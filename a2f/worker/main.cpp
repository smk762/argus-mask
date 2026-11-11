// a2f-worker: holds one Audio2Face regression model loaded and turns 16 kHz mono float32
// PCM into blendshape weights as the audio arrives.
//
// It speaks a framed binary protocol on stdin/stdout, so the process that owns the network
// (server.py) needs no CUDA. Every message is: 1 byte type, 4 bytes little-endian payload
// length, payload.
//
//   in   'A'  audio: f32le samples, 16 kHz mono, any chunk size
//        'E'  end of stream: flush the tail, answer 'E', reset for the next stream
//   out  'H'  once, after the model has loaded: JSON {"names": [...], "fps": n, ...}
//        'F'  one finished frame: u32 index, i64 timestamp in samples, then f32 weights
//        'E'  the stream is flushed and the worker is ready for the next one
//        'X'  fatal error, UTF-8 text; the worker exits after sending it
//
// stdout carries only this protocol. The SDK and this program log to stderr.
//
//   a2f-worker <model.json> [--fps N] [--gpu-solver]
//   a2f-worker <model.json> --selftest <clip.f32>    (checks that a reset stream repeats exactly)
//
// SPDX-License-Identifier: MIT

#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::size_t kSampleRate = 16000;
constexpr std::uint32_t kMaxPayload = 1u << 24;  // 16 MiB: far more than any audio chunk

struct Destroyer {
  template <typename T> void operator()(T* obj) const { obj->Destroy(); }
};
template <typename T> using UniquePtr = std::unique_ptr<T, Destroyer>;

struct Frame {
  std::int64_t timestamp;  // in samples from the start of the stream
  std::vector<float> weights;
};

// ---- framed I/O on the real stdin/stdout ------------------------------------------------

int g_out = STDOUT_FILENO;

bool readAll(int fd, void* buffer, std::size_t size) {
  auto* p = static_cast<char*>(buffer);
  while (size > 0) {
    const ssize_t n = ::read(fd, p, size);
    if (n == 0) return false;
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += n;
    size -= static_cast<std::size_t>(n);
  }
  return true;
}

bool writeAll(int fd, const void* buffer, std::size_t size) {
  auto* p = static_cast<const char*>(buffer);
  while (size > 0) {
    const ssize_t n = ::write(fd, p, size);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += n;
    size -= static_cast<std::size_t>(n);
  }
  return true;
}

bool send(char type, const void* payload, std::uint32_t size) {
  char header[5];
  header[0] = type;
  std::memcpy(header + 1, &size, 4);  // x86-64 only: little-endian
  return writeAll(g_out, header, 5) && (size == 0 || writeAll(g_out, payload, size));
}

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "a2f-worker: %s\n", message.c_str());
  send('X', message.data(), static_cast<std::uint32_t>(message.size()));
  std::_Exit(1);
}

void check(std::error_code error, const char* what) {
  if (error) fail(std::string(what) + ": " + error.message());
}

std::string jsonString(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) out += c;
  }
  return out + "\"";
}

// ---- the model ----------------------------------------------------------------------------

class Model {
public:
  Model(const std::string& modelPath, std::size_t fps, bool gpuSolver) {
    check(nva2x::SetCudaDeviceIfNeeded(0), "select the CUDA device");

    const auto option = nva2f::IGeometryExecutor::ExecutionOption::SkinTongue;
    nva2f::IRegressionModel::IGeometryModelInfo* geometryInfo = nullptr;
    nva2f::IRegressionModel::IBlendshapeSolveModelInfo* solveInfo = nullptr;
    _bundle.reset(nva2f::ReadRegressionBlendshapeSolveExecutorBundle(
        1, modelPath.c_str(), option, gpuSolver, fps, 1, &geometryInfo, &solveInfo));
    if (!_bundle || !solveInfo) fail("could not load the model at " + modelPath + " (see stderr)");

    // Weights come out as the skin poses followed by the tongue poses.
    const auto params = solveInfo->GetExecutorCreationParameters(option);
    for (const auto* part : {params.initializationSkinParams, params.initializationTongueParams}) {
      if (!part) continue;
      for (std::size_t i = 0; i < part->data.poseNamesSize; ++i) _names.emplace_back(part->data.poseNames[i]);
    }
    solveInfo->Destroy();
    if (geometryInfo) geometryInfo->Destroy();

    auto& executor = _bundle->GetExecutor();
    if (executor.GetSamplingRate() != kSampleRate) fail("the model does not take 16 kHz audio");
    if (_names.size() != executor.GetWeightCount()) fail("the model's pose names do not match its weight count");
    _device = executor.GetResultType() == nva2f::IBlendshapeExecutor::ResultsType::DEVICE;

    // A constant neutral emotion for all time: one value at t=0 on a closed accumulator.
    // Audio2Emotion is deliberately not used.
    auto& emotion = _bundle->GetEmotionAccumulator(0);
    const std::vector<float> neutral(emotion.GetEmotionSize(), 0.0f);
    check(emotion.Accumulate(0, nva2x::HostTensorFloatConstView{neutral.data(), neutral.size()},
                             _bundle->GetCudaStream().Data()),
          "set the emotion");
    check(emotion.Close(), "close the emotion accumulator");

    if (_device) {
      _scratch.reset(nva2x::CreateHostTensorFloat(executor.GetWeightCount()));
      check(executor.SetResultsCallback(&Model::onDevice, this), "set the results callback");
    } else {
      check(executor.SetResultsCallback(&Model::onHost, this), "set the results callback");
    }
  }

  const std::vector<std::string>& names() const { return _names; }
  bool gpuSolver() const { return _device; }

  // Add audio and return every frame that can now be computed, in order.
  std::vector<Frame> feed(const float* samples, std::size_t count) {
    if (count > 0) {
      check(_bundle->GetAudioAccumulator(0).Accumulate(nva2x::HostTensorFloatConstView{samples, count},
                                                        _bundle->GetCudaStream().Data()),
            "accumulate audio");
    }
    return drain();
  }

  // End of the stream: the frames the model was holding back for audio that will not come.
  // Leaves the model reset, ready for a new stream starting at time zero.
  std::vector<Frame> finish() {
    auto& audio = _bundle->GetAudioAccumulator(0);
    check(audio.Close(), "close the audio");
    auto frames = drain();
    check(_bundle->GetExecutor().Reset(0), "reset the executor");
    check(audio.Reset(), "reset the audio accumulator");
    return frames;
  }

private:
  std::vector<Frame> drain() {
    auto& executor = _bundle->GetExecutor();
    while (nva2x::GetNbReadyTracks(executor) > 0) check(executor.Execute(nullptr), "execute");
    // The CPU solver finishes on its own threads; wait so that every frame of this batch is in.
    check(executor.Wait(0), "wait for the solver");
    auto& audio = _bundle->GetAudioAccumulator(0);
    if (!audio.IsClosed()) {
      // Let the accumulator reuse the memory of audio no frame will read again.
      check(audio.DropSamplesBefore(executor.GetNextAudioSampleToRead(0)), "drop used audio");
    }
    std::vector<Frame> frames;
    {
      const std::lock_guard<std::mutex> lock(_mutex);
      frames.swap(_pending);
    }
    if (!_error.empty()) fail(_error);
    std::stable_sort(frames.begin(), frames.end(),
                     [](const Frame& a, const Frame& b) { return a.timestamp < b.timestamp; });
    return frames;
  }

  static void onHost(void* self, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code error) {
    auto& model = *static_cast<Model*>(self);
    const std::lock_guard<std::mutex> lock(model._mutex);
    if (error) {
      model._error = "blendshape solve: " + error.message();
      return;
    }
    model._pending.push_back({results.timeStampCurrentFrame,
                              std::vector<float>(results.weights.Data(), results.weights.Data() + results.weights.Size())});
  }

  static bool onDevice(void* self, const nva2f::IBlendshapeExecutor::DeviceResults& results) {
    auto& model = *static_cast<Model*>(self);
    std::error_code error = nva2x::CopyDeviceToHost(*model._scratch, results.weights, results.cudaStream);
    if (!error) error = model._bundle->GetCudaStream().Synchronize();
    const std::lock_guard<std::mutex> lock(model._mutex);
    if (error) {
      model._error = "read the weights back: " + error.message();
      return false;
    }
    model._pending.push_back({results.timeStampCurrentFrame,
                              std::vector<float>(model._scratch->Data(), model._scratch->Data() + model._scratch->Size())});
    return true;
  }

  UniquePtr<nva2f::IBlendshapeExecutorBundle> _bundle;
  UniquePtr<nva2x::IHostTensorFloat> _scratch;
  std::vector<std::string> _names;
  bool _device{false};
  std::mutex _mutex;
  std::vector<Frame> _pending;
  std::string _error;
};

// ---- modes --------------------------------------------------------------------------------

void sendFrames(const std::vector<Frame>& frames, std::uint32_t& next) {
  std::vector<char> payload;
  for (const auto& frame : frames) {
    payload.resize(12 + frame.weights.size() * sizeof(float));
    std::memcpy(payload.data(), &next, 4);
    std::memcpy(payload.data() + 4, &frame.timestamp, 8);
    std::memcpy(payload.data() + 12, frame.weights.data(), frame.weights.size() * sizeof(float));
    if (!send('F', payload.data(), static_cast<std::uint32_t>(payload.size()))) std::_Exit(0);  // the reader went away
    ++next;
  }
}

int serve(Model& model, std::size_t fps, double loadSeconds) {
  std::string hello = "{\"names\":[";
  for (std::size_t i = 0; i < model.names().size(); ++i) hello += (i ? "," : "") + jsonString(model.names()[i]);
  hello += "],\"fps\":" + std::to_string(fps) + ",\"sample_rate\":" + std::to_string(kSampleRate) +
           ",\"gpu_solver\":" + (model.gpuSolver() ? "true" : "false") + ",\"load_s\":" + std::to_string(loadSeconds) + "}";
  if (!send('H', hello.data(), static_cast<std::uint32_t>(hello.size()))) return 0;

  std::vector<float> samples;
  std::uint32_t next = 0;
  while (true) {
    char header[5];
    if (!readAll(STDIN_FILENO, header, 5)) return 0;  // stdin closed: the server is done with us
    std::uint32_t size = 0;
    std::memcpy(&size, header + 1, 4);
    if (size > kMaxPayload) fail("a message is larger than 16 MiB");
    if (header[0] == 'A') {
      if (size % sizeof(float) != 0) fail("audio is not a whole number of float32 samples");
      samples.resize(size / sizeof(float));
      if (size > 0 && !readAll(STDIN_FILENO, samples.data(), size)) return 0;
      for (float& s : samples) {
        if (!std::isfinite(s)) s = 0.0f;  // one NaN would poison every later frame
      }
      sendFrames(model.feed(samples.data(), samples.size()), next);
    } else if (header[0] == 'E') {
      if (size != 0) fail("an end message carries no payload");
      sendFrames(model.finish(), next);
      if (!send('E', nullptr, 0)) return 0;
      next = 0;
    } else {
      fail(std::string("unknown message type '") + header[0] + "'");
    }
  }
}

// Runs one clip three ways and compares: fed whole; fed in 40 ms chunks after a reset; and
// again after an unrelated stream (the clip reversed) and another reset. A stream must not
// depend on the one before it, and chunking must not change the result.
int selftest(Model& model, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::vector<float> clip(raw.size() / sizeof(float));
  std::memcpy(clip.data(), raw.data(), clip.size() * sizeof(float));
  if (clip.empty()) {
    std::fprintf(stderr, "selftest: %s is empty or unreadable\n", path.c_str());
    return 2;
  }
  auto run = [&](const std::vector<float>& audio, std::size_t chunk) {
    std::vector<Frame> all;
    for (std::size_t i = 0; i < audio.size(); i += chunk) {
      auto part = model.feed(audio.data() + i, std::min(chunk, audio.size() - i));
      all.insert(all.end(), part.begin(), part.end());
    }
    auto tail = model.finish();
    all.insert(all.end(), tail.begin(), tail.end());
    return all;
  };
  auto difference = [](const std::vector<Frame>& a, const std::vector<Frame>& b) {
    if (a.size() != b.size()) return 1e9;
    double worst = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a[i].timestamp != b[i].timestamp) return 1e9;
      for (std::size_t k = 0; k < a[i].weights.size(); ++k)
        worst = std::max(worst, static_cast<double>(std::fabs(a[i].weights[k] - b[i].weights[k])));
    }
    return worst;
  };
  const auto whole = run(clip, clip.size());
  const auto chunked = run(clip, 640);
  std::vector<float> reversed(clip.rbegin(), clip.rend());
  const auto other = run(reversed, 640);
  const auto again = run(clip, 640);
  bool ordered = true;
  for (std::size_t i = 1; i < whole.size(); ++i) ordered = ordered && whole[i].timestamp > whole[i - 1].timestamp;
  const double dChunk = difference(whole, chunked), dReset = difference(chunked, again), dOther = difference(chunked, other);
  std::fprintf(stderr,
               "selftest: frames=%zu ordered=%d first_ts=%lld last_ts=%lld\n"
               "selftest: whole vs chunked max|diff|=%g\n"
               "selftest: chunked vs same clip after another stream and a reset max|diff|=%g\n"
               "selftest: (control) chunked vs the other clip max|diff|=%g\n",
               whole.size(), ordered ? 1 : 0, whole.empty() ? 0LL : static_cast<long long>(whole.front().timestamp),
               whole.empty() ? 0LL : static_cast<long long>(whole.back().timestamp), dChunk, dReset, dOther);
  const bool ok = !whole.empty() && ordered && dReset < 1e-4 && dChunk < 1e-4;
  std::fprintf(stderr, "selftest: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Keep the protocol stream clean: anything the SDK or its libraries print to stdout goes
  // to stderr instead, and the protocol keeps a private copy of the real stdout.
  g_out = ::dup(STDOUT_FILENO);
  ::dup2(STDERR_FILENO, STDOUT_FILENO);

  if (argc < 2) {
    std::fprintf(stderr, "usage: a2f-worker <model.json> [--fps N] [--gpu-solver] [--selftest clip.f32]\n");
    return 2;
  }
  const std::string modelPath = argv[1];
  std::size_t fps = 30;
  bool gpuSolver = false;
  std::string selftestPath;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--fps" && i + 1 < argc) fps = static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (arg == "--gpu-solver") gpuSolver = true;
    else if (arg == "--selftest" && i + 1 < argc) selftestPath = argv[++i];
    else fail("unknown argument " + arg);
  }
  if (fps < 1 || fps > 240) fail("--fps must be between 1 and 240");

  const auto start = std::chrono::steady_clock::now();
  Model model(modelPath, fps, gpuSolver);
  const double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::fprintf(stderr, "a2f-worker: loaded %s in %.2f s (fps %zu, %s solver, %zu weights)\n", modelPath.c_str(),
               loadSeconds, fps, model.gpuSolver() ? "GPU" : "CPU", model.names().size());

  if (!selftestPath.empty()) return selftest(model, selftestPath);
  return serve(model, fps, loadSeconds);
}
