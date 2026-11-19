from __future__ import annotations

import numpy as np

from argus_mask.audio import DRIVER_RATE
from argus_mask.prosody import Prosody

FPS = 30


def test_audio_that_arrives_after_its_cues_lands_at_its_own_index() -> None:
    """Cues may run ahead of the audio; trimming must not move the stream's start past what has arrived."""
    total = 4 * DRIVER_RATE
    ramp = np.arange(total, dtype=np.float32) / total  # each sample says where it belongs
    prosody = Prosody(FPS)
    arrived = DRIVER_RATE // 2
    prosody.push(ramp[:arrived])
    for frame in range(3 * FPS):  # 3 s of cues: past the half second that has come
        prosody.cue(frame)
    prosody.push(ramp[arrived:])  # the rest, late
    end = 3 * DRIVER_RATE + 1000
    assert np.array_equal(prosody._window(end, 640), ramp[end - 640 : end])
