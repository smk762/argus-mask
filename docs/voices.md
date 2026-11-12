# Making a voice

This guide covers the **reference clip** a cloning TTS service (Miso, XTTS) makes a voice from: what a good one sounds like, how to record one, how to make a synthetic one, and how to get more than one delivery from the same voice. The editor's **Voices** page (`/voices.html`) checks clips against everything below, makes the voice on the TTS service, and keeps a record of where it came from.

What the model does with your clip:
- **It copies all of it.** A one-shot clone copies the voice, but also the room, the background noise, the microphone's bandwidth, the pace and the mood of the delivery. A tense line makes a tense voice; a noisy clip makes a noisy voice.
- **It reads the clip against its transcript.** The transcript tells the model which sounds are which words. A wrong or loose transcript teaches it to say words that are not there, which you hear later as invented phrases.

A careful 15 seconds saves more time than any later fix.

## Where a voice comes from

A voice is a likeness, like a face ([#19](https://github.com/smk762/argus-mask/issues/19)), and the same two tiers apply ([README.md § Personal and cleared](../README.md#personal-and-cleared)). For your own speakers at home, any voice you may lawfully use privately is fine: pick **Personal use** and note where the clips came from. A voice you will show or share must be **cleared**: one of the first three sources below. The Voices page records which:

| Source | What it is | What the record keeps |
|---|---|---|
| **Generated** | A TTS service's own voice, or a voice designed from a description: no real person | What made it: the service, the speaker or description, the temperature |
| **Your own** | You, recorded by you | That it is yours |
| **A consenting speaker** | Someone paid and released for this use ([#26](https://github.com/smk762/argus-mask/issues/26)) | Their name as the release gives it, and the release's sha256 (never the document) |
| **Personal use** | Anyone else, for your own speakers only: a game character, a narrator. Never cleared | Where the clips came from, in your words |

A cleared voice is never a clip of anyone who hasn't agreed to it: not taken from a film, show, game, video, podcast, audiobook or call, and not someone doing an impression of a real person. The page asks you to confirm this before it makes a cleared voice. A clip cut from a recording is also usually a bad reference anyway, with music, effects and room sound under the voice (the checks below catch it).

## What a good reference looks like

| Want | Avoid | Why |
|---|---|---|
| 12 to 20 s of speech, in one to three clips | under 6 s; over 20 s | Too little and the voice wanders; Miso keeps only the first 20 s |
| One speaker, speaking to someone | reading a list; acting; shouting; laughing | The delivery is copied into every line the voice says |
| The register you want to hear: calm and clear for an assistant | an argument, a joke, a whisper | Same reason: a reference of one mood gives a voice of one mood |
| A quiet, soft room; mic 15 to 30 cm away | fans, traffic, a bare room with echo, music | Noise and echo are copied; the check wants the gaps 30 dB below the speech |
| Peaks around -6 to -3 dBFS | any clipping; peaks below -24 dBFS | Clipping crackles in the clone; a quiet clip's noise comes up with it |
| The original file: WAV or FLAC, 44.1 or 48 kHz | MP3s, video soundtracks, phone or video-call recordings | Lossy and phone audio lose the top of the voice: the clone sounds muffled |
| About 0.2 s of quiet at each end | a word cut at the start or end; seconds of silence | A cut word teaches a cut word |
| Complete sentences, including a question | fragments; one long run-on | The model learns how sentences rise and fall and end |
| Numbers written as words in the script ("three", "nine thirty") | digits | The transcript must say exactly what is spoken |

The **transcript** is word for word what is said, with normal capitals and punctuation: "Good morning. Shall we start?" not "good morning shall we start". Include any "um" or repeated word you didn't cut. If you edit the audio, edit the transcript to match.

## Recording your own

1. **Room:** a small furnished room (curtains, sofa, a bed, a wardrobe of clothes) beats a big bare one. Turn off fans and close the window.
2. **Mic:** any decent USB or headset mic, 15 to 30 cm away, a little off to the side of your mouth so breath doesn't hit it. A phone's voice-memo app is usable if set to its best quality, held at the same distance.
3. **Level:** set the gain so your loudest words peak around -6 dBFS. Record a few seconds of silence first and listen: you should hear nothing.
4. **Script:** read one of the scripts on the Voices page's Generate tab (one per delivery), or write your own: three or four sentences, about 40 to 50 words, in the voice you want to hear back. Read it as if speaking to someone across a table. Do several takes and keep the best.
5. **Edit:** trim to about 0.2 s before the first word and after the last. Don't add noise reduction, compression or EQ: they smear the voice, and the clone copies the smear. Save as WAV (16 or 24-bit).
6. **Check:** upload it on the Voices page with its transcript. Fix anything marked bad, and read the warnings.

## A synthetic voice

A synthetic voice belongs to no one: no consent is needed, and none can be withdrawn. Two routes, both on the Voices page's **Generate** tab. In both, the TTS reads a script you wrote, so the transcript is known word for word: the one part of a reference that is otherwise easy to get wrong.

**1. The service's own speakers (Miso).** Miso has no fixed voices: without a reference, speaker `0` or `1` comes out as a different voice each time. Generate candidates and keep the one you like. Each takes about a minute on an RTX 3090.

Unconditioned, the model also wanders. In testing, one in four candidates read its script cleanly; the others repeated a sentence, added one after a long pause ("Add the table."), or clipped. The checks catch most of this: for a generated candidate the pace check is tighter (a model that repeats itself reads too few words a second), and long pauses and clipping are flagged. **Listen to every candidate to the end before you use it.**

**2. A described voice (agent-composer's Qwen3-TTS).** A service that designs voices takes a description, for example "female, warm, mid-thirties, light British accent", sent as `voice_description`. Point the page at that service (its `/synthesise` URL) to generate, then make the voice on Miso from the result: the description gives the timbre, Miso the quality. Check the model's licence on its model card before you rely on its output.

For either route:
- **Sound-alike check:** a model can land on a voice that sounds like someone famous. Listen honestly; if it sounds like someone you could name, generate another.
- **The record:** the page fills in "what made it" from the candidate (service, speaker or description, temperature). Keep it.

## One voice, several deliveries

Miso has no tone or emotion setting. Its API accepts `emotion`, `instruction` and `voice_description` for compatibility with other services, but ignores them. Temperature is not tone either: lower is steadier from one sentence to the next, higher is more varied, in the same mood.

The delivery comes from the reference. So make **one voice per delivery**, and pick the one a broadcast needs:

- `ginger-neutral`: an everyday assistant
- `ginger-warm`: friendly, reassuring
- `ginger-brisk`: short and to the point
- `ginger-serious`: something is wrong

The Voices page names them this way (name, then delivery), and has a script per delivery: written in that register, so the reading lands there without acting.

To keep it the **same voice** across deliveries:
- **Recorded:** the same speaker, room, mic and distance, in one session. Only the script and the way it is read change.
- **Generated:** a new candidate is a new voice, so don't generate each delivery from scratch. Make the first delivery, then use **that voice** to read the next delivery's script (Try a voice, or `say` with `record`), check the reading, and make it a voice of its own. The timbre carries over, and the delivery follows the script.

On the Broadcast page, pick the delivery as the voice. A temperature of about 0.7 keeps it steady.

## The checks

What the Voices page measures, per clip and for the set. A clip with a bad check is not made into a voice; warnings are yours to judge.

| Check | Good | Warn | Bad |
|---|---|---|---|
| Speech length (per clip) | 6 to 20 s | 2 to 6 s; over 20 s | under 2 s |
| Speech in all (the set) | 12 to 20 s | 6 to 12 s; over 20 s | under 6 s |
| Sample rate | 22.05 kHz and up | 16 kHz | under 16 kHz |
| Bandwidth: how high the voice reaches above the noise | 7 kHz and up | 4 to 7 kHz, or cut short by noise | under 4 kHz |
| Clipping | none | a few samples | more than 1 in 10,000 |
| Level | peaks above -24 dBFS | quieter | |
| Background noise: gaps against speech | 30 dB below | 20 to 30 dB | under 20 dB |
| Start and end | 0.05 to 1.5 s of quiet | speech at the edge; long silence | |
| Pauses | none over 1.5 s | a longer one | |
| Delivery: pitch spread | 1.2 to 7 semitones | flatter (monotone) or wider (acted) | |
| Transcript vs audio | 1.8 to 3.8 words/s (generated: 2.7 to 4.2) | outside it | |
| Transcript | capitals, a full stop, no digits | any of those missing | none |
| One voice (the set) | clips within 3 semitones and 6 dB | further apart | |

`argus_mask.voice` is the code: pure numpy, the same clip gives the same report.

## What a good reference doesn't fix

- **Tone drift between sentences.** `say` sends one sentence at a time, and each is generated with only the reference as context, so neighbouring sentences can land in slightly different moods. `say` already evens their loudness and softens their edges. The real fix is to give each sentence the previous one as context, which needs a change to Miso.
- **Runaway generation.** Now and then the model keeps talking after the text ends. Miso caps a request at 30 s; a cap scaled to the text's length would stop it sooner, which also needs a change to Miso.
- **Numbers.** `say` writes numbers and times as words before sending them (`--as-is` turns it off), because the model guesses at digits.
