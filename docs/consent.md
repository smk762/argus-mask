# Consent records

A real person's face or voice is used here only under a signed release ([docs/likeness-release.md](likeness-release.md)) and a **consent record** that says what the release covers. The record decides whether a consented face is served, and whether a consented voice may be made or spoken. Revoking it stops both, and removes what was made ([#26](https://github.com/smk762/argus-mask/issues/26)).

A record holds **hashes, never documents**: the sha256 of the signed release, of the reference images and of the reference recording. It never holds the release itself, the media, or any copy of the ID the person's age was checked against. It does name the person, so it is kept readable by its owner only, and is never served. The server answers only whether a use is allowed and why (`GET /face/{id}/consent?use=video`).

## What a record says

Records live at `<assets>/.consent/<release_sha256>.json`, one per release. A face's provenance and a voice's `voice.json` name the release by its hash, which is how they find their record.

| Field | Meaning |
|---|---|
| `release_sha256` | The signed release |
| `subject` | The person, as the release names them. A face's `subject` and a voice's speaker must match it exactly, spaces at the ends aside |
| `collected_by` | Who ran the signing session |
| `signed_on`, `expires_on` | When it was signed; the first day it no longer holds (none: no expiry). Dates are read as UTC days |
| `age_check` | Who checked the person's age against ID, and when. Never the ID |
| `face`, `voice` | Which the release covers. Each is a separate opt-in |
| `deliveries` | For a voice: the deliveries covered (`neutral`, `warm`, …). Each is its own voice |
| `uses` | `avatar` (a live face on a screen), `video` (`record` output), `speech` (the voice with no face) and `training` (fitting a model on the captures). A use not listed is not covered |
| `reference_images`, `reference_recordings` | Hashes of the references taken at the signing ([faces](consented-faces.md#the-reference-images), [voices](voices.md#a-consenting-speaker-the-session)) |
| `exclusions`, `media`, `territories`, `notes` | As the release says them, for people to read. Code doesn't check them |
| `revocable`, `notice_days`, `on_revoke` | The revocation terms: whether, with what notice, and what happens to recordings already made (`delete` or `keep-labelled`) |
| `revocation` | Set by `consent revoke`: when it was asked for, when it takes effect, and who recorded it |

A record is checked in a fixed order, and the first failure is the answer. Mask and curator ([smk762/argus-curator#133](https://github.com/smk762/argus-curator/issues/133)) refuse for the same reasons, in the same words:

1. `revoked`
2. `not-yet-signed`
3. `expired`
4. `likeness-not-covered`: the face, or the voice
5. `use-not-covered`
6. `delivery-not-covered`: voices only

The record and its checks are argus-cortex's (`argus_cortex.consent`, 0.5.0), shared by every package. Mask adds rules of its own for the records it writes: a delivery is named in lower case as the Voices page names it (`warm`, not `Warm`), and names and notes have a length limit. "Today" is the UTC day, as cortex reads a record's dates, so a revocation dated today stops use at once wherever you are.

## Commands

```bash
# Write the record (the files are read only to hash them)
argus-mask consent new --release signed-release.pdf --subject "A. Person" --collected-by "You" \
    --signed-on 2026-11-20 --age-checked-by "You" --face --voice --delivery neutral --delivery warm \
    --use avatar --use video --reference-image front-1.jpg --reference-image front-2.jpg \
    --reference-recording reference.wav --revocable --notice-days 30 --on-revoke delete

argus-mask consent list            # every record, and whether it holds today
argus-mask consent revoke 3fa9c2e1 --by "You"   # by the release's sha256, or its first 8+ characters
```

A release has one record. A re-signed or amended release has a new hash, so it gets a new record. If a mistake was made, revoke the old record and write a new one.

## Where the record is enforced

| Where | Use checked | When it fails |
|---|---|---|
| The server, serving a consented face (`/faces`, `/face/{id}`, the viewer socket) | `avatar` | The face is not served; `inspect` says why |
| `record`, and recording from the Broadcast page | `video` | The recording is refused before it starts |
| The Voices page, making a consented voice | the voice and its delivery | The voice is not made |
| `say`, `POST /speak`, the Broadcast page, the Voices page's **Try** | `avatar` (`video` too when the Broadcast page records) | Nothing is sent to the TTS service |

A consented face carries `disclosure` in its manifest, and a stream in a consented voice carries it in its header. The viewer shows the label on every screen, kiosks included, and in every recording, whatever `?hud` says.

**Limits.** A voice is checked only where its `voice.json` can be read: on the machine whose Voices page made it (`ARGUS_MASK_VOICES_DIR`). A clone that someone made on a TTS service directly isn't known here at all. `record` on its own checks the face, not the voice: it can't tell which voice a session used, only that the voice was a consenting person's (from the label).

## Revoking

`argus-mask consent revoke` marks the record revoked, effective from today unless you give `--effective-on`. From that day nothing under the release is served or spoken. Then it removes what was made:

- each voice made under the release is deleted from the TTS service it was made on (`DELETE /clones/<id>`, as the Voices page's Delete does), and its record and clips move to the voices' `.trash/`;
- each face under the release moves to `assets/.trash/`;
- recordings already made aren't tracked, so it reminds you what the release agreed (`on_revoke`). Deal with them as it says.

If a TTS service can't be reached, that voice stays where it is. It is refused, because its record is revoked. The command fails and names the voice, and running it again finishes the job (the first revocation stands). A revocation dated in the future removes nothing yet: run the command again on or after that day.

The record itself stays, revoked, as the account of what was agreed and when it ended. Empty the `.trash/` folders, and delete the source captures and the signed release, as the release's retention terms say.
