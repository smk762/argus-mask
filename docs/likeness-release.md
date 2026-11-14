# Likeness release: a template

> **This is not legal advice.** It is a starting point for a lawyer. Have one review it for each jurisdiction you use it in, before anyone signs it. Image rights, personality rights, data protection (a face and a voice are biometric data in many places) and employment law all differ between countries, and sometimes between states.

This release is for paying a person to be an avatar: their **face**, their **voice**, or both, used by argus-mask to say new things. A stock model release usually doesn't cover a digital double saying new things. A voice actor's usual contract covers recorded lines, not a clone that speaks new ones. A release written for this use is needed ([#26](https://github.com/smk762/argus-mask/issues/26)).

Once it is signed, its sha256 is recorded with `argus-mask consent new` ([docs/consent.md](consent.md)). The record holds the hash, never the document. Keep the document under § 9.

Fill in everything in `[brackets]`. Where the release offers a choice (☐), tick one and strike out the other before signing.

---

## Likeness release

**Between** `[name of the person]` ("the Person") **and** `[name of the operator, and organisation if any]` ("the Operator").

**Date of signing:** `[YYYY-MM-DD]`  **Place:** `[place]`

### 1. What this covers

The Person allows the Operator to make and use an AI avatar from their likeness, as set out below. Each part is a separate choice. The Person may agree to one and not another.

| | Agreed? |
|---|---|
| **Face:** a 3D or image-based model made from photos or video of the Person's face and head | ☐ yes ☐ no |
| **Voice:** a synthetic voice made from recordings of the Person speaking | ☐ yes ☐ no |

**Voice deliveries.** Each delivery (way of speaking) recorded becomes its own voice. The Person agrees only to the deliveries ticked: ☐ neutral ☐ warm ☐ bright ☐ calm ☐ brisk ☐ serious ☐ playful ☐ other: `[describe]`

### 2. Uses

The avatar may be used only for the uses ticked:

| Use | Meaning | Agreed? |
|---|---|---|
| **Live avatar** | The face shown on a screen, speaking live | ☐ yes ☐ no |
| **Video** | Recorded clips of the avatar | ☐ yes ☐ no |
| **Speech** | The voice heard on its own, with no face shown | ☐ yes ☐ no |
| **Training** | The recordings used to train or fine-tune a model, not only as a one-shot reference | ☐ yes ☐ no (**no** unless ticked) |

**Media:** `[for example: screens in the Operator's premises; the Operator's website; social media]`
**Territories:** `[for example: worldwide; the UK only]`
**Term:** from the date of signing ☐ with no end date ☐ until `[YYYY-MM-DD]`

### 3. Uses that are never allowed

Whatever § 2 says, the avatar is never used for:

1. sexual or sexualised content;
2. political speech, campaigning, or any message for or against a party, candidate, cause or policy;
3. endorsing a product, service, person or organisation, unless the Person approves that endorsement in writing;
4. impersonating the Person to anyone: presenting the avatar as the Person themselves, or using it to act in the Person's name;
5. for the voice:
   - phone calls or voicemail to anyone who doesn't know they are hearing an AI avatar;
   - voice authentication of any kind: banks, phone services, smart locks, or any system that recognises a person by their voice;
   - singing or performance outside the uses in § 2.

**Further exclusions the Person adds:** `[list, or "none"]`

### 4. Disclosure

Everything the avatar produces is labelled as an AI avatar of a real person who has consented. Audio heard without a face (§ 2, Speech) carries a spoken or written label, since nobody can see a face. argus-mask shows this label on every screen and in every recording, and no setting turns it off.

### 5. Revocation

☐ The Person may revoke this release, by written notice to the Operator, with `[N]` days' notice.
☐ This release may not be revoked during its term, except as the law allows. (Many jurisdictions let a person withdraw consent to the use of their biometric or personal data at any time. Check with a lawyer.)

When it is revoked, from the day it takes effect:

- the avatar stops being used;
- the face and voice models are deleted;
- each voice is deleted from every TTS service it was made on;
- the reference recordings and source captures are deleted (§ 9);
- recordings already made under § 2 are: ☐ deleted ☐ kept, labelled as in § 4, and not used again.

The Operator confirms in writing when this is done.

### 6. Payment and credit

**Payment:** `[amount, currency, when, how]`
**Credit:** ☐ the Person is credited as `[name as credited]` where `[where]` ☐ no credit

### 7. New uses

☐ Any use not in § 2 needs the Person's written approval first, under a new or amended release.
☐ `[other arrangement]`

### 8. Age

The Person confirms they are 18 or over. The Operator has checked this against a government-issued photo ID at signing. The Operator records that the check was done, when, and by whom. **No copy of the ID is made or kept.**

### 9. What is kept, by whom, and for how long

| What | Kept by | For how long |
|---|---|---|
| This signed release | `[who]` | `[the term, plus N years]` |
| Source captures (photos, video, voice recordings) | `[who]` | `[until the models are built / the term / …]` |
| Reference images and reference recording from the signing session (they bind this release to the Person's face and voice) | `[who]` | the term |
| The built face and voice models | `[who]` | the term, or until revoked |

The Person may ask what is held about them, and have a copy.

### 10. Signatures

| | Name | Signature | Date |
|---|---|---|---|
| The Person | | | |
| The Operator | | | |
| Age checked by (§ 8) | | | |

---

## After signing

1. Take the **reference images** and read the **reference recording** at the same session. See [docs/consented-faces.md](consented-faces.md) and [docs/voices.md § A consenting speaker](voices.md#a-consenting-speaker-the-session).
2. Write the record from the signed file and those references. Mask reads them only to hash them:

   ```bash
   argus-mask consent new --release signed-release.pdf --subject "<name as on the release>" \
       --collected-by "<you>" --signed-on <YYYY-MM-DD> --age-checked-by "<who checked the ID>" \
       --face --voice --delivery neutral --delivery warm --use avatar --use video \
       --reference-image front-1.jpg --reference-image front-2.jpg --reference-recording reference.wav \
       --revocable --notice-days 30 --on-revoke delete
   ```

   The record mirrors §§ 1, 2, 3, 5 and 8. Add the § 3 extras with `--exclusion`, and § 2's media and territories with `--media` and `--territory`.
3. Store the signed document as § 9 says, **not** in the assets directory and never in a repository.
