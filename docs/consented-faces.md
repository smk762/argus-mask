# Capturing a consenting person's face

This is the sibling of [docs/synthetic-faces.md](synthetic-faces.md), for the other route to a real-looking face: a real person, paid and released for this use ([#26](https://github.com/smk762/argus-mask/issues/26)). Synthetic portraits remain the default. Use this route only when you want a particular, real person.

Before anything is captured, the person signs a release that covers the face ([docs/likeness-release.md](likeness-release.md)), and their age is checked against ID. Nothing in this guide happens without that.

## What a good capture looks like

The routes that turn photos into a face want the same things whether the face is real or generated:

| Want | Avoid | Why |
|---|---|---|
| Face straight to camera, head level | three-quarter view, tilt, looking away | The models assume a frontal neutral pose; yaw and tilt get baked in |
| Neutral expression, lips closed and relaxed | smile, open mouth, visible teeth, raised brows | Expression is animated as change from neutral; a smile in the source can never fully relax |
| Eyes open, looking into the lens | squint, half-closed lids, sideways gaze | Blinks and gaze are applied from this state |
| Soft, even, frontal light: two softboxes at 45°, or a large window on an overcast day | hard key, rim light, coloured light, deep shadows | Light is baked into the face and moves with it |
| Plain mid-grey background | busy scenes, other people | Easy, clean matting, and nobody else in frame |
| Forehead, ears, jaw and neck visible; head and shoulders | hair over the eyes or brows, hats, glasses, hands, high collars | Anything covering the face gets fused into it |
| Light or no makeup | heavy contour, glitter | Contour fights the viewer's own shading |
| Sharp throughout, the camera's full resolution, no filters | beauty mode, portrait-mode blur, a phone's HDR "look" | Detail is what makes it read as real |

Tie long hair back where you can. Loose strands are the hardest part for every route.

### More than one view

One-shot routes need only a frontal portrait. Multi-view routes ([#17](https://github.com/smk762/argus-mask/issues/17)) need the head from all round:

- **A slow turntable:** the person sits on a swivel chair and turns slowly through 360° (about 20 s), the camera fixed at eye height, 1 to 1.5 m away. Shoot video at 4K if you can.
- **Or a fixed camera ring:** several cameras fired together, which avoids any change of expression between views.

Keep the lighting, background, camera settings (fixed exposure, white balance and focus) and expression identical across views. View-to-view lighting changes look like texture to the trainer.

## The reference images

At the signing session, take a small set of **reference images**: three to five frontal photos, as in the table above, in the same light. They bind the release to this person's face:

- their hashes go into the consent record (`consent new --reference-image`);
- curator ([smk762/argus-curator#133](https://github.com/smk762/argus-curator/issues/133)) matches every image it exports for this person against them by face embedding, and leaves out any frame that shows someone else.

Keep the reference images as the release says (§ 9), not in the assets directory.

## From capture to a served face

1. **Record the consent:** `argus-mask consent new …` (see [docs/consent.md](consent.md)). It prints the release's sha256.
2. **Build the face** from the captures by whichever route you use. Keep the source captures outside the assets root.
3. **Write its manifest** with the `consented` kind, the subject exactly as the release names them, and the release's sha256:

   ```bash
   argus-mask inspect assets/<face-id>/head.glb --write-manifest --id <face-id> --name "…" \
       --kind consented --subject "<name as on the release>" --release-sha256 <sha256> \
       --source "Captured <date> under the release above; built with <route>" --license "per the release"
   ```

4. **Check it:** `argus-mask inspect assets/<face-id>` says whether the server will serve it, and if not, why. A consented face is served only while its record covers the face and the use: `avatar` to be shown on a screen, `video` as well to be recorded. Every screen and every recording shows the disclosure label: *An AI avatar of a real person, shown with their consent.*

If the release is revoked, `argus-mask consent revoke` stops the face being served and moves it to `assets/.trash/` ([docs/consent.md](consent.md#revoking)).

## Checklist

- [ ] The release is signed, covers the face, and the person's age was checked against ID.
- [ ] Reference images taken at the signing session, frontal, in the capture's light.
- [ ] `consent new` run, with the reference images; its sha256 noted.
- [ ] Captures as in the table; for multi-view, the lighting and settings fixed across views.
- [ ] The manifest says `consented`, with the subject as the release names them and the release's sha256.
- [ ] `inspect` on the face directory shows no refusal.
- [ ] Source captures and the signed release are kept as the release says, outside the assets root and out of any repository.
