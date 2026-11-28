# Accessories spike (#67)

Layer a **committed** procedural Three.js accessory on a face (splat or mesh), prove it tracks the head bone and stays deterministic in record mode, then decide whether img2threejs is worth adopting for authoring toward [#60](https://github.com/smk762/argus-mask/issues/60).

## Method

- **Reference:** synthetic PNG [`docs/media/earpiece-ref.png`](media/earpiece-ref.png) (repo-owned; not a product photo).
- **Factory:** [`viewer/accessories/earpiece.js`](../src/argus_mask/viewer/accessories/earpiece.js) — `createEarpiece()` → `THREE.Group` from primitives. Written to the img2threejs *output contract* (reviewed code, no agent at runtime). img2threejs itself is an Apache-2.0 **agent skill** (`npx img2threejs@latest`), not a library; a full vision forge was not required to answer the layering question.
- **Attach:** [`viewer/accessories.js`](../src/argus_mask/viewer/accessories.js). Mesh: parent to `head` / `neck_01`. Splat: the vendored renderer draws `threeScene` *under* the Gaussians and hides the avatar mesh, so a bone child would be invisible or buried — a `BoneOverlay` copies the bone's world pose (unit scale) and draws **after** the splat pass.
- **Enable:** `?accessory=earpiece` on the viewer (spike only; no `FaceManifest` field yet).

## Numbers (splat-01, record mode, `life=0`)

Playwright poses the same weights twice at yaw 0° and twice at yaw 18° (`scripts/accessories_spike.py`, writing to `work/accessories-spike/`):

| Pose | sha256[:16] (both passes) | Match |
|---|---|---|
| yaw 0° | `8ad5008515b0c616` | yes |
| yaw 18° | `24e00ddecc9b989f` | yes |

Measured locally on `splat-01`: same-pose frames bit-identical; turn changes the frame; `__mask.accessory` reports `{ kind: "splat", bone: "head", overlay: true }`. Cost is a few dozen triangles + one extra `renderer.render` after the splat pass.

## Decision

**Adopt the runtime pattern; do not depend on img2threejs for shipping.**

Keep: committed factories under `viewer/accessories/`, `BoneOverlay` for splat heads, mesh bone parenting, `?accessory=` for local checks. A follow-up for #60 can add optional `FaceManifest.accessories` (module id, bone, TRS) and cortex wire bump once a cleared set exists.

Drop / defer: running the img2threejs agent forge in CI or at serve time; borrowing its vision quality loop. Use it optionally when *authoring* a new factory, then review and commit the JS like any other viewer code.

No demo default, no guided step, no wire change in this PR.
