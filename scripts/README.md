# Building a stock head

`src/argus_mask/editor/build_head.py` builds a head for argus-mask with [MPFB](https://static.makehumancommunity.org/mpfb.html), the MakeHuman plugin for Blender, and exports it as a `.glb` with the 52 ARKit face shapes. It runs headless. A preset under `stock_heads/` is the whole recipe, so another synthetic head is another preset.

Blender and MPFB are authoring tools. Nothing in the package imports them, and neither is needed to run argus-mask.

**To get the head without building it**, run `python scripts/fetch_stock_head.py` (or `make head`) from the repo root. It downloads the built model named in `stock_heads/<id>.lock.json`, checks its size and sha256, and writes it with its manifest under `assets/`. The rest of this page is for building a head yourself.

## What you need

| What | Version used | Where from | sha256 |
|---|---|---|---|
| Blender | 4.5.15 LTS, Linux x64 | <https://download.blender.org/release/Blender4.5/> | `0d2b8bb2efe97e6c8eb57de6b7322f8800e1984e3125b60decca5f6771e4773e` |
| MPFB extension | 2.0.18 | <https://extensions.blender.org/add-ons/mpfb/> | `c075b3e9b94b8e5556a5f03f4653602d02ea306de08126e8f55ed6dee28a2dad` |
| MakeHuman system assets pack | `makehuman_system_assets_cc0.zip` | <https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html> | `b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107` |
| Faceunits 01 pack | `faceunits01.zip` | <https://static.makehumancommunity.org/assets/assetpacks/faceunits01.html> | `d113107bd7eb59f3af4df6fc0ec29bfcc593f496d0b336aec14f086a80ce7146` |

MPFB needs Blender 4.2 or later. Other versions may work; these are the ones the first head was built with.

## Set up, once

Keep Blender's user files out of your home directory by pointing `BLENDER_USER_RESOURCES` at a directory of your choice. `$WORK` below is any scratch directory.

```bash
export BLENDER_USER_RESOURCES=$WORK/user

# Blender: unpack the tarball; no install step.
mkdir -p $WORK/blender && tar -xJf blender-4.5.15-linux-x64.tar.xz -C $WORK/blender --strip-components=1

# MPFB: install the extension zip and enable it.
$WORK/blender/blender -b --command extension install-file -r user_default --enable add-on-mpfb-v2.0.18.zip

# Asset packs: unzip both into MPFB's user data directory. Each zip holds top-level
# directories (skins/, hair/, eyes/, targets/, packs/ ...) that merge into it.
DATA=$BLENDER_USER_RESOURCES/extensions/.user/user_default/mpfb/data
mkdir -p $DATA
python3 -m zipfile -e makehuman_system_assets_cc0.zip $DATA
python3 -m zipfile -e faceunits01.zip $DATA
```

## Build a head

From the repo root:

```bash
export BLENDER_USER_RESOURCES=$WORK/user
$WORK/blender/blender -b -P src/argus_mask/editor/build_head.py -- \
    --preset scripts/stock_heads/stock-01.json --out assets/stock-01/head.glb
```

It writes the model and, beside it, `head.recipe.json`: the versions, assets, triangle counts and textures actually used. The script exits with status 2 and writes nothing if the preset cannot be built or the result is over budget.

Then write the manifest. The provenance must say enough to rebuild the head:

```bash
argus-mask inspect assets/stock-01/head.glb --write-manifest \
    --id stock-01 --name "Stock head 01" --kind stock \
    --source "Built with src/argus_mask/editor/build_head.py from scripts/stock_heads/stock-01.json: Blender 4.5.15 LTS, MPFB 2.0.18, asset packs makehuman_system_assets and faceunits01" \
    --license "CC0-1.0" \
    --distance 0.9 --height -0.08 \
    --notes "MakeHuman base mesh and targets (bundled with MPFB 2.0.18); skin young_caucasian_male, hair short02, eyes high-poly (brown), eyebrow001, eyelashes01, teeth_base, tongue01 (makehuman_system_assets pack, CC0); ARKit face units by Mika Suominen (faceunits01 pack, CC0). Phenotype: adult, age macro 0.6. A generated character, not a real person. Revision 2: tongue flattened to lie on the floor of the mouth, and it drops with the chin under jawOpen; upper teeth lowered 2.5 mm; inside of the mouth darkened with vertex colours; mouthPucker scaled 1.5x; teeth texture brightened; brows and lashes blended instead of cut out."
```

`--distance` and `--height` set the camera framing; the default (0.6 m) is too close for a bust. The model's origin is on the centre line at eye height, halfway through the skull, and the face looks down +Z. A rebuild changes the model's sha256, so write the manifest again after each one.

`assets/` is gitignored: model files are never committed.

## Publishing a head

To let clones download a head you built, upload the `.glb` to the repo's `stock-heads` GitHub release under a new file name (never replace a published file: lock files pin its sha256), then write `stock_heads/<id>.lock.json` with the download `url`, the file's `size` in bytes, and the `manifest` you just wrote. `tests/test_fetch_stock_head.py` checks every committed lock. The release's tag has no digits in it on purpose: `hatch-vcs` reads any tag with a number as a version.

## The preset

| Field | Meaning |
|---|---|
| `phenotype` | MakeHuman's macro settings, each 0 to 1: `gender`, `age`, `muscle`, `weight`, `proportions`, `height`, `cupsize`, `firmness`, and `race` (`asian`, `caucasian`, `african`). |
| `assets` | Paths under MPFB's data directory: `skin` (`.mhmat`), and `eyes`, `eyebrows`, `eyelashes`, `teeth`, `tongue`, `hair` (`.mhclo`). Hair is optional. `clothes` is a list of up to three `.mhclo` paths under `clothes/` (an outfit, a hat); MPFB hides the skin under them, so they add few triangles. The system pack's clothes are CC0. |
| `targets` | MakeHuman modelling targets beyond the phenotype, from a fixed list (`MODIFIERS` in `build_head.py`): face shape (`head-square`, `head-round`, `head-oval`, `head-invertedtriangular`, `head-diamond`, 0 to 1), chin and jaw (`chin-bones`, `chin-width`, `chin-height`, `chin-prominent`, -1 to 1; `chin-triangle`, 0 to 1), nose (`nose-volume`, `nose-scale-horiz`, `nose-scale-vert`, `nose-scale-depth`, `nose-point-width`, `nose-point`, `nose-hump`, `nose-flaring`) and lips (`mouth-upperlip-volume`, `mouth-lowerlip-volume`, `mouth-upperlip-height`, `mouth-lowerlip-height`, `mouth-scale-horiz`, `mouth-cupidsbow`, `mouth-angles`, `mouth-philtrum-volume`), -1 to 1 unless noted. They are baked with the phenotype, so the face shapes, teeth and clothes fit the result. CC0, bundled with MPFB. |
| `bust` | Where the body is cut: `height_m` below the top of the head, `half_width_m` either side of the centre line. |
| `max_texture_px` | Cap for opaque textures, stored as JPEG. |
| `max_alpha_texture_px` | Cap for textures with real transparency (hair, lashes, brows), stored as PNG. |
| `max_triangles` | The build fails above this. |
| `mouth.tongue_flatten` | Squashes the tongue's rest shape toward its underside: 1 leaves it, 0.6 makes it 60% as tall. Setting this (or `tongue_drop_mm`) also makes the whole tongue drop with its tip under `jawOpen`. |
| `mouth.tongue_drop_mm` | Lowers the tongue's rest shape by this much more. |
| `mouth.upper_teeth_drop_mm` | Lowers the upper teeth, so their edge shows below the upper lip in an open mouth. |
| `mouth.cavity_shade` | How dark the back of the mouth gets, as a multiplier on its texture (0.02 is nearly black). Leave it out for no shading. |
| `shape_scale` | Face shapes to exaggerate or soften, by name: `{"mouthPucker": 1.5}` moves every vertex 1.5 times as far. |
| `texture_gain` | Brightness multiplier for a part's texture, by part: `{"teeth": 1.3}`. |
| `blend_parts` | Parts whose transparency stays blended (soft edges) instead of becoming a cut-out. Only for parts that lie flat on the skin: `["eyebrows", "eyelashes"]`. |

**Adults only.** `phenotype.age` below 0.5 is refused. On MakeHuman's scale 0.5 is about 25 years and 0.35 about 18, but below 0.5 the mesh blends in the child target, so the recipe starts at the young-adult target and goes no lower.

## What the script does to the model

- Bakes the phenotype into the mesh, then loads the 52 face units and has MPFB interpolate them onto the eyes, lashes, brows, teeth, tongue and hair. glTF cannot drive one mesh's morph targets from another, so each part carries its own copies of the shapes that move it.
- Removes MakeHuman's helper geometry, cuts the bust with three planes, and centres the head.
- Converts opaque textures to JPEG and drops normal maps.
- Exports without Draco, meshopt or KTX2, with sparse morph targets and no morph normals.
- Rewrites blended materials as alpha cut-outs (`MASK`, cutoff 0.35), because blended hair sorts wrongly against itself and the face in a real-time renderer. Parts listed in `blend_parts` stay blended.
- Fixes the mouth, which MakeHuman's assets leave wrong for speech (see below).

## The mouth

As MPFB fits them, three things make an open mouth look wrong, and the `mouth` settings correct each:

- **The tongue fills the opening.** MakeHuman's tongue is a dome as tall as the palate, higher than the lower teeth all the way round. The jaw shapes reach it as the jaw's own rotation, so its back half, near the hinge, barely moves when the mouth opens. Seen from the front, the dome still reaches the upper lip. The recipe flattens the rest shape so it lies below the crowns of the lower teeth, and under `jawOpen` moves the whole tongue as its tip moves. `tongueOut` keeps the full, raised shape MPFB gave it.
- **The upper teeth never show.** Their edge sits level with the lip's, while the lower teeth stand clear of the lower lip, so only lower teeth are seen. The recipe lowers the upper teeth.
- **The back of the mouth is lit like a cheek.** The viewer has no shadows. The recipe writes a vertex colour that darkens the head's mouth lining, the molars and the back of the tongue with depth behind the lips. It finds the lining by casting rays forward from the vertices around the mouth: the ones that hit the face are inside.

The lips are slightly parted at rest, and a thin line of teeth shows through. That is MakeHuman's neutral face, not something the recipe adds.

## Licences of what goes into `stock-01`

| Part | Asset | Licence | Stated at |
|---|---|---|---|
| Base mesh, modelling targets | bundled with MPFB (`data/3dobjs/base.obj`, `data/targets/`) | CC0 | file headers ("explicitly released as CC0 in september 2020"); <https://static.makehumancommunity.org/mpfb/faq/can_i_sell_models.html> |
| ARKit face shapes | `faceunits01` pack, by Mika Suominen | CC0 | `packs/faceunits01.json` in the pack (every target); <https://static.makehumancommunity.org/assets/assetpacks.html> ("The assets are shared under CC0") |
| Skin, hair, eyes, eyebrows, eyelashes, teeth, tongue | `makehuman_system_assets` pack | CC0 | `packs/makehuman_system_assets.json` in the pack (all 94 assets); each `.mhmat` / `.mhclo` header; <https://static.makehumancommunity.org/assets/assetpacks.html> ("System assets, shared under CC0") |

MPFB itself is GPL-3.0-or-later. It is the tool that builds the head; none of its code is in the exported model.

Other asset packs on the same site are CC-BY (Hair 02, Hair 03 and others). A preset that uses one must say so in the manifest's provenance, with the author.
