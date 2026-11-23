# Making a synthetic face for input

This guide covers making a **source portrait**: one still image of a person who does not exist, made with an image model such as SDXL. Photoreal face routes start from a portrait like this:

- a Gaussian-splat head avatar built from one image ([#17](https://github.com/smk762/argus-mask/issues/17));
- a 2D talking-head model used by `record` ([#18](https://github.com/smk762/argus-mask/issues/18)).

The splat route has landed: a LAM head built from one portrait is served as a `.zip` beside the `.glb` heads (see [README.md § Faces](../README.md#faces)). The 2D talking-head route is still planned. This guide is about making inputs that work, and that you are allowed to use. Most of it also applies to a reference image for a new MakeHuman preset.

What these models do with your image:
- **One-shot avatar models** build a face from your image and move it by the *difference* from that face.
- **Talking-head models** warp the image itself.

Either way, whatever is in the image (a smile, a shadow, a strand of hair over an eye) becomes part of the face for good. A careful input saves more time than any later fix.

## Personal or synthetic

There are two tiers ([README.md § Personal and cleared](../README.md#personal-and-cleared)):

- **Personal:** a portrait for your own screens at home. Use whatever you may lawfully use privately: a character LoRA, a film still, a fan-art style. Record the inputs anyway (see [Recording provenance](#recording-provenance)) and build the face as `kind: personal`. Generate through [argus-darkroom](https://github.com/smk762/argus-darkroom), so the same legal floor applies as for any other image: what darkroom refuses is not a face either.
- **Synthetic, and so cleared:** a face you may show, share, demo or ship. It is a person who does not exist, so it involves no consent, and it has to be synthetic all the way through. The rest of this section is the bar for it.

The rule for showing a face is **no real person's face, and no owned character, without a release**. A synthetic face must meet all of these:

- **Inputs:**
  - Don't name real people, named characters, or "in the style of" a real person in the prompt.
  - No img2img, inpainting base, IP-Adapter, InstantID, PhotoMaker, or face-swap from a photo of a real person.
  - No LoRA or fine-tune trained on a real person's face or on a specific character. A character LoRA reproduces the character: the output is derived from it, however far the prompt pushes.
  
  A real person's face as an input makes the output *their* likeness, and a character as input makes it that character. Either can still be cleared, as `consented`, under a release (see [#19](https://github.com/smk762/argus-mask/issues/19)). Without one it is `personal`, never `synthetic`.
- **Look-alike check:** image models sometimes reproduce famous faces anyway. Before using a portrait, run a reverse image search and look at it honestly. If it looks like someone you could name, throw it away and change the seed. This is a best-effort check. The real guarantee is that nothing real went in.
- **18 and over.** Write the age into the prompt ("a woman in her twenties", "a man in his forties"), and put "child, minor" in the negative prompt. Discard anything that reads as under 18. This is the same floor the head recipe uses ([spec.md § The stock head](../spec.md#the-stock-head)).
- **The generator's license:**
  - SDXL base 1.0 is CreativeML Open RAIL++-M, which allows this use under its use restrictions.
  - Community "photoreal" fine-tunes each have their own license, and some don't say what they were trained on. Prefer the base model, or a fine-tune whose license and training data are stated.

## What a good source portrait looks like

| Want | Avoid | Why |
|---|---|---|
| Face straight to camera, head level | three-quarter view, tilt, looking away | The models assume a frontal neutral pose; yaw and tilt get baked in |
| Neutral expression, lips closed and relaxed | smile, open mouth, visible teeth, raised brows | Expression is animated as change from neutral; a smile in the source can never fully relax |
| Eyes open, looking into the lens | squint, half-closed lids, sideways gaze | Blinks and gaze are applied from this state |
| Soft, even, frontal light | hard key, rim light, coloured light, deep shadows | Light is baked into the face and moves with it; a shadow under the nose follows the nose when the head turns |
| Plain mid-grey background | busy scenes, bokeh, other people | Easy, clean matting |
| Forehead, ears, jaw and neck all visible; head and shoulders | hair over the eyes or brows, hats, glasses, hands, high collars | Anything covering the face gets fused into it |
| Light or no makeup, no face paint | heavy contour, glitter | Contour fights the viewer's own shading |
| Sharp throughout, natural skin texture | beauty filter, plastic skin, strong depth-of-field blur | Detail is what makes it read as real |

Keep long hair off the face and shoulders where you can. Loose strands are the hardest part for every route.

## Generating with SDXL

These starting points work well with SDXL base 1.0, in ComfyUI or diffusers. Tune them to your setup.

- **Size:** portrait shape at SDXL's native pixel count, 896×1152 or 832×1216. Don't generate at 512: the face comes out mushy.
- **Sampler:** DPM++ 2M Karras, 30–40 steps, CFG 5–7. A higher CFG gives the "airbrushed" look.
- **Refiner:** optional. It sharpens skin, but check it hasn't changed the identity.
- **Seeds:** fix the seed and change the prompt, or fix the prompt and sweep seeds. Record both either way.

Prompt template (swap in your own description):

```
studio portrait photograph of a woman in her thirties, short dark curly hair tied back,
head and shoulders, facing the camera, head level, neutral relaxed expression, lips closed,
eyes open looking into the lens, soft even diffused front lighting, plain mid-grey background,
sharp focus, natural skin texture, 85mm lens, f/8
```

Negative prompt:

```
smile, open mouth, teeth, raised eyebrows, head tilt, profile, three-quarter view, looking away,
glasses, hat, hands, hair over face, heavy makeup, harsh shadows, rim light, coloured light,
bokeh, blur, beauty filter, airbrushed, cartoon, 3d render, painting, watermark, text,
child, minor
```

Generate a batch (16–32), shortlist using the table above, then refine.

### Fixing a near-miss

- **Small blemishes in the image** (a stray hair, an earring): inpaint with the *same* model, prompt and seed, masked tightly. Don't inpaint the eyes or mouth. Regenerate instead.
- **Upscaling:** a 2–4× upscaler model, then downscale to what the target model wants. This beats generating at a large size.
- **Avoid face restorers** (GFPGAN, CodeFormer). They smooth skin and quietly change the identity, which is fatal if you need more than one view of the same person.
- **No real-person tools.** Never fix a portrait with a tool that takes a reference face, unless that reference is itself one of your synthetic portraits.

## For a LAM head

A LAM splat head ([lam/README.md](../lam/README.md)) is built from the one portrait, so it knows nothing the portrait did not show. Everything in the table above holds, and three things matter more:

- **Hair must cover the hairline and the temples.** LAM invents the sides of the head. Where the forehead runs bare up to a centre parting, or the hair is pulled back off the temples, a pale gap opens as soon as the head tips back or turns (from about 7° and 12° on `splat-01`). A fringe, or hair that sweeps across the top of the forehead and down past the temples, hides it: `syn-w101` holds to 20° either way. This overrides "forehead visible" above: the brows and eyes stay clear, the hairline does not have to.
- **Flat light, and no smile.** The head keeps the portrait's light and expression for good. Ask for a large soft light directly in front and a relaxed, closed mouth.
- **Head and shoulders, plain dark background.** The bust fades out below the collarbone, so a low neckline or a plain top reads best; the background is matted away.

### Guided flow (prefer this)

The editor's [guided head builder](../README.md#guided-head-builder) (`/guided.html`) runs source → portrait checks → LAM build → likeness / rest / turns / speech / age → pick → scene → save. It keeps the build record, refuses unsettled steps, measures turn limits, fits the rest set, and writes the face the server serves. Open it once darkroom, curator, retina OCR and the LAM image are configured (README § Configuration). Say the age in the subject (18 or over), and never a name.

The subsections below are what that flow checks and scores. Use them when reading a refusal on the page, or when you want the same checks by hand.

### By hand (CLI)

Make a batch, build a head from each, and pick from a contact sheet, never from the portraits alone: two portraits that look alike can give heads that fail in different places.

```bash
# 1. Portraits: FLUX.1 schnell through argus-darkroom, no LoRA, so the face is synthetic. One per seed, each with its record.
python lam/portraits.py --subject "a woman in her thirties, short dark hair with a fringe over the forehead, green eyes, \
    wearing a charcoal crew-neck top" --seeds 501-508 --out work/portraits

# 2. Heads, one per portrait (about 2 minutes and 9 GB of VRAM each).
lam/make_heads.sh $(for p in work/portraits/*.png; do n=$(basename $p .png); echo "s$n=$p"; done)

# 3. Add each as a face (argus-mask inspect --write-manifest), then draw the sheet from a running server.
python lam/sheet.py --url http://127.0.0.1:8109 --out work/sheet.jpg s501 s502 s503
```

The sheet draws each head with its limits off: from the front, turned 12° and 20° each way, tipped back and forward, and with the jaw at 0.35 and 0.6. Reject a head with a gap at the hairline or a smear at the temple inside 12°. If the best head still gives out early, give its manifest its own `limits` (README § Faces) and the viewer will keep it inside them.

Say the age in the subject (18 or over), and never a name. The subject is the only part of the prompt to change; the rest is the recipe.

### Pick the style: realistic character art, not a photograph

LAM keeps a face's outline (rendered at the portrait's own camera, the landmarks land within about 3% of the eyes' spacing) but softens what is inside it: the lids sit lower over the iris, the lips thin, the skin loses its fine detail. A photograph comes out of that heavier and older than it went in; an idealised face with smooth, painted skin survives it (#151). So `--style` picks the template around the subject (the same templates live in `argus_mask.portraits`, which the guided editor's source step uses too, #167):

| `--style` | Look | Verdict on the first batches |
|---|---|---|
| `character` | realistic adult proportions and light, a smooth painted finish | the pick: `lam-c1011` and `lam-c1022` |
| `stylised` | a digital painting, idealised | good; a little more illustrated |
| `photo` (CLI default) | a studio photograph | a believable portrait, a head that reads about ten years older |
| `semireal` | a 3D render | too doll-like, and reads young: don't use it |

The guided editor defaults to `character`. Every style is a text prompt only, so the face stays `synthetic`. Say "late twenties" or older in the subject: a painted style pulls a face younger. Reject a portrait with a signature or text in it (FLUX paints one now and then), however clean its face.

### Check the portrait before any GPU time

The guided editor's portrait step (#168) runs before LAM: curator `face_source` (one adult identity, frontal, sharp, eyes open), retina OCR for a painted signature, and mask's own framing / expression / lighting / temples / photograph advisory. A temple-bare crop and a picture with text are caught; a photograph is an advisory with a restyle offer (#167). A required check that could not run (`not_checked`) blocks the build.

### Score likeness, then fit its rest set

After the head is built, `lam/likeness.py` scores it against the portrait at the portrait's own camera (colour per region, DINOv2 identity, lid aperture, lip fullness, landmark misfit). The scores that ranked character heads above photo heads on #151's set are kept; detail did not separate them and is not. The guided editor runs this as a job after the build.

### Measure how far it turns

LAM invents the sides of the head. Where the hairline or temples were bare in the portrait, a gap opens as the head turns or tips back — each head at its own angle (`splat-01` ~12° yaw / ~7° back; `syn-w101` holds ~20° / 8°). The guided turns step (#170) draws the contact-sheet poses with limits off, scores the skin-share rise at the outer temple (yaw) and top hairline (tip back), and writes measured `limits` into the build record (copied onto the face manifest on save, #176):

```bash
python lam/measure_turns.py --url http://127.0.0.1:8109 --face splat-01 --out work/splat-01.turns.json
```

Needs playwright and pillow. The editor runs it as a job (`POST /guided/{id}/candidates/{candidate}/turns`).

### Compare apparent age

LAM brings a photograph out about ten years older; character art holds its age (#151). The guided age step (#172) asks argus-curator `POST /faces` (MiVOLO) for the age of the portrait-crop and of the head render at the portrait's camera, writes both and the difference, and flags when the head reads much older (default ≥ 7 years). This is a likeness score, not the 18-or-over check (that one is on the portrait before any build).

### Score the mouth in speech

A head can look right at rest and stutter in speech. Lip-parting shapes held under speech fought every closure (#161: lips closed 42% → 62% of frames, direction reversals 52 → 38 once those shapes drop out under speech, in line with `splat-01`). The guided speech step (#171) plays one fixed, cleared take (`talk.json`, no TTS) on the candidate and on `splat-01`, reads the lip gap per frame with MediaPipe, and writes closed-share / reversals / opening p95 into the record (stutter and oversized mouth are advisories):

```bash
python lam/measure_speech.py --url http://127.0.0.1:8109 --face splat-01 \
    --take src/argus_mask/editor/page/talk.json --model face_landmarker.task --out work/splat-01.speech.json
```

### Fit its rest set to its portrait

The viewer's default rest set (`warm`) closes the lids a little, which on a LAM head's already low lids reads as hooded. `lam/fit_rest.py` searches a rest set for each head instead: the viewer draws the head with each candidate, and MediaPipe scores it against the portrait (eyes, lips and brows placed against the outline; how much iris the lids show; how full the lips are; MediaPipe's own ARKit scores). About 3½ minutes per head on the CPU, against a running server:

```bash
python lam/fit_rest.py --url http://127.0.0.1:8109 --face lam-c1022 --portrait assets/lam-c1022/source/portrait.png \
    --model face_landmarker.task --out work/lam-c1022.rest.json
```

It writes a rest set (`fit-<face>`) to define in `expressions.json` and choose for the face, its score against `warm`'s, and both renders. A slight smile is kept where the portrait has one. A splat head is drawn with a long lens by default (23° from 1.6 m, as the portrait was taken): from close up at a wide angle the same head reads narrower and longer.

`FIT_SHAPE=1 lam/make_heads.sh ...` also searches the head's FLAME shape against the portrait at its own camera (about 10 minutes more per head). It helped `lam-s622` a little (misfit 0.033 to 0.026); it is off by default.

## More than one view

One-shot routes need only the single frontal portrait. Prefer them.

Some splat methods train on many views of the same person. SDXL can't hold one identity across separate generations. Two ways that keep it synthetic:

1. **Many views from your portrait:** run your frontal portrait through a multi-view or image-to-video model to get a slow turntable. Then check every frame against the frontal one for identity drift. A face embedding distance gives a quick number.
2. **Identity-locked generation:** generate the extra views with an identity adapter (IP-Adapter FaceID, InstantID), using *your synthetic portrait* as the reference. Don't use these adapters with a photo of a real person.

Keep the lighting and background identical across views. View-to-view lighting changes look like texture to the trainer.

## Recording provenance

A face is served only with provenance, so record it **as you generate** and keep it beside the image. Reconstructing it later doesn't work. `assets/` is gitignored, so none of this is committed.

```
assets/<face-id>/source/
  portrait.png          # the chosen image, at full size, unedited
  portrait.final.png    # after any inpaint and upscale
  generation.json       # see below
  workflow.json         # the ComfyUI workflow or the diffusers script, as run
```

Until the face has its `.glb` and `manifest.json`, the server lists `<face-id>` as refused (no manifest); that is expected. Don't keep `source/` inside a face the editor made (one with a `preset.json`): exporting it again replaces the whole directory, and deleting it moves the directory to `assets/.trash/`. Keep that face's source material in a directory of its own.

`generation.json` should hold everything needed to make the image again:
- the model, with its file name and sha256;
- any refiner, upscaler, LoRA or adapter, each with its sha256 and license;
- the prompt and negative prompt;
- the seed, sampler, steps, CFG and size;
- the tool versions;
- every edit made afterwards;
- the look-alike check you did, with its date.

When the face is built, the provenance carries this through. With today's `inspect`:

```bash
argus-mask inspect assets/<face-id>/head.glb --write-manifest \
    --id <face-id> --name "…" --kind synthetic \
    --source "Built from assets/<face-id>/source/portrait.final.png: SDXL base 1.0 (sha256 …), ComfyUI …, seed …; see generation.json" \
    --license "<license you apply> (generator: CreativeML Open RAIL++-M)" \
    --notes "A generated character, not a real person. No real-person inputs. Look-alike check: <what, when>."
```

A personal face needs only its source; the licence is optional:

```bash
argus-mask inspect assets/<face-id>/head.glb --write-manifest --id <face-id> --name "…" \
    --source "AureaNuance_XL (sha256 …) + a character LoRA (sha256 …), seed …; see generation.json"
```

The model, its sha256, the seed and the prompt also have fields of their own, in the provenance's `generator`. Add them to either command with `--generator "SDXL base 1.0" --generator-sha256 <sha256> --generator-seed <seed> --prompt "…"`. Only a `synthetic` or `personal` face takes a generator.

## Checklist

- [ ] Personal or synthetic decided, and `kind` set to match.
- [ ] For synthetic: no real person or named character in the prompt, the image inputs, or the LoRAs.
- [ ] The person reads as 18 or over.
- [ ] Look-alike check done and recorded.
- [ ] Frontal, level, neutral, lips closed, eyes open to the lens.
- [ ] Even light, plain background, nothing covering the face.
- [ ] Sharp, natural skin, no face restorer.
- [ ] `generation.json` and the workflow saved beside the image.
- [ ] Generator and add-on licenses allow the use, and are recorded.
