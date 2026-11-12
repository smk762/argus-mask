# Making a synthetic face for input

This guide covers making a **source portrait**: one still image of a person who does not exist, made with an image model such as SDXL. Photoreal face routes start from a portrait like this:

- a Gaussian-splat head avatar built from one image ([#17](https://github.com/smk762/argus-mask/issues/17));
- a 2D talking-head model used by `record` ([#18](https://github.com/smk762/argus-mask/issues/18)).

Neither route is in argus-mask yet. Today a served face must be a `.glb` (see [README.md § Faces](../README.md#faces)). This guide is about making inputs that will work, and that you are allowed to use, once a route lands. Most of it also applies to a reference image for a new MakeHuman preset.

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

[#19](https://github.com/smk762/argus-mask/issues/19) will add a dedicated provenance kind for generated faces, which carries the model, license, seed and prompt as fields. Until then they go in `source` and `notes` as above.

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
