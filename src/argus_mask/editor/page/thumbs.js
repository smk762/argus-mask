// Stills of faces for cards and lists: one hidden stage takes a still of each face in turn, so a page
// of faces needs no live canvas per face. Saved faces and Broadcast both use it.

import { Stage, modelUrl } from './stage.js';

let stage = null;   // made on first use: a page that shows no still builds no WebGL context
const cache = new Map();  // `${place}/${id}/${sha}/${w}x${h}` -> data URL
let queue = Promise.resolve();

// A still of *id* (from the editor's `place`: 'export' or 'preview') as a data URL, at rest, *width* × *height*.
// Rejects with the reason when the face cannot be loaded (the editor does not have it, the model is refused).
export function still(id, place = 'export', width = 240, height = 288) {
  const turn = queue.then(async () => {
    const reply = await fetch(`/face/${encodeURIComponent(id)}?from=${place}`);
    const manifest = await reply.json().catch(() => ({}));
    if (!reply.ok) throw new Error(String(manifest.detail ?? reply.statusText));
    const key = `${place}/${id}/${manifest.model.sha256}/${width}x${height}`;
    if (!cache.has(key)) {
      stage ??= new Stage({ animate: false });
      await stage.load(modelUrl(id, manifest, `from=${place}&v=${manifest.model.sha256}`), manifest.framing || {}, manifest.model.format, { limits: manifest.limits });
      cache.set(key, await stage.snapshot(width, height));
    }
    return cache.get(key);
  });
  queue = turn.catch(() => {});
  return turn;
}
