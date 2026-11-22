# Splat heads in the viewer: the renderer's traps

`viewer/splat.js` draws `lam-splat` faces with Myned AI's gsplat-flame-avatar-renderer (1.4.0, vendored). It works around these traps. Read this before changing `splat.js` or upgrading the renderer.

1. **Scale.** LAM's `skin.glb` keeps a 0.01 scale on its node while its vertices are in metres. `lipBudget` multiplies by world scale, so `readSkin` measures with every node's scale reset to 1. Otherwise the budget is null.
2. **The neck.** The renderer's skeleton is a full-body rig (`neckLower`, `neckUpper`, `head`), not FLAME's `neck`, so `getNeckPose({neck})` moves nothing. `SplatHead.pose()` (the `getNeckPose` callback) restores the rest pose from the file and turns those bones itself, about world axes. It then calls `boneRoot.updateMatrixWorld`, because the renderer reads the skeleton before it updates world matrices, and a pose would otherwise show a frame late.
3. **LAM's clips must not move anything.** Don't call `getChatState`, and keep the mixer's actions stopped.
4. **The render loop.** The renderer's rAF loop is cancelled after every `render()`. The page drives it: `step()` live, `exact()` in record mode.
5. **Exact frames.** `exact()` waits for any depth sort in flight (a worker), runs one for this pose and camera, and waits for it. Then it draws in a rAF. Only after that may `record` take a screenshot.
6. **Resizing.** The renderer's ResizeObserver is disconnected; `fit()` sizes the canvas before each draw. The observer's callback runs after rAF callbacks, and resizing clears the canvas, so a frame drawn that tick would be presented blank.
7. **Loading.** `create()` resolves after the first section of a progressive load. Wait for `!isLoadingOrUnloading() && splatMesh.finalBuild`, or the first recorded frames race the final build.
8. **The reveal.** Record mode sets the reveal to Instant, because the fade-in reads the clock.
9. **URLs.** The renderer wants a URL ending in `.zip`, because it parses a name from it. That is why `/face/{id}/model/{file}` exists.
10. **Backdrops.** A backdrop is the background of the renderer's own `threeScene`, rendered with its WebGL context (`setScene`). The renderer draws that scene only if a child is visible, so an empty group stands in. Over a backdrop, the camera moves to `SPLAT_CLOSE` unless the manifest frames the face.
11. **One build at a time.** Two splat renderers building at once stall each other, so the editor's stages queue their builds (`oneSplatAtATime`).
12. **Controls.** The renderer builds no OrbitControls (its `setupControls` is commented out). If a later version does, never let them `listenToKeyEvents(window)`: W, A, S and D would be lost to text boxes.
13. **The head turn is limited.** A LAM head is made from one portrait, so the side it never showed smears: past about 20° of yaw the far temple, tipped back past about 10° a pale gap at the hairline (sooner where the hair is parted off the forehead). `SplatHead.set()` eases every turn towards the face's limits (`manifest.limits`, else `TURN` in `splat.js`) and never passes them. This lives in the viewer, never in a take. `?turn=0` switches it off, to see where a head gives out.
14. **The mouth has no inside.** Past a `jawOpen` of about 0.35 to 0.5 every LAM head shows a flat pink mouth, so the jaw cap is not a per-head tuning problem. `limits.jaw` moves the knee for a face; `?jaw=` overrides it.
15. **The soft edge.** The splats thin into ragged wisps below the collarbone. `SplatHead.veil()` draws a strip over the bottom of the picture after the splats, with the same WebGL context, fading to what is behind the head between `FADE.top` and `FADE.bottom` (heights in the head's space). It shows the backdrop's own texture, or the background colour, as three.js draws a scene's background, so the join can't be seen. It is placed from the camera alone and drawn in `step()` and `exact()`, so a recording has it.

It needs no COOP/COEP headers: it rendered on a non-secure, non-isolated origin in headless Chromium.
