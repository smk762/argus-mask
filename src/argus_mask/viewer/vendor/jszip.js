// JSZip 3.10.1 (MIT or GPLv3: LICENSE-jszip.md) as an ES module, for the splat renderer's `import JSZip from 'jszip'`.
// The release ships CommonJS and a UMD build (jszip.min.js, unchanged); with no module system on the page, the UMD
// build sets window.JSZip, and this hands it on. Not part of the npm release: written for argus-mask.
import './jszip.min.js';

export default globalThis.JSZip;
