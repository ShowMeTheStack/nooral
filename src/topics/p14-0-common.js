// Part 14, generative models: the families, the stored training runs (S.P14D, made by tests/ref/p14_runs.mjs from
// the engines on fixed seeds) read back into numbers, images and weights, the models the topics run live (a trained
// VAE and a trained 2-D diffusion denoiser, rebuilt from their stored weights), and the pieces of picture the topics
// share: a 16×16 image, a row of images, a latent plane with the prior's rings, an image strip with names.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'generative', order, title, blurb });
  fam('p14-ae', 10, 'Autoencoders and VAEs', 'Squeezing an image through a few numbers and rebuilding it; a variational autoencoder whose codes form a map you can walk; the reparameterisation trick that lets gradients pass through a random draw; and the KL term pulling against reconstruction.');
  fam('p14-gan', 20, 'Generative adversarial networks', 'A generator and a discriminator in a duel on points in the plane and on small digits, the discriminator’s view of the plane, and the mode collapse that comes when the duel is unbalanced.');
  fam('p14-diffusion', 30, 'Diffusion', 'Data dissolved into noise step by step, a network that learns to predict the noise, and sampling that runs the chain backwards, on points in the plane and on 16×16 images.');
  fam('p14-families', 40, 'The families compared', 'Autoencoders, VAEs, GANs and diffusion side by side: what each learns, how each draws a new sample, and what it costs.');

  const P = (S.P14 = {}), H = () => S.Kit.ml;
  const memo = new Map();
  /** a value computed once per page (or per test run) and kept */
  P.once = (key, f) => { if (!memo.has(key)) memo.set(key, f()); return memo.get(key); };
  P.D = () => S.P14D;

  // ---- base64 by hand (the checks run without atob or Buffer) ----------------------------------------------------
  const AL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/', REV = new Int16Array(128).fill(-1);
  for (let i = 0; i < 64; i++) REV[AL.charCodeAt(i)] = i;
  P.bytes = (str) => {
    const clean = str.replace(/=+$/, ''), n = Math.floor((clean.length * 3) / 4), out = new Uint8Array(n);
    let buf = 0, bits = 0, k = 0;
    for (let i = 0; i < clean.length; i++) { buf = (buf << 6) | REV[clean.charCodeAt(i)]; bits += 6; if (bits >= 8) { bits -= 8; out[k++] = (buf >> bits) & 255; } }
    return out;
  };
  /** bytes as values in [0, 1] */
  P.u8 = (str) => Float32Array.from(P.bytes(str), (b) => b / 255);
  /** 16-bit integers with a scale: {s, d} */
  P.i16 = (o) => { const b = P.bytes(o.d), n = b.length / 2, out = new Float32Array(n); for (let i = 0; i < n; i++) { let v = b[2 * i] | (b[2 * i + 1] << 8); if (v >= 32768) v -= 65536; out[i] = v * o.s; } return out; };
  /** a flat block of images, 256 pixels each, as a list */
  P.split = (flat, n) => { n = n || 256; const out = []; for (let i = 0; i < flat.length; i += n) out.push(flat.subarray(i, i + n)); return out; };
  P.imgs = (str) => P.split(P.u8(str));
  P.pairs = (flat) => { const out = []; for (let i = 0; i < flat.length; i += 2) out.push([flat[i], flat[i + 1]]); return out; };
  P.ink = (px) => { let s = 0; for (const v of px) s += v; return s; };

  // ---- the models run live ---------------------------------------------------------------------------------------
  /** the trained 2-D VAE (β = 1) with its test images: S.Gen.vae built as in training, its weights set from the data */
  P.vae = () => P.once('vae', () => {
    const J = P.D().vae1, v = S.Gen.vae(Object.assign({}, J.o, { n: 32 }));
    S.Gen.setParams(v.net, J.weights.map(P.i16));
    v.map = v.latentMap();
    // the centre of each digit's codes
    v.centres = {};
    for (const c of [...new Set(v.map.map((m) => m.y))].sort()) { const ms = v.map.filter((m) => m.y === c); v.centres[c] = [ms.reduce((s, m) => s + m.z[0], 0) / ms.length, ms.reduce((s, m) => s + m.z[1], 0) / ms.length]; }
    return v;
  });
  /** the trained 2-D diffusion model: S.Gen.diffusion2d built as in training, weights from the data */
  P.diff2d = () => P.once('diff2d', () => {
    const J = P.D().diff2d, d = S.Gen.diffusion2d(J.o);
    S.Gen.setParams(d.net, J.weights.map(P.i16));
    return d;
  });
  /** the test digits the autoencoders were shown (as S.Gen.autoencoder makes them) */
  P.aeTest = () => P.once('aeTest', () => S.Gen.images('digits', { n: 128, size: 16, seed: 1001 }));

  // ---- pictures -----------------------------------------------------------------------------------------------------
  /** a 16×16 grey image as a raster, top-left at (x, y), w wide */
  P.img = (key, x, y, w, px, cls) => ({ key, kind: 'raster', x: H().r2(x), y: H().r2(y), w, h: w, nx: 16, ny: 16, vals: Array.from(px), scheme: 'ink', max: 1, smooth: false, cls: 'p14img ' + (cls || ''), layer: 'items' });
  /** a row of images with a gap; cls(i) adds classes (a lit one) */
  P.row = (prefix, x, y, w, list, gap, cls) => list.map((px, i) => P.img(prefix + ':' + i, x + i * (w + (gap === undefined ? 6 : gap)), y, w, px, cls ? cls(i) : ''));
  /** a frame round a lit image */
  P.ring = (key, x, y, w, cls) => ({ key, kind: 'path', d: `M${H().r2(x - 3)},${H().r2(y - 3)} h${w + 6} v${w + 6} h${-(w + 6)} Z`, cls: 'p14ring ' + (cls || ''), audit: 'ignore', layer: 'top', box: { x: x - 3, y: y - 3, w: w + 6, h: w + 6 } });
  /** the prior's 1σ and 2σ circles on latent axes */
  P.rings = (ax, key) => [1, 2].map((r) => {
    const pts = []; for (let i = 0; i <= 64; i++) { const t = (2 * Math.PI * i) / 64; pts.push([ax.X(r * Math.cos(t)), ax.Y(r * Math.sin(t))]); }
    return { key: key + r, kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ',' + p[1]).join(' ') + ' Z', cls: 'p14prior', pathAudit: 'edge-x', layer: 'lines', audit: 'ignore', box: H().boxOf(pts) };
  });
  /** latent axes over [−R, R]² */
  P.latentAxes = (prefix, x, y, w, R, o) => S.Kit.curve.axes(Object.assign({ prefix, x, y, w, h: w, xr: [-R, R], yr: [-R, R], xticks: S.Kit.curve.ticks(-R, R, 4), yticks: S.Kit.curve.ticks(-R, R, 4), xname: 'z₁', yname: 'z₂' }, o || {}));
  /** class colours: digits 0, 1, 3 and 7 of the VAE as c0..c3 */
  P.VAE_CLASSES = [0, 1, 3, 7];
  P.cOf = (digit) => 'c' + Math.max(0, P.VAE_CLASSES.indexOf(digit));
  P.digitName = (d) => ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'][d];
  /** "a 7" / "an 8" */
  P.aDigit = (d) => (d === 8 ? 'an ' : 'a ') + d;
  /** the same with the digit quoted for the narration */
  P.aQ = (q, d) => (d === 8 ? 'an ' : 'a ') + q.n(d);
  /** a step of a run as an index into a curve kept every k steps */
  P.at = (arr, step, k) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(step / k) - 1))];
  P.ticksN = (n) => S.Kit.curve.ticks(0, n, 4);
})(globalThis.Nooral = globalThis.Nooral || {});
