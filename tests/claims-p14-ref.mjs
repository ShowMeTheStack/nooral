// Independent working-out for the part 14 claims check: the stored runs read with Node's own base64, the trained
// networks evaluated by hand-written matrix products (not the engine's tensors), the losses, distances and
// coverage written from their definitions, and the diffusion schedule from its formula.
export const b64 = (s) => new Uint8Array(Buffer.from(s, 'base64'));
export const u8 = (s) => Array.from(b64(s), (b) => b / 255);
export const i16 = (o) => { const b = Buffer.from(o.d, 'base64'), out = []; for (let i = 0; i < b.length; i += 2) out.push(b.readInt16LE(i) * o.s); return out; };
export const chunks = (a, n) => { const out = []; for (let i = 0; i < a.length; i += n) out.push(a.slice(i, i + n)); return out; };
export const pairs = (a) => chunks(a, 2);
export const rel = (a, b, t) => Math.abs(a - b) <= t * Math.max(1, Math.abs(a), Math.abs(b));

/** a dense layer: W stored [in, out] row by row, then b */
export const dense = (x, W, b, nOut) => { const out = new Float64Array(nOut); for (let j = 0; j < nOut; j++) out[j] = b[j]; for (let i = 0; i < x.length; i++) { const v = x[i]; if (v) for (let j = 0; j < nOut; j++) out[j] += v * W[i * nOut + j]; } return out; };
export const relu = (a) => a.map((v) => (v > 0 ? v : 0));
export const sigm = (v) => 1 / (1 + Math.exp(-v));

/** the VAE from its stored weights: enc 256 → 64 → 4 (μ, log σ²), dec 2 → 64 → 256 */
export function vaeNet(weights) {
  const [W1, b1, W2, b2, W3, b3, W4, b4] = weights.map(i16), h = b1.length;
  const enc = (x) => { const o = dense(relu(dense(x, W1, b1, h)), W2, b2, 4); return { mu: [o[0], o[1]], lv: [Math.max(-8, Math.min(6, o[2])), Math.max(-8, Math.min(6, o[3]))] }; };
  const logits = (z) => dense(relu(dense(z, W3, b3, h)), W4, b4, 256);
  return { enc, logits, dec: (z) => Array.from(logits(z), sigm) };
}
/** binary cross-entropy with logits, summed over the pixels */
export const bce = (lg, x) => { let s = 0; for (let k = 0; k < x.length; k++) { const l = lg[k]; s += Math.max(l, 0) - l * x[k] + Math.log1p(Math.exp(-Math.abs(l))); } return s; };
export const corr = (a, b) => { const n = a.length; let ma = 0, mb = 0; for (let k = 0; k < n; k++) { ma += a[k]; mb += b[k]; } ma /= n; mb /= n; let s = 0, sa = 0, sb = 0; for (let k = 0; k < n; k++) { const x = a[k] - ma, y = b[k] - mb; s += x * y; sa += x * x; sb += y * y; } return s / Math.sqrt(sa * sb || 1); };
export const classMeans = (X, y, digits) => Object.fromEntries(digits.map((d) => { const m = new Float64Array(256); let n = 0; y.forEach((c, i) => { if (c !== d) return; n++; for (let k = 0; k < 256; k++) m[k] += X[i * 256 + k]; }); return [d, Array.from(m, (v) => v / n)]; }));
export const likeBest = (im, means) => { let b = -1, bc = -Infinity; for (const [d, m] of Object.entries(means)) { const c = corr(im, m); if (c > bc) { bc = c; b = +d; } } return { digit: b, corr: bc }; };
export const ink = (px) => px.reduce((s, v) => s + v, 0);
export const mae = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;

/** coverage of modes by samples, from its definition: counted near a mode when within r of the nearest mode */
export function coverage(P, modes, r) {
  const counts = modes.map(() => 0), nearest = modes.map(() => 0);
  for (const p of P) { let b = 0, bd = Infinity; modes.forEach((m, i) => { const d = Math.hypot(p[0] - m[0], p[1] - m[1]); if (d < bd) { bd = d; b = i; } }); nearest[b]++; if (bd <= r) counts[b]++; }
  const fair = Math.max(1, 0.2 * (P.length / modes.length)), H = nearest.reduce((h, c) => (c ? h - (c / P.length) * Math.log(c / P.length) : h), 0);
  return { counts, covered: counts.filter((c) => c >= fair).length, effective: Math.exp(H) };
}
export const chamfer = (A, B) => A.reduce((s, a) => s + Math.min(...B.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1]))), 0) / A.length;
/** sliced Wasserstein-1 over k evenly spaced directions, sorted projections matched by quantile */
export function swd(A, B, k = 32) {
  let tot = 0;
  for (let d = 0; d < k; d++) {
    const th = (Math.PI * d) / k, pa = A.map((p) => p[0] * Math.cos(th) + p[1] * Math.sin(th)).sort((x, y) => x - y), pb = B.map((p) => p[0] * Math.cos(th) + p[1] * Math.sin(th)).sort((x, y) => x - y), m = Math.max(pa.length, pb.length);
    let s = 0; for (let i = 0; i < m; i++) s += Math.abs(pa[Math.floor((i * pa.length) / m)] - pb[Math.floor((i * pb.length) / m)]);
    tot += s / m;
  }
  return tot / k;
}
/** the diffusion schedule from its formulas */
export function schedule(n, kind) {
  const beta = [0], ab = [1];
  for (let t = 1; t <= n; t++) {
    if (kind === 'cosine') { const f = (u) => Math.cos(((u / n + 0.008) / 1.008) * (Math.PI / 2)) ** 2; beta.push(Math.min(0.999, 1 - f(t) / f(t - 1))); }
    else { const b0 = 1e-4 * (1000 / n), b1 = Math.min(0.5, 0.02 * (1000 / n)); beta.push(b0 + ((b1 - b0) * (t - 1)) / Math.max(1, n - 1)); }
    ab.push(ab[t - 1] * (1 - beta[t]));
  }
  return { beta, ab };
}
/** the 2-D denoiser from its stored weights: [x, time features] → 64 → 64 → 64 → 2 (ReLU) */
export function denoiser(weights, T, temb = 16) {
  const [W1, b1, W2, b2, W3, b3, W4, b4] = weights.map(i16), h = b1.length, half = temb / 2;
  const feat = (t) => { const out = new Array(temb); for (let j = 0; j < half; j++) { const f = Math.pow(1000, -j / Math.max(1, half - 1)), a = (t / T) * 1000 * f; out[j] = Math.sin(a); out[half + j] = Math.cos(a); } return out; };
  return (x, t) => Array.from(dense(relu(dense(relu(dense(relu(dense(x.concat(feat(t)), W1, b1, h)), W2, b2, h)), W3, b3, h)), W4, b4, 2));
}
