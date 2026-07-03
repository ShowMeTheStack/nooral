// Pictures for the sample scenes (S.Toy): convolution and pooling on a small generated digit with filters written
// out by hand (edges and a corner), and a tiny variational autoencoder trained on generated strokes, whose
// two-number latent space can be walked and decoded.
(function (S) {
  'use strict';
  const T = (S.Toy = S.Toy || {});

  // ---- convolution ----------------------------------------------------------------------------------------------
  /** a valid (no padding) 2-D convolution of an n×n image (flat, rows first) with a k×k filter: an (n−k+1)² map */
  T.conv = function (img, n, filt, bias) {
    const k = Math.round(Math.sqrt(filt.length)), m = n - k + 1, out = new Array(m * m);
    for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
      let s = bias || 0;
      for (let b = 0; b < k; b++) for (let a = 0; a < k; a++) s += img[(j + b) * n + i + a] * filt[b * k + a];
      out[j * m + i] = s;
    }
    return out;
  };
  T.relu = (v) => v.map((x) => Math.max(0, x));
  /** 2×2 max pooling of an n×n map (n even, or the last row and column dropped) */
  T.pool = function (img, n) {
    const m = Math.floor(n / 2), out = new Array(m * m), arg = new Array(m * m);
    for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
      let best = -Infinity, at = 0;
      for (const [a, b] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const v = img[(2 * j + b) * n + 2 * i + a]; if (v > best) { best = v; at = (2 * j + b) * n + 2 * i + a; } }
      out[j * m + i] = best; arg[j * m + i] = at;
    }
    return { out, arg, m };
  };
  /** the first layer's filters: edges that rise to the right, fall to the right, run across, and a diagonal */
  T.filters1 = [
    { name: 'left edge', f: [-1, 0, 1, -1, 0, 1, -1, 0, 1] },
    { name: 'right edge', f: [1, 0, -1, 1, 0, -1, 1, 0, -1] },
    { name: 'top edge', f: [-1, -1, -1, 0, 0, 0, 1, 1, 1] },
    { name: 'diagonal', f: [0, 1, 1, -1, 0, 1, -1, -1, 0] },
  ];
  /** a little network on a 16×16 image: conv 3×3 ×4 → ReLU → pool → conv 3×3 (one filter over all four maps) */
  T.cnn = function (img) {
    const n = Math.round(Math.sqrt(img.length));
    const c1 = T.filters1.map((F) => T.conv(img, n, F.f, -0.4)), m1 = n - 2;
    const r1 = c1.map(T.relu), p1 = r1.map((x) => T.pool(x, m1)), m2 = p1[0].m;
    // the second layer: a "corner" detector that adds a left edge and a top edge seen together, minus the rest
    const w2 = [[0.6, 0.6, 0, 0.6, 0.6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0.6, 0.6, 0.6, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -0.4, 0, 0, 0, 0]];
    let c2 = new Array((m2 - 2) * (m2 - 2)).fill(-1);
    w2.forEach((f, ch) => { const part = T.conv(p1[ch].out, m2, f, 0); c2 = c2.map((v, i) => v + part[i]); });
    return { n, c1, r1, m1, p1: p1.map((p) => p.out), parg: p1.map((p) => p.arg), m2, w2, c2: T.relu(c2), m3: m2 - 2 };
  };

  // ---- a variational autoencoder ----------------------------------------------------------------------------------
  /** a stroke through the middle of a size×size picture, at an angle and of a length */
  T.stroke = function (size, angle, len) {
    const n = size, img = new Array(n * n), c = (n - 1) / 2, dx = Math.cos(angle) * len / 2, dy = Math.sin(angle) * len / 2;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const vx = 2 * dx, vy = 2 * dy, l2 = vx * vx + vy * vy, t = Math.max(0, Math.min(1, ((i - c + dx) * vx + (j - c + dy) * vy) / (l2 || 1)));
      const d = Math.hypot(i - (c - dx + t * vx), j - (c - dy + t * vy));
      img[j * n + i] = Math.max(0, Math.min(1, 1.3 - d / 0.9));
    }
    return img;
  };
  /**
   * Train an encoder 144 → 24 → (μ, log σ²) of 2 and a decoder 2 → 24 → 144 on strokes of every angle and length,
   * by Adam on the binary cross-entropy plus the KL term. Snapshots {step, loss, dec, enc} at the steps asked.
   */
  T.vae = function (o) {
    o = o || {};
    const size = o.size || 12, P = size * size, r = T.rng(o.seed || 17), data = [];
    for (let i = 0; i < (o.n || 300); i++) { const a = r() * Math.PI, l = 3 + r() * 8; data.push({ img: T.stroke(size, a, l), angle: a, len: l }); }
    const enc = T.dense([P, o.hidden || 24, 4], 31), dec = T.dense([2, o.hidden || 24, P], 37);
    const keep = o.keep || [0, 60, 250, 1000], snaps = [], batch = o.batch || 16, sig = T.sigmoid;
    for (let step = 0; step <= Math.max(...keep); step++) {
      const Ge = enc.zeros(), Gd = dec.zeros();
      let L = 0;
      for (let b = 0; b < batch; b++) {
        const x = data[r.int(data.length)].img, Ae = enc.forward(x), h = Ae[2], mu = [h[0], h[1]], lv = [Math.max(-6, Math.min(4, h[2])), Math.max(-6, Math.min(4, h[3]))];
        const eps = [r.normal(), r.normal()], sd = lv.map((v) => Math.exp(v / 2)), z = [mu[0] + sd[0] * eps[0], mu[1] + sd[1] * eps[1]];
        const Ad = dec.forward(z), logits = Ad[2], p = logits.map(sig);
        for (let k = 0; k < P; k++) L -= x[k] * Math.log(Math.max(1e-7, p[k])) + (1 - x[k]) * Math.log(Math.max(1e-7, 1 - p[k]));
        L += 0.5 * (mu[0] ** 2 + mu[1] ** 2 + Math.exp(lv[0]) + Math.exp(lv[1]) - 2 - lv[0] - lv[1]);
        const dz = dec.backward(Ad, p.map((v, k) => v - x[k]), Gd);
        // through the sampling: z = μ + σ·ε
        const dmu = [dz[0] + mu[0], dz[1] + mu[1]], dlv = [0, 1].map((k) => dz[k] * eps[k] * sd[k] / 2 + 0.5 * (Math.exp(lv[k]) - 1));
        enc.backward(Ae, [dmu[0], dmu[1], dlv[0], dlv[1]], Ge);
      }
      if (keep.includes(step)) snaps.push({ step, loss: L / batch, dec: dec.copy(), enc: enc.copy() });
      enc.step(Ge, 0.004, 1 / batch); dec.step(Gd, 0.004, 1 / batch);
    }
    const run = (Pm, x) => { let a = x; Pm.W.forEach((W, l) => { a = W.map((row, i) => { let s = Pm.b[l][i]; for (let j = 0; j < row.length; j++) s += row[j] * a[j]; return l < Pm.W.length - 1 ? Math.tanh(s) : s; }); }); return a; };
    const decode = (snap, z) => run(snap.dec, z).map(sig);
    const encode = (snap, x) => run(snap.enc, x).slice(0, 2);
    return { snaps, data, decode, encode, size };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
