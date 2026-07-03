// Small, deterministic engines for the sample scenes of the demo page (S.Toy), until the full engines land: seeded
// random numbers, our own 2-D datasets, digits drawn by a stroke model, and a short original corpus.
(function (S) {
  'use strict';
  const T = (S.Toy = S.Toy || {});

  /** mulberry32, with normal() and int(n) */
  T.rng = function (seed) {
    let s = (seed === undefined ? 1 : seed) >>> 0;
    const r = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    r.normal = () => { if (spare !== null) { const v = spare; spare = null; return v; } let u = 0; while (u < 1e-12) u = r(); const v = r(), m = Math.sqrt(-2 * Math.log(u)); spare = m * Math.sin(2 * Math.PI * v); return m * Math.cos(2 * Math.PI * v); };
    r.int = (n) => Math.floor(r() * n);
    r.shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = r.int(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    return r;
  };

  /** two interleaving half-moons: {X: [[x, y]..], y: [0|1..]} */
  T.moons = function (n, noise, seed) {
    const r = T.rng(seed === undefined ? 7 : seed), X = [], y = [], half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
      const c = i < half ? 0 : 1, k = c ? i - half : i, m = c ? n - half : half, t = (Math.PI * (k + 0.5)) / m;
      const p = c ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)];
      X.push([p[0] + noise * r.normal(), p[1] + noise * r.normal()]);
      y.push(c);
    }
    const order = r.shuffle([...X.keys()]);
    return { X: order.map((i) => X[i]), y: order.map((i) => y[i]) };
  };
  /** Gaussian blobs around the given centres */
  T.blobs = function (n, centres, spread, seed) {
    const r = T.rng(seed === undefined ? 11 : seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const c = i % centres.length; X.push([centres[c][0] + spread * r.normal(), centres[c][1] + spread * r.normal()]); y.push(c); }
    return { X, y };
  };
  /** a split into train and test, shuffled with its own seed */
  T.split = function (data, testFrac, seed) {
    const r = T.rng(seed === undefined ? 3 : seed), idx = r.shuffle([...data.X.keys()]), nt = Math.round(idx.length * testFrac);
    const pick = (ix) => ({ X: ix.map((i) => data.X[i]), y: ix.map((i) => data.y[i]), idx: ix });
    return { train: pick(idx.slice(nt)), test: pick(idx.slice(0, nt)) };
  };

  // ---- digits drawn by a stroke model -------------------------------------------------------------------------
  // each digit is a few strokes in a unit square (y down); a sample is jittered, slanted, and drawn with a soft pen
  const STROKES = {
    0: [[[0.5, 0.12], [0.75, 0.25], [0.8, 0.5], [0.75, 0.75], [0.5, 0.88], [0.25, 0.75], [0.2, 0.5], [0.25, 0.25], [0.5, 0.12]]],
    1: [[[0.35, 0.28], [0.55, 0.12], [0.55, 0.88]]],
    2: [[[0.25, 0.28], [0.4, 0.14], [0.62, 0.14], [0.75, 0.3], [0.68, 0.5], [0.25, 0.86], [0.8, 0.86]]],
    3: [[[0.25, 0.18], [0.5, 0.12], [0.72, 0.22], [0.7, 0.4], [0.48, 0.5], [0.72, 0.6], [0.74, 0.78], [0.5, 0.88], [0.25, 0.82]]],
    4: [[[0.62, 0.88], [0.62, 0.12], [0.2, 0.62], [0.82, 0.62]]],
    7: [[[0.2, 0.15], [0.8, 0.15], [0.45, 0.88]], [[0.35, 0.5], [0.68, 0.5]]],
  };
  T.digit = function (d, size, seed) {
    const r = T.rng(seed === undefined ? 5 : seed), n = size || 16, img = new Array(n * n).fill(0);
    const slant = 0.12 * r.normal(), sx = 1 + 0.06 * r.normal(), sy = 1 + 0.05 * r.normal(), dx = 0.03 * r.normal(), dy = 0.03 * r.normal();
    const pen = 0.075 + 0.01 * r();
    const strokes = STROKES[d].map((st) => st.map(([x, y]) => [(x - 0.5) * sx + 0.5 + (y - 0.5) * -slant + dx + 0.015 * r.normal(), (y - 0.5) * sy + 0.5 + dy + 0.015 * r.normal()]));
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const px = (i + 0.5) / n, py = (j + 0.5) / n;
      let best = 9;
      for (const st of strokes) for (let k = 1; k < st.length; k++) {
        const [ax, ay] = st[k - 1], [bx, by] = st[k], vx = bx - ax, vy = by - ay, l2 = vx * vx + vy * vy;
        const t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (l2 || 1)));
        best = Math.min(best, Math.hypot(px - ax - t * vx, py - ay - t * vy));
      }
      img[j * n + i] = Math.max(0, Math.min(1, 1.25 - best / pen));
    }
    return img;
  };

  /** our own short corpus: plain sentences about a lighthouse keeper and her garden */
  T.corpus = 'the keeper rose at dawn and lit the lamp. she walked to the garden and the gulls cried over the rocks. ' +
    'she planted beans in rows and watered the rows at noon. at night the lamp turned and the ships passed the rocks. ' +
    'the garden grew green and the beans rose tall. she sat by the lamp and read to the cat. the cat slept on the warm stones. ' +
    'in the morning the sea was calm and the keeper sang. she counted the ships and wrote their names in a book. ' +
    'the wind turned cold and the gulls went home. she closed the door and the lamp kept turning in the dark. ';
})(globalThis.Nooral = globalThis.Nooral || {});
