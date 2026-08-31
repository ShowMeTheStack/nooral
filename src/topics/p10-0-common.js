// Part 10, convolutional networks: the families, the stored training runs (S.P10D, made by tests/ref/p10_runs.mjs)
// read back into weights, images and curves, the trained networks rebuilt from them, the small pictures and filters
// the convolution topics start from, and the pieces of picture the topics share (an image as a raster, a grid of
// numbers coloured by sign, a window outline).
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'cnn', order, title, blurb });
  fam('p10-conv', 10, 'Convolution and pooling', 'A small filter sliding over a picture, one dot product per place; padding, stride and dilation, which decide where it lands; pooling, which shrinks a map; and filters that read several channels at once.');
  fam('p10-inside', 20, 'Inside a trained network', 'A digit flowing through a network trained here, map by map; the filters it learned, drawn as pictures; and how far each unit can see.');
  fam('p10-train', 30, 'Training a CNN', 'A small network trained on generated digits and shapes, with its curves and its confusion matrix; augmentation, which stretches a small training set; and residual connections, which let gradients through deep stacks.');
  fam('p10-explain', 40, 'What the network looks at', 'Saliency maps from the gradient of a class score with respect to every pixel, and Grad-CAM from the last convolution’s maps weighted by their gradients.');

  const P = (S.P10 = {}), H = () => S.Kit.ml;
  const memo = new Map();
  P.once = (key, f) => { if (!memo.has(key)) memo.set(key, f()); return memo.get(key); };
  P.D = () => S.P10D;
  P.r2 = (v) => Math.round(v * 100) / 100;

  // ---- base64 by hand (the checks run without atob or Buffer) ----------------------------------------------------
  const AL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/', REV = new Int16Array(128).fill(-1);
  for (let i = 0; i < 64; i++) REV[AL.charCodeAt(i)] = i;
  P.bytes = (str) => {
    const clean = str.replace(/=+$/, ''), n = Math.floor((clean.length * 3) / 4), out = new Uint8Array(n);
    let buf = 0, bits = 0, k = 0;
    for (let i = 0; i < clean.length; i++) { buf = ((buf << 6) | REV[clean.charCodeAt(i)]) & 0xffffff; bits += 6; if (bits >= 8) { bits -= 8; out[k++] = (buf >> bits) & 255; } }
    return out;
  };
  P.u8 = (str) => Float32Array.from(P.bytes(str), (b) => b / 255);
  P.f32 = (str) => { const b = P.bytes(str); return new Float32Array(b.buffer, 0, b.length >> 2).slice(); };

  // ---- the trained networks --------------------------------------------------------------------------------------
  P.SETS = { digits: 'generated digits', shapes: 'generated shapes' };
  /** the trained CNN of a run (digits or shapes), rebuilt from its stored weights */
  P.net = (run) => P.once('net:' + run, () => {
    const J = P.D()[run], net = S.DL.build('cnn', { size: 16, channels: 1, classes: J.classes.length, filters: [6, 12], kernel: 5, hidden: [32], seed: 1 });
    const state = {};
    for (const [n, w] of Object.entries(J.weights)) state[n] = { shape: w.shape, data: Array.from(P.f32(w.d)) };
    net.loadState(state);
    net.eval();
    return net;
  });
  /** the validation images of a run: [{px, y}] */
  P.images = (run) => P.once('img:' + run, () => { const J = P.D()[run], all = P.u8(J.images); return [...J.y].map((c, i) => ({ px: all.subarray(i * 256, (i + 1) * 256), y: +c })); });
  P.classes = (run) => P.D()[run].classes.map(String);
  /** the forward pass of one image through a trained net: every named output, as plain arrays with shapes */
  P.forward = (run, px) => {
    const net = P.net(run), x = S.T.tensor(Array.from(px), [1, 1, 16, 16]);
    net.predict(x);
    const o = {};
    for (const [k, t] of Object.entries(net.outputs)) o[k] = { shape: t.shape.slice(), v: Array.from(t.values()) };
    const lg = o.logits.v, m = Math.max(...lg), e = lg.map((z) => Math.exp(z - m)), s = e.reduce((a, b) => a + b, 0);
    o.probs = e.map((z) => z / s);
    return o;
  };
  /** the parameters of a trained net by name, as {shape, v} */
  P.param = (run, name) => { const p = P.net(run).namedParameters().find(([n]) => n === name)[1]; return { shape: p.shape.slice(), v: Array.from(p.values()) }; };
  /** a choice of image: the k-th validation image of class c */
  P.pick = (run, c, k) => { const all = P.images(run), list = all.map((e, i) => [e, i]).filter(([e]) => e.y === c); const [e, i] = list[Math.min(k, list.length - 1)]; return { px: e.px, y: e.y, index: i }; };

  // ---- small pictures and filters for the convolution topics -----------------------------------------------------
  const grid = (rows) => rows.map((r) => [...r].map((c) => (c === '#' ? 1 : c === '+' ? 0.5 : 0)));
  P.PICTURES = {
    bar: { label: 'a vertical bar (6 × 6)', px: grid(['..##..', '..##..', '..##..', '..##..', '..##..', '..##..']) },
    corner: { label: 'a corner (6 × 6)', px: grid(['......', '.####.', '.#....', '.#....', '.#....', '......']) },
    diagonal: { label: 'a diagonal stroke (6 × 6)', px: grid(['#+....', '+#+...', '.+#+..', '..+#+.', '...+#+', '....+#']) },
    plus: { label: 'a plus sign (7 × 7)', px: grid(['...#...', '...#...', '...#...', '#######', '...#...', '...#...', '...#...']) },
    seven: { label: 'a small 7 (7 × 7)', px: grid(['.#####.', '.....#.', '....#..', '...#...', '...#...', '..#....', '..#....']) },
  };
  P.KERNELS = {
    vedge: { label: 'vertical edge', w: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
    hedge: { label: 'horizontal edge', w: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
    diag: { label: 'diagonal line', w: [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]] },
    blur: { label: 'box blur (every weight one ninth)', w: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)) },
    sharpen: { label: 'sharpen', w: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  };
  P.options = (o) => Object.keys(o).map((k) => ({ value: k, label: o[k].label }));

  // ---- pictures -----------------------------------------------------------------------------------------------------
  /** an image of n × n grey levels as a raster, top-left at (x, y), w wide */
  P.img = (key, x, y, w, px, o) => {
    o = o || {};
    const n = o.n || Math.round(Math.sqrt(px.length)), m = o.m || n;
    return { key, kind: 'raster', x: P.r2(x), y: P.r2(y), w, h: o.h || (w * m) / n, nx: n, ny: m, vals: Array.from(px), scheme: o.scheme || 'ink', max: o.max || 1, smooth: false, cls: 'p10img ' + (o.cls || ''), layer: o.layer || 'items' };
  };
  /** a matrix of numbers drawn as cells coloured by sign and size (rows of numbers) */
  P.cells = (prefix, x, y, M, o) => {
    o = o || {};
    return S.Kit.heat.grid({ prefix, x, y, cw: o.cw || 34, ch: o.ch || o.cw || 34, vals: M, scheme: o.scheme || 'div', max: o.max || 1, gap: o.gap === undefined ? 2 : o.gap, size: o.size, fmt: o.fmt === undefined ? (v) => P.num(v, o.d) : o.fmt, cls: o.cls, title: o.title, titleSize: o.titleSize, rows: o.rows, cols: o.cols, rx: o.rx });
  };
  /** a short number for a cell: 1, 0.5, −1.33 (at most d decimals, no trailing zeros) */
  P.num = (v, d) => { const s = String(+(+v).toFixed(d === undefined ? 2 : d)); return (s === '-0' ? '0' : s).replace(/^-/, '−'); };
  /** a rectangle outline (a window, a field) */
  P.box = (key, x, y, w, h, cls, layer) => ({ key, kind: 'path', d: `M${P.r2(x)},${P.r2(y)} h${P.r2(w)} v${P.r2(h)} h${P.r2(-w)} Z`, cls: 'win ' + (cls || ''), audit: 'ignore', layer: layer || 'top', box: { x, y, w, h } });
  P.text = (key, x, y, text, cls, size, anchor) => H().text(key, x, y, text, cls, size, anchor);
  /** rows of an n×n flat array */
  P.rows = (v, n, m) => Array.from({ length: m || n }, (_, i) => Array.from(v.slice(i * n, (i + 1) * n)));
  /** the k-th n×n map of a flat [C, n, n] array */
  P.map = (v, k, n) => v.slice(k * n * n, (k + 1) * n * n);
  P.maxAbs = (v) => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0) || 1;
})(globalThis.Nooral = globalThis.Nooral || {});
(function (S) {
  'use strict';
  const P = S.P10;
  /** the three inputs that choose one validation image: the set, the class and which example of it */
  P.imageInputs = (set, cls, k) => [
    { id: 'set', kind: 'choice', label: 'Images', def: set, options: Object.keys(P.SETS).map((v) => ({ value: v, label: P.SETS[v] })) },
    { id: 'cls', kind: 'number', label: 'Class (digit, or shape 0 circle, 1 square, 2 triangle, 3 star)', def: cls, min: 0, max: 9, check: (v, inp) => (v >= P.D()[inp.set].classes.length ? 'Class: the shapes are 0 to ' + (P.D()[inp.set].classes.length - 1) : null) },
    { id: 'k', kind: 'number', label: 'Example', def: k, min: 0, max: 20 },
  ];
  /** a class's name: “7”, “square” */
  P.cname = (run, c) => String(P.D()[run].classes[c]);
  /** “the digit 7” / “a square” */
  P.cwords = (run, c) => (run === 'digits' ? 'the digit ' + c : (/^[aeiou]/.test(P.cname(run, c)) ? 'an ' : 'a ') + P.cname(run, c));
  /** bars of class chances, the largest lit: from (x, y), rows rowH apart */
  P.chances = (prefix, x, y, run, probs, o) => {
    o = o || {};
    const names = P.D()[run].classes.map(String), best = probs.indexOf(Math.max(...probs));
    return S.Kit.dt.hbars({ prefix, x, y, w: o.w || 210, rowH: o.rowH || 30, vals: probs, names, max: 1, nameW: o.nameW || (run === 'digits' ? 22 : 60), size: 12, fmt: (v) => Math.round(v * 100) + '%', cls: (i) => (i === best ? 'p10best' : 'p10other') });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
(function (S) {
  'use strict';
  const P = S.P10;
  /** a class's name in narration: a digit is quoted (the claims check reads it back), a shape is a word */
  P.qname = (q, run, c) => (run === 'digits' ? q.n(c) : P.cname(run, c));
  /** “a square” / “an 8” */
  P.an = (w) => (/^(8|11|18|[aeiou])/i.test(String(w)) ? 'an ' : 'a ') + w;
})(globalThis.Nooral = globalThis.Nooral || {});
(function (S) {
  'use strict';
  const P = S.P10;
  /** “the digit 7” (the 7 quoted) / “a square” */
  P.qwords = (q, run, c) => (run === 'digits' ? 'the digit ' + q.n(c) : P.cwords(run, c));
})(globalThis.Nooral = globalThis.Nooral || {});
