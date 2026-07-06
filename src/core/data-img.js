// Generated images: handwritten-style digits from a stroke model, geometric shapes and simple icons. Each image
// comes from its own seed (the set's seed and the image's index), so a set is identical every time it is made.
// Pixels are grey levels in [0, 1], 1 is ink; an image set is { images: Float32Array(n·w·h), w, h, y, get(i), X }.
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});
  const TAU = 2 * Math.PI;
  const arc = (cx, cy, rx, ry, a0, a1, k = 24) => Array.from({ length: k + 1 }, (_, i) => { const a = a0 + ((a1 - a0) * i) / k; return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)]; });

  // the strokes of each digit in a unit box (x to the right, y downwards)
  const DIGIT_STROKES = [
    () => [arc(0.5, 0.5, 0.27, 0.4, -Math.PI / 2, 1.5 * Math.PI, 32)],
    () => [[[0.36, 0.24], [0.52, 0.1], [0.52, 0.9]]],
    () => [[...arc(0.5, 0.33, 0.24, 0.22, Math.PI + 0.25, TAU + 0.55, 18), [0.24, 0.9], [0.78, 0.9]]],
    () => [[...arc(0.48, 0.3, 0.22, 0.19, Math.PI + 0.5, TAU + Math.PI / 2, 18), ...arc(0.48, 0.7, 0.26, 0.2, -Math.PI / 2, Math.PI - 0.5, 18)]],
    () => [[[0.64, 0.9], [0.64, 0.1], [0.2, 0.64], [0.82, 0.64]]],
    () => [[[0.74, 0.1], [0.34, 0.1], [0.3, 0.46], ...arc(0.48, 0.66, 0.27, 0.24, -2.3, Math.PI - 0.35, 20)]],
    () => [[[0.68, 0.1], [0.5, 0.2], [0.34, 0.4], ...arc(0.5, 0.68, 0.24, 0.22, Math.PI, 3 * Math.PI, 26)]],
    () => [[[0.22, 0.1], [0.78, 0.1], [0.42, 0.9]]],
    () => [arc(0.5, 0.29, 0.2, 0.19, 0, TAU, 24), arc(0.5, 0.7, 0.25, 0.21, 0, TAU, 26)],
    () => [[...arc(0.5, 0.33, 0.23, 0.22, 0, TAU, 26), [0.71, 0.6], [0.58, 0.9]]],
  ];

  const segDist2 = (px, py, ax, ay, bx, by) => {
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
    let t = L ? ((px - ax) * dx + (py - ay) * dy) / L : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
    const ex = ax + t * dx - px, ey = ay + t * dy - py; return ex * ex + ey * ey;
  };
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const inside = (poly, x, y) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
  const edgeDist = (poly, x, y, closed) => { let m = Infinity; const n = poly.length; for (let i = 0; i < (closed ? n : n - 1); i++) { const a = poly[i], b = poly[(i + 1) % n]; m = Math.min(m, segDist2(x, y, a[0], a[1], b[0], b[1])); } return Math.sqrt(m); };

  /**
   * paint parts into a size×size grey image; every part is { pts (pixel coordinates), mode: 'stroke' | 'fill' |
   * 'cut', width }. Coverage is anti-aliased by the distance from the pixel centre to the edge.
   */
  D.paint = (parts, size) => {
    const out = new Float32Array(size * size);
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
      const x = c + 0.5, y = r + 0.5; let v = 0;
      for (const p of parts) {
        if (p.mode === 'stroke') v = Math.max(v, clamp01(p.width / 2 + 0.5 - edgeDist(p.pts, x, y, false)));
        else { const d = edgeDist(p.pts, x, y, true), cov = inside(p.pts, x, y) ? clamp01(0.5 + d) : clamp01(0.5 - d); v = p.mode === 'cut' ? Math.min(v, 1 - cov) : Math.max(v, cov); }
      }
      out[r * size + c] = v;
    }
    return out;
  };

  /** a random affine map from the unit box into pixels: slant, rotation, scale and shift */
  const jitterMap = (rnd, size, box, amt) => {
    const slant = rnd.normal(0, 0.18 * amt), rot = rnd.normal(0, 0.1 * amt), sx = 1 - amt * rnd.range(0, 0.22), sy = 1 - amt * rnd.range(0, 0.12);
    const tx = rnd.normal(0, 0.035 * amt), ty = rnd.normal(0, 0.035 * amt), co = Math.cos(rot), si = Math.sin(rot), m = (size - box) / 2;
    return ([x, y]) => { let u = (x - 0.5) * sx, v = (y - 0.5) * sy; u -= slant * v; return [m + box * (0.5 + tx + co * u - si * v), m + box * (0.5 + ty + si * u + co * v)]; };
  };

  /**
   * drawDigit(d, {size: 28, seed: 1, jitter: 1}) → Float32Array(size²): the strokes of digit d, slanted, rotated,
   * scaled, shifted and wobbled at random, with a random pen thickness. jitter 0 draws the clean template.
   */
  D.drawDigit = (d, opt = {}) => {
    const size = opt.size || 28, amt = opt.jitter == null ? 1 : opt.jitter, rnd = D.rng(opt.seed == null ? 1 : opt.seed);
    const box = size * (size >= 28 ? 0.72 : 0.78), map = jitterMap(rnd, size, box, amt);
    const width = (size / 28) * (amt ? rnd.range(1.6, 3.4) : 2.4);
    const parts = DIGIT_STROKES[d]().map((stroke) => {
      const ox = rnd.normal(0, 0.02 * amt), oy = rnd.normal(0, 0.02 * amt), fx = rnd.range(0, TAU), wob = 0.018 * amt;
      return { mode: 'stroke', width, pts: stroke.map(([x, y], i) => map([x + ox + wob * Math.sin(fx + i * 0.5), y + oy + wob * Math.cos(fx + i * 0.37)])) };
    });
    return D.paint(parts, size);
  };

  const imageSet = (n, w, h, y, draw, extra) => {
    const images = new Float32Array(n * w * h);
    for (let i = 0; i < n; i++) images.set(draw(i), i * w * h);
    const get = (i) => images.subarray(i * w * h, (i + 1) * w * h);
    return Object.assign({ images, w, h, y, n, get, X: Array.from({ length: n }, (_, i) => get(i)), features: [{ name: 'pixels', type: 'image', w, h }] }, extra || {});
  };
  const balancedLabels = (n, k, rnd) => rnd.shuffle(D.even(n, k).flatMap((m, c) => new Array(m).fill(c)));
  const imgSeed = (seed, i) => (Math.imul(seed >>> 0, 2654435761) + i * 40503 + 17) >>> 0;

  D.digits = (opt = {}) => {
    const n = opt.n || 1000, size = opt.size || 28, seed = opt.seed || 1, rnd = D.rng(seed), y = balancedLabels(n, 10, rnd);
    return imageSet(n, size, size, y, (i) => D.drawDigit(y[i], { size, seed: imgSeed(seed, i), jitter: opt.jitter }), { classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], target: { name: 'digit', type: 'category' } });
  };
  D.register('digits', { title: 'Drawn digits', kind: 'image', story: 'Digits 0 to 9 drawn by a pen model: each digit is a few strokes, and every image gets its own slant, tilt, size, position, wobble and pen thickness.', rule: 'Template strokes per digit in a unit box; per image a random shear (sd 0.18), rotation (sd 0.1 rad), scale (x 0.78–1, y 0.88–1), shift (sd 3.5%), per-stroke offset and sinusoidal wobble, pen width 1.6–3.4 px at 28×28; painted with anti-aliased edges. Image i uses its own seed from the set seed and i.', source: 'Generated by Nooral from a seed', licence: 'GPL-3.0-or-later (part of Nooral)', size: '1000 images by default, 28×28 or 16×16 (opt.size)', features: ['pixels (grey 0–1)'], target: 'digit 0–9', balance: 'equal tenths (the first digits take any remainder)' }, D.digits);

  // shapes: polygons in a unit box around (0, 0) with radius 1
  const ngon = (k, r0 = 0) => Array.from({ length: k }, (_, i) => [Math.cos(r0 + (TAU * i) / k), Math.sin(r0 + (TAU * i) / k)]);
  const SHAPES = {
    circle: () => [ngon(40)],
    square: () => [ngon(4, Math.PI / 4).map(([x, y]) => [x * 0.95, y * 0.95])],
    triangle: () => [ngon(3, -Math.PI / 2).map(([x, y]) => [x, y + 0.2])],
    star: () => [Array.from({ length: 10 }, (_, i) => { const a = -Math.PI / 2 + (TAU * i) / 10, r = i % 2 ? 0.45 : 1; return [r * Math.cos(a), r * Math.sin(a)]; })],
  };
  D.shapeNames = Object.keys(SHAPES);
  /** drawShape(name, {size: 28, seed, outline: false}) → Float32Array: a shape at a random size, place and angle */
  D.drawShape = (name, opt = {}) => {
    const size = opt.size || 28, rnd = D.rng(opt.seed == null ? 1 : opt.seed), rad = size * rnd.range(0.22, 0.38), rot = name === 'circle' ? 0 : rnd.range(-0.5, 0.5);
    const cx = size / 2 + rnd.range(-1, 1) * (size / 2 - rad - 1), cy = size / 2 + rnd.range(-1, 1) * (size / 2 - rad - 1), co = Math.cos(rot), si = Math.sin(rot);
    const parts = SHAPES[name]().map((poly) => ({ mode: opt.outline ? 'stroke' : 'fill', width: (size / 28) * 2.2, pts: (opt.outline ? [...poly, poly[0]] : poly).map(([x, y]) => [cx + rad * (co * x - si * y), cy + rad * (si * x + co * y)]) }));
    return D.paint(parts, size);
  };
  D.shapes = (opt = {}) => {
    const n = opt.n || 400, size = opt.size || 28, seed = opt.seed || 1, rnd = D.rng(seed), y = balancedLabels(n, 4, rnd);
    return imageSet(n, size, size, y, (i) => D.drawShape(D.shapeNames[y[i]], { size, seed: imgSeed(seed, i), outline: opt.outline }), { classes: D.shapeNames.slice(), target: { name: 'shape', type: 'category' } });
  };
  D.register('shapes', { title: 'Shapes', kind: 'image', story: 'Circles, squares, triangles and stars of different sizes, places and angles.', rule: 'Radius 22–38% of the image, centre anywhere the shape fits, rotation ±0.5 rad (not for circles); filled, or outlined with opt.outline.', source: 'Generated by Nooral from a seed', licence: 'GPL-3.0-or-later (part of Nooral)', size: '400 images by default, 28×28 or 16×16', features: ['pixels (grey 0–1)'], target: 'shape (circle, square, triangle, star)', balance: 'equal quarters' }, D.shapes);

  // icons: unions of filled parts in a unit box (0..1), with cut-outs
  const heart = Array.from({ length: 40 }, (_, i) => { const t = (TAU * i) / 40, x = 16 * Math.sin(t) ** 3, y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t); return [0.5 + x / 36, 0.47 - y / 36]; });
  const ICONS = {
    house: [{ mode: 'fill', pts: [[0.2, 0.5], [0.8, 0.5], [0.8, 0.88], [0.2, 0.88]] }, { mode: 'fill', pts: [[0.1, 0.52], [0.5, 0.14], [0.9, 0.52]] }, { mode: 'cut', pts: [[0.43, 0.66], [0.57, 0.66], [0.57, 0.88], [0.43, 0.88]] }],
    heart: [{ mode: 'fill', pts: heart }],
    arrow: [{ mode: 'fill', pts: [[0.1, 0.42], [0.55, 0.42], [0.55, 0.22], [0.9, 0.5], [0.55, 0.78], [0.55, 0.58], [0.1, 0.58]] }],
    plus: [{ mode: 'fill', pts: [[0.4, 0.12], [0.6, 0.12], [0.6, 0.4], [0.88, 0.4], [0.88, 0.6], [0.6, 0.6], [0.6, 0.88], [0.4, 0.88], [0.4, 0.6], [0.12, 0.6], [0.12, 0.4], [0.4, 0.4]] }],
    moon: [{ mode: 'fill', pts: ngon(40).map(([x, y]) => [0.5 + 0.38 * x, 0.5 + 0.38 * y]) }, { mode: 'cut', pts: ngon(40).map(([x, y]) => [0.66 + 0.32 * x, 0.4 + 0.32 * y]) }],
    tree: [{ mode: 'fill', pts: [[0.5, 0.08], [0.82, 0.68], [0.18, 0.68]] }, { mode: 'fill', pts: [[0.43, 0.66], [0.57, 0.66], [0.57, 0.92], [0.43, 0.92]] }],
  };
  D.iconNames = Object.keys(ICONS);
  /** drawIcon(name, {size: 28, seed}) → Float32Array: an icon with a little random scale, shift and tilt */
  D.drawIcon = (name, opt = {}) => {
    const size = opt.size || 28, rnd = D.rng(opt.seed == null ? 1 : opt.seed), map = jitterMap(rnd, size, size * 0.86, opt.jitter == null ? 0.7 : opt.jitter);
    return D.paint(ICONS[name].map((p) => ({ mode: p.mode, width: 2, pts: p.pts.map(map) })), size);
  };
  D.icons = (opt = {}) => {
    const n = opt.n || 300, size = opt.size || 28, seed = opt.seed || 1, rnd = D.rng(seed), y = balancedLabels(n, D.iconNames.length, rnd);
    return imageSet(n, size, size, y, (i) => D.drawIcon(D.iconNames[y[i]], { size, seed: imgSeed(seed, i) }), { classes: D.iconNames.slice(), target: { name: 'icon', type: 'category' } });
  };
  D.register('icons', { title: 'Icons', kind: 'image', story: 'Six simple icons (house, heart, arrow, plus, moon, tree), each drawn a little tilted, shifted and resized.', rule: 'Fixed filled outlines (the house door and the moon\'s bite are cut out) under a random slant, tilt, scale and shift at 0.7 of the digit jitter.', source: 'Generated by Nooral from a seed', licence: 'GPL-3.0-or-later (part of Nooral)', size: '300 images by default, 28×28 or 16×16', features: ['pixels (grey 0–1)'], target: 'icon (house, heart, arrow, plus, moon, tree)', balance: 'equal sixths' }, D.icons);

  /** a picture of an image as text rows (for tests and quick looks): ' .:-=+*#%@' by grey level */
  D.ascii = (px, w) => { const ramp = ' .:-=+*#%@', rows = []; for (let i = 0; i < px.length; i += w) rows.push(Array.from(px.subarray ? px.subarray(i, i + w) : px.slice(i, i + w), (v) => ramp[Math.min(9, Math.floor(v * 10))]).join('')); return rows.join('\n'); };
})(globalThis.Nooral = globalThis.Nooral || {});
