// Part 6, kernels: a ring round a disc cannot be split by a line, but lift every point to the height z = x₁² + x₂²
// and a flat plane splits them; the plane cuts back to a circle in the plane of the data. The kernel computes the
// dot products of lifted points without lifting. Then the polynomial kernel by degree, and the RBF kernel by γ,
// with what a too-large γ does to points it has not seen.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const lift = (x) => [x[0], x[1], x[0] * x[0] + x[1] * x[1]];

  S.topic({
    id: 'svm-kernel-lift', family: 'p06-kernels', order: 10, title: 'The kernel trick as a lift into 3-D',
    blurb: 'No straight line splits a ring from the disc inside it. Give every point a third coordinate, its squared distance from the centre, z = x₁² + x₂²: the disc stays low and the ring rises. Now a flat plane splits them, and where the plane meets the bowl is a circle, the curved boundary back in two dimensions. A kernel is the shortcut: K(x, x′) = φ(x)·φ(x′) gives the lifted dot product straight from x and x′.',
    inputs: [
      { id: 'n', kind: 'number', label: 'Points', def: 60, min: 20, max: 120 },
      { id: 'noise', kind: 'slider', label: 'Noise', def: 0.08, min: 0, max: 0.2, step: 0.01 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A clean ring', values: { n: 60, noise: 0.05, seed: 1 } },
      { label: 'A noisy ring', values: { n: 80, noise: 0.15, seed: 2 } },
      { label: 'Few points', values: { n: 24, noise: 0.08, seed: 3 } },
    ],
    counters: [{ id: 'dims', label: 'dimensions' }, { id: 'sv', label: 'support vectors' }],
    run(inp) {
      const d = S.Data.circles({ n: inp.n, noise: inp.noise, factor: 0.45, seed: inp.seed }), X = d.X, y = d.y, n = X.length;
      const Z = X.map(lift), F = H().frames(['dims', 'sv']), q = F.q;
      const flat = P.fit(X, y, { C: 1 }), up = P.fit(Z, y, { C: 10 });
      const [w1, w2, w3] = up.w, b = up.b;
      // the plane w·φ + b = 0 meets the bowl z = r² in a circle: w3·r² + w1·x₁ + w2·x₂ + b = 0
      const cx = -w1 / (2 * w3), cy = -w2 / (2 * w3), rad = Math.sqrt(Math.max(0, cx * cx + cy * cy - b / w3));
      const f2 = (x) => up.f(lift(x));
      const acc2 = ML().accuracy(y, X.map((x) => (f2(x) > 0 ? 1 : 0)));
      // the 3-D scene on the left
      const zs = 0.85, A = 1.45, view = (az, el) => H().view3({ cx: 215, cy: 290, s: 112, az, el });
      const scene = (V, rise, opts) => {
        opts = opts || {};
        // the floor of the box, the vertical axis at its far corner, and the names of the floor's edges
        const fl = [[-A, -A], [A, -A], [A, A], [-A, A]].map(([u, v]) => Object.assign(V([u, v, 0]), { u, v }));
        const items = [{ key: 'lf:floor', kind: 'path', d: fl.map((p, k) => (k ? 'L' : 'M') + p.x + ',' + p.y).join(' ') + ' Z', cls: 'axis', audit: 'ignore', layer: 'back', z: 1, box: H().boxOf(fl) }];
        const far = fl.reduce((a, p) => (p.d < a.d ? p : a), fl[0]), top = V([far.u, far.v, 1.5 * zs]);
        items.push({ key: 'lf:zax', kind: 'path', d: `M${far.x},${far.y} L${top.x},${top.y}`, cls: 'axis', audit: 'ignore', layer: 'back', z: 1 });
        items.push(H().text('lf:zn', top.x, top.y - 10, 'z', 'axname', 13, 'middle'));
        const c0 = V([0, 0, 0]);
        for (const [k, nm] of [[0, 'x₁'], [1, 'x₂']]) {
          // the nearer of the two floor edges that run along this axis
          const pairs = k === 0 ? [[fl[0], fl[1]], [fl[3], fl[2]]] : [[fl[0], fl[3]], [fl[1], fl[2]]];
          const near = pairs.reduce((a, p) => (p[0].d + p[1].d > a[0].d + a[1].d ? p : a), pairs[0]);
          const mx = (near[0].x + near[1].x) / 2, my = (near[0].y + near[1].y) / 2, dx = mx - c0.x, dy = my - c0.y, L = Math.hypot(dx, dy) || 1;
          items.push(H().text('lf:n' + k, mx + (dx / L) * 20, my + (dy / L) * 20 + 4, nm, 'axname', 13, 'middle'));
        }
        if (opts.rings) for (const L of [0.25, 0.6, 1, 1.4]) {
          const pts = []; for (let k = 0; k <= 48; k++) { const t = (2 * Math.PI * k) / 48, r = Math.sqrt(L); pts.push(V([r * Math.cos(t), r * Math.sin(t), L * zs])); }
          items.push({ key: 'lf:ring' + L, kind: 'path', d: pts.map((p, k) => (k ? 'L' : 'M') + p.x + ',' + p.y).join(' '), cls: 'mlring', audit: 'ignore', layer: 'back', z: 1 });
        }
        if (opts.plane) {
          const c = [[-1.4, -1.4], [1.4, -1.4], [1.4, 1.4], [-1.4, 1.4]].map(([u, v]) => V([u, v, (-(b + w1 * u + w2 * v) / w3) * zs]));
          items.push({ key: 'lf:plane', kind: 'path', d: c.map((p, k) => (k ? 'L' : 'M') + p.x + ',' + p.y).join(' ') + ' Z', cls: 'mlplane', audit: 'ignore', layer: 'back', z: 3, box: H().boxOf(c) });
        }
        Z.forEach((p, i) => { const s = V([p[0], p[1], rise * p[2] * zs]); items.push({ key: 'lf:p' + i, kind: 'dot', x: s.x, y: s.y, r: 4.4, cls: 'sp c' + y[i] + (opts.sv && up.alpha[i] > 0 ? ' ' : ''), audit: 'overlap-ok', layer: 'items', z: Math.round(s.d * 100) }); });
        if (opts.sv) for (const i of up.sv) { const s = V([Z[i][0], Z[i][1], Z[i][2] * zs]); items.push({ key: 'lf:r' + i, kind: 'dot', x: s.x, y: s.y, r: 8, cls: 'ring act', audit: 'overlap-ok', layer: 'items', z: 9999 }); }
        return items;
      };
      // the 2-D picture on the right
      const ax = P.axes('l2', X, { x: 520, y: 60, w: 260, h: 260 }, { xr: [-1.45, 1.45], yr: [-1.45, 1.45] });
      const right = (which) => {
        const items = [...ax.items];
        if (which === 'line') items.push(...P.boundary(ax, 'l2', flat, { n: 60, margins: false }));
        if (which === 'circle') items.push(...P.boundary(ax, 'l2', { f: f2 }, { n: 60, margins: false }));
        items.push(...S.Kit.scatter.points(ax, 'l2:p', X, y, { r: 3.4 }));
        items.push(H().text('l2:t', 520, 30, which === 'circle' ? 'back in 2-D: a circle' : 'in 2-D', 'strong', 14));
        return items;
      };
      F.set('dims', 2);
      F.push([...scene(view(0, Math.PI / 2), 0), ...right('line')], `${q.n(n)} points: an orange disc inside a blue ring, seen from above. The best straight line, on the right, gets only ${q.pct(flat.acc)} of them right: a line cannot go round the disc.`, { phase: 'flat', n, acc: flat.acc });
      F.push([...scene(view(0.5, 0.42), 0, { rings: false }), ...right('line')], `Tilt the view: the same points lie flat on the floor of a 3-D box. The third axis will be z = x₁² + x₂², the squared distance from the centre.`, { phase: 'tilt' });
      F.set('dims', 3);
      const zin = Math.max(...Z.filter((_, i) => y[i] === 1).map((p) => p[2])), zout = Math.min(...Z.filter((_, i) => y[i] === 0).map((p) => p[2]));
      F.push([...scene(view(0.5, 0.42), 1, { rings: true }), ...right('line')], `Lift every point to its height z = x₁² + x₂², on the bowl drawn in dashed rings. The disc stays low, up to z = ${q(zin)}, and the ring rises, from z = ${q(zout)} upwards${zout > zin ? ': a gap opens between them' : ': the noisiest points still mix, but most of the classes have parted'}.`, { phase: 'lift', zin, zout });
      F.set('sv', up.sv.length);
      F.push([...scene(view(0.9, 0.3), 1, { rings: true, plane: true, sv: true }), ...right('line')], `A linear SVM on the three coordinates finds a flat plane between the two layers: ${q.t(w1, 2, true)}·x₁${q.t(w2)}·x₂${q.t(w3)}·z${q.t(b)} = 0, with ${q.n(up.sv.length)} support vectors (ringed). It gets ${q.pct(up.acc)} of the lifted points right.`, { phase: 'plane', w: up.w, b, nsv: up.sv.length, acc: up.acc });
      F.push([...scene(view(0.9, 0.3), 1, { rings: true, plane: true, sv: true }), ...right('circle')], `Where the plane meets the bowl, z = x₁² + x₂², so the boundary in 2-D is ${q.t(w3, 2, true)}·(x₁² + x₂²)${q.t(w1)}·x₁${q.t(w2)}·x₂${q.t(b)} = 0: a circle centred at (${q(cx)}, ${q(cy)}) with radius ${q(rad, 3)}. A straight cut in 3-D is a curved cut in 2-D, and ${q.pct(acc2)} of the points are now right.`, { phase: 'circle', cx, cy, rad, acc: acc2, w: up.w, b });
      const i0 = 0, j0 = X.findIndex((_, j) => y[j] !== y[i0]), a = X[i0], c = X[j0];
      const viaPhi = lift(a).reduce((s, v, k) => s + v * lift(c)[k], 0), dotac = a[0] * c[0] + a[1] * c[1];
      const viaK = dotac + (a[0] * a[0] + a[1] * a[1]) * (c[0] * c[0] + c[1] * c[1]);
      const pair = [S.Kit.scatter.ring(ax, 'l2:a', a[0], a[1], 'hot', 8), S.Kit.scatter.ring(ax, 'l2:c', c[0], c[1], 'hot', 8), H().text('l2:an', ax.X(a[0]) + (a[0] > 0 ? 12 : -12), ax.Y(a[1]) - 10, 'x', 'strong', 14, a[0] > 0 ? 'start' : 'end'), H().text('l2:cn', ax.X(c[0]) + (c[0] > 0 ? 12 : -12), ax.Y(c[1]) - 10, 'x′', 'strong', 14, c[0] > 0 ? 'start' : 'end')];
      F.push([...scene(view(1.2, 0.28), 1, { rings: true, plane: true, sv: true }), ...right('circle'), ...pair], `The SVM only ever uses dot products of lifted points, so it never needs φ itself. Take x = (${q(a[0])}, ${q(a[1])}) and x′ = (${q(c[0])}, ${q(c[1])}): lifting both and multiplying gives φ(x)·φ(x′) = ${q(viaPhi, 4)}, and the kernel K(x, x′) = x·x′ + ‖x‖²‖x′‖² gives ${q(viaK, 4)} straight away. That shortcut is the kernel trick.`, { phase: 'kernel', a, c, viaPhi, viaK });
      // on a phone one picture at a time, large enough to read: the lift in 3-D, then (once the circle is found) the
      // boundary back in 2-D
      F.list.forEach((f, k) => { f.narrowView = k < 4 ? { x: -40, y: 60, w: 510, h: 450 } : { x: 470, y: 14, w: 330, h: 360 }; });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The lift and its kernel', text: ['φ(x₁, x₂) = (x₁, x₂, x₁² + x₂²)', 'a plane in 3-D: w·φ(x) + b = 0', 'in 2-D: w₃(x₁² + x₂²) + w₁x₁ + w₂x₂ + b = 0, a circle', 'K(x, x′) = φ(x)·φ(x′) = x·x′ + ‖x‖²‖x′‖²', 'the SVM needs only K, never φ'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'tilt', pick: 'option', ask: 'Each point rises to z = x₁² + x₂². Which class ends up higher?', options: ['the blue ring', 'the orange disc', 'they rise equally'], answer: 'the blue ring', explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'plane', pick: 'option', ask: 'What shape does the plane’s boundary make back in the 2-D picture?', options: ['a circle', 'a straight line', 'two lines', 'a parabola'], answer: 'a circle', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'For the lift φ(x) = (x₁, x₂, x₁² + x₂²), what is the kernel K(x, x′)?', options: ['x·x′ + ‖x‖²‖x′‖²', 'x·x′', '‖x − x′‖²', '(x·x′)²'], answer: 0, why: 'Multiply the coordinates pairwise and add: x₁x₁′ + x₂x₂′ + (x₁² + x₂²)(x₁′² + x₂′²).' },
      { q: 'Why can the kernel trick be cheaper than lifting the points?', options: ['K is computed from the original coordinates, however many features φ has', 'kernels need fewer points', 'kernels avoid the dual problem', 'lifting loses information'], answer: 0, why: 'For the RBF kernel φ even has infinitely many features, yet K is one exponential.' },
      { q: 'A flat plane in the lifted space becomes what in the original plane?', options: ['a curve: here a circle', 'always a straight line', 'a single point', 'nothing: it cannot be drawn'], answer: 0, why: 'The plane is linear in φ, and φ is quadratic in x, so its trace is a conic.' },
      { q: 'Which quantity does the SVM need from the lifted points?', options: ['their dot products', 'their coordinates one by one', 'their distances to the origin only', 'their class means'], answer: 0, why: 'Both the dual problem and the decision function use the data only through φ(xᵢ)·φ(xⱼ).' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // The polynomial kernel
  // ---------------------------------------------------------------------------------------------------------------
  const nMono = (deg) => ((deg + 1) * (deg + 2)) / 2;
  S.topic({
    id: 'svm-poly', family: 'p06-kernels', order: 20, title: 'The polynomial kernel',
    blurb: 'K(x, x′) = (γ·x·x′ + r)^d is the dot product of two lifted points whose features are every product of the inputs up to degree d. With two inputs that is 6 features at degree 2 and 21 at degree 5, but the kernel costs one dot product and a power. Raise d and the boundary can bend more.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'xor', options: P.setOptions(['xor', 'moons', 'circles']) },
      { id: 'C', kind: 'slider', label: 'C', def: 1, min: 0.1, max: 100, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'XOR corners', values: { set: 'xor', C: 1, seed: 1 } },
      { label: 'Two moons', values: { set: 'moons', C: 10, seed: 2 } },
      { label: 'A ring round a disc', values: { set: 'circles', C: 1, seed: 1 } },
    ],
    counters: [{ id: 'deg', label: 'degree' }, { id: 'feat', label: 'lifted features' }],
    run(inp) {
      const d = P.data(inp.set, { seed: inp.seed }), X = d.X, y = d.y, n = X.length, Cv = +H().sg(inp.C, 2);
      const F = H().frames(['deg', 'feat']), q = F.q, DEG = [1, 2, 3, 4, 5];
      const fits = DEG.map((k) => P.fit(X, y, { kernel: 'poly', degree: k, gamma: 1, coef0: 1, C: Cv }));
      const ax = P.axes('pk', X, { x: 40, y: 24, w: 400, h: 400 }, { pad: 0.25 });
      const ac = C().axes({ prefix: 'pa', x: 540, y: 50, w: 240, h: 150, xr: [0.6, 5.4], yr: [0, 1], xticks: DEG, yticks: [0, 0.5, 1], fmtY: (v) => Math.round(v * 100) + '%', xname: 'degree d', yname: 'training accuracy' });
      const side = (k) => {
        const items = [...ac.items, C().line(ac, 'pa:l', DEG.slice(0, k + 1).map((g, j) => [g, fits[j].acc]), 'act')];
        DEG.slice(0, k + 1).forEach((g, j) => items.push(C().dot(ac, 'pa:d' + j, g, fits[j].acc, '', j === k ? 6 : 3.5)));
        items.push(H().text('pk:feat', 500, 280, 'degree ' + DEG[k] + ': ' + nMono(DEG[k]) + ' lifted features', 'strong', 14));
        items.push(H().text('pk:svn', 500, 304, fits[k].sv.length + ' support vectors', 'legend', 13));
        return items;
      };
      F.push([...ax.items, ...S.Kit.scatter.points(ax, 'pk:p', X, y, { r: 4.6 }), ...ac.items], `${q.n(n)} points. We fit an SVM with the polynomial kernel K(x, x′) = (x·x′ + 1)^d, with C = ${q.sg(Cv, 2)}, for degrees d from ${q.n(DEG[0])} to ${q.n(DEG[DEG.length - 1])}.`, { phase: 'data', n, C: Cv, degs: DEG });
      DEG.forEach((g, k) => {
        const fit = fits[k];
        F.set('deg', g).set('feat', nMono(g));
        let say = `Degree ${q.n(g)}: the kernel stands for ${q.n(nMono(g))} features, every product of x₁ and x₂ up to degree ${q.n(g)}; ${q.n(fit.sv.length)} support vectors, training accuracy ${q.pct(fit.acc)}.`;
        if (g === 1) say += ' The first degree is a straight line, no better than a linear SVM.';
        else if (g === 2) say += ' The quadratic kernel adds x₁², x₂² and x₁x₂: the boundary can be an ellipse, a parabola or a pair of crossing lines.';
        else if (g === 5) say += ' Higher degrees bend more, and their values grow fast away from the data, so scaling the inputs matters.';
        F.push([...ax.items, ...P.boundary(ax, 'pk', fit, { n: 80 }), ...P.points(ax, 'pk', X, y, fit, { r: 4.6 }), ...side(k)], say, { phase: 'deg', deg: g, feat: nMono(g), nsv: fit.sv.length, acc: fit.acc, k });
      });
      // the kernel against the explicit features at degree 2, for one pair of points
      const a = X[0], c = X[1], kv = (a[0] * c[0] + a[1] * c[1] + 1) ** 2, s2 = Math.SQRT2;
      const phi = (x) => [1, s2 * x[0], s2 * x[1], x[0] * x[0], x[1] * x[1], s2 * x[0] * x[1]];
      const ex = phi(a).reduce((s, v, k) => s + v * phi(c)[k], 0);
      F.set('deg', 2).set('feat', 6);
      F.push([...ax.items, ...P.boundary(ax, 'pk', fits[1], { n: 80 }), ...P.points(ax, 'pk', X, y, fits[1], { r: 4.6 }), ...side(1)], `Check the shortcut for the quadratic kernel and the points (${q(a[0])}, ${q(a[1])}) and (${q(c[0])}, ${q(c[1])}): the kernel (x·x′ + 1)² gives ${q(kv, 4)}; the six features (1, √2x₁, √2x₂, x₁², x₂², √2x₁x₂) multiplied pairwise and added give ${q(ex, 4)}. The same number, without ever building the features.`, { phase: 'check', a, c, kv, ex });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The polynomial kernel', text: ['K(x, x′) = (γ·x·x′ + r)^d', 'here γ = 1 and r = 1', 'features: all products of the inputs up to degree d', 'p inputs, degree d: C(p + d, d) features', 'p = 2: 3, 6, 10, 15, 21 for d = 1 … 5'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr, inp) => f.meta.phase === 'deg' && f.meta.deg === 1, pick: 'option', ask: 'How many lifted features does the degree-2 kernel stand for, with two inputs?', options: ['6', '4', '3', '9'], answer: '6', explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => f.meta.phase === 'deg' && f.meta.deg === 2, pick: 'option', ask: 'At degree 3, will the training accuracy be at least as high as at degree 2?', options: ['yes', 'no'], answer: (fr, i) => (fr[i + 1].meta.acc >= fr[i].meta.acc ? 'yes' : 'no'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'With two inputs, how many features does a degree-3 polynomial kernel stand for (r > 0)?', options: ['10', '6', '9', '8'], answer: 0, why: 'All monomials of x₁, x₂ up to degree 3: 1 + 2 + 3 + 4 = 10.' },
      { q: 'What does the degree-1 polynomial kernel with r = 0 reduce to?', options: ['the linear kernel x·x′', 'the RBF kernel', 'a constant', 'nothing useful'], answer: 0, why: '(γ·x·x′)¹ is a scaled dot product.' },
      { q: 'Why scale the inputs before a high-degree polynomial kernel?', options: ['powers of large numbers explode and swamp the rest', 'the kernel needs integers', 'scaling raises the degree', 'it is not needed'], answer: 0, why: '(x·x′ + 1)^5 grows like ‖x‖¹⁰: unscaled features make the kernel matrix badly conditioned.' },
      { q: 'What does raising the degree do to the boundary?', options: ['lets it bend more, with more risk of overfitting', 'makes it straighter', 'removes the support vectors', 'widens the margin'], answer: 0, why: 'More features mean a more flexible boundary.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
