// Part 10, receptive fields: how big a patch of the input one unit of a layer depends on. Computed two ways: by the
// rule r ← r + (k − 1) · d · jump, jump ← jump · s, layer by layer; and by the engine, as the pixels where the gradient
// of that one unit with respect to the input is not zero (every weight set to one, average pooling, so no path is
// cut). The two must agree; the gradient also shows that the middle of the field counts most.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const cv = (k, s, d, p) => ({ t: 'conv', k, s: s || 1, d: d || 1, p: p || 0 });
  const pool = () => ({ t: 'pool', k: 2, s: 2, d: 1, p: 0 });
  const ARCH = {
    three: { label: 'three 3 × 3 convolutions', N: 20, L: [cv(3), cv(3), cv(3)] },
    pooled: { label: '3 × 3 convolutions with pooling between', N: 24, L: [cv(3), pool(), cv(3), pool(), cv(3)] },
    dilated: { label: 'dilated 3 × 3: dilation 1, 2, 4', N: 20, L: [cv(3, 1, 1), cv(3, 1, 2), cv(3, 1, 4)] },
    strided: { label: '3 × 3 convolutions with stride 2', N: 24, L: [cv(3, 2), cv(3, 2), cv(3, 2)] },
    ours: { label: 'the network trained here (5 × 5, pool, 5 × 5, pool)', N: 16, L: [cv(5, 1, 1, 2), pool(), cv(5), pool()] },
    big: { label: 'one 7 × 7 convolution', N: 20, L: [cv(7)] },
  };
  const name = (l) => (l.t === 'pool' ? '2 × 2 pool' : l.k + ' × ' + l.k + ' conv' + (l.s > 1 ? ', stride ' + l.s : '') + (l.d > 1 ? ', dilation ' + l.d : ''));

  S.topic({
    id: 'cnn-field', family: 'p10-inside', order: 30, title: 'Receptive fields: how far a unit can see',
    blurb: 'A unit in the first layer sees only its kernel’s patch of the picture. A unit in the next layer sees a patch of first-layer units, each with its own patch, so it sees more. Pooling, stride and dilation make the field grow faster. Here the field is worked out layer by layer by a rule, and checked by the engine: the input pixels with a non-zero gradient for that one unit are exactly its field.',
    inputs: [{ id: 'arch', kind: 'choice', label: 'Layers', def: 'pooled', options: P.options(ARCH) }],
    presets: Object.keys(ARCH).map((k) => ({ label: ARCH[k].label.replace(/^./, (c) => c.toUpperCase()), values: { arch: k } })),
    counters: [{ id: 'layers', label: 'layers' }],
    run(inp) {
      const T = S.T, A = ARCH[inp.arch], N = A.N, F = H().frames(['layers']), q = F.q;
      const qname = (l) => (l.t === 'pool' ? 'two-by-two pool' : `${q.n(l.k)} × ${q.n(l.k)} conv` + (l.s > 1 ? `, stride ${q.n(l.s)}` : '') + (l.d > 1 ? `, dilation ${q.n(l.d)}` : ''));
      const PX = Math.min(18, Math.floor(380 / N)), SZ = PX * N, TOP = 40, TX = SZ + 50;
      // the rule
      let r = 1, j = 1;
      const rows = A.L.map((l) => { const r0 = r, j0 = j; r = r + (l.k - 1) * l.d * j; j = j * l.s; return { r0, j0, r, j }; });
      // the engine: the gradient of one unit of layer L with respect to the input
      const probe = (L) => {
        const x = T.param(T.ones([1, 1, N, N]));
        let h = x;
        for (const l of A.L.slice(0, L)) h = l.t === 'pool' ? T.avgPool2d(h, 2, 2) : T.conv2d(h, T.full([1, 1, l.k, l.k], 1 / (l.k * l.k)), null, { stride: l.s, padding: l.p, dilation: l.d });
        const M = h.shape[2], u = Math.floor((M - 1) / 2), mask = new Array(M * M).fill(0);
        mask[u * M + u] = 1;
        T.sum(T.mul(h, T.tensor(mask, h.shape))).backward();
        const g = Array.from(x.grad.values());
        let lo = [N, N], hi = [-1, -1];
        g.forEach((v, i) => { if (v !== 0) { const a = Math.floor(i / N), b = i % N; lo = [Math.min(lo[0], a), Math.min(lo[1], b)]; hi = [Math.max(hi[0], a), Math.max(hi[1], b)]; } });
        // the same interval by the rule, mapped back layer by layer and clipped to the picture
        let a = u, b = u;
        for (const l of A.L.slice(0, L).reverse()) { a = a * l.s - l.p; b = b * l.s - l.p + (l.k - 1) * l.d; }
        return { g, M, u, lo, hi, a, b, ca: Math.max(0, a), cb: Math.min(N - 1, b), nz: g.filter((v) => v !== 0).length };
      };
      const table = (lit) => S.Kit.dt.table({ prefix: 'tb', x: TX, y: TOP, cols: [{ name: 'layer', w: 190 }, { name: 'field', w: 64 }, { name: 'jump', w: 60 }], rows: [['input pixel', '1', '1']].concat(A.L.map((l, i) => [name(l), i < lit ? String(rows[i].r) : '', i < lit ? String(rows[i].j) : ''])), rowH: 28, headH: 28, size: 13, cls: (i) => (i === lit ? 'act' : '') });
      const pic = (pr, L) => {
        const items = [P.text('h:in', 0, TOP - 14, 'input, ' + N + ' × ' + N + ': gradient of one unit', 'axname', 13)];
        const gm = pr ? Math.max(...pr.g) : 1;
        items.push({ key: 'in', kind: 'raster', x: 0, y: TOP, w: SZ, h: SZ, nx: N, ny: N, vals: pr ? pr.g : new Array(N * N).fill(0), scheme: 'seq', max: gm || 1, smooth: false, cls: 'p10img', layer: 'items' });
        let gl = '';
        for (let i = 1; i < N; i++) gl += `M${i * PX},${TOP} v${SZ} M0,${TOP + i * PX} h${SZ} `;
        items.push({ key: 'gl', kind: 'path', d: gl, cls: 'grid', audit: 'ignore', layer: 'items', z: 3 });
        if (pr) items.push(P.box('field', pr.ca * PX, TOP + pr.ca * PX, (pr.cb - pr.ca + 1) * PX, (pr.cb - pr.ca + 1) * PX, 'field'));
        const tb = table(L);
        items.push(...tb.items);
        items.push(...H().lines('rule', TX, TOP + tb.h + 30, [['field ← field + (k − 1) · d · jump', 'math'], ['jump ← jump · s', 'math'], ['shaded: ∂ unit / ∂ pixel (not zero = in the field)', 'legend']], 13, 22));
        return items;
      };
      F.push(pic(null, 0), `A pixel sees only itself: field ${q.n(1)}, and one step in the input is one pixel (jump ${q.n(1)}). The layers are: ${A.L.map(qname).join(', ')}.`, { phase: 'start', one: 1, layers: A.L });
      A.L.forEach((l, i) => {
        F.count('layers');
        const R = rows[i], pr = probe(i + 1), w = pr.hi[1] - pr.lo[1] + 1, clipped = pr.a < 0 || pr.b > N - 1;
        const head = `Layer ${q.n(i + 1)}, ${qname(l)}. `;
        const how = l.t === 'pool'
          ? `Pooling, moving two at a time, adds one step of the current jump, ${q.n(R.j0)}, so the field is ${q.n(R.r)}, and it doubles the jump to ${q.n(R.j)}`
          : `A ${q.n(l.k)} × ${q.n(l.k)} kernel` + (l.d > 1 ? ` with dilation ${q.n(l.d)}` : '') + ` adds (k − 1) · d · jump = ${q.n(l.k - 1)} × ${q.n(l.d)} × ${q.n(R.j0)} = ${q.n(R.r - R.r0)}: field ${q.n(R.r)}` + (l.s > 1 ? `, and its stride ${q.n(l.s)} makes the jump ${q.n(R.j)}` : '');
        const check = clipped ? ` The engine’s gradient is non-zero on ${q.n(w)} × ${q.n(w)} pixels: the rest of the ${q.n(R.r)} × ${q.n(R.r)} field lies on the padding.` : ` The engine agrees: the gradient of the middle unit is non-zero across exactly ${q.n(w)} × ${q.n(w)} pixels` + (pr.nz < w * w ? `, though only ${q.n(pr.nz)} of them are touched: dilation leaves gaps.` : '.');
        F.push(pic(pr, i + 1), head + how + '.' + check, { phase: 'layer', l: i + 1, k: l.k, d: l.d, s: l.s, r: R.r, r0: R.r0, j: R.j, j0: R.j0, add: R.r - R.r0, w, km1: l.k - 1, nz: pr.nz, ww: w * w, lo: pr.lo, hi: pr.hi, a: pr.a, b: pr.b, N, clipped });
      });
      const last = probe(A.L.length), mid = Math.floor((last.ca + last.cb) / 2), centre = last.g[mid * N + mid], corner = last.g[last.lo[0] * N + last.lo[1]];
      F.push(pic(last, A.L.length + 1), `The last layer’s units each see ${q.n(rows[rows.length - 1].r)} × ${q.n(rows[rows.length - 1].r)} pixels. ` + (centre / corner > 1.5 ? `Not all equally: the middle pixel’s gradient is ${q.sg(centre / corner, 3)} times the corner’s, because it reaches the unit along many more paths. The field that matters in practice is smaller than its edges suggest.` : `Here the middle pixel counts no more than the corner (its gradient is ${q.sg(centre / corner, 3)} times the corner’s): with these layers the paths to the unit are spread evenly.`), { phase: 'end', r: rows[rows.length - 1].r, ratio: centre / corner, centre, corner });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The receptive field', text: ['r₀ = 1, j₀ = 1 (one input pixel)', 'rₗ = rₗ₋₁ + (kₗ − 1) · dₗ · jₗ₋₁', 'jₗ = jₗ₋₁ · sₗ', 'pooling counts as a kernel with its own k and s'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start', once: true, pick: 'option',
      ask: 'How wide will the last layer’s field be?',
      options: (fr) => { const r = fr.filter((f) => f.meta.phase === 'layer').slice(-1)[0].meta.r, c = new Set([r, r - 2, r + 4, Math.max(3, Math.round(r / 2))]); let e = r + 1; while (c.size < 4) c.add(e++); return [...c].filter((v) => v > 0).sort((a, b) => a - b).map((v) => v + ' pixels'); },
      answer: (fr) => fr.filter((f) => f.meta.phase === 'layer').slice(-1)[0].meta.r + ' pixels',
      explain: (fr) => fr.filter((f) => f.meta.phase === 'layer').slice(-1)[0].say,
    }],
    quiz: [
      { q: 'Three 3 × 3 convolutions with stride 1 give a field of…', options: ['7 × 7', '9 × 9', '3 × 3', '27 × 27'], answer: 0, why: '1 + 2 + 2 + 2.' },
      { q: 'After a 2 × 2 pool with stride 2, a following 3 × 3 convolution adds…', options: ['4 to the field', '2 to the field', 'nothing', '9 to the field'], answer: 0, why: '(3 − 1) times the jump of 2.' },
      { q: 'Dilation lets a 3 × 3 kernel…', options: ['cover a wider field with the same nine weights', 'use more weights', 'shrink the map', 'skip the bias'], answer: 0, why: 'Its taps are spread d apart.' },
      { q: 'Within a unit’s field, which pixels usually matter most?', options: ['those near the middle', 'the corners', 'all equally', 'only the edges'], answer: 0, why: 'More paths lead from the middle pixels to the unit.' },
      { q: 'Two stacked 3 × 3 convolutions see the same field as one 5 × 5. What do they save?', options: ['weights: 18 against 25 per channel', 'nothing', 'the bias', 'pooling'], answer: 0, why: '2 · 9 = 18 < 25, and they add a second ReLU.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
