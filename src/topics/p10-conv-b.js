// Part 10, padding, stride and dilation: where the kernel's taps land. Padding adds a border of zeros, stride is how
// far the window jumps between places, dilation spreads the taps apart. The output size follows from the three, and
// the final frame counts how often each pixel is read (from the gradient of the summed output, all weights one).
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const TOP = 40;
  const num = (id, label, def, min, max) => ({ id, kind: 'number', label, def, min, max });

  S.topic({
    id: 'conv-geometry', family: 'p10-conv', order: 20, title: 'Padding, stride and dilation',
    blurb: 'Three settings decide where a kernel lands. Padding puts a border of zeros round the picture so the kernel can sit over the edges; stride makes the window jump several pixels at a time; dilation spreads the kernel’s taps apart so it covers more ground with the same weights. Step the window along and watch the output size come out of the formula.',
    inputs: [
      num('n', 'Picture size n', 6, 3, 8), num('k', 'Kernel size k', 3, 1, 5), num('p', 'Padding p', 1, 0, 2), num('s', 'Stride s', 1, 1, 3), num('d', 'Dilation d', 1, 1, 3),
    ],
    check: (inp) => ((inp.k - 1) * inp.d + 1 > inp.n + 2 * inp.p ? 'The kernel spans ' + ((inp.k - 1) * inp.d + 1) + ' pixels, more than the padded picture (' + (inp.n + 2 * inp.p) + '): make it smaller, or pad more' : null),
    presets: [
      { label: '“Same” padding: 3 × 3, p = 1', values: { n: 6, k: 3, p: 1, s: 1, d: 1 } },
      { label: 'No padding (“valid”)', values: { n: 6, k: 3, p: 0, s: 1, d: 1 } },
      { label: 'Stride 2 halves the size', values: { n: 8, k: 3, p: 1, s: 2, d: 1 } },
      { label: 'Stride bigger than the kernel skips pixels', values: { n: 7, k: 2, p: 0, s: 3, d: 1 } },
      { label: 'Dilation 2: a 3 × 3 kernel spanning 5', values: { n: 8, k: 3, p: 0, s: 1, d: 2 } },
      { label: 'A 1 × 1 kernel', values: { n: 5, k: 1, p: 0, s: 1, d: 1 } },
    ],
    counters: [{ id: 'places', label: 'places shown' }],
    run(inp) {
      const T = S.T, { n, k, p, s, d } = inp, opts = { stride: s, padding: p, dilation: d }, N = n + 2 * p, span = (k - 1) * d + 1;
      const [Ho, Wo] = T.convOut(n, n, k, k, opts);
      // how often each pixel is read: the gradient of the summed output with every weight 1
      const x = T.param(T.ones([1, 1, n, n])), y = T.conv2d(x, T.ones([1, 1, k, k]), null, opts);
      T.sum(y).backward();
      const uses = Array.from(x.grad.values());
      const c = Math.min(34, Math.floor(300 / N)), G = 2, step = c + G, OX = N * step + 70;
      const F = H().frames(['places']), q = F.q;
      const inGrid = (taps, showUses) => {
        const M = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { const r = i - p, cc = j - p; return r < 0 || cc < 0 || r >= n || cc >= n ? 0 : showUses ? uses[r * n + cc] : 0; }));
        const umax = Math.max(1, ...uses);
        return P.cells('in', 0, TOP, M, { cw: c, scheme: 'seq', max: umax * 1.25, size: c < 30 ? 11 : 12, fmt: (v) => '', cls: (i, j) => { const pad = i < p || j < p || i >= n + p || j >= n + p; return [pad ? 'pad' : '', taps && taps.has(i + ',' + j) ? 'tap' : ''].join(' '); } }).items.map((it) => {
          const [i, j] = it.key.split(':c')[1].split('_').map(Number), pad = i < p || j < p || i >= n + p || j >= n + p;
          return Object.assign(it, { text: pad ? '0' : showUses ? String(uses[(i - p) * n + (j - p)]) : '', cls: it.cls + (pad ? ' wait' : '') });
        });
      };
      // how many of each window's taps land on the picture itself (not on padding)
      const real = Array.from({ length: Ho }, (_, i) => Array.from({ length: Wo }, (_, j) => { let r = 0; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const y = i * s - p + a * d, x = j * s - p + b * d; if (y >= 0 && x >= 0 && y < n && x < n) r++; } return r; }));
      const outGrid = (done, cur) => P.cells('out', OX, TOP, real, { cw: c, scheme: 'seq', max: k * k * 1.25, d: 0, size: c < 30 ? 11 : 12, cls: (i, j) => (cur && cur[0] === i && cur[1] === j ? 'act' : '') }).items.map((it) => { const [i, j] = it.key.split(':c')[1].split('_').map(Number); return done(i, j) ? it : Object.assign(it, { text: '', cls: it.cls + ' wait', paint: null }); });
      const titles = [P.text('t:in', 0, TOP - 14, p ? 'picture ' + n + ' × ' + n + ', padded to ' + N + ' × ' + N : 'picture ' + n + ' × ' + n, 'axname', 13), P.text('t:out', OX, TOP - 14, 'output ' + Ho + ' × ' + Wo + ': real pixels read', 'axname', 13)];
      const legend = H().lines('lg', 0, TOP + N * step + 28, [['k = ' + k + ', padding ' + p + ', stride ' + s + ', dilation ' + d, 'math'], ['the kernel spans ' + span + ' pixels: (k − 1) · d + 1', 'legend']], 13, 22);
      const intro = `${n === 8 ? 'An' : 'A'} ${q.n(n)} × ${q.n(n)} picture` + (p ? ` with ${q.n(p)} ring${p > 1 ? 's' : ''} of zeros round it (dashed)` : ' with no padding') + `, a ${q.n(k)} × ${q.n(k)} kernel, stride ${q.n(s)} and dilation ${q.n(d)}. `;
      const formula = `The kernel spans ${q.n(span)} pixel${span === 1 ? '' : 's'} and the padded picture ${q.n(N)}, so it fits (${q.n(N)} − ${q.n(span)}) / ${q.n(s)}, rounded down, plus one: ${q.n(Ho)} time${Ho === 1 ? '' : 's'} each way: the output is ${q.n(Ho)} × ${q.n(Wo)}`;
      F.push([...titles, ...inGrid(null, false), ...outGrid(() => false), ...legend],
        intro + formula + '.', { phase: 'start', n, k, p, s, d, Ho, Wo, span, N });
      const places = [];
      for (let j = 0; j < Wo; j++) places.push([0, j]);
      if (Ho > 1) places.push([1, 0]);
      const done = new Set();
      for (const [oi, oj] of places) {
        F.count('places');
        done.add(oi + ',' + oj);
        const taps = new Set(), r0 = oi * s, c0 = oj * s;
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) taps.add(r0 + a * d + ',' + (c0 + b * d));
        let padTaps = 0;
        for (const t of taps) { const [i, j] = t.split(',').map(Number); if (i < p || j < p || i >= n + p || j >= n + p) padTaps++; }
        const box = P.box('win', c0 * step - 3, TOP + r0 * step - 3, span * step - G + 6, span * step - G + 6, 'act');
        const moved = oj > 0 ? `The window moved ${q.n(s)} pixel${s > 1 ? 's' : ''} right from the last place. ` : oi > 0 ? `Next row: the window drops ${q.n(s)} pixel${s > 1 ? 's' : ''} and starts again at the left. ` : '';
        F.push([...titles, ...inGrid(taps, false), box, ...outGrid((i, j) => done.has(i + ',' + j), [oi, oj]), ...legend],
          moved + `Output row ${q.n(oi + 1)}, column ${q.n(oj + 1)}: its ${q.n(k * k)} taps (outlined) start at padded row ${q.n(r0 + 1)}, column ${q.n(c0 + 1)}` + (d > 1 ? `, ${q.n(d)} apart` : '') + (padTaps ? `; ${q.n(padTaps)} of them fall on padding zeros.` : '; none falls on padding.'), { phase: 'place', oi, oj, r0, c0, padTaps, kk: k * k, s, d, row: oi + 1, col: oj + 1, pr: r0 + 1, pc: c0 + 1 });
      }
      const zero = uses.filter((u) => u === 0).length, most = Math.max(...uses), corner = uses[0];
      F.push([...titles, ...inGrid(null, true), ...outGrid(() => true), ...legend],
        `Every place visited: ${q.n(Ho * Wo)} output${Ho * Wo === 1 ? '' : 's'}. The numbers count how often each pixel was read, the gradient of the summed output when every weight is one: the busiest pixel is read ${q.n(most)} time${most === 1 ? '' : 's'}, the top-left corner ${q.n(corner)}.` + (zero ? ` ${q.n(zero)} pixel${zero === 1 ? ' is' : 's are'} never read at all: the stride jumps over ${zero === 1 ? 'it' : 'them'}.` : ' Every pixel is read at least once.'), { phase: 'end', count: Ho * Wo, most, corner, zero, uses });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Output size', text: ['out = ⌊(n + 2p − d(k − 1) − 1) / s⌋ + 1', 'span of the kernel = d(k − 1) + 1', '“same”: p = (k − 1) / 2 with s = 1 keeps n', 'stride s divides the size by about s'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start', once: true, pick: 'option',
      ask: 'How large will the output map be?',
      options: (fr) => { const m = fr[0].meta, c = new Set([m.Ho, m.n - m.k + 1, m.n, m.n + 2 * m.p - m.k + 1, Math.floor(m.n / m.s)].filter((v) => v > 0)); let extra = m.Ho + 1; while (c.size < 4) c.add(extra++); return [...c].sort((a, b) => a - b).map((v) => v + ' × ' + v); },
      answer: (fr) => fr[0].meta.Ho + ' × ' + fr[0].meta.Wo,
      explain: (fr) => fr[0].say,
    }],
    quiz: [
      { q: 'A 3 × 3 kernel with padding 1 and stride 1 on a 10 × 10 picture gives…', options: ['10 × 10', '8 × 8', '12 × 12', '5 × 5'], answer: 0, why: '(10 + 2 − 3) / 1 + 1 = 10: “same” padding.' },
      { q: 'Stride 2 with a 3 × 3 kernel and padding 1 on a 16 × 16 picture gives…', options: ['8 × 8', '16 × 16', '7 × 7', '14 × 14'], answer: 0, why: '⌊(16 + 2 − 3) / 2⌋ + 1 = 8.' },
      { q: 'A 3 × 3 kernel with dilation 2 covers…', options: ['a 5 × 5 area with 9 weights', 'a 6 × 6 area with 36 weights', 'a 3 × 3 area', 'a 9 × 9 area with 81 weights'], answer: 0, why: 'Span (3 − 1) · 2 + 1 = 5, still nine taps.' },
      { q: 'When can a convolution skip pixels completely?', options: ['when the stride is larger than the kernel’s span', 'whenever padding is used', 'when dilation is 1', 'never'], answer: 0, why: 'The window jumps past the pixels between its places.' },
      { q: 'What value do the padding pixels hold?', options: ['zero', 'the nearest edge pixel always', 'one', 'random noise'], answer: 0, why: 'Zero padding is the usual choice; they add nothing to the sums.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
