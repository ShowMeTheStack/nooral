// Part 10, pooling: a window slides over a feature map and keeps one number of each patch, the largest (max) or the
// mean (average); the map shrinks, and a small shift of the input changes little of the result.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const TOP = 40;
  const TOY = [[0, 1, 0, 0, 2, 0], [3, 0, 0, 0, 0, 1], [0, 0, 4, 1, 0, 0], [0, 2, 0, 0, 0, 5], [1, 0, 0, 3, 0, 0], [0, 0, 2, 0, 1, 0]];
  const MAPS = {
    toy: { label: 'a small map of whole numbers (6 × 6)', make: () => ({ M: TOY }), say: () => 'a made-up 6 × 6 map after ReLU' },
    digit: {
      label: 'a real map: the trained network’s first layer on a 7 (8 × 8 crop)',
      make: () => P.once('poolDigit', () => {
        const im = P.pick('digits', 7, 0), o = P.forward('digits', im.px), v = o.relu1.v;
        const tot = [0, 1, 2, 3, 4, 5].map((c) => P.map(v, c, 16).reduce((a, b) => a + b, 0)), c = tot.indexOf(Math.max(...tot));
        const M = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => v[c * 256 + (i + 4) * 16 + j + 4]));
        return { M, channel: c, seven: 7 };
      }),
      say: (q, src) => 'channel ' + q.n(src.channel + 1) + ' of the trained network’s first layer (after ReLU) for a ' + q.n(src.seven) + ', its middle 8 × 8',
    },
  };

  P.POOLMAPS = MAPS;
  S.topic({
    id: 'conv-pool', family: 'p10-conv', order: 30, title: 'Pooling: shrinking a map',
    blurb: 'After a convolution and ReLU, a pooling layer slides a small window over each map and keeps one number per patch: the largest (max pooling) or the mean (average pooling). The map gets smaller, the next layer sees a wider area for the same cost, and a pattern that moves by a pixel often gives the same pooled value.',
    inputs: [
      { id: 'map', kind: 'choice', label: 'Feature map', def: 'toy', options: Object.keys(MAPS).map((k) => ({ value: k, label: MAPS[k].label })) },
      { id: 'kind', kind: 'choice', label: 'Pooling', def: 'max', options: [{ value: 'max', label: 'max' }, { value: 'avg', label: 'average' }] },
      { id: 'size', kind: 'number', label: 'Window', def: 2, min: 2, max: 3 },
      { id: 'stride', kind: 'number', label: 'Stride', def: 2, min: 1, max: 3 },
    ],
    presets: [
      { label: 'Max pooling, 2 × 2, stride 2', values: { map: 'toy', kind: 'max', size: 2, stride: 2 } },
      { label: 'Average pooling, 2 × 2', values: { map: 'toy', kind: 'avg', size: 2, stride: 2 } },
      { label: 'Overlapping: 3 × 3, stride 1', values: { map: 'toy', kind: 'max', size: 3, stride: 1 } },
      { label: 'A real map from a trained network', values: { map: 'digit', kind: 'max', size: 2, stride: 2 } },
      { label: 'The real map, averaged', values: { map: 'digit', kind: 'avg', size: 2, stride: 2 } },
    ],
    counters: [{ id: 'windows', label: 'windows pooled' }],
    run(inp) {
      const T = S.T, src = MAPS[inp.map].make(), M = src.M, n = M.length, k = inp.size, s = inp.stride;
      const x = T.tensor(M.flat(), [1, 1, n, n]), pool = (t) => Array.from((inp.kind === 'max' ? T.maxPool2d(t, k, s) : T.avgPool2d(t, k, s)).values());
      const m = Math.floor((n - k) / s) + 1, out = pool(x), vmax = Math.max(...M.flat()) || 1;
      const c = n > 6 ? 36 : 42, step = c + 2, OX = n * step + 64, d = inp.map === 'toy' ? 0 : 2;
      const F = H().frames(['windows']), q = F.q, word = inp.kind === 'max' ? 'largest' : 'mean';
      const grid = (prefix, x0, y0, lit, best, extra) => P.cells(prefix, x0, y0, M, { cw: c, scheme: 'seq', max: vmax, d, size: 12, cls: (i, j) => [lit && i >= lit[0] && i < lit[0] + k && j >= lit[1] && j < lit[1] + k ? 'lit' : '', best && best[0] === i && best[1] === j ? 'tap' : '', extra ? extra(i, j) || '' : ''].join(' ') }).items;
      const outGrid = (prefix, y0, vals, done, cur, mark) => P.cells(prefix, OX, y0, P.rows(vals, m), { cw: c, scheme: 'seq', max: vmax, d: inp.kind === 'avg' ? 2 : d, size: 12, cls: (i, j) => [cur && cur[0] === i && cur[1] === j ? 'act' : '', mark ? mark(i, j) || '' : ''].join(' ') }).items.map((it, idx) => (done(Math.floor(idx / m), idx % m) ? it : Object.assign({}, it, { text: '', cls: it.cls + ' wait', paint: null })));
      const titles = [P.text('t:in', 0, TOP - 14, 'feature map, ' + n + ' × ' + n, 'axname', 13), P.text('t:out', OX, TOP - 14, (inp.kind === 'max' ? 'max' : 'average') + '-pooled, ' + m + ' × ' + m, 'axname', 13)];
      F.push([...titles, ...grid('in', 0, TOP), ...outGrid('out', TOP, out, () => false)],
        `The input is ${MAPS[inp.map].say(q, src)}. A ${q.n(k)} × ${q.n(k)} window moving ${q.n(s)} at a time fits ${q.n(m)} times each way, so the pooled map is ${q.n(m)} × ${q.n(m)}` + (s < k ? '; the windows overlap.' : s > k ? '; some values are never looked at.' : `: ${q.n(n * n)} numbers become ${q.n(m * m)}.`), { phase: 'start', n, k, s, m, nn: n * n, mm: m * m, channel: src.channel === undefined ? null : src.channel + 1, seven: src.seven });
      const places = [];
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) places.push([i, j]);
      const shown = places.length <= 16 ? places : places.slice(0, m + 1);
      shown.forEach(([i, j], idx) => {
        F.count('windows');
        const r0 = i * s, c0 = j * s, vals = [];
        let best = null;
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const v = M[r0 + a][c0 + b]; vals.push(v); if (!best || v > best.v) best = { v, r: r0 + a, c: c0 + b }; }
        const val = out[i * m + j];
        const say = inp.kind === 'max'
          ? `Window at row ${q.n(r0 + 1)}, column ${q.n(c0 + 1)}: the largest of ${H().list(vals.map((v) => q.sg(v, 3)))} is ${q.sg(val, 3)} (ringed), and only it goes on.`
          : `Window at row ${q.n(r0 + 1)}, column ${q.n(c0 + 1)}: the mean of ${H().list(vals.map((v) => q.sg(v, 3)))} is ${q.sg(val, 3)}.`;
        F.push([...titles, ...grid('in', 0, TOP, [r0, c0], inp.kind === 'max' ? [best.r, best.c] : null), P.box('win', c0 * step - 3, TOP + r0 * step - 3, k * step + 4, k * step + 4, 'act'), ...outGrid('out', TOP, out, (a, b) => a * m + b <= i * m + j, [i, j])],
          say, { phase: 'window', i, j, r0, c0, row: r0 + 1, col: c0 + 1, vals, val });
      });
      // a shift by one pixel to the right: how many pooled values stay the same?
      const sh = M.map((r) => [0].concat(r.slice(0, n - 1))), out2 = pool(T.tensor(sh.flat(), [1, 1, n, n]));
      const same = out.filter((v, i) => Math.abs(v - out2[i]) < 1e-6).length, y2 = TOP + m * step + 58;
      F.push([...titles, ...grid('in', 0, TOP, null, null, (i, j) => (j === 0 ? 'pad' : '')).map((it) => { const [a, b] = it.key.split(':c')[1].split('_').map(Number); const v = sh[a][b]; return Object.assign(it, { text: P.num(v, d), paint: S.Kit.heat.colour(v, 'seq', vmax) }); }),
        ...outGrid('out', TOP, out, () => true), P.text('t:out2', OX, y2 - 14, 'after shifting the input', 'axname', 13), ...outGrid('out2', y2, out2, () => true, null, (i, j) => (Math.abs(out[i * m + j] - out2[i * m + j]) < 1e-6 ? 'used' : ''))],
        `Now the whole map moves one pixel to the right (a column of zeros comes in on the left). Pooling again, ${q.n(same)} of the ${q.n(m * m)} pooled values are exactly as before (outlined below).` + (inp.kind === 'max' ? ' A largest value that stays inside its window is not disturbed by the shift: this is the small tolerance to position that pooling buys.' : ' Averages shift a little with every move, so fewer of them stay the same than with max pooling.'), { phase: 'shift', same, mm: m * m, out2 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Pooling', text: ['max:     y[i, j] = max over the window of x', 'average: y[i, j] = mean over the window of x', 'size: ⌊(n − k) / s⌋ + 1 each way', 'no weights to learn; applied to each map on its own'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start', once: true, pick: 'item',
      ask: 'Click the cell of the input map that the first pooled value will come from (for average pooling: any cell of its window).',
      answer: (fr) => { const w = fr.find((f) => f.meta.phase === 'window').meta, k = fr[0].meta.k, out = []; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const r = w.r0 + a, c = w.c0 + b; if (w.vals[a * k + b] === w.val || fr.find((f) => f.meta.phase === 'window').say.includes('mean')) out.push('in:c' + r + '_' + c); } return out; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'window').say,
    }],
    quiz: [
      { q: '2 × 2 max pooling with stride 2 turns a 16 × 16 map into…', options: ['8 × 8', '15 × 15', '4 × 4', '14 × 14'], answer: 0, why: 'Each 2 × 2 block becomes one value.' },
      { q: 'How many weights does a max-pooling layer learn?', options: ['none', 'one per window', 'four', 'one per map'], answer: 0, why: 'It only compares numbers.' },
      { q: 'Max pooling of the window 0, 3, 1, 2 gives…', options: ['3', '1.5', '6', '0'], answer: 0, why: 'The largest value.' },
      { q: 'Average pooling of the window 0, 3, 1, 2 gives…', options: ['1.5', '3', '6', '2'], answer: 0, why: '(0 + 3 + 1 + 2) / 4.' },
      { q: 'Why does pooling help a network cope with a digit drawn one pixel to the left?', options: ['the largest value in a window often stays in the same window', 'it rotates the picture back', 'it adds padding', 'it learns where the digit is'], answer: 0, why: 'A small shift often keeps the peak inside the same window.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
