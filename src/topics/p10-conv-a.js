// Part 10, convolution as a sliding kernel: a 3 × 3 kernel visits every place of a small picture; at each place the
// nine pixels under it are multiplied by the nine weights and added, and that sum is one number of the output map.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const C = 38, G = 2, K0 = 320, OUTX = 524, TOP = 40;

  S.topic({
    id: 'conv-slide', family: 'p10-conv', order: 10, title: 'Convolution: a kernel sliding over a picture',
    blurb: 'A convolution layer is one small grid of weights, the kernel, slid over the picture. At each place the pixels under it are multiplied by the weights and added up: a dot product. That number goes into the output map at the same place. Watch every product, every sum, and the map that comes out: it is bright wherever the picture looks like the kernel.',
    inputs: [
      { id: 'picture', kind: 'choice', label: 'Picture', def: 'corner', options: P.options(P.PICTURES) },
      { id: 'kernel', kind: 'choice', label: 'Kernel', def: 'vedge', options: P.options(P.KERNELS) },
      { id: 'bias', kind: 'choice', label: 'Bias', def: '0', options: [{ value: '0', label: '0' }, { value: '-1', label: '−1' }, { value: '1', label: '+1' }] },
    ],
    presets: [
      { label: 'Vertical edges of a corner', values: { picture: 'corner', kernel: 'vedge', bias: '0' } },
      { label: 'Horizontal edges of a corner', values: { picture: 'corner', kernel: 'hedge', bias: '0' } },
      { label: 'A bar, edge kernel', values: { picture: 'bar', kernel: 'vedge', bias: '0' } },
      { label: 'Diagonal detector on a diagonal', values: { picture: 'diagonal', kernel: 'diag', bias: '-1' } },
      { label: 'Blurring a plus', values: { picture: 'plus', kernel: 'blur', bias: '0' } },
      { label: 'Sharpening a 7', values: { picture: 'seven', kernel: 'sharpen', bias: '0' } },
    ],
    counters: [{ id: 'places', label: 'places visited' }, { id: 'mults', label: 'multiplications' }],
    run(inp) {
      const T = S.T, pic = P.PICTURES[inp.picture].px, ker = P.KERNELS[inp.kernel].w, bias = +inp.bias, n = pic.length, k = 3, m = n - k + 1;
      const x = T.tensor(pic.flat(), [1, 1, n, n]), w = T.tensor(ker.flat(), [1, 1, k, k]), b = T.tensor([bias]);
      const ex = T.explain.conv2d(x, w, b), out = Array.from(ex.out.values()), wins = ex.trace.filter((s) => s.kind === 'window');
      const F = H().frames(['places', 'mults']), q = F.q, wmax = P.maxAbs(ker.flat()), omax = P.maxAbs(out);
      const outY = TOP, prodY = TOP + 3 * (C + G) + 58;
      const kerItems = P.cells('ks', K0, TOP, ker, { cw: C, max: wmax, d: 2, size: 12 }).items;
      const titles = [P.text('t:in', 0, TOP - 14, 'picture, ' + n + ' × ' + n, 'axname', 13), P.text('t:k', K0, TOP - 14, 'kernel, 3 × 3', 'axname', 13), P.text('t:out', OUTX, TOP - 14, 'output map, ' + m + ' × ' + m, 'axname', 13)];
      const input = (lit) => P.cells('in', 0, TOP, pic, { cw: C, scheme: 'seq', max: 1, d: 1, size: 12, cls: (i, j) => (lit && i >= lit[0] && i < lit[0] + k && j >= lit[1] && j < lit[1] + k ? 'lit' : '') }).items;
      const output = (done, cur) => {
        const g = P.cells('out', OUTX, outY, P.rows(out, m), { cw: C, max: omax, d: 2, size: 12, cls: (i, j) => (cur && cur[0] === i && cur[1] === j ? 'act' : '') });
        return g.items.map((it, idx) => (done(Math.floor(idx / m), idx % m) ? it : Object.assign({}, it, { text: '', cls: it.cls + ' wait', paint: null })));
      };
      const prods = (win) => {
        if (!win) return [];
        const M = [0, 1, 2].map((i) => [0, 1, 2].map((j) => win.window[i * 3 + j].value * win.window[i * 3 + j].weight));
        return [P.text('t:p', K0, prodY - 14, 'pixel × weight', 'axname', 13), ...P.cells('pr', K0, prodY, M, { cw: 52, ch: C, max: wmax, d: 2, size: 12 }).items];
      };
      const winBox = (oy, ox) => P.box('win', ox * (C + G) - 3, oy * (C + G) - 3 + TOP, k * (C + G) - G + 6, k * (C + G) - G + 6, 'act');
      F.push([...titles, ...input(null), ...kerItems, ...output(() => false)],
        `The picture is a ${q.n(n)} × ${q.n(n)} grid of pixel values (1 is ink, 0 is paper). The kernel is a 3 × 3 grid of weights (the ${P.KERNELS[inp.kernel].label} kernel). It fits in ${q.n(m)} places across and ${q.n(m)} down, so the output map is ${q.n(m)} × ${q.n(m)}.`, { phase: 'start', n, m });
      wins.forEach((s, idx) => {
        F.count('places').count('mults', 9);
        const oy = s.oy, ox = s.ox, terms = s.window.filter((t) => t.value !== 0 && t.weight !== 0);
        const where = `Row ${q.n(oy + 1)}, column ${q.n(ox + 1)}: `;
        const sum = terms.length ? terms.map((t, i) => q.sg(t.value, 3) + ' × ' + q.sg(t.weight, 3)).join(' + ') : '';
        const say = where + (terms.length === 1 ? `the only product that is not zero is ${sum}` : terms.length ? `the products that are not zero add up as ${sum}` : 'every product is zero (no ink meets a weight)') + (bias ? `; with the bias ${q.sg(bias, 3)}` : '') + `, so the output there is ${q.sg(s.value, 3)}.`;
        F.push([...titles, ...input([oy, ox]), winBox(oy, ox), ...kerItems, ...prods(s), ...output((i, j) => i * m + j <= idx, [oy, ox]), P.text('sum', K0, prodY + 3 * (C + G) + 20, 'sum' + (bias ? ' + bias' : '') + ' = ' + P.num(s.value, 2), 'strong', 14)],
          say, { phase: 'window', row: oy + 1, col: ox + 1, oy, ox, terms: terms.map((t) => [t.value, t.weight]), value: s.value, bias });
      });
      const hi = out.indexOf(Math.max(...out)), lo = out.indexOf(Math.min(...out)), neg = out.filter((v) => v < 0).length;
      const best = [Math.floor(hi / m), hi % m];
      F.push([...titles, ...input(best), winBox(best[0], best[1]), ...kerItems, ...output(() => true, best)],
        `The whole map. Its largest value, ${q.sg(out[hi], 3)}, is at row ${q.n(best[0] + 1)}, column ${q.n(best[1] + 1)} (boxed): there the picture looks most like the kernel. The smallest is ${q.sg(out[lo], 3)}, ${neg === m * m ? `and all ${q.n(m * m)} values are negative` : neg ? `and ${q.n(neg)} of the ${q.n(m * m)} values are negative` : `and none of the ${q.n(m * m)} values is negative`}. The same ${q.n(9)} weights were used at every place: that sharing is what makes it a convolution.`, { phase: 'end', max: out[hi], min: out[lo], neg, count: m * m, best, row: best[0] + 1, col: best[1] + 1, nine: 9 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'One output value', text: ['y[i, j] = b + Σᵤ Σᵥ x[i + u, j + v] · w[u, v]', 'u, v run over the kernel: 0, 1, 2', 'output size: n − k + 1 each way', 'the same w at every place (weight sharing)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start', once: true, pick: 'item',
      ask: 'Click the output cell you expect to be the largest.',
      answer: (fr) => { const e = fr[fr.length - 1].meta, m = fr[0].meta.m, all = []; fr.filter((f) => f.meta.phase === 'window' && Math.abs(f.meta.value - e.max) < 1e-6).forEach((f) => all.push('out:c' + f.meta.oy + '_' + f.meta.ox)); return all.length ? all : ['out:c' + Math.floor(0 / m) + '_0']; },
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'A 3 × 3 kernel slides over an 8 × 8 picture with no padding and stride 1. The output is…', options: ['6 × 6', '8 × 8', '5 × 5', '3 × 3'], answer: 0, why: '8 − 3 + 1 = 6 places each way.' },
      { q: 'How many weights does one 3 × 3 kernel on a one-channel picture have, bias aside?', options: ['9, whatever the picture’s size', 'one per pixel', '3', '81'], answer: 0, why: 'The same nine weights are used at every place.' },
      { q: 'The vertical-edge kernel [−1 0 1] in every row gives a large positive value where…', options: ['ink lies to the right of paper', 'ink lies to the left of paper', 'the window is all ink', 'the window is all paper'], answer: 0, why: 'Right pixels are added, left ones subtracted.' },
      { q: 'A blur kernel of nine weights 1/9 gives at each place…', options: ['the mean of the nine pixels', 'the largest pixel', 'the edge strength', 'zero'], answer: 0, why: 'Each pixel counts one ninth.' },
      { q: 'Why does a convolution find a pattern wherever it is in the picture?', options: ['the same kernel is applied at every place', 'each place has its own weights', 'the picture is flattened first', 'pooling moves the pattern'], answer: 0, why: 'Shared weights: the response follows the pattern as it moves.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
