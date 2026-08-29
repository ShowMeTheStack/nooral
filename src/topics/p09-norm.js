// Part 9, batch and layer normalisation on a batch small enough to read: batch norm makes each feature (a column)
// have mean 0 and spread 1 across the batch; layer norm does the same for each example (a row) across its features;
// both then scale by γ and shift by β. Batch norm's output for one example depends on the rest of the batch, layer
// norm's does not. Last, a deep tanh stack with badly scaled weights, with and without batch norm after each layer.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  P.NORM_BATCHES = {
    scales: { label: 'features on very different scales', rows: [[52, 0.8, -3], [61, 1.4, 2], [47, 0.6, 5], [70, 2.2, -1]], swap: [55, 3.9, 0], names: ['f₁', 'f₂', 'f₃'] },
    odd: { label: 'one example far from the rest', rows: [[1.2, 0.4, 0.9], [0.8, 0.6, 1.1], [1, 0.5, 1], [6, 3, -4]], swap: [1.1, 0.5, 0.8], names: ['f₁', 'f₂', 'f₃'] },
    small: { label: 'small numbers near zero', rows: [[0.02, -0.01, 0.03], [0.05, 0.02, -0.02], [-0.01, 0.04, 0.01], [0.03, 0, 0.05]], swap: [0.2, -0.1, 0.1], names: ['f₁', 'f₂', 'f₃'] },
  };
  const fmtCell = (v) => (Math.abs(v) >= 10 ? H().fx(v, 1) : H().fx(v, 2));

  S.topic({
    id: 'p09-norm', family: 'p09-reg', order: 50, title: 'Batch normalisation and layer normalisation',
    blurb: 'The numbers flowing through a network drift in size from layer to layer and during training, which makes learning slow and touchy. Normalisation resets them: batch norm makes every feature have mean 0 and spread 1 over the examples of the batch, layer norm does it for every example over its own features. Then a learned scale γ and shift β give the layer back whatever range suits it.',
    inputs: [
      { id: 'batch', kind: 'choice', label: 'Batch', def: 'scales', options: Object.keys(P.NORM_BATCHES).map((k) => ({ value: k, label: P.NORM_BATCHES[k].label })) },
      { id: 'gamma', kind: 'slider', label: 'Scale γ', def: 1, min: 0.5, max: 2, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'beta', kind: 'slider', label: 'Shift β', def: 0, min: -1, max: 1, step: 0.1, fmt: (v) => H().fx(v, 1) },
    ],
    presets: [
      { label: 'Mixed scales', values: { batch: 'scales', gamma: 1, beta: 0 } },
      { label: 'An odd example', values: { batch: 'odd', gamma: 1, beta: 0 } },
      { label: 'With γ = 2, β = 0.5', values: { batch: 'scales', gamma: 2, beta: 0.5 } },
      { label: 'Tiny numbers', values: { batch: 'small', gamma: 1, beta: 0 } },
    ],
    counters: [],
    run(inp) {
      const T = S.T, F = H().frames([]), q = F.q, B = P.NORM_BATCHES[inp.batch], X = B.rows, n = X.length, c = X[0].length, eps = 1e-5;
      const x = T.tensor(X), gam = T.full([c], inp.gamma), bet = T.full([c], inp.beta);
      const bn = T.batchNorm(x, gam, bet, { training: true, eps }), ln = T.layerNorm(x, gam, bet, eps);
      const colMean = Array.from(T.mean(x, 0).values()), colStd = Array.from(T.variance(x, 0).values()).map(Math.sqrt);
      const rowMean = Array.from(T.mean(x, 1).values()), rowStd = Array.from(T.variance(x, 1).values()).map(Math.sqrt);
      const BN = Array.from({ length: n }, (_, i) => Array.from(bn.values()).slice(i * c, i * c + c)), LN = Array.from({ length: n }, (_, i) => Array.from(ln.values()).slice(i * c, i * c + c));
      const big = Math.max(...X.flat().map(Math.abs));
      const grid = (prefix, x0, y0, M, title, o) => S.Kit.heat.grid(Object.assign({ prefix, x: x0, y: y0, cw: 78, ch: 34, vals: M, scheme: 'div', max: o && o.max ? o.max : 2.2, fmt: fmtCell, rows: M.map((_, i) => 'x' + P.sub(i + 1)), cols: B.names, title, size: 14, labelSize: 14, titleSize: 13 }, o || {})).items;
      const stats = (prefix, x0, y0, vals, label, cls) => [H().text(prefix + ':l', x0 - 8, y0 + 22, label, 'legend', 13, 'end'), ...vals.map((v, j) => ({ key: prefix + ':' + j, kind: 'cell', x: x0 + j * 80, y: y0, w: 78, h: 30, rx: 3, text: fmtCell(v), size: 14, cls: 'dt ' + (cls || '') }))];
      const rowStats = (prefix, x0, y0, m, s) => [H().text(prefix + ':h', x0 + 39, y0 - 8, 'mean', 'legend', 13, 'middle'), H().text(prefix + ':h2', x0 + 119, y0 - 8, 'spread', 'legend', 13, 'middle'), ...m.map((v, i) => ({ key: prefix + ':m' + i, kind: 'cell', x: x0, y: y0 + i * 36, w: 78, h: 34, rx: 3, text: fmtCell(v), size: 14, cls: 'dt lit' })), ...s.map((v, i) => ({ key: prefix + ':s' + i, kind: 'cell', x: x0 + 80, y: y0 + i * 36, w: 78, h: 34, rx: 3, text: fmtCell(v), size: 14, cls: 'dt lit' }))];
      const left = (o) => grid('nx', 80, 70, X, 'the batch: ' + n + ' examples × ' + c + ' features', Object.assign({ max: big }, o || {}));
      F.push(left(), `A batch of ${q.n(n)} examples (rows) with ${q.n(c)} features (columns), as they might arrive at a layer. Feature f₁ runs from ${q(Math.min(...X.map((r) => r[0])), 2)} to ${q(Math.max(...X.map((r) => r[0])), 2)}, feature f₂ from ${q(Math.min(...X.map((r) => r[1])), 2)} to ${q(Math.max(...X.map((r) => r[1])), 2)}: the next layer would see wildly different scales.`, { phase: 'start', X, n, c }, { hl: { eq: [] } });
      F.push([...left({ colCls: () => 'act' }), ...stats('bm', 80, 230, colMean, 'mean', 'lit'), ...stats('bs', 80, 264, colStd, 'spread', 'lit')],
        `Batch norm works column by column, over the batch. Feature f₁ has mean ${q(colMean[0], 3)} and spread (standard deviation) ${q(colStd[0], 3)}; f₂ has ${q(colMean[1], 3)} and ${q(colStd[1], 3)}; f₃ has ${q(colMean[2], 3)} and ${q(colStd[2], 3)}.`, { phase: 'bnstats', colMean, colStd }, { hl: { eq: [0] } });
      F.push([...left(), ...stats('bm', 80, 230, colMean, 'mean', 'lit'), ...stats('bs', 80, 264, colStd, 'spread', 'lit'), ...grid('nb', 500, 70, BN, 'batch norm')],
        `Each value has its column’s mean subtracted and is divided by its column’s spread (with a tiny ε for safety), then scaled by γ = ${q(inp.gamma, 1)} and shifted by β = ${q(inp.beta, 1)}. The first example’s f₁ becomes (${q(X[0][0], 2)} − ${q(colMean[0], 3)}) / ${q(colStd[0], 3)} × γ + β = ${q(BN[0][0], 3)}. Every column now has mean β and spread γ, whatever its scale was.`, { phase: 'bn', BN, x00: X[0][0], gamma: inp.gamma, beta: inp.beta, m0: colMean[0], s0: colStd[0] }, { hl: { eq: [0, 1] } });
      F.push([...left({ rowCls: () => 'act' }), ...rowStats('lr', 330, 70, rowMean, rowStd)],
        `Layer norm works row by row instead: each example over its own ${q.n(c)} features. Example x₁ has mean ${q(rowMean[0], 3)} and spread ${q(rowStd[0], 3)}; example x${P.sub(n)} has ${q(rowMean[n - 1], 3)} and ${q(rowStd[n - 1], 3)}.`, { phase: 'lnstats', rowMean, rowStd, c }, { hl: { eq: [2] } });
      F.push([...left(), ...rowStats('lr', 330, 70, rowMean, rowStd), ...grid('nl', 580, 70, LN, 'layer norm')],
        `Each example is centred on its own mean and divided by its own spread, then scaled and shifted by γ and β. Example x₁ becomes ${P.qvec(q, LN[0], 2)}. It never looks at the other examples, so it works the same with a batch of one, which is why transformers and recurrent networks use it.`, { phase: 'ln', LN }, { hl: { eq: [2, 1] } });
      // swap the last example: batch norm of x1 changes, layer norm of x1 does not
      const X2 = X.slice(0, n - 1).concat([B.swap]), x2 = T.tensor(X2);
      const bn2 = Array.from(T.batchNorm(x2, gam, bet, { training: true, eps }).values()).slice(0, c), ln2 = Array.from(T.layerNorm(x2, gam, bet, eps).values()).slice(0, c);
      F.push([...grid('nx', 80, 70, X2, 'the batch, last example replaced', { max: big, rowCls: (i) => (i === n - 1 ? 'act' : '') }), ...grid('nb', 560, 70, [BN[0], bn2], 'batch norm of x₁', { rows: ['before', 'after'] }), ...grid('nl', 560, 205, [LN[0], ln2], 'layer norm of x₁', { rows: ['before', 'after'] })],
        `Replace only the last example with ${P.qvec(q, B.swap, 2)}. The first example has not changed, yet its batch norm output moves from ${P.qvec(q, BN[0], 2)} to ${P.qvec(q, bn2, 2)}, because the column statistics moved. Its layer norm output stays ${P.qvec(q, ln2, 2)}. Batch norm therefore keeps running averages of the statistics during training and uses those at test time.`, { phase: 'swap', bn1: BN[0], bn2, ln1: LN[0], ln2, swap: B.swap }, { hl: { eq: [3] } });
      // a deep tanh stack with badly scaled weights, with and without batch norm after each layer
      const deep = (useBN) => {
        const g = T.rng(4), D = 10, W = 32;
        let h = T.normal([64, W], 0, 1, g);
        const out = [];
        for (let l = 0; l < D; l++) {
          let z = T.matmul(h, T.normal([W, W], 0, 0.06, g));
          if (useBN) z = T.batchNorm(z, null, null, { training: true });
          h = T.tanh(z);
          out.push(Math.sqrt(Array.from(T.variance(h, null).values())[0]));
        }
        return out;
      };
      const plain = deep(false), withBN = deep(true), lg = (v) => Math.log10(Math.max(1e-9, v));
      const lo = Math.floor(Math.min(...plain.map(lg), -1));
      const ax = P.ax({ prefix: 'dn', x: 80, y: 70, w: 460, h: 230, xr: [0.6, 10.4], yr: [lo, 0.3], xticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], yticks: Array.from({ length: -lo + 1 }, (_, k) => lo + k), fmtY: (v) => '10' + String(v).split('').map((ch) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(ch)]).join(''), xname: 'layer', yname: 'spread of the activations' });
      const items = [...ax.items, C().line(ax, 'dn:a', plain.map((v, l) => [l + 1, lg(v)]), 'p09val'), C().line(ax, 'dn:b', withBN.map((v, l) => [l + 1, lg(v)]), 'p09train'), ...plain.map((v, l) => P.dot(ax, 'dn:ad' + l, l + 1, lg(v), 'p09val', 4)), ...withBN.map((v, l) => P.dot(ax, 'dn:bd' + l, l + 1, lg(v), 'p09train', 4)), ...C().legend('dn:lg', 600, 120, [{ text: 'plain tanh layers', cls: 'p09val' }, { text: 'with batch norm', cls: 'p09train' }], { dir: 'col', size: 14 })];
      F.push(items, `Why it helps: ${q.n(plain.length)} tanh layers of ${q.n(32)} neurons with weights drawn too small (standard deviation ${q(0.06, 2)}). Without normalisation the spread of the activations falls from ${q.sg(plain[0], 2)} to ${q.sg(plain[plain.length - 1], 2)} by the last layer. With batch norm before each tanh it is reset every layer and ends at ${q.sg(withBN[withBN.length - 1], 2)}: the signal keeps its size however the weights were drawn.`, { phase: 'deep', plain, withBN, depth: plain.length, width: 32, std: 0.06 }, { hl: { eq: [4] } });
      P.narrow(F.list, (m) => ({ start: { x: 20, y: 30, w: 400, h: 240 }, bnstats: { x: 20, y: 30, w: 400, h: 290 }, bn: { x: 470, y: 30, w: 330, h: 220 }, lnstats: { x: 280, y: 30, w: 250, h: 220 }, ln: { x: 530, y: 30, w: 320, h: 220 }, swap: { x: 500, y: 40, w: 340, h: 260 }, deep: { x: 20, y: 40, w: 560, h: 330 } }[m.phase] || null));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Normalising', text: ['batch norm: per feature j, over the batch: x̂ᵢⱼ = (xᵢⱼ − μⱼ) / √(σⱼ² + ε)', 'then y = γ x̂ + β  (γ, β learned)', 'layer norm: per example i, over its features: (xᵢⱼ − μᵢ) / √(σᵢ² + ε)', 'batch norm depends on the batch: running averages at test time', 'effect: activations keep a steady size, layer after layer'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'ln', once: true, pick: 'option',
      ask: 'Replace only the last example of the batch. Which normalised output of the first example changes?',
      options: ['batch norm only', 'layer norm only', 'both', 'neither'],
      answer: () => 'batch norm only',
      explain: (fr) => fr.find((f) => f.meta.phase === 'swap').say,
    }],
    quiz: [
      { q: 'Batch normalisation computes its mean and spread…', options: ['per feature, over the examples of the batch', 'per example, over its features', 'over the whole dataset once', 'per weight'], answer: 0, why: 'Each column of the batch is normalised.' },
      { q: 'Why does layer normalisation suit a batch of one example?', options: ['it only uses that example’s own features', 'it ignores the features', 'it needs no γ and β', 'it uses running averages'], answer: 0, why: 'Batch statistics from one example would be meaningless; layer norm never needs them.' },
      { q: 'What do γ and β do after normalising?', options: ['let the layer learn any scale and offset it needs', 'undo the normalisation every time', 'set the learning rate', 'pick the batch'], answer: 0, why: 'Normalising fixes the range; γ and β make it a learned choice.' },
      { q: 'At test time, batch norm uses…', options: ['running averages of the mean and spread kept during training', 'the statistics of the test batch', 'zero and one', 'layer norm instead'], answer: 0, why: 'Predictions must not depend on what else is in the batch.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
