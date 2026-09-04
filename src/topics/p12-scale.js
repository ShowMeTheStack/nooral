// Part 12, why attention divides its scores by √dₖ: random queries and keys of growing size, their dot products
// spreading out like √dₖ, and the softmax of unscaled scores collapsing onto a single key (with gradients that
// vanish), while the scaled ones stay spread. Measured with S.T.attention, with and without its scaling.
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml, C = () => S.Kit.curve;
  /** a point on a plot; points may sit on one another */
  const dotOk = (...a) => Object.assign(C().dot(...a), { audit: 'overlap-ok', point: false });
  P.SCALE_DIMS = [2, 4, 8, 16, 32, 64, 128, 256];
  /** m queries against n keys each, entries drawn from a standard normal: {q: [m·d], k: [m·n·d]} (float32, row by row) */
  P.scaleDraw = (seed, d, n, m) => {
    const g = S.T.rng(seed * 1000 + d), q = new Float32Array(m * d), k = new Float32Array(m * n * d);
    for (let r = 0; r < m; r++) { for (let j = 0; j < d; j++) q[r * d + j] = g.normal(); for (let j = 0; j < n * d; j++) k[r * n * d + j] = g.normal(); }
    return { q, k };
  };
  /** the statistics for one size: spread of the raw scores, mean largest weight and mean Σ p(1 − p), unscaled and scaled */
  P.scaleStats = (seed, d, n, m) => {
    const T = S.T, { q, k } = P.scaleDraw(seed, d, n, m), Q = T.make(q, [m, 1, d]), K = T.make(k, [m, n, d]), V = T.zeros([m, n, 1]);
    const un = T.attention(Q, K, V, { scale: 1 }), sc = T.attention(Q, K, V);
    const raw = un.attn.scores.values(), wu = un.attn.weights.values(), ws = sc.attn.weights.values();
    let s1 = 0, s2 = 0;
    for (const v of raw) { s1 += v; s2 += v * v; }
    const mean = s1 / raw.length, sd = Math.sqrt(s2 / raw.length - mean * mean);
    const rowStats = (w) => { let mx = 0, gr = 0, sharp = 0; for (let r = 0; r < m; r++) { let best = 0, g = 0; for (let j = 0; j < n; j++) { const p = w[r * n + j]; best = Math.max(best, p); g += p * (1 - p); } mx += best; gr += g; if (best > 0.9) sharp++; } return { max: mx / m, grad: gr / m, sharp: sharp / m }; };
    return { d, sd, un: rowStats(wu), sc: rowStats(ws), ex: { raw: Array.from(raw.subarray(0, n)), wu: Array.from(wu.subarray(0, n)), ws: Array.from(ws.subarray(0, n)) } };
  };

  S.topic({
    id: 'attention-scale', family: 'p12-attn', order: 40, title: 'Why the scores are divided by √dₖ',
    blurb: 'A dot product adds up dₖ products, so with longer queries and keys the scores spread wider: their typical size grows like √dₖ. Fed to a softmax, wide scores make one key take nearly all the weight, and the gradient that training needs almost vanishes. Dividing by √dₖ brings the spread back to about 1 whatever the size. Here random queries and keys of size 2 to 256 are measured both ways.',
    inputs: [
      { id: 'n', kind: 'choice', label: 'Keys per query', def: '8', options: [{ value: '4', label: '4' }, { value: '8', label: '8' }, { value: '16', label: '16' }] },
      { id: 'show', kind: 'choice', label: 'Example row at dₖ', def: '128', options: P.SCALE_DIMS.slice(2).map((d) => ({ value: String(d), label: String(d) })) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: '8 keys', values: { n: '8', show: '128', seed: 1 } },
      { label: '4 keys', values: { n: '4', show: '64', seed: 2 } },
      { label: '16 keys, dₖ = 256', values: { n: '16', show: '256', seed: 3 } },
      { label: 'Small dₖ = 8', values: { n: '8', show: '8', seed: 5 } },
    ],
    counters: [{ id: 'sizes', label: 'sizes measured' }],
    run(inp) {
      const F = H().frames(['sizes']), q = F.q, n = +inp.n, m = 100, show = +inp.show;
      const stats = P.SCALE_DIMS.map((d) => P.scaleStats(inp.seed, d, n, m));
      const lx = (d) => Math.log2(d), xt = P.SCALE_DIMS.map(lx), fmtX = (u) => String(Math.round(Math.pow(2, u)));
      const maxSd = Math.ceil(Math.max(...stats.map((s) => s.sd)) / 4) * 4;
      const A = C().axes({ prefix: 'sa', x: 70, y: 56, w: 300, h: 220, xr: [1, 8], yr: [0, maxSd], xticks: xt, fmtX, xname: 'dₖ (size of q and k)', yname: 'spread of the scores (std)' });
      const B = C().axes({ prefix: 'sb', x: 470, y: 56, w: 300, h: 220, xr: [1, 8], yr: [0, 1], xticks: xt, fmtX, yticks: [0, 0.25, 0.5, 0.75, 1], xname: 'dₖ', yname: 'largest weight in a row (mean)' });
      const legend = C().legend('sl', 470, 336, [{ text: 'q·k (unscaled)', cls: 'p12c' }, { text: 'q·k / √dₖ (scaled)', cls: 'p12a' }], { size: 13 });
      const curves = (upto) => {
        const s = stats.slice(0, upto), items = [];
        if (s.length > 1) {
          items.push(C().line(A, 'sa:u', s.map((x) => [lx(x.d), x.sd]), 'p12c'), C().line(A, 'sa:s', s.map((x) => [lx(x.d), x.sd / Math.sqrt(x.d)]), 'p12a'));
          items.push(C().line(B, 'sb:u', s.map((x) => [lx(x.d), x.un.max]), 'p12c'), C().line(B, 'sb:s', s.map((x) => [lx(x.d), x.sc.max]), 'p12a'));
        }
        s.forEach((x, k) => items.push(dotOk(A, 'sa:du' + k, lx(x.d), x.sd, 'p12c'), dotOk(A, 'sa:ds' + k, lx(x.d), x.sd / Math.sqrt(x.d), 'p12a'), dotOk(B, 'sb:du' + k, lx(x.d), x.un.max, 'p12c'), dotOk(B, 'sb:ds' + k, lx(x.d), x.sc.max, 'p12a')));
        return items;
      };
      const base = [...A.items, ...B.items, ...legend];
      F.push([...base], `Each query is compared with ${P.qpl(q, n, 'key')}; ${q.n(m)} queries are drawn for every size, their numbers from a normal distribution with mean 0 and spread 1. Left: how widely the scores spread. Right: how much of a row’s weight its largest entry takes on average (${q(1 / n, 3)} would be perfectly even).`, { phase: 'start', n, m, even: 1 / n });
      stats.forEach((x, k) => {
        F.count('sizes');
        F.push([...base, ...curves(k + 1)], `dₖ = ${q.n(x.d)}: the raw scores have spread ${q(x.sd, 2)}, close to √${q.n(x.d)} = ${q(Math.sqrt(x.d), 2)}; after dividing by it the spread is ${q(x.sd / Math.sqrt(x.d), 2)}. The largest weight in a row is ${q(x.un.max, 2)} unscaled and ${q(x.sc.max, 2)} scaled.` + (x.un.sharp > 0.5 ? ` Unscaled, ${q.pct(x.un.sharp)} of the rows give one key more than ${q.pct(0.9)} of the weight.` : ''), { phase: 'size', d: x.d, sd: x.sd, root: Math.sqrt(x.d), sdScaled: x.sd / Math.sqrt(x.d), un: x.un.max, sc: x.sc.max, sharp: x.un.sharp, pct90: 0.9 });
      });
      const e = stats.find((x) => x.d === show), bw = 300 / n - 4;
      const bars = (key, x0, w, title, cls) => [H().text(key + ':t', x0, 392, title, 'strong', 13), ...w.map((p, j) => ({ key: key + ':b' + j, kind: 'cell', x: P.r2(x0 + j * (bw + 4)), y: P.r2(480 - 70 * p), w: P.r2(bw), h: P.r2(Math.max(1, 70 * p)), rx: 2, text: '', cls: 'p12bar ' + cls, audit: 'ignore' })), { key: key + ':base', kind: 'line', x1: x0, y1: 481, x2: x0 + 300, y2: 481, cls: 'faint', audit: 'ignore' }];
      const bu = P.argmax(e.ex.wu);
      F.push([...base, ...curves(stats.length), ...bars('xu', 70, e.ex.wu, 'one row at dₖ = ' + show + ', unscaled', 'hot'), ...bars('xs', 470, e.ex.ws, 'the same row, scaled', '')],
        `One query at dₖ = ${q.n(show)}: its raw scores run from ${q(Math.min(...e.ex.raw), 1)} to ${q(Math.max(...e.ex.raw), 1)}. Unscaled, the softmax gives key ${q.n(bu)} a weight of ${q(e.ex.wu[bu], 3)}; scaled, the same key gets ${q(e.ex.ws[bu], 3)} and the others keep a share.`, { phase: 'example', d: show, raw: e.ex.raw, wu: e.ex.wu, ws: e.ex.ws, bu });
      const big = stats[stats.length - 1];
      F.push([...base, ...curves(stats.length), ...bars('xu', 70, e.ex.wu, 'one row at dₖ = ' + show + ', unscaled', 'hot'), ...bars('xs', 470, e.ex.ws, 'the same row, scaled', '')],
        `Why it matters for learning: a weight p changes with its score at rate p(1 − p), which is near 0 when p is near 0 or 1. Summed over a row and averaged, that rate is ${q(big.un.grad, 3)} unscaled at dₖ = ${q.n(big.d)} against ${q(big.sc.grad, 3)} scaled: unscaled attention barely learns. Dividing by √dₖ keeps the scores near spread 1 at every size.`, { phase: 'grad', d: big.d, gu: big.un.grad, gs: big.sc.grad });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The spread of a dot product', text: ['q · k = q₁k₁ + … + q_d k_d', 'each term: mean 0, variance 1', 'so q · k has variance dₖ, spread √dₖ', 'q · k / √dₖ has spread about 1'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'size' && f.meta.d === 16, once: true, pick: 'option',
      ask: 'At dₖ = 256, will the unscaled scores spread about twice, four times or sixteen times as wide as at dₖ = 16?',
      options: ['about twice', 'about four times', 'about sixteen times'],
      answer: 'about four times',
      explain: 'The spread grows like √dₖ: √256 / √16 = 16 / 4 = 4.',
    }],
    quiz: [
      { q: 'Queries and keys of size 100 with independent entries of spread 1: the dot product has a spread of about…', options: ['10', '100', '1', '50'], answer: 0, why: 'Variance adds up over 100 terms, so the spread is √100.' },
      { q: 'Without scaling, large dₖ makes the softmax…', options: ['put almost all the weight on one key', 'give every key the same weight', 'produce negative weights', 'stop adding up to 1'], answer: 0, why: 'Wide scores push the exponentials far apart.' },
      { q: 'Why is a nearly one-hot softmax bad for training?', options: ['its gradient p(1 − p) is nearly 0', 'it needs more memory', 'it cannot be computed', 'it makes the values larger'], answer: 0, why: 'Weights stuck near 0 or 1 hardly respond to their scores.' },
      { q: 'Doubling dₖ from 64 to 128 changes the scaling factor from 1/8 to…', options: ['about 1/11.3', '1/16', '1/128', '1/4'], answer: 0, why: '√128 ≈ 11.3.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
