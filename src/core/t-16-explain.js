// Step-by-step traces of the core ops on small inputs, for lessons and exercises. Each returns {out, trace} where
// trace is a list of steps {kind, say, ...data}, say being one sentence with the actual numbers.
//   S.T.explain.matmul(a, b)            one step per output cell: the row, the column and their dot product
//   S.T.explain.softmax(x)              the shift by the maximum, the exponentials, the sum and the shares
//   S.T.explain.conv2d(x, w, b?, opts)  one step per output position of the first image: the window and the sum
//   S.T.explain.attention(q, k, v, {causal, mask})  scores, scaling, masking, softmax per query, weighted sums
//   S.T.explain.backward(loss)          the forward and backward traces of a recorded graph (t-01)
(function (S) {
  'use strict';
  const T = S.T, f = (x) => T.fmt(x), step = T.step;
  const E = (T.explain = {});
  const rowsOf = (t) => t.toArray();

  E.matmul = (a, b) => {
    const A = rowsOf(T.as(a)), Bm = rowsOf(T.as(b)), M = A.length, K = Bm.length, N = Bm[0].length, trace = [], out = [];
    if (A[0].length !== K) throw new Error('inner sizes differ');
    trace.push(step('shape', `A is ${M}×${K} and B is ${K}×${N}, so the product is ${M}×${N}: each entry is a row of A times a column of B.`, { M, K, N }));
    for (let i = 0; i < M; i++) {
      out.push([]);
      for (let j = 0; j < N; j++) {
        const terms = A[i].map((x, k) => x * Bm[k][j]), s = terms.reduce((p, q) => p + q, 0);
        out[i].push(s);
        trace.push(step('cell', `C[${i}][${j}] = ${A[i].map((x, k) => f(x) + '·' + f(Bm[k][j])).join(' + ')} = ${f(s)}.`, { i, j, row: A[i], col: Bm.map((r) => r[j]), terms, value: s }));
      }
    }
    return { out: T.tensor(out), trace };
  };

  E.softmax = (x) => {
    const v = Array.from(T.as(x).values()), m = Math.max(...v), sh = v.map((z) => z - m), ex = sh.map(Math.exp), s = ex.reduce((p, q) => p + q, 0), p = ex.map((e) => e / s);
    const trace = [
      step('max', `The largest score is ${f(m)}; subtracting it from every score keeps the exponentials from overflowing: ${sh.map(f).join(', ')}.`, { max: m, shifted: sh }),
      step('exp', `The exponentials are ${ex.map(f).join(', ')}.`, { exp: ex }),
      step('sum', `They add up to ${f(s)}.`, { sum: s }),
      step('divide', `Dividing each by the sum gives probabilities ${p.map(f).join(', ')}, which add up to 1.`, { probs: p }),
    ];
    return { out: T.tensor(p), trace };
  };

  E.conv2d = (x, w, b, opts = {}) => {
    const out = T.noGrad(() => T.conv2d(x, w, b, opts)), [, C, H, W] = x.shape, [O, Cg, kh, kw] = w.shape, Ho = out.shape[2], Wo = out.shape[3];
    const pad = Array.isArray(opts.padding) ? opts.padding : [opts.padding || 0, opts.padding || 0], st = Array.isArray(opts.stride) ? opts.stride : [opts.stride || 1, opts.stride || 1];
    const dl = Array.isArray(opts.dilation) ? opts.dilation : [opts.dilation || 1, opts.dilation || 1];
    const trace = [step('shape', `A ${H}×${W} image with ${C} channel${C > 1 ? 's' : ''} and ${O} filter${O > 1 ? 's' : ''} of ${kh}×${kw} (stride ${st[0]}, padding ${pad[0]}) gives ${O} map${O > 1 ? 's' : ''} of ${Ho}×${Wo}.`, { H, W, C, O, kh, kw, Ho, Wo })];
    const groups = opts.groups || 1, Og = O / groups;
    for (let o = 0; o < O; o++) for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
      const window = [];
      let s = b ? b.values()[o] : 0;
      const c0 = Math.floor(o / Og) * Cg;
      for (let c = 0; c < Cg; c++) for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
        const y = oy * st[0] - pad[0] + i * dl[0], xx = ox * st[1] - pad[1] + j * dl[1];
        const xv = y >= 0 && y < H && xx >= 0 && xx < W ? x.get(0, c0 + c, y, xx) : 0, wv = w.get(o, c, i, j);
        window.push({ c: c0 + c, y, x: xx, value: xv, weight: wv }); s += xv * wv;
      }
      trace.push(step('window', `Filter ${o} over the window at row ${oy * st[0] - pad[0]}, column ${ox * st[1] - pad[1]}: the ${window.length} products add up to ${f(s)}${b ? ' (with the bias)' : ''}.`, { o, oy, ox, window, value: s }));
    }
    return { out, trace };
  };

  E.attention = (q, k, v, opts = {}) => {
    const Q = rowsOf(q), K = rowsOf(k), V = rowsOf(v), d = Q[0].length, sc = 1 / Math.sqrt(d), trace = [], out = [], weights = [];
    const mask = opts.mask ? rowsOf(T.as(opts.mask)) : null;
    trace.push(step('setup', `${Q.length} queries attend over ${K.length} keys of width ${d}; scores are divided by √${d} = ${f(Math.sqrt(d))}.`, { d, scale: sc }));
    Q.forEach((qr, i) => {
      const raw = K.map((kr) => kr.reduce((s, x, j) => s + x * qr[j], 0)), s = raw.map((z) => z * sc);
      const allowed = K.map((_, j) => !(opts.causal && j > i) && !(mask && !mask[i][j]));
      const m = Math.max(...s.filter((_, j) => allowed[j])), e = s.map((z, j) => (allowed[j] ? Math.exp(z - m) : 0)), tot = e.reduce((p, x) => p + x, 0), w = e.map((x) => x / tot);
      const o = V[0].map((_, c) => w.reduce((p, wj, j) => p + wj * V[j][c], 0));
      weights.push(w); out.push(o);
      trace.push(step('scores', `Query ${i} scores the keys ${raw.map(f).join(', ')}; scaled: ${s.map(f).join(', ')}.`, { i, raw, scaled: s }));
      if (allowed.some((a) => !a)) trace.push(step('mask', `Query ${i} may not look at key${allowed.filter((a) => !a).length > 1 ? 's' : ''} ${allowed.map((a, j) => (a ? null : j)).filter((j) => j !== null).join(', ')}, so those scores become −∞.`, { i, allowed }));
      trace.push(step('softmax', `Softmax turns them into weights ${w.map(f).join(', ')}.`, { i, weights: w }));
      trace.push(step('mix', `The output for query ${i} is the weighted sum of the values: ${o.map(f).join(', ')}.`, { i, out: o }));
    });
    return { out: T.tensor(out), weights: T.tensor(weights), trace };
  };

  E.backward = (loss) => ({ forward: T.traceForward(loss), backward: T.traceBackward(loss), graph: T.graph(loss) });
})(globalThis.Nooral = globalThis.Nooral || {});
