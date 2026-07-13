// Regression metrics (MAE, MSE, RMSE, R², explained variance, MAPE, residuals), top-k accuracy for rankings,
// and two text measures: perplexity from token probabilities and a BLEU-like clipped n-gram precision.
(function (S) {
  'use strict';
  const M = (S.Metrics = S.Metrics || {});
  const f = (x) => M.f(x), st = (k, s, d) => M.step(k, s, d);
  const EPS = 2.220446049250313e-16;

  M.residuals = (y, yHat) => {
    M.check(y, yHat, 'residuals');
    const r = y.map((v, i) => v - yHat[i]);
    let big = 0; r.forEach((v, i) => { if (Math.abs(v) > Math.abs(r[big])) big = i; });
    return { value: r, residuals: r, largest: big, trace: [st('residuals', 'Residual = actual − predicted for each of ' + r.length + ' examples; the largest is ' + f(r[big]) + ' at example ' + big + ' (actual ' + f(y[big]) + ', predicted ' + f(yHat[big]) + ').', { residuals: r, largest: big })] };
  };

  M.mae = (y, yHat) => {
    const r = M.residuals(y, yHat).value, s = M.sum(r.map(Math.abs)), value = s / r.length;
    return { value, trace: [st('mae', 'MAE = Σ|actual − predicted| / n = ' + f(s) + ' / ' + r.length + ' = ' + f(value) + '.', { value })] };
  };
  M.mse = (y, yHat) => {
    const r = M.residuals(y, yHat).value, s = M.sum(r.map((v) => v * v)), value = s / r.length;
    return { value, trace: [st('mse', 'MSE = Σ(actual − predicted)² / n = ' + f(s) + ' / ' + r.length + ' = ' + f(value) + '.', { value })] };
  };
  M.rmse = (y, yHat) => {
    const m = M.mse(y, yHat), value = Math.sqrt(m.value);
    return { value, trace: [...m.trace, st('rmse', 'RMSE = √MSE = √' + f(m.value) + ' = ' + f(value) + ', back in the units of the target.', { value })] };
  };
  /** R² = 1 − SS_res / SS_tot; a constant target gives 1 for a perfect fit and 0 otherwise */
  M.r2 = (y, yHat) => {
    const r = M.residuals(y, yHat).value, mu = M.mean(y);
    const ssRes = M.sum(r.map((v) => v * v)), ssTot = M.sum(y.map((v) => (v - mu) ** 2));
    const value = ssTot === 0 ? (ssRes === 0 ? 1 : 0) : 1 - ssRes / ssTot;
    return { value, ssRes, ssTot, trace: [
      st('sst', 'Total spread around the mean ' + f(mu) + ': SS_tot = ' + f(ssTot) + '.', { ssTot, mean: mu }),
      st('ssr', 'Left unexplained by the model: SS_res = ' + f(ssRes) + '.', { ssRes }),
      st('r2', ssTot === 0 ? 'The target is constant, so R² is set to ' + value + '.' : 'R² = 1 − ' + f(ssRes) + ' / ' + f(ssTot) + ' = ' + f(value) + '.', { value })] };
  };
  /** explained variance = 1 − Var(residual) / Var(target): unlike R², a constant offset is not penalised */
  M.explainedVariance = (y, yHat) => {
    const r = M.residuals(y, yHat).value, n = y.length;
    const var_ = (a) => { const m = M.mean(a); return M.sum(a.map((v) => (v - m) ** 2)) / n; };
    const vr = var_(r), vy = var_(y), value = vy === 0 ? (vr === 0 ? 1 : 0) : 1 - vr / vy;
    return { value, trace: [st('ev', 'Explained variance = 1 − Var(residuals) / Var(target) = 1 − ' + f(vr) + ' / ' + f(vy) + ' = ' + f(value) + '; the mean residual ' + f(M.mean(r)) + ' is not counted.', { value })] };
  };
  /** mean absolute percentage error, as a fraction (0.1 means 10%); a zero target is guarded by a tiny ε */
  M.mape = (y, yHat) => {
    const p = y.map((v, i) => Math.abs(v - yHat[i]) / Math.max(Math.abs(v), EPS)), value = M.mean(p);
    return { value, trace: [st('mape', 'MAPE = mean of |actual − predicted| / |actual| = ' + M.pct(value) + (y.some((v) => v === 0) ? '; a zero actual value makes this explode, which is why MAPE is risky near 0.' : '.'), { value })] };
  };
  M.regressionReport = (y, yHat) => ({ mae: M.mae(y, yHat).value, mse: M.mse(y, yHat).value, rmse: M.rmse(y, yHat).value, r2: M.r2(y, yHat).value, explainedVariance: M.explainedVariance(y, yHat).value, mape: M.mape(y, yHat).value, residuals: M.residuals(y, yHat).value });

  /**
   * topK(yTrue, scores, k, {labels}): a hit when the true label is among the k highest-scoring classes.
   * scores rows have one column per sorted label; among equal scores the later column ranks first.
   * With a 1-D list of positive scores (two classes), k = 1 thresholds at 0.5 (or 0 for unbounded scores).
   */
  M.topK = (yTrue, scores, k = 1, opt = {}) => {
    const labels = opt.labels || M.labels(yTrue), ix = M.index(labels);
    let hits;
    if (scores.length && typeof scores[0] === 'number') {
      const t = Math.min(...scores) >= 0 && Math.max(...scores) <= 1 ? 0.5 : 0;
      hits = scores.map((s, i) => (k >= 2 ? true : (s > t ? 1 : 0) === ix(yTrue[i])));
    } else {
      hits = scores.map((row, i) => {
        const order = Array.from(row, (v, j) => j).sort((a, b) => row[b] - row[a] || b - a);
        return order.slice(0, k).includes(ix(yTrue[i]));
      });
    }
    const c = hits.filter(Boolean).length, value = c / hits.length;
    return { value, hits, trace: [st('topk', 'The true class is among the top ' + k + ' guesses for ' + c + ' of ' + hits.length + ' examples, so top-' + k + ' accuracy = ' + f(value) + '.', { hits, k })] };
  };

  /** perplexity = exp(mean −ln p) over the probabilities a model gave to each actual next token */
  M.perplexity = (probs) => {
    const nll = probs.map((p) => -Math.log(Math.max(p, 1e-300))), mean = M.mean(nll), value = Math.exp(mean);
    return { value, meanNll: mean, trace: [
      st('nll', 'The model gave the actual tokens probabilities whose mean −ln p is ' + f(mean) + ' nats over ' + probs.length + ' tokens.', { nll }),
      st('ppl', 'Perplexity = e^' + f(mean) + ' = ' + f(value) + ': the model is as unsure as if it chose among about ' + Math.round(value) + ' equally likely tokens.', { value })] };
  };

  const tokens = (s) => (Array.isArray(s) ? s : String(s).toLowerCase().match(/[\p{L}\p{N}']+|[^\s\p{L}\p{N}]/gu) || []);
  /** references: one string, or a list of strings or token lists */
  const refList = (refs) => (typeof refs === 'string' ? [refs] : refs);
  const grams = (t, n) => { const m = new Map(); for (let i = 0; i + n <= t.length; i++) { const g = t.slice(i, i + n).join(' '); m.set(g, (m.get(g) || 0) + 1); } return m; };
  /**
   * ngramPrecision(candidate, references, n): clipped n-gram precision — each candidate n-gram counts at most as
   * often as it appears in the most generous reference.
   */
  M.ngramPrecision = (cand, refs, n = 1) => {
    const c = grams(tokens(cand), n), rs = refList(refs).map((r) => grams(tokens(r), n));
    let match = 0, total = 0; const detail = [];
    for (const [g, k] of c) { const cap = Math.max(0, ...rs.map((r) => r.get(g) || 0)), m = Math.min(k, cap); match += m; total += k; detail.push({ gram: g, count: k, clip: cap, matched: m }); }
    const value = M.div(match, total);
    return { value, match, total, detail, trace: [st('ngram', n + '-gram precision: ' + match + ' of the ' + total + ' candidate ' + n + '-grams also appear in a reference (each counted at most as often as there) = ' + f(value) + '.', { n, match, total })] };
  };
  /** bleuLike(candidate, references, maxN = 4): geometric mean of clipped 1..N-gram precisions times a brevity penalty */
  M.bleuLike = (cand, refs, maxN = 4) => {
    const rl = refList(refs).map((r) => tokens(r).length), cl = tokens(cand).length;
    const closest = rl.reduce((a, b) => (Math.abs(b - cl) < Math.abs(a - cl) || (Math.abs(b - cl) === Math.abs(a - cl) && b < a) ? b : a), rl[0]);
    const ps = []; for (let n = 1; n <= maxN; n++) ps.push(M.ngramPrecision(cand, refList(refs), n));
    const bp = cl === 0 ? 0 : cl >= closest ? 1 : Math.exp(1 - closest / cl);
    const value = ps.some((p) => p.value === 0) ? 0 : bp * Math.exp(M.mean(ps.map((p) => Math.log(p.value))));
    return { value, precisions: ps.map((p) => p.value), brevity: bp, trace: [...ps.flatMap((p) => p.trace),
      st('brevity', 'The candidate has ' + cl + ' tokens against a reference of ' + closest + ', so the brevity penalty is ' + f(bp) + '.', { bp }),
      st('bleu', 'Score = brevity × geometric mean of the ' + maxN + ' precisions = ' + f(value) + (value === 0 ? ' (one precision is 0, so the product is 0).' : '.'), { value })] };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
