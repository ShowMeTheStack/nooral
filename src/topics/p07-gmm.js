// Part 7, Gaussian mixtures: EM from a rough start, iteration by iteration, the ellipses (one and two standard
// deviations) breathing as the means, covariances and weights are re-estimated from the responsibilities; and
// the number of components chosen by BIC, the log-likelihood with a price for every parameter.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const TYPES = [{ value: 'full', label: 'full (any tilted ellipse)' }, { value: 'diag', label: 'diagonal (ellipses along the axes)' }, { value: 'spherical', label: 'spherical (circles)' }];
  /** any covariance form as a 2 × 2 matrix */
  P.cov2 = (c, type) => (type === 'full' ? c : type === 'diag' ? [[c[0], 0], [0, c[1]]] : [[c, 0], [0, c]]);
  const ells = (ax, key, means, covs, type) => means.flatMap((m, j) => { const c = P.cov2(covs[j], type); return [H().ellipse(ax, key + 'a' + j, m, c, 1, 'p7ell c' + (j % 6)), H().ellipse(ax, key + 'b' + j, m, c, 2, 'p7ell outer c' + (j % 6))]; });
  const reach = (means, covs, type) => means.flatMap((m, j) => { const c = P.cov2(covs[j], type), a = 2 * Math.sqrt(c[0][0]), b = 2 * Math.sqrt(c[1][1]); return [[m[0] - a, m[1] - b], [m[0] + a, m[1] + b]]; });
  const softPoints = (ax, key, X, R) => P.points(ax, key, X, R.map((r) => ML().argmax(r) % 6), { opacity: (i) => 0.25 + 0.75 * ((Math.max(...R[i]) - 1 / R[i].length) / (1 - 1 / R[i].length || 1)), r: 4.4 });

  S.topic({
    id: 'gmm-em', family: 'p07-gmm', order: 10, title: 'Gaussian mixtures fitted by EM',
    blurb: 'A Gaussian mixture says each point came from one of k Gaussian bumps, each with a mean, a covariance (the shape of its ellipse) and a weight. EM fits them by alternating two steps. E: for every point, work out the responsibility of each component, the probability that the point came from it. M: re-estimate each component from all the points, weighted by those responsibilities. The log-likelihood never falls, and the ellipses settle onto the groups.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'stretched', options: P.setOptions(['stretched', 'three', 'touching', 'uneven', 'iris']) },
      { id: 'k', kind: 'number', label: 'Components', def: 3, min: 1, max: 5 },
      { id: 'type', kind: 'choice', label: 'Covariance', def: 'full', options: TYPES },
      { id: 'seed', kind: 'number', label: 'Seed (the starting points)', def: 2, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Stretched groups, full covariances', values: { set: 'stretched', k: 3, type: 'full', seed: 2 } },
      { label: 'The same with circles only', values: { set: 'stretched', k: 3, type: 'spherical', seed: 2 } },
      { label: 'Groups that touch', values: { set: 'touching', k: 3, type: 'full', seed: 5 } },
      { label: 'Uneven sizes', values: { set: 'uneven', k: 3, type: 'full', seed: 3 } },
      { label: 'Iris petals, diagonal', values: { set: 'iris', k: 3, type: 'diag', seed: 1 } },
    ],
    counters: [{ id: 'iter', label: 'EM iterations' }],
    run(inp) {
      const d = P.data(inp.set, {}), X = d.X, n = X.length, k = inp.k, type = inp.type, F = H().frames(['iter']), q = F.q;
      // a rough start: k points picked by the seed as means, equal weights, a wide round covariance
      const pick = ML().rng(inp.seed).sample(n, k), spread = ML().mean(ML().colVars(X)), means0 = pick.map((i) => X[i].slice());
      const cov0 = means0.map(() => (type === 'full' ? [[spread, 0], [0, spread]] : type === 'diag' ? [spread, spread] : spread));
      const g = new (ML().GaussianMixture)({ k, covarianceType: type, meansInit: means0, weightsInit: new Array(k).fill(1 / k), covariancesInit: cov0, tol: 1e-4, maxIter: 200 }).fit(X);
      const its = g.trace.filter((s) => s.kind === 'em'), N = its.length;
      const box = { x: 40, y: 20, w: 430, h: 430 }, extra = reach(means0, cov0, type).concat(...its.map((s) => reach(s.means, s.covariances, type)));
      const ax = P.axes('gm', X, box, { extra, pad: 0.02, xname: d.names[0], yname: d.names[1] });
      const LA = { x: 560, y: 50, w: 210, h: 150 }, lls = its.map((s) => s.ll), llMin = Math.min(...lls), llMax = Math.max(...lls), pad = (llMax - llMin) * 0.08 || 0.1;
      const lax = C().axes(Object.assign({ prefix: 'gl', xr: [1, Math.max(2, N)], yr: [llMin - pad, llMax + pad], xticks: C().ticks(1, Math.max(2, N), 3).filter((v) => v >= 1), yticks: C().ticks(llMin, llMax, 3), fmtY: (v) => H().fx(v, 2), xname: 'iteration', yname: 'mean log-likelihood' }, LA));
      const weightRows = (w) => w.map((v, j) => [...P.row('gw:' + j, 545, 290 + j * 24, j % 6, H().pct(v) + ' of the points')]).flat();
      F.push([...ax.items, ...P.points(ax, 'gm:p', X, null), ...ells(ax, 'gm:e', means0, cov0, type), ...lax.items, H().head('gw:h', 545, 262, 'Weights')].concat(weightRows(new Array(k).fill(1 / k))),
        `${P.qpl(q, k, 'component')}${k === 1 ? ' starts' : ' start'} on randomly picked points, ${k === 1 ? 'with' : 'each with'} a wide round covariance (variance ${q(spread, 3)}) and equal weight. The solid ellipse is one standard deviation, the dashed one two.`, { phase: 'start', k, spread, means: means0, covariances: cov0, type });
      const show = [...new Set([0, 1, 2, 3, 4, 5, 7, 10, 15, 22, 32, 45, 65, 90, 130, N - 1])].filter((t) => t < N);
      show.forEach((t, u) => {
        F.set('iter', t + 1);
        const st = its[t], prevMeans = t === 0 ? means0 : its[t - 1].means, prev = t === 0 ? { means: means0, covariances: cov0, weights: new Array(k).fill(1 / k) } : its[t - 1];
        const moved = st.means.map((m, j) => Math.sqrt(ML().sqdist(m, prevMeans[j]))), far = moved.indexOf(Math.max(...moved));
        // the responsibilities after this M-step: the points' colours
        const R = t + 1 < N ? its[t + 1].resp : g.predictProba(X);
        const items = [...ax.items, ...softPoints(ax, 'gm:p', X, R), ...ells(ax, 'gm:e', st.means, st.covariances, type), ...lax.items, C().line(lax, 'gl:l', lls.slice(0, t + 1).map((v, i) => [i + 1, v]), 'act'), C().dot(lax, 'gl:d', t + 1, st.ll, '', 5), H().head('gw:h', 545, 262, 'Weights')].concat(weightRows(st.weights));
        let say;
        const skipped = () => (u > 0 && t - show[u - 1] > 1 ? `After ${q.n(t - show[u - 1] - 1)} more: ` : '');
        if (t === 0) say = `Iteration ${q.n(1)}. E-step: each point gets a responsibility from each component; with the start’s parameters the mean log-likelihood is ${q(st.ll, 3)}. M-step: each component takes the responsibility-weighted mean and covariance of the points. The ${P.NAMES[far]} mean moves furthest, ${q(moved[far], 3)}.`;
        else if (t === N - 1) say = `Iteration ${q.n(t + 1)}: the log-likelihood rose by less than the tolerance, so EM stops at ${q(st.ll, 4)}${g.converged ? '' : ', the iteration limit'}. Points are coloured by their most responsible component and faded when the responsibility is split.`;
        else say = skipped() + `iteration ${q.n(t + 1)}: mean log-likelihood ${q(st.ll, 3)}` + (t > 0 ? `, up ${q.sg(st.ll - its[t - 1].ll, 2)}` : '') + `. The ${P.NAMES[far]} component moves furthest, ${q(moved[far], 3)}; the weights are ${H().list(st.weights.map((w) => q.pct(w)))}.`;
        if (say[0] === 'i') say = 'I' + say.slice(1);
        F.push(items, say, { phase: 'em', t, iter: t + 1, ll: st.ll, gain: t > 0 ? st.ll - its[t - 1].ll : null, moved, far, weights: st.weights, means: st.means, covariances: st.covariances, skipped: u > 0 ? t - show[u - 1] - 1 : 0, prev: { means: prev.means, covariances: prev.covariances, weights: prev.weights }, type });
      });
      // a point whose responsibility is most split
      const Rf = g.predictProba(X), unsure = Rf.map((r) => Math.max(...r)).reduce((b, v, i, a) => (v < a[b] ? i : b), 0), ru = Rf[unsure].map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]);
      if (k > 1) {
        const items = [...ax.items, ...softPoints(ax, 'gm:p', X, Rf), ...ells(ax, 'gm:e', g.means, g.covariances, type), S.Kit.scatter.ring(ax, 'gm:u', X[unsure][0], X[unsure][1], 'act', 10), ...lax.items, C().line(lax, 'gl:l', lls.map((v, i) => [i + 1, v]), 'act'), H().head('gw:h', 545, 262, 'Point ' + unsure)];
        ru.slice(0, 3).forEach(([v, j], r) => items.push(...P.row('gr:' + r, 545, 290 + r * 24, j % 6, H().pct(v) + ' ' + P.NAMES[j])));
        F.push(items, `Unlike k-means, the answer is soft. The ringed point ${q.n(unsure)} is ${q.pct(ru[0][0])} ${P.NAMES[ru[0][1]]} and ${q.pct(ru[1][0])} ${P.NAMES[ru[1][1]]}: it lies where two ellipses overlap. The fitted mixture has BIC ${q(g.bic(X), 1)}.`, { phase: 'soft', unsure, resp: Rf[unsure], bic: g.bic(X), r0: ru[0][0], r1: ru[1][0], means: g.means, covariances: g.covariances, weights: g.weights, type });
      }
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'EM for a Gaussian mixture', text: ['p(x) = Σⱼ πⱼ N(x | μⱼ, Σⱼ)', 'E: rᵢⱼ = πⱼ N(xᵢ | μⱼ, Σⱼ) / Σₗ πₗ N(xᵢ | μₗ, Σₗ)', 'M: Nⱼ = Σᵢ rᵢⱼ, πⱼ = Nⱼ / n', '   μⱼ = Σᵢ rᵢⱼ xᵢ / Nⱼ', '   Σⱼ = Σᵢ rᵢⱼ (xᵢ − μⱼ)(xᵢ − μⱼ)ᵀ / Nⱼ', 'the log-likelihood never falls'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'em' && f.meta.t === 1 && fr[i + 1] && fr[i + 1].meta.phase === 'em', pick: 'option', ask: 'Can the log-likelihood be lower at a later iteration?', options: ['no: EM never lowers it', 'yes, when a mean moves far', 'yes, when the weights change'], answer: 'no: EM never lowers it', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'A point has responsibilities 0.7 and 0.3. In the M-step it counts…', options: ['as 0.7 of a point for the first component and 0.3 for the second', 'only for the first component', 'for neither until it is sure', 'equally for both'], answer: 0, why: 'Every point contributes to every component in proportion to its responsibility.' },
      { q: 'How does k-means relate to a Gaussian mixture?', options: ['k-means is like EM with hard assignments and equal round covariances', 'they are unrelated', 'k-means is EM with full covariances', 'a mixture cannot have k components'], answer: 0, why: 'Shrink every covariance to the same tiny circle and the responsibilities become 0 or 1: k-means.' },
      { q: 'What do the weights πⱼ add up to?', options: ['1', 'k', 'n', 'it depends on the covariances'], answer: 0, why: 'They are the probabilities of each component.' },
      { q: 'Why might full covariances fit stretched groups better than spherical ones?', options: ['they can tilt and stretch their ellipses', 'they have fewer parameters', 'they converge in one step', 'they ignore the weights'], answer: 0, why: 'A spherical component can only be a circle, so it needs more of them to cover a long, tilted group.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'gmm-bic', family: 'p07-gmm', order: 20, title: 'How many components? BIC',
    blurb: 'More components always raise the likelihood, so the likelihood alone would choose as many as possible. The Bayesian information criterion charges for every parameter: BIC = −2·log L + p·ln n, with p the number of parameters and n the number of points. The model with the lowest BIC balances fit against size. Each covariance shape gives its own curve.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'stretched', options: P.setOptions(['stretched', 'three', 'four', 'touching', 'uneven', 'iris']) },
      { id: 'type', kind: 'choice', label: 'Covariance shown', def: 'full', options: TYPES },
    ],
    presets: [
      { label: 'Stretched groups', values: { set: 'stretched', type: 'full' } },
      { label: 'Stretched groups, circles only', values: { set: 'stretched', type: 'spherical' } },
      { label: 'Four round groups', values: { set: 'four', type: 'spherical' } },
      { label: 'Iris petals', values: { set: 'iris', type: 'full' } },
    ],
    counters: [{ id: 'k', label: 'components tried' }],
    run(inp) {
      const d = P.data(inp.set, {}), X = d.X, n = X.length, F = H().frames(['k']), q = F.q, KS = [1, 2, 3, 4, 5, 6];
      const fits = {};
      for (const t of TYPES) fits[t.value] = KS.map((k) => new (ML().GaussianMixture)({ k, covarianceType: t.value, seed: 0, trace: false }).fit(X));
      const bic = {};
      for (const t of TYPES) bic[t.value] = fits[t.value].map((g) => g.bic(X));
      const type = inp.type, B = bic[type], best = B.indexOf(Math.min(...B));
      const all = TYPES.flatMap((t) => bic[t.value]), bmin = Math.min(...all), bmax = Math.max(...all), pd = (bmax - bmin) * 0.08;
      const box = { x: 40, y: 20, w: 430, h: 430 }, ext = fits[type].flatMap((g) => reach(g.means, g.covariances, type));
      const ax = P.axes('gb', X, box, { extra: ext, pad: 0.02, xname: d.names[0], yname: d.names[1] });
      const BA = { x: 570, y: 50, w: 200, h: 200 };
      const bax = C().axes(Object.assign({ prefix: 'bc', xr: [0.6, 6.4], yr: [bmin - pd, bmax + pd], xticks: KS, yticks: C().ticks(bmin, bmax, 4), xname: 'components', yname: 'BIC (lower is better)' }, BA));
      const curve = (tv, upto, cls) => C().line(bax, 'bc:' + tv, bic[tv].slice(0, upto).map((v, i) => [i + 1, v]), cls);
      KS.forEach((k, i) => {
        F.set('k', k);
        const g = fits[type][i], ll = g.score(X) * n, p = g.nParameters();
        const items = [...ax.items, ...softPoints(ax, 'gb:p', X, g.predictProba(X)), ...ells(ax, 'gb:e', g.means, g.covariances, type), ...bax.items, curve(type, k, 'act'), C().dot(bax, 'bc:d', k, B[i], '', 5)];
        items.push(H().text('gb:r1', 545, 300, 'k = ' + k + ': log L = ' + H().fx(ll, 1), 'strong', 13), H().text('gb:r2', 545, 324, p + ' parameters, ln n = ' + H().fx(Math.log(n), 2), 'legend', 13), H().text('gb:r3', 545, 348, 'BIC = ' + H().fx(B[i], 1), 'legend', 13));
        F.push(items, `k = ${q.n(k)}: the log-likelihood is ${q(ll, 1)} with ${P.qpl(q, p, 'parameter')}, so BIC = −2 × (${q(ll, 1)}) + ${q.n(p)} × ln ${q.n(n)} = ${q(B[i], 1)}.` + (i > 0 ? (B[i] < B[i - 1] ? ' Lower than before: the extra component pays for itself.' : ' Higher than before: the better fit does not pay for the extra parameters.') : ''), { phase: 'k', k, ll, p, n, bic: B[i], means: g.means, covariances: g.covariances, weights: g.weights, type });
      });
      const items = [...ax.items, ...softPoints(ax, 'gb:p', X, fits[type][best].predictProba(X)), ...ells(ax, 'gb:e', fits[type][best].means, fits[type][best].covariances, type), ...bax.items];
      const cls = { full: 'c0', diag: 'c1', spherical: 'c2' };
      for (const t of TYPES) items.push(curve(t.value, 6, cls[t.value]));
      items.push(...C().legend('bc:lg', 545, 300, TYPES.map((t) => ({ text: t.value + ': best k = ' + (bic[t.value].indexOf(Math.min(...bic[t.value])) + 1), cls: cls[t.value] })), { dir: 'col', size: 13 }));
      const bestOf = (tv) => bic[tv].indexOf(Math.min(...bic[tv])) + 1, overall = TYPES.map((t) => [Math.min(...bic[t.value]), t.value]).sort((a, b) => a[0] - b[0])[0];
      F.push(items, `With ${type} covariances the lowest BIC is ${q(B[best], 1)}, at k = ${q.n(best + 1)}, drawn here. The other shapes: full is lowest at k = ${q.n(bestOf('full'))}, diagonal at ${q.n(bestOf('diag'))} and spherical at ${q.n(bestOf('spherical'))}. Overall the best is ${overall[1]} with BIC ${q(overall[0], 1)}.`, { phase: 'best', best: best + 1, bic: B[best], bests: [bestOf('full'), bestOf('diag'), bestOf('spherical')], overall: overall[0], all: bic });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'BIC', text: ['BIC = −2·log L + p·ln n', 'p for k components in d dimensions:', '  full: k·d(d+1)/2 + k·d + k − 1', '  diagonal: k·d + k·d + k − 1', '  spherical: k + k·d + k − 1', 'choose the model with the lowest BIC'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'k' && f.meta.k === 1, pick: 'option', ask: 'At which number of components will this BIC curve be lowest?', options: ['1', '2', '3', '4', '5', '6'], answer: (fr) => String(fr[fr.length - 1].meta.best), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'How many parameters has a full-covariance mixture of 2 components in 2-D?', options: ['11', '10', '12', '6'], answer: 0, why: 'Covariances 2 × 3, means 2 × 2, weights 2 − 1: 6 + 4 + 1.' },
      { q: 'Why not pick k by the highest likelihood?', options: ['the likelihood keeps rising as components are added', 'the likelihood is hard to compute', 'it always picks k = 1', 'it ignores the weights'], answer: 0, why: 'A bigger mixture can always fit at least as well, down to one tiny Gaussian per point.' },
      { q: 'BIC’s penalty for each parameter grows with…', options: ['ln n, the logarithm of the number of points', 'n', 'k²', 'nothing: it is fixed'], answer: 0, why: 'p·ln n: with more data the bar for adding parameters rises slowly.' },
      { q: 'Is the lower BIC or the higher BIC better?', options: ['lower', 'higher', 'closer to zero', 'it depends on k'], answer: 0, why: 'BIC = −2·log L + penalty: a good fit makes −2·log L small.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
