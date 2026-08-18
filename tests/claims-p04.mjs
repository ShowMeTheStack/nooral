// Part 4, nearest neighbours and probabilistic models: every preset and random inputs of each topic; every number
// in the narration must be in the frame's values (tests/claims-p03-lib.mjs), and those values are worked out again
// here independently: neighbours by brute force, votes and weights by hand, naive Bayes statistics straight from the
// data and the raw messages. The engines themselves are first compared with the scikit-learn references.
// Usage: node tests/claims-p04.mjs [topic id ...]   Prints "claims-p04 N passed, M failed".
import { readFileSync } from 'node:fs';
import { setup, runPart, near } from './claims-p03-lib.mjs';
import { rng as mulberry } from './load.mjs';

const { S, t, rnd } = setup('claims-p04');
const only = process.argv.slice(2);
const pick = (r, a) => a[Math.floor(r() * a.length)];
const between = (r, a, b, step) => Math.round((a + (b - a) * r()) / step) * step;
const ref = (f) => JSON.parse(readFileSync(new URL('./ref/classic/' + f + '.json', import.meta.url), 'utf8'));

// ---- the engines against scikit-learn (the topics rest on them) ----
{
  const D = ref('data'), K = ref('knn'), G = ref('generative');
  const [Xc, yc] = [D.cls3.X, D.cls3.y];
  for (const [key, p] of [['e5u', { k: 5 }], ['e7d', { k: 7, weights: 'distance' }], ['m4u', { k: 4, metric: 'manhattan' }], ['cheb', { k: 5, metric: 'chebyshev' }]]) {
    if (!K[key]) continue;
    const prm = K[key].params || {};
    const m = S.ML.KNeighborsClassifier({ k: prm.n_neighbors || p.k, weights: prm.weights || p.weights || 'uniform', metric: prm.metric || p.metric || 'minkowski', p: prm.p || 2 }).fit(Xc, yc);
    const P = m.predictProba(K.Q);
    t.ok(P.every((r, i) => r.every((v, j) => near(v, K[key].proba[i][j], 1e-9))), 'k-NN ' + key + ' probabilities match scikit-learn');
  }
  const g = S.ML.GaussianNB().fit(D.cls3.X, D.cls3.y);
  t.ok(g.predictProba(D.cls3.X).every((r, i) => r.every((v, j) => near(v, G.cls3.gnb[i][j], 1e-6))), 'GaussianNB matches scikit-learn');
  for (const key of ['mnb_1', 'mnb_0.1']) {
    const m = S.ML.MultinomialNB({ alpha: G[key].alpha }).fit(D.counts.X, D.counts.y);
    t.ok(m.featureLogProb.every((r, k) => r.every((v, j) => near(v, G[key].logp[k][j], 1e-9))), 'MultinomialNB ' + key + ' log-probabilities match scikit-learn');
  }
}

// ---- independent pieces ----
const dist = (a, b, p) => { const d = a.map((v, j) => Math.abs(v - b[j])); return p === 1 ? d.reduce((s, v) => s + v, 0) : p === Infinity ? Math.max(...d) : Math.sqrt(d.reduce((s, v) => s + v * v, 0)); };
/** the k nearest by brute force: [{i, d}] sorted by distance, then index */
const nearest = (X, q, k, p) => X.map((x, i) => ({ i, d: dist(x, q, p || 2) })).sort((a, b) => a.d - b.d || a.i - b.i).slice(0, k);
const vote = (X, y, q, k) => { const nb = nearest(X, q, k), K = Math.max(...y) + 1, c = new Array(K).fill(0); nb.forEach((o) => c[y[o.i]]++); let b = 0; for (let j = 1; j < K; j++) if (c[j] > c[b]) b = j; return b; };
const err = (X, y, Q, yq, k) => Q.filter((q, i) => vote(X, y, q, k) !== yq[i]).length / Q.length;
const queryOf = (S, inp, X) => { const r = S.PL4.square(X); return [r.xr[0] + inp.qx * (r.xr[1] - r.xr[0]), r.yr[0] + inp.qy * (r.yr[1] - r.yr[0])].map((v) => S.Data.round(v, 2)); };
const SETS = ['blobs', 'moons', 'circles', 'flowers', 'iris'];

const mods = {
  'knn-voronoi': {
    random: (r) => ({ set: pick(r, SETS), n: 3 + Math.floor(r() * 38), noise: between(r, 0, 1, 0.05), seed: 1 + Math.floor(r() * 999), qx: between(r, 0.02, 0.98, 0.01), qy: between(r, 0.02, 0.98, 0.01) }),
    check({ r, fail, pass, input }) {
      const d = S.PL4.data(input.set, input.n, input.seed, input.noise), q = queryOf(S, input, d.X), nb = nearest(d.X, q, 2), m = r.frames[0].meta;
      (m.nearest === nb[0].i && near(m.dNear, nb[0].d, 1e-12) ? pass : fail)('nearest point ' + m.nearest + ' at ' + m.dNear + ', brute force ' + nb[0].i + ' at ' + nb[0].d);
      const qf = r.frames.find((f) => f.meta.phase === 'query').meta;
      if (nb[1]) (near(qf.second, nb[1].d, 1e-12) ? pass : fail)('second distance');
      const self = d.X.filter((x, i) => d.y[nearest(d.X, x, 1)[0].i] === d.y[i]).length / d.X.length;
      (near(m.trainAcc, self, 1e-12) ? pass : fail)('training accuracy ' + m.trainAcc + ' vs ' + self);
      (m.nCross <= m.walls ? pass : fail)('walls');
      // the cells: every cell vertex is no nearer any other point than to its own (a sample of the plane agrees)
      const rg = S.PL4.square(d.X), cells = S.PL4.voronoi(d.X, rg.xr, rg.yr);
      let ok = true;
      cells.forEach((poly, i) => poly.forEach((v) => { if (nearest(d.X, v, 1)[0].d < dist(v, d.X[i]) - 1e-7) ok = false; }));
      (ok ? pass : fail)('every Voronoi vertex is nearest its own point');
      const area = cells.reduce((s, poly) => { let a = 0; for (let i = 0; i < poly.length; i++) { const p = poly[i], o = poly[(i + 1) % poly.length]; a += p[0] * o[1] - o[0] * p[1]; } return s + Math.abs(a) / 2; }, 0);
      const dupFree = new Set(d.X.map((x) => x.join(','))).size === d.X.length;
      if (dupFree) (near(area, (rg.xr[1] - rg.xr[0]) * (rg.yr[1] - rg.yr[0]), 1e-7) ? pass : fail)('the cells tile the plot: ' + area);
    },
  },
  'knn-k': {
    nRandom: 3,
    random: (r) => { const n = 20 + Math.floor(r() * 60); const ks = [1, 2 + Math.floor(r() * 5), 9 + Math.floor(r() * 10), n].map(String); return { set: pick(r, SETS), n, noise: between(r, 0, 1, 0.05), seed: 1 + Math.floor(r() * 999), ks }; },
    check({ r, fail, pass, input }) {
      const d = S.PL4.data(input.set, input.n, input.seed, input.noise), te = S.PL4.data(input.set, input.n, input.seed + 1000, input.noise);
      let best = null;
      for (const f of r.frames) {
        const k = f.meta.k, tr = err(d.X, d.y, d.X, d.y, k), ts = err(d.X, d.y, te.X, te.y, k);
        (near(f.meta.trErr, tr, 1e-12) && near(f.meta.teErr, ts, 1e-12) ? pass : fail)('k = ' + k + ': errors ' + f.meta.trErr + ', ' + f.meta.teErr + ' vs brute force ' + tr + ', ' + ts);
        if (!best || ts < best.ts) best = { k, ts };
        (f.meta.bestK === best.k ? pass : fail)('best k so far');
      }
    },
  },
  'knn-metrics': {
    random: (r) => ({ set: pick(r, SETS), n: 5 + Math.floor(r() * 100), noise: between(r, 0, 1, 0.05), seed: 1 + Math.floor(r() * 999), k: 1 + Math.floor(r() * 5), qx: between(r, 0.02, 0.98, 0.01), qy: between(r, 0.02, 0.98, 0.01) }),
    check({ r, fail, pass, input }) {
      const d = S.PL4.data(input.set, input.n, input.seed, input.noise), q = queryOf(S, input, d.X), K = Math.max(...d.y) + 1;
      const sets = [2, 1, Infinity].map((p) => nearest(d.X, q, input.k, p));
      r.frames.filter((f) => f.meta.phase === 'metric').forEach((f) => {
        const nb = sets[f.meta.s], votes = new Array(K).fill(0);
        nb.forEach((o) => votes[d.y[o.i]]++);
        (near(f.meta.r, nb[nb.length - 1].d, 1e-12) ? pass : fail)('metric ' + f.meta.s + ': k-th distance ' + f.meta.r + ' vs ' + nb[nb.length - 1].d);
        (votes.every((v, c) => v === (f.meta.votes[c] || 0)) ? pass : fail)('metric ' + f.meta.s + ' votes');
        if (f.meta.s) { const prev = new Set(sets[f.meta.s - 1].map((o) => o.i)); (f.meta.same === nb.filter((o) => prev.has(o.i)).length ? pass : fail)('shared neighbours'); }
      });
      const all = sets.map((s) => new Set(s.map((o) => o.i))), common = [...all[0]].filter((i) => all[1].has(i) && all[2].has(i)).length;
      (r.frames[r.frames.length - 1].meta.common === common ? pass : fail)('common neighbours');
    },
  },
  'knn-weights': {
    random: (r) => ({ set: pick(r, SETS), n: 5 + Math.floor(r() * 100), noise: between(r, 0, 1, 0.05), seed: 1 + Math.floor(r() * 999), k: 1 + Math.floor(r() * 5), qx: between(r, 0.02, 0.98, 0.01), qy: between(r, 0.02, 0.98, 0.01) }),
    check({ r, fail, pass, input }) {
      const d = S.PL4.data(input.set, input.n, input.seed, input.noise), q = queryOf(S, input, d.X), K = Math.max(...d.y) + 1, k = input.k;
      const wvote = (x, uniform) => { const nb = nearest(d.X, x, k), t = new Array(K).fill(0); const z = nb.some((o) => o.d === 0); nb.forEach((o) => { t[d.y[o.i]] += uniform ? 1 : z ? (o.d === 0 ? 1 : 0) : 1 / o.d; }); return t; };
      const tu = wvote(q, true), td = wvote(q, false), am = (t) => t.indexOf(Math.max(...t));
      const fu = r.frames.find((f) => f.meta.phase === 'uniform').meta, fd = r.frames.find((f) => f.meta.phase === 'distance').meta;
      (tu.every((v, c) => near(v, fu.totU[c])) && near(fu.p, tu[am(tu)] / tu.reduce((a, b) => a + b)) ? pass : fail)('uniform votes');
      (td.every((v, c) => near(v, fd.totD[c], 1e-9)) && near(fd.p, td[am(td)] / td.reduce((a, b) => a + b), 1e-9) ? pass : fail)('weighted votes ' + td + ' vs ' + fd.totD);
      const accOf = (uniform) => d.X.filter((x, i) => am(wvote(x, uniform)) === d.y[i]).length / d.X.length;
      const ft = r.frames.find((f) => f.meta.phase === 'train').meta;
      (near(ft.accU, accOf(true)) && near(ft.accD, accOf(false)) ? pass : fail)('training accuracies');
    },
  },
  'kd-search': {
    random: (r) => ({ n: 4 + Math.floor(r() * 57), leaf: 1 + Math.floor(r() * 8), k: 1 + Math.floor(r() * 4), seed: 1 + Math.floor(r() * 999), qx: between(r, 0, 10, 0.1), qy: between(r, 0, 10, 0.1) }),
    check({ r, fail, pass, input }) {
      const g = mulberry(input.seed), X = Array.from({ length: input.n }, () => [S.Data.round(g() * 10, 2), S.Data.round(g() * 10, 2)]), q = [input.qx, input.qy];
      const k = Math.min(input.k, input.n), nb = nearest(X, q, k), last = r.frames[r.frames.length - 1].meta;
      (nb.every((o, t) => near(o.d, last.found[t], 1e-12)) ? pass : fail)('found distances ' + last.found + ' vs brute force ' + nb.map((o) => o.d));
      (last.same === 1 ? pass : fail)('same as brute force');
      (last.dist <= input.n && near(last.frac, last.dist / input.n) ? pass : fail)('distances counted');
      r.frames.forEach((f, i) => {
        if (f.meta.phase === 'prune') (f.meta.gap > f.meta.kth ? pass : fail)('frame ' + i + ': a skipped box must be farther than the k-th distance');
        if (f.meta.phase === 'descend') ((f.meta.qd === q[0] || f.meta.qd === q[1]) ? pass : fail)('frame ' + i + ': query coordinate');
      });
    },
  },
  'curse-dim': {
    nRandom: 3,
    random: (r) => ({ n: 20 + Math.floor(r() * 300), dims: [1, 3, 30, 300].map(String), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, input }) {
      for (const f of r.frames) {
        const d = f.meta.d, g = mulberry(input.seed * 7919 + d), q = Array.from({ length: d }, () => g());
        const ds = Array.from({ length: input.n }, () => { let s = 0; for (let j = 0; j < d; j++) { const v = g() - q[j]; s += v * v; } return Math.sqrt(s); });
        const mn = Math.min(...ds), mx = Math.max(...ds), mean = ds.reduce((a, b) => a + b) / ds.length, sd = Math.sqrt(ds.reduce((a, v) => a + (v - mean) ** 2, 0) / ds.length);
        (near(f.meta.mn, mn, 1e-9) && near(f.meta.mx, mx, 1e-9) && near(f.meta.mean, mean, 1e-9) && near(f.meta.cv, sd / mean, 1e-7) && near(f.meta.ratio, mn / mx, 1e-9) ? pass : fail)('d = ' + d + ': distance statistics');
      }
    },
  },
  'nb-gauss': {
    random: (r) => ({ set: pick(r, ['flowers', 'iris', 'blobs', 'tilted', 'moons']), n: 12 + Math.floor(r() * 150), seed: 1 + Math.floor(r() * 999), qx: between(r, 0.02, 0.98, 0.01), qy: between(r, 0.02, 0.98, 0.01) }),
    check({ r, fail, pass, input }) {
      const d = input.set === 'tilted' ? S.PL.gauss2(input.n, input.seed, [3.6, 3.4], [6.2, 6.6], [1.3, 1.3, 0.85], [1.3, 1.3, 0.85]) : S.PL4.data(input.set, input.n, input.seed, 0.5);
      const X = d.X, y = d.y, n = X.length, K = Math.max(...y) + 1;
      const colVar = (rows, j) => { const m = rows.reduce((s, x) => s + x[j], 0) / rows.length; return rows.reduce((s, x) => s + (x[j] - m) ** 2, 0) / rows.length; };
      const eps = 1e-9 * Math.max(colVar(X, 0), colVar(X, 1));
      const st = Array.from({ length: K }, (_, k) => { const g = X.filter((x, i) => y[i] === k); return { prior: g.length / n, mean: [0, 1].map((j) => g.reduce((s, x) => s + x[j], 0) / g.length), var: [0, 1].map((j) => colVar(g, j) + eps) }; });
      r.frames.filter((f) => f.meta.phase === 'class').forEach((f) => { const o = st[f.meta.k]; (near(f.meta.prior, o.prior) && f.meta.mean.every((v, j) => near(v, o.mean[j], 1e-9)) && f.meta.var.every((v, j) => near(v, o.var[j], 1e-7)) ? pass : fail)('class ' + f.meta.k + ' statistics'); });
      const pm = r.frames[r.frames.length - 1].meta, q = pm.q;
      const joint = st.map((o) => Math.log(o.prior) + [0, 1].reduce((s, j) => s - 0.5 * Math.log(2 * Math.PI * o.var[j]) - (q[j] - o.mean[j]) ** 2 / (2 * o.var[j]), 0));
      const mx = Math.max(...joint), z = joint.reduce((s, v) => s + Math.exp(v - mx), 0), post = joint.map((v) => Math.exp(v - mx) / z);
      (joint.every((v, k) => near(v, pm.joint[k], 1e-7)) ? pass : fail)('joint log-likelihoods ' + pm.joint + ' vs ' + joint);
      (post.every((v, k) => near(v, pm.post[k], 1e-7)) ? pass : fail)('posteriors');
      (pm.terms.every((t, k) => near(pm.prior[k] + t[0] + t[1], joint[k], 1e-7)) ? pass : fail)('the terms add up');
    },
  },
};

// ---- the spam filter, from the raw messages ----
const toks = (s) => String(s).toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g) || [];
function nbIndependent(text, alpha) {
  const rows = S.Data.textSets.spam.filter((r) => r[0] !== text.trim());
  const cnt = [new Map(), new Map()], N = [0, 0], docs = [0, 0], vocab = new Set();
  for (const [t, c] of rows) { docs[c]++; for (const w of toks(t)) { cnt[c].set(w, (cnt[c].get(w) || 0) + 1); N[c]++; vocab.add(w); } }
  const V = vocab.size, p = (w, c) => ((cnt[c].get(w) || 0) + alpha) / (N[c] + alpha * V);
  const lo0 = Math.log(docs[1] / docs[0]);
  let run = lo0;
  const steps = toks(text).filter((w) => vocab.has(w)).map((w) => { const d = Math.log(p(w, 1)) - Math.log(p(w, 0)); run += d; return { w, c: [cnt[0].get(w) || 0, cnt[1].get(w) || 0], p: [p(w, 0), p(w, 1)], d, run }; });
  return { V, N, docs, lo0, steps, final: run, post: 1 / (1 + Math.exp(-run)), p };
}
const MSGS = ['Are you free for lunch? Reply today', 'Win cash now', 'See you at the station tomorrow', 'Your prize is waiting, claim it today', 'Can you call me back', 'Free free free', 'The train is late again, I will be there in twenty minutes.', 'zebra free'];
mods['nb-spam'] = {
  random: (r) => ({ text: pick(r, MSGS), alpha: between(r, 0.1, 3, 0.1) }),
  check({ r, fail, pass, input }) {
    const ref = nbIndependent(input.text, input.alpha), f0 = r.frames[0].meta;
    (f0.V === ref.V && f0.N[0] === ref.N[0] && f0.N[1] === ref.N[1] && near(f0.lo0, ref.lo0) ? pass : fail)('counts: V ' + f0.V + ' vs ' + ref.V);
    const ws = r.frames.filter((f) => f.meta.phase === 'word');
    (ws.length === ref.steps.length ? pass : fail)('words scored');
    ws.forEach((f, i) => { const s = ref.steps[i]; (s && f.meta.c[0] === s.c[0] && f.meta.c[1] === s.c[1] && near(f.meta.p[0], s.p[0], 1e-9) && near(f.meta.p[1], s.p[1], 1e-9) && near(f.meta.d, s.d, 1e-9) && near(f.meta.run, s.run, 1e-9) ? pass : fail)('word ' + i); });
    const v = r.frames[r.frames.length - 1].meta;
    (near(v.final, ref.final, 1e-9) && near(v.post, ref.post, 1e-9) ? pass : fail)('posterior ' + v.post + ' vs ' + ref.post);
  },
};
mods['nb-smoothing'] = {
  random: (r) => ({ text: pick(r, MSGS.filter((m) => !/zebra/.test(m))), alphas: ['0', String(between(r, 0.01, 1, 0.01)), String(between(r, 1, 20, 1))] }),
  check({ r, fail, pass, input }) {
    r.frames.filter((f) => f.meta.phase === 'alpha').forEach((f) => {
      const a = f.meta.a === 0 ? 1e-10 : f.meta.a, ref = nbIndependent(input.text, a);
      (near(f.meta.post, ref.post, 1e-7) ? pass : fail)('α = ' + f.meta.a + ': P(spam) ' + f.meta.post + ' vs ' + ref.post);
      const zs = ref.steps.filter((s) => s.c[1] === 0).map((s) => s.w), zh = ref.steps.filter((s) => s.c[0] === 0).map((s) => s.w);
      (f.meta.zs === new Set(zs).size && f.meta.zh === new Set(zh).size ? pass : fail)('unseen words');
      const crit = zs[0] || ref.steps.find((s) => s.c[0] === 0 || s.c[1] === 0)?.w;
      if (crit) { const c = zs.length ? 1 : (ref.steps.find((s) => s.w === crit).c[1] === 0 ? 1 : 0); (near(f.meta.lnPz, Math.log(ref.p(crit, c)), 1e-9) ? pass : fail)('ln P of the unseen word'); }
    });
  },
};

const res = runPart(S, t, rnd, /^p04-/, mods, only);
t.ok(res.runs > 0, 'some runs');
t.done();
