// Part 7 claims: PCA by power iteration on our own covariance matrix, the embeddings and the isolation forest.
import { phase } from './claims-lib-p06-08.mjs';

/** the sample covariance (n − 1), written out */
export function cov(X) {
  const n = X.length, d = X[0].length, m = Array.from({ length: d }, (_, j) => X.reduce((s, x) => s + x[j], 0) / n);
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const x of X) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += ((x[a] - m[a]) * (x[b] - m[b])) / (n - 1);
  return { C, m };
}
/** the top k eigenpairs of a symmetric matrix by power iteration with deflation */
export function topEig(C0, k) {
  const d = C0.length, C = C0.map((r) => r.slice()), out = [];
  for (let t = 0; t < k; t++) {
    let v = Array.from({ length: d }, (_, i) => 1 + ((i * 7919) % 13) / 13), lam = 0;
    for (let it = 0; it < 3000; it++) {
      const w = C.map((r) => r.reduce((s, x, j) => s + x * v[j], 0)), nn = Math.hypot(...w);
      if (nn === 0) break;
      const nv = w.map((x) => x / nn), diff = Math.hypot(...nv.map((x, i) => x - v[i]));
      v = nv; lam = nn;
      if (diff < 1e-13) break;
    }
    out.push({ lam, v });
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] -= lam * v[a] * v[b];
  }
  return out;
}
const sameDir = (u, v) => Math.abs(Math.abs(u.reduce((s, x, i) => s + x * v[i], 0)) - 1);

export function pcaChecks(S) {
  const P = S.P07;
  return {
    'pca-project': {
      check(c) {
        const f0 = phase(c.frames, 'data')[0], n = f0.meta.n;
        const pcs = phase(c.frames, 'pcs')[0];
        const X = P.SETS2[c.input.set].make().X, own = cov(X), C = own.C, [e1, e2] = topEig(C, 2);
        [0, 1].forEach((a) => { c.near(f0.meta.mu[a], own.m[a], 1e-12, 'the mean'); [0, 1].forEach((b) => c.near(f0.meta.cov[a][b], C[a][b], 1e-9, 'the covariance')); });
        c.near(pcs.meta.l1, e1.lam, 1e-8, 'λ₁'); c.near(pcs.meta.l2, e2.lam, 1e-8, 'λ₂');
        c.ok(sameDir(pcs.meta.u1, e1.v) < 1e-8, 'PC1 is the top eigenvector');
        c.ok(sameDir(pcs.meta.u2, e2.v) < 1e-6, 'PC2 is the second eigenvector');
        c.near(pcs.meta.ratio[0], e1.lam / (e1.lam + e2.lam), 1e-9, 'share of PC1');
        for (const f of phase(c.frames, 'dir')) {
          const u = [Math.cos(f.meta.th), Math.sin(f.meta.th)], v = u[0] * u[0] * C[0][0] + 2 * u[0] * u[1] * C[0][1] + u[1] * u[1] * C[1][1];
          c.near(f.meta.v, v, 1e-9, 'variance along a direction is uᵀCu');
          c.ok(f.meta.v <= e1.lam + 1e-9, 'no direction beats PC1');
        }
        const pj = phase(c.frames, 'project')[0];
        c.near(pj.meta.err, (n - 1) * e2.lam, 1e-8, 'the error of keeping PC1 is (n − 1)·λ₂');
      },
    },
    'pca-variance': {
      check(c) {
        const sc = phase(c.frames, 'scree')[0], dim = sc.meta.dim;
        const data = { wine: () => P.std(S.Data.load('wine').X), iris: () => S.Data.load('iris').X, penguins: null }[c.input.set];
        if (!data) { c.ok(Math.abs(sc.meta.cum[dim - 1] - 1) < 1e-9, 'the shares add to 1'); return; }
        const X = data(), { C } = cov(X), eig = topEig(C, dim), tot = eig.reduce((s, e) => s + e.lam, 0);
        eig.slice(0, 3).forEach((e, j) => c.near(sc.meta.r[j], e.lam / tot, 1e-7, 'share of PC' + (j + 1)));
        let run = 0, k90 = 0; for (const e of eig) { run += e.lam / tot; k90++; if (run >= 0.9) break; }
        c.eq(sc.meta.k90, k90, 'components for 90%');
        for (const f of phase(c.frames, 'load')) c.ok(sameDir(f.meta.u, eig[f.meta.j].v) < 1e-6, 'loadings of PC' + (f.meta.j + 1));
      },
    },
    'pca-recon': {
      check(c) {
        const d = S.Data.digits({ n: 150, size: 10, seed: 3 }), X = d.X.map((r) => Array.from(r)), { C, m } = cov(X), eig = topEig(C, 5), tot = C.reduce((s, r, i) => s + r[i], 0);
        const cf = phase(c.frames, 'comps')[0];
        c.near(cf.meta.r[0], eig[0].lam / tot, 1e-6, 'share of PC1');
        for (const f of phase(c.frames, 'rebuild')) {
          if (f.meta.k > 5) continue;
          f.meta.ids.forEach((i, t) => {
            const xc = X[i].map((v, j) => v - m[j]), rec = m.map((mm, j) => mm + eig.slice(0, f.meta.k).reduce((s, e) => s + e.v.reduce((a, w, q) => a + w * xc[q], 0) * e.v[j], 0));
            c.near(f.meta.errs[t], X[i].reduce((s, v, j) => s + (v - rec[j]) ** 2, 0), 1e-6, 'rebuild error, k = ' + f.meta.k);
          });
        }
        const rb = phase(c.frames, 'rebuild');
        for (let t = 1; t < rb.length; t++) rb[t].meta.errs.forEach((e, u) => c.ok(e <= rb[t - 1].meta.errs[u] + 1e-9, 'errors never rise with more components'));
      },
    },
  };
}
const sqd = (a, b) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0);
/** t-SNE's joint probabilities, our own way: bisection on each row's precision until its entropy is ln(perplexity) */
export function tsneP(X, perp) {
  const n = X.length, P = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const d2 = X.map((x) => sqd(X[i], x));
    let lo = 0, hi = Infinity, beta = 1, p = null;
    for (let it = 0; it < 200; it++) {
      p = d2.map((d, j) => (j === i ? 0 : Math.exp(-beta * d)));
      const Z = p.reduce((a, b) => a + b, 0) || 1e-300; p = p.map((v) => v / Z);
      const Hh = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
      if (Math.abs(Hh - Math.log(perp)) < 1e-10) break;
      if (Hh > Math.log(perp)) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
    }
    p.forEach((v, j) => { P[i][j] += v; P[j][i] += v; });
  }
  const s = P.flat().reduce((a, b) => a + b, 0);
  return P.map((r, i) => r.map((v, j) => (i === j ? 0 : Math.max(v / s, 2.220446049250313e-16))));
}
export function kl(P, Y, exag) {
  const n = Y.length; let Z = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Z += 1 / (1 + sqd(Y[i], Y[j]));
  let s = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { const p = P[i][j] * exag, q = Math.max(1 / (1 + sqd(Y[i], Y[j])) / Z, 2.220446049250313e-16); s += p * Math.log(p / q); }
  return s;
}
/** trustworthiness from its definition */
export function trust(X, Y, k) {
  const n = X.length, rank = (A, i) => { const o = [...Array(n).keys()].filter((j) => j !== i).sort((u, v) => sqd(A[i], A[u]) - sqd(A[i], A[v]) || u - v); const r = new Array(n).fill(0); o.forEach((j, t) => (r[j] = t + 1)); return { o, r }; };
  let t = 0;
  for (let i = 0; i < n; i++) { const rx = rank(X, i).r, ny = rank(Y, i).o.slice(0, k); for (const j of ny) t += Math.max(0, rx[j] - k); }
  return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * t;
}
export function embedChecks(S) {
  const P = S.P07;
  return {
    'tsne-run': {
      check(c) {
        const X = P.HIGH[c.input.set].make().X, Pm = tsneP(X, c.input.perp);
        for (const f of c.frames) c.near(f.meta.kl, kl(Pm, f.meta.Y, f.meta.exag), 2e-3, 'KL at iteration ' + f.meta.t);
        const lf = c.frames[c.frames.length - 1];
        c.near(lf.meta.trust, trust(X, lf.meta.Yfinal, 5), 1e-12, 'trustworthiness');
      },
    },
    'umap-run': {
      check(c) {
        const X = P.HIGH[c.input.set].make().X, n = X.length, k = c.input.k, g = c.frames[0].meta, ex = g.ex;
        const nbrs = X.map((a, i) => [...Array(n).keys()].filter((j) => j !== i).map((j) => [Math.sqrt(sqd(a, X[j])), j]).sort((u, v) => u[0] - v[0] || u[1] - v[1]).slice(0, k));
        c.near(g.rho, nbrs[ex][0][0], 1e-12, 'ρ is the nearest distance');
        c.near(nbrs[ex].reduce((s, [d]) => s + Math.exp(-Math.max(0, d - g.rho) / g.sigma), 0), Math.log2(k), 1e-3, 'the weights add to log₂ k');
        const pairs = new Set(); nbrs.forEach((l, i) => l.forEach(([, j]) => pairs.add(Math.min(i, j) + ',' + Math.max(i, j))));
        c.eq(g.edges, pairs.size, 'edges after merging both directions');
        c.eq(g.nb, [...pairs].filter((p) => p.split(',').map(Number).includes(ex)).length, 'edges at the example point');
        for (const f of c.frames.slice(1)) c.near(f.meta.alpha, 1 - (f.meta.ep - 1) / 200, 1e-12, 'step size at epoch ' + f.meta.ep);
      },
    },
  };
}
const cOf = (m) => (m <= 1 ? 0 : m === 2 ? 1 : 2 * (Math.log(m - 1) + 0.5772156649015329) - (2 * (m - 1)) / m);
/** walk a tree from its root: the cuts passed and the leaf's size */
const walk = (node, x) => { let d = 0; while (node.left) { node = x[node.feature] < node.threshold ? node.left : node.right; d++; } return d + cOf(node.size); };
export function forestChecks(S) {
  const P = S.P07, ML = S.ML;
  return {
    'iforest-cuts': {
      random: (r) => ({ set: r() < 0.5 ? 'cloud' : 'two', seed: 1 + Math.floor(r() * 300) }),
      randomRuns: 4,
      check(c) {
        const X = P.OUT[c.input.set].make(2).X, f = new ML.IsolationForest({ nEstimators: 100, seed: c.input.seed }).fit(X), m = Math.min(256, X.length);
        const mean = X.map((x) => f.trees.reduce((s, t) => s + walk(t, x), 0) / f.trees.length);
        const sc = phase(c.frames, 'scores')[0];
        c.eq(sc.meta.odd, mean.indexOf(Math.min(...mean)), 'the most isolated point');
        c.near(sc.meta.hO, mean[sc.meta.odd], 1e-12, 'mean path of the odd point');
        c.near(sc.meta.hN, mean[sc.meta.normal], 1e-12, 'mean path of the crowded point');
        c.near(sc.meta.c, cOf(m), 1e-12, 'c(m)');
        c.near(sc.meta.sO, 2 ** (-sc.meta.hO / cOf(m)), 1e-12, 'score of the odd point');
        c.near(sc.meta.sN, 2 ** (-sc.meta.hN / cOf(m)), 1e-12, 'score of the crowded point');
        for (const who of ['odd', 'normal']) {
          const fr = phase(c.frames, who), x = X[sc.meta[who]];
          let rows = X.map((_, i) => i);
          fr.forEach((g) => { c.eq(g.meta.before, rows.length, who + ' cut ' + g.meta.cut + ': points before'); rows = rows.filter((i) => (X[i][g.meta.feature] < g.meta.threshold) === (x[g.meta.feature] < g.meta.threshold)); c.eq(g.meta.after, rows.length, who + ' cut ' + g.meta.cut + ': points after'); });
          const last = fr[fr.length - 1].meta;
          c.near(last.h, walk(f.trees[0], x), 1e-12, who + ': path length in tree 1');
          c.near(last.h, last.depth + cOf(rows.length), 1e-12, who + ': cuts plus c(points left)');
        }
      },
    },
    'iforest-scores': {
      extra: [{ set: 'cloud', cont: 0.15, trees: 40 }, { set: 'pumps', cont: 0.02, trees: 10 }],
      check(c) {
        const d = P.OUT[c.input.set].make(2), X = d.X, f = new ML.IsolationForest({ nEstimators: c.input.trees, seed: 1 }).fit(X), m = Math.min(256, X.length);
        const s = X.map((x) => 2 ** (-(f.trees.reduce((a, t) => a + walk(t, x), 0) / f.trees.length) / cOf(m)));
        const fl = phase(c.frames, 'flag')[0], sorted = s.slice().sort((a, b) => b - a);
        c.near(fl.meta.thr, sorted[Math.min(X.length - 1, Math.floor(c.input.cont * X.length))] , 1e-12, 'the threshold is the score at the contamination rank');
        const flagged = s.map((v) => v > fl.meta.thr);
        c.eq(fl.meta.nf, flagged.filter(Boolean).length, 'points flagged');
        const tr = phase(c.frames, 'truth')[0], tp = flagged.filter((v, i) => v && d.truth[i]).length, pos = d.truth.filter(Boolean).length;
        c.eq([tr.meta.tp, tr.meta.pos], [tp, pos], 'true positives and odd points');
        const fd = phase(c.frames, 'field')[0];
        c.near(fd.meta.topScore, Math.max(...s), 1e-12, 'the highest score');
      },
    },
  };
}
