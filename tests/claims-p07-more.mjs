// Part 7 claims beyond k-means: hierarchical clustering by brute force (cluster distances from the member points
// every time, no update formula), DBSCAN neighbour counts, EM steps, PCA by power iteration, the t-SNE cost from P
// and the drawn layout, the UMAP graph's own conditions, and isolation paths by walking the trees.
import { phase } from './claims-lib-p06-08.mjs';
import { pcaChecks, embedChecks, forestChecks } from './claims-p07-more2.mjs';

const dist = (a, b) => Math.sqrt(a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0));
/** agglomerative clustering from the definitions: [{A, B, dist}] in merge order (members as point lists) */
export function bruteLinkage(X, link) {
  let cl = X.map((_, i) => [i]);
  const out = [];
  const dAB = (A, B) => {
    if (link === 'ward') { const m = (S) => [0, 1].map((j) => S.reduce((s, i) => s + X[i][j], 0) / S.length); return Math.sqrt((2 * A.length * B.length) / (A.length + B.length)) * dist(m(A), m(B)); }
    const ds = []; for (const i of A) for (const j of B) ds.push(dist(X[i], X[j]));
    return link === 'single' ? Math.min(...ds) : link === 'complete' ? Math.max(...ds) : ds.reduce((a, b) => a + b, 0) / ds.length;
  };
  while (cl.length > 1) {
    let best = Infinity, bi = 0, bj = 1;
    for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) { const d = dAB(cl[i], cl[j]); if (d < best - 1e-12) { best = d; bi = i; bj = j; } }
    out.push({ A: cl[bi], B: cl[bj], dist: best });
    const merged = cl[bi].concat(cl[bj]);
    cl = cl.filter((_, k) => k !== bi && k !== bj).concat([merged]);
  }
  return out;
}
/** flat clusters below a height, from a brute merge list */
export function cutBrute(merges, n, h) {
  const lab = [...Array(n).keys()];
  for (const m of merges) if (m.dist <= h) { const t = lab[m.A[0]]; for (const i of m.B) for (let j = 0; j < n; j++) if (lab[j] === lab[i]) lab[j] = t; }
  return lab;
}
const sizesOf = (lab) => { const c = new Map(); for (const l of lab) c.set(l, (c.get(l) || 0) + 1); return [...c.values()].sort((a, b) => b - a); };

export function moreChecks(S) {
  const P = S.P07;
  return Object.assign({
    'hc-build': {
      random: (r) => ({ set: ['three', 'bridge', 'stretched', 'moons'][Math.floor(r() * 4)], n: 6 + Math.floor(r() * 25), link: ['single', 'complete', 'average', 'ward'][Math.floor(r() * 4)], cut: 0.02 + Math.round(r() * 96) / 100 }),
      check(c) {
        const X = P.data(c.input.set, { n: c.input.n, seed: 2 }).X, n = X.length, br = bruteLinkage(X, c.input.link);
        phase(c.frames, 'merge').forEach((f, t) => {
          c.near(f.meta.dist, br[t].dist, 1e-9, 'merge ' + (t + 1) + ' distance');
          c.eq([f.meta.sa, f.meta.sb].sort((a, b) => a - b), [br[t].A.length, br[t].B.length].sort((a, b) => a - b), 'merge ' + (t + 1) + ' sizes');
          c.eq(f.meta.left, n - t - 1, 'clusters left');
        });
        const cut = phase(c.frames, 'cut')[0], top = br[n - 2].dist;
        c.near(cut.meta.cutH, c.input.cut * top, 1e-12, 'cut height');
        const lab = cutBrute(br, n, cut.meta.cutH);
        c.eq(cut.meta.k, new Set(lab).size, 'clusters below the cut');
        c.eq(cut.meta.sizes, sizesOf(lab), 'cluster sizes below the cut');
      },
    },
    'hc-linkage': {
      random: (r) => ({ set: ['bridge', 'three', 'stretched', 'moons', 'uneven'][Math.floor(r() * 5)], n: 10 + Math.floor(r() * 30), k: 2 + Math.floor(r() * 5) }),
      randomRuns: 4,
      check(c) {
        const X = P.data(c.input.set, { n: c.input.n, seed: 2 }).X, n = X.length, k = c.input.k, labs = {};
        for (const f of phase(c.frames, 'tree')) {
          const br = bruteLinkage(X, f.meta.link);
          c.near(f.meta.top, br[n - 2].dist, 1e-9, f.meta.link + ' top join');
          const h = (br[n - k - 1].dist + br[n - k].dist) / 2, lab = cutBrute(br, n, h);
          c.eq(f.meta.sizes, sizesOf(lab), f.meta.link + ' sizes at k');
          labs[f.meta.link] = lab;
        }
        const agree = (a, b) => { let s = 0, t = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { t++; if ((a[i] === a[j]) === (b[i] === b[j])) s++; } return s / t; };
        const ag = phase(c.frames, 'agree')[0];
        c.near(ag.meta.sw, agree(labs.single, labs.ward), 1e-12, 'single vs Ward');
        c.near(ag.meta.cw, agree(labs.complete, labs.ward), 1e-12, 'complete vs Ward');
        c.near(ag.meta.aw, agree(labs.average, labs.ward), 1e-12, 'average vs Ward');
      },
    },
  }, dbscanChecks(S), gmmChecks(S), pcaChecks(S), embedChecks(S), forestChecks(S));
}

/** DBSCAN from the definition: neighbourhoods by brute force, clusters by breadth-first search from cores */
export function dbscan(X, eps, mp) {
  const n = X.length, N = X.map((a) => X.map((_, j) => j).filter((j) => dist(a, X[j]) <= eps)), core = N.map((l) => l.length >= mp), lab = new Array(n).fill(-1), sizes = [], seeds = [];
  for (let i = 0; i < n; i++) {
    if (!core[i] || lab[i] >= 0) continue;
    const c = sizes.length, queue = [i]; lab[i] = c; let size = 1; seeds.push(i);
    while (queue.length) { const p = queue.shift(); if (!core[p]) continue; for (const j of N[p]) if (lab[j] < 0) { lab[j] = c; size++; queue.push(j); } }
    sizes.push(size);
  }
  const nB = lab.filter((l, i) => l >= 0 && !core[i]).length, nN = lab.filter((l) => l < 0).length;
  return { N, core, lab, sizes, seeds, nCore: core.filter(Boolean).length, nB, nN, k: sizes.length };
}
export function dbscanChecks(S) {
  const P = S.P07;
  return {
    'dbscan-core': {
      random: (r) => ({ set: ['moons', 'noisy', 'circles', 'three', 'uneven'][Math.floor(r() * 5)], eps: 0.1 + Math.round(r() * 28) * 0.05, minPts: 2 + Math.floor(r() * 19) }),
      check(c) {
        const X = P.data(c.input.set, {}).X, R = dbscan(X, c.input.eps, c.input.minPts);
        const cf = phase(c.frames, 'core')[0];
        c.eq(cf.meta.count, R.N[cf.meta.ex].length, 'neighbours of the example');
        c.eq(cf.meta.nCore, R.nCore, 'core points');
        phase(c.frames, 'grow').forEach((f) => { c.eq(f.meta.seed, R.seeds[f.meta.c], 'cluster seed'); c.eq(f.meta.size, R.sizes[f.meta.c], 'cluster size'); });
        const dn = phase(c.frames, 'done')[0];
        c.eq([dn.meta.clusters, dn.meta.nCore, dn.meta.nB, dn.meta.nN], [R.k, R.nCore, R.nB, R.nN], 'clusters, core, border, noise');
      },
    },
    'dbscan-params': {
      random: (r) => ({ set: ['noisy', 'moons', 'circles', 'three', 'uneven'][Math.floor(r() * 5)], minPts: 2 + Math.floor(r() * 19) }),
      randomRuns: 4,
      check(c) {
        const X = P.data(c.input.set, {}).X, n = X.length, mp = c.input.minPts;
        const kd = X.map((a) => X.map((b) => dist(a, b)).sort((u, v) => u - v)[mp - 1]).sort((a, b) => a - b);
        let b = 0, bd = -Infinity; kd.forEach((y, i) => { const d = i / (n - 1) - (y - kd[0]) / (kd[n - 1] - kd[0]); if (d > bd) { bd = d; b = i; } });
        const kf = phase(c.frames, 'kdist')[0];
        c.near(kf.meta.epsS, kd[b], 1e-12, 'ε at the bend');
        for (const f of phase(c.frames, 'eps').concat(phase(c.frames, 'minpts'))) { const R = dbscan(X, f.meta.eps, f.meta.minPts || mp); c.eq([f.meta.clusters, f.meta.noise], [R.k, R.nN], 'clusters and noise at ε ' + f.meta.eps); }
      },
    },
  };
}
const cov2 = (c, t) => (t === 'full' ? c : t === 'diag' ? [[c[0], 0], [0, c[1]]] : [[c, 0], [0, c]]);
/** the density of a 2-D Gaussian, written out */
const gauss = (x, m, S) => { const det = S[0][0] * S[1][1] - S[0][1] * S[1][0], dx = x[0] - m[0], dy = x[1] - m[1], qf = (S[1][1] * dx * dx - (S[0][1] + S[1][0]) * dx * dy + S[0][0] * dy * dy) / det; return Math.exp(-qf / 2) / (2 * Math.PI * Math.sqrt(det)); };
/** one E-step: responsibilities and the mean log-likelihood */
export function eStep(X, p, type) {
  let ll = 0;
  const R = X.map((x) => { const w = p.means.map((m, j) => p.weights[j] * gauss(x, m, cov2(p.covariances[j], type))), s = w.reduce((a, b) => a + b, 0); ll += Math.log(s); return w.map((v) => v / s); });
  return { R, ll: ll / X.length };
}
/** one M-step (the same small ridge on the variances as the engine, 1e-6) */
export function mStep(X, R, type) {
  const n = X.length, K = R[0].length, out = { means: [], covariances: [], weights: [] };
  for (let j = 0; j < K; j++) {
    const N = R.reduce((s, r) => s + r[j], 0), m = [0, 1].map((t) => X.reduce((s, x, i) => s + R[i][j] * x[t], 0) / N);
    const C = [[0, 0], [0, 0]];
    X.forEach((x, i) => { for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) C[a][b] += (R[i][j] * (x[a] - m[a]) * (x[b] - m[b])) / N; });
    out.means.push(m); out.weights.push(N / n);
    out.covariances.push(type === 'full' ? [[C[0][0] + 1e-6, C[0][1]], [C[1][0], C[1][1] + 1e-6]] : type === 'diag' ? [C[0][0] + 1e-6, C[1][1] + 1e-6] : (C[0][0] + C[1][1]) / 2 + 1e-6);
  }
  return out;
}
const nPar = (k, type) => (type === 'full' ? 3 * k : type === 'diag' ? 2 * k : k) + 2 * k + k - 1;
export function gmmChecks(S) {
  const P = S.P07, flat = (v) => [v].flat(3);
  return {
    'gmm-em': {
      random: (r) => ({ set: ['stretched', 'three', 'touching', 'uneven', 'iris'][Math.floor(r() * 5)], k: 1 + Math.floor(r() * 5), type: ['full', 'diag', 'spherical'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 300) }),
      check(c) {
        const X = P.data(c.input.set, {}).X, type = c.input.type;
        const v = [0, 1].map((t) => { const m = X.reduce((s, x) => s + x[t], 0) / X.length; return X.reduce((s, x) => s + (x[t] - m) ** 2, 0) / X.length; });
        c.near(phase(c.frames, 'start')[0].meta.spread, (v[0] + v[1]) / 2, 1e-12, 'the starting variance');
        for (const f of phase(c.frames, 'em')) {
          const e = eStep(X, f.meta.prev, type), m = mStep(X, e.R, type);
          c.near(f.meta.ll, e.ll, 1e-7, 'mean log-likelihood of iteration ' + f.meta.iter);
          flat(m.means).forEach((x, i) => c.near(flat(f.meta.means)[i], x, 1e-6, 'M-step means'));
          flat(m.weights).forEach((x, i) => c.near(flat(f.meta.weights)[i], x, 1e-6, 'M-step weights'));
          flat(m.covariances).forEach((x, i) => c.near(flat(f.meta.covariances)[i], x, 1e-5, 'M-step covariances'));
          f.meta.moved.forEach((d, j) => c.near(d, Math.hypot(f.meta.means[j][0] - f.meta.prev.means[j][0], f.meta.means[j][1] - f.meta.prev.means[j][1]), 1e-12, 'how far a mean moved'));
        }
        const sf = phase(c.frames, 'soft')[0];
        if (sf) {
          const e = eStep(X, sf.meta, type), k = c.input.k;
          e.R[sf.meta.unsure].forEach((r, j) => c.near(sf.meta.resp[j], r, 1e-6, 'responsibilities of the unsure point'));
          const lo = e.R.map((r) => Math.max(...r)); c.near(lo[sf.meta.unsure], Math.min(...lo), 1e-9, 'the most split point');
          c.near(sf.meta.bic, -2 * e.ll * X.length + nPar(k, type) * Math.log(X.length), 1e-6, 'BIC');
        }
      },
    },
    'gmm-bic': {
      extra: [{ set: 'touching', type: 'diag' }, { set: 'uneven', type: 'full' }, { set: 'three', type: 'spherical' }],
      check(c) {
        const X = P.data(c.input.set, {}).X, n = X.length;
        for (const f of phase(c.frames, 'k')) {
          const e = eStep(X, f.meta, f.meta.type);
          c.near(f.meta.ll, e.ll * n, 1e-6, 'log-likelihood at k = ' + f.meta.k);
          c.eq(f.meta.p, nPar(f.meta.k, f.meta.type), 'parameters at k = ' + f.meta.k);
          c.near(f.meta.bic, -2 * e.ll * n + f.meta.p * Math.log(n), 1e-6, 'BIC at k = ' + f.meta.k);
        }
        const b = phase(c.frames, 'best')[0], all = b.meta.all;
        const arg = (a) => a.indexOf(Math.min(...a)) + 1;
        c.eq(b.meta.bests, [arg(all.full), arg(all.diag), arg(all.spherical)], 'best k of each shape');
        c.near(b.meta.overall, Math.min(...all.full, ...all.diag, ...all.spherical), 1e-12, 'overall best BIC');
        c.eq(b.meta.best, arg(all[c.input.type]), 'best k shown');
      },
    },
  };
}
