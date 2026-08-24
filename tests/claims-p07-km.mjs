// Part 7 claims: k-means. Lloyd's steps replayed with our own nearest-centre and mean code, the D² shares of
// k-means++ recomputed from the chosen points, the elbow from our own Lloyd runs and the silhouette from scratch.
import { phase } from './claims-lib-p06-08.mjs';

const sq = (a, b) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0);
export const nearest = (x, C) => { let b = 0; for (let j = 1; j < C.length; j++) if (sq(x, C[j]) < sq(x, C[b])) b = j; return b; };
export const means = (X, lab, C) => C.map((c, j) => { const m = X.filter((_, i) => lab[i] === j); return m.length ? c.map((_, t) => m.reduce((s, x) => s + x[t], 0) / m.length) : c.slice(); });
export const inertia = (X, lab, C) => X.reduce((s, x, i) => s + sq(x, C[lab[i]]), 0);
/** silhouette of every point, from the definition */
export function silhouette(X, lab) {
  const n = X.length, ids = [...new Set(lab)];
  return X.map((x, i) => {
    const avg = (c) => { let s = 0, m = 0; for (let j = 0; j < n; j++) if (j !== i && lab[j] === c) { s += Math.sqrt(sq(x, X[j])); m++; } return m ? s / m : null; };
    const a = avg(lab[i]);
    if (a === null) return 0;
    const b = Math.min(...ids.filter((c) => c !== lab[i]).map(avg).filter((v) => v !== null));
    return (b - a) / Math.max(a, b);
  });
}

export function kmChecks(S) {
  const P = S.P07;
  return {
    'km-steps': {
      random: (r) => ({ set: ['three', 'four', 'touching', 'uneven', 'stretched', 'iris'][Math.floor(r() * 6)], k: 1 + Math.floor(r() * 6), init: ['poor', 'random', 'k-means++'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 300) }),
      check(c) {
        const X = P.data(c.input.set, { seed: c.input.seed }).X, f0 = phase(c.frames, 'init')[0];
        if (c.input.init === 'poor') { const want = X.map((x, i) => [x[0] + x[1], i]).sort((a, b) => a[0] - b[0]).slice(0, c.input.k).map(([, i]) => X[i]); c.eq(f0.meta.centers, want, 'the poor start'); }
        let Cs = f0.meta.centers, prev = null;
        for (const f of c.frames.slice(1)) {
          if (f.meta.phase === 'assign') {
            const lab = X.map((x) => nearest(x, Cs));
            c.near(f.meta.inertia, inertia(X, lab, Cs), 1e-9, 'inertia after assigning');
            c.eq(f.meta.sizes, Cs.map((_, j) => lab.filter((l) => l === j).length), 'cluster sizes');
            c.eq(f.meta.changed, prev ? lab.filter((l, i) => l !== prev[i]).length : X.length, 'points that changed');
            prev = lab;
          } else {
            const NC = means(X, prev, Cs);
            NC.forEach((m, j) => { c.near(f.meta.centers[j][0], m[0], 1e-9, 'a centre moves to the mean'); c.near(f.meta.centers[j][1], m[1], 1e-9, 'a centre moves to the mean'); });
            f.meta.moved.forEach((d, j) => c.near(d, Math.sqrt(sq(NC[j], Cs[j])), 1e-9, 'how far a centre moved'));
            c.near(f.meta.inertia, inertia(X, prev, NC), 1e-9, 'inertia after the move');
            Cs = NC;
          }
        }
      },
    },
    'km-plusplus': {
      random: (r) => ({ set: ['three', 'four', 'uneven', 'touching', 'penguins'][Math.floor(r() * 5)], k: 2 + Math.floor(r() * 5), seed: 1 + Math.floor(r() * 300) }),
      check(c) {
        const X = P.data(c.input.set, { seed: 1 }).X;
        for (const f of phase(c.frames, 'seed')) {
          const Cs = f.meta.idx.map((i) => X[i]), d2 = X.map((x) => Math.min(...Cs.map((cc) => sq(x, cc))));
          const tot = d2.reduce((a, b) => a + b, 0);
          c.near(f.meta.total, tot, 1e-9, 'total D²');
          c.near(f.meta.maxShare, Math.max(...d2) / tot, 1e-9, 'the largest share');
          c.eq(f.meta.far, d2.indexOf(Math.max(...d2)), 'the furthest point');
          c.eq(f.meta.trials, 2 + Math.floor(Math.log(c.input.k)), 'candidates per centre');
          if (f.meta.m > 1) {
            const B = f.meta.idx.slice(0, -1).map((i) => X[i]), e = X.map((x) => Math.min(...B.map((cc) => sq(x, cc)))), te = e.reduce((a, b) => a + b, 0);
            c.near(f.meta.share, e[f.meta.idx[f.meta.m - 1]] / te, 1e-9, 'the chosen point’s share');
          }
        }
        const cmp = phase(c.frames, 'compare')[0], runs = cmp.meta.runs, best = Math.min(...runs.flat());
        c.near(cmp.meta.best, best, 1e-12, 'best inertia');
        [0, 1].forEach((j) => { c.near(cmp.meta.mean[j], runs.reduce((a, r) => a + r[j], 0) / 10, 1e-12, 'mean inertia'); c.eq(cmp.meta.hits[j], runs.filter((r) => r[j] <= best * (1 + 1e-6)).length, 'runs reaching the best'); });
      },
    },
      'km-elbow': {
      extra: [{ set: 'uneven', kmax: 6 }, { set: 'penguins', kmax: 10 }, { set: 'three', kmax: 4 }],
      check(c) {
        const X = P.data(c.input.set, {}).X, n = X.length, mean = [0, 1].map((t) => X.reduce((s, x) => s + x[t], 0) / n);
        const I = c.frames[c.frames.length - 1].meta.I;
        c.near(I[0], X.reduce((s, x) => s + sq(x, mean), 0), 1e-9, 'inertia with one centre is the total spread');
        for (const f of phase(c.frames, 'k')) {
          const lab = X.map((x) => nearest(x, f.meta.centers));
          c.near(f.meta.inertia, inertia(X, lab, f.meta.centers), 1e-9, 'inertia of the drawn centres');
          means(X, lab, f.meta.centers).forEach((m, j) => c.near(Math.hypot(m[0] - f.meta.centers[j][0], m[1] - f.meta.centers[j][1]), 0, 1e-6, 'each centre is the mean of its points'));
          if (f.meta.k > 1) c.near(f.meta.drop, (I[f.meta.k - 2] - I[f.meta.k - 1]) / I[f.meta.k - 2], 1e-12, 'the drop');
        }
        // the elbow: furthest below the chord, by our own arithmetic
        const lo = Math.min(...I), hi = Math.max(...I), K = I.length;
        let best = 1, bd = -Infinity;
        I.forEach((v, i) => { const dd = (1 - i / (K - 1)) - (v - lo) / (hi - lo); if (dd > bd) { bd = dd; best = i + 1; } });
        c.eq(phase(c.frames, 'elbow')[0].meta.elbow, best, 'the elbow');
      },
    },
    'km-silhouette': {
      random: (r) => ({ set: ['three', 'four', 'touching', 'uneven', 'iris', 'penguins'][Math.floor(r() * 6)], k: 2 + Math.floor(r() * 5) }),
      randomRuns: 4,
      check(c) {
        const X = P.data(c.input.set, {}).X, f0 = phase(c.frames, 'clusters')[0], lab = X.map((x) => nearest(x, f0.meta.centers)), s = silhouette(X, lab);
        c.eq(f0.meta.sizes, f0.meta.centers.map((_, j) => lab.filter((l) => l === j).length), 'cluster sizes');
        const d = (i, j) => Math.sqrt(sq(X[i], X[j]));
        for (const f of phase(c.frames, 'typical').concat(phase(c.frames, 'worst'))) {
          const i = f.meta.i, own = X.map((_, j) => j).filter((j) => j !== i && lab[j] === lab[i]);
          c.near(f.meta.a, own.reduce((t, j) => t + d(i, j), 0) / own.length, 1e-9, 'a');
          const other = X.map((_, j) => j).filter((j) => lab[j] === f.meta.nb);
          c.near(f.meta.b, other.reduce((t, j) => t + d(i, j), 0) / other.length, 1e-9, 'b');
          c.near(f.meta.s, s[i], 1e-9, 's');
        }
        c.eq(phase(c.frames, 'worst')[0].meta.i, s.indexOf(Math.min(...s)), 'the worst point');
        const pl = phase(c.frames, 'plot')[0];
        c.near(pl.meta.mean, s.reduce((a, b) => a + b, 0) / s.length, 1e-9, 'mean silhouette');
        c.eq(pl.meta.negs, s.filter((v) => v < 0).length, 'negative silhouettes');
        const ks = phase(c.frames, 'ks')[0];
        ks.meta.ks.forEach((kk, t) => { const l2 = X.map((x) => nearest(x, ks.meta.centersByK[t])), s2 = silhouette(X, l2); c.near(ks.meta.scores[t], s2.reduce((a, b) => a + b, 0) / s2.length, 1e-9, 'mean silhouette at k = ' + kk); });
      },
    },
  };
}
