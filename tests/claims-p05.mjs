// Part 5, trees and ensembles: every preset and random inputs of each topic; every number in the narration must be in
// the frame's values (tests/claims-p03-lib.mjs), and those values are worked out again here apart from the topic:
// impurities of every candidate threshold from class counts, leaf means, the weakest-link pruning formula, bootstrap
// and out-of-bag rows from the draws, AdaBoost's error, say and reweighting, gradient boosting's residuals, and
// accuracies by walking the trees by hand. Usage: node tests/claims-p05.mjs [topic id ...]
import { setup, runPart } from './claims-p03-lib.mjs';

const { S, t, rnd } = setup('claims-p05');
const only = process.argv.slice(2);
const pick = (r, a) => a[Math.floor(r() * a.length)];
const int = (r, a, b) => a + Math.floor(r() * (b - a + 1));
const step = (r, a, b, s) => Math.round((a + (b - a) * r()) / s) * s;
const SETS = ['moons', 'circles', 'xor', 'blobs', 'flowers', 'exam'];

// ---- independent pieces ----
const counts = (ys, K) => { const c = new Array(K).fill(0); for (const v of ys) c[v]++; return c; };
function imp(crit, c) {
  const W = c.reduce((a, b) => a + b, 0);
  if (!W) return 0;
  if (crit === 'entropy') return -c.reduce((s, v) => (v ? s + (v / W) * Math.log2(v / W) : s), 0);
  return 1 - c.reduce((s, v) => s + (v / W) ** 2, 0);
}
const mse = (ys) => { if (!ys.length) return 0; const m = ys.reduce((a, b) => a + b, 0) / ys.length; return ys.reduce((s, v) => s + (v - m) ** 2, 0) / ys.length; };
/** every midpoint cut of one feature over some rows, with the weighted child score */
function cuts(X, rows, f, score, minLeaf = 1) {
  const s = rows.slice().sort((a, b) => X[a][f] - X[b][f]), out = [];
  for (let p = 1; p < s.length; p++) {
    if (X[s[p]][f] <= X[s[p - 1]][f] + 1e-7) continue;
    if (p < minLeaf || s.length - p < minLeaf) continue;
    const L = s.slice(0, p), R = s.slice(p);
    out.push({ t: (X[s[p - 1]][f] + X[s[p]][f]) / 2, g: (L.length * score(L) + R.length * score(R)) / s.length });
  }
  return out;
}
/** the rows that reach a node, by walking the tree's thresholds from the root */
function rowsAt(T, X, id) {
  return X.map((_, i) => i).filter((i) => { let n = T.nodes[0]; for (;;) { if (n.id === id) return true; if (n.left < 0) return false; n = T.nodes[X[i][n.feature] <= n.threshold ? n.left : n.right]; } });
}
const walk = (T, x) => { let n = T.nodes[0]; while (n.left >= 0) n = T.nodes[x[n.feature] <= n.threshold ? n.left : n.right]; return n; };
const argmax = (a) => a.indexOf(Math.max(...a));

const mods = {
  'cart-grow': {
    random: (r) => ({ set: pick(r, SETS), n: int(r, 12, 120), depth: int(r, 1, 3), criterion: pick(r, ['gini', 'entropy']), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.PL5.data2(input.set, input.n, input.seed), X = d.X, y = d.y, K = d.K;
      const T = S.ML.DecisionTreeClassifier({ criterion: input.criterion, maxDepth: input.depth, seed: 0 }).fit(X, y).tree;
      const score = (rows) => imp(input.criterion, counts(rows.map((i) => y[i]), K));
      for (const f of r.frames) {
        const m = f.meta;
        if (m.phase === 'cands') {
          const rows = rowsAt(T, X, m.node);
          (rows.length === m.nd ? pass : fail)('node ' + m.node + ' rows');
          (near(score(rows), m.imp, 1e-9) ? pass : fail)('node ' + m.node + ' impurity');
          const all = [0, 1].map((ft) => cuts(X, rows, ft, score));
          ((all[0].length + all[1].length) === m.tried ? pass : fail)('node ' + m.node + ': ' + m.tried + ' cuts, by hand ' + (all[0].length + all[1].length));
          const best = Math.min(...all[m.bestF].map((c) => c.g));
          (near(best, m.best.g, 1e-9) ? pass : fail)('node ' + m.node + ' best score ' + m.best.g + ' by hand ' + best);
          (all[m.bestF].some((c) => near(c.t, m.best.t, 1e-9) && near(c.g, best, 1e-9)) ? pass : fail)('node ' + m.node + ' best threshold');
          if (m.other) (near(Math.min(...all[1 - m.bestF].map((c) => c.g)), m.other.g, 1e-9) ? pass : fail)('other feature best');
          if (m.other) (m.other.g >= best - 1e-12 ? pass : fail)('the chosen feature is the best');
        }
        if (m.phase === 'split') {
          const node = T.nodes[m.node], L = rowsAt(T, X, node.left), R = rowsAt(T, X, node.right);
          (JSON.stringify(counts(L.map((i) => y[i]), K)) === JSON.stringify(m.L) && JSON.stringify(counts(R.map((i) => y[i]), K)) === JSON.stringify(m.R) ? pass : fail)('child counts');
          (near(score(L), m.il) && near(score(R), m.ir) && near((L.length * score(L) + R.length * score(R)) / (L.length + R.length), m.child) ? pass : fail)('child impurities');
        }
        if (m.phase === 'done') {
          const right = X.filter((x, i) => argmax(walk(T, x).value) === y[i]).length;
          (right === m.right ? pass : fail)('accuracy ' + m.right + ' by hand ' + right);
        }
      }
    },
  },
  'reg-tree': {
    random: (r) => ({ shape: pick(r, ['sine', 'step', 'poly', 'abs', 'linear']), n: int(r, 8, 100), noise: step(r, 0, 1, 0.05), depth: int(r, 1, 3), minLeaf: int(r, 1, 8), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.Data.curve({ n: input.n, shape: input.shape, noise: input.noise, seed: input.seed }), X = d.X, y = d.y;
      const T = S.ML.DecisionTreeRegressor({ maxDepth: input.depth, minSamplesLeaf: input.minLeaf, seed: 0 }).fit(X, y).tree;
      const score = (rows) => mse(rows.map((i) => y[i]));
      const mean = (rows) => rows.reduce((s, i) => s + y[i], 0) / rows.length;
      for (const f of r.frames) {
        const m = f.meta;
        if (m.phase === 'root') (near(m.mean, mean(X.map((_, i) => i))) && near(m.mse, mse(y)) ? pass : fail)('root mean and error');
        if (m.phase === 'cands') {
          const rows = rowsAt(T, X, m.node), cs = cuts(X, rows, 0, score, input.minLeaf), best = Math.min(...cs.map((c) => c.g));
          (cs.length === m.tried ? pass : fail)('cuts tried ' + m.tried + ' by hand ' + cs.length);
          (near(best, m.g, 1e-7) ? pass : fail)('best cut score ' + m.g + ' by hand ' + best);
          (near(mean(rows), m.mean) ? pass : fail)('node mean');
        }
        if (m.phase === 'split') {
          const node = T.nodes[m.node], L = rowsAt(T, X, node.left), R = rowsAt(T, X, node.right), all = rowsAt(T, X, m.node);
          (near(mean(L), m.ml) && near(mean(R), m.mr) && L.length === m.nl ? pass : fail)('child means');
          (near(score(all) * all.length, m.before, 1e-7) && near(score(L) * L.length + score(R) * R.length, m.after, 1e-7) ? pass : fail)('sums of squares');
        }
        if (m.phase === 'done') {
          const p = X.map((x) => walk(T, x).value);
          (near(m.mse, y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0) / y.length) ? pass : fail)('final error');
        }
      }
    },
  },
  'tree-size': {
    random: (r) => ({ set: pick(r, SETS), knob: pick(r, ['depth', 'leaf']), n: int(r, 30, 160), noise: step(r, 0.05, 0.6, 0.05), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const tr = S.PL5.data2(input.set, input.n, input.seed, input.noise), te = S.PL5.data2(input.set, 300, input.seed + 1000, input.noise);
      const vals = input.knob === 'depth' ? [1, 2, 3, 4, 5, 6, 8, 12] : [40, 25, 15, 10, 6, 3, 2, 1];
      let best = -1;
      r.frames.forEach((f, k) => {
        const m = f.meta, T = S.ML.DecisionTreeClassifier(input.knob === 'depth' ? { maxDepth: vals[k], seed: 0 } : { minSamplesLeaf: vals[k], seed: 0 }).fit(tr.X, tr.y).tree;
        const acc = (d) => d.X.filter((x, i) => argmax(walk(T, x).value) === d.y[i]).length / d.X.length;
        const leaves = T.nodes.filter((n) => n.left < 0);
        (near(m.train, acc(tr)) && near(m.test, acc(te)) ? pass : fail)('accuracies at ' + vals[k]);
        (m.leaves === leaves.length && m.depth === Math.max(...T.nodes.map((n) => n.depth)) ? pass : fail)('leaves and depth');
        if (input.knob === 'depth') (m.depth <= vals[k] ? pass : fail)('depth within the limit');
        else (leaves.every((n) => n.n >= vals[k]) ? pass : fail)('every leaf holds at least ' + vals[k]);
        best = Math.max(best, acc(te));
      });
      (near(r.frames[r.frames.length - 1].meta.best, best) ? pass : fail)('best test accuracy');
    },
  },
  'ccp-prune': {
    random: (r) => ({ set: pick(r, SETS), n: int(r, 16, 60), noise: step(r, 0.05, 0.6, 0.05), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const tr = S.PL5.data2(input.set, input.n, input.seed, input.noise), te = S.PL5.data2(input.set, 300, input.seed + 1000, input.noise);
      const T = S.ML.DecisionTreeClassifier({ seed: 0 }).fit(tr.X, tr.y).tree, N = T.nodes, W0 = N[0].n, K = tr.K;
      // node impurities again from the rows that reach each node
      const R = N.map((n) => { const rows = rowsAt(T, tr.X, n.id); return (rows.length / W0) * imp('gini', counts(rows.map((i) => tr.y[i]), K)); });
      const cut = new Set();
      const isLeaf = (i) => N[i].left < 0 || cut.has(i);
      const alive = (i) => { for (let p = N[i].parent; p >= 0; p = N[p].parent) if (cut.has(p)) return false; return true; };
      const leavesOf = (i) => (isLeaf(i) ? [i] : [...leavesOf(N[i].left), ...leavesOf(N[i].right)]);
      r.frames.forEach((f) => {
        const m = f.meta, lv = leavesOf(0);
        (lv.length === m.leaves ? pass : fail)('leaf count');
        (near(lv.reduce((s, i) => s + R[i], 0), m.Rt, 1e-9) ? pass : fail)('total leaf impurity ' + m.Rt);
        const acc = te.X.filter((x, i) => { let n = N[0]; while (!isLeaf(n.id)) n = N[x[n.feature] <= n.threshold ? n.left : n.right]; return argmax(n.value) === te.y[i]; }).length / 300;
        (near(acc, m.acc) ? pass : fail)('test accuracy');
        if (m.node !== undefined) {
          let gmin = Infinity;
          for (const n of N) if (!isLeaf(n.id) && alive(n.id)) { const l = leavesOf(n.id); gmin = Math.min(gmin, (R[n.id] - l.reduce((s, i) => s + R[i], 0)) / (l.length - 1)); }
          (near(m.g, gmin, 1e-9) ? pass : fail)('weakest link g ' + m.g + ' by hand ' + gmin);
          (near(Math.max(0, m.g), m.nextAlpha, 1e-9) ? pass : fail)('g is the next alpha');
          cut.add(m.node);
        }
      });
    },
  },
  'bagging': {
    random: (r) => ({ set: pick(r, SETS), n: int(r, 20, 120), B: int(r, 2, 9), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.PL5.data2(input.set, input.n, input.seed), te = S.PL5.data2(input.set, 300, input.seed + 1000), n = d.X.length;
      // the draws again, from the same generator
      const rng = S.ML.rng(input.seed), draws = [], seeds = [];
      for (let b = 0; b < input.B; b++) { draws.push(Array.from({ length: n }, () => rng.int(n))); seeds.push(1 + rng.int(0x7ffffffe)); }
      const m = S.ML.RandomForestClassifier({ nEstimators: input.B, maxFeatures: null, bootstrapIndices: draws, treeSeeds: seeds }).fit(d.X, d.y);
      const vote = (b, x) => { const s = new Array(d.K).fill(0); for (let t = 0; t < b; t++) walk(m.trees[t], x).value.forEach((q, k) => (s[k] += q)); return s; };
      for (const f of r.frames) {
        const mm = f.meta;
        if (mm.phase !== 'tree') continue;
        const b = mm.b - 1, cnt = new Array(n).fill(0);
        draws[b].forEach((i) => cnt[i]++);
        (cnt.filter((c) => c > 0).length === mm.distinct && cnt.filter((c) => c > 1).length === mm.twice && cnt.filter((c) => c === 0).length === mm.oob ? pass : fail)('bootstrap counts of tree ' + mm.b);
        const ta = te.X.filter((x, i) => argmax(walk(m.trees[b], x).value) === te.y[i]).length / 300, ba = te.X.filter((x, i) => argmax(vote(mm.b, x)) === te.y[i]).length / 300;
        (near(ta, mm.treeAcc) && near(ba, mm.bagAcc) ? pass : fail)('accuracies of tree ' + mm.b);
        (m.trees[b].nodes.filter((nd) => nd.left < 0).length === mm.leaves ? pass : fail)('leaves');
      }
    },
  },
  'rf-oob': {
    random: (r) => ({ set: pick(r, SETS), n: int(r, 16, 60), B: int(r, 2, 12), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.PL5.data2(input.set, input.n, input.seed), te = S.PL5.data2(input.set, 300, input.seed + 1000), n = d.X.length;
      const rng = S.ML.rng(input.seed), draws = [], seeds = [];
      for (let b = 0; b < input.B; b++) { draws.push(Array.from({ length: n }, () => rng.int(n))); seeds.push(1 + rng.int(0x7ffffffe)); }
      const m = S.ML.RandomForestClassifier({ nEstimators: input.B, maxFeatures: 'sqrt', oobScore: true, bootstrapIndices: draws, treeSeeds: seeds }).fit(d.X, d.y);
      const votes = d.X.map(() => new Array(d.K).fill(0)), nv = new Array(n).fill(0);
      r.frames.slice(1).forEach((f, b) => {
        const mm = f.meta, inBag = new Set(draws[b]);
        (n - inBag.size === mm.oob ? pass : fail)('out-of-bag rows of tree ' + (b + 1));
        for (let i = 0; i < n; i++) if (!inBag.has(i)) { walk(m.trees[b], d.X[i]).value.forEach((q, k) => (votes[i][k] += q)); nv[i]++; }
        const voted = nv.filter((v) => v).length, right = d.X.filter((_, i) => nv[i] && argmax(votes[i]) === d.y[i]).length;
        (voted === mm.voted && right === mm.right ? pass : fail)('votes after tree ' + (b + 1));
        if (mm.oobScore !== undefined) (voted === n && near(mm.oobScore, right / n) ? pass : fail)('the engine OOB score ' + mm.oobScore + ' by hand ' + right / n);
      });
    },
  },
  'rf-features': {
    random: (r) => { const table = pick(r, ['wildflowers', 'exam', 'loans']), d = { wildflowers: 4, exam: 3, loans: 6 }[table]; return { table, mf: int(r, 1, d), n: int(r, 40, 300), depth: int(r, 2, 3), trees: int(r, 5, 40), seed: int(r, 1, 999) }; },
    check({ r, input, fail, pass, near }) {
      const t = S.PL5.table(input.table, input.n, input.seed), d = t.names.length, n = t.X.length;
      const rng = S.ML.rng(input.seed), draws = [], seeds = [];
      for (let b = 0; b < input.trees; b++) { draws.push(Array.from({ length: n }, () => rng.int(n))); seeds.push(1 + rng.int(0x7ffffffe)); }
      const mfa = input.mf >= d ? null : input.mf;
      const rf = S.ML.RandomForestClassifier({ nEstimators: input.trees, maxFeatures: mfa, maxDepth: input.depth, bootstrapIndices: draws, treeSeeds: seeds }).fit(t.X, t.y);
      const T = rf.trees[0], w = new Array(n).fill(0);
      draws[0].forEach((i) => w[i]++);
      const wscore = (rows) => { const c = new Array(t.K).fill(0); let W = 0; rows.forEach((i) => { c[t.y[i]] += w[i]; W += w[i]; }); return { g: imp('gini', c), W }; };
      for (const f of r.frames) {
        const m = f.meta;
        if (m.phase !== 'split') continue;
        const node = T.nodes[m.node];
        (node && node.feature === m.feature && near(node.threshold, m.t) ? pass : fail)('node ' + m.node + ' is the forest’s first tree');
        (m.seen.length <= Math.max(input.mf, 1) || m.seen.length <= d ? pass : fail)('features seen');
        const rows = rowsAt(T, t.X, m.node).filter((i) => w[i] > 0), W = rows.reduce((s, i) => s + w[i], 0);
        m.seen.forEach((j, k) => {
          const s = rows.slice().sort((a, b) => t.X[a][j] - t.X[b][j]);
          let best = Infinity;
          for (let p = 1; p < s.length; p++) {
            if (t.X[s[p]][j] <= t.X[s[p - 1]][j] + 1e-7) continue;
            const L = wscore(s.slice(0, p)), R = wscore(s.slice(p));
            best = Math.min(best, (L.W * L.g + R.W * R.g) / W);
          }
          (near(best, m.scores[k], 1e-9) ? pass : fail)('node ' + m.node + ' feature ' + j + ': ' + m.scores[k] + ' by hand ' + best);
        });
        (m.scores[m.seen.indexOf(m.feature)] <= Math.min(...m.scores) + 1e-12 ? pass : fail)('the split uses the best feature it saw');
        (m.hidden === d - m.seen.length ? pass : fail)('hidden features');
      }
      const L = r.frames[r.frames.length - 1].meta, roots = new Array(d).fill(0);
      rf.trees.forEach((tt) => { if (tt.nodes[0].left >= 0) roots[tt.nodes[0].feature]++; });
      (JSON.stringify(roots) === JSON.stringify(L.rootsRf) ? pass : fail)('root features');
    },
  },
  'rf-importance': {
    random: (r) => ({ table: pick(r, ['wildflowers', 'exam', 'loans']), extra: pick(r, ['noise', 'copy', 'both', 'none']), n: int(r, 60, 300), trees: int(r, 5, 30), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const t = S.PL5.withExtra(S.PL5.table(input.table, input.n, input.seed), input.extra, input.seed + 77), te = S.PL5.withExtra(S.PL5.table(input.table, 300, input.seed + 1000), input.extra, input.seed + 1077);
      const m = S.ML.RandomForestClassifier({ nEstimators: input.trees, maxFeatures: 'sqrt', seed: input.seed }).fit(t.X, t.y), d = t.names.length;
      // impurity importance from the trees' nodes
      const tot = new Array(d).fill(0);
      for (const T of m.trees) {
        const v = new Array(d).fill(0);
        for (const nd of T.nodes) if (nd.left >= 0) { const L = T.nodes[nd.left], R = T.nodes[nd.right]; v[nd.feature] += nd.w * nd.impurity - L.w * L.impurity - R.w * R.impurity; }
        const s = v.reduce((a, b) => a + b, 0);
        v.forEach((x, j) => (tot[j] += s > 0 ? x / s / m.trees.length : 0));
      }
      const st = tot.reduce((a, b) => a + b, 0), impHand = tot.map((x) => x / st);
      (impHand.every((v, j) => near(v, r.frames[0].meta.imp[j], 1e-9)) ? pass : fail)('impurity importance by hand');
      const acc = (X) => te.y.filter((v, i) => m.predict([X[i]])[0] === v).length / te.y.length;
      const base = acc(te.X);
      (near(base, r.frames[1].meta.baseline) ? pass : fail)('baseline');
      const sh = S.PL5.firstShuffles(te.X, 5, input.seed);
      for (const f of r.frames) {
        if (f.meta.phase !== 'perm') continue;
        const j = f.meta.feature, X = te.X.map((x, i) => { const z = x.slice(); z[j] = sh[j][i]; return z; });
        (near(acc(X), f.meta.first, 1e-12) ? pass : fail)('first shuffle of ' + t.names[j] + ' drawn as the engine draws it');
      }
    },
  },
  'adaboost': {
    random: (r) => ({ set: pick(r, ['moons', 'circles', 'xor', 'exam']), n: int(r, 16, 90), rounds: int(r, 2, 14), depth: int(r, 1, 2), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.PL5.data2(input.set, input.n, input.seed), n = d.X.length;
      const m = S.ML.AdaBoostClassifier({ nEstimators: input.rounds, maxDepth: input.depth, seed: input.seed }).fit(d.X, d.y);
      const M = m.trees.length, stop = m.trace.some((st) => st.kind === 'stop');
      let w = new Array(n).fill(1 / n);
      const alphas = [];
      r.frames.filter((f) => f.meta.phase === 'round').forEach((f, k) => {
        const pred = d.X.map((x) => argmax(walk(m.trees[k], x).value)), wrong = pred.map((c, i) => c !== d.y[i]);
        const W = w.reduce((a, b) => a + b, 0), err = w.reduce((s, v, i) => s + (wrong[i] ? v : 0), 0) / W;
        (near(err, f.meta.err, 1e-9) && wrong.filter(Boolean).length === f.meta.wrongN ? pass : fail)('round ' + (k + 1) + ' error ' + f.meta.err + ' by hand ' + err);
        const alpha = err > 0 ? Math.log((1 - err) / err) : f.meta.alpha;
        (near(alpha, f.meta.alpha, 1e-9) ? pass : fail)('round ' + (k + 1) + ' alpha');
        alphas.push(alpha);
        const vote = d.X.map((x) => alphas.reduce((s, a, j) => s + a * (argmax(walk(m.trees[j], x).value) === 1 ? 1 : -1), 0));
        const acc = d.X.filter((x, i) => (vote[i] > 0 ? 1 : 0) === d.y[i]).length / n;
        (near(acc, f.meta.acc) ? pass : fail)('vote accuracy after round ' + (k + 1));
        if (err > 0 && (k < M - 1 || stop)) { w = w.map((v, i) => (wrong[i] ? v * Math.exp(alpha) : v)); const s2 = w.reduce((a, b) => a + b, 0); w = w.map((v) => v / s2); }
      });
      (near(Math.max(...w), r.frames[r.frames.length - 1].meta.heavy, 1e-9) ? pass : fail)('heaviest final weight');
    },
  },
  'gb-residuals': {
    random: (r) => ({ shape: pick(r, ['sine', 'step', 'poly', 'abs']), n: int(r, 12, 90), stages: int(r, 1, 12), depth: int(r, 1, 3), lr: step(r, 0.1, 1, 0.05), noise: step(r, 0, 0.8, 0.05), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.Data.curve({ n: input.n, shape: input.shape, noise: input.noise, seed: input.seed }), X = d.X, y = d.y;
      const m = S.ML.GradientBoostingRegressor({ nEstimators: input.stages, learningRate: input.lr, maxDepth: input.depth, seed: input.seed }).fit(X, y);
      let F = y.map(() => y.reduce((a, b) => a + b, 0) / y.length);
      (near(F[0], r.frames[0].meta.init) ? pass : fail)('the start is the mean');
      r.frames.slice(1).forEach((f, k) => {
        const T = m.stages[k][0], res = y.map((v, i) => v - F[i]);
        const big = res.reduce((b, v, i) => (Math.abs(v) > Math.abs(res[b]) ? i : b), 0);
        (near(res[big], f.meta.resBig) ? pass : fail)('largest residual at stage ' + (k + 1));
        // each leaf holds the mean residual of its rows
        const leaves = T.nodes.filter((nd) => nd.left < 0);
        (leaves.every((nd) => { const rows = X.map((x, i) => i).filter((i) => walk(T, X[i]).id === nd.id); return !rows.length || near(nd.value, rows.reduce((s, i) => s + res[i], 0) / rows.length, 1e-9); }) ? pass : fail)('leaves are mean residuals at stage ' + (k + 1));
        F = F.map((v, i) => v + input.lr * walk(T, X[i]).value);
        (near(y.reduce((s, v, i) => s + (v - F[i]) ** 2, 0) / y.length, f.meta.mse, 1e-9) ? pass : fail)('error after stage ' + (k + 1));
      });
    },
  },
  'gb-shrinkage': {
    nRandom: 3,
    random: (r) => ({ shape: pick(r, ['sine', 'step', 'poly', 'abs']), n: int(r, 15, 80), stages: int(r, 10, 60), depth: int(r, 1, 4), noise: step(r, 0.05, 0.8, 0.05), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const tr = S.Data.curve({ n: input.n, shape: input.shape, noise: input.noise, seed: input.seed }), te = S.Data.curve({ n: 200, shape: input.shape, noise: input.noise, seed: input.seed + 1000 });
      [1, 0.3, 0.1].forEach((lr, k) => {
        const m = S.ML.GradientBoostingRegressor({ nEstimators: input.stages, learningRate: lr, maxDepth: input.depth, seed: input.seed }).fit(tr.X, tr.y);
        const mean = tr.y.reduce((a, b) => a + b, 0) / tr.y.length;
        let Ft = tr.y.map(() => mean), Fe = te.y.map(() => mean);
        const at = new Map(r.frames.map((f) => [f.meta.s, f.meta]));
        for (let s = 1; s <= input.stages; s++) {
          const T = m.stages[s - 1][0];
          Ft = Ft.map((v, i) => v + lr * walk(T, tr.X[i]).value); Fe = Fe.map((v, i) => v + lr * walk(T, te.X[i]).value);
          const mm = at.get(s);
          if (!mm) continue;
          const a = tr.y.reduce((q, v, i) => q + (v - Ft[i]) ** 2, 0) / tr.y.length, b = te.y.reduce((q, v, i) => q + (v - Fe[i]) ** 2, 0) / 200;
          (near(a, mm.train[k], 1e-9) && near(b, mm.test[k], 1e-9) ? pass : fail)('rate ' + lr + ' stage ' + s);
        }
      });
    },
  },
  'ensemble-compare': {
    nRandom: 3,
    random: (r) => ({ set: pick(r, SETS), n: int(r, 30, 120), noise: step(r, 0.05, 0.6, 0.05), trees: int(r, 5, 20), seed: int(r, 1, 999) }),
    check({ r, input, fail, pass, near }) {
      const d = S.PL5.data2(input.set, input.n, input.seed, input.noise), te = S.PL5.data2(input.set, 300, input.seed + 1000, input.noise), B = input.trees, K = d.K;
      const avg = (trees, x) => { const s = new Array(K).fill(0); trees.forEach((T) => walk(T, x).value.forEach((q, k) => (s[k] += q))); return s; };
      const t1 = S.ML.DecisionTreeClassifier({ seed: 0 }).fit(d.X, d.y), bag = S.ML.RandomForestClassifier({ nEstimators: B, maxFeatures: null, seed: input.seed }).fit(d.X, d.y), rf = S.ML.RandomForestClassifier({ nEstimators: B, maxFeatures: 'sqrt', seed: input.seed }).fit(d.X, d.y);
      const ada = S.ML.AdaBoostClassifier({ nEstimators: 2 * B, maxDepth: 1, seed: input.seed }).fit(d.X, d.y), gb = S.ML.GradientBoostingClassifier({ nEstimators: 2 * B, learningRate: 0.1, maxDepth: 2, seed: input.seed }).fit(d.X, d.y);
      const preds = [
        (x) => argmax(walk(t1.tree, x).value), (x) => argmax(avg(bag.trees, x)), (x) => argmax(avg(rf.trees, x)),
        (x) => { const s = new Array(K).fill(0); ada.trees.forEach((T, j) => { const c = argmax(walk(T, x).value); for (let k = 0; k < K; k++) s[k] += k === c ? ada.alphas[j] : -ada.alphas[j] / (K - 1); }); return argmax(s); },
        (x) => { const raw = gb.init.slice(); gb.stages.forEach((st) => st.forEach((T, k) => (raw[k] += 0.1 * walk(T, x).value))); return raw.length === 1 ? (raw[0] > 0 ? 1 : 0) : argmax(raw); },
      ];
      r.frames.filter((f) => f.meta.phase === 'model').forEach((f) => {
        const p = preds[f.meta.k - 1];
        const a = te.X.filter((x, i) => p(x) === te.y[i]).length / 300, b = d.X.filter((x, i) => p(x) === d.y[i]).length / d.X.length;
        (near(a, f.meta.test) && near(b, f.meta.train) ? pass : fail)('model ' + f.meta.k + ' accuracies ' + f.meta.test + '/' + f.meta.train + ' by hand ' + a + '/' + b);
      });
    },
  },
};

const res = runPart(S, t, rnd, /^p05-/, mods, only);
t.ok(res.runs > 0, 'some runs');
t.done();
