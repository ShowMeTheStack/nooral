// Shapley explanations: exact Shapley values (up to 10 features), Kernel SHAP (weighted linear regression over
// coalitions) and Tree SHAP for a decision tree (path-dependent, polynomial time).
// A model is a function from a list of rows to a list of numbers (a score or the probability of one class).
// A coalition S of features "present" is valued as v(S) = mean over the background rows b of f(x on S, b elsewhere).
(function (S) {
  'use strict';
  const E = (S.Explain = S.Explain || {});
  const M = () => S.Metrics;
  const f = (x) => M().f(x, 4), st = (k, s, d) => M().step(k, s, d);
  const names = (opt, m) => opt.names || Array.from({ length: m }, (_, i) => 'x' + i);
  const popc = (m) => { let c = 0; while (m) { c += m & 1; m >>>= 1; } return c; };
  const fact = (k) => { let r = 1; for (let i = 2; i <= k; i++) r *= i; return r; };
  const binom = (n, k) => fact(n) / (fact(k) * fact(n - k));
  const asModel = (m) => (typeof m === 'function' ? m : (X) => m.predict(X));

  /** v(mask) for every mask in the list, evaluated in one batch */
  const coalitionValues = (model, x, bg, masks) => {
    const rows = [];
    for (const m of masks) for (const b of bg) rows.push(b.map((bv, j) => ((m >>> j) & 1 ? x[j] : bv)));
    const out = model(rows), v = new Array(masks.length).fill(0);
    masks.forEach((_, k) => { for (let r = 0; r < bg.length; r++) v[k] += out[k * bg.length + r]; v[k] /= bg.length; });
    return v;
  };

  const explainTrace = (phi, base, fx, nm) => {
    const order = phi.map((_, i) => i).sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
    return [
      st('base', 'With no feature known, the model gives on average ' + f(base) + '; for this example it gives ' + f(fx) + '.', { base, fx }),
      ...order.map((i) => st('feature', nm[i] + (phi[i] >= 0 ? ' pushes the output up by ' : ' pulls the output down by ') + f(Math.abs(phi[i])) + '.', { feature: i, phi: phi[i] })),
      st('sum', 'The contributions add up: ' + f(base) + ' + (' + phi.map((p) => f(p)).join(') + (') + ') = ' + f(base + phi.reduce((a, b) => a + b, 0)) + '.', { phi }),
    ];
  };

  /**
   * shapExact(model, x, background, {names}) → { phi, base, fx, values, trace }
   * φᵢ = Σ over coalitions S without i of |S|!(M−|S|−1)!/M! · (v(S ∪ {i}) − v(S)); at most 10 features.
   */
  E.shapExact = (model, x, background, opt = {}) => {
    const m = x.length; model = asModel(model);
    if (m > 10) throw new Error('exact Shapley values need 2^M model calls per background row; use Kernel SHAP for more than 10 features (here ' + m + ')');
    const masks = Array.from({ length: 1 << m }, (_, i) => i), v = coalitionValues(model, x, background, masks);
    const w = Array.from({ length: m }, (_, s) => (fact(s) * fact(m - s - 1)) / fact(m));
    const phi = new Array(m).fill(0);
    for (let i = 0; i < m; i++) for (let k = 0; k < masks.length; k++) if (!((k >>> i) & 1)) phi[i] += w[popc(k)] * (v[k | (1 << i)] - v[k]);
    const base = v[0], fx = v[masks.length - 1];
    return { phi, base, fx, values: v, trace: [st('coalitions', 'Evaluated all ' + masks.length + ' coalitions of ' + m + ' features against ' + background.length + ' background rows.'), ...explainTrace(phi, base, fx, names(opt, m))] };
  };

  // weighted least squares with Σφ = Δ, by eliminating the last feature, then Gaussian elimination
  const solveConstrained = (Z, y, w, delta) => {
    const m = Z[0].length;
    if (m === 1) return [delta];
    const p = m - 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
    Z.forEach((z, k) => {
      const r = z.slice(0, p).map((v) => v - z[p]), t = y[k] - z[p] * delta;
      for (let i = 0; i < p; i++) { b[i] += w[k] * r[i] * t; for (let j = 0; j < p; j++) A[i][j] += w[k] * r[i] * r[j]; }
    });
    for (let i = 0; i < p; i++) A[i][i] += 1e-12;
    for (let c = 0; c < p; c++) {
      let piv = c; for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
      for (let r = c + 1; r < p; r++) { const q = A[r][c] / A[c][c]; for (let k = c; k < p; k++) A[r][k] -= q * A[c][k]; b[r] -= q * b[c]; }
    }
    const phi = new Array(p).fill(0);
    for (let i = p - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < p; k++) s -= A[i][k] * phi[k]; phi[i] = s / A[i][i]; }
    return [...phi, delta - phi.reduce((a, c) => a + c, 0)];
  };

  /**
   * kernelShap(model, x, background, {nSamples: 2048, seed: 1, names}) → { phi, base, fx, coalitions, trace }
   * Small coalition sizes (and their complements) are enumerated while the budget allows; the rest are sampled
   * with the Shapley kernel. With a budget of at least 2^M − 2 every coalition is used and φ is exact.
   */
  E.kernelShap = (model, x, background, opt = {}) => {
    const m = x.length; model = asModel(model);
    const budget = opt.nSamples || 2048, rnd = S.Data.rng(opt.seed || 1);
    const [base, fx] = coalitionValues(model, x, background, [0, m >= 31 ? -1 >>> 0 : (1 << m) - 1]);
    const kern = (s) => (m - 1) / (binom(m, s) * s * (m - s));
    const coal = new Map(), add = (mask, w) => coal.set(mask, (coal.get(mask) || 0) + w);
    const nSizes = Math.ceil((m - 1) / 2), done = new Set();
    let left = budget;
    if (m > 1) {
      const sizeMass = (s) => binom(m, s) * kern(s) * (s !== m - s ? 2 : 1);
      for (let s = 1; s <= nSizes; s++) {
        const paired = s !== m - s, count = binom(m, s) * (paired ? 2 : 1);
        let rest = 0; for (let q = s; q <= nSizes; q++) rest += sizeMass(q);
        if (left * sizeMass(s) / rest < count - 1e-8) break;
        for (let mask = 0; mask < 1 << m; mask++) if (popc(mask) === s) { add(mask, kern(s)); if (paired) add(((1 << m) - 1) ^ mask, kern(s)); }
        done.add(s); left -= count;
      }
      const todo = []; for (let s = 1; s <= nSizes; s++) if (!done.has(s)) todo.push(s);
      if (todo.length && left > 1) {
        const mass = todo.map(sizeMass), total = mass.reduce((a, b) => a + b, 0), draws = new Map();
        let drawn = 0;
        for (let k = 0; k < Math.floor(left / 2); k++) {
          let u = rnd() * total, i = 0; while (i < todo.length - 1 && u > mass[i]) u -= mass[i++];
          const s = todo[i], idx = Array.from({ length: m }, (_, j) => j);
          for (let j = 0; j < s; j++) { const r = j + Math.floor(rnd() * (m - j)); [idx[j], idx[r]] = [idx[r], idx[j]]; }
          let mask = 0; for (let j = 0; j < s; j++) mask |= 1 << idx[j];
          draws.set(mask, (draws.get(mask) || 0) + 1); drawn++;
          if (s !== m - s) { const c = ((1 << m) - 1) ^ mask; draws.set(c, (draws.get(c) || 0) + 1); drawn++; }
        }
        for (const [mask, c] of draws) add(mask, (total * c) / drawn);
      }
    }
    const masks = [...coal.keys()], w = masks.map((k) => coal.get(k));
    const v = masks.length ? coalitionValues(model, x, background, masks) : [];
    const Z = masks.map((k) => Array.from({ length: m }, (_, j) => (k >>> j) & 1));
    const phi = m === 1 ? [fx - base] : solveConstrained(Z, v.map((q) => q - base), w, fx - base);
    return { phi, base, fx, coalitions: masks.length, trace: [
      st('coalitions', (done.size === nSizes ? 'Used every one of the ' + masks.length + ' coalitions' : 'Enumerated coalition sizes ' + ([...done].join(', ') || 'none') + ' and sampled the rest, ' + masks.length + ' distinct coalitions in all') + ', weighted by the Shapley kernel.', { coalitions: masks.length }),
      st('regression', 'A weighted linear fit of the coalition values on which features are present, constrained to add up to ' + f(fx - base) + ', gives the contributions.'),
      ...explainTrace(phi, base, fx, names(opt, m))] };
  };

  /**
   * a tree for Tree SHAP from a list of nodes {feature, threshold, left, right, value, cover} (x ≤ threshold goes
   * left, left < 0 marks a leaf). Also accepts a tree grown by S.ML.growTree (cover from w or n; value[output] when
   * the value is a list of class probabilities).
   */
  E.treeFromNodes = (nodes, opt = {}) => ({ float32: !!nodes.float32, nodes: (nodes.nodes || nodes).map((n) => ({
    feature: n.feature, threshold: n.threshold, left: n.left, right: n.right,
    value: Array.isArray(n.value) || ArrayBuffer.isView(n.value) ? n.value[opt.output || 0] : n.value,
    cover: n.cover != null ? n.cover : n.w != null ? n.w : n.n })) });
  E.treePredict = (tree, x) => { if (tree.float32) x = x.map(Math.fround); let n = tree.nodes[0]; while (n.left >= 0) n = tree.nodes[x[n.feature] <= n.threshold ? n.left : n.right]; return n.value; };

  /** treeShap(tree, x, {names}) → { phi, base, fx, trace }: exact Shapley values for the tree's own expectation */
  E.treeShap = (tree, x, opt = {}) => {
    if (tree.float32) x = x.map(Math.fround);
    const N = tree.nodes, m = x.length, phi = new Array(m).fill(0);
    const extend = (path, pz, po, pi) => {
      const l = path.length, p = path.map((e) => Object.assign({}, e));
      p.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 });
      for (let i = l - 1; i >= 0; i--) { p[i + 1].w += (po * p[i].w * (i + 1)) / (l + 1); p[i].w = (pz * p[i].w * (l - i)) / (l + 1); }
      return p;
    };
    const unwind = (path, i) => {
      const l = path.length - 1, p = path.map((e) => Object.assign({}, e)), o = p[i].o, z = p[i].z;
      let n = p[l].w;
      for (let j = l - 1; j >= 0; j--) {
        if (o !== 0) { const t = p[j].w; p[j].w = (n * (l + 1)) / ((j + 1) * o); n = t - (p[j].w * z * (l - j)) / (l + 1); }
        else p[j].w = (p[j].w * (l + 1)) / (z * (l - j));
      }
      for (let j = i; j < l; j++) { p[j].d = p[j + 1].d; p[j].z = p[j + 1].z; p[j].o = p[j + 1].o; }
      p.pop(); return p;
    };
    const recurse = (j, path, pz, po, pi) => {
      path = extend(path, pz, po, pi);
      const nd = N[j];
      if (nd.left < 0) {
        for (let i = 1; i < path.length; i++) { const w = unwind(path, i).reduce((a, e) => a + e.w, 0); phi[path[i].d] += w * (path[i].o - path[i].z) * nd.value; }
        return;
      }
      const hot = x[nd.feature] <= nd.threshold ? nd.left : nd.right, cold = hot === nd.left ? nd.right : nd.left;
      let iz = 1, io = 1;
      const k = path.findIndex((e, q) => q > 0 && e.d === nd.feature);
      if (k > 0) { iz = path[k].z; io = path[k].o; path = unwind(path, k); }
      recurse(hot, path, (iz * N[hot].cover) / nd.cover, io, nd.feature);
      recurse(cold, path, (iz * N[cold].cover) / nd.cover, 0, nd.feature);
    };
    recurse(0, [], 1, 1, -1);
    const expect = (j) => (N[j].left < 0 ? N[j].value : (N[N[j].left].cover * expect(N[j].left) + N[N[j].right].cover * expect(N[j].right)) / N[j].cover);
    const base = expect(0), fx = E.treePredict(tree, x);
    return { phi, base, fx, trace: [st('paths', 'Followed every root-to-leaf path once, tracking what share of each path the known features would take.'), ...explainTrace(phi, base, fx, names(opt, m))] };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
