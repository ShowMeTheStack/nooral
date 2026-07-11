// CART decision trees for classification (Gini, entropy) and regression (squared error), grown
// depth first with every candidate split's impurity in the trace, plus cost-complexity pruning.
// The node layout (preorder, left before right) and split rules follow the usual CART conventions:
// thresholds sit halfway between neighbouring distinct values and x <= threshold goes left.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x, 4);
  const EPS = 2.220446049250313e-16, FEATURE_THRESHOLD = 1e-7;

  // impurity of weighted statistics: classes -> counts array, regression -> {w, s, ss}
  function impurity(crit, st, W) {
    if (W <= 0) return 0;
    if (crit === 'gini') { let s = 0; for (const c of st) s += c * c; return 1 - s / (W * W); }
    if (crit === 'entropy') { let s = 0; for (const c of st) if (c > 0) { const p = c / W; s -= p * (Math.log(p) / Math.LN2); } return s; }
    const m = st.s / W; return st.ss / W - m * m;
  }
  // the quantity a split maximises (larger is better); its ordering equals that of the impurity decrease
  function proxy(crit, L, R, Wl, Wr) {
    if (crit === 'gini' || crit === 'entropy') return -Wr * impurity(crit, R, Wr) - Wl * impurity(crit, L, Wl);
    if (crit === 'friedman_mse') { const diff = Wr * L.s - Wl * R.s; return (diff * diff) / (Wl * Wr); }
    return (L.s * L.s) / Wl + (R.s * R.s) / Wr;
  }

  // the splitter's random numbers: a 32-bit xorshift whose state is the tree's seed
  function randR(seed) {
    let s = seed >>> 0 || 1;
    return (lo, hi) => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return lo + ((s % 0x80000000) % (hi - lo)); };
  }

  // Grow a tree. y: class indices (nClasses given) or numbers. w: sample weights (0 drops a sample).
  // At every node the features are drawn in a random order (seeded); with maxFeatures < d only that many
  // non-constant features are searched. Features found constant are remembered for the node's subtree.
  ML.growTree = function (X, y, opts) {
    const o = Object.assign({ criterion: 'gini', maxDepth: Infinity, minSamplesSplit: 2, minSamplesLeaf: 1, maxFeatures: null, minImpurityDecrease: 0, seed: 1, tr: null, nClasses: 0 }, opts);
    const d = X[0].length, crit = o.criterion, cls = crit === 'gini' || crit === 'entropy', K = o.nClasses;
    const w = o.w || X.map(() => 1), tr = o.tr, rand = randR(o.seed);
    // float32: work on single-precision copies of the features, as libraries that store X in float32 do
    if (o.float32) X = X.map((r) => r.map(Math.fround));
    const all = ML.range(X.length).filter((i) => w[i] !== 0);
    let Wtot = 0; for (const i of all) Wtot += w[i];
    const nodes = [], imp = new Array(d).fill(0), features = ML.range(d), constant = ML.range(d);
    const stats = (idx) => {
      if (cls) { const c = new Array(K).fill(0); for (const i of idx) c[y[i]] += w[i]; return c; }
      const s = { w: 0, s: 0, ss: 0 }; for (const i of idx) { s.w += w[i]; s.s += w[i] * y[i]; s.ss += w[i] * y[i] * y[i]; } return s;
    };
    const maxF = o.maxFeatures == null ? d : Math.max(1, Math.min(d, o.maxFeatures));
    const swap = (a, i, j) => { const t = a[i]; a[i] = a[j]; a[j] = t; };

    const grow = (idx, depth, parent, nKnown, givenImpurity) => {
      const st = stats(idx);
      let W = 0; for (const i of idx) W += w[i];
      const nodeImp = givenImpurity == null ? impurity(crit === 'friedman_mse' ? 'mse' : crit, st, W) : givenImpurity;
      const node = { id: nodes.length, depth, n: idx.length, w: W, impurity: nodeImp, left: -1, right: -1, feature: -1, threshold: -2, parent };
      node.value = cls ? st.map((c) => c / W) : st.s / W;
      nodes.push(node);
      let leaf = depth >= o.maxDepth || idx.length < o.minSamplesSplit || idx.length < 2 * o.minSamplesLeaf || node.impurity <= EPS;
      let best = null, nTotal = nKnown, cur = idx;
      const cand = tr ? [] : null;
      if (!leaf) {
        let fi = d, visited = 0, found = 0, drawn = 0;
        while (fi > nTotal && (visited < maxF || visited <= found + drawn)) {
          visited++;
          let fj = rand(drawn, fi - found);
          if (fj < nKnown) { swap(features, drawn, fj); drawn++; continue; }
          fj += found;
          const j = features[fj];
          const s = cur.slice().sort((a, b) => X[a][j] - X[b][j] || a - b);
          cur = s; // the node's samples stay in the order of the last feature examined
          if (X[s[s.length - 1]][j] <= X[s[0]][j] + FEATURE_THRESHOLD) { swap(features, fj, nTotal); found++; nTotal++; continue; }
          fi--; swap(features, fi, fj);
          const L = cls ? new Array(K).fill(0) : { w: 0, s: 0, ss: 0 };
          let Wl = 0, pos = 0;
          const feat = cand ? { feature: j, thresholds: [], scores: [] } : null;
          for (let p = 1; p < s.length; p++) {
            const i = s[p - 1];
            if (cls) { L[y[i]] += w[i]; Wl += w[i]; }
            if (X[s[p]][j] <= X[i][j] + FEATURE_THRESHOLD) continue;
            if (p < o.minSamplesLeaf || s.length - p < o.minSamplesLeaf) continue;
            if (!cls) {
              // regression sums move from the last evaluated position, forwards or back from the end,
              // whichever is shorter (this fixes the rounding, and with it the choice between near-ties)
              if (p - pos <= s.length - p) for (let q = pos; q < p; q++) { const t = s[q]; L.w += w[t]; L.s += w[t] * y[t]; L.ss += w[t] * y[t] * y[t]; }
              else { L.w = st.w; L.s = st.s; L.ss = st.ss; for (let q = s.length - 1; q >= p; q--) { const t = s[q]; L.w -= w[t]; L.s -= w[t] * y[t]; L.ss -= w[t] * y[t] * y[t]; } }
              pos = p; Wl = L.w;
            }
            const R = cls ? st.map((c, k) => c - L[k]) : { w: st.w - L.w, s: st.s - L.s, ss: st.ss - L.ss }, Wr = W - Wl;
            const score = proxy(crit, L, R, Wl, Wr);
            let thr = X[i][j] / 2 + X[s[p]][j] / 2;
            if (thr === X[s[p]][j] || !isFinite(thr)) thr = X[i][j];
            if (feat) { feat.thresholds.push(thr); feat.scores.push(cls ? -score / W : (Wl * impurity('mse', L, Wl) + Wr * impurity('mse', R, Wr)) / W); }
            if (!best || score > best.score) best = { feature: j, threshold: thr, score, L: cls ? L.slice() : Object.assign({}, L), R, Wl, Wr, pos: p, sorted: s };
          }
          if (feat) cand.push(feat);
        }
        for (let k = 0; k < nKnown; k++) features[k] = constant[k];
        for (let k = nKnown; k < nKnown + found; k++) constant[k] = features[k];
        if (!best) leaf = true;
        else {
          const ic = crit === 'friedman_mse' ? 'mse' : crit;
          best.il = impurity(ic, best.L, best.Wl); best.ir = impurity(ic, best.R, best.Wr);
          best.child = (best.Wl * best.il + best.Wr * best.ir) / W;
          const improvement = (W / Wtot) * (node.impurity - best.child);
          if (improvement + EPS < o.minImpurityDecrease) leaf = true;
        }
      }
      if (tr) {
        tr.add('node', 'Node ' + node.id + ' at depth ' + depth + ' holds ' + idx.length + ' samples with ' + crit + ' ' + f(node.impurity) + '.', { node: node.id, depth, n: idx.length, impurity: node.impurity, value: node.value });
        if (cand && cand.length) tr.add('candidates', 'Tried ' + cand.reduce((a, c) => a + c.thresholds.length, 0) + ' thresholds over ' + cand.length + ' features; the weighted child impurity is lowest, ' + (best ? f(best.child) : '-') + ', for x' + (best ? best.feature : '-') + ' <= ' + (best ? f(best.threshold) : '-') + '.', { node: node.id, candidates: cand });
      }
      if (leaf) { if (tr) tr.add('leaf', 'Node ' + node.id + ' becomes a leaf predicting ' + (cls ? ML.fmtv(node.value) : f(node.value)) + '.', { node: node.id, value: node.value }); return node.id; }
      node.feature = best.feature; node.threshold = best.threshold;
      if (tr) tr.add('split', 'Node ' + node.id + ' splits on x' + best.feature + ' <= ' + f(best.threshold) + ': impurity ' + f(node.impurity) + ' falls to ' + f(best.il) + ' (left) and ' + f(best.ir) + ' (right).', { node: node.id, feature: best.feature, threshold: best.threshold, il: best.il, ir: best.ir });
      // in-place partition of the current order around the threshold (fixes the children's sample order)
      const part = cur.slice();
      let pp = 0, end = part.length;
      while (pp < end) { if (X[part[pp]][best.feature] <= best.threshold) pp++; else { end--; const t = part[pp]; part[pp] = part[end]; part[end] = t; } }
      const li = part.slice(0, pp), ri = part.slice(pp);
      node.left = grow(li, depth + 1, node.id, nTotal, best.il);
      node.right = grow(ri, depth + 1, node.id, nTotal, best.ir);
      return node.id;
    };
    grow(all, 0, -1, 0, null);
    // importance: the weighted impurity decrease of every split, summed per feature
    nodes.forEach((n) => { if (n.left >= 0) { const l = nodes[n.left], r = nodes[n.right]; imp[n.feature] += n.w * n.impurity - l.w * l.impurity - r.w * r.impurity; } });
    const tot = ML.sum(imp);
    return { nodes, classification: cls, nFeatures: d, float32: !!o.float32, importances: tot > 0 ? imp.map((v) => v / tot) : imp, rawImportance: imp };
  };

  ML.treeApply = function (T, x) { if (T.float32) x = x.map(Math.fround); let n = T.nodes[0]; while (n.left >= 0) n = T.nodes[x[n.feature] <= n.threshold ? n.left : n.right]; return n.id; };
  ML.treePath = function (T, x) { if (T.float32) x = x.map(Math.fround); const p = []; let n = T.nodes[0]; for (;;) { p.push(n.id); if (n.left < 0) return p; n = T.nodes[x[n.feature] <= n.threshold ? n.left : n.right]; } };
  ML.treeDepth = (T) => Math.max(...T.nodes.map((n) => n.depth));
  ML.treeLeaves = (T) => T.nodes.filter((n) => n.left < 0).length;
  // arrays in the common export layout
  ML.treeArrays = (T) => ({
    childrenLeft: T.nodes.map((n) => n.left), childrenRight: T.nodes.map((n) => n.right),
    feature: T.nodes.map((n) => (n.left < 0 ? -2 : n.feature)), threshold: T.nodes.map((n) => (n.left < 0 ? -2 : n.threshold)),
    impurity: T.nodes.map((n) => n.impurity), nNodeSamples: T.nodes.map((n) => n.n), value: T.nodes.map((n) => n.value),
  });

  // weakest-link pruning. Returns the path (alphas and total leaf impurity) and, given ccpAlpha, a pruned copy.
  ML.ccpPrune = function (T, ccpAlpha) {
    const N = T.nodes, W0 = N[0].w, rNode = N.map((n) => (n.w / W0) * n.impurity);
    const leafOf = N.map((n) => n.left < 0), alive = N.map(() => true);
    const rBranch = new Array(N.length).fill(0), nLeaves = new Array(N.length).fill(0);
    const refresh = () => {
      for (let i = N.length - 1; i >= 0; i--) {
        if (!alive[i]) continue;
        if (leafOf[i]) { rBranch[i] = rNode[i]; nLeaves[i] = 1; } else { const l = N[i].left, r = N[i].right; rBranch[i] = rBranch[l] + rBranch[r]; nLeaves[i] = nLeaves[l] + nLeaves[r]; }
      }
    };
    refresh();
    const alphas = [0], impurities = [rBranch[0]], prunedAt = [];
    for (;;) {
      let a = Infinity, at = -1;
      for (let i = 0; i < N.length; i++) {
        if (!alive[i] || leafOf[i]) continue;
        const g = (rNode[i] - rBranch[i]) / (nLeaves[i] - 1);
        if (g < a) { a = g; at = i; }
      }
      if (at < 0) break;
      if (ccpAlpha != null && a > ccpAlpha) break;
      leafOf[at] = true;
      const kill = (i) => { if (N[i].left < 0) return; for (const c of [N[i].left, N[i].right]) { alive[c] = false; kill(c); } };
      kill(at);
      refresh();
      alphas.push(Math.max(a, 0)); impurities.push(rBranch[0]); prunedAt.push(at);
    }
    // the surviving tree, renumbered in preorder
    const map = new Map(), nodes = [];
    const copy = (i, parent) => {
      const n = Object.assign({}, N[i], { id: nodes.length, parent });
      map.set(i, n.id); nodes.push(n);
      if (leafOf[i]) { n.left = -1; n.right = -1; n.feature = -1; n.threshold = -2; return n.id; }
      n.left = copy(N[i].left, n.id); n.right = copy(N[i].right, n.id);
      return n.id;
    };
    copy(0, -1);
    const imp = new Array(T.nFeatures).fill(0);
    nodes.forEach((n) => { if (n.left >= 0) { const l = nodes[n.left], r = nodes[n.right]; imp[n.feature] += n.w * n.impurity - l.w * l.impurity - r.w * r.impurity; } });
    const tot = ML.sum(imp);
    return { alphas, impurities, prunedAt, tree: Object.assign({}, T, { nodes, importances: tot > 0 ? imp.map((v) => v / tot) : imp }) };
  };

  const treeDefaults = { classWeight: null, criterion: 'gini', maxDepth: null, minSamplesSplit: 2, minSamplesLeaf: 1, maxFeatures: null, minImpurityDecrease: 0, ccpAlpha: 0, seed: 0, float32: false };
  function treeFit(X, y, tr, w) {
    const p = this.params, cls = this.kind === 'classifier';
    let yy = y;
    if (cls) { const e = ML.encode(y); this.classes = e.classes; yy = e.idx; }
    // class weights multiply each row's weight (as scikit-learn's class_weight)
    const CW = cls ? ML.classWeights(p.classWeight, this.classes, yy) : null;
    if (CW) { const w0 = w || p.sampleWeight; w = yy.map((k, i) => (w0 ? w0[i] : 1) * CW[k]); }
    const mf = ML.resolveMaxFeatures ? ML.resolveMaxFeatures(p.maxFeatures, X[0].length) : null;
    let T = ML.growTree(X, yy, { criterion: cls ? p.criterion : p.criterion === 'friedman_mse' ? 'friedman_mse' : 'mse', maxDepth: p.maxDepth == null ? Infinity : p.maxDepth, minSamplesSplit: p.minSamplesSplit, minSamplesLeaf: p.minSamplesLeaf, maxFeatures: mf, minImpurityDecrease: p.minImpurityDecrease, seed: p.seed, tr: tr.on ? tr : null, nClasses: cls ? this.classes.length : 0, w: w || p.sampleWeight, float32: p.float32 });
    if (p.ccpAlpha > 0) {
      const pr = ML.ccpPrune(T, p.ccpAlpha);
      tr.add('prune', 'Cost-complexity pruning with alpha = ' + f(p.ccpAlpha) + ' collapsed ' + pr.prunedAt.length + ' subtrees, leaving ' + ML.treeLeaves(pr.tree) + ' leaves.', { prunedAt: pr.prunedAt });
      T = pr.tree;
    }
    this.tree = T;
    this.featureImportances = T.importances;
    this.depth = ML.treeDepth(T); this.nLeaves = ML.treeLeaves(T); this.nodeCount = T.nodes.length;
  }
  const treeMethods = {
    apply(X) { return X.map((x) => ML.treeApply(this.tree, x)); },
    decisionPath(x) { return ML.treePath(this.tree, x); },
    costComplexityPath(X, y) { const t = this.clone({ ccpAlpha: 0 }).fit(X, y); const r = ML.ccpPrune(t.tree); return { alphas: r.alphas, impurities: r.impurities }; },
    arrays() { return ML.treeArrays(this.tree); },
    // fit with sample weights (bootstrap counts, boosting weights)
    fitWeighted(X, y, w) { const tr = ML.tracer(this.params); treeFit.call(this, X, y, tr, w); this.trace = tr.steps; this.fitted = true; return this; },
  };
  ML.model('DecisionTreeClassifier', Object.assign({
    kind: 'classifier', defaults: Object.assign({}, treeDefaults),
    fit: treeFit,
    predictProba(X) { return X.map((x) => this.tree.nodes[ML.treeApply(this.tree, x)].value.slice()); },
    predict(X) { return this._argmaxPredict(X); },
  }, treeMethods));
  ML.model('DecisionTreeRegressor', Object.assign({
    kind: 'regressor', defaults: Object.assign({}, treeDefaults, { criterion: 'mse' }),
    fit: treeFit,
    predict(X) { return X.map((x) => this.tree.nodes[ML.treeApply(this.tree, x)].value); },
  }, treeMethods));
})();
