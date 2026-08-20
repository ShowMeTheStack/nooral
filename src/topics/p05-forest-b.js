// Part 5, random forests: the features each split may look at (a random few, drawn afresh at every node, so the
// trees differ even on similar samples), and the two ways to rank features: impurity importance, summed from the
// training splits, against permutation importance, the accuracy lost on test rows when one column is shuffled.
(function (S) {
  'use strict';
  const H = S.PL, T5 = S.PL5, ML = () => S.ML;
  const n$ = H.n;
  /** bars from a zero line: values (may be slightly negative), names on the left, numbers on the right */
  const bars = (key, x, y, w, rowH, vals, names, o) => {
    o = o || {};
    const max = o.max || Math.max(1e-9, ...vals.map(Math.abs)), z = x + (o.neg ? 24 : 0), W = w - (z - x), items = [];
    items.push({ key: key + ':z', kind: 'path', d: `M${z},${y - 3} v${vals.length * rowH}`, cls: 'axis', audit: 'ignore', layer: 'back' });
    vals.forEach((v, i) => {
      if (v === null) return;
      const cy = y + i * rowH, bw = (Math.abs(v) / max) * W;
      if (bw > 0.4) items.push({ key: key + ':b' + i, kind: 'cell', x: H.r2(v >= 0 ? z : z - Math.min(bw, z - x)), y: H.r2(cy), w: H.r2(v >= 0 ? bw : Math.min(bw, z - x)), h: rowH - 8, rx: 2, text: '', cls: 'pl-pbar ' + (o.cls ? o.cls(i) : ''), audit: 'ignore' });
      if (o.err && o.err[i] > 0) { const e = (o.err[i] / max) * W, c = v >= 0 ? z + bw : z - bw, yy = cy + (rowH - 8) / 2; items.push({ key: key + ':e' + i, kind: 'path', d: `M${H.r2(c - e)},${H.r2(yy)} H${H.r2(c + e)} M${H.r2(c - e)},${H.r2(yy - 4)} v8 M${H.r2(c + e)},${H.r2(yy - 4)} v8`, cls: 'p5-err', audit: 'ignore', layer: 'lines' }); }
      items.push(H.text(key + ':v' + i, x + w + 10, cy + rowH / 2, n$(v, 2), 'tick', 12));
    });
    if (names) names.forEach((t, i) => items.push(H.text(key + ':n' + i, x - 8, y + i * rowH + rowH / 2, t, o.ncls ? o.ncls(i) : 'tick', 12, 'end')));
    return items;
  };
  T5.bars = bars;
  const mfArg = (mf, d) => (mf >= d ? null : mf);

  S.topic({
    id: 'rf-features', family: 'p05-forest', order: 20, title: 'Random forests: a few features per split',
    blurb: 'Bagged trees still look alike: each one starts with the strongest feature at its root. A random forest also draws, at every node, a random handful of the features (max features), and the split may only use those. Weaker features get their turn, the trees disagree more, and their vote improves. Here every split of the first tree is shown with the features it was allowed to try.',
    inputs: [
      { id: 'table', kind: 'choice', label: 'data', def: 'loans', options: T5.TABLES },
      { id: 'mf', kind: 'number', label: 'max features', def: 2, min: 1, max: 6 },
      { id: 'n', kind: 'number', label: 'training rows', def: 150, min: 40, max: 400 },
      { id: 'depth', kind: 'number', label: 'max depth', def: 3, min: 2, max: 3 },
      { id: 'trees', kind: 'number', label: 'trees', def: 30, min: 5, max: 60 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Small loans, 2 of 6 features', values: { table: 'loans', mf: 2, n: 150, depth: 3, trees: 30, seed: 2 } },
      { label: 'Small loans, all 6: bagging', values: { table: 'loans', mf: 6, n: 150, depth: 3, trees: 30, seed: 2 } },
      { label: 'Wildflowers, 1 of 4', values: { table: 'wildflowers', mf: 1, n: 120, depth: 3, trees: 30, seed: 4 } },
      { label: 'Exam, 2 of 3', values: { table: 'exam', mf: 2, n: 120, depth: 2, trees: 20, seed: 5 } },
    ],
    check: (inp) => { const d = T5.table(inp.table, 40, 1).names.length; return inp.mf > d ? 'max features: at most ' + d + ' for this data' : null; },
    counters: [{ id: 'splits', label: 'splits' }],
    run(inp) {
      const t = T5.table(inp.table, inp.n, inp.seed), te = T5.table(inp.table, 300, inp.seed + 1000), d = t.names.length, mf = inp.mf;
      const { draws, seeds } = T5.draws(t.X.length, inp.trees, inp.seed);
      const forest = (m) => ML().RandomForestClassifier({ nEstimators: inp.trees, maxFeatures: mfArg(m, d), maxDepth: inp.depth, bootstrapIndices: draws, treeSeeds: seeds }).fit(t.X, t.y);
      const rf = forest(mf), bag = forest(d);
      const w = new Array(t.X.length).fill(0);
      draws[0].forEach((i) => w[i]++);
      const tr = ML().tracer({});
      const T = ML().growTree(t.X, t.y, { criterion: 'gini', maxDepth: inp.depth, maxFeatures: mf >= d ? null : mf, seed: seeds[0], nClasses: t.K, w, tr });
      const N = T.nodes, F = new S.Frames(['splits']);
      const testAcc = (m) => T5.acc(te.y, m.predict(te.X));
      const rootCount = (m) => { const c = new Array(d).fill(0); m.trees.forEach((tt) => { if (tt.nodes[0].left >= 0) c[tt.nodes[0].feature]++; }); return c; };
      const rcRf = rootCount(rf), rcBag = rootCount(bag), accRf = testAcc(rf), accBag = testAcc(bag);
      // the grid: one row per split node, one column per feature
      const inner = N.filter((x) => x.left >= 0).map((x) => x.id), cw = Math.min(84, 400 / d), rh = 28, gx = 100, gy = 56;
      const cands = {};
      for (const st of tr.steps) if (st.kind === 'candidates') cands[st.node] = st.candidates;
      const best = (c) => Math.min(...c.scores);
      const grid = (upto, act) => {
        const items = [H.title('rf:gt', 0, 24, 'the first tree: what each split could see')];
        t.names.forEach((nm, j) => items.push(H.text('rf:h' + j, gx + j * cw + cw / 2, gy - 10, nm, 'tick', 12, 'middle')));
        inner.forEach((id, r) => {
          const shown = r <= upto, nd = N[id];
          items.push(H.text('rf:r' + r, gx - 10, gy + r * rh + rh / 2 + 1, 'node ' + id + ' · ' + nd.n, id === act ? 'pl-lab act' : 'tick', 12, 'end'));
          for (let j = 0; j < d; j++) {
            const c = shown ? (cands[id] || []).find((q) => q.feature === j) : null;
            items.push({ key: 'rf:c' + r + '_' + j, kind: 'cell', x: H.r2(gx + j * cw), y: gy + r * rh, w: H.r2(cw - 4), h: rh - 4, rx: 3, text: c ? n$(best(c), 3) : '', size: 12, cls: 'p5-feat' + (c ? (nd.feature === j ? ' won' : ' seen') : '') });
          }
        });
        return items;
      };
      const dia = T5.diagram(T, { prefix: 'rf', x: 0, y: 330, w: 760, h: 200, label: (nd, open) => (open ? [t.names[nd.feature] + ' ≤ ' + n$(nd.threshold, 3)] : [t.classes[T5.major(nd)]]), cls: (nd, open) => (open ? '' : 'leaf c' + T5.major(nd)), bar: false, boxH: 30, maxW: 110, size: 12 });
      const right = (show) => {
        if (!show) return [];
        const items = [H.title('rf:bt', 560, 24, 'root feature of ' + inp.trees + ' trees')];
        items.push(H.text('rf:b1', 560, 52, 'max features ' + mf + ': ' + H.pct(accRf) + ' on test', 'pl-lab act', 13));
        items.push(...bars('rf:x', 620, 66, 100, 20, rcRf, t.names, { cls: () => '' }));
        const y2 = 80 + d * 20;
        items.push(H.text('rf:b2', 560, y2, 'all ' + d + ' (bagging): ' + H.pct(accBag) + ' on test', 'pl-lab', 13));
        items.push(...bars('rf:y', 620, y2 + 14, 100, 20, rcBag, t.names, { cls: () => 'soft' }));
        return items;
      };
      const open = new Set();
      F.add({ items: [...grid(-1, -1), ...dia.items(open, { act: 0 })], say: t.X.length + ' rows with ' + d + ' features. Each split of a tree draws ' + H.plural(mf, 'feature') + ' at random' + (mf >= d ? ' (all of them: this is plain bagging)' : '') + ' and searches only those. The first tree’s bootstrap sample has ' + w.filter((v) => v > 0).length + ' different rows.', meta: { phase: 'start', n: t.X.length, d, mf, distinct: w.filter((v) => v > 0).length } });
      inner.forEach((id, r) => {
        const nd = N[id], c = cands[id] || [], seen = c.map((q) => t.names[q.feature]);
        open.add(id);
        F.set('splits', r + 1);
        const others = c.filter((q) => q.feature !== nd.feature);
        const say = 'Node ' + id + ' (' + nd.n + ' rows) may try ' + H.list(seen) + '. The best cut is on ' + t.names[nd.feature] + ' ≤ ' + n$(nd.threshold, 3) + ', weighted Gini ' + n$(best(c.find((q) => q.feature === nd.feature)), 3) + (others.length ? '; ' + H.list(others.map((q) => t.names[q.feature] + ' reaches ' + n$(best(q), 3))) + '.' : '.') + (seen.length < mf ? ' It drew ' + H.plural(mf, 'feature') + ', but ' + (mf - seen.length === 1 ? 'the other one is' : 'the other ' + (mf - seen.length) + ' are') + ' constant in this node, so there is nothing to cut on.' : '') + (mf < d ? ' The other ' + H.plural(d - mf, 'feature') + (d - mf === 1 ? ' was' : ' were') + ' not offered here.' : '');
        F.add({ items: [...grid(r, id), ...dia.items(open, { act: id })], say, meta: { phase: 'split', node: id, nd: nd.n, seen: c.map((q) => q.feature), scores: c.map(best), feature: nd.feature, t: nd.threshold, hidden: d - seen.length, drawn: mf, notOffered: d - mf, constant: Math.max(0, mf - seen.length) } });
      });
      const top = (c) => c.indexOf(Math.max(...c));
      F.add({ items: [...grid(inner.length, -1), ...dia.items(open, {}), ...right(true)], say: 'Over ' + inp.trees + ' trees: with ' + mf + ' feature' + (mf === 1 ? '' : 's') + ' per split, ' + H.plural(rcRf.filter((v) => v > 0).length, 'different feature') + ' start a tree (' + t.names[top(rcRf)] + ' most, ' + rcRf[top(rcRf)] + ' times); with all ' + d + ', ' + H.plural(rcBag.filter((v) => v > 0).length, 'feature') + ' do. On 300 test rows the forest scores ' + H.pct(accRf) + ' and the bagged trees ' + H.pct(accBag) + '.', meta: { phase: 'done', d, mf, trees: inp.trees, rootsRf: rcRf, rootsBag: rcBag, nRf: rcRf.filter((v) => v > 0).length, nBag: rcBag.filter((v) => v > 0).length, accRf, accBag, nte: 300 } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Random forest', text: ['for each tree: a bootstrap sample', 'at each node: draw m of the d features', '  split on the best of those m only', 'usual m: √d (classes), d/3 (numbers)', 'm = d is bagging'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start' && f.meta.mf < f.meta.d, pick: 'option', ask: 'Across all the trees, will more than one feature appear at the root?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.nRf > 1 ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
      { at: (f) => f.meta.phase === 'start' && f.meta.mf >= f.meta.d, pick: 'option', ask: 'With every feature offered, will the trees all start with the same feature?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.nBag === 1 ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What does a random forest add to bagging?', options: ['a random subset of features at every split', 'a random subset of rows at every split', 'random thresholds', 'pruning'], answer: 0, why: 'The bootstrap is the same; the feature draw at each node is new.' },
      { q: 'Why does drawing features help the vote?', options: ['the trees become less alike, so their errors cancel more', 'each tree becomes more accurate', 'it uses less memory', 'it removes noise features'], answer: 0, why: 'Averaging helps most when the averaged errors are not correlated.' },
      { q: 'A common choice of max features for classification with 16 features is…', options: ['4', '16', '1', '8'], answer: 0, why: '√16 = 4.' },
      { q: 'Max features equal to the number of features gives…', options: ['bagged trees', 'a single tree', 'boosting', 'a linear model'], answer: 0, why: 'Every split sees every feature, so only the bootstrap differs.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Impurity importance against permutation importance
  // ---------------------------------------------------------------------------------------------------------------
  const EXTRA = [{ value: 'noise', label: 'a column of pure noise' }, { value: 'copy', label: 'a near copy of the first feature' }, { value: 'both', label: 'both' }, { value: 'none', label: 'nothing' }];
  /** a table with the extra columns added (noise: Gaussian; copy: the first feature plus a little noise) */
  T5.withExtra = (t, extra, seed) => {
    const r = S.Data.rng(seed), names = t.names.slice(), X = t.X.map((x) => x.slice());
    const sd = Math.sqrt(ML().colVars(t.X)[0]) || 1;
    if (extra === 'copy' || extra === 'both') { names.push(t.names[0] + ' copy'); X.forEach((x, i) => x.push(S.Data.round(t.X[i][0] + r.normal(0, 0.15 * sd), 3))); }
    if (extra === 'noise' || extra === 'both') { names.push('noise'); X.forEach((x) => x.push(S.Data.round(r.normal(0, 1), 3))); }
    return { X, y: t.y, names, classes: t.classes, K: t.K };
  };
  /** the first shuffle of each column, as the permutation-importance engine draws it */
  T5.firstShuffles = (X, reps, seed) => {
    const rnd = S.Data.rng(seed || 1), d = X[0].length, out = [];
    for (let j = 0; j < d; j++) for (let r = 0; r < reps; r++) {
      const col = X.map((x) => x[j]);
      for (let i = col.length - 1; i > 0; i--) { const k = Math.floor(rnd() * (i + 1)); [col[i], col[k]] = [col[k], col[i]]; }
      if (r === 0) out.push(col);
    }
    return out;
  };

  S.topic({
    id: 'rf-importance', family: 'p05-forest', order: 40, title: 'Impurity against permutation importance',
    blurb: 'Two ways to ask a forest which features matter. Impurity importance adds up, for each feature, how much its splits lowered the impurity while the trees were trained. Permutation importance shuffles one column of fresh test data, breaking its link to the target, and measures how much accuracy is lost. A column of pure noise gets some impurity importance (the trees did split on it, fitting noise) but almost no permutation importance; two near copies of one feature share the credit between them.',
    inputs: [
      { id: 'table', kind: 'choice', label: 'data', def: 'exam', options: T5.TABLES },
      { id: 'extra', kind: 'choice', label: 'add', def: 'both', options: EXTRA },
      { id: 'n', kind: 'number', label: 'training rows', def: 200, min: 60, max: 400 },
      { id: 'trees', kind: 'number', label: 'trees', def: 25, min: 5, max: 60 },
      { id: 'seed', kind: 'number', label: 'seed', def: 3, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Exam with noise and a copy', values: { table: 'exam', extra: 'both', n: 200, trees: 25, seed: 3 } },
      { label: 'Exam with a noise column', values: { table: 'exam', extra: 'noise', n: 200, trees: 25, seed: 3 } },
      { label: 'Wildflowers with a near copy', values: { table: 'wildflowers', extra: 'copy', n: 150, trees: 25, seed: 2 } },
      { label: 'Small loans as they are', values: { table: 'loans', extra: 'none', n: 300, trees: 25, seed: 4 } },
    ],
    counters: [{ id: 'shuffled', label: 'columns shuffled' }],
    run(inp) {
      const t = T5.withExtra(T5.table(inp.table, inp.n, inp.seed), inp.extra, inp.seed + 77), te = T5.withExtra(T5.table(inp.table, 300, inp.seed + 1000), inp.extra, inp.seed + 1077);
      const d = t.names.length, reps = 5;
      const m = ML().RandomForestClassifier({ nEstimators: inp.trees, maxFeatures: 'sqrt', seed: inp.seed }).fit(t.X, t.y);
      const imp = m.featureImportances, perm = S.Explain.permutationImportance(m, te.X, te.y, { nRepeats: reps, seed: inp.seed, names: t.names });
      const shuf = T5.firstShuffles(te.X, reps, inp.seed);
      const F = new S.Frames(['shuffled']);
      const rowH = Math.min(30, 200 / d), pmax = Math.max(1e-6, ...perm.mean.map((v, j) => v + perm.std[j]));
      const extraCls = (j) => (t.names[j] === 'noise' || /copy$/.test(t.names[j]) ? 'pl-lab hot' : 'tick');
      const left = [H.title('ri:lt', 20, 24, 'impurity importance (training splits)'), ...bars('ri:l', 110, 50, 190, rowH, imp, t.names, { max: Math.max(...imp), ncls: extraCls })];
      const rightHead = [H.title('ri:rt', 420, 24, 'permutation importance (test accuracy lost)')];
      const pvals = new Array(d).fill(null);
      const right = () => [...rightHead, ...bars('ri:r', 510, 50, 170, rowH, pvals, t.names, { max: pmax, neg: true, err: perm.std, cls: () => 'hot', ncls: extraCls })];
      const cw = Math.min(96, 700 / d), ty = 90 + d * rowH;
      const table = (j) => {
        const items = [H.text('ri:tt', 20, ty - 4, j === null ? 'the first test rows' : 'the first test rows, with ' + t.names[j] + ' shuffled', 'pl-lab' + (j === null ? '' : ' hot'), 13)];
        t.names.forEach((nm, c) => items.push({ key: 'ri:h' + c, kind: 'cell', x: H.r2(20 + c * cw), y: ty + 8, w: H.r2(cw - 4), h: 22, rx: 3, text: nm, size: 12, cls: 'p5-col head' + (c === j ? ' shuf' : '') }));
        for (let i = 0; i < 6; i++) t.names.forEach((nm, c) => items.push({ key: 'ri:v' + i + '_' + c, kind: 'cell', x: H.r2(20 + c * cw), y: ty + 34 + i * 24, w: H.r2(cw - 4), h: 21, rx: 3, text: n$(c === j ? shuf[c][i] : te.X[i][c], 3), size: 12, cls: 'p5-col' + (c === j ? ' shuf' : '') }));
        items.push(H.text('ri:y', 20 + d * cw + 6, ty + 25, 'class', 'tick', 12));
        for (let i = 0; i < 6; i++) items.push(H.text('ri:yv' + i, 20 + d * cw + 6, ty + 49 + i * 24, t.classes[te.y[i]], 'pl-lab c' + te.y[i], 12));
        return items;
      };
      const topImp = imp.indexOf(Math.max(...imp));
      F.add({ items: [...left, ...right(), ...table(null)], say: 'A forest of ' + inp.trees + ' trees on ' + t.X.length + ' rows. Summed over its splits, ' + t.names[topImp] + ' did the most to lower the impurity: ' + n$(imp[topImp], 3) + ' of the total.' + (inp.extra === 'noise' || inp.extra === 'both' ? ' Even the noise column gets ' + n$(imp[t.names.indexOf('noise')], 3) + '.' : ''), meta: { phase: 'impurity', trees: inp.trees, n: t.X.length, imp, top: imp[topImp], noise: t.names.includes('noise') ? imp[t.names.indexOf('noise')] : null } });
      const base = perm.trace[0];
      F.add({ items: [...left, ...right(), ...table(null)], say: 'Now the test rows, untouched: the forest gets ' + H.pct(base.baseline) + ' of 300 right. Each column will be shuffled ' + reps + ' times; how far the accuracy falls is that feature’s permutation importance.', meta: { phase: 'baseline', baseline: base.baseline, nte: 300, reps } });
      perm.trace.slice(1).forEach((st, k) => {
        const j = st.feature;
        pvals[j] = perm.mean[j];
        F.set('shuffled', k + 1);
        const isNoise = t.names[j] === 'noise';
        F.add({ items: [...left, ...right(), ...table(j)], say: 'Shuffling ' + t.names[j] + ' ' + reps + ' times costs ' + n$(st.mean * 100, 2) + ' points of accuracy on average (spread ' + n$(st.std * 100, 2) + ')' + (Math.abs(st.mean) < 0.005 ? (isNoise ? ': the forest does not really need the noise, whatever its splits said.' : ': the forest hardly needs it.') : '.'), meta: { phase: 'perm', reps, feature: j, mean: st.mean * 100, std: st.std * 100, raw: st.mean, first: base.baseline - perm.drops[j][0] } });
      });
      const rank = (v) => v.map((x, j) => j).sort((a, b) => v[b] - v[a]);
      const ri = rank(imp), rp = rank(perm.mean);
      let say = 'Impurity ranks ' + H.list(ri.map((j) => t.names[j])) + '; permutation ranks ' + H.list(rp.map((j) => t.names[j])) + '.';
      const ci = t.names.findIndex((x) => /copy$/.test(x)), pair = ci >= 0 ? [perm.mean[0] * 100, perm.mean[ci] * 100] : null;
      if (ci >= 0) {
        // the copy duplicates column 0; say which of the pair the forest can do without, from the measured drops
        const po = perm.mean[0] * 100, pc = perm.mean[ci] * 100, small = (v) => v < 2, pts = (v) => n$(v, 2) + ' points';
        say += ' The near copy of ' + t.names[0] + ' splits the credit with it: ' + (small(po) && small(pc) ? 'shuffling either one alone costs little (' + pts(po) + ' and ' + pts(pc) + '), because the other still carries the information.'
          : small(pc) ? 'shuffling the copy alone costs little (' + pts(pc) + '), because ' + t.names[0] + ' still carries the information; ' + t.names[0] + ' itself still costs ' + pts(po) + '.'
          : small(po) ? 'shuffling ' + t.names[0] + ' alone costs little (' + pts(po) + '), because the copy still carries the information; the copy itself still costs ' + pts(pc) + '.'
          : 'even so, shuffling either one alone still costs ' + pts(po) + ' and ' + pts(pc) + '.');
      }
      F.add({ items: [...left, ...right(), ...table(null)], say, meta: { phase: 'done', ri, rp, pair } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Importance', text: ['impurity: Σ over splits on j of', '  (n_t / n)·(I(t) − weighted I(children))', '  averaged over trees, scaled to sum 1', 'permutation: score(test) −', '  score(test with column j shuffled)'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr, inp) => f.meta.phase === 'baseline' && (inp.extra === 'noise' || inp.extra === 'both'), pick: 'option', ask: 'Shuffling the noise column will cost the forest…', options: ['almost nothing', 'more than any other column', 'about as much as its impurity importance suggests'], answer: (fr) => { const f = fr.find((x) => x.meta.phase === 'perm' && fr[0].meta.noise !== null && x.meta.feature === fr[0].meta.imp.length - 1); return f && Math.abs(f.meta.raw) < 0.02 ? 'almost nothing' : 'about as much as its impurity importance suggests'; }, explain: 'Splits on noise fit the training rows only; on test rows the noise carries no information.' },
      { at: (f, i, fr, inp) => f.meta.phase === 'baseline' && !(inp.extra === 'noise' || inp.extra === 'both'), pick: 'option', ask: 'Will the two methods agree on the most important feature?', options: ['yes', 'no'], answer: (fr) => { const L = fr[fr.length - 1].meta; return L.ri[0] === L.rp[0] ? 'yes' : 'no'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Which importance is measured on data the model did not train on?', options: ['permutation importance (on a test set)', 'impurity importance', 'both', 'neither'], answer: 0, why: 'Impurity importance comes from the training splits themselves.' },
      { q: 'Why can a noise feature get impurity importance?', options: ['trees split on it to fit noise in the training rows', 'noise is always important', 'it was scaled wrongly', 'it cannot'], answer: 0, why: 'A continuous feature offers many thresholds, and some happen to separate training rows.' },
      { q: 'Two near-identical features: what does permutation importance show?', options: ['each looks unimportant alone, because the other covers for it', 'both look twice as important', 'one is always zero and the other full', 'an error'], answer: 0, why: 'Shuffling one leaves the information in the other.' },
      { q: 'A feature has permutation importance slightly below zero. That means…', options: ['shuffling it did not hurt; the small change is chance', 'it is harmful and must be removed', 'the model is broken', 'it is the most important'], answer: 0, why: 'Random shuffles can even help by luck; read it as about zero.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
