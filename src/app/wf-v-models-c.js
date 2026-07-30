// The pictures of the tree families: a decision tree growing split by split (the tree, the plane being cut box by box,
// and the impurity of every threshold the split was chosen from), a forest's first trees beside the vote of all of
// them, AdaBoost's rows growing heavier as they are missed, and gradient boosting climbing stage by stage.
(function (S) {
  'use strict';
  const { h } = S.UI;
  const WF = S.WF, M = S.WMV, P = M.pics;
  const r2 = (v) => Math.round(v * 100) / 100;
  const K = () => S.Kit;
  const tw = (t, fs) => S.geom.textWidth(String(t), fs);

  // ---- the decision tree ------------------------------------------------------------------------------------
  /** a node's words: the question it asks, or the answer it gives */
  const nodeLines = (view, n, I) => {
    const name = I.features[n.f] || 'x' + n.f, clf = Array.isArray(n.v);
    const verdict = clf ? M.cname(view, n.v.indexOf(Math.max(...n.v))) : M.num(n.v);
    const imp = (I.criterion === 'entropy' ? 'entropy ' : clf ? 'gini ' : 'error ') + M.num(n.imp, 2) + ' · ' + n.n;
    return { ask: name + ' ≤ ' + M.num(n.t), verdict, imp };
  };
  /**
   * the tree as it stood after k splits, cut to fit maxLeaves leaves across: the path to the newest split is opened
   * first, then the nodes holding the most rows, so the split being shown is always drawn with its context
   */
  const treeAt = (view, I, k, maxLeaves, whole) => {
    const byId = new Map(I.nodes.map((n) => [n.id, n]));
    const split = new Set(I.grow.slice(0, k).map((g) => g.node));
    const cur = k > 0 && !whole ? I.grow[k - 1].node : -1;
    const parent = new Map();
    for (const n of I.nodes) if (n.l >= 0) { parent.set(n.l, n.id); parent.set(n.r, n.id); }
    const path = new Set();
    for (let x = cur; x >= 0 && byId.has(x); x = parent.has(x) ? parent.get(x) : -1) path.add(x);
    // open nodes one at a time while the leaves fit: the path first, then by rows held
    const open = new Set();
    let leavesNow = 1;
    const queue = [byId.get(0)];
    while (queue.length) {
      queue.sort((a, b) => (path.has(b.id) ? 1 : 0) - (path.has(a.id) ? 1 : 0) || b.n - a.n);
      const n = queue.shift();
      if (!split.has(n.id) || n.l < 0) continue;
      if (leavesNow + 1 > maxLeaves) continue;
      open.add(n.id); leavesNow++;
      queue.push(byId.get(n.l), byId.get(n.r));
    }
    const leaves = (n) => (split.has(n.id) && n.l >= 0 ? leaves(byId.get(n.l)) + leaves(byId.get(n.r)) : 1);
    const make = (n, edge) => {
      const L = nodeLines(view, n, I), grown = split.has(n.id) && n.l >= 0;
      const node = { id: n.id, edge, kids: [], bar: Array.isArray(n.v) ? n.v.map((p) => p * n.n) : null, cls: (n.id === cur ? 'act ' : '') + (grown ? '' : 'leaf ' + (Array.isArray(n.v) ? 'c' + (n.v.indexOf(Math.max(...n.v)) % 4) : '')) };
      if (open.has(n.id)) { node.lines = [L.ask, L.imp]; node.kids = [make(byId.get(n.l), 'yes'), make(byId.get(n.r), 'no')]; }
      else if (grown) { node.lines = ['… ' + leaves(n) + ' leaves', L.imp]; node.cls = 'more'; }
      else node.lines = [L.verdict, L.imp];
      return node;
    };
    return make(byId.get(0), null);
  };
  /** what the tree after k splits says at a point: the class mix (or value) of the node it reaches */
  const reach = (I, split, x) => { const byId = I._byId || (I._byId = new Map(I.nodes.map((n) => [n.id, n]))); let n = byId.get(0); while (split.has(n.id) && n.l >= 0) n = byId.get(x[n.f] <= n.t ? n.l : n.r); return n; };
  /** the box of the plane a node covers */
  const regionOf = (I, id, xr, yr) => {
    const byId = I._byId || (I._byId = new Map(I.nodes.map((n) => [n.id, n])));
    const b = { x0: xr[0], x1: xr[1], y0: yr[0], y1: yr[1] };
    let n = byId.get(id);
    while (n && n.id !== 0) {
      const p = I.nodes.find((q) => q.l === n.id || q.r === n.id);
      if (!p) break;
      const left = p.l === n.id;
      if (p.f === 0) { if (left) b.x1 = Math.min(b.x1, p.t); else b.x0 = Math.max(b.x0, p.t); } else if (left) b.y1 = Math.min(b.y1, p.t); else b.y0 = Math.max(b.y0, p.t);
      n = p;
    }
    return b;
  };
  P.tree = function (view, ctx) {
    const I = view.internals;
    if (!I || !I.nodes) return P.generic(view, ctx);
    const fs = 12;
    const texts = I.nodes.flatMap((n) => { const L = nodeLines(view, n, I); return [L.ask, L.verdict, L.imp]; });
    const boxW = Math.max(84, Math.min(160, Math.max(...texts.map((t) => tw(t, fs))) + 16)), boxH = 50, gap = 10;
    const maxLeaves = Math.max(2, Math.floor((ctx.W - 10) / (boxW + gap)));
    const pl = view.plane, nSplits = I.grow.length;
    const wide = !ctx.narrow && ctx.W >= 720;
    const frames = [];
    for (let k = 0; k <= nSplits + 1; k++) {
      const fin = k === nSplits + 1, kk = Math.min(k, nSplits);
      const root = treeAt(view, I, kk, maxLeaves, fin);
      K().tree.layout(root, { x: 0, y: boxH / 2 + 4, levelH: 80, boxW, gapX: gap });
      const tb = K().tree.bounds(root, boxW, boxH), dx = (ctx.W - tb.w) / 2 - tb.x;
      const shift = (n) => { n.x = r2(n.x + dx); (n.kids || []).forEach(shift); };
      shift(root);
      const items = K().tree.draw(root, { prefix: 'tr', boxW, boxH, size: fs });
      let y = tb.y + tb.h + 30, side = null;
      const g = kk > 0 && !fin ? I.grow[kk - 1] : null;
      // the plane, cut by the splits so far
      if (pl) {
        const split = new Set(I.grow.slice(0, kk).map((s) => s.node)), n = fin ? pl.n : 28;
        const corners = [];
        for (let j = 0; j <= n; j++) { const v = pl.yr[1] - ((pl.yr[1] - pl.yr[0]) * j) / n; for (let i = 0; i <= n; i++) corners.push([pl.xr[0] + ((pl.xr[1] - pl.xr[0]) * i) / n, v]); }
        const vals = [], vals2 = [];
        if (fin) { vals.push(...pl.vals); if (pl.vals2) vals2.push(...pl.vals2); } else for (const c of corners) { const node = reach(I, split, c); if (!Array.isArray(node.v)) vals.push(node.v); else if (pl.scheme === 'prob') vals.push(node.v[1]); else { vals.push(node.v.indexOf(Math.max(...node.v))); vals2.push(Math.max(...node.v)); } }
        const pw = wide ? Math.min(460, ctx.W * 0.5) : ctx.W - 8, box = M.plotBox(pw, { ratio: 0.78, maxH: 340, top: y + 48 });
        const pic = M.plane('tp', pl, box, { vals, vals2: vals2.length ? vals2 : undefined, n, scheme: pl.scheme, score: fin ? pl.score || null : null });
        items.push(...pic.items);
        // every cut so far, the newest lit
        if (!fin) I.grow.slice(0, kk).forEach((s, i) => {
          const b = regionOf(I, s.node, pl.xr, pl.yr), ax = pic.ax;
          const d = s.f === 0 ? `M${ax.X(s.t)},${ax.Y(Math.max(b.y0, pl.yr[0]))} V${ax.Y(Math.min(b.y1, pl.yr[1]))}` : `M${ax.X(Math.max(b.x0, pl.xr[0]))},${ax.Y(s.t)} H${ax.X(Math.min(b.x1, pl.xr[1]))}`;
          items.push({ key: 'tp:cut' + i, kind: 'path', d, cls: 'cut' + (i === kk - 1 ? ' act' : ''), pathAudit: 'edge-x', layer: 'lines' });
        });
        items.push({ key: 'tp:t', kind: 'text', x: box.x, y: y + 6, text: fin ? 'what the grown tree says everywhere' : 'the plane, cut by the splits so far', size: 13, cls: 'axname' });
        y = box.y + box.h + 50;
        if (wide) side = { x: box.x + box.w + 70, y: box.y + 10, w: ctx.W - box.x - box.w - 100, h: Math.min(220, box.h - 60) };
      }
      // the impurity after every threshold of the chosen feature at this split: the curve the split was read from
      const cb = side || { x: 50, y: y + 30, w: Math.min(520, ctx.W - 80), h: 150 };
      if (g && g.curve && g.curve.length > 1 && cb.w > 140) {
        const sp = M.series('ic', cb, [{ key: 'c', pts: g.curve, cls: 'act' }], { names: [I.features[g.f] + ' threshold', 'impurity after'] });
        sp.items.push(K().curve.vline(sp.ax, 'ic:at', g.t, 'hot'));
        sp.items.push({ key: 'ic:t', kind: 'text', x: cb.x - 44, y: cb.y - 30, text: 'node ' + g.node + ': ' + g.tried + ' thresholds tried; the lowest wins', size: 13, cls: 'axname' });
        items.push(...sp.items);
        if (!side) y = cb.y + cb.h + 50;
      }
      const say = fin ? 'The grown tree: ' + I.leaves + ' leaves, each giving ' + (view.task === 'regression' ? 'the mean of its training rows' : 'the class most of its training rows have') + '. The deepest boxes hold only a handful of rows; pruning or a smaller depth would merge them.' : kk === 0 ? 'The root holds every training row. The tree tries every feature and every threshold, and keeps the question that leaves the two groups purest.' : g.say + (g.tried ? ' It weighed ' + g.tried + ' thresholds to find it.' : '');
      frames.push({ items, view: { x: 0, y: 0, w: ctx.W, h: Math.max(y, tb.y + tb.h + 20) }, say, title: fin ? 'the grown tree' : kk === 0 ? 'the root' : 'split ' + kk });
    }
    ctx.el.appendChild(M.facts([['depth', I.depth], ['leaves', I.leaves], ['nodes', I.nodeCount], I.importances && I.importances.length ? ['most used feature', I.features[I.importances.indexOf(Math.max(...I.importances))]] : null]));
    ctx.el.appendChild(M.legend(view) || h('span'));
    M.stepper(ctx.el, frames, { label: 'The tree growing split by split', key: 'tree', mem: ctx.mem, start: frames.length - 1 });
    if (I.grow.length < I.nodeCount / 2 - 0.5) ctx.el.appendChild(M.sub('The first ' + I.grow.length + ' splits are replayed; the tree has ' + Math.floor(I.nodeCount / 2) + ' in all.'));
  };

  // ---- the random forest ---------------------------------------------------------------------------------------------
  P.forest = function (view, ctx) {
    const I = view.internals, pl = view.plane;
    ctx.el.appendChild(M.facts([['trees', I.n], I.oob !== undefined && I.oob !== null ? ['out-of-bag score', M.num(I.oob)] : null]));
    if (pl && view.trees) {
      const cols = ctx.W >= 900 ? 4 : 2, cw = Math.floor((ctx.W - 10) / cols), items = [];
      let maxY = 0;
      const one = (k, title, opts) => {
        const c = k % cols, r = Math.floor(k / cols), box = { x: c * cw + 34, y: r * (Math.round((cw - 50) * 0.8) + 62) + 34, w: cw - 50, h: Math.round((cw - 50) * 0.8) };
        const pic = M.plane('f' + k, pl, box, Object.assign({ r: 2.6 }, opts));
        pic.items.forEach((it) => { if (it.cls === 'tick' || /:t:/.test(it.key)) it.size = 11; });
        items.push(...pic.items.filter((it) => !/:xn$|:yn$/.test(it.key)), { key: 'f' + k + ':title', kind: 'text', x: box.x, y: box.y - 10, text: title, size: 13, cls: 'axname' });
        maxY = Math.max(maxY, box.y + box.h + 30);
      };
      view.trees.forEach((t, k) => one(k, 'tree ' + (k + 1), { vals: t.vals, n: t.n, scheme: 'prob', score: t.vals.map((v) => v - 0.5), levels: [0] }));
      one(view.trees.length, 'all ' + I.n + ' trees voting', {});
      ctx.el.appendChild(M.legend(view) || h('span'));
      M.still(ctx.el, items, { x: 0, y: 0, w: ctx.W, h: maxY }, 'A few trees of the forest and their vote');
      ctx.el.appendChild(M.sub('Each tree saw its own bootstrap sample of the rows and a random few features at each split, so each draws its own jagged boundary; their average is smoother than any of them.'));
    } else P.generic(view, ctx);
    if (I.importances && I.importances.length) {
      const b = M.bars('im', I.features.slice(0, 20), I.importances.slice(0, 20), { x: 0, y: 26, w: ctx.W - 16, size: 13 });
      b.items.push({ key: 'im:t', kind: 'text', x: 0, y: 14, text: ctx.narrow ? 'feature importances' : 'how much each feature cut the impurity, over all the trees', size: 13, cls: 'axname' });
      M.still(ctx.el, b.items, { x: -4, y: 0, w: ctx.W - 8, h: b.h + 36 }, 'Feature importances');
    }
  };

  // ---- AdaBoost: the rows it misses grow heavier --------------------------------------------------------------------
  P.adaboost = function (view, ctx) {
    const I = view.internals, pl = view.plane;
    ctx.el.appendChild(M.facts([['rounds', I.n], ['first error', M.num(I.errors[0])], ['last error', M.num(I.errors[I.errors.length - 1])]]));
    if (!pl || !I.weights || !I.weights.length) return P.generic(view, ctx);
    const L = M.split(ctx, 0), box = L.plot, frames = [];
    I.weights.forEach((w, k) => {
      const rOf = (pk) => { const row = pl.points.idx[pk]; const ww = row < w.length ? w[row] : 1; return Math.max(1.6, Math.min(11, 3.3 * Math.sqrt(ww))); };
      const pic = M.plane('ab', pl, box, { vals: pl.vals.map(() => 0.5), scheme: 'prob', noLevels: true, rOf });
      const s = I.stumps[k];
      if (s && (s.f === 0 || s.f === 1)) pic.items.push({ key: 'ab:cut', kind: 'path', d: s.f === 0 ? `M${pic.ax.X(s.t)},${box.y} V${box.y + box.h}` : `M${box.x},${pic.ax.Y(s.t)} H${box.x + box.w}`, cls: 'cut act', pathAudit: 'edge-x', layer: 'lines' });
      frames.push({ items: pic.items, view: M.viewOf(box), title: 'round ' + (k + 1), say: (I.says[k] || '') + ' A dot’s size is the weight of its row going into this round.' });
    });
    const fin = M.plane('ab', pl, box, {});
    frames.push({ items: fin.items, view: M.viewOf(box), title: 'the vote', say: 'All ' + I.n + ' rounds vote, each with its say α (from `' + M.num(Math.max(...I.alphas)) + '` down to `' + M.num(Math.min(...I.alphas)) + '`): the shading is the result.' });
    ctx.el.appendChild(M.legend(view) || h('span'));
    M.stepper(ctx.el, frames, { label: 'AdaBoost round by round', key: 'ab', mem: ctx.mem, start: frames.length - 1 });
  };

  // ---- gradient boosting: stage by stage -----------------------------------------------------------------------
  P.gboost = function (view, ctx) {
    const I = view.internals, st = view.stages || [];
    ctx.el.appendChild(M.facts([['stages', I.n], ['learning rate', M.num(I.lr)], ['first training loss', M.num(I.trainLoss[0])], ['last', M.num(I.trainLoss[I.trainLoss.length - 1])]]));
    if (!st.length) return P.generic(view, ctx);
    const frames = [], pl = view.plane, ln = view.line;
    const lossBox = (top, left, w) => ({ x: left + 50, y: top, w: Math.max(150, w - 70), h: 150 });
    st.forEach((s) => {
      let items = [], y = 0;
      if (pl) {
        const L = M.split(ctx, 300), pic = M.plane('gb', pl, L.plot, { vals: s.vals, n: s.n, scheme: pl.scheme === 'value' ? 'value' : 'prob', score: pl.scheme === 'value' ? null : s.vals.map((v) => v - 0.5), levels: [0] });
        items = pic.items; y = L.plot.y + L.plot.h + 46;
        const lb = L.side ? lossBox(L.plot.y + 40, L.side.x, L.side.w) : lossBox(y + 30, 0, ctx.W);
        const sp = M.series('gl', lb, [{ key: 'l', pts: I.trainLoss.map((v, i) => [i + 1, v]), cls: 'act' }], { names: ['stage', 'training loss'] });
        sp.items.push(K().curve.vline(sp.ax, 'gl:at', s.k, 'hot'));
        items.push(...sp.items);
        y = Math.max(y, lb.y + lb.h + 46);
      } else if (ln) {
        const box = M.plotBox(ctx.W - 8, { ratio: 0.55, maxH: 320 }), ax = M.axes('gc', box, ln.xr, ln.yr, [ln.name, ln.target]);
        items = ax.items.slice();
        ln.points.X.forEach((p, k) => items.push({ key: 'gc:p' + k, kind: 'dot', x: ax.X(p[0]), y: ax.Y(ln.points.y[k]), r: 3.2, cls: 'sp c0', audit: 'overlap-ok', layer: 'items' }));
        items.push(K().curve.line(ax, 'gc:fit', ln.xs.map((x, i) => [x, s.ys[i]]), 'act'));
        y = box.y + box.h + 46;
      }
      frames.push({ items, view: { x: 0, y: 0, w: ctx.W, h: y }, title: 'stage ' + s.k, say: 'After ' + s.k + ' stage' + (s.k === 1 ? '' : 's') + ': training loss `' + M.num(I.trainLoss[s.k - 1]) + '`. Each new tree fits what the model so far still gets wrong, and joins shrunk by the learning rate `' + M.num(I.lr) + '`.' });
    });
    ctx.el.appendChild(M.legend(view) || h('span'));
    M.stepper(ctx.el, frames, { label: 'Boosting stage by stage', key: 'gb', mem: ctx.mem, start: frames.length - 1 });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
