// Part 4, searching fast: the k-d tree search traced box by box (descend to the near side, check a leaf, skip a box
// that lies wholly outside the search ball), and the curse of dimensionality: in many dimensions the nearest and the
// farthest point are almost equally far, so "nearest" stops meaning much.
(function (S) {
  'use strict';
  const H = S.PL, P = S.PL4, ML = () => S.ML;
  const n$ = H.n;

  /** n points uniform in [0, 10]², from a seed */
  const uniform = (n, seed) => H.once('kdu:' + n + ':' + seed, () => { const r = ML().rng(seed); return Array.from({ length: n }, () => [S.Data.round(r() * 10, 2), S.Data.round(r() * 10, 2)]); });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'kd-search', family: 'p04-kd', order: 10, title: 'A k-d tree search, traced',
    blurb: 'A k-d tree splits the points at the median of the widest coordinate, again and again, until each leaf holds a few points. A query walks down to the leaf on its own side first, which gives a good candidate quickly; then, on the way back, any box farther away than the current k-th distance is skipped whole. Most distances are never computed.',
    inputs: [
      { id: 'n', kind: 'number', label: 'points', def: 30, min: 4, max: 60 },
      { id: 'leaf', kind: 'number', label: 'leaf size', def: 3, min: 1, max: 8 },
      { id: 'k', kind: 'number', label: 'k', def: 1, min: 1, max: 5, check: (v, inp) => (v > inp.n ? 'k: at most the number of points' : null) },
      { id: 'seed', kind: 'number', label: 'seed', def: 7, min: 1, max: 9999 },
      { id: 'qx', kind: 'slider', label: 'query x', def: 6.3, min: 0, max: 10, step: 0.1 },
      { id: 'qy', kind: 'slider', label: 'query y', def: 3.6, min: 0, max: 10, step: 0.1 },
    ],
    presets: [
      { label: '30 points, the nearest one', values: { n: 30, leaf: 3, k: 1, seed: 7, qx: 6.3, qy: 3.6 } },
      { label: 'Three nearest', values: { n: 40, leaf: 3, k: 3, seed: 11, qx: 4.2, qy: 5.5 } },
      { label: 'A query near a split', values: { n: 30, leaf: 2, k: 1, seed: 7, qx: 5, qy: 5 } },
      { label: 'Big leaves', values: { n: 40, leaf: 8, k: 2, seed: 4, qx: 2.5, qy: 7.5 } },
      { label: 'A query far outside', values: { n: 24, leaf: 2, k: 1, seed: 9, qx: 10, qy: 0 } },
    ],
    counters: [{ id: 'dist', label: 'distances computed' }, { id: 'skip', label: 'boxes skipped' }],
    run(inp) {
      const X = uniform(inp.n, inp.seed), n = X.length, k = inp.k, q = [inp.qx, inp.qy];
      const T = ML().KDTree(X, { leafSize: inp.leaf }), steps = [], found = T.query(q, k, 2, steps);
      const ax = H.axes({ prefix: 'kd', x: 40, y: 26, w: 430, h: 430, xr: [0, 10], yr: [0, 10], xticks: 2, yticks: 2, xname: 'x₀', yname: 'x₁', grid: false });
      // the tree's nodes in preorder, with a diagram of small dots
      const nodes = [];
      const walk = (nd, parent) => { const id = nodes.length; nodes.push({ nd, id, parent }); const o = { id, kids: [] }; if (nd.left) { o.kids.push(walk(nd.left, id)); o.kids.push(walk(nd.right, id)); } return o; };
      const root = walk(T.root, -1), byLo = new Map(nodes.map((o) => [o.nd.lo, o]));
      const leaves = nodes.filter((o) => !o.nd.left).length, unit = Math.min(20, 250 / leaves), depth = Math.max(...nodes.map((o) => o.nd.depth));
      S.Kit.tree.layout(root, { x: 515, y: 48, levelH: Math.min(40, 160 / Math.max(1, depth)), boxW: unit * 0.6, gapX: unit * 0.4 });
      const pos = []; const put = (o) => { pos[o.id] = [o.x, o.y]; o.kids.forEach(put); }; put(root);
      const R = 515, ty = 48 + Math.min(40, 160 / Math.max(1, depth)) * depth + 30;
      const box = (nd, key, cls) => ({ key, kind: 'path', d: `M${ax.X(nd.lo[0])},${ax.Y(nd.lo[1])} H${ax.X(nd.hi[0])} V${ax.Y(nd.hi[1])} H${ax.X(nd.lo[0])} Z`, cls: 'p4-kd ' + cls, audit: 'ignore', layer: 'back', z: cls === 'seen' ? 2 : 4 });
      const splitLine = (nd, key, cls) => (nd.left ? (nd.dim === 0 ? H.seg(key, ax.X(nd.split), ax.Y(nd.lo[1]), ax.X(nd.split), ax.Y(nd.hi[1]), 'p4-kd split ' + (cls || ''), 'ignore') : H.seg(key, ax.X(nd.lo[0]), ax.Y(nd.split), ax.X(nd.hi[0]), ax.Y(nd.split), 'p4-kd split ' + (cls || ''), 'ignore')) : null);
      const allSplits = nodes.map((o) => splitLine(o.nd, 'kd:s' + o.id, 'soft')).filter(Boolean);
      const F = new S.Frames(['dist', 'skip']);
      const seen = new Set(), pruned = new Set(), checked = new Set();
      let best = [], kth = Infinity;
      const ball = () => (kth < Infinity ? [{ key: 'kd:ball', kind: 'path', d: (() => { const pts = []; for (let i = 0; i <= 72; i++) { const a = (2 * Math.PI * i) / 72; pts.push([Math.max(0, Math.min(10, q[0] + kth * Math.cos(a))), Math.max(0, Math.min(10, q[1] + kth * Math.sin(a)))]); } return pts.map((p, i) => (i ? 'L' : 'M') + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' ') + ' Z'; })(), cls: 'p4-ball', audit: 'ignore', layer: 'back', z: 5 }] : []);
      const treeItems = (cur, curCls) => {
        const it = [];
        nodes.forEach((o) => {
          const cls = o.id === cur ? curCls : pruned.has(o.id) ? 'prune' : seen.has(o.id) ? 'seen' : '';
          it.push({ key: 'kd:t' + o.id, kind: 'state', x: pos[o.id][0], y: pos[o.id][1], r: Math.max(3.5, Math.min(7, unit * 0.32)), text: '', cls: 'p4-tn ' + cls });
          if (o.parent >= 0) it.push({ key: 'kd:te' + o.id, kind: 'edge', from: 'kd:t' + o.parent, to: 'kd:t' + o.id, head: false, cls: 'p4-te' });
        });
        return it;
      };
      const pts = () => S.Kit.scatter.points(ax, 'kd:p', X, X.map(() => 0), { r: 4.5, cls: (i) => 'cn' + (best.includes(i) ? ' act' : checked.has(i) ? ' big' : '') });
      const probe = { key: 'kd:q', kind: 'dot', x: ax.X(q[0]), y: ax.Y(q[1]), r: 6.5, cls: 'pl-probe act', audit: 'overlap-ok', layer: 'items', z: 6 };
      const panel = (lines) => P.eqs('kd:eq', R, ty, lines, 14, 25);
      const frame = (cur, curCls, curBox, extra, lines, say, meta) => F.add({ items: [...ax.items, ...nodes.filter((o) => seen.has(o.id) && !o.nd.left).map((o) => box(o.nd, 'kd:v' + o.id, 'seen')), ...nodes.filter((o) => pruned.has(o.id)).map((o) => box(o.nd, 'kd:x' + o.id, 'prune')), ...(curBox ? [box(curBox, 'kd:cur', curCls)] : []), ...allSplits, ...(extra || []), ...ball(), ...pts(), probe, ...treeItems(cur, curCls), ...panel(lines)], say, meta: Object.assign({ kth: kth === Infinity ? 0 : kth, dist: checked.size, n, k }, meta) });
      frame(-1, '', null, [], [['query (' + n$(q[0]) + ', ' + n$(q[1]) + ')', ''], ['k = ' + k + ', leaf size ' + inp.leaf, 'muted'], [leaves + ' leaves, depth ' + depth, 'muted']], 'The tree splits ' + n + ' points into ' + leaves + ' leaves of at most ' + inp.leaf + ' (each dashed line is a median split of the wider side). The query at (' + n$(q[0]) + ', ' + n$(q[1]) + ') wants its ' + H.plural(k, 'nearest point') + '.', { phase: 'start', leaves, depth });
      for (const st of steps) {
        const o = byLo.get(st.lo);
        if (st.kind === 'descend') {
          seen.add(o.id);
          frame(o.id, 'act', o.nd, [splitLine(o.nd, 'kd:hs', 'act')], [['depth ' + o.nd.depth + ': x' + H.sub(st.dim) + ' = ' + n$(q[st.dim]) + ' vs ' + n$(st.split), 'act'], ['near side first', 'muted'], ['distances so far: ' + checked.size, '']], st.say, { phase: 'descend', depth: o.nd.depth, split: st.split, qd: q[st.dim] });
        } else if (st.kind === 'leaf') {
          seen.add(o.id);
          o.nd.idx.forEach((i) => checked.add(i));
          best = st.best.slice();
          F.set('dist', checked.size);
          kth = best.length >= k ? Math.max(...best.map((i) => ML().dist(X[i], q))) : Infinity;
          frame(o.id, 'leaf', o.nd, [], [['leaf: ' + H.plural(o.nd.n, 'point') + ' checked', 'good'], ['k-th distance ' + (kth < Infinity ? n$(kth) : '∞'), 'act'], ['distances so far: ' + checked.size, '']], st.say.replace(/Infinity/, 'still unbounded') + (kth < Infinity ? ' The circle is the search ball: only boxes that reach inside it can hold anything nearer.' : ''), { phase: 'leaf', depth: o.nd.depth, npts: o.nd.n });
        } else {
          pruned.add(o.id);
          F.count('skip');
          let under = 0; const cnt = (nd) => { under += nd.left ? 0 : nd.n; if (nd.left) { cnt(nd.left); cnt(nd.right); } }; cnt(o.nd);
          const gap = (() => { let s = 0; for (let j = 0; j < 2; j++) { const g = q[j] < st.lo[j] ? st.lo[j] - q[j] : q[j] > st.hi[j] ? q[j] - st.hi[j] : 0; s += g * g; } return Math.sqrt(s); })();
          frame(o.id, 'prune', null, [], [['box ' + n$(gap) + ' away > ' + n$(kth), 'bad'], [H.plural(under, 'point') + ' never measured', 'muted'], ['distances so far: ' + checked.size, '']], st.say + ' Its ' + H.plural(under, 'point') + ' are never measured.', { phase: 'prune', depth: o.nd.depth, gap, under });
        }
      }
      const brute = X.map((p, i) => [ML().dist(p, q), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k);
      const same = brute.every((b, t) => b[1] === found.idx[t]);
      best = found.idx.slice();
      frame(-1, '', null, [], [['found: ' + found.dist.map((v) => n$(v)).join(', '), 'good'], [checked.size + ' of ' + n + ' distances computed', 'act'], [same ? 'the same as brute force' : 'brute force differs', same ? 'good' : 'bad']], 'Done: the ' + H.plural(k, 'nearest point') + ' at ' + H.list(found.dist.map((v) => n$(v))) + ', found after ' + checked.size + ' of the ' + n + ' distances (' + H.pct(checked.size / n) + ')' + (pruned.size ? ', with ' + H.plural(pruned.size, 'box', 'boxes') + ' skipped whole.' : '.') + (same ? ' Brute force over all ' + n + ' gives the same answer.' : ''), { phase: 'done', found: found.dist, frac: checked.size / n, skipped: pruned.size, same: same ? 1 : 0 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'k-d tree search', text: ['build: split at the median of the widest side', '  until a node holds ≤ leaf size points', 'query: go to the near child first;', 'skip a box if its distance to the query', '  exceeds the current k-th distance', 'about log n boxes in low dimensions'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'start', pick: 'option', ask: 'Will the search skip at least one box?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.skipped > 0 ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A box is skipped when…', options: ['its nearest possible point is farther than the current k-th distance', 'it holds fewer than k points', 'it is on the far side of a split', 'the query is outside it'], answer: 0, why: 'Nothing inside it could beat the k neighbours already found.' },
      { q: 'Why search the near side of a split first?', options: ['it probably holds the nearest points, which shrinks the ball early', 'it is always smaller', 'the far side is never searched', 'to save memory'], answer: 0, why: 'A small search ball makes later boxes easy to skip.' },
      { q: 'Does a k-d tree give the same neighbours as brute force?', options: ['yes, exactly', 'only approximately', 'only for k = 1', 'only in two dimensions'], answer: 0, why: 'It only skips boxes that provably hold nothing nearer.' },
      { q: 'Where do k-d trees stop helping?', options: ['in many dimensions, where few boxes can be skipped', 'with many points', 'with small k', 'with integer features'], answer: 0, why: 'In high dimensions the ball reaches into almost every box.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'curse-dim', family: 'p04-kd', order: 20, title: 'The curse of dimensionality',
    blurb: 'Scatter points at random in a unit cube and measure their distances to a query point. In two dimensions some points are close and some far. In hundreds of dimensions every distance is nearly the same: the histogram of distance divided by the largest distance crowds toward 1, and the gap between the nearest and the farthest point shrinks. "Nearest" stops telling points apart.',
    inputs: [
      { id: 'n', kind: 'number', label: 'points', def: 300, min: 20, max: 1000 },
      { id: 'dims', kind: 'list', label: 'dimensions', def: ['1', '2', '5', '10', '50', '200', '1000'], min: 2, max: 9, check: (v) => { for (const s of v) { const d = +s; if (!Number.isInteger(d) || d < 1 || d > 2000) return 'dimensions: whole numbers from 1 to 2000'; } return null; } },
      { id: 'seed', kind: 'number', label: 'seed', def: 1, min: 1, max: 9999 },
    ],
    presets: [
      { label: '300 points, 1 to 1000 dimensions', values: { n: 300, dims: ['1', '2', '5', '10', '50', '200', '1000'], seed: 1 } },
      { label: 'Few points', values: { n: 30, dims: ['2', '10', '100', '1000'], seed: 4 } },
      { label: 'Low dimensions only', values: { n: 500, dims: ['1', '2', '3', '4', '6', '8'], seed: 2 } },
      { label: 'Very high', values: { n: 200, dims: ['10', '100', '500', '2000'], seed: 3 } },
    ],
    counters: [{ id: 'd', label: 'dimensions' }],
    run(inp) {
      const dims = [...new Set(inp.dims.map(Number))].sort((a, b) => a - b), n = inp.n;
      const hx = { x: 50, y: 40, w: 400, h: 260 }, cx = { x: 530, y: 40, w: 220, h: 220 };
      const rows = [];
      const stats = dims.map((d) => H.once('cd:' + [n, d, inp.seed].join(':'), () => {
        const r = ML().rng(inp.seed * 7919 + d), q = Array.from({ length: d }, () => r());
        const ds = Array.from({ length: n }, () => ML().dist(Array.from({ length: d }, () => r()), q));
        const mn = Math.min(...ds), mx = Math.max(...ds), mean = ML().mean(ds), sd = Math.sqrt(ML().mean(ds.map((v) => (v - mean) ** 2)));
        return { d, ds, mn, mx, mean, sd, contrast: (mx - mn) / mn, ratio: mn / mx };
      }));
      const cmax = Math.max(...stats.map((s) => s.contrast)), lg = (v) => Math.log10(v);
      const yTop = Math.ceil(lg(cmax) + 0.01), yBot = Math.min(-1, Math.floor(lg(Math.min(...stats.map((s) => s.contrast)))));
      const cax = H.axes({ prefix: 'cc', x: cx.x, y: cx.y, w: cx.w, h: cx.h, xr: [0, Math.max(1, lg(Math.max(...dims)) * 1.06)], yr: [yBot, yTop], xticks: [0, 1, 2, 3].filter((v) => v <= Math.max(1, lg(Math.max(...dims)) * 1.06)), yticks: 1, fmtX: (v) => String(Math.round(Math.pow(10, v))), fmtY: (v) => String(+Math.pow(10, v).toPrecision(2)), xname: 'dimensions', yname: '(far − near) / near', grid: true });
      const F = new S.Frames(['d']);
      stats.forEach((st, s) => {
        F.set('d', st.d);
        const bins = 25, cnt = new Array(bins).fill(0);
        st.ds.forEach((v) => cnt[Math.min(bins - 1, Math.floor((v / st.mx) * bins))]++);
        const ymax = Math.max(...cnt), yt = ymax > 100 ? 50 : ymax > 40 ? 20 : ymax > 15 ? 5 : 2;
        const ax = H.axes({ prefix: 'cd', x: hx.x, y: hx.y, w: hx.w, h: hx.h, xr: [0, 1], yr: [0, Math.ceil(ymax / yt) * yt], xticks: 0.2, yticks: yt, xname: 'distance / largest distance', yname: 'points', grid: true });
        const items = [...ax.items];
        cnt.forEach((c, b) => { if (c) items.push({ key: 'cd:b' + b, kind: 'cell', x: H.r2(ax.X(b / bins) + 1), y: ax.Y(c), w: H.r2(hx.w / bins - 2), h: H.r2(ax.Y(0) - ax.Y(c)), rx: 1, text: '', cls: 'p4-hist', audit: 'ignore' }); });
        items.push(S.Kit.curve.vline(ax, 'cd:mn', st.ratio, 'act'));
        rows.push(st);
        items.push(...cax.items, S.Kit.curve.line(cax, 'cc:l', rows.map((r) => [lg(r.d), lg(r.contrast)]), 'act'), ...rows.map((r, i) => S.Kit.curve.dot(cax, 'cc:d' + i, lg(r.d), lg(r.contrast), i === s ? '' : 'muted', i === s ? 5 : 3.5)));
        items.push(...P.eqs('cd:eq', 50, 364, [[st.d + (st.d === 1 ? ' dimension' : ' dimensions') + ', ' + n + ' points', 'act'], ['nearest ' + n$(st.mn) + ', farthest ' + n$(st.mx) + ', mean ' + n$(st.mean), ''], ['nearest / farthest = ' + n$(st.ratio) + '   (dashed line)', ''], ['spread / mean = ' + n$(st.sd / st.mean), 'muted']], 15, 28));
        const say = st.d + (st.d === 1 ? ' dimension' : ' dimensions') + ': the nearest of ' + n + ' random points is ' + n$(st.mn) + ' from the query and the farthest ' + n$(st.mx) + ', a ratio of ' + n$(st.ratio) + '. ' +
          (s === 0 ? 'Distances spread over the whole range.' : st.ratio > 0.5 ? 'The histogram has bunched up near the right: almost every point is about as far as the farthest.' : 'The distances are bunching up as the dimension grows.') +
          ' The relative gap (far − near) / near is ' + n$(st.contrast) + '.';
        F.add({ items, say, meta: { phase: 'dim', d: st.d, n, mn: st.mn, mx: st.mx, mean: st.mean, ratio: st.ratio, contrast: st.contrast, cv: st.sd / st.mean } });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Distance concentration', text: ['x, q uniform in [0, 1]ᵈ', 'E‖x − q‖² = d/6, so the distance grows like √d', 'but its spread stays about the same,', 'so (max − min) / min → 0 as d grows', 'k-NN and k-d trees suffer; reduce d first'].join('\n') }] } };
    },
    predict: [
      { at: (f, i) => i === 0, pick: 'option', ask: 'In the highest dimension, will the nearest point be less than half as far as the farthest?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.ratio < 0.5 ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'In a 1000-dimensional unit cube, distances between random points…', options: ['are all nearly the same', 'are mostly near 0', 'vary from 0 to the diagonal evenly', 'cannot be computed'], answer: 0, why: 'They concentrate around √(d/6) with a spread that does not grow.' },
      { q: 'Why does this hurt k-NN?', options: ['the nearest neighbours are hardly nearer than any other point', 'distances overflow', 'k must equal d', 'votes tie'], answer: 0, why: 'Neighbours chosen by tiny differences in distance carry little information.' },
      { q: 'What helps in high dimensions?', options: ['reducing or selecting features first', 'a larger k only', 'Chebyshev distance', 'more decimal places'], answer: 0, why: 'Real data often lie near a lower-dimensional surface, where distances mean something again.' },
      { q: 'Roughly how does the typical distance between random points in [0, 1]ᵈ grow?', options: ['like √d', 'like d', 'like log d', 'it stays near 1'], answer: 0, why: 'The squared distance is a sum of d terms with mean 1/6.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
