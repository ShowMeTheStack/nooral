// A CART tree growing split by split on the moons. For the node being split, every candidate threshold on x and on
// y is tried: the weighted Gini impurity of the two halves is drawn as a curve along each axis of the scatter, and
// the lowest point of the two curves is the split taken. The tree grows beside the scatter, which is shaded by the
// leaves' majority class.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, SC = () => S.Kit.scatter, TR = () => S.Kit.tree, box = D.box;
  const F = ['x', 'y'];
  D.add({
    id: 'cart-grow', order: 20, tag: 'Decision tree · CART', short: 'Tree growing', title: 'A CART tree growing split by split',
    blurb: 'At each node every threshold halfway between two neighbouring values is tried, on x and on y. The curves along the axes are the weighted Gini impurity of the two halves each threshold would make; the lowest point wins.',
    frames({ narrow }) {
      const data = D.moons(), T = S.Toy;
      const xr = [-1.6, 2.6], yr = [-1.15, 1.6];
      const tree = D.once('cart3', () => T.cart(data.X, data.y, { maxDepth: 3, box: [xr, yr] }));
      const L = narrow
        ? { view: box(-6, -10, 392, 760), sc: { x: 30, y: 74, w: 290, h: 200 }, sx: 52, tree: { x: 4, y: 400 }, bw: 60, bh: 46, gap: 4, lv: 82, fs: 12, short: true }
        : { view: box(-10, -14, 1180, 480), sc: { x: 40, y: 104, w: 420, h: 290 }, sx: 70, tree: { x: 572, y: 60 }, bw: 92, bh: 62, gap: 10, lv: 116, fs: 12.5, short: false };
      const ax = C().axes(Object.assign({ prefix: 'ct', xr, yr, xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1], xname: narrow ? '' : 'x', grid: false }, L.sc));
      const made = new Set();
      // a node's box for the tree diagram
      const label = (n, active) => {
        const c = n.counts, lines = [];
        if (n.split && made.has(n.id)) lines.push(F[n.split.f] + ' ≤ ' + D.fix(n.split.t));
        else lines.push(c[0] > c[1] ? 'blue' : c[0] < c[1] ? 'orange' : 'tie');
        if (!L.short) lines.push('gini ' + n.gini.toFixed(3));
        lines.push(c[0] + ' | ' + c[1]);
        return { id: n.id, lines, bar: c, cls: (made.has(n.id) ? '' : 'leaf c' + (c[0] >= c[1] ? 0 : 1)) + (active ? ' act' : ''), kids: made.has(n.id) ? n.kids.map((k, i) => label(k, false)) : [], edge: '' };
      };
      const diagram = (activeId) => {
        const root = label(tree.root, false);
        const mark = (n) => { if (n.id === activeId) n.cls += ' act'; (n.kids || []).forEach((k, i) => { k.edge = i ? 'no' : 'yes'; mark(k); }); };
        mark(root);
        TR().layout(root, { x: L.tree.x, y: L.tree.y, levelH: L.lv, boxW: L.bw, gapX: L.gap });
        // centre the tree in its space
        const b = TR().bounds(root, L.bw, L.bh), want = narrow ? 192 : 572 + 290, dx = want - (b.x + b.w / 2);
        const shift = (n) => { n.x = Math.round((n.x + dx) * 100) / 100; (n.kids || []).forEach(shift); };
        shift(root);
        return TR().draw(root, { prefix: 'cg', boxW: L.bw, boxH: L.bh, size: L.fs, edgeSize: 12 });
      };
      const region = () => {
        const leafOf = (p) => T.cartLeaf(tree.root, p, made);
        return SC().field(ax, 'ct:field', (u, v) => { const n = leafOf([u, v]), tot = n.counts[0] + n.counts[1]; const k = n.counts[0] >= n.counts[1] ? 0 : 1; return [k, Math.abs(n.counts[0] - n.counts[1]) / tot]; }, { n: narrow ? 70 : 100, scheme: 'classes', smooth: false });
      };
      const splitLines = () => {
        let d = '';
        const walk = (n) => { if (!n.kids || !made.has(n.id)) return; const [bx, by] = n.box, t = n.split.t; d += n.split.f === 0 ? `M${ax.X(t)},${ax.Y(Math.min(by[1], yr[1]))} V${ax.Y(Math.max(by[0], yr[0]))} ` : `M${ax.X(Math.max(bx[0], xr[0]))},${ax.Y(t)} H${ax.X(Math.min(bx[1], xr[1]))} `; n.kids.forEach(walk); };
        walk(tree.root);
        return d ? [{ key: 'ct:splits', kind: 'path', d: d.trim(), cls: 'level', pathAudit: 'edge-x' }] : [];
      };
      // the Gini curves for one node: along the top for thresholds on x, along the right for thresholds on y
      const curves = (step) => {
        const items = [], g0 = 0.1, gmax = 0.5, top = L.sc.y - 12, H = L.sx - 16, right = L.sc.x + L.sc.w + 12, W = L.sx - 16;
        const sc = (g) => (Math.max(g0, Math.min(gmax, g)) - g0) / (gmax - g0);
        items.push({ key: 'gx:base', kind: 'path', d: `M${L.sc.x},${top} h${L.sc.w}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        items.push({ key: 'gy:base', kind: 'path', d: `M${right},${L.sc.y} v${L.sc.h}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        items.push({ key: 'gx:name', kind: 'text', x: L.sc.x, y: top - H - 10, text: narrow ? 'Gini of each cut on x' : 'weighted Gini of each cut on x', size: 12, cls: 'axname' });
        items.push({ key: 'gy:name', kind: 'text', x: right + W, y: L.sc.y + L.sc.h + 32, text: 'cuts on y', size: 12, anchor: 'end', cls: 'axname' });
        if (!step) return items;
        for (const f of [0, 1]) {
          const cs = step.cands.filter((c) => c.f === f).sort((a, b) => a.t - b.t);
          if (!cs.length) continue;
          const pt = (c) => (f === 0 ? [ax.X(c.t), top - H * sc(c.g)] : [right + W * sc(c.g), ax.Y(c.t)]);
          const d = cs.map((c, i) => (i ? 'L' : 'M') + pt(c).map((v) => Math.round(v * 10) / 10).join(',')).join(' ');
          const win = step.best.f === f;
          items.push({ key: 'g' + f + ':c', kind: 'path', d, cls: 'series thin ' + (win ? 'act' : 'muted'), pathAudit: 'edge-x' });
          const m = cs.reduce((a, b) => (b.g < a.g ? b : a));
          const p = pt(m);
          items.push({ key: 'g' + f + ':min', kind: 'dot', x: p[0], y: p[1], r: 4.5, cls: 'mark ' + (win ? '' : 'muted'), point: true });
          const txt = (win ? 'best ' : '') + F[f] + ' ≤ ' + D.fix(m.t) + ': ' + m.g.toFixed(3);
          if (f === 0) items.push({ key: 'g0:best', kind: 'text', x: L.sc.x + L.sc.w, y: top - H - 10, text: txt, anchor: 'end', size: 12, cls: win ? 'act' : 'axname' });
          else items.push({ key: 'g1:best', kind: 'text', x: right + W, y: L.sc.y + L.sc.h + 50, text: txt, anchor: 'end', size: 12, cls: win ? 'act' : 'axname' });
        }
        return items;
      };
      const frames = [];
      const scatter = (focus) => SC().points(ax, 'ct:p', data.X, data.y, { r: narrow ? 3.2 : 4, cls: (i) => (focus && !focus.has(i) ? 'dim' : '') });
      frames.push({ view: L.view, items: [...ax.items, region(), ...scatter(), ...curves(null), ...diagram(-1)], say: `The root holds all \`120\` points, \`60\` of each class: Gini impurity \`1 − 0.5² − 0.5² = 0.500\`. A leaf predicts its majority; here it is a tie.` });
      tree.steps.forEach((st, k) => {
        const n = st.node, focus = new Set(n.idx), best = st.best;
        const other = st.cands.filter((c) => c.f !== best.f).reduce((a, b) => (b.g < a.g ? b : a), { g: 1 });
        frames.push({ view: L.view, items: [...ax.items, region(), ...splitLines(), ...scatter(focus), ...curves(st), ...diagram(n.id)], say: `Node of \`${n.idx.length}\` points (\`${n.counts[0]}\` blue, \`${n.counts[1]}\` orange, Gini \`${n.gini.toFixed(3)}\`): ${st.cands.length} candidate cuts. The best on ${F[best.f]} is \`${F[best.f]} ≤ ${D.fix(best.t)}\` with weighted Gini \`${best.g.toFixed(3)}\`; the best on ${F[1 - best.f]} only reaches \`${other.g.toFixed(3)}\`.` });
        made.add(n.id);
        const [a, b] = n.kids;
        frames.push({ view: L.view, items: [...ax.items, region(), ...splitLines(), ...scatter(focus), ...curves(null), ...diagram(n.id)], say: `Split: \`${a.counts[0]} | ${a.counts[1]}\` go left (Gini \`${a.gini.toFixed(3)}\`), \`${b.counts[0]} | ${b.counts[1]}\` go right (Gini \`${b.gini.toFixed(3)}\`). The impurity falls from \`${n.gini.toFixed(3)}\` to \`${best.g.toFixed(3)}\`` + (k === tree.steps.length - 1 ? '. The tree has reached its depth limit of 3.' : '.') });
      });
      let ok = 0;
      data.X.forEach((p, i) => { const n = T.cartLeaf(tree.root, p, made); if ((n.counts[0] >= n.counts[1] ? 0 : 1) === data.y[i]) ok++; });
      frames.push({ view: L.view, items: [...ax.items, region(), ...splitLines(), ...scatter(), ...curves(null), ...diagram(-1)], say: `The grown tree: ${tree.steps.length} splits, ${tree.steps.length + 1} leaves, \`${ok}\` of \`120\` points (\`${D.pct(ok / 120)}\`) on the side of their leaf’s majority. Every boundary is a straight cut along an axis: a staircase round the moons.` });
      return frames;
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
