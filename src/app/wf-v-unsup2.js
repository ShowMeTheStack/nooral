// The viewers of hierarchical clustering (a dendrogram you can cut with a slider, coloured by the clusters the cut
// makes, with the rows in the same colours), the Gaussian mixture (its clouds breathing round by round of EM, with the
// log-likelihood climbing) and the maps (t-SNE and UMAP, replayed from their snapshots).
(function (S) {
  'use strict';
  const { h } = S.UI;
  const V = S.WFV, r2 = (v) => Math.round(v * 100) / 100;
  const C = () => S.Kit.curve;

  /** flat labels from merges [[a, b, dist, size]..] cut at a height */
  const cutAt = (merges, n, height) => {
    const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i), find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    merges.forEach((m, t) => { if (m[2] <= height) { parent[find(m[0])] = n + t; parent[find(m[1])] = n + t; } });
    const roots = new Map();
    return Array.from({ length: n }, (_, i) => { const r = find(i); if (!roots.has(r)) roots.set(r, roots.size); return roots.get(r); });
  };
  /** the leaves of the merge tree in drawing order */
  const leafOrder = (merges, n) => { const out = [], walk = (id) => { if (id < n) out.push(id); else { const m = merges[id - n]; walk(m[0]); walk(m[1]); } }; if (merges.length) walk(n + merges.length - 1); else out.push(0); return out; };

  V.register('hierarchical', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, n = view.n, merges = view.merges, narrow = ctx.narrow;
    const top = merges.length ? merges[merges.length - 1][2] : 1;
    let height = view.cut;
    const facts = h('div');
    const fig = h('div'), sc = h('div');
    const draw = () => {
      const labels = cutAt(merges, n, height), k = new Set(labels).size;
      facts.replaceChildren(V.facts([['linkage', view.linkage], ['cut height', V.num(height, 3)], ['clusters', k], view.silhouette !== null && Math.abs(height - view.cut) < 1e-12 ? ['silhouette', V.num(view.silhouette, 3)] : null, view.ari !== null && Math.abs(height - view.cut) < 1e-12 ? ['agreement with the classes (ARI)', V.num(view.ari, 3)] : null]));
      // the dendrogram: leaves along the bottom, each merge a ⊓ at its height
      const order = leafOrder(merges, n), pos = new Array(2 * n - 1), hgt = new Array(2 * n - 1).fill(0), lab = new Array(2 * n - 1);
      const box = { x: 54, y: 30, w: W - 54 - 20, h: narrow ? 220 : 280 };
      order.forEach((leaf, i) => { pos[leaf] = box.x + ((i + 0.5) * box.w) / n; lab[leaf] = labels[leaf]; });
      const Y = (d) => box.y + box.h - (d / (top * 1.05)) * box.h;
      const items = [];
      const ax = C().axes({ prefix: 'hd', x: box.x, y: box.y, w: box.w, h: box.h, xr: [0, 1], yr: [0, top * 1.05], xticks: 0, yticks: C().ticks(0, top * 1.05, 4), yname: 'merge height (' + view.linkage + ' distance)', grid: false });
      items.push(...ax.items);
      const paths = new Map();
      merges.forEach((m, t) => {
        const id = n + t, a = m[0], b = m[1];
        pos[id] = (pos[a] + pos[b]) / 2; hgt[id] = m[2];
        lab[id] = lab[a] === lab[b] && m[2] <= height ? lab[a] : -1;
        const cls = (c) => (c >= 0 ? 'c' + (c % 8) : 'x');
        const seg = (child) => `M${r2(pos[child])},${r2(Y(hgt[child]))} V${r2(Y(m[2]))}`;
        const add = (c, d) => paths.set(cls(c), (paths.get(cls(c)) || '') + d + ' ');
        add(lab[a] >= 0 && m[2] <= height ? lab[a] : -1, seg(a)); add(lab[b] >= 0 && m[2] <= height ? lab[b] : -1, seg(b));
        add(lab[id], `M${r2(pos[a])},${r2(Y(m[2]))} H${r2(pos[b])}`);
      });
      for (const [c, d] of paths) items.push({ key: 'hd:p' + c, kind: 'path', d, cls: 'wfd-dendro ' + (c === 'x' ? '' : c), audit: 'ignore' });
      items.push({ key: 'hd:cut', kind: 'path', d: `M${box.x},${r2(Y(height))} H${box.x + box.w}`, cls: 'wfd-cut', audit: 'ignore' });
      items.push(V.text('hd:cl', box.x + box.w, Y(height) - 7, 'cut: ' + k + ' cluster' + (k === 1 ? '' : 's'), { anchor: 'end', size: 13, cls: 'wfd-cutlabel' }));
      if (view.y && n <= 120) order.forEach((leaf, i) => items.push({ key: 'hd:l' + i, kind: 'dot', x: r2(pos[leaf]), y: box.y + box.h + 10, r: Math.min(4, (box.w / n) * 0.42), cls: 'sp c' + (view.y[leaf] % 8), audit: 'overlap-ok' }));
      const Hd = box.y + box.h + (view.y && n <= 120 ? 30 : 16);
      if (view.y && n <= 120) items.push(V.text('hd:lt', box.x, Hd + 10, 'the dots under the leaves: each row’s true class', { size: 12, cls: 'tick' }));
      fig.innerHTML = '';
      V.stepper(fig, [{ items, view: V.view(W, Hd + 20) }], { label: 'The dendrogram' });
      // the rows, coloured by the clusters of this cut
      sc.innerHTML = '';
      const pb = V.plotBox(Math.min(W, 720), narrow ? 280 : 320, 48), ax2 = V.axes('hs', view.P, pb, view.names);
      V.stepper(sc, [{ items: ax2.items.concat(V.dots(ax2, 'hs:p', view.P, view.keep.map((i) => labels[i]))), view: V.view(W, pb.y + pb.h + 46) }], { label: 'Rows by cluster' });
    };
    el.appendChild(facts);
    const range = h('input', { type: 'range', min: 0, max: 1000, value: Math.round((1000 * height) / (top * 1.05)), 'aria-label': 'Cut height' });
    const out = h('output', V.num(height, 3));
    const use = h('button.btn.small', { type: 'button' }, 'Use this cut');
    range.addEventListener('input', () => { height = (+range.value / 1000) * top * 1.05; out.textContent = V.num(height, 3); draw(); });
    use.addEventListener('click', () => { if (ctx.api && ctx.api.setParams) ctx.api.setParams({ cut: 'height', height: +height.toPrecision(4) }); });
    el.appendChild(h('div.wfd-row', h('label', 'Cut the tree at', range, out), use));
    el.appendChild(fig);
    el.appendChild(V.title('The rows, coloured by cluster', '(' + view.how + ')'));
    el.appendChild(sc);
    draw();
    el.appendChild(V.note('Each ⊓ joins two groups at the distance where they merged. Long vertical stretches mean well-separated groups: a cut across them is a natural number of clusters.'));
  }));

  // ---- the Gaussian mixture -------------------------------------------------------------------------------------------
  const ellipsePath = (ax, c) => {
    const th = (c.angle * Math.PI) / 180, pts = [];
    for (let i = 0; i <= 48; i++) { const t = (i / 48) * 2 * Math.PI, u = c.x + c.rx * Math.cos(t) * Math.cos(th) - c.ry * Math.sin(t) * Math.sin(th), v = c.y + c.rx * Math.cos(t) * Math.sin(th) + c.ry * Math.sin(t) * Math.cos(th); pts.push((i ? 'L' : 'M') + ax.X(u) + ',' + ax.Y(v)); }
    return pts.join(' ') + ' Z';
  };
  V.register('gmm', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, L = V.clusterLayout(W, ctx.narrow);
    el.appendChild(V.facts([['components', view.k], ['shape', view.cov], ['EM rounds', view.iterations], ['mean log-likelihood', V.num(view.ll, 4)], ['BIC', V.num(view.bic, 5)], view.ari !== null ? ['agreement with the classes (ARI)', V.num(view.ari, 3)] : null]));
    if (!view.twoD) el.appendChild(V.note('The rows and the clouds are drawn on ' + view.how + ': each ellipse is its cloud seen through that projection.'));
    const all = view.P.concat(view.em.flatMap((s) => s.clouds.flatMap((c) => [[c.x - c.rx, c.y - c.rx], [c.x + c.rx, c.y + c.rx]])));
    const pr = all.filter((p) => isFinite(p[0]) && isFinite(p[1]));
    const ax = V.axes('gm', view.P.concat(pr.length > 4 * view.P.length ? [] : pr).slice(0, 4000), L.sc, view.names);
    const lls = view.em.map((s) => s.ll), lo = Math.min(...lls), hi = Math.max(...lls);
    const lax = C().axes({ prefix: 'gl', x: L.side.x, y: L.side.y, w: L.side.w, h: 150, xr: [1, Math.max(2, view.em.length)], yr: [lo - (hi - lo) * 0.08 - 1e-9, hi + (hi - lo) * 0.08 + 1e-9], xticks: C().ticks(1, Math.max(2, view.em.length), 5), yticks: C().ticks(lo, hi, 3), xname: 'EM round', yname: V.fit('mean log-likelihood', L.side.w + 30, 13, true) });
    const H = Math.max(L.sc.y + L.sc.h + 46, L.side.y + 150 + 50);
    const frames = view.em.map((s, k) => {
      const items = [...ax.items];
      items.push(...V.dots(ax, 'gm:p', view.P, view.labels, { cls: (i) => (view.certainty[i] < 0.8 ? 'soft' : '') }));
      s.clouds.forEach((c, j) => { items.push({ key: 'gm:e' + j, kind: 'path', d: ellipsePath(ax, c), cls: 'wfd-cloud c' + (j % 8), audit: 'ignore', layer: 'lines' }); items.push({ key: 'gm:m' + j, kind: 'state', x: ax.X(c.x), y: ax.Y(c.y), w: 14, h: 14, rx: 7, text: '', cls: 'wfd-centre c' + (j % 8), audit: 'overlap-ok' }); });
      items.push(...lax.items, C().line(lax, 'gl:l', lls.slice(0, k + 1).map((v, i) => [i + 1, v]), 'act'), C().dot(lax, 'gl:d', k + 1, s.ll, ''));
      return { items, view: V.view(W, H), title: 'round ' + s.iter, say: 'EM round `' + s.iter + '`: weights `' + s.clouds.map((c) => V.num(c.w, 2)).join(', ') + '`, mean log-likelihood `' + V.num(s.ll, 4) + '`. Faded rows are shared between clouds (no cloud owns them by 80% or more).' };
    });
    V.stepper(el, frames, { label: 'EM, round by round', start: frames.length - 1 });
    el.appendChild(V.table(['component', 'weight', 'rows it owns most'], view.clouds.map((c, j) => ['component ' + j, V.num(c.w, 3), view.sizes[j]])));
    el.appendChild(V.note('Each round can only raise the likelihood, so the curve climbs and flattens. Unlike k-means, a row may belong partly to two clouds, and a cloud may be long and tilted.'));
  }));

  // ---- t-SNE and UMAP --------------------------------------------------------------------------------------------------
  const mapViewer = V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, narrow = ctx.narrow;
    el.appendChild(V.facts([['rows', view.n], ['columns squeezed', view.d + ' → 2'], view.kl !== null ? ['KL divergence', V.num(view.kl, 3)] : null, view.trust !== null ? ['neighbours kept (trustworthiness)', V.num(view.trust, 3)] : null]));
    const box = V.plotBox(Math.min(W, narrow ? W : 760), narrow ? 300 : 420, 24);
    const labels = view.y;
    const frames = view.snaps.map((s, k) => {
      // each snapshot fitted to the box: the map's own units mean nothing, only who sits near whom
      const xr = V.range(s.Y.map((p) => p[0])), yr = V.range(s.Y.map((p) => p[1]));
      const ax = C().axes({ prefix: 'mp', x: box.x, y: box.y, w: box.w, h: box.h, xr, yr, xticks: 0, yticks: 0, frame: 'box', grid: false, xname: view.label + ' 1', yname: view.label + ' 2' });
      const items = ax.items.concat(V.dots(ax, 'mp:p', s.Y, labels));
      if (view.classes && labels) items.push(...V.legend('mp:lg', box.x, box.y + box.h + 66, view.classes.slice(0, 12).map((c, i) => ({ text: String(c), c: i })), W - box.x - 12).items);
      return { items, view: V.view(W, box.y + box.h + 50 + (labels && view.classes ? Math.ceil(view.classes.length / (narrow ? 2 : 5)) * 22 + 22 : 0)), title: s.iter !== null ? (view.kind === 'umap' ? 'epoch ' : 'iteration ') + s.iter : '', say: s.say };
    });
    V.stepper(el, frames, { label: view.label + ', step by step', start: frames.length - 1, hold: 500 });
    el.appendChild(V.note(view.kind === 'tsne' ? 'Rows that are close in the table are pulled close on the map. Sizes of the groups and the gaps between them are not to be trusted; which rows sit together is. The colours are the classes, which the map never saw.' : 'Neighbours in the table stay neighbours on the map, and more of the large-scale shape survives than with t-SNE. The colours are the classes, which the map never saw.'));
  });
  V.register(['tsne', 'umap'], mapViewer);
})(globalThis.Nooral = globalThis.Nooral || {});
