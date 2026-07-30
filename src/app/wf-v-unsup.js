// The viewers of k-means and DBSCAN. k-means replays its rounds: every row joins its nearest centre (on a 2-D table
// the plane is shaded by the region each centre owns), then each centre moves to the mean of its rows, leaving a
// trail; beside it the inertia falls round by round, and the elbow shows the inertia for k = 1 to 8. DBSCAN shows
// core points, border points and noise, with the ε circle round some core points, cluster by cluster.
(function (S) {
  'use strict';
  const V = S.WFV, r2 = (v) => Math.round(v * 100) / 100;
  const SC = () => S.Kit.scatter, C = () => S.Kit.curve;

  /** the layout of a clustering picture: the scatter, and a column (or a band under it) for the small plots */
  V.clusterLayout = function (W, narrow) {
    if (narrow) { const sc = V.plotBox(W, 300, 44); return { sc, side: { x: 60, y: sc.y + sc.h + 70, w: W - 80 }, stacked: true }; }
    const sw = Math.round(Math.min(W * 0.6, 720)), sc = V.plotBox(sw, 400, 48);
    return { sc, side: { x: sw + 56, y: 40, w: Math.min(W - sw - 80, 420) }, stacked: false };
  };
  const cname = (j) => 'cluster ' + j;

  V.register('kmeans', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, L = V.clusterLayout(W, ctx.narrow);
    el.appendChild(V.facts([['k', view.k], ['rounds', view.iterations], ['inertia', V.num(view.inertia, 4)], view.silhouette !== null ? ['silhouette', V.num(view.silhouette, 3)] : null, view.ari !== null ? ['agreement with the classes (ARI)', V.num(view.ari, 3)] : null]));
    if (!view.twoD) el.appendChild(V.note('This table has ' + (view.names[0] === 'PC1' ? 'more than two columns' : 'one column') + ': the rows are drawn on ' + view.how + ', so a centre’s region is not a straight-edged patch here. The clustering itself used every column.'));
    const all = view.P.concat(view.steps.flatMap((s) => s.centers));
    const ax = V.axes('km', all, L.sc, view.names);
    const steps = view.steps, inertias = steps.map((s) => s.inertia);
    const imax = Math.max(...inertias, 1e-9);
    const plotW = L.side.w, ih = 140;
    const iax = C().axes({ prefix: 'ki', x: L.side.x, y: L.side.y, w: plotW, h: ih, xr: [1, Math.max(2, steps.length)], yr: [0, imax * 1.08], xticks: C().ticks(1, Math.max(2, steps.length), Math.min(6, steps.length)), yticks: C().ticks(0, imax * 1.08, 3), xname: 'half-round', yname: V.fit('inertia (sum of squared distances)', plotW + 30, 13, true), fmtY: V.tick });
    const elbowY = L.side.y + ih + 90;
    const eax = view.elbow ? C().axes({ prefix: 'ke', x: L.side.x, y: elbowY, w: plotW, h: 120, xr: [1, view.elbow.length], yr: [0, Math.max(...view.elbow) * 1.08], xticks: view.elbow.map((_, i) => i + 1), yticks: C().ticks(0, Math.max(...view.elbow) * 1.08, 3), xname: 'k', yname: V.fit('the elbow: inertia for each k', plotW + 30, 13, true), fmtY: V.tick }) : null;
    const H = Math.max(L.sc.y + L.sc.h + 46, (eax ? elbowY + 120 : L.side.y + ih) + 44);
    const centreItems = (Cs, k0, trailUpto) => {
      const out = [];
      Cs.forEach((c, j) => {
        const trail = [view.start ? view.start[j] : null].concat(steps.slice(0, trailUpto + 1).filter((s) => s.kind === 'update').map((s) => s.centers[j])).filter(Boolean);
        if (trail.length > 1) out.push({ key: 'km:tr' + j, kind: 'path', d: trail.map((p, i) => (i ? 'L' : 'M') + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' '), cls: 'wfd-trail', pathAudit: 'edge-x', layer: 'lines' });
        out.push({ key: 'km:c' + j, kind: 'state', x: ax.X(c[0]), y: ax.Y(c[1]), w: 20, h: 20, rx: 4, text: '', cls: 'wfd-centre c' + (j % 8), audit: 'overlap-ok' });
      });
      return out;
    };
    const frame = (k) => {
      const st = steps[k], items = [...ax.items];
      const Cs = st ? (st.kind === 'assign' ? st.centers : steps[k - 1] ? steps[k - 1].centers : view.start) : view.start;
      const labels = st ? st.labels : null;
      if (st && view.twoD) items.push(SC().field(ax, 'km:f', (u, v) => { let b = 0, bd = Infinity; Cs.forEach((c, j) => { const d = (u - c[0]) ** 2 + (v - c[1]) ** 2; if (d < bd) { bd = d; b = j; } }); return [b % 8, 0.5]; }, { n: ctx.narrow ? 90 : 140, scheme: 'classes', smooth: false }));
      items.push(...V.dots(ax, 'km:p', view.P, labels, labels ? null : { cls: () => 'cn' }));
      items.push(...centreItems(st ? st.centers : view.start || [], k, k));
      items.push(...iax.items);
      if (k >= 0) {
        const pts = inertias.slice(0, k + 1).map((v, i) => [i + 1, v]);
        items.push(C().line(iax, 'ki:l', pts, 'act'));
        items.push(C().dot(iax, 'ki:d', k + 1, inertias[k], ''));
      }
      if (eax) {
        items.push(...eax.items, C().line(eax, 'ke:l', view.elbow.map((v, i) => [i + 1, v]), 'c0'));
        view.elbow.forEach((v, i) => items.push(C().dot(eax, 'ke:d' + i, i + 1, v, i + 1 === view.k ? '' : 'c0', i + 1 === view.k ? 5.5 : 3.5)));
      }
      const sizes = labels ? Array.from({ length: view.k }, (_, j) => labels.filter((l) => l === j).length) : null;
      let say;
      if (k < 0) say = 'The ' + view.k + ' first centres, chosen by ' + (view.start ? 'the starting rule' : 'k-means') + '. No row belongs to a centre yet.';
      else if (st.kind === 'assign') say = 'Round ' + (st.iter + 1) + ', assign: every row joins its nearest centre. Inertia `' + V.num(st.inertia, 4) + '`' + (sizes ? '; the drawn rows split `' + sizes.join(', ') + '`.' : '.');
      else say = 'Round ' + (st.iter + 1) + ', move: each centre goes to the mean of its rows; together they moved `' + V.num(st.shift, 3) + '`' + (k === steps.length - 1 ? '. Nothing more changes: k-means has converged.' : '.');
      return { items, view: V.view(W, H), say, title: k < 0 ? 'start' : st.kind };
    };
    const frames = [];
    if (view.start) frames.push(frame(-1));
    steps.forEach((_, k) => frames.push(frame(k)));
    V.stepper(el, frames, { label: 'k-means, round by round', start: frames.length - 1 });
    const lg = view.sizes.map((n, j) => [cname(j), n]);
    el.appendChild(V.table(['cluster', 'rows'], lg));
    el.appendChild(V.note('k-means always finds k groups, whether or not the data has them. Read the elbow: past the k where the inertia stops falling fast, more centres only split real groups.'));
  }));

  // ---- DBSCAN ------------------------------------------------------------------------------------------------------
  V.register('dbscan', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, L = V.clusterLayout(W, ctx.narrow);
    el.appendChild(V.facts([['clusters', view.clusters], ['core points', view.core], ['border points', view.border], ['noise', view.noise, view.noise ? 'bad' : ''], ['ε', view.eps], ['minPts', view.minPts], view.ari !== null ? ['agreement with the classes (ARI)', V.num(view.ari, 3)] : null]));
    if (!view.twoD) el.appendChild(V.note('The rows are drawn on ' + view.how + '; the neighbourhoods of radius ε were measured with every column, so no ε circle is drawn.'));
    const ax = V.axes('db', view.P, L.sc, view.names);
    const H = L.sc.y + L.sc.h + 50;
    if (L.stacked) L.side.y = H + 50;
    // ε circles round a few core points, spread across the clusters
    const ring = [];
    if (view.twoD) {
      const seen = new Set();
      view.kinds.forEach((k, i) => { if (k === 'core' && ring.length < 6 && !seen.has(view.labels[i])) { seen.add(view.labels[i]); ring.push(i); } });
    }
    const rx = view.twoD ? Math.abs(ax.X(view.eps) - ax.X(0)) : 0, ry = view.twoD ? Math.abs(ax.Y(view.eps) - ax.Y(0)) : 0;
    // the k-distance curve: each row's distance to its minPts-th neighbour, sorted; ε sits where it bends upward
    const kd = [];
    if (view.kdist && view.kdist.length) {
      const n = view.kdist.length, top = Math.max(view.kdist[n - 1], view.eps) * 1.08;
      const kax = C().axes({ prefix: 'kd', x: L.side.x, y: L.side.y, w: L.side.w, h: 170, xr: [0, n], yr: [0, top], xticks: C().ticks(0, n, 4), yticks: C().ticks(0, top, 3), xname: 'rows, sorted', yname: V.fit('radius that holds ' + view.minPts + ' rows (itself included)', L.side.w + 30, 13, true) });
      kd.push(...kax.items, C().line(kax, 'kd:l', view.kdist.map((d, i) => [i + 1, d]), 'c0'), C().hline(kax, 'kd:eps', view.eps, 'act'));
      kd.push(V.text('kd:et', L.side.x + 6, kax.Y(view.eps) - 7, 'ε = ' + view.eps, { size: 12, cls: 'strong' }));
      kd.push(V.text('kd:n', L.side.x, L.side.y + 170 + 54, V.fit('Rows below the line are core points.', L.side.w, 12), { size: 12, cls: 'tick' }));
    }
    const frame = (labels, focus, say, title) => {
      const items = [...ax.items];
      ring.forEach((i, t) => { const x = ax.X(view.P[i][0]), y = ax.Y(view.P[i][1]); items.push({ key: 'db:e' + t, kind: 'path', d: `M${r2(x - rx)},${r2(y)} a${r2(rx)},${r2(ry)} 0 1,0 ${r2(2 * rx)},0 a${r2(rx)},${r2(ry)} 0 1,0 ${r2(-2 * rx)},0`, cls: 'wfd-eps', audit: 'ignore', layer: 'back', z: 3 }); });
      items.push(...V.dots(ax, 'db:p', view.P, labels, { cls: (i) => (view.kinds[i] === 'core' ? 'core' : view.kinds[i] === 'border' ? 'border' : 'noise') + (focus !== null && labels[i] !== focus && labels[i] >= 0 ? ' soft' : ''), rOf: (i) => (view.kinds[i] === 'core' ? 4.4 : 3.4) }));
      const lg = V.legend('db:lg', 12, H - 10, [{ text: 'core point (ringed)', c: 0 }, { text: 'border point', c: 0 }, { text: 'noise', c: -1 }], W - 24);
      lg.items[0].cls += ' sp core'; lg.items[2].cls += ' sp border'; lg.items[2].r = 4; lg.items[4].cls += ' sp cn noise';
      items.push(...lg.items, ...kd);
      return { items, view: V.view(W, L.stacked && kd.length ? L.side.y + 170 + 70 : Math.max(H + lg.h, L.side.y + 170 + 70)), say, title };
    };
    const frames = [frame(view.labels.map(() => -1), null, 'First every row counts its neighbours within ε = `' + view.eps + '`. `' + view.core + '` rows have at least `' + view.minPts + '` and are core points; no cluster exists yet.', 'core points')];
    view.grow.forEach((g, i) => frames.push(frame(g.labels, g.cluster, 'Cluster `' + g.cluster + '` grows from one core point to every core point it can reach in steps of at most ε, and takes in the border points on its edge.', 'cluster ' + g.cluster)));
    frames.push(frame(view.labels, null, '`' + view.clusters + '` cluster' + (view.clusters === 1 ? '' : 's') + '. ' + (view.noise === 0 ? 'No row is noise: every row is a core point or within ε of one.' : view.noise === 1 ? 'The one hollow grey row is noise: not core, and not within ε of a core point.' : 'The `' + view.noise + '` hollow grey rows are noise: not core, and not within ε of a core point.'), 'done'));
    V.stepper(el, frames, { label: 'DBSCAN, cluster by cluster', start: frames.length - 1 });
    el.appendChild(V.note('Try a smaller ε: clusters break up and noise grows. A larger ε merges clusters that should stay apart. DBSCAN needs no k, and can follow curved shapes that k-means would cut through.'));
  }));
})(globalThis.Nooral = globalThis.Nooral || {});
