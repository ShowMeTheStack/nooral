// k-means on three blobs from a poor start: assign each point to its nearest centre (the shading shows every
// centre's region), move each centre to the mean of its points, and repeat until nothing changes. The inertia
// (the sum of squared distances to the centres) falls at every half-step.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, SC = () => S.Kit.scatter, box = D.box;
  const NAMES = ['blue', 'orange', 'green'];
  D.add({
    id: 'kmeans-steps', order: 40, tag: 'Clustering · k-means', short: 'k-means steps', title: 'k-means: assign, move the centres, converge',
    blurb: 'Lloyd’s algorithm with k = 3, started from three centres placed badly on purpose. Each assignment can only lower the inertia, and so can each move, so it must stop.',
    frames({ narrow }) {
      const T = S.Toy, data = D.once('blobs3', () => T.blobs(90, [[-1.5, 1], [1.5, 1.2], [0, -1.4]], 0.55, 11));
      const run = D.once('kmeans3', () => T.kmeans(data.X, [[-2.4, -2.2], [-1.4, -2.6], [-0.6, -2.4]]));
      const xr = [-3.4, 3.2], yr = [-3.1, 2.8];
      const L = narrow
        ? { view: box(-6, -14, 392, 712), sc: { x: 34, y: 20, w: 330, h: 300 }, ic: { x: 50, y: 440, w: 300, h: 130 } }
        : { view: box(-10, -14, 1180, 534), sc: { x: 50, y: 24, w: 500, h: 450 }, ic: { x: 700, y: 60, w: 420, h: 220 } };
      const ax = C().axes(Object.assign({ prefix: 'km', xr, yr, xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2], xname: 'x', yname: 'y' }, L.sc));
      const steps = run.steps, imax = steps.reduce((a, s) => Math.max(a, s.inertia || 0), 0);
      const iax = C().axes(Object.assign({ prefix: 'ki', xr: [0, steps.length - 1], yr: [0, Math.ceil(imax / 100) * 100], xticks: 0, yticks: [0, 200, 400, 600].filter((v) => v <= Math.ceil(imax / 100) * 100), yname: 'inertia: the sum of squared distances', xname: 'half-steps' }, L.ic));
      return steps.map((st, k) => {
        const items = [...ax.items];
        if (st.kind !== 'init') {
          // the regions belong to the centres the points were assigned to (before a move, the old centres)
          const Cs = st.kind === 'move' ? steps[k - 1].C : st.C;
          items.push(SC().field(ax, 'km:f', (u, v) => { let b = 0; Cs.forEach((c, j) => { if ((u - c[0]) ** 2 + (v - c[1]) ** 2 < (u - Cs[b][0]) ** 2 + (v - Cs[b][1]) ** 2) b = j; }); return [b, 0.5]; }, { n: narrow ? 110 : 170, scheme: 'classes', smooth: false }));
        }
        items.push(...SC().points(ax, 'km:p', data.X, st.lab.map((l) => (l < 0 ? 'n' : l)), { r: narrow ? 3.2 : 4.2 }));
        // the centres, and the trail each has followed
        st.C.forEach((c, j) => {
          const trail = steps.slice(0, k + 1).filter((s) => s.kind !== 'assign').map((s) => s.C[j]);
          if (trail.length > 1) items.push({ key: 'km:tr' + j, kind: 'path', d: trail.map((p, i) => (i ? 'L' : 'M') + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' '), cls: 'trail c' + j, pathAudit: 'edge-x', layer: 'lines' });
          items.push({ key: 'km:c' + j, kind: 'state', x: ax.X(c[0]), y: ax.Y(c[1]), w: narrow ? 18 : 22, h: narrow ? 18 : 22, rx: 4, text: '', cls: 'centre c' + j, audit: 'overlap-ok' });
        });
        const done = steps.slice(0, k + 1).filter((s) => s.inertia !== null);
        items.push(...iax.items);
        if (done.length) {
          items.push(C().line(iax, 'ki:l', done.map((s) => [steps.indexOf(s), s.inertia]), 'act'));
          items.push(C().dot(iax, 'ki:d', k, st.inertia, ''));
          items.push({ key: 'ki:v', kind: 'text', x: L.ic.x + L.ic.w, y: L.ic.y - 10, text: `now ${st.inertia.toFixed(1)}`, anchor: 'end', size: 13, cls: 'strong' });
        }
        const counts = [0, 1, 2].map((j) => st.lab.filter((l) => l === j).length);
        // the centres, as a small table
        const ty = L.ic.y + L.ic.h + (narrow ? 56 : 80);
        st.C.forEach((c, j) => {
          const y = ty + j * 24;
          items.push({ key: 'kt:s' + j, kind: 'dot', x: L.ic.x + 6, y: y - 4.5, r: 6, cls: 'swatch c' + j });
          items.push({ key: 'kt:t' + j, kind: 'text', x: L.ic.x + 20, y, text: `${NAMES[j]} centre at (${D.fix(c[0])}, ${D.fix(c[1])})` + (st.kind === 'init' ? '' : `, ${counts[j]} points`), size: 13, cls: 'legend' });
        });
        let say;
        if (st.kind === 'init') say = 'Three centres start bunched together low on the left, far from two of the three groups. No point is assigned yet.';
        else if (st.kind === 'assign') say = (st.changed ? `Assign: each point joins its nearest centre (\`${st.changed}\` point${st.changed === 1 ? '' : 's'} changed). ` : 'Assign again: no point changes centre, so the next move would change nothing. k-means has converged. ') + `Sizes \`${counts.join(', ')}\`, inertia \`${st.inertia.toFixed(1)}\`.`;
        else {
          const far = st.moved.indexOf(Math.max(...st.moved));
          say = `Move: each centre goes to the mean of its points. The ${NAMES[far]} centre moves furthest, by \`${st.moved[far].toFixed(2)}\`; the inertia falls to \`${st.inertia.toFixed(1)}\`.`;
        }
        return { view: L.view, items, say, title: st.kind === 'init' ? 'start' : st.kind };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
