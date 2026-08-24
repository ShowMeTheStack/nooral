// Part 7, k-means: Lloyd's two steps (every point joins its nearest centre, every centre moves to the mean of its
// points) drawn from the engine's trace until nothing changes, and k-means++ seeding, where each new centre is drawn
// in proportion to the squared distance to the nearest centre already chosen.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;

  const nearestField = (ax, key, Cs) => S.Kit.scatter.field(ax, key, (u, v) => { let b = 0, bd = Infinity; Cs.forEach((c, j) => { const d = (u - c[0]) ** 2 + (v - c[1]) ** 2; if (d < bd) { bd = d; b = j; } }); return [b, 0.55]; }, { n: 150, scheme: 'classes', smooth: false });
  const inertiaOf = (X, labels, Cs) => X.reduce((s, x, i) => s + ML().sqdist(x, Cs[labels[i]]), 0);
  /** a poor start: the k points nearest the bottom-left corner */
  const poorStart = (X, k) => X.map((x, i) => [x[0] + x[1], i]).sort((a, b) => a[0] - b[0]).slice(0, k).map(([, i]) => X[i].slice());

  S.topic({
    id: 'km-steps', family: 'p07-kmeans', order: 10, title: 'k-means step by step',
    blurb: 'k-means looks for k centres so that every point is close to one of them. It repeats two steps: assign each point to its nearest centre (the shaded regions), then move each centre to the mean of the points it was given. Neither step can raise the inertia, the sum of squared distances to the centres, so the loop must stop.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'three', options: P.setOptions(['three', 'four', 'touching', 'uneven', 'stretched', 'iris']) },
      { id: 'k', kind: 'number', label: 'k (clusters)', def: 3, min: 1, max: 6 },
      { id: 'init', kind: 'choice', label: 'Start', def: 'poor', options: [{ value: 'poor', label: 'a poor start (k points in a corner)' }, { value: 'random', label: 'k random points' }, { value: 'k-means++', label: 'k-means++' }] },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Three groups, a poor start', values: { set: 'three', k: 3, init: 'poor', seed: 1 } },
      { label: 'Four groups, random start', values: { set: 'four', k: 4, init: 'random', seed: 3 } },
      { label: 'k too small', values: { set: 'four', k: 2, init: 'random', seed: 2 } },
      { label: 'Stretched groups (k-means struggles)', values: { set: 'stretched', k: 3, init: 'k-means++', seed: 1 } },
      { label: 'Uneven sizes', values: { set: 'uneven', k: 3, init: 'poor', seed: 1 } },
      { label: 'Iris petals', values: { set: 'iris', k: 3, init: 'random', seed: 5 } },
    ],
    counters: [{ id: 'iter', label: 'iterations' }, { id: 'moved', label: 'points that changed' }],
    run(inp) {
      const d = P.data(inp.set, { seed: inp.seed }), X = d.X, k = inp.k, F = H().frames(['iter', 'moved']), q = F.q;
      const init = inp.init === 'poor' ? poorStart(X, k) : inp.init;
      const km = new (ML().KMeans)({ k, init, seed: inp.seed, maxIter: 50 }).fit(X);
      const tr = km.trace, assigns = tr.filter((s) => s.kind === 'assign'), updates = tr.filter((s) => s.kind === 'update');
      const C0 = assigns[0].centers;
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('km', X, box, { extra: C0, xname: d.names[0], yname: d.names[1] });
      const steps = [{ kind: 'init', C: C0, inertia: null }];
      assigns.forEach((a, t) => { steps.push({ kind: 'assign', C: a.centers, lab: a.labels, inertia: a.inertia, t }); steps.push({ kind: 'update', C: updates[t].centers, lab: a.labels, inertia: inertiaOf(X, a.labels, updates[t].centers), shift: updates[t].shift, t }); });
      // the last pass assigns and changes nothing: show it as the end
      const imax = Math.max(...steps.filter((s) => s.inertia !== null).map((s) => s.inertia));
      const IA = { x: 545, y: 50, w: 225, h: 150 };
      const iax = C().axes(Object.assign({ prefix: 'ki', xr: [0, Math.max(2, steps.length - 1)], yr: [0, imax * 1.08], xticks: 0, yticks: C().ticks(0, imax, 3).filter((v) => v <= imax * 1.08), yname: 'inertia', xname: 'half-steps' }, IA));
      let prevLab = null;
      steps.forEach((st, s) => {
        const items = [...ax.items];
        if (st.kind !== 'init') items.push(nearestField(ax, 'km:f', st.kind === 'update' ? steps[s - 1].C : st.C));
        items.push(...P.points(ax, 'km:p', X, st.lab || null, { r: 4.4 }));
        st.C.forEach((c, j) => {
          const trail = steps.slice(0, s + 1).filter((z) => z.kind !== 'assign').map((z) => z.C[j]);
          if (trail.length > 1) items.push({ key: 'km:tr' + j, kind: 'path', d: trail.map((p, i) => (i ? 'L' : 'M') + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' '), cls: 'trail c' + j, pathAudit: 'edge-x', layer: 'lines' });
          items.push(P.centre(ax, 'km:c' + j, c, j));
        });
        items.push(...iax.items);
        const done = steps.slice(0, s + 1).map((z, i) => [i, z.inertia]).filter((p) => p[1] !== null);
        if (done.length) { items.push(C().line(iax, 'ki:l', done, 'act')); items.push(C().dot(iax, 'ki:d', s, st.inertia, '')); }
        const sizes = st.lab ? P.sizes(st.lab, k) : null;
        items.push(H().head('kt:h', 530, 270, 'The centres'));
        st.C.forEach((c, j) => items.push(...P.row('kt:' + j, 530, 300 + j * 24, j, `(${H().fx(c[0])}, ${H().fx(c[1])})` + (sizes ? `, ${H().pl(sizes[j], 'point')}` : ''))));
        let say, meta = { phase: st.kind, centers: st.C, inertia: st.inertia, t: st.t };
        if (st.kind === 'init') {
          const cs = P.qpl(q, k, 'centre');
          const st1 = k === 1 ? 'starts' : 'start';
          say = (inp.init === 'poor' ? `${cs} ${st1} on the points nearest the bottom-left corner, a poor start on purpose.` : inp.init === 'random' ? `${cs} ${st1} on points picked at random.` : `${cs} ${st1} where k-means++ put them, spread out by its D² rule.`) + ` No point is assigned yet. k-means will alternate two steps until no point changes its centre.`;
          meta.k = k;
        } else if (st.kind === 'assign') {
          F.count('iter');
          const changed = prevLab ? st.lab.filter((l, i) => l !== prevLab[i]).length : X.length;
          F.set('moved', changed);
          prevLab = st.lab;
          st.changedNone = changed === 0;
          meta = Object.assign(meta, { changed, sizes, assignments: st.t + 1 });
          if (s === 1) say = `Assign: every point joins its nearest centre; the shading shows each centre’s region. The clusters have ${H().list(sizes.map((n) => q.n(n)))} points, and the inertia, the sum of squared distances to the centres, is ${q(st.inertia, 2)}.`;
          else if (changed === 0) say = `Assign again: not one point changes its centre, so the next move would change nothing. k-means has converged after ${q.n(st.t + 1)} assignments, with inertia ${q(st.inertia, 2)}.`;
          else say = `Assign: with the centres in their new places, ${q.n(changed)} ${changed === 1 ? 'point changes' : 'points change'} cluster. The inertia falls to ${q(st.inertia, 2)}.`;
        } else {
          const moved = st.C.map((c, j) => Math.sqrt(ML().sqdist(c, steps[s - 1].C[j]))), far = moved.indexOf(Math.max(...moved));
          meta = Object.assign(meta, { moved, far, sizes, shiftTotal: Math.sqrt(st.shift) });
          if (Math.max(...moved) === 0) say = `Move: every centre is already the mean of its points, so none moves. The inertia stays at ${q(st.inertia, 2)}.`;
          else say = `Move: each centre goes to the mean of its points. The ${P.NAMES[far]} centre moves furthest, by ${q(moved[far], 3)}, and the inertia falls to ${q(st.inertia, 2)}.`;
          if (s === steps.length - 1 && steps[s - 1].changedNone) say += ' Nothing changed in the assignment before this move, so the loop stops.';
          else if (s === steps.length - 1) say += ` The centres moved ${q.sg(st.shift === undefined ? 0 : Math.sqrt(st.shift), 2)} in all, less than the engine’s tolerance, so it stops here.`;
        }
        F.push(items, say, meta);
      });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Lloyd’s algorithm', text: ['repeat:', '  assign: cᵢ = the nearest centre to xᵢ', '  move: μⱼ = the mean of the points with cᵢ = j', 'until no cᵢ changes', 'inertia = Σᵢ ‖xᵢ − μ(cᵢ)‖²', 'each step can only lower it'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'assign' && fr[i + 1] && fr[i + 1].meta.phase === 'update' && Math.max(...fr[i + 1].meta.moved) > 0, once: true, pick: 'option', ask: 'Which centre will move furthest when the centres move to their means?', options: (fr, i) => fr[i].meta.centers.map((_, j) => S.P07.NAMES[j]), answer: (fr, i) => S.P07.NAMES[fr[i + 1].meta.far], explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => f.meta.phase === 'update' && fr[i + 1] && fr[i + 1].meta.phase === 'assign', once: true, pick: 'option', ask: 'After this move, can the inertia at the next assignment be higher than now?', options: ['no: assigning to the nearest centre can only lower it', 'yes, if many points change', 'yes, if a centre moved far'], answer: 'no: assigning to the nearest centre can only lower it', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'Points (0, 0), (2, 0) and (4, 3) form one cluster. Where does its centre move?', options: ['(2, 1)', '(2, 0)', '(3, 1.5)', '(4, 3)'], answer: 0, why: 'The mean: ((0 + 2 + 4) / 3, (0 + 0 + 3) / 3) = (2, 1).' },
      { q: 'Why must k-means stop?', options: ['each step can only lower the inertia, and there are finitely many assignments', 'it runs a fixed number of steps', 'the centres always reach the true groups', 'the learning rate shrinks'], answer: 0, why: 'The inertia never rises and no assignment can repeat with a lower value, so it reaches a point where nothing changes.' },
      { q: 'Does k-means always find the lowest possible inertia?', options: ['no: it stops at a local minimum that depends on the start', 'yes, always', 'yes, if k is right', 'only for two clusters'], answer: 0, why: 'Different starts can end in different clusterings; that is why good seeding and several restarts help.' },
      { q: 'What shape are the regions a set of centres carves out?', options: ['convex polygons (a Voronoi diagram)', 'circles', 'any shape', 'squares on a grid'], answer: 0, why: 'Each region is where one centre is nearest: an intersection of half-planes, so k-means cannot follow a curved or stretched group.' },
      { q: 'What does the inertia measure?', options: ['the sum of squared distances from each point to its centre', 'the number of clusters', 'the distance between the centres', 'how many points changed'], answer: 0, why: 'It is the quantity both steps lower.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'km-plusplus', family: 'p07-kmeans', order: 20, title: 'k-means++: seeding the centres far apart',
    blurb: 'k-means only finds a local minimum, and a bad start can trap it. k-means++ starts better: the first centre is a random point, and each next centre is drawn with probability proportional to D², the squared distance from a point to its nearest centre so far. Points far from every centre are likely picks, so the centres spread over the groups.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'four', options: P.setOptions(['three', 'four', 'uneven', 'touching', 'penguins']) },
      { id: 'k', kind: 'number', label: 'k (clusters)', def: 4, min: 2, max: 6 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Four groups', values: { set: 'four', k: 4, seed: 1 } },
      { label: 'Three groups', values: { set: 'three', k: 3, seed: 4 } },
      { label: 'Uneven sizes', values: { set: 'uneven', k: 3, seed: 2 } },
      { label: 'Penguins, k = 3', values: { set: 'penguins', k: 3, seed: 1 } },
    ],
    counters: [{ id: 'centres', label: 'centres chosen' }],
    run(inp) {
      const d = P.data(inp.set, { seed: 1 }), X = d.X, k = inp.k, F = H().frames(['centres']), q = F.q;
      const tr = ML().tracer({}), centres = ML().kmeansPlusPlus(X, k, ML().rng(inp.seed), tr), seeds = tr.steps;
      const trials = 2 + Math.floor(Math.log(k));
      const idx = centres.map((c) => X.findIndex((x) => x[0] === c[0] && x[1] === c[1]));
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('pp', X, box, { xname: d.names[0], yname: d.names[1] });
      const d2Of = (m) => X.map((x) => Math.min(...centres.slice(0, m).map((c) => ML().sqdist(x, c))));
      const labOf = (m) => X.map((x) => { let b = 0; for (let j = 1; j < m; j++) if (ML().sqdist(x, centres[j]) < ML().sqdist(x, centres[b])) b = j; return b; });
      const panel = (m, d2, total, lab) => {
        const it = [H().head('pp:h', 530, 40, 'Where the D² is'), H().text('pp:h2', 530, 62, 'share of the total D² in each centre’s region', 'axname', 12)];
        for (let j = 0; j < m; j++) {
          const sh = d2.reduce((a, v, i) => a + (lab[i] === j ? v : 0), 0) / total, y = 82 + j * 30, w = 150 * sh;
          it.push(...P.row('pp:r' + j, 530, y + 14, j, P.NAMES[j]));
          it.push({ key: 'pp:b' + j, kind: 'cell', x: 610, y: y + 2, w: Math.max(2, w), h: 16, rx: 2, text: '', cls: 'bar c' + j, audit: 'ignore' });
          it.push(H().text('pp:v' + j, 616 + w, y + 14, H().pct(sh), 'legend', 12));
        }
        it.push(H().text('pp:tot', 530, 290, 'total D² = ' + H().fx(total, 1), 'strong', 13));
        it.push(H().text('pp:m', 530, 314, H().pl(m, 'centre') + ' chosen', 'legend', 13));
        return it;
      };
      for (let m = 1; m <= k; m++) {
        F.set('centres', m);
        const d2 = d2Of(m), total = d2.reduce((a, b) => a + b, 0), lab = labOf(m);
        const maxShare = Math.max(...d2) / total;
        const items = [...ax.items, ...P.points(ax, 'pp:p', X, lab, { r: (i) => 2 + 5 * Math.sqrt(d2[i] / Math.max(...d2)) })];
        if (m > 1) items.push(S.Kit.scatter.ring(ax, 'pp:new', centres[m - 1][0], centres[m - 1][1], 'act', 15));
        centres.slice(0, m).forEach((c, j) => items.push(P.centre(ax, 'pp:c' + j, c, j)));
        items.push(...panel(m, d2, total, lab));
        const far = d2.indexOf(Math.max(...d2));
        const before = m > 1 ? d2Of(m - 1) : null, share = m > 1 ? before[idx[m - 1]] / before.reduce((a, b) => a + b, 0) : null;
        let say;
        if (m === 1) say = `The first centre is point ${q.n(idx[0])}, picked at random. Each point is drawn with a size that grows with D², its squared distance to that centre: the total D² is ${q(total, 1)}, and the far point ${q.n(far)} alone holds ${q.pct(maxShare)} of it.`;
        else say = `Centre ${q.n(m)} is point ${q.n(idx[m - 1])}. The engine drew ${q.n(trials)} candidates in proportion to D² and kept the one that lowers the total most; this point held ${q.pct(share)} of the D² before. The total D² falls to ${q(total, 1)}.`;
        F.push(items, say, { phase: 'seed', m, idx: idx.slice(0, m), total, far, maxShare, trials, share, d2max: Math.max(...d2) });
      }
      // ten seeds, k-means after ++ against k-means after random starts
      const runs = [];
      for (let s = 1; s <= 10; s++) runs.push([new (ML().KMeans)({ k, seed: s, init: 'k-means++', trace: false }).fit(X).inertia, new (ML().KMeans)({ k, seed: s, init: 'random', trace: false }).fit(X).inertia]);
      const best = Math.min(...runs.flat()), hit = (j) => runs.filter((r) => r[j] <= best * (1 + 1e-6)).length;
      const meanOf = (j) => runs.reduce((a, r) => a + r[j], 0) / runs.length;
      const RA = { x: 575, y: 190, w: 195, h: 220 }, rmax = Math.max(...runs.flat());
      const rax = C().axes(Object.assign({ prefix: 'pr', xr: [0.5, 10.5], yr: [0, rmax * 1.1], xticks: [1, 5, 10], yticks: C().ticks(0, rmax, 4).filter((v) => v <= rmax * 1.1), xname: 'seed', yname: 'final inertia' }, RA));
      const items = [...ax.items, ...P.points(ax, 'pp:p', X, labOf(k), { r: 4.4 })];
      centres.forEach((c, j) => items.push(P.centre(ax, 'pp:c' + j, c, j)));
      items.push(H().head('pp:h', 530, 40, 'Ten seeds, then k-means'), ...rax.items);
      runs.forEach((r, s) => { items.push(C().dot(rax, 'pr:a' + s, s + 1 - 0.15, r[0], 'c0', 4)); items.push(C().dot(rax, 'pr:b' + s, s + 1 + 0.15, r[1], 'c1', 4)); });
      items.push(...C().legend('pr:lg', 530, 90, [{ text: 'after k-means++', cls: 'c0', dot: true }, { text: 'after a random start', cls: 'c1', dot: true }], { dir: 'col', size: 13 }));
      F.push(items, `These are the starting centres; k-means then takes over. Over ten seeds, k-means started this way ends with mean inertia ${q(meanOf(0), 1)} and reaches the best value found, ${q(best, 1)}, in ${q.n(hit(0))} of the ten runs. Started from ${P.qpl(q, k, 'random point')}, the mean is ${q(meanOf(1), 1)} and the best is reached in ${q.n(hit(1))} of ten.`, { phase: 'compare', runs, best, mean: [meanOf(0), meanOf(1)], hits: [hit(0), hit(1)], k });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'k-means++', text: ['μ₁ = a point chosen at random', 'D(x) = distance from x to its nearest chosen centre', 'next centre: x with probability D(x)² / Σ D²', 'greedy version: draw a few, keep the best', 'then run k-means from these centres'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'seed' && f.meta.m === 1 && fr[i + 1] && fr[i + 1].meta.phase === 'seed', pick: 'option', ask: 'Where is the next centre most likely to land?', options: ['far from the first centre', 'right next to the first centre', 'at the mean of all the points', 'anywhere, with equal chance'], answer: 'far from the first centre', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'Three points have D² = 1, 3 and 6. With what probability is the third chosen?', options: ['0.6', '0.5', '0.33', '1'], answer: 0, why: '6 / (1 + 3 + 6) = 0.6.' },
      { q: 'Why square the distance rather than use it as it is?', options: ['it favours far points even more strongly, and matches the inertia', 'squares are faster to compute', 'to make every probability equal', 'distances can be negative'], answer: 0, why: 'D² weighting is what gives k-means++ its guarantee of being within a log factor of the best inertia in expectation.' },
      { q: 'Can a point that is already a centre be chosen again?', options: ['no: its D² is 0', 'yes, often', 'only for the last centre', 'only with a random start'], answer: 0, why: 'A chosen point is at distance 0 from a centre, so its probability is 0.' },
      { q: 'What does k-means++ change in k-means?', options: ['only the starting centres', 'the assignment step', 'the distance used', 'the number of clusters'], answer: 0, why: 'After seeding, the usual assign-and-move loop runs.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
