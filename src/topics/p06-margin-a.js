// Part 6, the margin: many lines split two groups, but one leaves the widest empty street between them. Three
// candidate lines are drawn with the widest street each direction allows, then the SVM's own line, whose street
// rests on its support vectors; deleting a point that is not a support vector changes nothing, deleting one that
// is moves the line.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const BIG_C = 1000;

  const separable = (inp) => { const d = P.data(inp.set, { n: inp.n, seed: inp.seed }); return Math.max(...P.fit(d.X, d.y, { C: BIG_C, tol: 1e-6 }).slack) < 1e-3; };

  S.topic({
    id: 'svm-max-margin', family: 'p06-margin', order: 10, title: 'The widest street: maximum margin and support vectors',
    blurb: 'When a straight line can split two classes, infinitely many can. A support vector machine picks the one with the widest empty street on either side, the maximum margin. The street’s edges touch a few points, the support vectors: they alone decide the line, and every other point could move or vanish without changing it.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'apart', options: P.setOptions(['apart', 'close']) },
      { id: 'n', kind: 'number', label: 'Points', def: 24, min: 8, max: 60 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999, check: (v, inp) => (Number.isInteger(v) && Number.isInteger(inp.n) && inp.n >= 8 && inp.n <= 60 && !separable(inp) ? 'With this seed the two groups overlap, so no empty street exists: try another seed, or see the soft margin' : null) },
    ],
    presets: [
      { label: 'Well apart', values: { set: 'apart', n: 24, seed: 1 } },
      { label: 'Close together', values: { set: 'close', n: 24, seed: 1 } },
      { label: 'A narrow gap', values: { set: 'close', n: 24, seed: 6 } },
      { label: 'Two support vectors only', values: { set: 'apart', n: 24, seed: 2 } },
      { label: 'Many points', values: { set: 'apart', n: 60, seed: 3 } },
    ],
    counters: [{ id: 'lines', label: 'lines tried' }, { id: 'sv', label: 'support vectors' }],
    run(inp) {
      const d = P.data(inp.set, { n: inp.n, seed: inp.seed }), X = d.X, y = d.y, n = X.length;
      const fit = P.fit(X, y, { C: BIG_C, tol: 1e-6 }), F = H().frames(['lines', 'sv']), q = F.q;
      const th0 = Math.atan2(fit.w[1], fit.w[0]);
      // three other directions, each with the widest street it allows
      const cands = [];
      for (const dd of [40, -28, 16, -12, 8, -6, 4]) { const s = P.streetAt(X, y, th0 + (dd * Math.PI) / 180); if (s && cands.length < 3) cands.push(s); }
      const box = { x: 40, y: 24, w: 420, h: 420 }, ax = P.axes('mm', X, box, { pad: 0.45 });
      const RX = 520, bars = [];
      const barPanel = (shown, best, after) => {
        const items = [H().head('bp:t', RX, 50, 'Width of each street')];
        const all = shown.map((s, k) => ({ name: 'line ' + 'ABC'[k], w: 2 * s.h, cls: '' })).concat(best ? [{ name: 'the SVM', w: fit.width, cls: 'act' }] : []).concat(after ? [{ name: 'refitted', w: after, cls: 'act' }] : []);
        const wmax = Math.max(fit.width, after || 0, ...cands.map((s) => 2 * s.h));
        all.forEach((b, k) => {
          const yy = 80 + k * 40, len = (150 * b.w) / wmax;
          items.push(H().text('bp:n' + k, RX, yy + 17, b.name, b.cls ? 'strong' : 'legend', 13));
          items.push({ key: 'bp:b' + k, kind: 'cell', x: RX + 72, y: yy + 4, w: Math.max(2, len), h: 18, rx: 3, text: '', cls: 'bar ' + (b.cls ? 'c1' : 'soft'), audit: 'ignore' });
          items.push(H().text('bp:v' + k, RX + 80 + len, yy + 17, H().fx(b.w, 3), 'legend', 13));
        });
        return items;
      };
      const base = () => [...ax.items];
      const street = (key, s, cls) => [P.line(ax, key + ':c', s.n, s.c, cls), P.line(ax, key + ':l', s.n, s.c - s.h, 'margin soft'), P.line(ax, key + ':h', s.n, s.c + s.h, 'margin soft')];
      const pts = (fitShown, only) => P.points(ax, 'mm', X, y, fitShown, { r: 5, cls: only });

      F.push([...base(), ...pts(null), ...barPanel([], false)],
        `${q.n(n)} points in two classes, blue and orange. A straight line can split them, and so can many others: tilt it a little or slide it and it still splits them. Which line should we trust with new points?`, { phase: 'data', n });
      cands.forEach((s, k) => {
        F.count('lines');
        const items = [...base(), ...street('cand', s, 'soft'), ...pts(null), ...barPanel(cands.slice(0, k + 1), false)];
        items.push(S.Kit.scatter.ring(ax, 'mm:lim0', X[s.lo][0], X[s.lo][1], 'hot', 9), S.Kit.scatter.ring(ax, 'mm:lim1', X[s.hi][0], X[s.hi][1], 'hot', 9));
        F.push(items, `Line ${'ABC'[k]}: for this direction, the widest empty street is ${q(2 * s.h, 3)} wide. It is stopped by two points, one of each class, ringed in yellow; any wider and a point would be inside it.`, { phase: 'cand', k, width: 2 * s.h, th: s.th, lo: s.lo, hi: s.hi });
      });
      F.set('sv', fit.sv.length);
      const svItems = () => [...base(), ...P.boundary(ax, 'mm', fit), ...pts(fit), ...barPanel(cands, true), ...P.ringLegend('mm:lg', RX, 330, true)];
      const nB = fit.sv.filter((i) => y[i] === 0).length, nO = fit.sv.length - nB;
      F.push(svItems(), `The SVM turns and slides the line to make the street as wide as it can: ${q(fit.width, 3)}, wider than any line tried. Its edges, the dashed lines f = ±1, touch ${q.n(fit.sv.length)} points, ${q.n(nB)} blue and ${q.n(nO)} orange: the support vectors, ringed.`, { phase: 'svm', width: fit.width, sv: fit.sv, nsv: fit.sv.length, nB, nO, w: fit.w, b: fit.b });
      const eq = `w = (${H().fx(fit.w[0])}, ${H().fx(fit.w[1])}), b = ${H().fx(fit.b)}`;
      F.push([...svItems(), H().text('mm:eq', RX, 420, 'f(x) = w·x + b', 'strong', 14), H().text('mm:eq2', RX, 444, eq, 'legend', 13)],
        `The line is f(x) = w·x + b = 0 with w = (${q(fit.w[0])}, ${q(fit.w[1])}) and b = ${q(fit.b)}. The margins are f = ±1, so the street’s width is 2 / ‖w‖ = 2 / ${q(fit.norm, 3)} = ${q(fit.width, 3)}: a shorter w means a wider street, and the SVM finds the shortest w that keeps every point at |f| ≥ 1 on its own side.`, { phase: 'eq', w: fit.w, b: fit.b, norm: fit.norm, width: fit.width });
      // delete a point that is not a support vector: nothing moves
      const far = fit.yf.map((v, i) => [v, i]).filter(([, i]) => fit.alpha[i] === 0).sort((a, b) => b[0] - a[0])[0][1];
      const keep = (drop) => X.map((_, i) => i).filter((i) => i !== drop);
      const fitA = P.fit(keep(far).map((i) => X[i]), keep(far).map((i) => y[i]), { C: BIG_C, tol: 1e-6 });
      const mapItems = (fitB, drop) => {
        const idx = keep(drop), XB = idx.map((i) => X[i]), yB = idx.map((i) => y[i]), items = [...base(), ...P.boundary(ax, 'mm', fitB)];
        items.push(...S.Kit.scatter.points(ax, 'mm:p', XB, yB, { r: 5 }).map((it, k) => Object.assign(it, { key: 'mm:p:' + idx[k] })));
        fitB.sv.forEach((j) => items.push(S.Kit.scatter.ring(ax, 'mm:r' + idx[j], XB[j][0], XB[j][1], fitB.alpha[j] < fitB.C * (1 - 1e-6) ? 'act' : '', 9)));
        items.push({ key: 'mm:gone', kind: 'dot', x: ax.X(X[drop][0]), y: ax.Y(X[drop][1]), r: 5, cls: 'sp c' + y[drop], opacity: 0.3, audit: 'overlap-ok' });
        items.push(...P.ringLegend('mm:lg', RX, 330, true));
        return items;
      };
      const moved = (fa) => Math.hypot(fa.w[0] - fit.w[0], fa.w[1] - fit.w[1]) + Math.abs(fa.b - fit.b);
      F.push([...mapItems(fitA, far), ...barPanel(cands, true, fitA.width)], `Delete the faded point, the ${y[far] ? 'orange' : 'blue'} one deepest in its own side (it has α = 0). Fit again: the width is ${q(fitA.width, 3)}, w and b change by ${q.sg(moved(fitA), 2)} in all, which is only the solver’s rounding. A point off the street has no say.`, { phase: 'drop-free', drop: far, width: fitA.width, moved: moved(fitA) });
      const sv = fit.sv.slice().sort((a, b) => fit.alpha[b] - fit.alpha[a])[0];
      const fitB = P.fit(keep(sv).map((i) => X[i]), keep(sv).map((i) => y[i]), { C: BIG_C, tol: 1e-6 });
      F.set('sv', fitB.sv.length);
      F.push([...mapItems(fitB, sv), ...barPanel(cands, true, fitB.width)], `Now delete the support vector with the largest α instead (${y[sv] ? 'orange' : 'blue'}). The street was resting on it, so the line turns and the street widens to ${q(fitB.width, 3)}, now held by ${q.n(fitB.sv.length)} support vectors. Only the support vectors matter.`, { phase: 'drop-sv', drop: sv, width: fitB.width, nsv: fitB.sv.length });
      // on a phone, the plot alone (the side panel's numbers are in the narration)
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 470, h: 480 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The maximum margin', text: ['f(x) = w·x + b; predict class 1 where f > 0', 'the margins: f(x) = +1 and f(x) = −1', 'width of the street = 2 / ‖w‖', 'minimise ½‖w‖² subject to yᵢ f(xᵢ) ≥ 1 (yᵢ = ±1)', 'support vectors: the points with yᵢ f(xᵢ) = 1'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'svm', pick: 'item', ask: 'Where will the widest street rest? Click a point you think will be a support vector.', answer: (fr, i) => fr[i + 1].meta.sv.map((k) => 'mm:p:' + k), explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'eq', pick: 'option', ask: 'Delete a point far from the street, on its own side. What happens to the line?', options: ['nothing: it stays where it is', 'it moves towards the deleted point', 'it moves away from it', 'the street gets narrower'], answer: 'nothing: it stays where it is', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'The street of an SVM is 2 / ‖w‖ wide. If ‖w‖ = 4, how wide is it?', options: ['0.5', '2', '4', '8'], answer: 0, why: '2 / 4 = 0.5.' },
      { q: 'Which points decide where a hard-margin SVM puts its line?', options: ['only the support vectors on the margins', 'every point equally', 'the points furthest from the line', 'the class means'], answer: 0, why: 'The constraints that hold with equality, yᵢ f(xᵢ) = 1, fix the solution; the rest could move freely as long as they stay off the street.' },
      { q: 'Why prefer the widest street?', options: ['new points near the classes are least likely to land on the wrong side', 'it uses fewer features', 'it is the fastest to compute', 'it always has zero training error'], answer: 0, why: 'A wide margin leaves room for noise in new points; it is also what the theory of margins rewards.' },
      { q: 'What does the SVM minimise to get the widest street?', options: ['½‖w‖², keeping yᵢ(w·xᵢ + b) ≥ 1', '‖w‖ alone, with no constraints', 'the number of support vectors', 'the training error only'], answer: 0, why: 'Width is 2 / ‖w‖, so the widest street is the shortest w that still meets every margin constraint.' },
      { q: 'At least how many support vectors does a hard-margin SVM in two classes have?', options: ['2: one from each class', '1', '3', 'as many as there are points'], answer: 0, why: 'The street must touch both classes, or it could be widened by sliding it.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
