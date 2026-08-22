// Part 6, the dual problem and SMO. The weights of a linear SVM are a sum over its support vectors, w = Σ αᵢyᵢxᵢ,
// drawn arrow by arrow; then the dual problem in the multipliers α, solved by SMO: each step picks the pair that
// breaks the optimality conditions most and moves just those two along the one line that keeps Σ αᵢyᵢ = 0, clipped
// to the box 0 ≤ α ≤ C.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const small = (inp, n) => S.Data.blobs({ n: n || inp.n, centers: [[-0.8, -0.5], [0.8, 0.5]], std: inp.spread, seed: inp.seed });

  S.topic({
    id: 'svm-dual', family: 'p06-solver', order: 10, title: 'The weights as a sum over support vectors',
    blurb: 'The solution of a linear SVM can be written w = Σ αᵢ yᵢ xᵢ: a weighted sum of the training points, where only the support vectors have αᵢ > 0. Each one pulls w towards itself if it is orange (y = +1) and away if it is blue (y = −1), by its multiplier α. The multipliers of the two classes balance, Σ αᵢ yᵢ = 0, and finding them is the dual problem.',
    inputs: [
      { id: 'n', kind: 'number', label: 'Points', def: 16, min: 6, max: 40 },
      { id: 'spread', kind: 'slider', label: 'Spread', def: 0.6, min: 0.3, max: 1, step: 0.05 },
      { id: 'C', kind: 'slider', label: 'C', def: 1, min: 0.05, max: 50, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Sixteen points, C = 1', values: { n: 16, spread: 0.6, C: 1, seed: 1 } },
      { label: 'Apart, C = 10', values: { n: 16, spread: 0.4, C: 10, seed: 2 } },
      { label: 'Overlapping, C = 0.2', values: { n: 24, spread: 0.9, C: 0.2, seed: 3 } },
    ],
    counters: [{ id: 'added', label: 'support vectors added' }],
    run(inp) {
      const d = small(inp), X = d.X, y = d.y, Cv = +H().sg(inp.C, 2), fit = P.fit(X, y, { C: Cv, tol: 1e-6 }), F = H().frames(['added']), q = F.q;
      const ys = y.map((v) => (v ? 1 : -1)), order = fit.sv.slice().sort((a, b) => fit.alpha[b] - fit.alpha[a]);
      const ax = P.axes('dv', X, { x: 40, y: 24, w: 400, h: 400 }, { pad: 0.35 });
      // the running sums of αᵢyᵢxᵢ, and a panel that fits them all with the same scale on both axes
      const sums = [[0, 0]];
      for (const i of order) { const p = sums[sums.length - 1]; sums.push([p[0] + fit.alpha[i] * ys[i] * X[i][0], p[1] + fit.alpha[i] * ys[i] * X[i][1]]); }
      const xs = sums.map((p) => p[0]), yv = sums.map((p) => p[1]), pad = 0.15 * Math.max(0.5, Math.max(...xs) - Math.min(...xs), Math.max(...yv) - Math.min(...yv));
      const vx = P.axes('vw', null, { x: 530, y: 50, w: 250, h: 250 }, { xr: [Math.min(...xs) - pad, Math.max(...xs) + pad], yr: [Math.min(...yv) - pad, Math.max(...yv) + pad], xname: 'w₁', yname: 'w₂' });
      const vec = (upto, lit) => {
        const items = [...vx.items];
        for (let k = 0; k < upto; k++) {
          const i = order[k], a = sums[k], b = sums[k + 1];
          if (Math.hypot(vx.X(b[0]) - vx.X(a[0]), vx.Y(b[1]) - vx.Y(a[1])) < 3) continue;
          items.push(H().arrow('vw:a' + k, [vx.X(a[0]), vx.Y(a[1])], [vx.X(b[0]), vx.Y(b[1])], (k === lit ? 'act ' : '') + 'c' + y[i]));
        }
        if (upto === order.length) items.push(H().arrow('vw:w', [vx.X(0), vx.Y(0)], [vx.X(fit.w[0]), vx.Y(fit.w[1])], 'act'));
        items.push({ key: 'vw:o', kind: 'dot', x: vx.X(0), y: vx.Y(0), r: 4, cls: 'goal', point: true });
        return items;
      };
      const plot = (lit) => {
        const items = [...ax.items, ...P.boundary(ax, 'dv', fit, { n: 70 }), ...P.points(ax, 'dv', X, y, fit, { r: 5 })];
        if (lit !== undefined) items.push(S.Kit.scatter.ring(ax, 'dv:lit', X[lit][0], X[lit][1], 'hot', 12));
        return items;
      };
      F.push([...plot(), ...vec(0)], `The linear SVM with C = ${q.sg(Cv, 2)} on ${q.n(X.length)} points has w = (${q(fit.w[0])}, ${q(fit.w[1])}). Only ${q.n(fit.sv.length)} points have a multiplier α > 0, the support vectors. On the right we build w from them alone, starting from zero.`, { phase: 'start', C: Cv, n: X.length, w: fit.w, nsv: fit.sv.length });
      order.forEach((i, k) => {
        F.count('added');
        const c = [fit.alpha[i] * ys[i] * X[i][0], fit.alpha[i] * ys[i] * X[i][1]];
        F.push([...plot(i), ...vec(k + 1, k)], `Add point ${q.n(i)}, ${ys[i] > 0 ? 'orange, y = +1' : 'blue, y = −1'}, with α = ${q(fit.alpha[i], 3)}${fit.alpha[i] >= Cv * (1 - 1e-6) ? ' (at the cap C)' : ''}: its term αyx = (${q(c[0])}, ${q(c[1])}) ${ys[i] > 0 ? 'points towards it' : 'points away from it'}. The sum so far is (${q(sums[k + 1][0])}, ${q(sums[k + 1][1])}).`, { phase: 'add', i, alpha: fit.alpha[i], term: c, sum: sums[k + 1], y: ys[i] });
      });
      const pos = order.filter((i) => ys[i] > 0).reduce((s, i) => s + fit.alpha[i], 0), neg = order.filter((i) => ys[i] < 0).reduce((s, i) => s + fit.alpha[i], 0);
      const half = 0.5 * fit.norm * fit.norm, dual = -fit.m.machines[0].objective;
      F.push([...plot(), ...vec(order.length)], `The arrows end exactly at w = (${q(fit.w[0])}, ${q(fit.w[1])}). The multipliers balance: the orange ones add up to ${q(pos, 4)} and the blue ones to ${q(neg, 4)}, which is the constraint Σ αᵢyᵢ = 0. The dual objective Σαᵢ − ½‖w‖² is ${q(dual, 3)}, the value the solver maximised over α.`, { phase: 'done', w: fit.w, pos, neg, dual, half, sumAlpha: pos + neg });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The dual problem', text: ['maximise Σ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)', 'subject to 0 ≤ αᵢ ≤ C and Σ αᵢ yᵢ = 0', 'then w = Σ αᵢ yᵢ xᵢ', 'αᵢ = 0: off the street, no say', '0 < αᵢ < C: on a margin', 'αᵢ = C: inside the street or wrong'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will a blue point’s term αyx point towards the blue point or away from it?', options: ['away from it', 'towards it'], answer: 'away from it', explain: 'Blue has y = −1, so its term is −α·x: it points away from the point.' },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'done', pick: 'option', ask: 'What do the multipliers of the orange points add up to, compared with the blue ones?', options: ['the same total', 'twice as much', 'zero', 'it depends on C only'], answer: 'the same total', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'Which training points appear in w = Σ αᵢyᵢxᵢ with a non-zero weight?', options: ['the support vectors', 'all of them', 'the misclassified ones only', 'the class means'], answer: 0, why: 'Every other point has αᵢ = 0.' },
      { q: 'Two support vectors: orange x = (1, 0) with α = 0.5 and blue x = (−1, 0) with α = 0.5. What is w?', options: ['(1, 0)', '(0, 0)', '(0.5, 0)', '(−1, 0)'], answer: 0, why: '0.5·(1, 0) − 0.5·(−1, 0) = (1, 0).' },
      { q: 'What does the constraint Σ αᵢyᵢ = 0 say?', options: ['the multipliers of the two classes have equal totals', 'every α is zero', 'there are as many orange as blue points', 'w has length 1'], answer: 0, why: 'Split the sum by y = +1 and y = −1.' },
      { q: 'Why does the dual matter for kernels?', options: ['it uses the data only through dot products xᵢ·xⱼ, which a kernel can replace', 'it has fewer constraints', 'it never needs C', 'it finds w directly'], answer: 0, why: 'Replace xᵢ·xⱼ with K(xᵢ, xⱼ) and the same solver fits a kernel SVM.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // SMO, pair by pair
  // ---------------------------------------------------------------------------------------------------------------
  /** replay the engine's SMO steps: the multipliers after each, and the solver's gradient, offset and objective */
  function replay(X, yy, kf, Cv, steps) {
    const n = X.length, K = X.map((a) => X.map((b) => kf(a, b))), a = new Array(n).fill(0), out = [];
    const state = () => {
      const G = a.map((_, t) => { let s = -1; for (let u = 0; u < n; u++) s += a[u] * yy[t] * yy[u] * K[t][u]; return s; });
      let ub = Infinity, lb = -Infinity, nf = 0, sf = 0;
      for (let t = 0; t < n; t++) { const g = yy[t] * G[t]; if (a[t] >= Cv) { if (yy[t] < 0) ub = Math.min(ub, g); else lb = Math.max(lb, g); } else if (a[t] <= 0) { if (yy[t] > 0) ub = Math.min(ub, g); else lb = Math.max(lb, g); } else { nf++; sf += g; } }
      const rho = nf ? sf / nf : (ub + lb) / 2;
      let dual = 0; for (let t = 0; t < n; t++) dual += a[t] - 0.5 * a[t] * (G[t] + 1);
      return { alpha: a.slice(), rho, dual };
    };
    out.push(Object.assign(state(), { step: null }));
    for (const s of steps) { const oi = a[s.i], oj = a[s.j]; a[s.i] = s.ai; a[s.j] = s.aj; out.push(Object.assign(state(), { step: s, oi, oj })); }
    return { states: out, K };
  }

  S.topic({
    id: 'svm-smo', family: 'p06-solver', order: 20, title: 'SMO: the dual solved two multipliers at a time',
    blurb: 'Sequential minimal optimisation solves the dual one pair at a time. Two is the fewest it can change: Σ αᵢyᵢ = 0 ties every move of one multiplier to another. For the chosen pair the constraint is a line through the box 0 ≤ αᵢ, αⱼ ≤ C; the best point on that line is found exactly, then clipped to the box. Each step picks the pair that breaks the optimality conditions most, and the solver stops when the largest violation, the gap, is below a tolerance.',
    inputs: [
      { id: 'n', kind: 'number', label: 'Points', def: 12, min: 6, max: 20 },
      { id: 'spread', kind: 'slider', label: 'Spread', def: 0.6, min: 0.3, max: 1, step: 0.05 },
      { id: 'kernel', kind: 'choice', label: 'Kernel', def: 'linear', options: [{ value: 'linear', label: 'linear' }, { value: 'rbf', label: 'RBF, γ = 1' }] },
      { id: 'C', kind: 'slider', label: 'C', def: 1, min: 0.1, max: 20, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Linear, C = 1', values: { n: 12, spread: 0.6, kernel: 'linear', C: 1, seed: 1 } },
      { label: 'Linear, C = 10', values: { n: 12, spread: 0.6, kernel: 'linear', C: 10, seed: 1 } },
      { label: 'RBF, C = 1', values: { n: 12, spread: 0.6, kernel: 'rbf', C: 1, seed: 2 } },
      { label: 'Overlapping, C = 0.5', values: { n: 16, spread: 0.9, kernel: 'linear', C: 0.5, seed: 3 } },
    ],
    counters: [{ id: 'steps', label: 'SMO steps' }, { id: 'sv', label: 'α > 0' }],
    run(inp) {
      const d = small(inp), X = d.X, y = d.y, n = X.length, Cv = +H().sg(inp.C, 2);
      const m = new (ML().SVC)({ kernel: inp.kernel, gamma: 1, C: Cv, tol: 1e-3 }).fit(X, y), steps = m.trace.filter((s) => s.kind === 'pair');
      P.record('svc', X, y, m.params, m);
      // the solver's own labels: blue (class 0) is +1, orange −1
      const yy = y.map((v) => (v === 0 ? 1 : -1)), R = replay(X, yy, m.kf, Cv, steps), st = R.states;
      const F = H().frames(['steps', 'sv']), q = F.q, cap = Cv * (1 - 1e-9);
      const ax = P.axes('sm', X, { x: 30, y: 24, w: 400, h: 400 }, { pad: 0.35 });
      const fOf = (s) => (x) => { let v = -s.rho; for (let t = 0; t < n; t++) if (s.alpha[t]) v += s.alpha[t] * yy[t] * m.kf(X[t], x); return -v; };
      // the box of the pair, the multipliers as bars and the gap per step
      const bx = { x: 520, y: 40, s: 120 }, BX = (v) => H().r2(bx.x + (v / Cv) * bx.s), BY = (v) => H().r2(bx.y + bx.s - (v / Cv) * bx.s);
      const boxView = (s) => {
        const items = [{ key: 'bx:sq', kind: 'path', d: `M${bx.x},${bx.y} h${bx.s} v${bx.s} h${-bx.s} Z`, cls: 'mlbox', audit: 'ignore', layer: 'back', box: { x: bx.x, y: bx.y, w: bx.s, h: bx.s } }];
        items.push(H().text('bx:c1', BX(Cv), bx.y + bx.s + 16, 'C', 'tick', 12, 'middle'), H().text('bx:c0', bx.x, bx.y + bx.s + 16, '0', 'tick', 12, 'middle'), H().text('bx:c2', bx.x - 6, bx.y + 4, 'C', 'tick', 12, 'end'));
        if (!s.step) { items.push(H().text('bx:t', bx.x + bx.s + 14, bx.y + 14, 'the pair’s box', 'axname', 13)); return items; }
        const { i, j } = s.step, same = yy[i] === yy[j], k = same ? s.oi + s.oj : s.oi - s.oj;
        // the constraint line through the box: same labels αi + αj = k, else αi − αj = k
        const ends = [];
        for (const ai of [0, Cv]) { const aj = same ? k - ai : ai - k; if (aj >= -1e-12 && aj <= Cv + 1e-12) ends.push([ai, aj]); }
        for (const aj of [0, Cv]) { const ai = same ? k - aj : k + aj; if (ai >= -1e-12 && ai <= Cv + 1e-12) ends.push([ai, aj]); }
        ends.sort((p, r) => p[0] - r[0]);
        if (ends.length >= 2) items.push({ key: 'bx:seg', kind: 'path', d: `M${BX(ends[0][0])},${BY(ends[0][1])} L${BX(ends[ends.length - 1][0])},${BY(ends[ends.length - 1][1])}`, cls: 'mlseg', pathAudit: 'edge-x', layer: 'lines' });
        items.push({ key: 'bx:old', kind: 'dot', x: BX(s.oi), y: BY(s.oj), r: 4.5, cls: 'mark muted', point: true });
        items.push({ key: 'bx:new', kind: 'dot', x: BX(s.alpha[i]), y: BY(s.alpha[j]), r: 6, cls: 'mark', point: true });
        items.push(H().text('bx:xi', bx.x + bx.s / 2, bx.y + bx.s + 34, 'α of point ' + i, 'axname', 13, 'middle'));
        items.push(H().text('bx:yj', bx.x + bx.s + 14, bx.y + bx.s / 2 + 4, 'α of point ' + j, 'axname', 13));
        return items;
      };
      const bars = (s) => {
        const items = [H().text('br:t', 520, 260, 'every α, against the cap C', 'axname', 13)], w = Math.min(18, 260 / n), h = 70, y0 = 280;
        items.push({ key: 'br:cap', kind: 'path', d: `M520,${y0} h${H().r2(n * w)}`, cls: 'guide act', audit: 'ignore', layer: 'back' });
        items.push({ key: 'br:base', kind: 'path', d: `M520,${y0 + h} h${H().r2(n * w)}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        s.alpha.forEach((v, t) => {
          const bh = Math.max(1, (v / Cv) * h), lit = s.step && (t === s.step.i || t === s.step.j);
          items.push({ key: 'br:b' + t, kind: 'cell', x: H().r2(520 + t * w + 1), y: H().r2(y0 + h - bh), w: H().r2(w - 2), h: H().r2(bh), rx: 1, text: '', cls: 'bar c' + y[t] + (lit ? '' : ' soft'), audit: 'ignore' });
        });
        return items;
      };
      const plot = (s) => {
        const items = [...ax.items], any = s.alpha.some((v) => v > 0);
        if (any) items.push(...P.boundary(ax, 'sm', { f: fOf(s) }, { n: 60 }));
        items.push(...S.Kit.scatter.points(ax, 'sm:p', X, y, { r: 5 }));
        s.alpha.forEach((v, t) => { if (v > 0) items.push(S.Kit.scatter.ring(ax, 'sm:r' + t, X[t][0], X[t][1], v < cap ? 'act' : '', 9)); });
        if (s.step) for (const t of [s.step.i, s.step.j]) items.push(S.Kit.scatter.ring(ax, 'sm:h' + t, X[t][0], X[t][1], 'hot', 13));
        return items;
      };
      const all = () => [...Array(st.length).keys()];
      const show = st.length <= 26 ? all() : [...new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...Array.from({ length: 10 }, (_, k) => Math.round(10 + (k * (st.length - 14)) / 10)), st.length - 4, st.length - 3, st.length - 2, st.length - 1])].filter((k) => k < st.length);
      F.push([...plot(st[0]), ...boxView(st[0]), ...bars(st[0])], `All ${q.n(n)} multipliers start at α = 0, which meets every constraint but solves nothing (dual objective ${q(st[0].dual)}). The solver labels blue +1 and orange −1, and with ${inp.kernel === 'rbf' ? 'the RBF kernel' : 'the linear kernel'} and C = ${q.sg(Cv, 2)} it will need ${q.n(steps.length)} steps.`, { phase: 'start', n, C: Cv, steps: steps.length, dual: st[0].dual });
      show.slice(1).forEach((k) => {
        const s = st[k], { i, j, gap } = s.step, clipped = [s.alpha[i], s.alpha[j]].some((v) => v <= 0 || v >= cap);
        F.set('steps', k).set('sv', s.alpha.filter((v) => v > 0).length);
        const skipped = k - (show[show.indexOf(k) - 1] || 0) - 1, skip = skipped > 0 ? `After ${q.n(skipped)} more ${skipped === 1 ? 'step' : 'steps'}, step ${q.n(k)}` : `Step ${q.n(k)}`;
        const name = (t) => (yy[t] > 0 ? 'blue' : 'orange') + ' point ' + q.n(t);
        F.push([...plot(s), ...boxView(s), ...bars(s)], `${skip}: the pair ${name(i)} and ${name(j)} breaks the optimality conditions most, by a gap of ${q.sg(gap, 3)}. Moving along their line, α of point ${q.n(i)} goes ${q.sg(s.oi, 4)} → ${q.sg(s.alpha[i], 4)} and α of point ${q.n(j)} ${q.sg(s.oj, 4)} → ${q.sg(s.alpha[j], 4)}${clipped ? ', stopped at the edge of the box' : ''}. Dual objective ${q(s.dual, 4)}.`, { phase: 'step', k, skipped, i, j, gap, oi: s.oi, oj: s.oj, ai: s.alpha[i], aj: s.alpha[j], dual: s.dual, alpha: s.alpha, prev: st[k - 1].alpha });
      });
      const fin = Object.assign({}, st[st.length - 1], { step: null }), nsv = fin.alpha.filter((v) => v > 0).length;
      F.push([...plot(fin), ...boxView(fin), ...bars(fin)], `No pair breaks the conditions by more than the tolerance ${q.sg(1e-3, 1)}, so SMO stops after ${q.n(steps.length)} steps with dual objective ${q(fin.dual, 4)} and ${q.n(nsv)} multipliers above zero: the support vectors, ringed. Every other α is still exactly 0.`, { phase: 'done', steps: steps.length, dual: fin.dual, nsv, tol: 1e-3, alpha: fin.alpha });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 440, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'SMO', text: ['maximise D(α) = Σ αᵢ − ½ Σ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ)', 'subject to 0 ≤ αᵢ ≤ C, Σ αᵢyᵢ = 0', 'pick the pair (i, j) that violates the', '  optimality conditions most', 'move along yᵢΔαᵢ + yⱼΔαⱼ = 0 to the best point,', '  clipped to the box [0, C]²', 'stop when the largest violation < tolerance'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'step' && fr[i + 1] && fr[i + 1].meta.phase === 'step', once: true, pick: 'option', ask: 'Will the dual objective at the next step be higher, lower or the same?', options: ['higher or the same', 'lower'], answer: (fr, i) => (fr[i + 1].meta.dual >= fr[i].meta.dual - 1e-12 ? 'higher or the same' : 'lower'), explain: 'Every SMO step solves its two-variable problem exactly, so the dual objective never falls.' },
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Why does SMO change two multipliers at a time, never one?', options: ['Σ αᵢyᵢ = 0 would break if only one moved', 'one at a time is too slow', 'the kernel needs pairs', 'C limits each step to two'], answer: 'Σ αᵢyᵢ = 0 would break if only one moved', explain: 'Moving a single α changes Σ αᵢyᵢ; a second one must move to put it back.' },
    ],
    quiz: [
      { q: 'The pair has yᵢ = yⱼ and αᵢ + αⱼ = 0.6, with C = 0.5. Which values can (αᵢ, αⱼ) take?', options: ['from (0.1, 0.5) to (0.5, 0.1)', 'from (0, 0.6) to (0.6, 0)', 'only (0.3, 0.3)', 'anything in [0, 0.5]²'], answer: 0, why: 'They stay on αᵢ + αⱼ = 0.6 inside the box: neither may exceed 0.5, so neither may fall below 0.1.' },
      { q: 'What does SMO do with the best point on the pair’s line if it lies outside the box?', options: ['clips it to the nearest end of the line inside the box', 'rejects the pair', 'raises C', 'moves a third multiplier'], answer: 0, why: 'The objective is a parabola along the line, so the best feasible point is the clipped one.' },
      { q: 'When does SMO stop?', options: ['when no pair violates the optimality conditions by more than a tolerance', 'after n steps', 'when every α equals C', 'when the training accuracy is 100%'], answer: 0, why: 'The gap measures how far the current α is from meeting the conditions.' },
      { q: 'Which multipliers are usually still exactly 0 at the end?', options: ['those of points outside the street on their own side', 'those of the support vectors', 'those of the misclassified points', 'none'], answer: 0, why: 'Points that need no say keep α = 0, which is why SVMs are sparse.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
