// Part 8, optimisation: the exercises. Every number is worked out here by the engines (the optimisers and schedules of
// S.T, run on the loss surfaces of the part's topics through S.P08.run), so the text, the answers and the hints agree.
(function (S) {
  'use strict';
  const C = S.Content, T = S.T, P = S.P08, WF = S.WF, ML = S.ML;
  if (!C || !T || !P || !WF || C.exercises.has('op-x-grad')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const z = (v) => (Math.abs(v) < 1e-12 ? 0 : v);
  const fv = (p, r) => '(' + p.map((v) => (r || r3)(z(v))).join(', ') + ')';
  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });
  const ex = (x) => C.exercise(x);
  const SF = P.SURF, rav = SF.ravine, tilt = SF.tilted;

  // =========================================================================================================
  // Chapter 1: going downhill
  // =========================================================================================================
  const GP = [1, -2], gt = tilt.g(...GP);
  ex({ id: 'op-x-grad', chapter: 'op-descent', title: 'A gradient by hand', level: 1, kind: 'table',
    text: ['The tilted valley has loss ' + tilt.formula + '. Fill in its gradient (∂L/∂u, ∂L/∂v) at (u, v) = (1, −2).'],
    rows: ['∂L/∂u', '∂L/∂v'], cols: ['value'], corner: 'at (1, −2)', cells: { '∂L/∂u|value': r3(gt[0]), '∂L/∂v|value': r3(gt[1]) }, tolerance: 0.0005,
    hints: ['∂L/∂u = 1.2u + 0.8v and ∂L/∂v = 0.8u + 1.2v (the ½ cancels the 2 from each square).', '∂L/∂u = 1.2 · 1 + 0.8 · (−2).', 'The gradient is ' + fv(gt) + ': the loss rises fastest that way, so a descent step goes the other way.'],
    solution: { '∂L/∂u|value': String(r3(gt[0])), '∂L/∂v|value': String(r3(gt[1])) },
    wrong: [{ '∂L/∂u|value': String(r3(2 * gt[0])), '∂L/∂v|value': String(r3(2 * gt[1])) }, { '∂L/∂u|value': String(r3(1.2 + 1.6 * -2)), '∂L/∂v|value': String(r3(1.6 + 1.2 * -2)) }] });

  const hev = ML.la.symEig(tilt.h(0, 0)).values, cond = Math.max(...hev) / Math.min(...hev);
  ex({ id: 'op-x-hessian', chapter: 'op-descent', title: 'How badly conditioned?', level: 2, kind: 'number',
    text: ['The tilted valley ' + tilt.formula + ' has the same Hessian everywhere, [[1.2, 0.8], [0.8, 1.2]]. What is the ratio of its largest eigenvalue to its smallest, the condition number that says how much of a ravine it is?'],
    answer: r3(cond), tolerance: 0.0005, whyNot: { '1.5': 'That is the ratio of the diagonal entries; the eigenvalues include the effect of the 0.8 off the diagonal.', '20': 'That is the long ravine, 0.1u² + 2v², with eigenvalues 4 and 0.2.' },
    hints: ['For [[a, b], [b, a]] the eigenvalues are a + b and a − b.', 'Here 1.2 + 0.8 = ' + r3(Math.max(...hev)) + ' and 1.2 − 0.8 = ' + r3(Math.min(...hev)) + '.', 'The ratio is ' + r3(Math.max(...hev)) + ' / ' + r3(Math.min(...hev)) + ' = ' + r3(cond) + ': a gentle valley beside the long ravine’s 20.'], solution: String(r3(cond)), wrong: ['1.5', '20'] });

  const g1 = P.run(rav, 'sgd', { lr: 0.1 }, 1), p1 = g1.pts[1], l1 = g1.loss[1];
  ex({ id: 'op-x-gd-step', chapter: 'op-descent', title: 'One step down the ravine', level: 1, kind: 'table',
    text: ['The ravine has loss ' + rav.formula + ', so its gradient is (0.2u, 4v). Take one gradient-descent step with η = 0.1 from (−4, 1.6).'],
    rows: ['u', 'v'], cols: ['gradient', 'after the step'], corner: 'from (−4, 1.6)',
    cells: { 'u|gradient': r3(g1.grads[0][0]), 'v|gradient': r3(g1.grads[0][1]), 'u|after the step': r3(p1[0]), 'v|after the step': r3(p1[1]) }, tolerance: 0.0005,
    hints: ['The gradient at (−4, 1.6) is (0.2 · (−4), 4 · 1.6).', 'It is (−0.8, 6.4); the step is −η times it, (0.08, −0.64).', 'The new point is ' + fv(p1) + '. The v coordinate moved eight times as far as u: the slope is steeper across the ravine.'],
    solution: { 'u|gradient': String(r3(g1.grads[0][0])), 'v|gradient': String(r3(g1.grads[0][1])), 'u|after the step': String(r3(p1[0])), 'v|after the step': String(r3(p1[1])) },
    wrong: [{ 'u|gradient': '-0.8', 'v|gradient': '6.4', 'u|after the step': '-4.08', 'v|after the step': '2.24' }, { 'u|gradient': '-0.8', 'v|gradient': '6.4', 'u|after the step': '-4.8', 'v|after the step': '8' }] });

  ex({ id: 'op-x-gd-loss', chapter: 'op-descent', title: 'The loss after the step', level: 1, kind: 'number',
    text: ['After one step from (−4, 1.6) with η = 0.1, gradient descent on the ravine ' + rav.formula + ' is at ' + fv(p1) + '. What is the loss there? (It was 6.72 at the start.) Give three decimals.'],
    answer: r3(l1), tolerance: 0.0015, whyNot: { '6.72': 'That is the loss at the start.', [String(r3(0.1 * p1[0] + 2 * p1[1]))]: 'Square u and v before multiplying.' },
    hints: ['L = 0.1u² + 2v².', '0.1 · 3.92² = ' + r3(0.1 * p1[0] ** 2) + ' and 2 · 0.96² = ' + r3(2 * p1[1] ** 2) + '.', 'L = ' + r3(l1) + ', down from 6.72: almost all of the drop came from v.'], solution: String(r3(l1)), wrong: ['6.72', String(r3(0.1 * p1[0] + 2 * p1[1]))] });

  const tr = P.run(tilt, 'sgd', { lr: 0.3 }, 25), tcurve = tr.loss.map((v, k) => [k, r3(v)]).filter(([k]) => k % 2 === 0 || k === 25);
  ex({ id: 'op-x-gd-curve', chapter: 'op-descent', title: 'The loss as descent goes on', level: 2, kind: 'predict-curve',
    text: ['Gradient descent with η = 0.3 walks down the tilted valley from (−2.5, 0.6), where the loss is 2.766. Sketch the loss over the first 25 steps.'],
    plot: { xr: [0, 26], yr: [0, 3], xname: 'step', yname: 'loss', given: tcurve.slice(0, 1), note: 'Drag from step 0 to step 25. The faint point is the start.' },
    answer: tcurve, tolerance: 0.15, checks: ['falling'],
    hints: ['Steps are long on steep ground and short near the bottom.', 'The first step alone takes the loss from 2.766 to 1.033.', 'It falls fast and then flattens: ' + tcurve.filter(([k]) => [0, 2, 6, 12, 24].includes(k)).map(([k, v]) => 'step ' + k + ': ' + v).join(', ') + '.'],
    solution: tcurve, wrong: [tcurve.map(([k]) => [k, r3(2.766 * (1 - k / 25))]), tcurve.map(([k]) => [k, 1.5])] });

  const gdRef = (p, q, start, lr, steps) => P.run({ f: (u, v) => 0.5 * (p * u * u + q * v * v), g: (u, v) => [p * u, q * v], start }, 'sgd', { lr }, steps).pts[steps];
  ex({ id: 'op-x-code-gd', chapter: 'op-descent', title: 'Write gradient descent', level: 2, kind: 'code',
    text: ['The loss L(u, v) = ½(p·u² + q·v²) has gradient (p·u, q·v). Write `descend(p, q, start, lr, steps)`: start is [u, v]; take that many gradient-descent steps with learning rate lr and return the final [u, v].'],
    fn: 'descend', params: ['p', 'q', 'start', 'lr', 'steps'], start: 'function descend(p, q, start, lr, steps) {\n  \n}', tolerance: 1e-9, ref: gdRef,
    cases: [[0.2, 4, [-4, 1.6], 0.1, 1], [0.2, 4, [-4, 1.6], 0.1, 10], [1, 1, [-2.4, 1.8], 0.5, 3], [1, 1, [-2.4, 1.8], 1, 1], [2, 0.5, [1, 1], 0.9, 20], [3, 1, [0.5, -0.5], 0.7, 5]],
    hints: ['Keep u and v in variables and loop steps times.', 'In each step compute both gradient parts from the current u and v first, then update both: u ← u − lr·p·u and v ← v − lr·q·v.', 'function descend(p, q, start, lr, steps) {\n  let [u, v] = start;\n  for (let k = 0; k < steps; k++) {\n    const gu = p * u, gv = q * v;\n    u -= lr * gu; v -= lr * gv;\n  }\n  return [u, v];\n}'],
    solution: 'function descend(p, q, start, lr, steps) {\n  let [u, v] = start;\n  for (let k = 0; k < steps; k++) {\n    const gu = p * u, gv = q * v;\n    u -= lr * gu; v -= lr * gv;\n  }\n  return [u, v];\n}',
    wrong: ['function descend(p, q, start, lr, steps) { let [u, v] = start; for (let k = 0; k < steps; k++) { u += lr * p * u; v += lr * q * v; } return [u, v]; }', 'function descend(p, q, start, lr, steps) { let [u, v] = start; for (let k = 0; k < steps; k++) { u -= lr * u; v -= lr * v; } return [u, v]; }'] });

  const LH = [3, 0.5], lmax = 2 / Math.max(...LH);
  ex({ id: 'op-x-lr-limit', chapter: 'op-descent', title: 'The largest safe step', level: 2, kind: 'number',
    text: ['Gradient descent on L(u, v) = 1.5u² + 0.25v². Above what learning rate does it diverge? Give three decimals.'],
    answer: r3(lmax), tolerance: 0.0015, whyNot: { '4': 'That is 2 / λmin; the steep direction, with the largest curvature, sets the limit.', [String(r3(1 / 3))]: 'That is 1 / λmax, the rate that lands on the floor of the steep direction in one step; the walk only diverges past twice that.', [String(r3(2 / 1.5))]: 'The curvature is the second derivative: 1.5u² has curvature 3, not 1.5.' },
    hints: ['The curvatures are the second derivatives: 3 for u and 0.5 for v.', 'Each step multiplies the error along a direction of curvature λ by 1 − ηλ; it must stay between −1 and 1.', 'The steep direction needs η < 2 / 3 = ' + r3(lmax) + '.'], solution: String(r3(lmax)), wrong: ['4', String(r3(1 / 3))] });

  ex({ id: 'op-x-lr-choice', chapter: 'op-descent', title: 'A loss that climbs', level: 1, kind: 'choice',
    text: ['A model is trained by gradient descent. Its training loss goes 2.1, 1.4, 1.9, 1.2, 2.6, 3.8, 7.9, 16.4, … What should you change first?'],
    options: ['Lower the learning rate', 'Raise the learning rate', 'Train for more steps', 'Add more parameters'], answer: 0,
    whyNot: { 1: 'The steps already overshoot; larger ones overshoot more.', 2: 'More steps of a diverging walk only climb further.', 3: 'The model’s size is not the problem: the steps are.' },
    hints: ['The loss falls, then rises with growing swings.', 'That is the walk jumping over the valley floor by more each time: the factor 1 − ηλ is beyond −1.', 'The learning rate is above 2/λmax: lower it.'], solution: 0, wrong: [1, 2] });

  const BX = [1, 2, 3], BY = [2, 3, 7], BA = 1, BB = 0, bres = BX.map((x, i) => BA * x + BB - BY[i]), bga = ML.mean(bres.map((r, i) => r * BX[i]));
  ex({ id: 'op-x-batch-grad', chapter: 'op-descent', title: 'A mini-batch gradient', level: 2, kind: 'number',
    text: ['A line y = a·x + b is fitted by half the mean squared error, L = ½ · mean((a·x + b − y)²). The current line has a = 1 and b = 0. A mini-batch holds three points: (1, 2), (2, 3) and (3, 7). What is ∂L/∂a on this mini-batch?'],
    answer: r3(bga), tolerance: 0.0005, whyNot: { [String(r3(-bga))]: 'Mind the sign: the residuals a·x + b − y are all negative here.', [String(r3(3 * bga))]: 'Take the mean over the batch, not the sum.', [String(r3(ML.mean(bres)))]: 'That is ∂L/∂b; for a, each residual is multiplied by its x.' },
    hints: ['∂L/∂a = mean((a·x + b − y) · x).', 'The residuals are ' + bres.join(', ') + '.', '(−1 · 1 + (−1) · 2 + (−4) · 3) / 3 = ' + r3(bga) + ': increasing a lowers this batch’s loss.'], solution: String(r3(bga)), wrong: [String(r3(-bga)), String(r3(3 * bga))] });

  ex({ id: 'op-x-sgd-choice', chapter: 'op-descent', title: 'Why stochastic descent never settles', level: 2, kind: 'choice',
    text: ['Stochastic descent on single examples, with a fixed learning rate, gets near the best fit quickly but then keeps wandering round it. Why?'],
    options: ['At the best fit the full gradient is zero, but each single example’s gradient is not, so every step still moves', 'The learning rate is too small', 'The loss surface changes as training goes on', 'Rounding errors in the computer'], answer: 0,
    whyNot: { 1: 'A smaller rate would make the wandering smaller, not cause it.', 2: 'The loss over the training set is fixed; only the example used at each step changes.', 3: 'The wandering is far larger than rounding: in the lesson, 0.1478 against the best 0.1206.' },
    hints: ['What is the gradient of the full loss at its minimum?', 'And the gradient of one example’s loss there?', 'The examples disagree: each pulls its own way even at the best fit. Only a falling rate, or larger batches, quiet them.'], solution: 0, wrong: [1, 3] });

  // =========================================================================================================
  // Chapter 2: faster ways down
  // =========================================================================================================
  const mo = P.run(rav, 'momentum', { lr: 0.1, momentum: 0.9 }, 2), mcells = {};
  ['step 1', 'step 2'].forEach((s, k) => { mcells[s + '|velocity u'] = r4(z(mo.state[k].v[0])); mcells[s + '|velocity v'] = r4(z(mo.state[k].v[1])); mcells[s + '|u after'] = r4(z(mo.pts[k + 1][0])); mcells[s + '|v after'] = r4(z(mo.pts[k + 1][1])); });
  ex({ id: 'op-x-momentum-table', chapter: 'op-faster', title: 'Two momentum steps', level: 3, kind: 'table',
    text: ['Momentum on the ravine ' + rav.formula + ' (gradient (0.2u, 4v)), from (−4, 1.6) with η = 0.1 and β = 0.9. The velocity starts as the first gradient; after that v ← βv + ∇L. Each step is θ ← θ − ηv. Fill in two steps.'],
    rows: ['step 1', 'step 2'], cols: ['velocity u', 'velocity v', 'u after', 'v after'], corner: '', cells: mcells, tolerance: 0.0015,
    hints: ['Step 1: v = ∇L(−4, 1.6) = (−0.8, 6.4), so θ = (−4, 1.6) − 0.1 · v.', 'Step 2: the gradient at ' + fv(mo.pts[1]) + ' is ' + fv(mo.grads[1]) + '; v = 0.9 · (−0.8, 6.4) + that.', 'Step 1: v = ' + fv(mo.state[0].v, r4) + ', θ = ' + fv(mo.pts[1], r4) + '. Step 2: v = ' + fv(mo.state[1].v, r4) + ', θ = ' + fv(mo.pts[2], r4) + ': the velocity across the ravine took v exactly to the floor.'],
    solution: Object.fromEntries(Object.entries(mcells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(mcells).map(([k, v]) => [k, String(v)])), { 'step 2|velocity u': String(r4(mo.grads[1][0])), 'step 2|velocity v': String(r4(mo.grads[1][1])), 'step 2|u after': String(r4(mo.pts[1][0] - 0.1 * mo.grads[1][0])), 'step 2|v after': String(r4(mo.pts[1][1] - 0.1 * mo.grads[1][1])) })] });

  ex({ id: 'op-x-ravine-choice', chapter: 'op-faster', title: 'An optimiser for a ravine', level: 1, kind: 'choice',
    text: ['A loss is a long, narrow ravine: steep across, nearly flat along. Plain gradient descent zigzags across it and creeps along the floor. What helps most?'],
    options: ['Momentum (or an adaptive method such as Adam)', 'Plain gradient descent with a larger rate', 'Plain gradient descent with a smaller rate', 'Starting from a random point each step'], answer: 0,
    whyNot: { 1: 'The rate is already limited by the steep direction: any larger and the zigzag explodes.', 2: 'A smaller rate tames the zigzag but makes the creep along the floor even slower.', 3: 'That throws away all progress.' },
    hints: ['The trouble is the mismatch between the two directions.', 'Across the ravine the gradient flips sign every step; along it, it always points the same way.', 'Momentum lets the flips cancel and the steady drift build up; an adaptive method gives each direction its own step. In the lesson momentum reached 5.13 × 10⁻⁴ where plain descent was still at 0.318.'], solution: 0, wrong: [1, 2] });

  const ad = P.run(rav, 'adam', { lr: 0.3 }, 2), acells = {}, arows = ['u', 'v'];
  arows.forEach((c, i) => { acells[c + '|m'] = r4(ad.state[0].m[i]); acells[c + '|s (mean of g²)'] = r4(ad.state[0].v[i]); acells[c + '|after the step'] = r4(ad.pts[1][i]); });
  ex({ id: 'op-x-adam-step', chapter: 'op-faster', title: 'Adam’s first step', level: 2, kind: 'table',
    text: ['Adam on the ravine from (−4, 1.6), where the gradient is (−0.8, 6.4), with η = 0.3, β₁ = 0.9 and β₂ = 0.999. Both moving averages start at 0: m ← β₁m + (1 − β₁)g, s ← β₂s + (1 − β₂)g². Fill in m, s and the new point after the first step (four decimals).'],
    rows: arows, cols: ['m', 's (mean of g²)', 'after the step'], corner: 'coordinate', cells: acells, tolerance: 0.00015,
    hints: ['m = 0.1 · g and s = 0.001 · g² after one step.', 'The bias corrections divide them by 1 − 0.9 = 0.1 and 1 − 0.999 = 0.001, so m̂ = g and ŝ = g²: the step is −η · g / |g|.', 'm = ' + fv(ad.state[0].m, r4) + ', s = ' + fv(ad.state[0].v, r4) + ', and each coordinate moves by 0.3 against its gradient: ' + fv(ad.pts[1], r4) + '.'],
    solution: Object.fromEntries(Object.entries(acells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(acells).map(([k, v]) => [k, String(v)])), { 'u|after the step': String(r4(-4 + 0.3 * 0.8)), 'v|after the step': String(r4(1.6 - 0.3 * 6.4)) }), Object.assign(Object.fromEntries(Object.entries(acells).map(([k, v]) => [k, String(v)])), { 'u|m': '-0.8', 'v|m': '6.4' })] });

  const c1 = 1 - 0.9 ** 2, c2 = 1 - 0.999 ** 2, mh = ad.state[1].m.map((v) => v / c1), vh = ad.state[1].v.map((v) => v / c2), a2 = {};
  arows.forEach((c, i) => { a2[c + '|m̂'] = r4(mh[i]); a2[c + '|ŝ'] = r4(vh[i]); a2[c + '|after the step'] = r4(ad.pts[2][i]); });
  ex({ id: 'op-x-adam-two', chapter: 'op-faster', title: 'Adam’s second step', level: 3, kind: 'table',
    text: ['Continue Adam from the last exercise. It is at (−3.7, 1.3) with m = (−0.08, 0.64) and s = (0.00064, 0.04096); the gradient there is ' + fv(ad.grads[1]) + '. This is step t = 2, so the corrections divide by 1 − 0.9² = 0.19 and 1 − 0.999² = 0.001999. Fill in m̂, ŝ and the new point (four decimals; ε is negligible).'],
    rows: arows, cols: ['m̂', 'ŝ', 'after the step'], corner: 'coordinate', cells: a2, tolerance: 0.0006,
    hints: ['First the averages: m = 0.9 · m + 0.1 · g and s = 0.999 · s + 0.001 · g².', 'm = ' + fv(ad.state[1].m, r4) + ' and s = (' + ad.state[1].v.map((v) => v.toFixed(6)).join(', ') + '); divide by 0.19 and 0.001999.', 'm̂ = ' + fv(mh, r4) + ', ŝ = ' + fv(vh, r4) + ', and θ = (−3.7, 1.3) − 0.3 · m̂ / √ŝ = ' + fv(ad.pts[2], r4) + ': again about 0.3 in each coordinate.'],
    solution: Object.fromEntries(Object.entries(a2).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(a2).map(([k, v]) => [k, String(v)])), { 'u|m̂': String(r4(ad.state[1].m[0])), 'v|m̂': String(r4(ad.state[1].m[1])) })] });

  const rm = P.run(rav, 'rmsprop', { lr: 0.1 }, 1), rlen = Math.abs(rm.pts[1][1] - rm.pts[0][1]);
  ex({ id: 'op-x-rmsprop', chapter: 'op-faster', title: 'RMSProp’s first step', level: 2, kind: 'number',
    text: ['RMSProp with η = 0.1 and ρ = 0.99 starts its average of squared gradients at 0: s ← 0.99s + 0.01g². Its first gradient’s v coordinate is 6.4. How far does v move in the first step, η · g / √s? Give two decimals.'],
    answer: r2(rlen), tolerance: 0.005, whyNot: { '0.1': 'That would need s = g²; after one step s is only 0.01g², so √s is a tenth of |g|.', '0.64': 'That is plain gradient descent, η · g.' },
    hints: ['After one step s = 0.01 · 6.4² = 0.4096.', '√s = 0.64, a tenth of the gradient.', 'The step is 0.1 · 6.4 / 0.64 = ' + r2(rlen) + ': ten times η, because nothing corrects the average for starting at zero. Adam’s bias correction fixes exactly this.'], solution: String(r2(rlen)), wrong: ['0.1', '0.64'] });

  const ag = P.run(SF.bowl, 'adagrad', { lr: 0.5 }, 2, { grad: (p, k) => [[3, 4][k], 0] }), agStep = Math.abs(ag.pts[2][0] - ag.pts[1][0]);
  ex({ id: 'op-x-adagrad', chapter: 'op-faster', title: 'AdaGrad’s shrinking steps', level: 2, kind: 'number',
    text: ['AdaGrad with η = 0.5 adds up the squared gradients of each coordinate, h ← h + g², and steps by η · g / √h. One coordinate sees the gradient 3, then 4. How long is its second step?'],
    answer: r3(agStep), tolerance: 0.0005, whyNot: { '0.5': 'That is the first step, 0.5 · 3 / √9, and also what you get from the second gradient alone. h is a running sum: 9 + 16 = 25.', '2': 'Divide by √h.' },
    hints: ['After the first step h = 9; after the second, 9 + 16.', '√25 = 5.', 'The step is 0.5 · 4 / 5 = ' + r3(agStep) + ', though the gradient grew: AdaGrad’s steps shrink as the sum grows.'], solution: String(r3(agStep)), wrong: ['0.5', '2'] });

  const adamRef = (w, g, m, v, t, lr) => T.float64(() => {
    const p = T.tensor(w.slice()), opt = T.optim.make('adam', [p], { lr });
    opt.state.set(p, { m: Float64Array.from(m), v: Float64Array.from(v), t: t - 1 });
    p.grad = T.tensor(g.slice()); opt.step();
    const s = opt.state.get(p);
    return [Array.from(p.values()), Array.from(s.m), Array.from(s.v)];
  });
  ex({ id: 'op-x-code-adam', chapter: 'op-faster', title: 'Write one Adam step', level: 3, kind: 'code',
    text: ['Write `adamStep(w, g, m, v, t, lr)`: w are the parameters, g their gradient, m and v the moving averages (all arrays of the same length), t the number of this step (1 for the first) and lr the learning rate. Use β₁ = 0.9, β₂ = 0.999 and ε = 1e−8. Return [new w, new m, new v], without changing the arrays you were given.'],
    fn: 'adamStep', params: ['w', 'g', 'm', 'v', 't', 'lr'], start: 'function adamStep(w, g, m, v, t, lr) {\n  \n}', tolerance: 1e-9, ref: adamRef,
    cases: [[[-4, 1.6], [-0.8, 6.4], [0, 0], [0, 0], 1, 0.3], [[-3.7, 1.3], [-0.74, 5.2], [-0.08, 0.64], [0.00064, 0.04096], 2, 0.3], [[1], [0.5], [0.1], [0.02], 10, 0.01], [[0, 0, 0], [1, -2, 0], [0, 0, 0], [0, 0, 0], 1, 0.001], [[2, -1], [0.3, 0.3], [0.2, -0.1], [0.05, 0.01], 3, 0.05]],
    hints: ['Update the averages first: m = 0.9m + 0.1g and v = 0.999v + 0.001g², coordinate by coordinate.', 'Correct them: m̂ = m / (1 − 0.9ᵗ) and v̂ = v / (1 − 0.999ᵗ). Then w − lr · m̂ / (√v̂ + ε).', 'function adamStep(w, g, m, v, t, lr) {\n  const b1 = 0.9, b2 = 0.999, eps = 1e-8;\n  const m2 = m.map((x, i) => b1 * x + (1 - b1) * g[i]);\n  const v2 = v.map((x, i) => b2 * x + (1 - b2) * g[i] * g[i]);\n  const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);\n  const w2 = w.map((x, i) => x - (lr * (m2[i] / c1)) / (Math.sqrt(v2[i] / c2) + eps));\n  return [w2, m2, v2];\n}'],
    solution: 'function adamStep(w, g, m, v, t, lr) {\n  const b1 = 0.9, b2 = 0.999, eps = 1e-8;\n  const m2 = m.map((x, i) => b1 * x + (1 - b1) * g[i]);\n  const v2 = v.map((x, i) => b2 * x + (1 - b2) * g[i] * g[i]);\n  const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);\n  const w2 = w.map((x, i) => x - (lr * (m2[i] / c1)) / (Math.sqrt(v2[i] / c2) + eps));\n  return [w2, m2, v2];\n}',
    wrong: ['function adamStep(w, g, m, v, t, lr) { const m2 = m.map((x, i) => 0.9 * x + 0.1 * g[i]); const v2 = v.map((x, i) => 0.999 * x + 0.001 * g[i] * g[i]); const w2 = w.map((x, i) => x - (lr * m2[i]) / (Math.sqrt(v2[i]) + 1e-8)); return [w2, m2, v2]; }',
      'function adamStep(w, g, m, v, t, lr) { const m2 = m.map((x, i) => 0.9 * x + 0.1 * g[i]); const v2 = v.map((x, i) => 0.999 * x + 0.001 * g[i]); const c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t); const w2 = w.map((x, i) => x - (lr * (m2[i] / c1)) / (Math.sqrt(Math.abs(v2[i]) / c2) + 1e-8)); return [w2, m2, v2]; }'] });

  ex({ id: 'op-x-plateau-choice', chapter: 'op-faster', title: 'Off a plateau', level: 1, kind: 'choice',
    text: ['A run starts far out on a plateau, where the loss is 0.9993 and the gradient only 0.00275 long. Which optimiser gets below loss 0.5 first?'],
    options: ['Adam', 'Plain gradient descent', 'They all take the same number of steps'], answer: 0,
    whyNot: { 1: 'Plain descent moves η times the tiny gradient: it took 114 steps in the lesson.', 2: 'They differ a great deal: 114, 29 and 24 steps.' },
    hints: ['Plain descent’s step is η times the gradient, which is tiny here.', 'Adam divides the gradient by the root of its own average square, so a small but steady gradient still gives a step of about η.', 'In the lesson: plain descent at step 114, momentum at 29, Adam at 24.'], solution: 0, wrong: [1, 2] });

  const rings = { data: { set: 'circles', n: 300, seed: 4, opts: { noise: 0.08 } }, hidden: { seed: 44, n: 600 }, metric: 'accuracy', target: 0.95, task: 'classification' };
  C.exercise(Object.assign({ id: 'op-x-wf-momentum', chapter: 'op-faster', title: 'A ring in fifteen epochs', level: 3, kind: 'workflow',
    text: ['A small neural network must learn a ring round a disc in at most 15 epochs of training. Choose the optimiser and the learning rate in the Train block so that it classifies 600 hidden points with accuracy at least 0.95.'],
    limits: { blocks: ['split', 'mlp', 'train', 'evaluate'], models: ['mlp'], epochs: 15 },
    hints: ['Train with the defaults first and look at the loss curve in the Train block: is it still falling steeply at the last epoch?', 'Fifteen epochs are few. Plain SGD at a small rate barely moves; momentum or Adam with a larger rate goes much further per epoch.', 'Split → MLP (8, 8, tanh) → Train (momentum, rate 0.1, 15 epochs) → Evaluate: accuracy 0.993, against 0.57 for plain SGD at the same rate.'],
    solution: pipe(rings.data, 'mlp', { hidden: [8, 8], act: 'tanh' }, { epochs: 15, optimizer: 'momentum', lr: 0.1 }),
    wrong: [pipe(rings.data, 'mlp', { hidden: [8, 8], act: 'tanh' }, { epochs: 15, optimizer: 'sgd', lr: 0.1 }), pipe(rings.data, 'mlp', { hidden: [8, 8], act: 'tanh' }, { epochs: 15, optimizer: 'adam', lr: 0.01 })] }, rings));

  // =========================================================================================================
  // Chapter 3: schedules
  // =========================================================================================================
  const stp = T.sched.step({ lr: 0.3, every: 20, gamma: 0.3 })(45);
  ex({ id: 'op-x-step-sched', chapter: 'op-sched', title: 'A step-decay rate', level: 1, kind: 'number',
    text: ['A step schedule starts at η = 0.3 and multiplies the rate by 0.3 every 20 steps (steps counted from 0). What is the rate at step 45?'],
    answer: r4(stp), tolerance: 0.00005, whyNot: { '0.09': 'That is after one drop; by step 45 there have been two, at steps 20 and 40.', [String(r4(0.3 * 0.3 ** 2.25))]: 'The drops come in whole steps: ⌊45 / 20⌋ = 2.', '0.0081': 'Only two drops so far, not three.' },
    hints: ['The rate is η · 0.3^⌊t / 20⌋.', '⌊45 / 20⌋ = 2.', '0.3 · 0.3² = ' + r4(stp) + '.'], solution: String(r4(stp)), wrong: ['0.09', '0.0081'] });

  const cos = T.sched.cosine({ lr: 0.4, min: 0, total: 100 })(25);
  ex({ id: 'op-x-cosine', chapter: 'op-sched', title: 'A cosine rate', level: 2, kind: 'number',
    text: ['A cosine schedule falls from η = 0.4 to 0 over 100 steps: η_t = 0.4 · ½(1 + cos(πt / 100)). What is the rate at step 25? Give four decimals.'],
    answer: r4(cos), tolerance: 0.00015, whyNot: { '0.3': 'That is a straight line from 0.4 to 0; the cosine falls more slowly at first.', '0.2': 'That is halfway, at step 50.', [String(r4(0.4 * Math.cos(Math.PI / 4)))]: 'Add 1 to the cosine and halve: ½(1 + cos(π/4)).' },
    hints: ['πt / 100 = π/4 at t = 25, and cos(π/4) = 0.7071.', '½(1 + 0.7071) = 0.8536.', '0.4 · 0.8536 = ' + r4(cos) + ': a quarter of the way through, the rate has fallen by only 15%.'], solution: String(r4(cos)), wrong: ['0.3', '0.2'] });

  const cfn = T.sched.cosine({ lr: 0.3, min: 0, total: 60 }), ccurve = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60].map((t) => [t, r4(cfn(t))]);
  ex({ id: 'op-x-sched-curve', chapter: 'op-sched', title: 'Sketch the cosine schedule', level: 2, kind: 'predict-curve',
    text: ['A cosine schedule takes the learning rate from 0.3 at step 0 to 0 at step 60. Sketch the rate over the 60 steps.'],
    plot: { xr: [0, 62], yr: [0, 0.33], xname: 'step', yname: 'learning rate', given: ccurve.slice(0, 1), note: 'Drag from step 0 to step 60. The faint point is the start.' },
    answer: ccurve, tolerance: 0.02, checks: ['falling'],
    hints: ['η_t = 0.3 · ½(1 + cos(πt / 60)).', 'At the middle, step 30, it is exactly half: 0.15. It falls slowly at the start and the end, fastest in the middle.', 'It goes ' + ccurve.filter(([t]) => t % 15 === 0).map(([t, v]) => 'step ' + t + ': ' + v).join(', ') + '.'],
    solution: ccurve, wrong: [ccurve.map(([t]) => [t, r4(0.3 * (1 - t / 60))]).map(([t, v]) => [t, t < 30 ? v - 0.05 : v + 0.05]), ccurve.map(([t]) => [t, 0.3])] });

  const wu = T.sched.warmup({ steps: 15, lr: 0.5 })(4);
  ex({ id: 'op-x-warmup', chapter: 'op-sched', title: 'A warm-up rate', level: 1, kind: 'number',
    text: ['A warm-up raises the rate in a straight line to η = 0.5 over the first 15 steps: η_t = 0.5 · (t + 1) / 15 for t = 0, 1, …, 14. What is the rate at step t = 4? Give four decimals.'],
    answer: r4(wu), tolerance: 0.00005, whyNot: { [String(r4(0.5 * 4 / 15))]: 'Steps count from 0, so step 4 is the fifth: (4 + 1) / 15.', '0.5': 'The full rate is reached only at the end of the warm-up.' },
    hints: ['t + 1 = 5.', '5 / 15 = 1/3.', '0.5 / 3 = ' + r4(wu) + '.'], solution: String(r4(wu)), wrong: [String(r4(0.5 * 4 / 15)), '0.5'] });

  ex({ id: 'op-x-warmup-choice', chapter: 'op-sched', title: 'Why warm up?', level: 2, kind: 'choice',
    text: ['Adam with a large rate, η = 0.5, climbs to a loss of 89 in its first few steps before coming down; with a 15-step warm-up it never goes above 7.64. Why does the warm-up help?'],
    options: ['Adam’s first steps are about η in every coordinate whatever the slope, because its averages hold only one or two gradients; small early rates keep those uninformed steps short', 'The warm-up makes the gradients smaller', 'The warm-up changes the loss surface', 'Adam needs a warm-up to start its momentum'], answer: 0,
    whyNot: { 1: 'The gradients are what they are; the warm-up scales the step, not the gradient.', 2: 'The surface is fixed; only the steps change.', 3: 'Adam’s averages start from the first gradient either way.' },
    hints: ['What is Adam’s first step, whatever the gradient?', 'm̂ = g and v̂ = g² after one step, so the step is η in each coordinate, even on a steep wall.', 'Early on, the averages know little about the terrain. Keeping the rate small until they do avoids the wild early jumps.'], solution: 0, wrong: [1, 3] });

  // =========================================================================================================
  // Chapter 4: the shape of the landscape
  // =========================================================================================================
  const dw = P.F1.double.f, ca = -2, cb = 0.4, cx = 0, lam = (cb - cx) / (cb - ca), chordAt = lam * dw(ca) + (1 - lam) * dw(cb), gap = dw(cx) - chordAt;
  ex({ id: 'op-x-chord', chapter: 'op-land', title: 'A chord under the graph', level: 3, kind: 'number',
    text: ['The double well L(x) = x⁴/4 − x² + 0.3x. Draw the chord from x = −2 to x = 0.4. At x = 0, how far is the graph above the chord? (A positive answer proves L is not convex.) Give three decimals.'],
    answer: r3(gap), tolerance: 0.0015, whyNot: { [String(r3(-gap))]: 'That is the chord above the graph: the sign says which is higher. Subtract the chord from L(0).', '0': 'Zero would mean the chord touches the graph at x = 0; work out the chord’s height there.' },
    hints: ['L(−2) = 4 − 4 − 0.6 = −0.6 and L(0.4) = ' + r4(dw(0.4)) + '.', 'x = 0 is a sixth of the way from 0.4 to −2, so the chord there is (1/6) · L(−2) + (5/6) · L(0.4) = ' + r4(chordAt) + '.', 'L(0) = 0, so the graph is ' + r3(gap) + ' above the chord: a chord passes under the graph, and the loss is not convex.'],
    solution: String(r3(gap)), wrong: [String(r3(-gap)), '0'] });

  // which of these are convex: every chord over a grid of pairs on or above the graph
  const FN = { a: (x) => x * x, b: (x) => Math.abs(x), c: (x) => Math.sin(x), d: (x) => x ** 4 / 4 - x * x, e: (x) => Math.exp(x), f: (x) => Math.log(Math.cosh(x)) };
  const isConvex = (f) => { const xs = []; for (let i = 0; i <= 40; i++) xs.push(-3 + (6 * i) / 40); for (const a of xs) for (const b of xs) if (b > a) for (let k = 1; k < 10; k++) { const l = k / 10; if (f(l * a + (1 - l) * b) > l * f(a) + (1 - l) * f(b) + 1e-12) return false; } return true; };
  const convex = Object.keys(FN).filter((k) => isConvex(FN[k]));
  ex({ id: 'op-x-convex-set', chapter: 'op-land', title: 'Which losses are convex?', level: 2, kind: 'set',
    text: ['Which of these functions are convex on the whole line? Write their letters as a set, like {a, b}.', { list: ['a: x²', 'b: |x|', 'c: sin x', 'd: x⁴/4 − x²', 'e: eˣ', 'f: log cosh x'] }],
    answer: '{' + convex.join(', ') + '}', caseless: true, placeholder: 'like {a, b}', whyNot: 'Test each with a chord, or by its curvature: convex means the chord is never below the graph.',
    hints: ['A smooth function of one variable is convex when its second derivative is never negative.', 'x² has 2, eˣ has eˣ, log cosh x has 1/cosh² x: all positive. sin x has −sin x, which changes sign; x⁴/4 − x² has 3x² − 2, negative near 0. |x| has a corner, but every chord lies above it.', 'The convex ones are {' + convex.join(', ') + '}.'],
    solution: '{' + convex.join(', ') + '}', wrong: ['{a, e, f}', '{a, b, d, e, f}'] });

  const pl = SF.plateau, pg = pl.g(3, 0), plen = Math.hypot(...pg);
  ex({ id: 'op-x-plateau-grad', chapter: 'op-land', title: 'How flat is the plateau?', level: 2, kind: 'number',
    text: ['The plateau round a pit has loss ' + pl.formula + ', with gradient (u·e, 2v·e) where e = exp(−(u² + 2v²)/2). How long is the gradient at (3, 0)? Give four decimals.'],
    answer: r4(plen), tolerance: 0.00015, whyNot: { '3': 'Multiply by e = exp(−4.5), which is tiny.', [String(r4(Math.exp(-4.5)))]: 'That is e alone; the gradient is u·e = 3e.' },
    hints: ['At (3, 0), e = exp(−9/2) = ' + r4(Math.exp(-4.5)) + '.', 'The gradient is (3e, 0).', 'Its length is ' + r4(plen) + ': with η = 1, plain descent would move only that far in a step, though the pit is 3 away.'], solution: String(r4(plen)), wrong: ['3', String(r4(Math.exp(-4.5)))] });

  const sh = ML.la.symEig(SF.saddle.h(0, 0)).values;
  ex({ id: 'op-x-saddle', chapter: 'op-land', title: 'A point where the gradient is zero', level: 2, kind: 'choice',
    text: ['The saddle surface ' + SF.saddle.formula + ' has zero gradient at (0, 0). Its Hessian there has eigenvalues ' + sh.map((v) => r3(v)).join(' and ') + '. What kind of point is it?'],
    options: ['A saddle: the loss rises along one direction and falls along the other', 'A minimum', 'A maximum', 'A plateau'], answer: 0,
    whyNot: { 1: 'A minimum needs every eigenvalue positive: the loss must rise in every direction.', 2: 'A maximum needs every eigenvalue negative.', 3: 'On a plateau the curvatures are near zero; here they are ±1.' },
    hints: ['The signs of the Hessian’s eigenvalues say how the loss curves along each direction.', 'One is positive (curving up along u), one negative (curving down along v).', 'Up along one direction, down along the other: a saddle. Gradient descent started exactly on the u axis would stall there; a tiny push in v lets it roll down to v = ±1.'], solution: 0, wrong: [1, 3] });
})(globalThis.Nooral = globalThis.Nooral || {});
