// Part 9, neural networks: the exercises. Every number is worked out here by the engines (the tensor engine's
// activations, autodiff, normalisation, dropout and optimisers, S.T, in double precision, and the networks of S.DL),
// so the text, the answers and the hints agree. The two curves come from the part's own experiments (S.P09: the
// initialisation stack and the regularisation run), read off once and written here; the claims check re-runs them.
(function (S) {
  'use strict';
  const C = S.Content, T = S.T, WF = S.WF;
  if (!C || !T || !WF || !S.DL || C.exercises.has('ne-x-neuron-out')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const f64 = (fn) => T.float64(fn);
  const one = (t) => t.values()[0];
  const gradOf = (t) => (t.grad ? Array.from(t.grad.values()) : null);
  const str = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));
  const sign = (v) => String(v).replace(/^-/, '−');
  const ex = (x) => C.exercise(x);
  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });
  /** an activation's value and slope at z, from the engine's forward and backward pass */
  const actAt = (kind, z) => f64(() => { const p = T.param(T.tensor([z])), a = T[kind](p); a.backward(); return { a: one(a), d: gradOf(p)[0] }; });

  // =========================================================================================================
  // Chapter 1: neurons and the forward pass
  // =========================================================================================================
  const N1 = f64(() => { const z = 0.4 * 2 + 0.9 * -1 - 0.3; return { z, a: one(T.tanh(T.tensor([z]))), s: one(T.sigmoid(T.tensor([z]))) }; });
  ex({ id: 'ne-x-neuron-out', chapter: 'ne-units', title: 'A neuron’s output', level: 1, kind: 'number',
    text: ['A tanh neuron has weights w₁ = 0.4 and w₂ = 0.9 and bias b = −0.3. What does it output for the input x₁ = 2, x₂ = −1? Give three decimals.'],
    answer: r3(N1.a), tolerance: 0.0015, whyNot: { [String(r3(N1.z))]: 'That is z, the weighted sum; the neuron passes it through tanh.', [String(r3(N1.s))]: 'That is the sigmoid of z; this neuron uses tanh.' },
    hints: ['First z = w₁x₁ + w₂x₂ + b, then a = tanh(z).', 'z = 0.4 · 2 + 0.9 · (−1) − 0.3 = ' + sign(r3(N1.z)) + '.', 'tanh(' + sign(r3(N1.z)) + ') = ' + sign(r3(N1.a)) + ': a small negative output, since z is just below zero where tanh is nearly straight.'],
    solution: String(r3(N1.a)), wrong: [String(r3(N1.z)), String(r3(N1.s))] });

  const RW = [1, -2], RB = 0.5, RP = { A: [1, 0], B: [0, 1], C: [2, 1] }, rcells = {};
  for (const [k, x] of Object.entries(RP)) f64(() => { const z = one(T.add(T.matmul(T.tensor([x]), T.tensor([[RW[0]], [RW[1]]])), RB)); rcells[k + '|z'] = r3(z); rcells[k + '|a'] = r3(one(T.relu(T.tensor([z])))); });
  ex({ id: 'ne-x-neuron-relu', chapter: 'ne-units', title: 'A ReLU neuron on three points', level: 1, kind: 'table',
    text: ['A ReLU neuron has weights w = (1, −2) and bias 0.5. Fill in its weighted sum z and its output a for the points A = (1, 0), B = (0, 1) and C = (2, 1).'],
    rows: ['A', 'B', 'C'], cols: ['z', 'a'], corner: 'point', cells: rcells, tolerance: 0.0005,
    hints: ['z = 1 · x₁ − 2 · x₂ + 0.5; a = max(0, z).', 'For B: z = 0 − 2 + 0.5 = −1.5, so the neuron is off.', 'A: z = ' + rcells['A|z'] + ', a = ' + rcells['A|a'] + '. B: z = −1.5, a = 0. C: z = ' + rcells['C|z'] + ', a = ' + rcells['C|a'] + '.'],
    solution: str(rcells), wrong: [Object.assign(str(rcells), { 'B|a': '-1.5' }), Object.assign(str(rcells), { 'A|z': '1', 'A|a': '1', 'C|z': '0', 'C|a': '0' })] });

  const LW = [3, 4], LB = -10, ldist = Math.abs(LB) / Math.hypot(...LW);
  ex({ id: 'ne-x-neuron-line', chapter: 'ne-units', title: 'Where a neuron switches', level: 2, kind: 'number',
    text: ['A neuron has weights w = (3, 4) and bias b = −10. Its output changes most across the line where z = 3x₁ + 4x₂ − 10 = 0. How far is that line from the origin?'],
    answer: r3(ldist), tolerance: 0.005, whyNot: { '10': 'That is |b|; divide it by the length of the weight vector.', '0.5': 'That is ‖w‖ / |b|, the wrong way up.', '-2': 'A distance is never negative: take |b|.' },
    hints: ['The line w · x + b = 0 is at right angles to w.', 'Its distance from the origin is |b| / ‖w‖, and ‖w‖ = √(3² + 4²) = 5.', '10 / 5 = ' + r3(ldist) + '. The point (1.2, 1.6), which is 2 along w’s direction, lies on the line.'],
    solution: String(r3(ldist)), wrong: ['10', '0.5'] });

  const SG = actAt('sigmoid', 0.8);
  ex({ id: 'ne-x-sig-slope', chapter: 'ne-units', title: 'The sigmoid’s slope', level: 1, kind: 'number',
    text: ['A sigmoid neuron has z = 0.8. What is the slope of its activation there, σ′(0.8)? Give four decimals.'],
    answer: r4(SG.d), tolerance: 0.0002, whyNot: { [String(r4(SG.a))]: 'That is σ(0.8), the output; the slope is σ(1 − σ).', '0.25': 'That is the slope at z = 0, the steepest point; at 0.8 it is a little less.' },
    hints: ['σ′(z) = σ(z) · (1 − σ(z)).', 'σ(0.8) = 1 / (1 + e^−0.8) = ' + r4(SG.a) + '.', r4(SG.a) + ' × ' + r4(1 - SG.a) + ' = ' + r4(SG.d) + '.'],
    solution: String(r4(SG.d)), wrong: [String(r4(SG.a)), '0.25'] });

  const scells = {};
  for (const k of ['sigmoid', 'tanh', 'relu']) for (const z of [-1, 2]) scells[k + '|at z = ' + sign(z)] = r3(actAt(k, z).d);
  ex({ id: 'ne-x-slopes-table', chapter: 'ne-units', title: 'Three slopes at two points', level: 2, kind: 'table',
    text: ['Fill in the slope of each activation at z = −1 and at z = 2 (three decimals). These are the factors a gradient is multiplied by as it passes back through the neuron.'],
    rows: ['sigmoid', 'tanh', 'relu'], cols: ['at z = −1', 'at z = 2'], corner: 'slope f′(z)', cells: scells, tolerance: 0.0015,
    hints: ['σ′ = σ(1 − σ); tanh′ = 1 − tanh²; ReLU′ = 1 for z > 0 and 0 for z < 0.', 'σ(−1) = 0.269 and σ(2) = 0.881; tanh(−1) = −0.762 and tanh(2) = 0.964.', 'sigmoid: ' + scells['sigmoid|at z = −1'] + ' and ' + scells['sigmoid|at z = 2'] + '; tanh: ' + scells['tanh|at z = −1'] + ' and ' + scells['tanh|at z = 2'] + '; ReLU: 0 and 1. Only ReLU passes a whole gradient, and only on its positive side.'],
    solution: str(scells), wrong: [Object.assign(str(scells), { 'relu|at z = −1': '1' }), Object.assign(str(scells), { 'sigmoid|at z = −1': '0.269', 'sigmoid|at z = 2': '0.881' })] });

  const TD = actAt('tanh', 1).d, vanish = TD ** 5;
  ex({ id: 'ne-x-vanish', chapter: 'ne-units', title: 'Five tanh layers', level: 2, kind: 'number',
    text: ['A gradient passes back through 5 tanh layers, and in each the neuron on its path has z = 1. What fraction of the gradient is left after the five slopes? Give four decimals.'],
    answer: r4(vanish), tolerance: 0.0002, whyNot: { [String(r4(TD))]: 'That is one layer’s slope; the gradient is multiplied by it five times.', [String(r4(5 * TD))]: 'The slopes multiply, they do not add.' },
    hints: ['Each layer multiplies the gradient by its slope, tanh′(1) = 1 − tanh²(1).', 'tanh(1) = 0.7616, so the slope is ' + r4(TD) + '.', r4(TD) + '⁵ = ' + r4(vanish) + ': about 1.3% is left after only five layers.'],
    solution: String(r4(vanish)), wrong: [String(r4(TD)), String(r4(5 * TD))] });

  ex({ id: 'ne-x-act-choice', chapter: 'ne-units', title: 'An activation for a deep network', level: 1, kind: 'choice',
    text: ['A network has 20 hidden layers. Which activation in the hidden layers gives the gradient the best chance of reaching the first layer?'],
    options: ['ReLU (or GELU)', 'Sigmoid', 'tanh', 'No activation at all'], answer: 0,
    whyNot: { 1: 'The sigmoid’s slope is at most 0.25: twenty of them leave less than 10⁻¹² of the gradient.', 2: 'tanh’s slope is 1 only at z = 0 and falls fast away from it; twenty layers usually lose most of the gradient.', 3: 'Without an activation the twenty layers collapse into one linear map: no curved boundary at all.' },
    hints: ['Backpropagation multiplies the slopes of all twenty layers.', 'Which activation has slope exactly 1 over a whole half of its range?', 'ReLU: on its positive side the gradient passes through unchanged. In the lesson, 20 ReLU layers left all of it.'], solution: 0, wrong: [1, 2] });

  const actRef = (kind, z, g) => f64(() => { const p = T.param(T.tensor(z)), a = T[kind](p); a.backward(T.tensor(g)); return [Array.from(a.values()), gradOf(p)]; });
  const actSol = 'function actPass(kind, z, g) {\n  const out = [], gIn = [];\n  for (let i = 0; i < z.length; i++) {\n    let a, d;\n    if (kind === \'relu\') { a = z[i] > 0 ? z[i] : 0; d = z[i] > 0 ? 1 : 0; }\n    else { a = 1 / (1 + Math.exp(-z[i])); d = a * (1 - a); }\n    out.push(a);\n    gIn.push(g[i] * d);\n  }\n  return [out, gIn];\n}';
  ex({ id: 'ne-x-code-act', chapter: 'ne-units', title: 'ReLU and sigmoid, forward and backward', level: 2, kind: 'code',
    text: ['Write `actPass(kind, z, g)`. `kind` is `\'relu\'` or `\'sigmoid\'`; `z` is an array of inputs; `g` is the gradient arriving from above, one number per input. Return `[out, gIn]`: the activation of each input, and the gradient passed back, `g[i] · f′(z[i])`. Take ReLU’s slope at exactly 0 to be 0.'],
    fn: 'actPass', params: ['kind', 'z', 'g'], start: 'function actPass(kind, z, g) {\n  \n}', tolerance: 1e-9, ref: actRef,
    cases: [['relu', [-2, 0.5, 3], [1, 1, 1]], ['relu', [1.5, -0.1, 0], [2, -3, 4]], ['sigmoid', [0], [1]], ['sigmoid', [0.8, -1, 2], [1, 1, 1]], ['sigmoid', [-3, 3], [0.5, -2]], ['relu', [], []]],
    hints: ['Loop over the inputs; compute the value first, then the slope at that input.', 'ReLU: value max(0, z), slope 1 if z > 0 else 0. Sigmoid: value σ = 1 / (1 + e^−z), slope σ(1 − σ), which reuses the value.', actSol],
    solution: actSol,
    wrong: ['function actPass(kind, z, g) { const out = z.map((v) => kind === \'relu\' ? Math.max(0, v) : 1 / (1 + Math.exp(-v))); return [out, g.slice()]; }', 'function actPass(kind, z, g) { const out = z.map((v) => kind === \'relu\' ? Math.max(0, v) : 1 / (1 + Math.exp(-v))); return [out, out.map((a, i) => g[i] * (kind === \'relu\' ? (z[i] > 0 ? 1 : 0) : a))]; }'] });

  const FX = [1, 2], FW = [[1, 0.5], [-0.5, 1]], FB = [0.5, -3], FV = [[1.5], [-2]], FC = 0.25, fcells = {};
  f64(() => {
    const z = T.add(T.matmul(T.tensor([FX]), T.tensor(FW)), T.tensor(FB)), h = T.relu(z), y = T.add(T.matmul(h, T.tensor(FV)), FC), zv = z.values(), hv = h.values();
    fcells['h₁|z'] = r3(zv[0]); fcells['h₁|value'] = r3(hv[0]); fcells['h₂|z'] = r3(zv[1]); fcells['h₂|value'] = r3(hv[1]); fcells['output|z'] = r3(one(y)); fcells['output|value'] = r3(one(y));
  });
  ex({ id: 'ne-x-forward', chapter: 'ne-units', title: 'A forward pass by hand', level: 2, kind: 'table',
    text: ['A network with two ReLU hidden neurons and one output with no activation. Hidden neuron h₁ has weights (1, −0.5) and bias 0.5; h₂ has weights (0.5, 1) and bias −3. The output is 1.5 · h₁ − 2 · h₂ + 0.25. Fill in every value for the input x = (1, 2).'],
    rows: ['h₁', 'h₂', 'output'], cols: ['z', 'value'], corner: 'x = (1, 2)', cells: fcells, tolerance: 0.0005,
    hints: ['Each hidden z is its weighted sum plus its bias; its value is max(0, z). The output has no activation, so its value is its z.', 'h₁: z = 1 − 1 + 0.5 = 0.5. h₂: z = 0.5 + 2 − 3 = −0.5, so h₂ is off.', 'h₁ = ' + fcells['h₁|value'] + ', h₂ = 0, and the output is 1.5 · 0.5 − 2 · 0 + 0.25 = ' + fcells['output|value'] + '.'],
    solution: str(fcells), wrong: [Object.assign(str(fcells), { 'h₂|value': '-0.5', 'output|z': '2', 'output|value': '2' }), Object.assign(str(fcells), { 'output|z': '0.75', 'output|value': '0.75' })] });

  const SC = [1.5, -0.5, 0.3], SP = f64(() => Array.from(T.softmax(T.tensor([SC]), -1).values()));
  ex({ id: 'ne-x-softmax', chapter: 'ne-units', title: 'Scores into a probability', level: 2, kind: 'number',
    text: ['A network’s three output scores are 1.5, −0.5 and 0.3. What probability does softmax give the first class? Give three decimals.'],
    answer: r3(SP[0]), tolerance: 0.0015, whyNot: { [String(r3(1.5 / 1.3))]: 'Softmax divides exponentials, e^score / Σ e^score, not the scores themselves.', [String(r3(1 / (1 + Math.exp(-1.5))))]: 'That is σ(1.5), the first score alone; softmax weighs it against the other two.' },
    hints: ['p₁ = e^1.5 / (e^1.5 + e^−0.5 + e^0.3).', 'e^1.5 = 4.482, e^−0.5 = 0.607, e^0.3 = 1.350.', '4.482 / 6.439 = ' + r3(SP[0]) + '. The others get ' + r3(SP[1]) + ' and ' + r3(SP[2]) + '.'],
    solution: String(r3(SP[0])), wrong: [String(r3(1.5 / 1.3)), String(r3(1 / (1 + Math.exp(-1.5))))] });

  const NW = S.DL.build('mlp', { inputs: 4, hidden: [16, 8], outputs: 3, seed: 1 }).numParams(), NWnb = 4 * 16 + 16 * 8 + 8 * 3;
  ex({ id: 'ne-x-count-weights', chapter: 'ne-units', title: 'Count the weights', level: 2, kind: 'number',
    text: ['A network takes 4 features, has hidden layers of 16 and 8 neurons, and 3 outputs. Every layer is dense, with a bias for each neuron. How many numbers does training adjust?'],
    answer: NW, tolerance: 0, whyNot: { [String(NWnb)]: 'Those are the weights only: every neuron also has a bias.', [String(4 + 16 + 8 + 3)]: 'That counts neurons, not the wires between them.' },
    hints: ['A dense layer from n inputs to m neurons has n · m weights and m biases.', '4 → 16: 64 + 16 = 80. 16 → 8: 128 + 8 = 136.', '8 → 3: 24 + 3 = 27; in all 80 + 136 + 27 = ' + NW + '.'],
    solution: String(NW), wrong: [String(NWnb), '31'] });

  // =========================================================================================================
  // Chapter 2: how a network learns
  // =========================================================================================================
  const BO = f64(() => { const v = T.param(T.tensor([1.5])), c = T.param(T.tensor([-0.2])), y = T.add(T.mul(v, 0.6), c), L = T.square(T.sub(y, 1)); L.backward(); return { y: one(y), gv: gradOf(v)[0], gc: gradOf(c)[0] }; });
  ex({ id: 'ne-x-bp-out', chapter: 'ne-learn', title: 'The gradient of an output weight', level: 1, kind: 'number',
    text: ['The output of a network is y = v · h + c with h = 0.6, v = 1.5 and c = −0.2. The loss is L = (y − t)² with target t = 1. What is ∂L/∂v?'],
    answer: r3(BO.gv), tolerance: 0.0005, whyNot: { [String(r3(BO.gc))]: 'That is ∂L/∂y (and ∂L/∂c); for v multiply by h, the input v multiplies.', [String(r3(-BO.gv))]: 'Mind the sign: y is below the target, so y − t is negative.' },
    hints: ['∂L/∂v = ∂L/∂y · ∂y/∂v = 2(y − t) · h.', 'y = 1.5 · 0.6 − 0.2 = ' + r3(BO.y) + ', so 2(y − t) = ' + r3(BO.gc) + '.', r3(BO.gc) + ' × 0.6 = ' + r3(BO.gv) + ': raising v would lower the loss.'],
    solution: String(r3(BO.gv)), wrong: [String(r3(BO.gc)), String(r3(-BO.gv))] });

  const BG = { x: 1.5, w: 0.4, b: -0.2, v: 2, t: 0.5 }, bcells = {};
  const bgRun = (o) => f64(() => {
    const w = T.param(T.tensor([o.w])), b = T.param(T.tensor([o.b])), v = T.param(T.tensor([o.v]));
    const u = T.add(T.mul(w, o.x), b), h = T.tanh(u), y = T.mul(v, h), L = T.square(T.sub(y, o.t));
    L.backward();
    return { L: one(L), y: one(y), h: one(h), u: one(u), gy: gradOf(y)[0], gh: gradOf(h)[0], gu: gradOf(u)[0], gw: gradOf(w)[0], gb: gradOf(b)[0], gv: gradOf(v)[0] };
  });
  const B0 = bgRun(BG);
  for (const [r, k] of [['∂L/∂y', 'gy'], ['∂L/∂h', 'gh'], ['∂L/∂u', 'gu'], ['∂L/∂w', 'gw'], ['∂L/∂b', 'gb'], ['∂L/∂v', 'gv']]) bcells[r + '|value'] = r3(B0[k]);
  ex({ id: 'ne-x-bp-graph', chapter: 'ne-learn', title: 'Backpropagation through a tiny graph', level: 3, kind: 'table',
    text: ['A one-neuron network as a graph: u = w · x + b, h = tanh(u), y = v · h, L = (y − t)². With x = 1.5, w = 0.4, b = −0.2, v = 2 and target t = 0.5, the forward pass gives u = ' + r3(B0.u) + ', h = ' + r3(B0.h) + ' and y = ' + r3(B0.y) + '. Fill in every gradient of the backward pass (three decimals).'],
    rows: ['∂L/∂y', '∂L/∂h', '∂L/∂u', '∂L/∂w', '∂L/∂b', '∂L/∂v'], cols: ['value'], corner: 'gradient', cells: bcells, tolerance: 0.002,
    hints: ['Start at the top: ∂L/∂y = 2(y − t). Then each node passes back the gradient times its local derivative.', 'y = v · h sends v · ∂L/∂y to h and h · ∂L/∂y to v. tanh has local derivative 1 − h² = ' + r3(1 - B0.h * B0.h) + '. u = w · x + b sends x · ∂L/∂u to w and ∂L/∂u to b.', '∂L/∂y = ' + bcells['∂L/∂y|value'] + ', ∂L/∂h = ' + bcells['∂L/∂h|value'] + ', ∂L/∂u = ' + bcells['∂L/∂u|value'] + ', ∂L/∂w = ' + bcells['∂L/∂w|value'] + ', ∂L/∂b = ' + bcells['∂L/∂b|value'] + ', ∂L/∂v = ' + bcells['∂L/∂v|value'] + '.'],
    solution: str(bcells), wrong: [Object.assign(str(bcells), { '∂L/∂u|value': String(bcells['∂L/∂h|value']), '∂L/∂w|value': String(r3(B0.gh * BG.x)), '∂L/∂b|value': String(bcells['∂L/∂h|value']) }), Object.assign(str(bcells), { '∂L/∂w|value': String(bcells['∂L/∂u|value']) })] });

  const BR = f64(() => {
    const w = T.param(T.tensor([[0.5], [-1]])), b = T.param(T.tensor([0.2])), z = T.add(T.matmul(T.tensor([[1, 2]]), w), b), h = T.relu(z), y = T.add(T.mul(h, 3), 0.4), L = T.square(T.sub(y, 2));
    L.backward();
    return { z: one(z), y: one(y), gw1: gradOf(w)[0], naive: 2 * (one(y) - 2) * 3 * 1 };
  });
  ex({ id: 'ne-x-bp-relu', chapter: 'ne-learn', title: 'A gradient through a switched-off ReLU', level: 2, kind: 'number',
    text: ['A ReLU neuron h = ReLU(w₁x₁ + w₂x₂ + b) with w = (0.5, −1) and b = 0.2 feeds the output y = 3h + 0.4, and L = (y − 2)². For the input x = (1, 2), what is ∂L/∂w₁?'],
    answer: r3(BR.gw1), tolerance: 0.0005, whyNot: { [String(r3(BR.naive))]: 'That skips the ReLU’s slope. Here z = ' + sign(r3(BR.z)) + ', so the neuron is off and its slope is 0.', [String(r3(2 * (BR.y - 2)))]: 'That is ∂L/∂y; keep going down the chain to w₁.' },
    hints: ['∂L/∂w₁ = 2(y − 2) · 3 · ReLU′(z) · x₁.', 'z = 0.5 · 1 − 1 · 2 + 0.2 = ' + sign(r3(BR.z)) + '.', 'z is negative, so ReLU′(z) = 0 and ∂L/∂w₁ = 0, though the loss is ' + r3((BR.y - 2) ** 2) + '. This example cannot teach w₁ anything.'],
    solution: String(r3(BR.gw1)), wrong: [String(r3(BR.naive)), String(r3(2 * (BR.y - 2)))] });

  const UW = [0.8, -1.2, 0.3], UG = [2.4, -0.6, 0], UE = 0.05;
  const upd = f64(() => { const p = T.param(T.tensor(UW)); p.grad = T.tensor(UG); new T.optim.SGD([p], { lr: UE }).step(); return Array.from(p.values()); });
  const ucells = {}; ['w₁', 'w₂', 'w₃'].forEach((n, i) => { ucells[n + '|after'] = r4(upd[i]); });
  ex({ id: 'ne-x-update', chapter: 'ne-learn', title: 'One weight update', level: 1, kind: 'table',
    text: ['Three weights are 0.8, −1.2 and 0.3; backpropagation gives their gradients as 2.4, −0.6 and 0. Take one gradient-descent step with learning rate η = 0.05: w ← w − η · ∂L/∂w.'],
    rows: ['w₁', 'w₂', 'w₃'], cols: ['after'], corner: 'weight', cells: ucells, tolerance: 0.0005,
    hints: ['Each weight moves against its own gradient, by η times it.', 'w₁: 0.8 − 0.05 · 2.4. w₂: −1.2 − 0.05 · (−0.6).', 'w₁ = ' + ucells['w₁|after'] + ', w₂ = ' + sign(ucells['w₂|after']) + ' (a negative gradient moves the weight up), w₃ = 0.3 (no gradient, no move).'],
    solution: str(ucells), wrong: [{ 'w₁|after': '0.92', 'w₂|after': '-1.23', 'w₃|after': '0.3' }, { 'w₁|after': '0.68', 'w₂|after': '-1.23', 'w₃|after': '0.3' }] });

  const B1 = f64(() => { const n = { x: BG.x, t: BG.t, w: BG.w - 0.1 * B0.gw, b: BG.b - 0.1 * B0.gb, v: BG.v - 0.1 * B0.gv }; return Object.assign(bgRun(n), { n }); });
  ex({ id: 'ne-x-update-loss', chapter: 'ne-learn', title: 'Does the step help?', level: 3, kind: 'number',
    text: ['Take the tiny graph of “Backpropagation through a tiny graph” (x = 1.5, w = 0.4, b = −0.2, v = 2, t = 0.5, loss ' + r4(B0.L) + '). Move all three weights one gradient step with η = 0.1, using the gradients found there, and run the forward pass again. What is the new loss? Give four decimals.'],
    answer: r4(B1.L), tolerance: 0.0003, whyNot: { [String(r4(B0.L))]: 'That is the loss before the step.', [String(r4(bgRun({ x: BG.x, t: BG.t, w: BG.w + 0.1 * B0.gw, b: BG.b + 0.1 * B0.gb, v: BG.v + 0.1 * B0.gv }).L))]: 'That moved the weights along the gradient, uphill: subtract η times it.' },
    hints: ['New weights: w − 0.1 · ∂L/∂w, b − 0.1 · ∂L/∂b, v − 0.1 · ∂L/∂v.', 'They are w = ' + r4(B1.n.w) + ', b = ' + r4(B1.n.b) + ', v = ' + r4(B1.n.v) + '.', 'Forward again: u = ' + r4(B1.u) + ', h = ' + r4(B1.h) + ', y = ' + r4(B1.y) + ', so L = ' + r4(B1.L) + (B1.L < B0.L ? ', down from ' + r4(B0.L) + '.' : ', up from ' + r4(B0.L) + ((B0.y - BG.t) * (B1.y - BG.t) < 0 ? ': the step overshoots, taking y from ' + r4(B0.y) + ' past the target ' + BG.t + ' to ' + r4(B1.y) + '. The gradient only promises a fall for a small enough step, and η = 0.1 is too large here.' : ': the step is too large for the gradient’s promise of a fall to hold.'))],
    solution: String(r4(B1.L)), wrong: [String(r4(B0.L)), String(r4(bgRun({ x: BG.x, t: BG.t, w: BG.w + 0.1 * B0.gw, b: BG.b + 0.1 * B0.gb, v: BG.v + 0.1 * B0.gv }).L))] });

  const denseRef = (X, W, G) => f64(() => {
    const x = T.param(T.tensor(X)), w = T.param(T.tensor(W)), b = T.param(T.zeros([W[0].length]));
    const y = T.add(T.matmul(x, w), b); y.backward(T.tensor(G));
    const rows = (t, n, m) => { const v = gradOf(t); return Array.from({ length: n }, (_, i) => v.slice(i * m, i * m + m)); };
    return [rows(w, W.length, W[0].length), gradOf(b), rows(x, X.length, X[0].length)];
  });
  const denseSol = 'function denseBackward(X, W, G) {\n  const n = X.length, d = W.length, m = W[0].length;\n  const dW = W.map((row) => row.map(() => 0)), db = new Array(m).fill(0);\n  const dX = X.map((row) => row.map(() => 0));\n  for (let i = 0; i < n; i++)\n    for (let k = 0; k < m; k++) {\n      db[k] += G[i][k];\n      for (let j = 0; j < d; j++) {\n        dW[j][k] += X[i][j] * G[i][k];\n        dX[i][j] += G[i][k] * W[j][k];\n      }\n    }\n  return [dW, db, dX];\n}';
  ex({ id: 'ne-x-code-dense', chapter: 'ne-learn', title: 'A dense layer’s backward pass', level: 3, kind: 'code',
    text: ['A dense layer computes Y = X·W + b for a batch: X is n × d (one row per example), W is d × m, b has m entries. Given the gradient G = ∂L/∂Y (n × m), write `denseBackward(X, W, G)` returning `[dW, db, dX]`: ∂L/∂W (d × m), ∂L/∂b (m numbers) and ∂L/∂X (n × d). Arrays are arrays of rows.'],
    fn: 'denseBackward', params: ['X', 'W', 'G'], start: 'function denseBackward(X, W, G) {\n  \n}', tolerance: 1e-9, ref: denseRef,
    cases: [[[[1, 2]], [[0.5], [-1]], [[1]]], [[[1, 2], [3, -1]], [[1, 0, 2], [-1, 1, 0.5]], [[1, 0, -1], [0.5, 2, 0]]], [[[0.2, -0.4, 1]], [[1, 2], [3, 4], [5, 6]], [[-2, 0.5]]], [[[1], [2], [3]], [[2, -1]], [[1, 1], [0, 1], [-1, 2]]]],
    hints: ['Each Y[i][k] = Σⱼ X[i][j] · W[j][k] + b[k]. Follow one product back: it sends X[i][j] · G[i][k] to W[j][k] and W[j][k] · G[i][k] to X[i][j].', 'So dW = Xᵀ·G, db is G summed over the examples (its column sums), and dX = G·Wᵀ.', denseSol],
    solution: denseSol,
    wrong: ['function denseBackward(X, W, G) { const n = X.length, d = W.length, m = W[0].length; const dW = W.map((r) => r.map(() => 0)), db = new Array(m).fill(0), dX = X.map((r) => r.map(() => 0)); for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) { db[k] += G[i][k]; for (let j = 0; j < d; j++) { dW[j][k] += X[i][j] * G[i][k]; dX[i][j] += G[i][k] * W[j][k]; } } return [dW, db.map((v) => v / n), dX]; }',
      'function denseBackward(X, W, G) { const n = X.length, d = W.length, m = W[0].length; const dW = W.map((r) => r.map(() => 0)), db = new Array(m).fill(0), dX = X.map((r) => r.map(() => 0)); for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) { db[k] += G[i][k]; for (let j = 0; j < d; j++) { dW[j][k] += X[i][j] * G[i][k]; dX[i][j] += G[i][k] * W[j][k]; } } return [dW, db, X]; }'] });

  const TW = 4, twins = 2 ** TW * [1, 2, 3, 4].reduce((a, b) => a * b, 1);
  ex({ id: 'ne-x-twins', chapter: 'ne-learn', title: 'How many twins?', level: 3, kind: 'number',
    text: ['A network has one hidden layer of 4 tanh neurons. Two changes leave the function it computes exactly the same: putting the hidden neurons in another order (with their wires), and flipping the signs of all of one neuron’s weights in and out (tanh is odd). Starting from one trained setting, how many settings can these two changes reach, the starting one included? (Assume no two neurons have the same weights.)'],
    answer: twins, tolerance: 0, whyNot: { '24': 'That counts the orders only; each neuron can also be flipped or not.', '16': 'That counts the sign flips only; the neurons can also be put in 4! orders.', '8': 'Each of the 4 neurons can be flipped or not, independently: 2⁴, not 2 · 4.' },
    hints: ['The two kinds of change are independent: count each, then multiply.', 'Orders: 4! = 24. Flips: each neuron flipped or not, 2⁴ = 16.', '24 × 16 = ' + twins + ' settings, all with exactly the same loss: the lesson’s pair of twins, for 4 neurons.'],
    solution: String(twins), wrong: ['24', '16'] });

  ex({ id: 'ne-x-saddle', chapter: 'ne-learn', title: 'All weights zero', level: 1, kind: 'choice',
    text: ['The network y = v · tanh(w · x) starts with w = 0 and v = 0. Backpropagation gives the gradient (0, 0). What does gradient descent do?'],
    options: ['It stays at (0, 0) for ever: every step is zero', 'It moves to the nearest minimum', 'It moves at random', 'It moves, but slowly'], answer: 0,
    whyNot: { 1: 'Each step is −η times the gradient, and the gradient is exactly zero.', 2: 'Plain gradient descent has no randomness; the noise of mini-batches would not help either, since every example’s gradient is zero there.', 3: 'A zero gradient gives a zero step, not a small one.' },
    hints: ['What is a step of gradient descent made of?', 'At (0, 0) the output is 0 for every input, and nudging w alone or v alone does not change that.', 'Zero gradient, zero step: the walk is stuck on the saddle. That is one reason weights start random, never all zero.'], solution: 0, wrong: [1, 3] });

  ex({ id: 'ne-x-init-choice', chapter: 'ne-learn', title: 'Weights for a ReLU network', level: 1, kind: 'choice',
    text: ['A deep network of dense layers with ReLU activations is about to be trained. How should its weights be drawn?'],
    options: ['He: variance 2 / fan-in', 'Xavier: variance 2 / (fan-in + fan-out)', 'Standard deviation 1 for every weight', 'All zero'], answer: 0,
    whyNot: { 1: 'Xavier suits tanh and sigmoid. ReLU zeroes about half of its inputs, so with Xavier the spread shrinks layer by layer: 0.14 after six layers in the lesson.', 2: 'With 64 inputs a neuron’s sum would have a spread near 8: in the lesson the signal grew to 36000 by the sixth layer.', 3: 'Every hidden neuron would compute the same thing and get the same gradient, for ever.' },
    hints: ['The goal: the spread of the activations stays about the same from layer to layer.', 'ReLU throws away the negative half, which halves the variance a layer passes on.', 'He’s factor of 2 makes up for that half: in the lesson the spread stayed between 0.82 and 1.1.'], solution: 0, wrong: [1, 2] });

  ex({ id: 'ne-x-zero-init', chapter: 'ne-learn', title: 'Every weight the same', level: 2, kind: 'choice',
    text: ['A network’s hidden layer has 8 tanh neurons, and every weight in the network starts at the same value, 0.1. What goes wrong in training?'],
    options: ['All 8 hidden neurons stay identical: they compute the same value and get the same gradient at every step', 'Nothing: training breaks the tie by itself', 'The gradients explode', 'The network cannot fit a straight line'], answer: 0,
    whyNot: { 1: 'Nothing in plain gradient descent tells the neurons apart: identical weights see the same inputs and the same gradients.', 2: 'Weights of 0.1 are small; nothing explodes.', 3: 'It can still fit what one neuron can fit, since the 8 act as one.' },
    hints: ['Look at two hidden neurons: same weights in, same weights out.', 'Same values forward, so the same gradients backward, so the same updates.', 'They stay copies of each other, so the layer is in effect one neuron. Random starting weights break this symmetry.'], solution: 0, wrong: [1, 2] });

  const xs = { 'Xavier|64 → 64': Math.sqrt(2 / (64 + 64)), 'Xavier|256 → 64': Math.sqrt(2 / (256 + 64)), 'He|64 → 64': Math.sqrt(2 / 64), 'He|256 → 64': Math.sqrt(2 / 256) }, xcells = {};
  for (const [k, v] of Object.entries(xs)) xcells[k] = r4(v);
  ex({ id: 'ne-x-xavier-std', chapter: 'ne-learn', title: 'Two rules, two layers', level: 2, kind: 'table',
    text: ['Fill in the standard deviation of the initial weights (four decimals) for a dense layer from 64 to 64 neurons and for one from 256 to 64. Xavier: variance 2 / (fan-in + fan-out). He: variance 2 / fan-in.'],
    rows: ['Xavier', 'He'], cols: ['64 → 64', '256 → 64'], corner: 'std of w', cells: xcells, tolerance: 0.0005,
    hints: ['The standard deviation is the square root of the variance.', 'Xavier for 64 → 64: √(2 / 128). He for 64 → 64: √(2 / 64).', 'Xavier: ' + xcells['Xavier|64 → 64'] + ' and ' + xcells['Xavier|256 → 64'] + '. He: ' + xcells['He|64 → 64'] + ' and ' + xcells['He|256 → 64'] + '. More inputs, smaller weights: the sum of more terms must keep the same spread.'],
    solution: str(xcells), wrong: [Object.assign(str(xcells), { 'Xavier|64 → 64': '0.0156', 'He|64 → 64': '0.0313' }), Object.assign(str(xcells), { 'He|256 → 64': String(r4(Math.sqrt(2 / 64))) })] });

  // the six-layer tanh stack with weights of std 0.05 (topic p09-init, seed 1): the activations' spread per layer
  const ISTD = [[0, 1.01], [1, 0.346], [2, 0.139], [3, 0.056], [4, 0.023], [5, 0.0099], [6, 0.004]];
  ex({ id: 'ne-x-init-curve', chapter: 'ne-learn', title: 'A signal fading layer by layer', level: 2, kind: 'predict-curve',
    text: ['Six dense tanh layers of 64 neurons, weights drawn with standard deviation 0.05, no training. Inputs with spread 1.01 go in. Sketch the spread (standard deviation) of the activations after each layer.'],
    plot: { xr: [0, 6], yr: [0, 1.1], xname: 'layer', yname: 'spread', given: ISTD.slice(0, 1), note: 'Drag from layer 0 (the inputs) to layer 6. The faint point is the input.' },
    answer: ISTD, tolerance: 0.08, checks: ['falling'],
    hints: ['A neuron sums 64 inputs times weights of spread 0.05: roughly √64 · 0.05 = 0.4 times the spread it receives.', 'Near zero, tanh is about the identity, so the factor of about 0.4 applies at every layer.', 'It falls fast and flattens near zero: 1.01, 0.346, 0.139, 0.056, 0.023, 0.0099, 0.004.'],
    solution: ISTD, wrong: [ISTD.map(([x]) => [x, 1.01]), ISTD.map(([x]) => [x, 0.4 + 0.1 * x])] });

  // =========================================================================================================
  // Chapter 3: fitting without memorising
  // =========================================================================================================
  // the regularisation run of the part (S.P09.regRun({})): the loss on 400 fresh points every so many steps
  const OFC = [[0, 1.011], [10, 0.351], [20, 0.354], [30, 0.373], [50, 0.509], [100, 0.777], [150, 0.96], [200, 1.123], [250, 1.209], [300, 1.26], [350, 1.311], [400, 1.349], [450, 1.38], [500, 1.409]];
  ex({ id: 'ne-x-overfit-curve', chapter: 'ne-reg', title: 'The loss on fresh points', level: 2, kind: 'predict-curve',
    text: ['A network with 1218 weights trains on only 30 noisy points of the two moons for 500 steps; its training loss falls all the way to 0.0002. Sketch its loss on 400 fresh points from the same moons over the 500 steps.'],
    plot: { xr: [0, 500], yr: [0, 1.6], xname: 'step', yname: 'fresh loss', given: OFC.slice(0, 2), note: 'Drag from step 0 to step 500. The faint points are steps 0 and 10.' },
    answer: OFC, tolerance: 0.15, checks: ['valley'],
    hints: ['At first the network learns the shape of the moons, which helps on fresh points too.', 'Then it bends its boundary round single noisy training points. Does that help on points it has never seen?', 'It bottoms out at 0.351 near step 10, then climbs: 0.509 at step 50, 0.96 at 150, 1.209 at 250, 1.409 at 500.'],
    solution: OFC, wrong: [OFC.map(([x]) => [x, r3(1.011 * Math.exp(-x / 60))]), OFC.map(([x]) => [x, 0.351])] });

  ex({ id: 'ne-x-gap-choice', chapter: 'ne-reg', title: 'Reading a gap', level: 1, kind: 'choice',
    text: ['A small network is trained on 40 images of digits. After 600 steps it gets 100% of the training images right and 78% of new ones. What is the most likely diagnosis?'],
    options: ['Overfitting: it has memorised its 40 images', 'Underfitting: it is too small to learn digits', 'The learning rate is too large', 'The test images are mislabelled'], answer: 0,
    whyNot: { 1: 'A network that cannot learn the task would not get every training image right.', 2: 'A rate that is too large stops the training accuracy from reaching 100%.', 3: 'Nothing points to that; a gap this size is the usual sign of too little data for the model.' },
    hints: ['Compare the two numbers, not just the second.', 'Perfect on what it trained on, much worse on what it has not seen.', 'Overfitting. The lesson’s cures: more data (or augmented copies), dropout, weight decay, early stopping. Part 10 shows augmentation lifting exactly this case to 97%.'], solution: 0, wrong: [1, 2] });

  ex({ id: 'ne-x-dropout-test', chapter: 'ne-reg', title: 'Dropout at test time', level: 1, kind: 'choice',
    text: ['A network was trained with inverted dropout, p = 0.5: each step, each hidden neuron was off with probability 0.5 and the survivors were doubled. What does dropout do when the trained network classifies a new example?'],
    options: ['Nothing: every neuron is on and nothing is scaled', 'Half the neurons are switched off at random', 'Every neuron’s output is halved', 'Every neuron’s output is doubled'], answer: 0,
    whyNot: { 1: 'That would make the predictions random, different every time for the same example.', 2: 'That is the old, non-inverted form; inverted dropout already scaled during training.', 3: 'The doubling was only for survivors during training, to keep the expected output the same.' },
    hints: ['During training a neuron’s expected output is 0.5 · 0 + 0.5 · 2h = h.', 'At test time we want that expected value, h, every time.', 'Inverted dropout does nothing at test time: the full network, unscaled.'], solution: 0, wrong: [1, 2] });

  const DS = f64(() => Math.max(...T.dropout(T.ones([1, 64]), 0.2, T.rng(5), true).values()));
  ex({ id: 'ne-x-dropout-scale', chapter: 'ne-reg', title: 'The survivors’ factor', level: 1, kind: 'number',
    text: ['Inverted dropout with p = 0.2 switches each neuron off with probability 0.2. By what factor is the output of each neuron that stays on multiplied during training?'],
    answer: DS, tolerance: 0.0005, whyNot: { '0.8': 'That is 1 − p, the chance of staying on; the survivors are divided by it.', '0.2': 'That is the chance of being switched off.', '5': 'That is 1 / p; divide by the chance of staying on, 1 − p.' },
    hints: ['The expected output must stay the same as with no dropout.', 'A neuron stays on with probability 1 − p = 0.8.', '1 / 0.8 = ' + DS + ': then 0.8 · 1.25h + 0.2 · 0 = h on average.'], solution: String(DS), wrong: ['0.8', '0.2'] });

  const DEC = f64(() => { const p = T.param(T.tensor([1.5])), o = new T.optim.AdamW([p], { lr: 0.01, weightDecay: 0.5 }); for (let k = 0; k < 200; k++) { p.grad = T.zeros([1]); o.step(); } return one(p); });
  ex({ id: 'ne-x-decay', chapter: 'ne-reg', title: 'A weight left to decay', level: 2, kind: 'number',
    text: ['A weight of 1.5 is trained with AdamW, learning rate η = 0.01 and weight decay λ = 0.5. The data never pushes it: its gradient is zero at every step. What is it after 200 steps? Give four decimals.'],
    answer: r4(DEC), tolerance: 0.0005, whyNot: { '1.5': 'With a zero gradient Adam’s own step is zero, but the decay still shrinks the weight every step.', [String(r4(1.5 * 0.99 ** 200))]: 'That uses 1 − η = 0.99 as the factor; the decay multiplies by 1 − ηλ = 0.995.', [String(r4(1.5 * 0.995))]: 'That is one step; the decay acts at every one of the 200.' },
    hints: ['Each step multiplies the weight by 1 − ηλ before the gradient step.', '1 − 0.01 · 0.5 = 0.995, two hundred times.', '1.5 · 0.995²⁰⁰ = ' + r4(DEC) + ': a weight the data does not need fades to about a third.'],
    solution: String(r4(DEC)), wrong: ['1.5', String(r4(1.5 * 0.99 ** 200))] });

  const EL = [0.62, 0.48, 0.44, 0.45, 0.41, 0.43, 0.42, 0.44, 0.46, 0.4], PAT = 3;
  const early = (() => { let best = 0, wait = 0; for (let i = 1; i < EL.length; i++) { if (EL[i] < EL[best]) { best = i; wait = 0; } else if (++wait >= PAT) return { stop: i + 1, best: best + 1, loss: EL[best] }; } return { stop: EL.length, best: best + 1, loss: EL[best] }; })();
  const ecells = { 'stops after check|value': early.stop, 'weights kept from check|value': early.best, 'their held-out loss|value': early.loss };
  ex({ id: 'ne-x-early', chapter: 'ne-reg', title: 'When early stopping stops', level: 2, kind: 'table',
    text: ['The held-out loss is checked every 100 steps. Checks 1 to 10 give:', { table: [['check', ...EL.map((_, i) => String(i + 1))], ['loss', ...EL.map(String)]] }, 'With patience ' + PAT + ' (stop after ' + PAT + ' checks in a row without a new lowest), after which check does training stop, which check’s weights are kept, and what is their held-out loss?'],
    rows: ['stops after check', 'weights kept from check', 'their held-out loss'], cols: ['value'], corner: 'patience ' + PAT, cells: ecells, tolerance: 0.0005,
    hints: ['Walk through the checks keeping the lowest so far and a counter of checks since it.', 'Check 4 (0.45) does not beat 0.44, but check 5 (0.41) does, and the counter starts again.', 'After 0.41 at check 5, checks 6, 7 and 8 are three in a row without beating it: stop after check ' + early.stop + ', with the weights of check ' + early.best + ' (loss ' + early.loss + '). The 0.40 at check 10 is never seen: patience is a trade-off.'],
    solution: str(ecells), wrong: [{ 'stops after check|value': '10', 'weights kept from check|value': '10', 'their held-out loss|value': '0.4' }, { 'stops after check|value': '4', 'weights kept from check|value': '3', 'their held-out loss|value': '0.44' }] });

  const ofit = { data: { set: 'moons', n: 60, seed: 9, opts: { noise: 0.35 } }, hidden: { seed: 99, n: 600 }, metric: 'accuracy', target: 0.85 };
  C.exercise(Object.assign({ id: 'ne-x-wf-overfit', chapter: 'ne-reg', title: 'Sixty noisy points', level: 3, kind: 'workflow',
    text: ['Only 60 noisy points of the two moons, and a network with two hidden layers of 64 neurons trained for 300 epochs: it memorises them. Change the network or its training so that it classifies 600 hidden points with accuracy at least 0.85.'],
    limits: { blocks: ['split', 'mlp', 'train', 'evaluate'], models: ['mlp'], epochs: 300 },
    starter: () => pipe(ofit.data, 'mlp', { hidden: [64, 64] }, { epochs: 300, lr: 0.01 }),
    hints: ['Run the starter and open the Train block: the training loss heads for zero while the validation loss climbs.', 'Any of the lesson’s cures works here: dropout in the MLP block, weight decay, or stopping far sooner. A smaller network memorises less too.', 'MLP (64, 64) with dropout 0.7, Train 300 epochs at 0.01: accuracy 0.875. Weight decay 0.03 gives 0.858; the starter alone gets 0.802.'],
    solution: pipe(ofit.data, 'mlp', { hidden: [64, 64], dropout: 0.7 }, { epochs: 300, lr: 0.01 }),
    wrong: [pipe(ofit.data, 'mlp', { hidden: [64, 64] }, { epochs: 300, lr: 0.01 }), pipe(ofit.data, 'mlp', { hidden: [64, 64], weightDecay: 0.01 }, { epochs: 300, lr: 0.01 })] }, ofit));

  ex({ id: 'ne-x-norm-choice', chapter: 'ne-reg', title: 'Normalising with tiny batches', level: 1, kind: 'choice',
    text: ['A model is so large that only 2 examples fit in memory at a time, so it trains on batches of 2. Which normalisation should its layers use?'],
    options: ['Layer normalisation', 'Batch normalisation', 'Either: they compute the same thing', 'Neither works with small batches'], answer: 0,
    whyNot: { 1: 'Batch norm estimates each feature’s mean and spread from the batch: from 2 examples those estimates are mostly noise.', 2: 'They standardise along different axes: over the batch, or over each example’s own features.', 3: 'Layer norm does not look at the other examples at all.' },
    hints: ['Over what does each method compute its mean and spread?', 'Batch norm: over the examples of the batch, per feature. Layer norm: over the features, per example.', 'Layer norm: each example is normalised with its own statistics, so the batch size does not matter.'], solution: 0, wrong: [1, 2] });

  const BNV = [2, 4, 6, 12], BNO = f64(() => Array.from(T.batchNorm(T.tensor(BNV.map((v) => [v]))).values())), bncells = {};
  BNV.forEach((v, i) => { bncells['x = ' + v + '|output'] = r3(BNO[i]); });
  const bnMean = BNV.reduce((a, b) => a + b, 0) / BNV.length, bnSd = Math.sqrt(BNV.reduce((a, b) => a + (b - bnMean) ** 2, 0) / BNV.length);
  ex({ id: 'ne-x-bn', chapter: 'ne-reg', title: 'Batch norm by hand', level: 2, kind: 'table',
    text: ['One feature takes the values 2, 4, 6 and 12 across a batch of four examples. Batch-normalise it with γ = 1 and β = 0, using the mean and the standard deviation of the batch (dividing by n, as batch norm does). Three decimals.'],
    rows: BNV.map((v) => 'x = ' + v), cols: ['output'], corner: 'example', cells: bncells, tolerance: 0.002,
    hints: ['Subtract the batch mean, then divide by the batch’s standard deviation.', 'The mean is ' + bnMean + '; the squared deviations are 16, 4, 0 and 36, so the variance is 56 / 4 = 14.', 'The spread is √14 = ' + r3(bnSd) + ', so the outputs are ' + BNO.map((v) => sign(r3(v))).join(', ') + '.'],
    solution: str(bncells), wrong: [Object.fromEntries(BNV.map((v) => ['x = ' + v + '|output', String(r3((v - bnMean) / Math.sqrt(56 / 3)))])), Object.fromEntries(BNV.map((v) => ['x = ' + v + '|output', String(r3(v - bnMean))]))] });
})(globalThis.Nooral = globalThis.Nooral || {});
