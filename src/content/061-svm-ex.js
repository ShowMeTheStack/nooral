// Part 6, support vector machines: the exercises. Every number is worked out here by the engines (S.ML's SMO solver
// and kernels, the topics' own runs) on small fixed inputs, so the text, the answers and the hints always agree.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, WF = S.WF;
  if (!C || !ML || !WF || C.exercises.has('sv-x-width')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const rbf = (g) => ML.kernel({ kernel: 'rbf' }, g), poly = (d) => ML.kernel({ kernel: 'poly', degree: d, coef0: 1 }, 1);
  /** the frames' data of a topic run on an input (from the topic itself) */
  const meta = (id, inp) => { const t = S.byId(id); if (!t) return []; const r = t.run(S.inputOf(t, Object.assign(S.inputOf(t), inp || {}))); return (r.frames || r).map((f) => f.meta || {}); };
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  const ex = (x) => C.exercise(x);

  // =========================================================================================================
  // Chapter 1: the widest street
  // =========================================================================================================
  const wW = [3, 4], width = 2 / Math.hypot(...wW);
  ex({ id: 'sv-x-width', chapter: 'sv-margin', title: 'How wide is the street?', level: 1, kind: 'number',
    text: ['A linear SVM has learned f(x) = 3x₁ + 4x₂ − 2. The edges of its street are where f = +1 and f = −1. How wide is the street?'],
    answer: r3(width), tolerance: 0.0005, whyNot: { '0.2': 'That is 1 / ‖w‖, the distance from the line to one edge: the street spans both sides.', '5': 'That is ‖w‖ itself; the width is 2 / ‖w‖.', '1': 'The width depends on w, not on the bias.' },
    hints: ['The width of the street is 2 / ‖w‖, with w = (3, 4).', '‖w‖ = √(3² + 4²) = 5.', 'Width = 2 / 5 = ' + r3(width) + '. The bias −2 only slides the street; it does not change its width.'], solution: String(r3(width)), wrong: ['0.2', '5'] });

  const PX = [[0, 0], [1, 0], [0, 1], [2, 2], [3, 2], [2, 3]], PY = [0, 0, 0, 1, 1, 1], NAMES = ['A', 'B', 'C', 'D', 'E', 'F'];
  const hard = new ML.SVC({ kernel: 'linear', C: 1000 }).fit(PX, PY), hm = hard.machines[0];
  const svNames = hard.support.map((i) => NAMES[i]).sort();
  const hw = [0, 1].map((j) => -hm.sv.reduce((s, i, t) => s + hm.coef[t] * PX[i][j], 0)), hb = hm.rho;
  const ptsText = 'Six parcels by weight and size: A (0, 0), B (1, 0) and C (0, 1) went in the blue van (y = −1); D (2, 2), E (3, 2) and F (2, 3) in the orange van (y = +1).';
  ex({ id: 'sv-x-sv-set', chapter: 'sv-margin', title: 'Which points hold the street?', level: 2, kind: 'set',
    text: [ptsText, 'A linear SVM with a hard margin separates them. Which points are its support vectors? Write them as a set, like {A, B}.'],
    answer: '{' + svNames.join(', ') + '}', caseless: true, placeholder: 'like {A, B}',
    whyNot: 'The support vectors are the points closest to the other class: the street rests on them.',
    hints: ['The two groups face each other along the direction (1, 1). Which points of each group are nearest the other?', 'Along x₁ + x₂, the blue points reach 1 at B and C, and the orange points start at 4 with D.', 'The street runs between x₁ + x₂ = 1 and x₁ + x₂ = 4, so its edges touch ' + svNames.join(', ') + '. The engine finds w = (' + r3(hw[0]) + ', ' + r3(hw[1]) + ') and b = ' + r3(hb) + ', with exactly these support vectors.'],
    solution: '{' + svNames.join(', ') + '}', wrong: ['{B, C}', '{A, B, C, D, E, F}', '{B, C, D, E, F}'] });

  const noA = new ML.SVC({ kernel: 'linear', C: 1000 }).fit(PX.slice(1), PY.slice(1)), nm = noA.machines[0];
  const nw = [0, 1].map((j) => -nm.sv.reduce((s, i, t) => s + nm.coef[t] * PX.slice(1)[i][j], 0));
  ex({ id: 'sv-x-remove', chapter: 'sv-margin', title: 'Delete a point', level: 1, kind: 'choice',
    text: [ptsText, 'The SVM is fitted, then parcel A is deleted and the SVM fitted again. What happens to the line?'],
    options: ['Nothing: A was not a support vector, so the line stays where it is', 'The street gets narrower', 'The street gets wider, because there are fewer points', 'The line turns towards A'], answer: 0,
    whyNot: { 1: 'Deleting a point can only free the street, never squeeze it.', 2: 'The street is held by B, C and D, which are all still there.', 3: 'A has α = 0: it has no pull on the line at all.' },
    hints: ['Which points does the line depend on?', 'Only the support vectors, here B, C and D, have α > 0.', 'A is not one of them, so the fit is unchanged: the engine gives w = (' + r3(nw[0]) + ', ' + r3(nw[1]) + ') again, as before.'], solution: 0, wrong: [1, 2] });

  ex({ id: 'sv-x-c-choice', chapter: 'sv-margin', title: 'What C does', level: 1, kind: 'choice',
    text: ['A soft-margin SVM is trained on overlapping classes with C = 1, then again with C = 100. Which is true of the second fit?'],
    options: ['Its street is narrower and fewer points sit inside it', 'Its street is wider and more points sit inside it', 'It has more support vectors', 'It is the same: C only changes how long training takes'], answer: 0,
    whyNot: { 1: 'That is what a smaller C does: slack becomes cheap.', 2: 'Fewer points pay slack, so fewer are support vectors: 14 at C = 1 and 8 at C = 100 in the lesson.', 3: 'C is part of the objective ½‖w‖² + C·Σξ, so it changes the answer.' },
    hints: ['C is the price of a unit of slack.', 'A dearer slack makes the solver keep points out of the street, even at the cost of a larger w.', 'Larger w means a narrower street (2 / ‖w‖), with fewer points inside it and fewer support vectors.'], solution: 0, wrong: [1, 2] });

  const sm = meta('svm-soft-c', { set: 'overlap', seed: 1, C: 3 }).filter((m) => m.phase === 'sweep');
  const wcurve = sm.map((m) => [Math.log10(m.C), r3(m.width)]);
  ex({ id: 'sv-x-c-curve', chapter: 'sv-margin', title: 'The street as C grows', level: 2, kind: 'predict-curve',
    text: ['The 40 overlapping points of the lesson are fitted with C = 0.01, 0.1, 1, 10 and 100. Sketch the width of the street against log₁₀ C, from −2 to 2.'],
    plot: { xr: [-2.3, 2.3], yr: [0, 6], xname: 'log₁₀ C', yname: 'street width', given: wcurve.slice(0, 1), note: 'Drag from log₁₀ C = −2 to 2. The faint point is C = 0.01.' },
    answer: wcurve, tolerance: 0.45, checks: ['falling'],
    hints: ['A dearer slack makes the street narrower.', 'The big change comes early: going from C = 0.01 to 0.1 already cuts the width by more than half.', 'The widths are ' + wcurve.map(([x, y]) => 'C = ' + r2(Math.pow(10, x)) + ': ' + y).join(', ') + '.'],
    solution: wcurve, wrong: [wcurve.map(([x]) => [x, 2]), wcurve.map(([x], i) => [x, wcurve[wcurve.length - 1 - i][1]])] });

  const HW = [1, -1], HB = 0.5, HX = [[2, 0], [0.5, 0.5], [1, 2], [2, 1]], HY = [1, 1, -1, -1];
  const hm2 = HX.map((x, i) => HY[i] * (dot(HW, x) + HB)), hl = hm2.map((m) => Math.max(0, 1 - m));
  const hrows = ['P (2, 0), y = +1', 'Q (0.5, 0.5), y = +1', 'R (1, 2), y = −1', 'T (2, 1), y = −1'], hcells = {};
  hrows.forEach((r, i) => { hcells[r + '|m = y·f(x)'] = r3(hm2[i]); hcells[r + '|hinge'] = r3(hl[i]); });
  ex({ id: 'sv-x-hinge-table', chapter: 'sv-margin', title: 'The hinge loss of four points', level: 2, kind: 'table',
    text: ['An SVM has f(x) = x₁ − x₂ + 0.5. For each point, fill in m = y·f(x) and its hinge loss max(0, 1 − m).'],
    rows: hrows, cols: ['m = y·f(x)', 'hinge'], corner: 'point', cells: hcells, tolerance: 0.0005,
    hints: ['f(P) = 2 − 0 + 0.5 = 2.5, and y = +1, so m = 2.5: safely outside the street, loss 0.', 'For the blue points y = −1 flips the sign: f(R) = 1 − 2 + 0.5 = −0.5, so m = +0.5.', hrows.map((r, i) => r.split(' ')[0] + ': m = ' + r3(hm2[i]) + ', hinge ' + r3(hl[i])).join('; ') + '. T is on the wrong side.'],
    solution: Object.fromEntries(Object.entries(hcells).map(([k, v]) => [k, String(v)])),
    // two slips: m taken as f(x), forgetting y; and the hinge as 1 − m without the max
    wrong: [Object.fromEntries(hrows.flatMap((r, i) => { const m = dot(HW, HX[i]) + HB; return [[r + '|m = y·f(x)', String(r3(m))], [r + '|hinge', String(r3(Math.max(0, 1 - m)))]]; })),
      Object.fromEntries(hrows.flatMap((r, i) => [[r + '|m = y·f(x)', String(r3(hm2[i]))], [r + '|hinge', String(r3(1 - hm2[i]))]]))] });

  const MS = [1.4, 0.3, -0.5], hsum = MS.reduce((s, m) => s + Math.max(0, 1 - m), 0);
  ex({ id: 'sv-x-hinge-sum', chapter: 'sv-margin', title: 'Add up the hinge', level: 1, kind: 'number',
    text: ['Three points have margins m = y·f(x) of ' + MS.map((v) => String(v).replace('-', '−')).join(', ') + '. What is their total hinge loss, Σ max(0, 1 − m)?'],
    answer: r3(hsum), tolerance: 0.0005, whyNot: { '1.5': 'The point at m = 0.3 is inside the street: it loses 0.7 too.', '1.8': 'A point with m = 1.4 is outside the street and loses nothing, not −0.4.', '4.2': 'Did you add 1 + m instead of 1 − m?' },
    hints: ['m = 1.4 is beyond the edge: loss 0.', 'm = 0.3 is inside the street: loss 1 − 0.3 = 0.7. m = −0.5 is on the wrong side: loss 1.5.', 'Total: 0 + 0.7 + 1.5 = ' + r3(hsum) + '.'], solution: String(r3(hsum)), wrong: ['1.5', '1.8'] });

  const objRef = (w, b, X, y, Cc) => 0.5 * dot(w, w) + Cc * X.reduce((s, x, i) => s + Math.max(0, 1 - y[i] * (dot(w, x) + b)), 0);
  ex({ id: 'sv-x-code-hinge', chapter: 'sv-margin', title: 'Write the soft-margin objective', level: 3, kind: 'code',
    text: ['Write `svmObjective(w, b, X, y, C)`: w is an array of weights, b a number, X an array of points (arrays as long as w), y their labels +1 or −1. Return ½‖w‖² + C · Σ max(0, 1 − yᵢ(w·xᵢ + b)).'],
    fn: 'svmObjective', params: ['w', 'b', 'X', 'y', 'C'], start: 'function svmObjective(w, b, X, y, C) {\n  \n}', tolerance: 1e-9, ref: objRef,
    cases: [[HW, HB, HX, HY, 1], [HW, HB, HX, HY, 10], [[3, 4], -2, [[1, 0], [0, 1], [0, 0]], [1, 1, -1], 0.5], [[0.5], 0, [[3], [-3], [0.5]], [1, -1, 1], 2], [[1, 1, 1], -1, [[1, 1, 1], [0, 0, 0]], [1, -1], 1]],
    hints: ['Two parts: half the sum of the squared weights, and C times the summed hinge.', 'For each point, m = y·(w·x + b); its hinge is Math.max(0, 1 − m).', 'function svmObjective(w, b, X, y, C) {\n  const dot = (a, c) => a.reduce((s, v, i) => s + v * c[i], 0);\n  let h = 0;\n  X.forEach((x, i) => { h += Math.max(0, 1 - y[i] * (dot(w, x) + b)); });\n  return 0.5 * dot(w, w) + C * h;\n}'],
    solution: 'function svmObjective(w, b, X, y, C) {\n  const dot = (a, c) => a.reduce((s, v, i) => s + v * c[i], 0);\n  let h = 0;\n  X.forEach((x, i) => { h += Math.max(0, 1 - y[i] * (dot(w, x) + b)); });\n  return 0.5 * dot(w, w) + C * h;\n}',
    wrong: ['function svmObjective(w, b, X, y, C) { const dot = (a, c) => a.reduce((s, v, i) => s + v * c[i], 0); let h = 0; X.forEach((x, i) => { h += 1 - y[i] * (dot(w, x) + b); }); return 0.5 * dot(w, w) + C * h; }',
      'function svmObjective(w, b, X, y, C) { const dot = (a, c) => a.reduce((s, v, i) => s + v * c[i], 0); let h = 0; X.forEach((x, i) => { h += Math.max(0, 1 - y[i] * (dot(w, x) + b)); }); return dot(w, w) + C * h; }'] });

  // =========================================================================================================
  // Chapter 2: kernels
  // =========================================================================================================
  const ka = [1, 2], kb = [2, 0], kv = rbf(0.5)(ka, kb);
  ex({ id: 'sv-x-rbf-value', chapter: 'sv-kernels', title: 'An RBF kernel value', level: 1, kind: 'number',
    text: ['With γ = 0.5, what is the RBF kernel K(x, x′) = exp(−γ‖x − x′‖²) for x = (1, 2) and x′ = (2, 0)? Give four decimals.'],
    answer: r4(kv), tolerance: 0.00015, whyNot: { [String(r4(Math.exp(-5)))]: 'That leaves out γ: multiply the squared distance by 0.5 first.', [String(r4(Math.exp(-0.5 * Math.sqrt(5))))]: 'Use the squared distance, 5, not the distance √5.', [String(r4(Math.exp(-0.5 * 3)))]: 'The squared distance is (1 − 2)² + (2 − 0)² = 5.' },
    hints: ['First the squared distance: (1 − 2)² + (2 − 0)².', 'It is 1 + 4 = 5, so the exponent is −0.5 · 5 = −2.5.', 'exp(−2.5) = ' + r4(kv) + '.'], solution: String(r4(kv)), wrong: [String(r4(Math.exp(-5))), String(r4(Math.exp(-0.5 * Math.sqrt(5))))] });

  const pa = [2, 1], pb = [1, 3], pv = poly(2)(pa, pb);
  ex({ id: 'sv-x-poly-value', chapter: 'sv-kernels', title: 'A polynomial kernel value', level: 1, kind: 'number',
    text: ['What is the polynomial kernel K(x, x′) = (x·x′ + 1)² for x = (2, 1) and x′ = (1, 3)?'],
    answer: pv, tolerance: 1e-9, whyNot: { '25': 'Add 1 to the dot product before squaring.', '26': 'Square the whole of x·x′ + 1, not only the dot product.', '6': 'The kernel is the square of x·x′ + 1.' },
    hints: ['x·x′ = 2 · 1 + 1 · 3.', 'That is 5; add 1 to get 6.', '6² = ' + pv + '. The same number as the dot product of the six features (1, √2x₁, √2x₂, x₁², x₂², √2x₁x₂) of both points.'], solution: String(pv), wrong: ['25', '26'] });

  const KP = { a: [0, 0], b: [1, 0], c: [1, 1] }, kcells = {};
  for (const [r, c] of [['a', 'a'], ['a', 'b'], ['a', 'c'], ['b', 'b'], ['b', 'c'], ['c', 'c']]) kcells[r + '|' + c] = r4(rbf(1)(KP[r], KP[c]));
  ex({ id: 'sv-x-kernel-table', chapter: 'sv-kernels', title: 'A kernel matrix', level: 2, kind: 'table',
    text: ['Three points: a = (0, 0), b = (1, 0) and c = (1, 1). Fill in the upper half of their RBF kernel matrix with γ = 1, K = exp(−‖x − x′‖²), to four decimals. (The matrix is symmetric, so the lower half is the same.)'],
    rows: ['a', 'b', 'c'], cols: ['a', 'b', 'c'], corner: 'K', cells: kcells, tolerance: 0.00015,
    hints: ['A point against itself is at distance 0: K = 1 on the diagonal.', 'a and b are 1 apart, squared distance 1: exp(−1). a and c are √2 apart, squared distance 2.', 'Diagonal 1; K(a, b) = K(b, c) = ' + kcells['a|b'] + '; K(a, c) = ' + kcells['a|c'] + '.'],
    solution: Object.fromEntries(Object.entries(kcells).map(([k, v]) => [k, String(v)])) });

  ex({ id: 'sv-x-ring-choice', chapter: 'sv-kernels', title: 'A kernel for a ring', level: 1, kind: 'choice',
    text: ['Sensors in a round pond and on the grass around it report only their map position (x₁, x₂), measured from the pond’s centre. Which kernels can let an SVM separate water from grass? Pick every one that can.'],
    options: ['The linear kernel x·x′', 'The polynomial kernel (x·x′ + 1)²', 'The RBF kernel exp(−γ‖x − x′‖²)', 'None: an SVM can only draw straight lines'], answer: [1, 2],
    whyNot: { 0: 'A straight line cannot go round the pond: the lesson’s best line got 53%.', 3: 'With a kernel the boundary is straight only in the lifted space; on the map it can be a circle.' },
    hints: ['The shore is a circle, x₁² + x₂² = r².', 'The degree-2 kernel includes the features x₁² and x₂², so a circle is a straight cut in its lifted space. The RBF kernel can bend any way.', 'Both the polynomial kernel of degree 2 and the RBF kernel can draw the circle; the linear kernel cannot.'], solution: [1, 2], wrong: [[0], [2]] });

  const LA = -2, LB1 = 0.4, LC = 1, lcx = -LB1 / (2 * LA), lr = Math.sqrt(lcx * lcx - LC / LA);
  ex({ id: 'sv-x-lift-radius', chapter: 'sv-kernels', title: 'The circle under the plane', level: 3, kind: 'number',
    text: ['Points are lifted to (x₁, x₂, z) with z = x₁² + x₂², and a linear SVM finds the plane −2z + 0.4x₁ + 1 = 0. On the original map, the boundary is a circle. What is its radius? Give three decimals.'],
    answer: r3(lr), tolerance: 0.0015, whyNot: { [String(r3(Math.sqrt(0.5)))]: 'That ignores the 0.4x₁ term, which moves the centre and changes the radius.', [String(r3(lr * lr))]: 'That is the radius squared.', '0.5': 'Divide through by −2 and complete the square in x₁.' },
    hints: ['Put z = x₁² + x₂² into the plane and divide by −2: x₁² + x₂² − 0.2x₁ − 0.5 = 0.', 'Complete the square: (x₁ − 0.1)² − 0.01 + x₂² − 0.5 = 0.', 'So (x₁ − ' + r3(lcx) + ')² + x₂² = ' + r3(lr * lr) + ': a circle centred at (' + r3(lcx) + ', 0) with radius √' + r3(lr * lr) + ' = ' + r3(lr) + '.'], solution: String(r3(lr)), wrong: [String(r3(Math.sqrt(0.5))), String(r3(lr * lr))] });

  const binom = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return Math.round(r); };
  const nfeat = binom(3 + 3, 3);
  ex({ id: 'sv-x-poly-count', chapter: 'sv-kernels', title: 'How many features does the kernel stand for?', level: 2, kind: 'number',
    text: ['With two inputs, the kernel (x·x′ + 1)^d stands for 3, 6, 10, 15 and 21 features for d = 1 to 5: every product of the inputs up to degree d, the constant included. How many does (x·x′ + 1)³ stand for with three inputs?'],
    answer: nfeat, tolerance: 1e-9, whyNot: { '10': 'That is for two inputs.', '27': 'That counts ordered products; x₁x₂ and x₂x₁ are the same feature.', '19': 'Remember the constant term.' },
    hints: ['Count by degree: degree 0 has 1 term, degree 1 has 3 (x₁, x₂, x₃).', 'Degree 2 has 6 (three squares and three pairs); degree 3 has 10.', '1 + 3 + 6 + 10 = ' + nfeat + ', which is the number of ways to choose 3 from 3 + 3 = 6.'], solution: String(nfeat), wrong: ['10', '27'] });

  const gm = meta('svm-rbf', { set: 'moons', C: 1, seed: 1 }).filter((m) => m.phase === 'gamma');
  const gcurve = gm.map((m) => [r3(Math.log10(m.gamma)), r3(m.test)]);
  ex({ id: 'sv-x-gamma-curve', chapter: 'sv-kernels', title: 'Held-back accuracy as γ grows', level: 2, kind: 'predict-curve',
    text: ['An RBF SVM with C = 1 learns 70 noisy moon points and is scored on 30 it never saw, for γ = 0.1, 0.3, 1, 3, 10, 30 and 100. Its training accuracy climbs from 90% to 100%. Sketch the accuracy on the 30 held-back points against log₁₀ γ.'],
    plot: { xr: [-1.2, 2.2], yr: [0.5, 1], xname: 'log₁₀ γ', yname: 'held-back accuracy', given: gcurve.slice(0, 1), note: 'Drag from log₁₀ γ = −1 to 2. The faint point is γ = 0.1.' },
    answer: gcurve, tolerance: 0.04, checks: ['peak'],
    hints: ['Small γ: wide bumps and a nearly straight boundary. Large γ: tiny bumps round every training point.', 'Both ends do worse on new points than the middle does.', 'The held-back accuracy is ' + gm.map((m) => Math.round(100 * m.test) + '%').join(', ') + ': it rises, stays flat over the middle, and falls at the large end.'],
    solution: gcurve, wrong: [gcurve.map(([x]) => [x, 0.83]), gcurve.map(([x], i) => [x, r3(0.77 + (0.23 * i) / (gcurve.length - 1))])] });

  ex({ id: 'sv-x-gamma-choice', chapter: 'sv-kernels', title: 'A boundary that wraps every point', level: 2, kind: 'choice',
    text: ['An RBF SVM with C = 100 on 70 moon points, scored on 30 more:', { table: [['γ', 'Support vectors', 'Training', 'Held back'], ['0.1', '28', '87%', '100%'], ['1', '14', '96%', '83%'], ['30', '59', '100%', '70%'], ['100', '69', '100%', '63%']] }, 'A colleague picks γ = 100 because it gets every training point right. What is wrong with that?'],
    options: ['It overfits: almost every point is a support vector and the held-back accuracy is the worst; choose γ (and C) by the held-back score', 'Nothing: 100% training accuracy is the goal', 'γ = 100 underfits: the boundary is nearly straight', 'The number of support vectors should be as large as possible'], answer: 0,
    whyNot: { 1: 'The model is for new points; on those it is the worst of the four.', 2: 'A large γ makes tiny bumps: the boundary bends round every point. That is the opposite of straight.', 3: 'A support vector for nearly every point is a sign the boundary is memorising, not generalising.' },
    hints: ['Compare the two accuracy columns.', 'At γ = 100, 69 of 70 points are support vectors: each has its own tiny bump.', 'The held-back score is best at γ = 0.1 (100%) and worst at γ = 100 (63%). Choose by the held-back score: the large γ overfits.'], solution: 0, wrong: [1, 2] });

  const rbfRef = (a, b, g) => rbf(g)(a, b);
  ex({ id: 'sv-x-code-rbf', chapter: 'sv-kernels', title: 'Write the RBF kernel', level: 2, kind: 'code',
    text: ['Write `rbf(a, b, gamma)`: a and b are points (arrays of the same length, any length), gamma a positive number. Return exp(−gamma · ‖a − b‖²).'],
    fn: 'rbf', params: ['a', 'b', 'gamma'], start: 'function rbf(a, b, gamma) {\n  \n}', tolerance: 1e-12, ref: rbfRef,
    cases: [[[1, 2], [2, 0], 0.5], [[0, 0], [0, 0], 3], [[1, 0, 0], [0, 1, 0], 1], [[0.3], [-0.7], 10], [[1, 2, 3, 4], [4, 3, 2, 1], 0.05], [[0, 0], [1, 1], 0.001]],
    hints: ['Add up the squared differences of the coordinates.', 'Then multiply by −gamma and take Math.exp.', 'function rbf(a, b, gamma) {\n  let d = 0;\n  for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2;\n  return Math.exp(-gamma * d);\n}'],
    solution: 'function rbf(a, b, gamma) {\n  let d = 0;\n  for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2;\n  return Math.exp(-gamma * d);\n}',
    wrong: ['function rbf(a, b, gamma) { let d = 0; for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2; return Math.exp(-gamma * Math.sqrt(d)); }', 'function rbf(a, b, gamma) { let d = 0; for (let i = 0; i < 2; i++) d += (a[i] - b[i]) ** 2; return Math.exp(-gamma * d); }'] });

  const ring = { data: { set: 'circles', n: 300, seed: 3, opts: { noise: 0.1 } }, hidden: { seed: 33, n: 600 }, metric: 'accuracy', target: 0.95, task: 'classification' };
  C.exercise(Object.assign({ id: 'sv-x-wf-ring', chapter: 'sv-kernels', title: 'An SVM round a ring', level: 2, kind: 'workflow',
    text: ['A ring of one class round a disc of the other. Build a workflow with a support vector machine that classifies 600 hidden points with accuracy at least 0.95.'],
    limits: { blocks: ['split', 'scale', 'svm', 'train', 'evaluate'], models: ['svm'] },
    hints: ['Try the linear kernel first and look at the boundary in the Train block.', 'A straight line cannot go round the disc. Switch the kernel to one that can bend.', 'Split → SVM with the RBF kernel (C = 1, γ automatic) → Train → Evaluate: accuracy 0.99 on the hidden points, against 0.56 for the linear kernel.'],
    solution: pipe(ring.data, 'svm', { kernel: 'rbf' }), wrong: [pipe(ring.data, 'svm', { kernel: 'linear' }), pipe(ring.data, 'svm', { kernel: 'linear', C: 100 })] }, ring));

  // =========================================================================================================
  // Chapter 3: inside the solver
  // =========================================================================================================
  const al = hm.coef.map((c) => r3(Math.abs(c))), ys = hm.sv.map((i) => (PY[i] === 1 ? 1 : -1));
  const ww = [0, 1].map((j) => r3(hm.sv.reduce((s, i, t) => s + al[t] * ys[t] * PX[i][j], 0)));
  const dIdx = hm.sv.find((i) => PY[i] === 1), bb = r3(1 - dot(ww, PX[dIdx]));
  const alText = hm.sv.map((i, t) => NAMES[i] + ' (' + PX[i].join(', ') + '), y = ' + (ys[t] > 0 ? '+1' : '−1') + ', α = ' + al[t]).join('; ');
  ex({ id: 'sv-x-w-from-alpha', chapter: 'sv-solver', title: 'Build w from the multipliers', level: 3, kind: 'table',
    text: [ptsText, 'The solver returns three support vectors: ' + alText + '. Every other α is 0. Fill in w = Σ αᵢyᵢxᵢ, and b from the orange support vector, which lies on the edge f = +1 (three decimals).'],
    rows: ['w₁', 'w₂', 'b'], cols: ['value'], corner: '', cells: { 'w₁|value': ww[0], 'w₂|value': ww[1], 'b|value': bb }, tolerance: 0.005,
    hints: ['Each support vector adds α·y times its coordinates; the blue ones (y = −1) subtract.', 'w = −' + al[0] + '·(1, 0) − ' + al[1] + '·(0, 1) + ' + al[2] + '·(2, 2).', 'w = (' + ww[0] + ', ' + ww[1] + '). On the edge f(D) = w·(2, 2) + b = 1, so b = 1 − ' + r3(dot(ww, PX[dIdx])) + ' = ' + String(bb).replace('-', '−') + ' (rounding the α moves b in the third decimal, so answers within 0.005 are accepted).'],
    solution: { 'w₁|value': String(ww[0]), 'w₂|value': String(ww[1]), 'b|value': String(bb) },
    wrong: [{ 'w₁|value': String(r3(ww[0] + 2 * al[0])), 'w₂|value': String(r3(ww[1] + 2 * al[1])), 'b|value': String(bb) }] });

  const AP = [0.3, 0.5], AN = [0.6], amiss = r3(AP[0] + AP[1] - AN[0]);
  ex({ id: 'sv-x-alpha-balance', chapter: 'sv-solver', title: 'The missing multiplier', level: 1, kind: 'number',
    text: ['An SVM has four support vectors. The two orange ones (y = +1) have α = ' + AP.join(' and ') + '; the blue ones (y = −1) have α = ' + AN[0] + ' and an unknown α₄. What is α₄?'],
    answer: amiss, tolerance: 0.0005, whyNot: { '1.4': 'The blue multipliers are subtracted in Σ αᵢyᵢ: they must balance the orange ones, not add to them.', '0.8': 'That is the orange total; the blue α₄ makes up the difference with the other blue one.' },
    hints: ['The multipliers satisfy Σ αᵢyᵢ = 0.', 'So the orange total equals the blue total: 0.3 + 0.5 = 0.6 + α₄.', 'α₄ = 0.8 − 0.6 = ' + amiss + '.'], solution: String(amiss), wrong: ['1.4', '0.8'] });

  ex({ id: 'sv-x-alpha-cap', chapter: 'sv-solver', title: 'What a multiplier says', level: 2, kind: 'choice',
    text: ['A soft-margin SVM with C = 1 is fitted. Point P ends with α = 1, point Q with α = 0.4 and point R with α = 0. Which description fits?'],
    options: ['P is inside the street or on the wrong side; Q lies exactly on an edge; R is outside the street on its own side', 'P is far outside the street; Q is inside it; R is on the wrong side', 'All three are support vectors', 'R matters most, because its α is smallest'], answer: 0,
    whyNot: { 1: 'A point far outside the street has no pull on the line: α = 0.', 2: 'R has α = 0, so it is not a support vector.', 3: 'α measures how hard a point pushes on the line; α = 0 means not at all.' },
    hints: ['α = 0: no say. 0 < α < C: on an edge. α = C: paying slack.', 'P has α = C = 1, the cap.', 'P pays slack (inside the street or across it), Q sits exactly on an edge, R is outside the street with no say.'], solution: 0, wrong: [1, 2] });

  const DSV = [[0, 0], [2, 0], [0, 2]], DAY = [0.8, -0.5, -0.3], DB = 0.1, DX = [1, 0];
  const dk = DSV.map((s) => rbf(0.5)(s, DX)), dval = DAY.reduce((s, a, i) => s + a * dk[i], 0) + DB;
  ex({ id: 'sv-x-decision', chapter: 'sv-solver', title: 'The decision value from support vectors', level: 3, kind: 'number',
    text: ['An RBF SVM with γ = 0.5 has three support vectors: (0, 0) with αy = +0.8, (2, 0) with αy = −0.5 and (0, 2) with αy = −0.3, and b = 0.1. What is f(x) = Σ αᵢyᵢ K(xᵢ, x) + b at x = (1, 0)? Give three decimals.'],
    answer: r3(dval), tolerance: 0.0015, whyNot: { [String(r3(DAY.reduce((s, a) => s + a, 0) + DB))]: 'Each αy must be multiplied by the kernel value between its support vector and x.', [String(r3(dval - DB))]: 'Add b at the end.' },
    hints: ['The squared distances from (1, 0) to the support vectors are 1, 1 and 5.', 'The kernel values are exp(−0.5) = ' + r4(dk[0]) + ', ' + r4(dk[1]) + ' and exp(−2.5) = ' + r4(dk[2]) + '.', 'f = 0.8 · ' + r4(dk[0]) + ' − 0.5 · ' + r4(dk[1]) + ' − 0.3 · ' + r4(dk[2]) + ' + 0.1 = ' + r3(dval) + ': positive, so the orange class (+1).'], solution: String(r3(dval)), wrong: [String(r3(DB)), String(r3(dval - DB))] });

  ex({ id: 'sv-x-smo-why', chapter: 'sv-solver', title: 'Why SMO moves two at a time', level: 2, kind: 'choice',
    text: ['SMO solves the dual problem by changing the multipliers α of two points at each step, never one. Why two?'],
    options: ['The constraint Σ αᵢyᵢ = 0 must hold after every step, and changing a single α would break it', 'Two kernel values are cheaper than one', 'Only two points can be support vectors at a time', 'So that each step tests one point of each class'], answer: 0,
    whyNot: { 1: 'One α would need the same kernel row; cost is not the reason.', 2: 'The finished SVM can have any number of support vectors.', 3: 'The lesson’s step 4 moved two blue points: the pair can be from the same class.' },
    hints: ['Which equality do all the multipliers satisfy together?', 'If only αᵢ changes, Σ αᵢyᵢ changes by yᵢ times the change.', 'With two, the change of one is offset by the other: yᵢΔαᵢ + yⱼΔαⱼ = 0. That is the smallest move that keeps the constraint.'], solution: 0, wrong: [1, 3] });

  // =========================================================================================================
  // Chapter 4: more classes, numbers and one class
  // =========================================================================================================
  const K6 = 6, novo = (K6 * (K6 - 1)) / 2;
  ex({ id: 'sv-x-ovo-count', chapter: 'sv-more', title: 'How many machines?', level: 1, kind: 'number',
    text: ['A fruit sorter tells 6 kinds of apple apart with SVMs. How many two-class machines does one-vs-one train?'],
    answer: novo, tolerance: 1e-9, whyNot: { '6': 'That is one-vs-rest: one machine per class.', '30': 'Each pair is counted twice that way: A against B is the same machine as B against A.', '36': 'A class is never trained against itself.' },
    hints: ['One machine for every pair of different kinds.', 'There are 6 · 5 ordered pairs, and each machine covers two of them.', '6 · 5 / 2 = ' + novo + '.'], solution: String(novo), wrong: ['6', '30'] });

  const VOTES = [['A', 'B', 'A'], ['A', 'C', 'C'], ['A', 'D', 'A'], ['B', 'C', 'C'], ['B', 'D', 'D'], ['C', 'D', 'C']];
  const tally = { A: 0, B: 0, C: 0, D: 0 };
  VOTES.forEach(([, , w]) => tally[w]++);
  const winner = ['A', 'B', 'C', 'D'][ML.argmax(['A', 'B', 'C', 'D'].map((k) => tally[k]))];
  ex({ id: 'sv-x-ovo-vote', chapter: 'sv-more', title: 'Count the votes', level: 2, kind: 'choice',
    text: ['Four classes A, B, C and D, and the six one-vs-one machines say, for a new point:', { table: [['Machine', 'Says']].concat(VOTES.map(([a, b, w]) => [a + ' vs ' + b, w])) }, 'Which class does the point get?'],
    options: ['A', 'B', 'C', 'D'], answer: ['A', 'B', 'C', 'D'].indexOf(winner),
    whyNot: { 0: 'A wins two of its three machines, but count every class’s votes.', 1: 'B wins none of its machines.', 3: 'D wins only against B.' },
    hints: ['Each machine gives one vote to the class it says.', 'Count: A ' + tally.A + ', B ' + tally.B + ', C ' + tally.C + ', D ' + tally.D + '.', winner + ' has the most votes, ' + tally[winner] + ': it wins every machine it is in.'], solution: ['A', 'B', 'C', 'D'].indexOf(winner), wrong: [0, 3] });

  const EPS = 0.2, EY = [[1, 0.9], [2, 2.5], [0.5, 0.2], [3, 3.2]], erows = EY.map(([y, f]) => 'y = ' + y + ', f(x) = ' + f), ecells = {};
  EY.forEach(([y, f], i) => { ecells[erows[i] + '|loss'] = r3(Math.max(0, Math.abs(y - f) - EPS)); });
  ex({ id: 'sv-x-eps-table', chapter: 'sv-more', title: 'The ε-insensitive loss', level: 2, kind: 'table',
    text: ['An SVR uses ε = 0.2. For each point fill in its loss max(0, |y − f(x)| − ε).'],
    rows: erows, cols: ['loss'], corner: 'point', cells: ecells, tolerance: 0.0005,
    hints: ['A point within 0.2 of the fit is inside the tube: loss 0.', 'The second point is 0.5 away: 0.3 beyond the tube.', erows.map((r, i) => r + ': ' + ecells[r + '|loss']).join('; ') + '. A point exactly on the tube’s edge, like the last, loses 0.'],
    solution: Object.fromEntries(Object.entries(ecells).map(([k, v]) => [k, String(v)])) });

  ex({ id: 'sv-x-nu', chapter: 'sv-more', title: 'What ν promises', level: 2, kind: 'choice',
    text: ['A one-class SVM with ν = 0.05 is fitted on 1000 minutes of pump readings. What does ν guarantee?'],
    options: ['At most 50 minutes fall outside the border, and at least 50 are support vectors', 'Exactly 50 minutes are flagged as faults', 'At least 50 minutes fall outside the border', '5% of the flagged minutes are real faults'], answer: 0,
    whyNot: { 1: 'ν is a bound, not an exact count: in the lesson ν = 0.1 left only 1 of 100 points outside.', 2: 'It is an upper bound on the outside share.', 3: 'The one-class SVM never sees the fault labels; ν says nothing about how many flags are right.' },
    hints: ['ν bounds two shares, one from each side.', 'The share outside the border is at most ν; the share of support vectors is at least ν.', '5% of 1000 is 50: at most 50 outside, at least 50 support vectors.'], solution: 0, wrong: [1, 2] });

  const flw = { data: { set: 'wildflowers', n: 300, seed: 5 }, hidden: { seed: 55, n: 300 }, metric: 'accuracy', target: 0.97, task: 'classification' };
  C.exercise(Object.assign({ id: 'sv-x-wf-flowers', chapter: 'sv-more', title: 'Three species, one scale', level: 3, kind: 'workflow',
    text: ['The wildflower table measures petals in centimetres, stems in tens of centimetres and leaves by count, for three species. Build a workflow with an SVM that names the species of 300 hidden flowers with accuracy at least 0.97.'],
    limits: { blocks: ['split', 'scale', 'svm', 'train', 'evaluate'], models: ['svm'] },
    hints: ['An SVM with three classes trains one machine per pair; the Train block shows them.', 'The RBF kernel measures distances, and the stem heights are the largest numbers in the table: they swamp the petals.', 'Split → Scale (standard) → SVM (RBF, C = 1) → Train → Evaluate: accuracy 0.987 on the hidden flowers, against 0.943 without scaling.'],
    solution: pipe(flw.data, 'svm', { kernel: 'rbf', C: 1 }, [['scale', { how: 'standard' }]]), wrong: [pipe(flw.data, 'svm', { kernel: 'rbf', C: 1 }), pipe(flw.data, 'svm', { kernel: 'linear', C: 0.01 })] }, flw));
})(globalThis.Nooral = globalThis.Nooral || {});
