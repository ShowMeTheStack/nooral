// Part 9, the forward pass of a small MLP with the value on every wire: one input point flows through a 2–h–2
// network neuron by neuron (each incoming wire carries its input times its weight, the neuron adds them with its
// bias and applies the activation), then softmax turns the two output scores into probabilities.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const DATA = { set: 'moons', n: 120, noise: 0.15, seed: 2 };
  /** the 2–h–2 network: trained on the moons for 300 full-batch Adam steps, or as initialised */
  P.fwdNet = (h, act, trained) => P.once(['fwd', h, act, trained].join(':'), () => {
    const d = P.data(DATA.set, DATA.n, DATA.noise, DATA.seed), S0 = P.SETS.moons;
    const r = P.train({ train: d, hidden: [h], act, lr: 0.05, steps: trained ? 300 : 0, seed: 3, evalEvery: 300, grid: { xr: S0.xr, yr: S0.yr, n: 60, w: 300, h: 200 }, snaps: [trained ? 300 : 0] });
    return { net: r.net, field: r.snaps[0].field, acc: r.evals[r.evals.length - 1].trainAcc, data: d };
  });

  S.topic({
    id: 'p09-forward', family: 'p09-mlp', order: 10, title: 'The forward pass: a value on every wire',
    blurb: 'A multilayer perceptron is neurons in layers, each layer feeding the next. Follow one point through a small network: every wire carries the value at its start times its weight, every neuron adds what arrives and its bias and applies its activation, and at the end softmax turns the output scores into probabilities. The colour of a wire is the sign of its weight and its thickness the size.',
    inputs: [
      { id: 'x1', kind: 'slider', label: 'Point x₁', def: 0.2, min: -1.5, max: 2.5, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'x2', kind: 'slider', label: 'Point x₂', def: 0.9, min: -1, max: 1.5, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'hidden', kind: 'number', label: 'Hidden neurons', def: 3, min: 2, max: 4 },
      { id: 'act', kind: 'choice', label: 'Activation', def: 'tanh', options: P.actOptions(['tanh', 'relu', 'sigmoid']) },
      { id: 'weights', kind: 'choice', label: 'Weights', def: 'trained', options: [{ value: 'trained', label: 'trained on the moons (300 steps)' }, { value: 'random', label: 'as initialised, before training' }] },
    ],
    presets: [
      { label: 'Upper moon, trained', values: { x1: 0.2, x2: 0.9, hidden: 3, act: 'tanh', weights: 'trained' } },
      { label: 'Lower moon, trained', values: { x1: 1.4, x2: -0.4, hidden: 3, act: 'tanh', weights: 'trained' } },
      { label: 'ReLU, four neurons', values: { x1: 0.5, x2: 0.2, hidden: 4, act: 'relu', weights: 'trained' } },
      { label: 'Before training', values: { x1: 0.2, x2: 0.9, hidden: 3, act: 'tanh', weights: 'random' } },
      { label: 'Two sigmoid neurons', values: { x1: -0.8, x2: 0.4, hidden: 2, act: 'sigmoid', weights: 'trained' } },
    ],
    counters: [{ id: 'mul', label: 'multiplications' }, { id: 'add', label: 'additions' }],
    run(inp) {
      const T = S.T, hN = inp.hidden, A = P.ACTS[inp.act], F = H().frames(['mul', 'add']), q = F.q, r2 = P.r2;
      const M = P.fwdNet(hN, inp.act, inp.weights === 'trained'), net = M.net, x = [inp.x1, inp.x2];
      const logitsT = net.predict(T.tensor([x])), z1 = Array.from(net.outputs.dense0.values()), a1 = Array.from(net.outputs.act0.values()), z2 = Array.from(logitsT.values());
      const p = Array.from(T.softmax(logitsT, -1).values());
      const W = P.netWeights(net), B = P.netBiases(net), wmax = Math.max(1, ...W.flat(2).map(Math.abs));
      const L = S.Kit.net.layout({ x: 60, y: 60, w: 380, h: 330, sizes: [2, hN, 2], r: 27 });
      const names = [['x₁', 'x₂'], Array.from({ length: hN }, (_, j) => 'h' + P.sub(j + 1)), ['score 0', 'score 1']];
      const vals = [x, a1, z2];
      /** the network with the values known so far: upto = [layer, neuron] (values shown through it) */
      const draw = (upto, focus) => {
        const items = [];
        W.forEach((Wl, l) => Wl.forEach((row, i) => row.forEach((w, j) => {
          const a = L.pos[l][j], b = L.pos[l + 1][i], dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len, pt = S.Kit.net.weightPaint(w, wmax, 5);
          const lit = focus && focus[0] === l + 1 && focus[1] === i;
          items.push({ key: `fw:w${l}_${i}_${j}`, kind: 'path', d: `M${r2(a.x + ux * 28)},${r2(a.y + uy * 28)} L${r2(b.x - ux * 28)},${r2(b.y - uy * 28)}`, cls: 'weight' + (focus && !lit ? ' faint' : ''), pathAudit: 'edge-x', paint: pt, layer: 'lines', z: lit ? 200 : Math.round(Math.abs(w) * 50) });
          if (lit) items.push({ key: `fw:c${l}_${i}_${j}`, kind: 'chip', x: r2(a.x + dx * 0.36), y: r2(a.y + dy * 0.36), text: H().fx(vals[l][j] * w), size: 14, cls: 'p09w act', layer: 'top' });
        })));
        L.pos.forEach((col, l) => col.forEach((pp, i) => {
          const known = l < upto[0] || (l === upto[0] && i <= upto[1]);
          items.push({ key: `fw:n${l}_${i}`, kind: 'state', x: pp.x, y: pp.y, r: 27, text: known ? H().fx(vals[l][i]) : '', sub: names[l][i], size: 13, cls: 'neuron ' + (l === 0 ? 'input ' : '') + (focus && focus[0] === l && focus[1] === i ? 'act' : '') });
        }));
        items.push(H().text('fw:t0', L.pos[0][0].x, 34, 'input', 'axname', 13, 'middle'), H().text('fw:t1', L.pos[1][0].x, 34, 'hidden (' + A.label + ')', 'axname', 13, 'middle'), H().text('fw:t2', L.pos[2][0].x, 34, 'output', 'axname', 13, 'middle'));
        return items;
      };
      const lines = [];
      const panel = () => lines.map((l, i) => H().text('fw:p:' + i, 600 + (/^\s/.test(l[0]) ? 30 : 0), 90 + i * 28, l[0].trim(), l[1], 15));
      const termsTxt = (ws, xs, cont) => ws.map((w, k) => (k || cont ? (w < 0 ? ' − ' : ' + ') + H().fx(Math.abs(w)) : H().fx(w)) + '·' + (xs[k] < 0 ? '(' + H().fx(xs[k]) + ')' : H().fx(xs[k]))).join('').replace(/^ /, '');
      F.push([...draw([0, 1], null), ...panel()], `The point (x₁, x₂) = (${q(x[0], 1)}, ${q(x[1], 1)}) enters a network with ${P.qpl(q, hN, 'hidden neuron')} (${A.label}) and two outputs, one score per class. The weights are ${inp.weights === 'trained' ? 'those after training on the two moons' : 'the random ones it starts with'}: blue wires are positive weights, vermilion negative, thicker ones larger.`, { phase: 'start', x, hN });
      for (let j = 0; j < hN; j++) {
        const ws = [W[0][j][0], W[0][j][1]];
        F.count('mul', 2).count('add', 2);
        lines.length = 0;
        lines.push(['Hidden neuron h' + P.sub(j + 1), 'strong'], ['z = ' + termsTxt(ws, x), 'math'], ['    ' + (B[0][j] < 0 ? '− ' : '+ ') + H().fx(Math.abs(B[0][j])) + ' = ' + H().fx(z1[j]), 'math'], ['h' + P.sub(j + 1) + ' = ' + A.label + '(' + H().fx(z1[j]) + ') = ' + H().fx(a1[j]), 'math']);
        F.push([...draw([1, j], [1, j]), ...panel()],
          `Hidden neuron h${P.sub(j + 1)}: its wires carry ${q(ws[0])} × ${q(x[0], 1)} = ${q(ws[0] * x[0])} and ${q(ws[1])} × ${q(x[1], 1)} = ${q(ws[1] * x[1])}. With its bias ${q(B[0][j])} they add to z = ${q(z1[j])}, and ${A.label}(z) = ${q(a1[j])}.`,
          { phase: 'hidden', j, ws, b: B[0][j], prods: [ws[0] * x[0], ws[1] * x[1]], z: z1[j], a: a1[j], x });
      }
      for (let k = 0; k < 2; k++) {
        const ws = W[1][k], prods = ws.map((w, j) => w * a1[j]);
        F.count('mul', hN).count('add', hN);
        lines.length = 0;
        lines.push(['Output ' + k, 'strong']);
        for (let a = 0; a < hN; a += 2) lines.push([(a ? '    ' : 's' + P.sub(k) + ' = ') + termsTxt(ws.slice(a, a + 2), a1.slice(a, a + 2), a > 0), 'math']);
        lines.push(['    ' + (B[1][k] < 0 ? '− ' : '+ ') + H().fx(Math.abs(B[1][k])) + ' = ' + H().fx(z2[k]), 'math']);
        F.push([...draw([2, k], [2, k]), ...panel()],
          `Output ${q.n(k)} reads every hidden neuron: its wires carry ${H().list(prods.map((v) => q(v)))}. With its bias ${q(B[1][k])} the score for class ${q.n(k)} is ${q(z2[k])}. The output neurons have no activation: their scores go to softmax.`,
          { phase: 'output', k, ws, b: B[1][k], prods, z: z2[k], h: a1 });
      }
      const win = p[1] > p[0] ? 1 : 0;
      lines.length = 0;
      lines.push(['Softmax', 'strong'], ['p = softmax(' + H().fx(z2[0]) + ', ' + H().fx(z2[1]) + ')', 'math'], ['    = (' + H().fx(p[0], 3) + ', ' + H().fx(p[1], 3) + ')', 'math'], ['class ' + win, 'strong']);
      const probsTxt = [H().text('fw:p0', L.pos[2][0].x + 40, L.pos[2][0].y + 5, 'p = ' + H().fx(p[0], 2), 'c0', 14), H().text('fw:p1', L.pos[2][1].x + 40, L.pos[2][1].y + 5, 'p = ' + H().fx(p[1], 2), 'c1', 14)];
      F.push([...draw([2, 1], null), ...panel(), ...probsTxt],
        `Softmax turns the scores into probabilities: e^score divided by the sum of both. Only the gap matters, ${q(z2[1] - z2[0])} here, so p(class 1) = σ(${q(z2[1] - z2[0])}) = ${q(p[1], 3)} and p(class 0) = ${q(p[0], 3)}. The network says class ${q.n(win)}.`,
        { phase: 'softmax', p, z: z2, gap: z2[1] - z2[0], win });
      // the same network everywhere
      const S0 = P.SETS.moons, ax = P.ax({ prefix: 'fp', x: 600, y: 80, w: 300, h: 200, xr: S0.xr, yr: S0.yr, xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1], xname: 'x₁', yname: 'x₂' });
      const plot = [...ax.items, ...P.boundary(ax, 'fp:b', M.field), ...S.Kit.scatter.points(ax, 'fp:pts', M.data.X, M.data.y, { r: 3 }), S.Kit.scatter.ring(ax, 'fp:ring', Math.max(S0.xr[0], Math.min(S0.xr[1], x[0])), Math.max(S0.yr[0], Math.min(S0.yr[1], x[1])), 'act', 9)];
      plot.push(H().text('fp:k', 600, 340, 'shading: p(class 1) at every point;', 'legend', 13), H().text('fp:k2', 600, 362, 'the line: p = ½', 'legend', 13));
      F.push([...draw([2, 1], null), ...probsTxt, ...plot],
        `The same arithmetic at every point of the plane gives the shading: the network’s p(class 1) everywhere, with its boundary where the two scores are equal. The ringed point is ours.` + (inp.weights === 'trained' ? ` These weights get ${q.pct(M.acc)} of the ${q.n(M.data.X.length)} training points right.` : ` Before training the boundary is wherever the random weights put it: ${q.pct(M.acc)} of the ${q.n(M.data.X.length)} training points are right.`),
        { phase: 'plane', acc: M.acc, n: M.data.X.length });
      P.narrow(F.list, (m) => (m.phase === 'plane' ? { x: 560, y: 40, w: 370, h: 350 } : { x: 20, y: 10, w: 540, h: 420 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The forward pass', text: ['hidden:  zⱼ = Σᵢ wᵢⱼ xᵢ + bⱼ,  hⱼ = f(zⱼ)', 'output:  sₖ = Σⱼ vⱼₖ hⱼ + cₖ', 'softmax: pₖ = e^sₖ / Σ e^s', 'two classes: p₁ = σ(s₁ − s₀)', 'in matrices: h = f(xW + b), s = hV + c'].join('\n') }] }, result: { W, B } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'output' && f.meta.k === 1, once: true, pick: 'option',
      ask: 'Which class will the network choose for this point?',
      options: ['class 0', 'class 1'],
      answer: (fr) => 'class ' + fr.find((f) => f.meta.phase === 'softmax').meta.win,
      explain: (fr) => fr.find((f) => f.meta.phase === 'softmax').say,
    }],
    quiz: [
      { q: 'A hidden layer has 3 neurons and the input has 2 numbers. How many weights (not biases) join them?', options: ['6', '5', '3', '9'], answer: 0, why: 'Every input is wired to every neuron: 2 × 3.' },
      { q: 'What does a wire into a neuron carry in the forward pass?', options: ['the value at its start times its weight', 'the weight alone', 'the gradient', 'the neuron’s bias'], answer: 0, why: 'The neuron then adds these products and its bias.' },
      { q: 'Two output scores are 1.2 and 0.2. Softmax gives class 0 a probability of about…', options: ['0.73', '0.5', '0.86', '0.27'], answer: 0, why: 'Only the gap of 1 matters: σ(1) ≈ 0.73.' },
      { q: 'Without any activation function, a network of many layers is…', options: ['still a linear function of its input', 'more powerful than with them', 'unable to learn', 'a decision tree'], answer: 0, why: 'A product of matrices is one matrix: the layers collapse into one linear map.' },
      { q: 'Adding the same number to both output scores changes the probabilities by…', options: ['nothing', 'a lot', 'only for class 0', 'it doubles them'], answer: 0, why: 'Softmax depends only on the differences between scores.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
