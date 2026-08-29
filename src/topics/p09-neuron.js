// Part 9, one neuron: each input times its weight, the products added with the bias into z, z squashed by the
// activation into the output a, with the slope there (from autodiff); then the same neuron over the whole input
// plane, where z = 0 is a straight line.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const slider = (id, label, def, min, max) => ({ id, kind: 'slider', label, def, min, max, step: 0.1, fmt: (v) => H().fx(v, 1) });

  S.topic({
    id: 'p09-neuron', family: 'p09-neuron', order: 10, title: 'One neuron: a weighted sum and a squash',
    blurb: 'A neuron multiplies each input by a weight, adds the products and a bias into one number z, and passes z through an activation function. Follow the numbers through one neuron, read its output and its slope off the activation curve, then see what the same neuron does at every point of the input plane: it draws a straight line and says how far each point lies on one side of it.',
    inputs: [
      slider('x1', 'Input x₁', 0.5, -2, 2), slider('x2', 'Input x₂', -1, -2, 2),
      slider('w1', 'Weight w₁', 1.2, -3, 3), slider('w2', 'Weight w₂', -0.8, -3, 3), slider('b', 'Bias b', 0.3, -3, 3),
      { id: 'act', kind: 'choice', label: 'Activation', def: 'sigmoid', options: P.actOptions() },
    ],
    presets: [
      { label: 'Sigmoid neuron', values: { x1: 0.5, x2: -1, w1: 1.2, w2: -0.8, b: 0.3, act: 'sigmoid' } },
      { label: 'Both inputs needed (AND-like)', values: { x1: 1, x2: 1, w1: 2, w2: 2, b: -3, act: 'sigmoid' } },
      { label: 'Saturated tanh', values: { x1: 2, x2: 1.5, w1: 2.5, w2: 1.5, b: 0.5, act: 'tanh' } },
      { label: 'ReLU switched off', values: { x1: -1, x2: 0.5, w1: 1.5, w2: -1, b: 0, act: 'relu' } },
      { label: 'GELU just below zero', values: { x1: -0.6, x2: 0.2, w1: 1, w2: 0.5, b: 0, act: 'gelu' } },
    ],
    counters: [{ id: 'mul', label: 'multiplications' }, { id: 'add', label: 'additions' }],
    run(inp) {
      const T = S.T, A = P.ACTS[inp.act], F = H().frames(['mul', 'add']), q = F.q, r2 = P.r2;
      const x = [inp.x1, inp.x2], w = [inp.w1, inp.w2], b = inp.b;
      // the neuron in the engine: z = x·W + b, a = f(z); backpropagation from a gives the slope f′(z) and ∂a/∂w
      const xt = T.tensor([x]), W = T.param(T.tensor([[w[0]], [w[1]]])), bt = T.param(T.tensor([b]));
      const zt = T.linear(xt, W, bt), at = T[inp.act](zt);
      at.backward();
      const z = zt.item(), a = at.item(), slope = zt.grad.item(), dw = Array.from(W.grad.values()), prod = [x[0] * w[0], x[1] * w[1]];
      const curve = P.actCurve(inp.act, T.linspace(-5, 5, 161).values());
      const zs = Array.from(T.linspace(-5, 5, 161).values());
      // the picture: inputs, the sum, the activation, the output; the curve on the right
      const N = { x1: [60, 110], x2: [60, 260], one: [60, 400], sum: [300, 250], f: [430, 250], out: [560, 250] };
      const node = (key, p, text, sub, cls) => ({ key, kind: 'state', x: p[0], y: p[1], r: 30, text, sub, size: 14, cls: 'neuron ' + (cls || '') });
      const chipAt = (k, from, t, text, cls) => ({ key: 'nn:c' + k, kind: 'chip', x: r2(from[0] + (N.sum[0] - from[0]) * t), y: r2(from[1] + (N.sum[1] - from[1]) * t), text, size: 13, cls: 'p09w ' + (cls || ''), layer: 'top' });
      const ax = P.ax({ prefix: 'ac', x: 660, y: 70, w: 250, h: 260, xr: [-5, 5], yr: A.range, xticks: [-4, -2, 0, 2, 4], yticks: inp.act === 'relu' || inp.act === 'gelu' ? [0, 1, 2, 3, 4, 5] : inp.act === 'tanh' ? [-1, -0.5, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1], xname: 'z', yname: A.label + '(z)' });
      const curveItems = (mark) => {
        const out = [...ax.items, C().line(ax, 'ac:f', zs.map((u, i) => [u, curve.f[i]]), 'act')];
        if (mark) {
          const zc = Math.max(-5, Math.min(5, z));
          out.push({ key: 'ac:gz', kind: 'path', d: `M${ax.X(zc)},${ax.Y(A.range[0])} V${ax.Y(a)} H${ax.X(-5)}`, cls: 'guide p09guide', audit: 'ignore', layer: 'back', z: 3 });
          const L = 1.2, y1 = a - slope * L, y2 = a + slope * L;
          out.push({ key: 'ac:tan', kind: 'path', d: `M${ax.X(zc - L)},${ax.Y(Math.max(A.range[0], Math.min(A.range[1], y1)))} L${ax.X(zc + L)},${ax.Y(Math.max(A.range[0], Math.min(A.range[1], y2)))}`, cls: 'p09tan', pathAudit: 'edge-x', layer: 'lines' });
          out.push(C().dot(ax, 'ac:pt', zc, a, 'p09hot', 6));
          out.push(H().text('ac:read', 660, 400, 'z = ' + H().fx(z) + '   a = ' + H().fx(a, 3), 'math', 14));
          out.push(H().text('ac:slope', 660, 424, 'slope f′(z) = ' + H().fx(slope, 3), 'math', 14));
        }
        return out;
      };
      const diagram = (st) => {
        const items = [];
        items.push(node('nn:x1', N.x1, H().fx(x[0], 1), 'x₁', 'input'), node('nn:x2', N.x2, H().fx(x[1], 1), 'x₂', 'input'), node('nn:one', N.one, '1', 'bias input', 'input'));
        items.push({ key: 'nn:sum', kind: 'state', x: N.sum[0], y: N.sum[1], r: 34, text: 'Σ', size: 22, cls: 'neuron ' + (st === 3 ? 'act' : '') });
        if (st >= 3) items.push(H().text('nn:z', N.sum[0], N.sum[1] + 60, 'z = ' + H().fx(z), 'math', 15, 'middle'));
        items.push({ key: 'nn:f', kind: 'state', x: N.f[0], y: N.f[1], w: 76, h: 46, rx: 8, text: A.label, size: 14, cls: 'neuron p09fn ' + (st === 4 ? 'act' : '') });
        items.push({ key: 'nn:out', kind: 'state', x: N.out[0], y: N.out[1], r: 30, text: st >= 4 ? H().fx(a, 2) : '?', sub: 'a', size: 14, cls: 'neuron ' + (st >= 4 ? 'act' : '') });
        for (const [k, p] of [['x1', N.x1], ['x2', N.x2], ['one', N.one]]) items.push({ key: 'nn:e' + k, kind: 'edge', from: 'nn:' + k, to: 'nn:sum', head: true, cls: 'p09wire ' + ((st === 1 && k === 'x1') || (st === 2 && k === 'x2') ? 'act' : '') });
        items.push({ key: 'nn:ef', kind: 'edge', from: 'nn:sum', to: 'nn:f', head: true, cls: 'p09wire' }, { key: 'nn:eo', kind: 'edge', from: 'nn:f', to: 'nn:out', head: true, cls: 'p09wire' });
        const wl = (k) => (st > k ? 'w' + P.sub(k) + 'x' + P.sub(k) + ' = ' + H().fx(prod[k - 1]) : 'w' + P.sub(k) + ' = ' + H().fx(w[k - 1], 1));
        items.push(chipAt('w1', N.x1, 0.45, wl(1), st === 1 ? 'act' : ''), chipAt('w2', N.x2, 0.45, wl(2), st === 2 ? 'act' : ''), chipAt('b', N.one, 0.45, 'b = ' + H().fx(b, 1), st === 3 ? 'act' : ''));
        return items;
      };
      const hl = (lines) => ({ eq: lines });
      F.push([...diagram(0), ...curveItems(false)],
        `A neuron with inputs x₁ = ${q(x[0], 1)} and x₂ = ${q(x[1], 1)}, weights w₁ = ${q(w[0], 1)} and w₂ = ${q(w[1], 1)}, and bias b = ${q(b, 1)}. It adds up the weighted inputs and the bias into z, then passes z through the ${A.label} curve on the right.`, { phase: 'start', x, w, b }, { hl: hl([0]) });
      F.count('mul');
      F.push([...diagram(1), ...curveItems(false)], `The first input times its weight: w₁ · x₁ = ${q(w[0], 1)} × ${q(x[0], 1)} = ${q(prod[0])}.` + (Math.abs(prod[0]) < 1e-9 ? ' A zero input passes nothing on, whatever its weight.' : prod[0] > 0 ? ' It pushes z up.' : ' It pulls z down.'), { phase: 'mul', k: 1, p: prod[0], w: w[0], x: x[0] }, { hl: hl([0]) });
      F.count('mul');
      F.push([...diagram(2), ...curveItems(false)], `The second: w₂ · x₂ = ${q(w[1], 1)} × ${q(x[1], 1)} = ${q(prod[1])}.` + (Math.abs(prod[1]) < 1e-9 ? ' Nothing passes on.' : prod[1] > 0 ? ' It pushes z up.' : ' It pulls z down.'), { phase: 'mul', k: 2, p: prod[1], w: w[1], x: x[1] }, { hl: hl([0]) });
      F.count('add', 2);
      F.push([...diagram(3), ...curveItems(false)], `Adding the products and the bias: z = ${q.t(prod[0], 2, true)}${q.t(prod[1], 2)}${q.t(b, 1)} = ${q(z)}. The bias shifts the sum whatever the inputs are.`, { phase: 'sum', z, terms: [prod[0], prod[1], b] }, { hl: hl([0]) });
      const what = inp.act === 'relu' ? (z <= 0 ? 'ReLU passes nothing below zero: the neuron is off, and its slope is zero, so no gradient will flow back through it for this input.' : 'ReLU passes a positive z unchanged, with slope one.')
        : Math.abs(slope) < 0.05 ? 'The curve is almost flat here: the neuron is saturated, so changing the weights a little hardly changes a.' : 'On the steep part of the curve, a small change in z moves a noticeably.';
      F.push([...diagram(4), ...curveItems(true)], `${A.label}(${q(z)}) = ${q(a, 3)}, the neuron’s output. The slope of the curve there is f′(z) = ${q(slope, 3)}, from backpropagation. ${what}`, { phase: 'act', z, a, slope }, { hl: hl([1, 2]) });
      F.push([...diagram(4), ...curveItems(true)], `The slope tells training how to move the weights: ∂a/∂w₁ = f′(z) · x₁ = ${q(dw[0], 3)} and ∂a/∂w₂ = f′(z) · x₂ = ${q(dw[1], 3)}. A weight whose input is large gets a large push; a weight on a saturated or switched-off neuron gets almost none.`, { phase: 'grad', dw, slope }, { hl: hl([3]) });
      // the whole plane
      const pax = P.ax({ prefix: 'pl', x: 660, y: 70, w: 250, h: 250, xr: [-2.5, 2.5], yr: [-2.5, 2.5], xticks: [-2, -1, 0, 1, 2], yticks: [-2, -1, 0, 1, 2], xname: 'x₁', yname: 'x₂' });
      const fieldOf = (u, v) => { const zz = w[0] * u + w[1] * v + b; return inp.act === 'sigmoid' ? 1 / (1 + Math.exp(-zz)) : inp.act === 'tanh' ? Math.tanh(zz) : zz; };
      const scheme = inp.act === 'sigmoid' ? 'prob' : 'div', max = inp.act === 'tanh' ? 1 : 3;
      const field = S.Kit.scatter.field(pax, 'pl:f', inp.act === 'sigmoid' ? fieldOf : (u, v) => (inp.act === 'tanh' ? fieldOf(u, v) : Math.max(-0.4, fieldOf(u, v))), { n: 60, scheme, max });
      const norm = Math.hypot(w[0], w[1]), dist = norm > 0 ? Math.abs(b) / norm : 0;
      const plane = [...pax.items, field];
      if (norm > 1e-9) plane.push(S.Kit.scatter.level(pax, 'pl:z', S.Kit.scatter.grid(pax, (u, v) => w[0] * u + w[1] * v + b, 40), 0, 'p09zero', { minPts: 1 }));
      plane.push(C().dot(pax, 'pl:pt', Math.max(-2.5, Math.min(2.5, x[0])), Math.max(-2.5, Math.min(2.5, x[1])), 'p09hot', 6));
      if (norm > 1e-9) { const s = 1.4 / norm; plane.push(H().arrow('pl:w', [pax.X(0), pax.Y(0)], [pax.X(w[0] * s), pax.Y(w[1] * s)], 'p09warrow')); }
      plane.push(H().text('pl:k', 660, 380, 'the line: z = 0;  arrow: (w₁, w₂)', 'legend', 13));
      plane.push(H().text('pl:k2', 660, 402, inp.act === 'sigmoid' ? 'shading: a, from 0 to 1' : 'blue: z > 0;  vermilion: z < 0', 'legend', 13));
      const side = z > 0 ? 'on the side the arrow points to' : z < 0 ? 'on the other side' : 'exactly on the line';
      F.push([...diagram(4), ...plane], norm > 1e-9
        ? `The same neuron at every point (x₁, x₂). Where z = 0 is a straight line, at right angles to the weight vector (w₁, w₂) and ${Math.abs(b) < 1e-9 ? 'through the origin, since there is no bias' : q(dist) + ' from the origin (|b| / ‖w‖)'}; a rises across it in the direction of the arrow. The marked input lies ${side}. One neuron can only split the plane with a line; a network adds many such lines up.`
        : 'With both weights zero the neuron ignores its inputs: z is the bias everywhere, and the plane has one colour.', { phase: 'plane', dist, norm, z }, { hl: hl([0, 4]) });
      P.narrow(F.list, (m) => (m.phase === 'act' || m.phase === 'grad' ? { x: 590, y: 30, w: 350, h: 420 } : m.phase === 'plane' ? { x: 600, y: 30, w: 340, h: 400 } : { x: 20, y: 40, w: 610, h: 420 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'One neuron', text: ['z = w₁x₁ + w₂x₂ + b', 'a = f(z)', 'f′(z): the slope of f at z', '∂a/∂wᵢ = f′(z) · xᵢ', 'z = 0 is a line, normal to w'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'mul' && f.meta.k === 2, once: true, pick: 'option',
      ask: 'Once the bias is added, will z be positive or negative?',
      options: ['positive', 'negative', 'zero'],
      answer: (fr) => { const z = fr.find((f) => f.meta.phase === 'sum').meta.z; return Math.abs(z) < 0.005 ? 'zero' : z > 0 ? 'positive' : 'negative'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'sum').say,
    }],
    quiz: [
      { q: 'A neuron computes z = w₁x₁ + w₂x₂ + b. What does the bias do?', options: ['shifts z by the same amount for every input', 'scales the inputs', 'chooses the activation', 'sets the learning rate'], answer: 0, why: 'b is added whatever the inputs are, so it moves the line z = 0 away from the origin.' },
      { q: 'Over the plane of its two inputs, the set where z = 0 is…', options: ['a straight line', 'a circle', 'a single point', 'a curve that depends on the activation'], answer: 0, why: 'w₁x₁ + w₂x₂ + b = 0 is the equation of a line; the activation only changes how a varies across it.' },
      { q: 'A sigmoid neuron has z = 8. Its slope f′(z) is…', options: ['nearly zero: it is saturated', 'about 1', 'exactly 0.5', 'negative'], answer: 0, why: 'The sigmoid is flat far from zero; f′(8) ≈ 0.0003.' },
      { q: 'A ReLU neuron has z = −1.5. How much do its weights learn from this input?', options: ['nothing: the slope is zero', 'a lot', 'only the bias learns', 'half as much as usual'], answer: 0, why: 'For z < 0 ReLU outputs zero with zero slope, so ∂a/∂w = f′(z)·x = 0.' },
      { q: '∂a/∂w₁ equals…', options: ['f′(z) · x₁', 'f′(z) · w₁', 'x₁ · w₁', 'f(z) · x₁'], answer: 0, why: 'z depends on w₁ through w₁x₁, so the chain rule gives f′(z) times x₁.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
