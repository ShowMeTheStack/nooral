// Part 9, backpropagation on the computational graph: a one-neuron network with an output weight and a squared
// error, built from the engine's scalar ops so every node is drawn. The forward pass fills in each node's value; the
// backward pass, from ∂L/∂L = 1, gives every node its gradient by the chain rule (the local derivative times the
// gradient arriving from above). Every parameter's gradient is then checked against finite differences, and one
// step of gradient descent lowers the loss.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml;
  const NODES = {
    x1: { at: [60, 50], label: 'x₁', kind: 'input' }, w1: { at: [60, 145], label: 'w₁', kind: 'param' },
    x2: { at: [60, 255], label: 'x₂', kind: 'input' }, w2: { at: [60, 350], label: 'w₂', kind: 'param' },
    m1: { at: [184, 97], label: 'm₁ = w₁·x₁', rule: 'mul' }, m2: { at: [184, 302], label: 'm₂ = w₂·x₂', rule: 'mul' },
    s: { at: [308, 200], label: 's = m₁ + m₂', rule: 'add' }, b: { at: [308, 330], label: 'b', kind: 'param' },
    u: { at: [432, 250], label: 'u = s + b', rule: 'add' }, h: { at: [556, 250], label: 'h = f(u)', rule: 'act' },
    v: { at: [556, 110], label: 'v', kind: 'param' }, m3: { at: [680, 180], label: 'm₃ = v·h', rule: 'mul' },
    c: { at: [680, 330], label: 'c', kind: 'param' }, y: { at: [804, 180], label: 'y = m₃ + c', rule: 'add' },
    t: { at: [804, 330], label: 't (target)', kind: 'input' }, d: { at: [928, 180], label: 'd = y − t', rule: 'sub' },
    L: { at: [928, 330], label: 'L = d²', rule: 'square' },
  };
  const PARAMS = ['w1', 'w2', 'b', 'v', 'c'];
  const BW = 112, BH = 68;
  /** the graph in the engine (float64): returns the named tensors and the loss */
  P.bpBuild = function (vals, act) {
    const T = S.T, n = {};
    for (const k of ['x1', 'x2', 't']) n[k] = T.tensor(vals[k]).named(k);
    for (const k of PARAMS) n[k] = T.param(T.tensor(vals[k]), k);
    n.m1 = T.mul(n.w1, n.x1).named('m1'); n.m2 = T.mul(n.w2, n.x2).named('m2');
    n.s = T.add(n.m1, n.m2).named('s'); n.u = T.add(n.s, n.b).named('u');
    n.h = T[act](n.u).named('h'); n.m3 = T.mul(n.v, n.h).named('m3');
    n.y = T.add(n.m3, n.c).named('y'); n.d = T.sub(n.y, n.t).named('d'); n.L = T.square(n.d).named('L');
    return n;
  };
  /** starting weights from a seed, rounded to one decimal so the sums can be followed by eye */
  P.bpWeights = (seed) => { const g = S.T.rng(seed); return Object.fromEntries(PARAMS.map((k) => [k, Math.round(g.normal(0, k === 'v' ? 1 : 0.8) * 10) / 10 || 0.1])); };

  S.topic({
    id: 'p09-backprop', family: 'p09-mlp', order: 20, title: 'Backpropagation on the computational graph',
    blurb: 'Training needs the slope of the loss with respect to every weight. Write the network as a graph of small operations, run it forward to get every value, then walk it backwards from the loss: each node passes back the gradient it received times its own local derivative. Every gradient is shown on its node, and each weight’s is checked against a finite difference.',
    inputs: [
      { id: 'x1', kind: 'slider', label: 'Input x₁', def: 0.5, min: -2, max: 2, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'x2', kind: 'slider', label: 'Input x₂', def: -1, min: -2, max: 2, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 't', kind: 'slider', label: 'Target t', def: 1, min: -2, max: 2, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'act', kind: 'choice', label: 'Activation f', def: 'tanh', options: P.actOptions(['tanh', 'sigmoid', 'relu']) },
      { id: 'seed', kind: 'number', label: 'Seed for the weights', def: 4, min: 1, max: 999 },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.1, min: 0.01, max: 1, step: 0.01, fmt: (v) => H().fx(v, 2) },
    ],
    presets: [
      { label: 'tanh neuron', values: { x1: 0.5, x2: -1, t: 1, act: 'tanh', seed: 4, lr: 0.1 } },
      { label: 'Sigmoid neuron', values: { x1: 1.5, x2: 0.5, t: -0.5, act: 'sigmoid', seed: 7, lr: 0.3 } },
      { label: 'ReLU that is off', values: { x1: 1, x2: 1, t: 1, act: 'relu', seed: 2, lr: 0.1 } },
      { label: 'ReLU that is on', values: { x1: 1, x2: -1, t: 2, act: 'relu', seed: 11, lr: 0.1 } },
      { label: 'Too large a step', values: { x1: 0.5, x2: -1, t: 1, act: 'tanh', seed: 4, lr: 1 } },
    ],
    counters: [{ id: 'fwd', label: 'values computed' }, { id: 'bwd', label: 'gradients computed' }],
    run(inp) {
      const T = S.T, F = H().frames(['fwd', 'bwd']), q = F.q, r2 = P.r2, A = P.ACTS[inp.act];
      const w0 = P.bpWeights(inp.seed), vals = Object.assign({ x1: inp.x1, x2: inp.x2, t: inp.t }, w0);
      return T.float64(() => {
        const n = P.bpBuild(vals, inp.act);
        n.L.backward();
        const G = T.graph(n.L), byName = Object.fromEntries(G.nodes.map((g) => [g.name, g]));
        const val = (k) => n[k].item(), grad = (k) => (n[k].grad ? n[k].grad.item() : null);
        const order = G.nodes.filter((g) => !g.leaf).map((g) => g.name);
        const fmtv = (v) => H().fx(v, 3);
        /** the graph with what is known: shown (values), grads (names with gradients), lit, back (node just given its gradient) */
        const draw = (shown, grads, lit, back) => {
          const items = [];
          for (const [a, b] of G.edges) {
            const A2 = byName[G.nodes.find((g) => g.id === a).name], B2 = byName[G.nodes.find((g) => g.id === b).name];
            const ka = A2.name, kb = B2.name, hot = lit === kb || (back && back.includes(ka) && lit === kb);
            items.push({ key: 'bp:e' + ka + '_' + kb, kind: 'edge', from: 'bp:' + ka, to: 'bp:' + kb, head: true, cls: 'p09wire ' + (hot ? 'act' : '') });
          }
          for (const [k, N] of Object.entries(NODES)) {
            const known = shown.has(k), leaf = !!N.kind, cls = ['p09g', leaf ? 'leaf' : '', N.kind === 'param' ? 'param' : '', lit === k ? 'act' : '', back && back.includes(k) ? 'back' : '', known ? '' : 'wait'].join(' ');
            const gl = N.kind === 'input' ? 'data' : grads.has(k) && grad(k) !== null ? '∂ ' + fmtv(grad(k)) : '∂ ?';
            items.push({ key: 'bp:' + k, kind: 'cell', x: N.at[0] - BW / 2, y: N.at[1] - BH / 2, w: BW, h: BH, rx: 7, lines: [N.label, known ? fmtv(val(k)) : '?', gl], size: 15, cls: cls + (grads.has(k) ? ' hasg' : '') });
          }
          return items;
        };
        const shown = new Set(['x1', 'x2', 't', ...PARAMS]), grads = new Set();
        const ruleTxt = (k) => {
          const g = byName[k], ps = n[k].ctx.parents.map((p) => p.name);
          return { ps, op: g.op };
        };
        F.push(draw(shown, grads, null, null), `The network as a graph of small operations: one neuron h = ${A.label}(w₁x₁ + w₂x₂ + b), an output y = v·h + c, and the squared error L = (y − t)². Inputs x₁ = ${q(vals.x1, 1)}, x₂ = ${q(vals.x2, 1)}, target t = ${q(vals.t, 1)}; the weights start at w₁ = ${q(vals.w1, 1)}, w₂ = ${q(vals.w2, 1)}, b = ${q(vals.b, 1)}, v = ${q(vals.v, 1)}, c = ${q(vals.c, 1)}.`, { phase: 'start', vals }, { hl: { eq: [0] } });
        // forward, node by node
        for (const k of order) {
          shown.add(k); F.count('fwd');
          const { ps } = ruleTxt(k), a = val(ps[0]), b = ps[1] ? val(ps[1]) : null, v = val(k);
          const sym = NODES[k].label.split(' = ')[0];
          const how = NODES[k].rule === 'mul' ? `${q(a, 3)} × ${q(b, 3)}` : NODES[k].rule === 'add' ? `${q(a, 3)} + ${q(b, 3)}` : NODES[k].rule === 'sub' ? `${q(a, 3)} − ${q(b, 3)}` : NODES[k].rule === 'square' ? (a < 0 ? `(${q(a, 3)})²` : `${q(a, 3)}²`) : `${A.label}(${q(a, 3)})`;
          F.push(draw(shown, grads, k, null), `Forward: ${sym} = ${how} = ${q(v, 3)}.` + (k === 'L' ? ' That is the loss: how far the output y is from the target, squared.' : k === 'h' && inp.act === 'relu' && v === 0 ? ' The ReLU is off: u is negative, so h is zero.' : ''), { phase: 'fwd', k, v, a, b }, { hl: { eq: [0] } });
        }
        // backward, from the loss
        grads.add('L'); F.count('bwd');
        F.push(draw(shown, grads, 'L', null), `Backward. Start at the loss with ∂L/∂L = ${q(grad('L'), 0)}. Each node will pass back the gradient it holds times its local derivative, the slope of its own little operation.`, { phase: 'seed', g: grad('L') }, { hl: { eq: [1] } });
        for (const k of order.slice().reverse()) {
          const { ps } = ruleTxt(k), g = grad(k), r = NODES[k].rule, trainable = ps.filter((p) => n[p].requiresGrad);
          trainable.forEach((p) => grads.add(p));
          F.count('bwd', trainable.length);
          const name = (p) => NODES[p].label.split(' = ')[0].replace(' (target)', '');
          let say;
          const local = {};
          if (r === 'square') { local[ps[0]] = 2 * val(ps[0]); say = `L = d²: the local derivative is 2d = ${q(local[ps[0]], 3)}, so ∂L/∂d = ${q(local[ps[0]], 3)} × ${q(g, 3)} = ${q(grad(ps[0]), 3)}.`; }
          else if (r === 'sub') { local[ps[0]] = 1; say = `d = y − t passes its gradient to y unchanged: ∂L/∂y = ${q(grad('y'), 3)}. (The target t is data, not a weight, so it needs no gradient.)`; }
          else if (r === 'add') { trainable.forEach((p) => { local[p] = 1; }); say = `${name(k)} = ${name(ps[0])} + ${name(ps[1])}: an addition passes the gradient ${q(g, 3)} to both of its inputs unchanged, so ` + trainable.map((p) => `∂L/∂${name(p)} = ${q(grad(p), 3)}`).join(' and ') + '.'; }
          else if (r === 'mul') { const [a, b] = ps; if (n[a].requiresGrad) local[a] = val(b); if (n[b].requiresGrad) local[b] = val(a); say = `${name(k)} = ${name(a)}·${name(b)}: a product sends each input the other input times the gradient. ` + trainable.map((p) => `∂L/∂${name(p)} = ${q(local[p], 3)} × ${q(g, 3)} = ${q(grad(p), 3)}`).join(', and ') + '.' + (trainable.length < 2 ? ' The input itself is data and gets no gradient.' : ''); }
          else { const hv = val('h'), uv = val('u'); local.u = inp.act === 'tanh' ? 1 - hv * hv : inp.act === 'sigmoid' ? hv * (1 - hv) : uv > 0 ? 1 : 0; say = `h = ${A.label}(u): the local derivative is ${inp.act === 'tanh' ? '1 − h²' : inp.act === 'sigmoid' ? 'h(1 − h)' : 'the slope of ReLU, one if u > 0 else zero,'} = ${q(local.u, 3)}, so ∂L/∂u = ${q(local.u, 3)} × ${q(g, 3)} = ${q(grad('u'), 3)}.` + (local.u === 0 ? ' Nothing gets through: no weight below this neuron can learn from this example.' : Math.abs(local.u) < 0.05 ? ' The neuron is saturated, so very little gets through.' : ''); }
          F.push(draw(shown, grads, k, trainable), say, { phase: 'bwd', k, g, local, grads: Object.fromEntries(trainable.map((p) => [p, grad(p)])) }, { hl: { eq: [1, r === 'mul' ? 2 : r === 'add' ? 3 : r === 'act' ? 4 : r === 'square' ? 5 : 3] } });
        }
        // the check: finite differences of the loss in each parameter, through the same engine graph
        const eps = 1e-6, lossAt = (over) => T.noGrad(() => P.bpBuild(Object.assign({}, vals, over), inp.act).L.item());
        const fd = PARAMS.map((k) => (lossAt({ [k]: vals[k] + eps }) - lossAt({ [k]: vals[k] - eps })) / (2 * eps));
        const ag = PARAMS.map((k) => grad(k)), worst = Math.max(...ag.map((g, i) => Math.abs(g - fd[i]) / Math.max(1e-8, Math.abs(g) + Math.abs(fd[i]))));
        const table = S.Kit.dt.table({ prefix: 'bp:tab', x: 280, y: 420, cols: [{ name: 'weight', w: 80 }, { name: 'backprop', w: 130 }, { name: 'finite difference', w: 170 }], rows: PARAMS.map((k, i) => [NODES[k].label, H().fx(ag[i], 5), H().fx(fd[i], 5)]), rowH: 27, headH: 29, size: 15 }).items;
        F.push([...draw(shown, grads, null, null), ...table], `Every weight now has its gradient. To check them, nudge each weight by ±${q.sg(eps, 1)} and see how the loss moves: (L(w + ε) − L(w − ε)) / 2ε. ` + PARAMS.map((k, i) => `${NODES[k].label}: ${q(ag[i], 4)} against ${q(fd[i], 4)}`).join('; ') + `. The worst relative difference is ${q.sg(worst, 2)}.`, { phase: 'check', ag, fd, worst, eps }, { hl: { eq: [6] } });
        // one step of gradient descent
        const next = Object.fromEntries(PARAMS.map((k) => [k, vals[k] - inp.lr * grad(k)])), L1 = lossAt(next), L0 = val('L');
        const big = PARAMS.map((k) => [Math.abs(grad(k)), k]).sort((a, b) => b[0] - a[0])[0][1];
        F.push([...draw(shown, grads, null, null), ...table], `One step of gradient descent moves every weight against its gradient: w ← w − η · ∂L/∂w with η = ${q(inp.lr, 2)}. The biggest move is ${NODES[big].label}’s, from ${q(vals[big], 3)} to ${q(next[big], 3)}. Running forward again, the loss goes from ${q(L0, 4)} to ${q(L1, 4)}` + (L1 < L0 ? '.' : ': higher, because the step was too large for this slope.'), { phase: 'step', L0, L1, big, from: vals[big], to: next[big], next }, { hl: { eq: [7] } });
        P.narrow(F.list, (m) => { if (m.phase === 'check' || m.phase === 'step') return { x: 260, y: 400, w: 420, h: 200 }; const cx = m.k ? NODES[m.k].at[0] : 180; return { x: Math.max(0, Math.min(620, cx - 190)), y: 0, w: 380, h: 410 }; });
        return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The chain rule on a graph', text: ['forward: each node computes its value from its inputs', 'backward: ∂L/∂input = local derivative × ∂L/∂node', 'product a·b: ∂/∂a = b, ∂/∂b = a', 'sum a + b: passes the gradient to both', 'activation: f′(u) (tanh: 1 − h², sigmoid: h(1 − h))', 'square d²: 2d', 'check: (L(w + ε) − L(w − ε)) / 2ε', 'update: w ← w − η ∂L/∂w'].join('\n') }] }, result: { vals } };
      });
    },
    predict: [{
      at: (f) => f.meta.phase === 'seed', once: true, pick: 'option',
      ask: 'Which weight will get the largest gradient (in size)?',
      options: ['w₁', 'w₂', 'b', 'v', 'c'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'check').meta, k = m.ag.map((g, i) => [Math.abs(g), i]).sort((a, b) => b[0] - a[0])[0][1]; return ['w₁', 'w₂', 'b', 'v', 'c'][k]; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'check').say,
    }],
    quiz: [
      { q: 'In backpropagation, an addition node z = a + b sends back to a…', options: ['the gradient it received, unchanged', 'the gradient times b', 'zero', 'half the gradient'], answer: 0, why: '∂z/∂a = 1, so ∂L/∂a = ∂L/∂z.' },
      { q: 'A product node z = a·b sends back to a…', options: ['b times the gradient it received', 'a times the gradient', 'the gradient unchanged', 'a·b'], answer: 0, why: '∂z/∂a = b.' },
      { q: 'Why is backpropagation run from the loss backwards rather than from each weight forwards?', options: ['one backward pass gives the gradient of every weight at once', 'forwards would give the wrong answer', 'the loss has no value until the end', 'weights cannot be differentiated forwards'], answer: 0, why: 'Reverse mode costs about one forward pass for all the gradients of a single scalar loss.' },
      { q: 'A finite-difference check computes…', options: ['(L(w + ε) − L(w − ε)) / 2ε for each weight', 'the loss twice and takes the larger', 'the gradient of the input', 'a second backward pass'], answer: 0, why: 'A central difference approximates the slope; it should agree with backpropagation to many digits.' },
      { q: 'A ReLU neuron has u < 0 for an example. The weights feeding it get from that example…', options: ['zero gradient', 'the largest gradient', 'a gradient equal to the input', 'a random gradient'], answer: 0, why: 'ReLU’s slope is zero there, so the chain rule multiplies everything below by zero.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
