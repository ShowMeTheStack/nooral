// Part 9, the playground: choose a 2-D set, one or two hidden layers and their width, and train; every hidden neuron
// is drawn as a small map of its own activation over the input plane, joined to the others by its weights, so the
// regions each neuron carves out can be watched as training shapes them into the final boundary.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const MAPN = 22;
  P.playRun = (inp) => P.once('play:' + JSON.stringify(inp), () => {
    const set = P.SETS[inp.data], d = P.data(inp.data, 200, set.noise, 3), hidden = Array.from({ length: +inp.layers }, () => inp.neurons);
    const steps = inp.steps, snaps = [...new Set([0, Math.round(steps / 20), Math.round(steps / 8), Math.round(steps / 3), Math.round((2 * steps) / 3), steps])];
    const r = P.train({ train: d, hidden, act: inp.act, lr: inp.lr, steps, evalEvery: Math.max(1, Math.round(steps / 50)), seed: inp.seed, snaps, grid: { xr: set.xr, yr: set.yr, n: 48, w: 1, h: 1 },
      snapExtra: (net) => ({ maps: P.neuronMaps(net, set.xr, set.yr, MAPN), W: P.netWeights(net) }) });
    r.data = d; r.hidden = hidden;
    return r;
  });

  S.topic({
    id: 'p09-playground', family: 'p09-play', order: 10, title: 'Playground: every neuron’s region',
    blurb: 'Pick a set of points, a number of layers and neurons, and train. Each hidden neuron is drawn as a small map of the input plane, blue where it fires positive and vermilion where negative, so you can see the simple regions of the first layer, the combined shapes of the second, and how the output adds them up into the boundary on the right.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'circles', options: P.setOptions(['moons', 'circles', 'xor', 'spirals']) },
      { id: 'layers', kind: 'choice', label: 'Hidden layers', def: '1', options: [{ value: '1', label: 'one' }, { value: '2', label: 'two' }] },
      { id: 'neurons', kind: 'number', label: 'Neurons per layer', def: 4, min: 2, max: 6 },
      { id: 'act', kind: 'choice', label: 'Activation', def: 'tanh', options: P.actOptions(['tanh', 'relu']) },
      { id: 'lr', kind: 'slider', label: 'Learning rate', def: 0.03, min: 0.003, max: 0.1, step: 0.001, fmt: (v) => H().fx(v, 3) },
      { id: 'steps', kind: 'number', label: 'Training steps', def: 400, min: 50, max: 1000 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A ring: four tanh neurons', values: { data: 'circles', layers: '1', neurons: 4, act: 'tanh', lr: 0.03, steps: 400, seed: 1 } },
      { label: 'A ring with only two neurons', values: { data: 'circles', layers: '1', neurons: 2, act: 'tanh', lr: 0.03, steps: 400, seed: 1 } },
      { label: 'XOR with ReLU', values: { data: 'xor', layers: '1', neurons: 4, act: 'relu', lr: 0.03, steps: 400, seed: 2 } },
      { label: 'Moons, two layers', values: { data: 'moons', layers: '2', neurons: 5, act: 'tanh', lr: 0.03, steps: 500, seed: 1 } },
      { label: 'Spirals, two layers of six', values: { data: 'spirals', layers: '2', neurons: 6, act: 'relu', lr: 0.03, steps: 800, seed: 1 } },
    ],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const F = H().frames(['step']), q = F.q, set = P.SETS[inp.data], r = P.playRun({ data: inp.data, layers: inp.layers, neurons: inp.neurons, act: inp.act, lr: inp.lr, steps: inp.steps, seed: inp.seed });
      const d = r.data, sizes = [2].concat(r.hidden, [2]), nW = P.weightsOf(r.hidden), A = P.ACTS[inp.act];
      const half = sizes.length === 4 ? 26 : 30, box = { x: 50, y: 60, w: sizes.length === 4 ? 470 : 400, h: 360 };
      const names = [['x₁', 'x₂'], [], [], []];
      names[sizes.length - 1] = ['0', '1'];
      const ax = P.ax({ prefix: 'pg', x: 620, y: 70, w: 280, h: 280, xr: set.xr, yr: set.yr, xticks: 1, yticks: 1, xname: 'x₁', yname: 'x₂', frame: 'box' });
      const norm = (m) => { const mx = Math.max(1e-9, ...m.map(Math.abs)); return m.map((v) => v / mx); };
      // the most useful last-layer neuron: the largest gap between its two output weights
      const infl = (W) => { const out = W[W.length - 1]; return out[0].map((_, j) => Math.abs(out[1][j] - out[0][j])); };
      const titles = sizes.map((_, l) => (l === 0 ? 'input' : l === sizes.length - 1 ? 'output' : 'layer ' + l + ' (' + A.label + ')'));
      r.snaps.forEach((s, k) => {
        F.set('step', s.step);
        const W = s.extra.W, wmax = Math.max(1, ...W.flat(2).map(Math.abs)), inf = infl(W), top = inf.indexOf(Math.max(...inf));
        const mn = P.mapNet({ prefix: 'pn', x: box.x, y: box.y, w: box.w, h: box.h, sizes, half, r: 20, weights: W, wmax, wide: 5, n: MAPN, maps: s.extra.maps.map((L) => L.map(norm)), names, lit: (l, i) => (k > 0 && l === sizes.length - 2 && i === top ? 'act' : '') });
        const items = [...mn.items, ...titles.map((t, l) => H().text('pn:t' + l, mn.L.pos[l][0].x, 34, t, 'axname', 13, 'middle'))];
        items.push(...ax.items, ...P.boundary(ax, 'pg:b', s.field), ...S.Kit.scatter.points(ax, 'pg:p', d.X, d.y, { r: 3.2, only: d.X.map((_, i) => i).filter((i) => ax.inside(d.X[i][0], d.X[i][1])) }));
        items.push(H().text('pg:r', 620, 400, 'step ' + s.step + ':  loss ' + H().fx(s.trainLoss, 3) + ',  ' + H().pct(s.trainAcc) + ' right', 'strong', 14));
        const say = k === 0
          ? `A network with ${r.hidden.length === 1 ? 'one hidden layer' : 'two hidden layers'} of ${P.qpl(q, inp.neurons, 'neuron')} (${A.label}), ${P.qpl(q, nW, 'weight')} in all, on ${q.n(d.X.length)} points. Each square is one hidden neuron over the input plane: blue where it is positive, vermilion where negative. Before training each first-layer neuron is a soft step across a random line; loss ${q(s.trainLoss, 3)}.`
          : `Step ${q.n(s.step)}: loss ${q(s.trainLoss, 3)}, ${q.pct(s.trainAcc)} of the points right. The ringed neuron, h${P.sub(top + 1)} of the last hidden layer, now matters most to the output (its two output weights differ by ${q(inf[top], 2)}).` + (k === r.snaps.length - 1 ? '' : r.hidden.length === 2 && k === 2 ? ' The second layer’s squares are no longer straight steps: each mixes the first layer’s regions.' : '');
        F.push(items, say, { phase: 'snap', step: s.step, loss: s.trainLoss, acc: s.trainAcc, top, inf: inf[top], nW, n: d.X.length, neurons: inp.neurons }, { hl: { eq: [k === 0 ? 0 : 1] } });
      });
      const last = r.snaps[r.snaps.length - 1];
      const good = last.trainAcc >= 0.95;
      F.push(F.list[F.list.length - 1].items, (good ? `Done: ${q.pct(last.trainAcc)} right. ` : `After ${q.n(last.step)} steps only ${q.pct(last.trainAcc)} are right. `) + (r.hidden.length === 1 ? `With one hidden layer the output can only add up the ${q.n(inp.neurons)} regions, each bounded by a straight line, so the boundary is built from at most that many straight pieces bent smooth.` : `With two layers the second combines the first layer’s regions before the output adds them: shapes made of shapes.`) + (good ? '' : ' Try more neurons, a second layer, more steps or another learning rate.'),
        { phase: 'end', acc: last.trainAcc, loss: last.trainLoss, step: last.step, neurons: inp.neurons }, { hl: { eq: [2] } });
      P.narrow(F.list, (m) => (m.phase === 'end' ? { x: 560, y: 40, w: 370, h: 380 } : { x: 20, y: 10, w: box.w + 70, h: 430 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'What each square shows', text: ['first layer: hⱼ = f(w₁ⱼx₁ + w₂ⱼx₂ + bⱼ), a soft step across a line', 'later layers: f(weighted sum of the squares before)', 'output: scores from the last squares, softmax to p', 'boundary: where p = ½'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'snap' && f.meta.step === 0, once: true, pick: 'option',
      ask: 'Will this network get at least 95% of the points right by the end?',
      options: ['yes', 'no'],
      answer: (fr) => (fr.find((f) => f.meta.phase === 'end').meta.acc >= 0.95 ? 'yes' : 'no'),
      explain: (fr) => fr.find((f) => f.meta.phase === 'end').say,
    }],
    quiz: [
      { q: 'A first-layer neuron of a network on 2-D inputs divides the plane…', options: ['with a straight line, softly', 'with a circle', 'into a checkerboard', 'not at all'], answer: 0, why: 'Its input is w₁x₁ + w₂x₂ + b, which is constant along lines.' },
      { q: 'Why can two tanh neurons in one hidden layer not enclose a ring of points?', options: ['two soft lines cannot surround a region', 'tanh is too slow', 'they need ReLU', 'rings need more data'], answer: 0, why: 'Enclosing a disc needs at least three lines; more make it rounder.' },
      { q: 'What does a second hidden layer add?', options: ['each neuron can combine the regions of the first layer into new shapes', 'more inputs', 'a higher learning rate', 'nothing that one wide layer cannot do as easily'], answer: 0, why: 'Depth composes simple pieces into complex ones, often with fewer neurons.' },
      { q: 'The spirals need many more steps than the circles because…', options: ['the boundary winds round several times, needing many regions placed just right', 'they have more points', 'they have three classes', 'tanh cannot fit them'], answer: 0, why: 'A complex boundary takes many neurons and long training to arrange.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
