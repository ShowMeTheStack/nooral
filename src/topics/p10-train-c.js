// Part 10, residual connections: a deep stack of 3 × 3 convolutions (4 channels on an 8 × 8 map) built twice from the
// same random weights, plain (h ← ReLU(conv h)) and residual (h ← h + ReLU(conv h) / √L), run forward and backward
// live; the size of the activations and of the gradient reaching every layer, on a log scale.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const INITS = { small: { label: 'small (half of He)', words: 'small (half the He size)', s: 0.5 }, he: { label: 'He (the usual)', words: 'He-sized', s: 1 }, large: { label: 'large (1.5 × He)', words: 'large (one and a half times the He size)', s: 1.5 } };
  const rms = (t) => Math.sqrt(Array.from(t.values()).reduce((s, v) => s + v * v, 0) / t.size);

  /** the stack, run forward and backward: RMS of every layer's output and of the gradient there */
  P.stack = function (o) {
    const T = S.T, g = T.rng(o.seed), Ch = 4, N = 8, L = o.depth;
    const x = T.param(T.tensor(Array.from({ length: Ch * N * N }, () => g.normal()), [1, Ch, N, N]));
    const Ws = Array.from({ length: L }, () => T.tensor(Array.from({ length: Ch * Ch * 9 }, () => g.normal() * o.scale * Math.sqrt(2 / (Ch * 9))), [Ch, Ch, 3, 3]));
    const r = T.tensor(Array.from({ length: Ch * N * N }, () => g.normal()), [1, Ch, N, N]);
    const hs = [x];
    let h = x;
    for (const W of Ws) { const f = T.relu(T.conv2d(h, W, null, { padding: 1 })); h = o.residual ? T.add(h, T.mul(f, 1 / Math.sqrt(L))) : f; hs.push(h); }
    T.sum(T.mul(h, r)).backward();
    return { act: hs.map(rms), grad: hs.map((t) => (t.grad ? rms(t.grad) : 0)), x: Array.from(x.values()), W: Ws.map((w) => Array.from(w.values())), r: Array.from(r.values()) };
  };

  S.topic({
    id: 'cnn-residual', family: 'p10-train', order: 30, title: 'Residual connections: a shortcut for the gradient',
    blurb: 'Stack many layers and the gradient that reaches the first ones is a product of many factors: it can shrink to nothing or blow up. A residual block adds its input back to its output, h + f(h), so there is always a path on which the gradient passes through unchanged. Build a deep stack both ways from the same weights and measure the gradient at every layer.',
    inputs: [
      { id: 'depth', kind: 'number', label: 'Layers', def: 20, min: 2, max: 40 },
      { id: 'init', kind: 'choice', label: 'Starting weights', def: 'small', options: P.options(INITS) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 9999 },
    ],
    presets: [
      { label: '20 layers, small weights: plain fades', values: { depth: 20, init: 'small', seed: 1 } },
      { label: '20 layers, He weights', values: { depth: 20, init: 'he', seed: 1 } },
      { label: '30 layers, large weights: plain explodes', values: { depth: 30, init: 'large', seed: 1 } },
      { label: 'Shallow: 3 layers', values: { depth: 3, init: 'small', seed: 2 } },
      { label: '40 layers, He weights', values: { depth: 40, init: 'he', seed: 7 } },
    ],
    counters: [{ id: 'passes', label: 'passes run' }],
    run(inp) {
      const L = inp.depth, sc = INITS[inp.init].s, plain = P.stack({ depth: L, scale: sc, seed: inp.seed, residual: false }), res = P.stack({ depth: L, scale: sc, seed: inp.seed, residual: true });
      const F = H().frames(['passes']), q = F.q;
      const lg = (v) => Math.log10(Math.max(v, 1e-30));
      const all = [...plain.act, ...plain.grad, ...res.act, ...res.grad].filter((v) => v > 0).map(lg);
      const lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all));
      const step = hi - lo > 8 ? 2 : 1, ticks = [];
      for (let v = lo; v <= hi; v += step) ticks.push(v);
      const pow = (v) => (v === 0 ? '1' : '10' + [...String(v)].map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]).join(''));
      const ga = C().axes({ prefix: 'ag', x: 384, y: 40, w: 336, h: 170, xr: [0, L], yr: [lo, hi], xticks: L > 20 ? 10 : 5, yticks: ticks, fmtY: pow, yname: 'gradient size at each layer (RMS)', xname: '' });
      const aa = C().axes({ prefix: 'aa', x: 384, y: 290, w: 336, h: 170, xr: [0, L], yr: [lo, hi], xticks: L > 20 ? 10 : 5, yticks: ticks, fmtY: pow, yname: 'activation size (RMS)', xname: 'layer' });
      const ser = (ax, key, v, cls) => C().line(ax, key, v.map((y, i) => [i, lg(y)]), cls);
      const blocks = (lit) => {
        const N = S.Kit.flow.net('bk'), items = [];
        const row = (id, y, skip) => {
          N.box(id + 'c', 105, y, 92, 34, 'conv 3 × 3', lit === id ? 'act' : '');
          N.box(id + 'r', 208, y, 58, 34, 'ReLU', lit === id ? 'act' : '');
          N.wire(id + 'i', [[16, y], N.port(id + 'c', 'w')], { head: true });
          N.wire(id + 'm', [N.port(id + 'c', 'e'), N.port(id + 'r', 'w')], { head: true });
          if (skip) {
            N.op(id + 'p', 270, y, 'add', lit === id ? 'act' : '', 11);
            N.wire(id + 'o', [N.port(id + 'r', 'e'), [259, y]], { head: true });
            N.wire(id + 's', [[36, y], [36, y - 44], [270, y - 44], [270, y - 11]], { head: true, cls: lit === id ? 'act' : '' });
            N.dot(36, y);
            N.wire(id + 'x', [[281, y], [316, y]], { head: true });
          } else N.wire(id + 'o', [N.port(id + 'r', 'e'), [316, y]], { head: true });
        };
        row('pl', 90, false);
        row('rs', 290, true);
        items.push(...N.items, P.text('bk:t1', 0, 40, 'plain block', 'axname', 13), P.text('bk:t2', 0, 196, 'residual block', 'axname', 13));
        items.push(...H().lines('bk:e', 0, 360, [['plain:  h ← ReLU(conv(h))', 'math'], ['residual:  h ← h + ReLU(conv(h)) / √L', 'math'], ['the same weights in both; L = ' + L, 'legend']], 13, 22));
        items.push(...C().legend('lg', 0, 446, [{ text: 'plain', cls: 'p10off' }, { text: 'residual', cls: 'p10on' }]));
        return items;
      };
      const pic = (lit, show) => {
        const items = [...blocks(lit), ...ga.items, ...aa.items];
        if (show.pa) items.push(ser(aa, 's:pa', plain.act, 'p10off'));
        if (show.pg) items.push(ser(ga, 's:pg', plain.grad, 'p10off'));
        if (show.ra) items.push(ser(aa, 's:ra', res.act, 'p10on'));
        if (show.rg) items.push(ser(ga, 's:rg', res.grad, 'p10on'));
        return items;
      };
      F.push(pic(null, {}), `A stack of ${q.n(L)} layers, each a 3 × 3 convolution over ${q.n(4)} channels of an ${q.n(8)} × ${q.n(8)} map, with ${INITS[inp.init].words} random weights (seed ${q.n(inp.seed)}). A random input goes in; the loss is a fixed random mix of the last layer’s outputs. Both stacks use exactly the same numbers.`, { phase: 'start', L, ch: 4, n: 8, seed: inp.seed });
      F.count('passes');
      F.push(pic('pl', { pa: true }), `Plain, forward: the activations start at size ${q.sg(plain.act[0], 3)} and end at ${q.sg(plain.act[L], 3)} after ${q.n(L)} layers.` + (plain.act[L] < 0.01 ? ' Each layer passes on a little less, and the signal all but vanishes.' : plain.act[L] > 100 ? ' Each layer passes on a little more, and the signal blows up.' : ' He weights keep it roughly level: that is what they are chosen for.'), { phase: 'plainF', a0: plain.act[0], aL: plain.act[L], L });
      F.count('passes');
      const pr = plain.grad[0] / plain.grad[L];
      F.push(pic('pl', { pa: true, pg: true }), `Plain, backward: the gradient is ${q.sg(plain.grad[L], 3)} at the last layer and ${q.sg(plain.grad[0], 3)} at the input, ${q.sg(pr, 3)} times as big.` + (pr < 0.01 ? ' The first layers would hardly learn: their gradient has vanished.' : pr > 100 ? ' The first layers would take huge steps: the gradient has exploded.' : ''), { phase: 'plainB', gL: plain.grad[L], g0: plain.grad[0], ratio: pr });
      F.count('passes');
      F.push(pic('rs', { pa: true, pg: true, ra: true }), `Residual, forward: every block adds a small correction to its input instead of replacing it, so the activations go from ${q.sg(res.act[0], 3)} to ${q.sg(res.act[L], 3)}.`, { phase: 'resF', a0: res.act[0], aL: res.act[L] });
      F.count('passes');
      const rr = res.grad[0] / res.grad[L];
      F.push(pic('rs', { pa: true, pg: true, ra: true, rg: true }), `Residual, backward: the gradient goes from ${q.sg(res.grad[L], 3)} at the last layer to ${q.sg(res.grad[0], 3)} at the input, ${q.sg(rr, 3)} times as big. The derivative of h + f(h) is 1 + f′: its identity part carries the gradient straight through every block, whatever f does.`, { phase: 'resB', gL: res.grad[L], g0: res.grad[0], ratio: rr });
      const gain = res.grad[0] / plain.grad[0];
      F.push(pic(null, { pa: true, pg: true, ra: true, rg: true }), `At the input the residual stack’s gradient is ${q.sg(gain, 3)} times the plain one’s.` + (gain > 10 ? ' The shortcut is what lets networks of a hundred layers train at all.' : gain < 0.1 ? ' Here the plain stack’s gradient has blown up, and the shortcut keeps the residual one in bounds.' : ' With so few layers, or such well-chosen weights, the plain stack copes too; the difference grows with depth.'), { phase: 'end', gain });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Why the shortcut helps', text: ['plain: ∂h_L/∂h_0 = Π (product of L layer Jacobians)', 'residual: each factor is I + ∂f/∂h', 'so ∂h_L/∂h_0 = I + (terms with f)', 'the identity path never shrinks the gradient'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'plainF', once: true, pick: 'option',
      ask: 'At the input, will the plain stack’s gradient be far smaller, far larger, or close to the residual stack’s?',
      options: ['far smaller (under a tenth)', 'close (within ten times)', 'far larger (over ten times)'],
      answer: (fr) => { const g = fr[fr.length - 1].meta.gain; return g > 10 ? 'far smaller (under a tenth)' : g < 0.1 ? 'far larger (over ten times)' : 'close (within ten times)'; },
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'A residual block computes…', options: ['h + f(h)', 'f(h) only', 'f(h) − h', 'h · f(h)'], answer: 0, why: 'It adds its input back.' },
      { q: 'Why does the gradient pass a residual block easily?', options: ['the derivative of h + f(h) contains the identity', 'f has no weights', 'ReLU is skipped', 'the block is linear'], answer: 0, why: '1 + f′: the 1 always gets through.' },
      { q: 'In a plain stack where each layer shrinks the gradient by half, after 20 layers it is about…', options: ['a millionth of its size', 'half', 'a twentieth', 'the same'], answer: 0, why: '0.5²⁰ ≈ 10⁻⁶.' },
      { q: 'What must match for h + f(h) to make sense?', options: ['the shapes of h and f(h)', 'the learning rates', 'the number of layers', 'nothing'], answer: 0, why: 'They are added element by element; otherwise a 1 × 1 convolution fixes the shape.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
