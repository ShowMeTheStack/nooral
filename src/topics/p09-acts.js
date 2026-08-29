// Part 9, activations and their slopes: sigmoid, tanh, ReLU and GELU drawn with their derivatives (the slopes from
// backpropagation through the engine's own ops), a chosen point with its tangent, where each one saturates, and what
// happens to a gradient that passes through many of them in a row.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const IDS = ['sigmoid', 'tanh', 'relu', 'gelu'];
  const FLAT = 0.05;

  S.topic({
    id: 'p09-activations', family: 'p09-neuron', order: 20, title: 'Activations and their slopes',
    blurb: 'The activation bends a neuron’s sum. Sigmoid and tanh squash every number into a narrow band; ReLU keeps the positive part and cuts the rest; GELU is a smooth ReLU. Their slopes matter as much as their shapes: a gradient going back through a layer is multiplied by the slope there, and slopes near zero starve the layers below.',
    inputs: [
      { id: 'act', kind: 'choice', label: 'Activation', def: 'tanh', options: P.actOptions(IDS) },
      { id: 'z0', kind: 'slider', label: 'Point z₀', def: 1.5, min: -5, max: 5, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'depth', kind: 'number', label: 'Layers in a row', def: 10, min: 2, max: 30 },
    ],
    presets: [
      { label: 'tanh at 1.5', values: { act: 'tanh', z0: 1.5, depth: 10 } },
      { label: 'Sigmoid at its centre', values: { act: 'sigmoid', z0: 0, depth: 10 } },
      { label: 'Sigmoid saturated', values: { act: 'sigmoid', z0: 4, depth: 5 } },
      { label: 'ReLU on the positive side', values: { act: 'relu', z0: 2, depth: 20 } },
      { label: 'GELU slightly negative', values: { act: 'gelu', z0: -0.5, depth: 6 } },
    ],
    counters: [],
    run(inp) {
      const T = S.T, A = P.ACTS[inp.act], F = H().frames([]), q = F.q, z0 = inp.z0, d = inp.depth;
      const zs = Array.from(T.linspace(-6, 6, 241).values());
      const all = Object.fromEntries(IDS.map((id) => [id, P.actCurve(id, zs)]));
      const at = Object.fromEntries(IDS.map((id) => { const c = P.actCurve(id, [z0]); return [id, { f: c.f[0], d: c.d[0] }]; }));
      const cur = all[inp.act], me = at[inp.act];
      const fmin = Math.min(...cur.f), fmax = Math.max(...cur.f), dmax = Math.max(...cur.d), dmaxAt = zs[cur.d.indexOf(dmax)];
      const yr = inp.act === 'relu' || inp.act === 'gelu' ? [-1, 6] : inp.act === 'tanh' ? [-1.2, 1.2] : [-0.1, 1.1];
      const ax = P.ax({ prefix: 'af', x: 50, y: 60, w: 330, h: 230, xr: [-6, 6], yr, xticks: [-6, -3, 0, 3, 6], yticks: inp.act === 'sigmoid' ? [0, 0.5, 1] : inp.act === 'tanh' ? [-1, 0, 1] : [0, 2, 4, 6], xname: 'z', yname: A.label + '(z)' });
      const dx = P.ax({ prefix: 'ad', x: 480, y: 60, w: 330, h: 230, xr: [-6, 6], yr: [-0.15, 1.15], xticks: [-6, -3, 0, 3, 6], yticks: [0, 0.25, 0.5, 0.75, 1], xname: 'z', yname: 'slope f′(z)' });
      const pair = { sigmoid: 'tanh', tanh: 'sigmoid', relu: 'gelu', gelu: 'relu' };
      const others = (axx, which) => IDS.filter((id) => id !== inp.act && (which === 'd' || id === pair[inp.act])).map((id) => C().line(axx, axx.o.prefix + ':o' + id, zs.map((u, i) => [u, all[id][which][i]]), 'p09alt thin'));
      const base = (o) => {
        o = o || {};
        const items = [...ax.items, ...dx.items];
        if (o.others) items.push(...others(ax, 'f'), ...others(dx, 'd'));
        items.push(C().line(ax, 'af:f', zs.map((u, i) => [u, cur.f[i]]), 'act'));
        if (o.slope) items.push(C().line(dx, 'ad:d', zs.map((u, i) => [u, cur.d[i]]), 'p09slope'));
        if (o.mark) {
          const L = 1.3, clip = (v) => Math.max(yr[0], Math.min(yr[1], v));
          items.push({ key: 'af:tan', kind: 'path', d: `M${ax.X(z0 - L)},${ax.Y(clip(me.f - me.d * L))} L${ax.X(z0 + L)},${ax.Y(clip(me.f + me.d * L))}`, cls: 'p09tan', pathAudit: 'edge-x', layer: 'lines' });
          items.push(C().dot(ax, 'af:pt', z0, me.f, 'p09hot', 6), C().vline(ax, 'af:g', z0, ''), C().vline(dx, 'ad:g', z0, ''));
          if (o.slope) items.push(C().dot(dx, 'ad:pt', z0, me.d, 'p09hot', 6));
        }
        if (o.sat) for (const [k, s] of o.sat.entries()) items.push({ key: 'ad:sat' + k, kind: 'region', x: dx.X(s[0]), y: dx.o.y, w: P.r2(dx.X(s[1]) - dx.X(s[0])), h: dx.o.h, cls: 'p09sat', layer: 'back', audit: 'ignore' });
        return items;
      };
      // where the slope is nearly flat: runs of grid points with |f′| < FLAT
      const flat = [];
      let start = null;
      zs.forEach((u, i) => { const f = Math.abs(cur.d[i]) < FLAT; if (f && start === null) start = u; if ((!f || i === zs.length - 1) && start !== null) { flat.push([start, f ? u : zs[i - 1]]); start = null; } });
      const flatCount = cur.d.filter((v) => Math.abs(v) < FLAT).length, flatShare = flatCount / zs.length;
      F.push(base({}), `${A.label} over z from ${q(-6, 0)} to ${q(6, 0)}: its output runs from ${q(fmin, 3)} to ${q(fmax, 3)}.` + (inp.act === 'sigmoid' ? ' It squashes every number into (0, 1), which reads like a probability.' : inp.act === 'tanh' ? ' It squashes every number into (−1, 1), centred on zero.' : inp.act === 'relu' ? ' It keeps positive numbers as they are and replaces negative ones with zero.' : ' It follows ReLU for large |z| but dips slightly below zero for small negative z, and has no corner.'), { phase: 'curve', fmin, fmax, range: [-6, 6] }, { hl: { eq: [IDS.indexOf(inp.act)] } });
      F.push(base({ mark: true }), `At z₀ = ${q(z0, 1)}: ${A.label}(z₀) = ${q(me.f, 3)}, and the tangent there has slope ${q(me.d, 3)} (from backpropagation through the engine’s ${A.label}).`, { phase: 'point', f: me.f, d: me.d, z0 }, { hl: { eq: [IDS.indexOf(inp.act)] } });
      F.push(base({ mark: true, slope: true }), `The slope at every z, on the right. Its largest value is ${q(dmax, 3)}, at z = ${q(dmaxAt, 2)}.` + (inp.act === 'sigmoid' ? ' Even at best the sigmoid passes on only a quarter of a gradient.' : inp.act === 'relu' ? ' ReLU’s slope is exactly one for every positive z: a gradient passes through unchanged.' : ''), { phase: 'slope', dmax, dmaxAt }, { hl: { eq: [IDS.length + IDS.indexOf(inp.act)] } });
      F.push(base({ mark: true, slope: true, sat: flat }), `Shaded: where the slope is below ${q(FLAT, 2)}, so the neuron hardly responds and passes back almost no gradient. That is ${q.pct(flatShare)} of this range.` + (inp.act === 'relu' ? ' For ReLU it is the whole negative side: a neuron stuck there is “dead” until its inputs change.' : inp.act === 'gelu' ? ' GELU is flat far on the negative side, like ReLU, but keeps a small slope near zero.' : ' Both tails are flat: large inputs saturate the neuron.'), { phase: 'sat', flatShare, flat: FLAT }, { hl: { eq: [IDS.length + IDS.indexOf(inp.act)] } });
      // all four at z0, and the slope multiplied through d layers
      const tab = (chain) => {
        const rows = IDS.map((id) => [P.ACTS[id].label, H().fx(at[id].f, 3), H().fx(at[id].d, 3)].concat(chain ? [H().sg(Math.pow(at[id].d, d), 2)] : []));
        const cols = [{ name: 'activation', w: 110 }, { name: 'f(z₀)', w: 90 }, { name: 'f′(z₀)', w: 90 }].concat(chain ? [{ name: 'f′(z₀)' + String(d).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join(''), w: 130 }] : []);
        return S.Kit.dt.table({ prefix: 'at', x: 50, y: 340, cols, rows, rowH: 26, headH: 28, size: 13, cls: (i) => (IDS[i] === inp.act ? 'act' : '') }).items;
      };
      F.push([...base({ mark: true, slope: true, others: true }), ...tab(false)], `All four at z₀ = ${q(z0, 1)} (dashed: ${P.ACTS[pair[inp.act]].label} beside ${A.label} on the left, the other slopes on the right): ` + IDS.map((id) => `${P.ACTS[id].label} gives ${q(at[id].f, 3)} with slope ${q(at[id].d, 3)}`).join(', ') + '.', { phase: 'compare', at: IDS.map((id) => [at[id].f, at[id].d]) }, { hl: { eq: [] } });
      const chain = IDS.map((id) => Math.pow(at[id].d, d)), mine = chain[IDS.indexOf(inp.act)];
      F.push([...base({ mark: true, slope: true, others: true }), ...tab(true)], `A gradient going back through ${q.n(d)} layers whose neurons all sit at z₀ is multiplied by the slope ${q.n(d)} times. For ${A.label} that ${mine > 1 + 1e-9 ? `multiplies it by ${q.sg(mine, 2)}: it has grown. Slopes above one make gradients explode as the depth grows.` : mine > 1 - 1e-9 ? `multiplies it by ${q.sg(mine, 2)}: all of it survives, which is what deep networks need.` : `leaves ${q.sg(mine, 2)} of it.`}` + (mine > 1 - 1e-9 ? '' : mine < 1e-3 ? ' It has all but vanished: the first layers would barely learn. This is the vanishing-gradient problem, and why deep networks moved to ReLU-like activations.' : mine > 0.5 ? ' Most of it survives, which is what deep networks need.' : ' A good share is lost on the way.'), { phase: 'chain', chain, mine, d }, { hl: { eq: [2 * IDS.length] } });
      P.narrow(F.list, (m) => (m.phase === 'curve' || m.phase === 'point' ? { x: 0, y: 20, w: 420, h: 320 } : m.phase === 'slope' || m.phase === 'sat' ? { x: 430, y: 20, w: 420, h: 320 } : { x: 20, y: 320, w: 560, h: 180 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The activations and their slopes', text: ['sigmoid σ(z) = 1 / (1 + e^−z)', 'tanh(z) = (e^z − e^−z) / (e^z + e^−z)', 'ReLU(z) = max(0, z)', 'GELU(z) ≈ ½ z (1 + tanh(√(2/π)(z + 0.044715 z³)))', 'σ′ = σ (1 − σ)', 'tanh′ = 1 − tanh²', 'ReLU′ = 1 if z > 0, else 0', 'GELU′: smooth, about 1.13 near z = 2', 'through d layers: the gradient × Π f′'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'sat', once: true, pick: 'option',
      ask: (fr, i, inp) => 'Multiply the slope at z₀ by itself ' + inp.depth + ' times. Roughly how much of a gradient is left?',
      options: ['almost all (over half)', 'some (between a thousandth and a half)', 'almost none (under a thousandth)'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'chain').meta.mine; return m > 0.5 ? 'almost all (over half)' : m >= 1e-3 ? 'some (between a thousandth and a half)' : 'almost none (under a thousandth)'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'chain').say,
    }],
    quiz: [
      { q: 'The largest slope of the sigmoid is…', options: ['0.25, at z = 0', '1, at z = 0', '0.5, everywhere', 'unbounded'], answer: 0, why: 'σ′ = σ(1 − σ) is largest when σ = 0.5, giving 0.25.' },
      { q: 'Why do deep networks prefer ReLU to sigmoid in the hidden layers?', options: ['its slope is 1 for positive inputs, so gradients do not shrink layer by layer', 'it is bounded', 'it is smooth everywhere', 'it outputs probabilities'], answer: 0, why: 'Products of sigmoid slopes (at most 0.25) shrink fast; ReLU passes gradients unchanged where it is active.' },
      { q: 'A “dead” ReLU neuron is one that…', options: ['has z < 0 for every input, so it never outputs or learns', 'outputs very large numbers', 'has a weight of exactly zero', 'was dropped by dropout'], answer: 0, why: 'With z < 0 everywhere its output and its slope are zero for all data.' },
      { q: 'tanh differs from the sigmoid mainly in that it is…', options: ['centred on zero, with outputs in (−1, 1)', 'unbounded', 'not differentiable', 'linear'], answer: 0, why: 'tanh(z) = 2σ(2z) − 1: the same shape, stretched to (−1, 1) and centred.' },
      { q: 'GELU differs from ReLU in that it…', options: ['is smooth and lets small negative values through', 'is bounded above', 'is always positive', 'has slope 0 for positive z'], answer: 0, why: 'GELU weighs z by how likely it is to be large; it has no corner at zero and a small dip below it.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
