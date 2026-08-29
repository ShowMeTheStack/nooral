// Part 9, initialisation: a deep stack of dense layers fed a batch of random inputs, its weights drawn too small,
// too large, or scaled by Xavier or He. Layer by layer the histogram of activations shows the signal dying, saturating
// or keeping its size; backpropagation then shows the same for the gradients. All four schemes compared at the end.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const WIDTH = 64, BATCH = 128, DEPTH = 6;
  P.INITS = {
    small: { label: 'too small: normal, std 0.05', short: 'std 0.05' },
    large: { label: 'too large: normal, std 1', short: 'std 1' },
    xavier: { label: 'Xavier: variance 2 / (in + out)', short: 'Xavier' },
    he: { label: 'He: variance 2 / in', short: 'He' },
  };
  /** the stack in the engine: activations and their gradients per layer (loss: half the mean square of the output) */
  P.initRun = (scheme, act, seed) => P.once(['init', scheme, act, seed].join(':'), () => initRun(scheme, act, seed));
  const initRun = function (scheme, act, seed) {
    const T = S.T, g = T.rng(seed), x = T.normal([BATCH, WIDTH], 0, 1, g), Ws = [];
    for (let l = 0; l < DEPTH; l++) {
      const sh = [WIDTH, WIDTH];
      Ws.push(T.param(scheme === 'small' ? T.normal(sh, 0, 0.05, g) : scheme === 'large' ? T.normal(sh, 0, 1, g) : scheme === 'xavier' ? T.xavier(sh, g, { dist: 'normal' }) : T.he(sh, g)));
    }
    const hs = [];
    let h = x;
    for (let l = 0; l < DEPTH; l++) { h = T[act](T.matmul(h, Ws[l])); hs.push(h); }
    T.mul(T.sum(T.square(h)), 0.5).backward();
    const std = (v) => { let s = 0, s2 = 0; for (const a of v) { s += a; s2 += a * a; } const m = s / v.length; return Math.sqrt(Math.max(0, s2 / v.length - m * m)); };
    const acts = hs.map((t) => t.values()), grads = hs.map((t) => t.grad.values());
    const flat = (v) => (act === 'relu' ? v.filter((a) => a === 0).length : v.filter((a) => Math.abs(a) > 0.95).length) / v.length;
    return { acts, xstd: std(x.values()), wstd: Ws.map((W) => std(W.values())), stds: acts.map(std), gstds: grads.map(std), flat: acts.map(flat) };
  };

  S.topic({
    id: 'p09-init', family: 'p09-init', order: 10, title: 'Initialisation: why the first weights matter',
    blurb: 'Before training, every weight is a random number, and the size of those numbers decides whether a deep network can learn at all. Send a batch of random inputs through six layers and watch each layer’s activations: with weights too small the signal shrinks to nothing, too large and it saturates or explodes. Xavier and He scale the weights to the width of the layer so the signal, and the gradient coming back, keep their size.',
    inputs: [
      { id: 'scheme', kind: 'choice', label: 'Initial weights', def: 'small', options: Object.keys(P.INITS).map((k) => ({ value: k, label: P.INITS[k].label })) },
      { id: 'act', kind: 'choice', label: 'Activation', def: 'tanh', options: P.actOptions(['tanh', 'relu']) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Too small, tanh', values: { scheme: 'small', act: 'tanh', seed: 1 } },
      { label: 'Too large, tanh', values: { scheme: 'large', act: 'tanh', seed: 1 } },
      { label: 'Xavier, tanh', values: { scheme: 'xavier', act: 'tanh', seed: 1 } },
      { label: 'Xavier, ReLU', values: { scheme: 'xavier', act: 'relu', seed: 1 } },
      { label: 'He, ReLU', values: { scheme: 'he', act: 'relu', seed: 1 } },
      { label: 'Too large, ReLU', values: { scheme: 'large', act: 'relu', seed: 1 } },
    ],
    counters: [{ id: 'layer', label: 'layers passed' }],
    run(inp) {
      const F = H().frames(['layer']), q = F.q, A = P.ACTS[inp.act], r = P.initRun(inp.scheme, inp.act, inp.seed), I = P.INITS[inp.scheme];
      const panel = (l) => ({ x: 40 + (l % 3) * 300, y: 50 + Math.floor(l / 3) * 170, w: 240, h: 100 });
      const hist = (l, on) => {
        const b = panel(l), v = r.acts[l], s = r.stds[l];
        const lo = inp.act === 'relu' ? 0 : -1, hi = inp.act === 'relu' ? Math.max(1e-6, 3 * Math.max(...r.stds)) : 1, nb = 30;
        const counts = new Array(nb).fill(0);
        for (const a of v) { const k = Math.floor(((a - lo) / (hi - lo)) * nb); counts[Math.max(0, Math.min(nb - 1, k))]++; }
        const share = counts.map((c) => c / v.length), ymax = Math.max(...share) * 1.08 || 1, edges = Array.from({ length: nb + 1 }, (_, k) => lo + ((hi - lo) * k) / nb);
        const ax = P.ax(Object.assign({ prefix: 'ih' + l, xr: [lo, hi], yr: [0, ymax], xticks: inp.act === 'relu' ? [0, hi / 2, hi] : [-1, -0.5, 0, 0.5, 1], yticks: 0, grid: false, fmtX: (u) => H().sg(u, 2) }, b));
        return [...ax.items, ...S.Kit.dt.hist(ax, 'ih' + l + ':b', share, edges, 'p09h'), H().text('ih' + l + ':t', b.x, b.y - 10, 'layer ' + (l + 1) + ':  std ' + H().sg(s, 2), on ? 'strong' : 'legend', 14)];
      };
      const empty = (l) => { const b = panel(l); return [{ key: 'ih' + l + ':e', kind: 'path', d: `M${b.x},${b.y + b.h} h${b.w}`, cls: 'axis', audit: 'ignore', layer: 'back' }, H().text('ih' + l + ':t', b.x, b.y - 10, 'layer ' + (l + 1), 'legend', 14)]; };
      const grid = (upto, lit) => Array.from({ length: DEPTH }, (_, l) => (l <= upto ? hist(l, l === lit) : empty(l))).flat();
      // judged against the layer before (the inputs for layer 1): a spread that keeps falling is not healthy, however large it still is
      const what = (l) => {
        const k = r.stds[l] / (l ? r.stds[l - 1] : r.xstd), pct = () => q.pct(k) + ' of ' + (l ? 'the layer before' : 'the inputs');
        if (r.stds[l] < 0.02) return 'shrinking towards zero';
        if (inp.act === 'tanh' && r.flat[l] > 0.3) return 'piling up at ±1, saturated';
        if (inp.act === 'relu' && r.stds[l] > 5) return 'growing out of hand';
        if (k < 0.6) return 'shrinking, to ' + pct();
        if (k < 0.9) return 'falling gently, to ' + pct();
        if (k > 1.6) return 'growing, to ' + pct();
        if (k > 1.15) return 'growing gently, to ' + pct();
        return 'keeping a healthy spread';
      };
      F.push(grid(-1, -1), `A stack of ${q.n(DEPTH)} dense layers, ${q.n(WIDTH)} neurons each, with ${A.label}, and no training yet. A batch of ${q.n(BATCH)} random inputs goes in, with standard deviation ${q(r.xstd, 2)}. The weights are drawn ${I.label.split(':')[0] === 'Xavier' || I.label.split(':')[0] === 'He' ? 'by the ' + I.short + ' rule' : I.label.split(':')[0]}, standard deviation ${q.sg(r.wstd[0], 2)}.`, { phase: 'start', xstd: r.xstd, wstd: r.wstd[0], depth: DEPTH, width: WIDTH, batch: BATCH }, { hl: { eq: [0] } });
      for (let l = 0; l < DEPTH; l++) {
        F.count('layer');
        const fl = r.flat[l];
        F.push(grid(l, l), `Layer ${q.n(l + 1)}: the activations have standard deviation ${q.sg(r.stds[l], 2)}, ${what(l)}.` + (inp.act === 'tanh' ? ` ${q.pct(fl)} of them are beyond ±${q(0.95, 2)}, where tanh is flat.` : ` ${q.pct(fl)} of them are exactly zero.`), { phase: 'layer', l: l + 1, std: r.stds[l], flat: fl, lim: 0.95, prevRatio: r.stds[l] / (l ? r.stds[l - 1] : r.xstd) }, { hl: { eq: [1] } });
      }
      // the forward and backward spread on one plot
      const lg = (v) => Math.log10(Math.max(1e-12, v)), ys = [...r.stds, ...r.gstds].map(lg), ylo = Math.floor(Math.min(...ys, -3)), yhi = Math.ceil(Math.max(...ys, 0.5));
      const sup = (k) => String(k).split('').map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]).join('');
      const tick = Math.max(1, Math.ceil((yhi - ylo) / 6));
      const ax = P.ax({ prefix: 'is', x: 40, y: 390, w: 420, h: 160, xr: [0.7, DEPTH + 0.3], yr: [ylo, yhi], xticks: Array.from({ length: DEPTH }, (_, k) => k + 1), yticks: Array.from({ length: Math.floor((yhi - ylo) / tick) + 1 }, (_, k) => ylo + k * tick), fmtY: (v) => '10' + sup(v), xname: 'layer', yname: 'standard deviation (log scale)' });
      const series = (gr) => {
        const out = [...ax.items, C().line(ax, 'is:a', r.stds.map((s, l) => [l + 1, lg(s)]), 'p09train'), ...r.stds.map((s, l) => P.dot(ax, 'is:ad' + l, l + 1, lg(s), 'p09train', 4))];
        if (gr) out.push(C().line(ax, 'is:g', r.gstds.map((s, l) => [l + 1, lg(s)]), 'p09val'), ...r.gstds.map((s, l) => P.dot(ax, 'is:gd' + l, l + 1, lg(s), 'p09val', 4)));
        out.push(...C().legend('is:lg', 520, 420, [{ text: 'activations, going forward', cls: 'p09train' }].concat(gr ? [{ text: 'gradients, coming back', cls: 'p09val' }] : []), { dir: 'col', size: 14 }));
        return out;
      };
      const ratio = r.stds[DEPTH - 1] / r.stds[0];
      F.push([...grid(DEPTH - 1, -1), ...series(false)], `Across the layers the spread goes from ${q.sg(r.stds[0], 2)} to ${q.sg(r.stds[DEPTH - 1], 2)}, a factor of ${q.sg(ratio, 2)}.` + (ratio < 0.2 ? ' Each layer multiplies the signal by less than one, and the last layers see almost nothing.' : ratio > 5 ? ' Each layer multiplies it by more than one.' : ratio < 0.4 ? ` It shrinks steadily, to ${q.pct(ratio)} from the first layer to the last: each layer passes on less than it receives, and a deeper stack would fade further.` : ratio < 0.8 ? ` It falls gently, to ${q.pct(ratio)} from the first layer to the last: each layer passes on a little less than it receives, nothing like the collapse of a bad start.` : ratio > 1.25 ? ` It grows gently, to ${q.pct(ratio)} from the first layer to the last: each layer passes on a little more than it receives.` : ' Roughly constant: each layer passes on about as much as it receives.'), { phase: 'forward', first: r.stds[0], last: r.stds[DEPTH - 1], ratio }, { hl: { eq: [2] } });
      const gr = r.gstds[0] / r.gstds[DEPTH - 1];
      F.push([...grid(DEPTH - 1, -1), ...series(true)], `Backpropagation runs the other way, from the loss L = ½ Σ h² over the last layer, so the gradient starts as the last layer’s own activations, spread ${q.sg(r.gstds[DEPTH - 1], 2)}; by the first layer it is ${q.sg(r.gstds[0], 2)}, a factor of ${q.sg(gr, 2)}.` + (gr < 0.1 ? ' The first layers would hardly learn: the gradient has vanished on the way back.' : gr > 10 ? ' The gradient grows on the way back: the first layers would take wild steps.' : ' It reaches the first layer in good shape.'), { phase: 'backward', gfirst: r.gstds[0], glast: r.gstds[DEPTH - 1], ratio: gr }, { hl: { eq: [3] } });
      // all four schemes
      const all = Object.keys(P.INITS).map((k) => { const o = P.initRun(k, inp.act, inp.seed); return [k, o.stds[DEPTH - 1], o.gstds[0]]; });
      const nameQ = (k) => (k === 'small' ? 'std ' + q(0.05, 2) : k === 'large' ? 'std ' + q(1, 0) : P.INITS[k].short);
      const tab = S.Kit.dt.table({ prefix: 'it', x: 520, y: 440, cols: [{ name: 'weights', w: 110 }, { name: 'last layer', w: 120 }, { name: 'first gradient', w: 140 }], rows: all.map(([k, a, g]) => [P.INITS[k].short, H().sg(a, 2), H().sg(g, 2)]), rowH: 25, headH: 27, size: 14, cls: (i) => (all[i][0] === inp.scheme ? 'act' : '') }).items;
      const good = inp.act === 'relu' ? 'he' : 'xavier', gi = all.find((a) => a[0] === good);
      F.push([...grid(DEPTH - 1, -1), ...series(true).filter((it) => !String(it.key).startsWith('is:lg')), ...tab], `The four ways side by side with ${A.label} (the standard deviation of the last layer’s activations, and of the first layer’s gradient): ` + all.map(([k, a, g]) => `${nameQ(k)} gives ${q.sg(a, 2)} and ${q.sg(g, 2)}`).join('; ') + `. For ${A.label} the ${P.INITS[good].short} rule keeps both from fading or saturating: ${inp.act === 'relu' ? 'ReLU zeroes half of its inputs, and He’s factor of two makes up for it.' : 'it balances the variance going forward and coming back.'}`, { phase: 'compare', all: all.map((a) => [a[1], a[2]]), good: gi[1], sizes: [0.05, 1] }, { hl: { eq: [4, 5] } });
      P.narrow(F.list, (m) => { if (m.phase === 'layer' || m.phase === 'start') { const b = panel(m.phase === 'start' ? 0 : m.l - 1); return { x: b.x - 20, y: b.y - 34, w: b.w + 40, h: b.h + 70 }; } return m.phase === 'compare' ? { x: 500, y: 420, w: 400, h: 160 } : { x: 0, y: 370, w: 520, h: 230 }; });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Keeping the variance', text: ['layer: h = f(h_prev · W), W of size n × n', 'Var(h) ≈ n · Var(w) · Var(h_prev) · (gain of f)', 'forward spread: std(h) per layer', 'backward spread: std(∂L/∂h) per layer, L = ½ Σ h²', 'Xavier: Var(w) = 2 / (n_in + n_out)  (tanh)', 'He: Var(w) = 2 / n_in  (ReLU)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'layer' && f.meta.l === 1, once: true, pick: 'option',
      ask: 'By the sixth layer, will the spread of the activations be much smaller, about the same, or much larger than at the first?',
      options: ['much smaller', 'about the same', 'much larger'],
      answer: (fr) => { const r = fr.find((f) => f.meta.phase === 'forward').meta.ratio; return r < 0.33 ? 'much smaller' : r > 3 ? 'much larger' : 'about the same'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'forward').say,
    }],
    quiz: [
      { q: 'A layer has 100 inputs. Under He initialisation the weights have standard deviation…', options: ['√(2/100) ≈ 0.14', '0.01', '1', '2/100'], answer: 0, why: 'He sets Var(w) = 2 / n_in.' },
      { q: 'Why not start every weight at zero?', options: ['every neuron in a layer would compute and learn the same thing', 'zero is not a number the engine allows', 'the loss would be infinite', 'the biases would explode'], answer: 0, why: 'Identical weights get identical gradients: the symmetry never breaks.' },
      { q: 'With tanh and weights that are too large, most activations end up…', options: ['near ±1, where the slope is almost zero', 'exactly zero', 'very large', 'negative'], answer: 0, why: 'Large sums saturate tanh, which stops the gradient.' },
      { q: 'He initialisation has a factor 2 that Xavier lacks because…', options: ['ReLU sets about half of its inputs to zero, halving the variance', 'ReLU is twice as steep', 'it uses two layers', 'it is for two classes'], answer: 0, why: 'Doubling the weight variance restores what ReLU throws away.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
