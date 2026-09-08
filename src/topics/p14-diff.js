// Part 14, diffusion: the forward process dissolving points and an image into noise (live, from the engine's
// schedule), the trained denoiser at work (its stored weights, run live), sampling 2-D points backwards step by step
// (live), and sampling 16×16 images (a stored run of the engine).
(function (S) {
  'use strict';
  const P = S.P14, H = () => S.Kit.ml, C = () => S.Kit.curve;
  P.SHAPES = ['circle', 'square', 'triangle', 'star'];
  /** a real shape image as the diffusion model sees it, in [−1, 1], with one fixed draw of noise per pixel */
  P.shapeImg = (cls) => P.once('shape' + cls, () => { const set = S.Gen.images('shapes', { n: 4, size: 16, seed: 1, classes: [cls] }); return Array.from(set.X.subarray(0, 256), (v) => 2 * v - 1); });
  P.toPix = (x) => Float32Array.from(x, (u) => Math.max(0, Math.min(1, (u + 1) / 2)));
  /** the forward process at the chosen steps, from the engine's schedule and its noising formula */
  P.forward = (inp) => {
    const G = S.Gen, sched = G.schedule({ steps: inp.T, kind: inp.kind }), data = G.dist2d(inp.data, { n: 2048, seed: 1 }), pts = data.X.slice(0, 256);
    const g = S.T.rng(31), eps = pts.map(() => [g.normal(), g.normal()]), x0 = P.shapeImg(P.SHAPES.indexOf(inp.shape)), g2 = S.T.rng(31), ei = x0.map(() => g2.normal());
    return { sched, pts, eps, x0, ei, at: (t) => ({ t, points: pts.map((p, i) => G.noise(p, t, sched, eps[i])), image: P.toPix(G.noise(x0, t, sched, ei)), ab: sched.alphaBar[t], beta: sched.beta[t] }) };
  };
  const plane = (pre, x, y, w, R, o) => C().axes(Object.assign({ prefix: pre, x, y, w, h: w, xr: [-R, R], yr: [-R, R], xticks: R >= 3 ? 1.5 : 1, yticks: R >= 3 ? 1.5 : 1, grid: false, xname: 'x₁', yname: 'x₂' }, o || {}));
  const dots = (ax, pre, pts, cls, r) => pts.filter((p) => ax.inside(p[0], p[1])).map((p, i) => ({ key: pre + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: r || 2.6, cls, audit: 'overlap-ok', layer: 'items' }));

  S.topic({
    id: 'gen-diff-forward', family: 'p14-diffusion', order: 10, title: 'Diffusion: dissolving data into noise',
    blurb: 'The forward process of a diffusion model needs no learning at all. At each of T steps it shrinks the data a little and adds a little Gaussian noise, by an amount set in a schedule β₁ … β_T. After enough steps nothing of the data is left. The useful fact is that x_t can be reached in one jump: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε, where ᾱ_t is the product of the (1 − β)s so far.',
    inputs: [
      { id: 'kind', kind: 'choice', label: 'Schedule', def: 'linear', options: [{ value: 'linear', label: 'linear β' }, { value: 'cosine', label: 'cosine ᾱ' }] },
      { id: 'T', kind: 'number', label: 'Steps T', def: 100, min: 20, max: 200 },
      { id: 'data', kind: 'choice', label: 'Points', def: 'moons', options: ['moons', 'ring', 'spiral', 'circle'].map((v) => ({ value: v, label: v })) },
      { id: 'shape', kind: 'choice', label: 'Image', def: 'circle', options: P.SHAPES.map((v) => ({ value: v, label: 'a ' + v })) },
    ],
    presets: [
      { label: 'Linear, 100 steps', values: { kind: 'linear', T: 100, data: 'moons', shape: 'circle' } },
      { label: 'Cosine, 100 steps', values: { kind: 'cosine', T: 100, data: 'moons', shape: 'circle' } },
      { label: 'A ring and a star', values: { kind: 'linear', T: 60, data: 'ring', shape: 'star' } },
      { label: 'Short: 20 steps', values: { kind: 'cosine', T: 20, data: 'spiral', shape: 'triangle' } },
    ],
    counters: [{ id: 't', label: 'noising steps' }],
    run(inp) {
      const fw = P.forward(inp), s = fw.sched, T = inp.T, F = H().frames(['t']), q = F.q;
      const ts = [...new Set([0, 1, Math.round(T * 0.05), Math.round(T * 0.1), Math.round(T * 0.2), Math.round(T * 0.35), Math.round(T * 0.5), Math.round(T * 0.7), T])].filter((t) => t <= T).sort((a, b) => a - b);
      const ax = plane('df', 40, 50, 300, 3), ac = C().axes({ prefix: 'dc', x: 540, y: 250, w: 240, h: 150, xr: [0, T], yr: [0, 1], xticks: P.ticksN(T), yticks: [0, 0.5, 1], xname: 't', yname: 'share of signal and noise' });
      const sig = [...Array(T + 1).keys()].map((t) => [t, Math.sqrt(s.alphaBar[t])]), noi = [...Array(T + 1).keys()].map((t) => [t, Math.sqrt(1 - s.alphaBar[t])]);
      const lg = C().legend('dcl', 540, 448, [{ text: '√ᾱ: signal', cls: 'p14sig' }, { text: '√(1 − ᾱ): noise', cls: 'p14noise' }], { dir: 'row', size: 12 });
      const strip = (t) => ts.filter((u) => u <= t).map((u, k) => P.img('st:' + u, 380 + k * 46, 50, 40, fw.at(u).image));
      for (const t of ts) {
        F.set('t', t);
        const a = fw.at(t), sg = Math.sqrt(a.ab), nz = Math.sqrt(1 - a.ab), p0 = fw.pts[0], p = a.points[0];
        const items = [...ax.items, ...dots(ax, 'df:p', a.points, 'p14fake'), { key: 'df:one', kind: 'dot', x: ax.X(Math.max(-3, Math.min(3, p[0]))), y: ax.Y(Math.max(-3, Math.min(3, p[1]))), r: 6, cls: 'p14z', layer: 'heads', audit: 'overlap-ok' }, H().text('df:h', 40, 24, 'The points at step ' + t, 'strong', 14)];
        items.push(P.img('di:big', 380, 120, 120, a.image, 'act'), H().text('di:h', 380, 24, 'The image along the way', 'strong', 14), ...strip(t));
        items.push(...ac.items, C().line(ac, 'dc:s', sig, 'p14sig'), C().line(ac, 'dc:n', noi, 'p14noise'), C().vline(ac, 'dc:now', t, 'act'), C().dot(ac, 'dc:ds', t, sg, 'p14sigdot'), ...(Math.abs(ac.Y(sg) - ac.Y(nz)) > 10 ? [C().dot(ac, 'dc:dn', t, nz, 'p14noisedot')] : []), ...lg);
        let say;
        if (t === 0) say = `At t = ${q.n(0)} these are the data: ${q.n(fw.pts.length)} points and a 16×16 ${inp.shape}. A ${inp.kind} schedule over ${q.n(T)} steps sets β from ${q.sg(s.beta[1], 3)} at the first step to ${q.sg(s.beta[T], 3)} at the last.`;
        else say = `t = ${q.n(t)}: β_t = ${q.sg(a.beta, 3)} and ᾱ_t = ${q.sg(a.ab, 3)}, so x_t = ${q(sg, 3)} · x₀ + ${q(nz, 3)} · ε. The marked point (${q(p0[0], 2)}, ${q(p0[1], 2)}) is now at (${q(p[0], 2)}, ${q(p[1], 2)}).` + (t === T ? ' Almost nothing of the data is left: this is what sampling will start from.' : '');
        F.push(items, say, { phase: t ? (t === T ? 'end' : 'step') : 'start', t, T, n: fw.pts.length, beta: a.beta, ab: a.ab, sg, nz, p0, p, b1: s.beta[1], bT: s.beta[T] });
      }
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The forward process', text: ['x_t = √(1 − β_t) · x_{t−1} + √β_t · ε_t', 'ᾱ_t = (1 − β₁)(1 − β₂) … (1 − β_t)', 'x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε,  ε ~ N(0, I)', inp.kind === 'linear' ? 'linear: β_t rises in a straight line' : 'cosine: ᾱ_t follows cos² of a quarter turn'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Halfway through, at t = T/2, how much of the signal is left (√ᾱ)?', options: ['more than half', 'less than half'], answer: (fr) => { const m = fr.find((f) => f.meta.t === Math.round(f.meta.T * 0.5)); return m.meta.sg > 0.5 ? 'more than half' : 'less than half'; }, explain: (fr) => fr.find((f) => f.meta.t === Math.round(f.meta.T * 0.5)).say },
    ],
    quiz: [
      { q: 'ᾱ_t = 0.36. What multiplies x₀ and what multiplies ε in x_t?', options: ['0.6 and 0.8', '0.36 and 0.64', '0.64 and 0.36', '0.8 and 0.6'], answer: 0, why: '√0.36 = 0.6 and √(1 − 0.36) = 0.8.' },
      { q: 'Why does the forward process need no training?', options: ['it is a fixed recipe of scaling and adding noise', 'it copies a trained network', 'it uses the data’s labels', 'it is learned by the denoiser'], answer: 0, why: 'Only the reverse direction is learned.' },
      { q: 'Why is it useful that x_t can be reached in one jump from x₀?', options: ['training can pick any t at random without simulating the steps before it', 'it makes sampling one step long', 'it removes the need for noise', 'it makes β constant'], answer: 0, why: 'Each training example is one draw of t, ε and x₀.' },
      { q: 'β₁ = 0.1 and β₂ = 0.2. What is ᾱ₂?', options: ['0.72', '0.7', '0.3', '0.02'], answer: 0, why: '(1 − 0.1)(1 − 0.2) = 0.9 × 0.8.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  /** the denoiser's predicted noise for points at step t (the engine's network, from its stored weights) */
  P.epsHat = (pts, t) => { const d = P.diff2d(), T = S.T; return T.noGrad(() => { const v = d.net.forward(T.tensor(pts), new Array(pts.length).fill(t)).values(); return P.pairs(Array.from(v)); }); };

  S.topic({
    id: 'gen-diff-denoise', family: 'p14-diffusion', order: 20, title: 'Learning to predict the noise',
    blurb: 'The reverse direction is learned. A network is shown a noised point x_t and the step t, and asked for the noise ε that was added; its loss is the squared error. Knowing the noise is knowing the way back: x̂₀ = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t is its guess at the clean point. This denoiser was trained on two moons for 3000 steps; the arrows show where it thinks the clean data are, from every place in the plane, at different noise levels.',
    inputs: [
      { id: 'ex', kind: 'number', label: 'Training example (1–50)', def: 1, min: 1, max: 50 },
      { id: 't', kind: 'number', label: 'Its step t (1–100)', def: 30, min: 1, max: 100 },
    ],
    presets: [{ label: 'A middle step', values: { ex: 1, t: 30 } }, { label: 'Almost clean', values: { ex: 4, t: 5 } }, { label: 'Almost pure noise', values: { ex: 7, t: 90 } }],
    counters: [{ id: 'step', label: 'frames' }],
    run(inp) {
      const d = P.diff2d(), s = d.sched, J = P.D().diff2d, X = d.data.X, F = H().frames(['step']), q = F.q;
      const g = S.T.rng(1000 + inp.ex), x0 = X[(inp.ex * 37) % X.length], eps = [g.normal(), g.normal()], t = inp.t, ab = s.alphaBar[t];
      const xt = S.Gen.noise(x0, t, s, eps), eh = P.epsHat([xt], t)[0], err = (eps[0] - eh[0]) ** 2 + (eps[1] - eh[1]) ** 2;
      const x0h = [(xt[0] - Math.sqrt(1 - ab) * eh[0]) / Math.sqrt(ab), (xt[1] - Math.sqrt(1 - ab) * eh[1]) / Math.sqrt(ab)];
      const ax = plane('dn', 40, 50, 340, 3), data = dots(ax, 'dn:d', X.slice(0, 300), 'p14real', 2.2);
      const pt = (key, p, cls, r) => ({ key, kind: 'dot', x: ax.X(Math.max(-3, Math.min(3, p[0]))), y: ax.Y(Math.max(-3, Math.min(3, p[1]))), r: r || 6, cls, layer: 'heads', audit: 'overlap-ok' });
      const arr = (key, a, b, cls) => H().arrow(key, [ax.X(a[0]), ax.Y(a[1])], [ax.X(Math.max(-3, Math.min(3, b[0]))), ax.Y(Math.max(-3, Math.min(3, b[1])))], cls);
      const lmax = Math.max(...J.loss), al = C().axes({ prefix: 'dl', x: 470, y: 300, w: 300, h: 120, xr: [0, J.steps], yr: [0, lmax * 1.05], xticks: P.ticksN(J.steps), yticks: C().ticks(0, lmax, 2), xname: 'training step', yname: 'loss: mean of ‖ε − ε̂‖² per coordinate' });
      const loss = [C().line(al, 'dl:l', J.loss.map((v, j) => [(j + 1) * 10, v]), 'p14loss'), ...al.items];
      const head = (text) => H().text('dn:h', 40, 24, text, 'strong', 14);
      const box = (rows) => H().lines('nb', 470, 70, rows, 13, 22);
      F.count('step');
      F.push([...ax.items, ...data, pt('dn:x0', x0, 'p14mu'), head('One training example'), ...loss, ...box([['a clean point x₀ from the moons', 'legend']])], `A training example: the clean point x₀ = (${q(x0[0], 2)}, ${q(x0[1], 2)}) from the moons, a step t = ${q.n(t)} drawn at random, and fresh noise ε = (${q(eps[0], 2)}, ${q(eps[1], 2)}).`, { phase: 'example', x0, t, eps });
      F.count('step');
      F.push([...ax.items, ...data, pt('dn:x0', x0, 'p14mu'), arr('dn:a1', x0, xt, 'p14arr'), pt('dn:xt', xt, 'p14z'), head('Noised in one jump'), ...loss, ...box([['x_t = √ᾱ · x₀ + √(1 − ᾱ) · ε', 'legend']])], `At t = ${q.n(t)}, ᾱ = ${q.sg(ab, 3)}, so x_t = ${q(Math.sqrt(ab), 3)} · x₀ + ${q(Math.sqrt(1 - ab), 3)} · ε = (${q(xt[0], 2)}, ${q(xt[1], 2)}). The network sees only x_t and t.`, { phase: 'noised', t, ab, sa: Math.sqrt(ab), sn: Math.sqrt(1 - ab), xt });
      F.count('step');
      F.push([...ax.items, ...data, pt('dn:x0', x0, 'p14mu'), pt('dn:xt', xt, 'p14z'), arr('dn:a2', xt, x0h, 'p14eps'), pt('dn:x0h', x0h, 'p14z soft'), head('The network’s guess'), ...loss, ...box([['x̂₀ = (x_t − √(1 − ᾱ) · ε̂) / √ᾱ', 'legend']])], `The denoiser predicts ε̂ = (${q(eh[0], 2)}, ${q(eh[1], 2)}) against the true (${q(eps[0], 2)}, ${q(eps[1], 2)}): a squared error of ${q.sg(err, 3)}. Undoing that noise gives its guess at the clean point, x̂₀ = (${q(x0h[0], 2)}, ${q(x0h[1], 2)}), ${q(Math.hypot(x0h[0] - x0[0], x0h[1] - x0[1]), 2)} from the true one.`, { phase: 'predict', eh, eps, err, x0h, x0, miss: Math.hypot(x0h[0] - x0[0], x0h[1] - x0[1]) });
      // the field of guesses at several noise levels
      const grid = []; for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) grid.push([-2.7 + (5.4 * b) / 10, 2.7 - (5.4 * a) / 10]);
      for (const tt of [90, 50, 20, 5]) {
        const e = P.epsHat(grid, tt), a2 = s.alphaBar[tt], guess = grid.map((p, i) => [(p[0] - Math.sqrt(1 - a2) * e[i][0]) / Math.sqrt(a2), (p[1] - Math.sqrt(1 - a2) * e[i][1]) / Math.sqrt(a2)]);
        const arrows = grid.map((p, i) => { const g2 = guess[i], dx = g2[0] - p[0], dy = g2[1] - p[1], L = Math.hypot(dx, dy), k = Math.min(1, 0.42 / (L || 1)); return arr('dn:f' + i, p, [p[0] + dx * k, p[1] + dy * k], 'p14eps'); });
        const before = S.Gen.chamfer(grid, X.slice(0, 600)), after = S.Gen.chamfer(guess, X.slice(0, 600));
        // how far the guesses sit from the centre of the data, against the data's own spread (medians)
        const ctr = [0, 1].map((k) => X.reduce((a, p) => a + p[k], 0) / X.length), med = (v) => { const w = v.slice().sort((a, b) => a - b); return w[w.length >> 1]; };
        const gSpread = med(guess.map((p) => Math.hypot(p[0] - ctr[0], p[1] - ctr[1]))), dSpread = med(X.map((p) => Math.hypot(p[0] - ctr[0], p[1] - ctr[1])));
        const why = () => (after > before
          ? `So little of x₀ is left in x_t that x̂₀ divides by √ᾱ = ${q.sg(Math.sqrt(a2), 2)}: every small error in ε̂ is magnified ${q.n(Math.round(1 / Math.sqrt(a2)))} times, and the guesses fly far past the data. At this noise level x̂₀ is not to be trusted.`
          : gSpread < 0.6 * dSpread ? 'With this much noise any place could have come from anywhere, so the guesses gather towards the middle of the data.'
          : (tt <= 20 ? 'With little noise left, each place is pulled to the nearest part of a moon.' : 'Even with this much noise, each place is pulled onto the moons.'));
        F.count('step');
        F.push([...ax.items, ...data, ...arrows, head('Its guesses from everywhere, t = ' + tt), ...loss, ...box([['each arrow points from x towards x̂₀', 'legend']])], `At t = ${q.n(tt)} the denoiser is asked about ${q.n(grid.length)} places on a grid. Its guesses x̂₀ lie a mean ${q(after, 2)} from the nearest data point, where the places themselves lie ${q(before, 2)} from it. ` + why(), { phase: 'field', t: tt, n: grid.length, before, after, gSpread, dSpread, sa: Math.sqrt(a2), magnify: Math.round(1 / Math.sqrt(a2)) });
      }
      F.count('step');
      const tail = J.loss.slice(-10).reduce((a, b) => a + b, 0) / 10;
      F.push([...ax.items, ...data, head('Training'), ...loss, ...box([['every step: a batch of x₀, t and ε; loss ‖ε − ε̂‖²', 'legend']])], `Training repeated this for ${q.n(J.steps)} steps of ${q.n(J.o.batch || 128)} examples each. The loss fell from ${q.sg(J.loss[0], 3)} to about ${q.sg(tail, 3)} over the last steps; it cannot reach ${q.n(0)}, since at small t the noise is a tiny nudge that cannot be told apart from the data’s own spread (at large t, ε is easy to read off, as x_t is almost all noise).`, { phase: 'train', steps: J.steps, batch: J.o.batch || 128, first: J.loss[0], tail, zero: 0 });
      for (const f of F.list) f.narrowView = { x: 10, y: 0, w: 400, h: 480 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Training the denoiser', text: ['draw x₀ from the data, t from 1 … T, ε ~ N(0, I)', 'x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε', 'loss = ‖ε − ε̂(x_t, t)‖²', 'x̂₀ = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'noised', pick: 'option', ask: 'Will the network’s guess x̂₀ land within 0.3 of the true clean point?', options: ['yes', 'no'], answer: (fr) => (fr.find((f) => f.meta.phase === 'predict').meta.miss <= 0.3 ? 'yes' : 'no'), explain: (fr) => fr.find((f) => f.meta.phase === 'predict').say },
    ],
    quiz: [
      { q: 'What is the denoiser trained to output?', options: ['the noise ε that was added to make x_t', 'the clean point directly, always', 'the step t', 'a probability that x_t is real'], answer: 0, why: 'Predicting ε is equivalent to predicting x₀, and trains more evenly across t.' },
      { q: 'Why must the denoiser be told t?', options: ['the same x_t means very different things at different noise levels', 't is the label', 'to pick the batch size', 'it does not need it'], answer: 0, why: 'At small t, x_t is nearly clean; at large t, nearly pure noise.' },
      { q: 'ᾱ = 0.25, x_t = 1 and ε̂ = 0.5 in one coordinate. What is x̂₀?', options: ['(1 − 0.866 × 0.5) / 0.5 ≈ 1.13', '0.5', '2', '0.25'], answer: 0, why: '√(1 − 0.25) ≈ 0.866 and √0.25 = 0.5.' },
      { q: 'Why does the loss not fall to zero?', options: ['at low noise, the tiny ε cannot be told apart from the data’s own spread', 'the network is too small to fit anything', 'the learning rate is zero', 'the loss is measured on the wrong data'], answer: 0, why: 'When x_t is almost x₀, many small nudges could have produced it, so the best guess at ε is an average that still leaves error. At high noise x_t is almost ε itself, and ε is easy to read off.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'gen-diff-sample', family: 'p14-diffusion', order: 30, title: 'Sampling: from noise back to the moons',
    blurb: 'To make new points, start from pure noise and run the chain backwards. At each step t the denoiser predicts the noise in x_t; a little of it is removed, x is rescaled, and (except at the last step) a little fresh noise is added. A hundred small corrections turn a Gaussian cloud into two moons. The trained denoiser runs here live.',
    inputs: [
      { id: 'n', kind: 'number', label: 'Points', def: 200, min: 50, max: 300 },
      { id: 'seed', kind: 'number', label: 'Seed of the starting noise', def: 2024, min: 1, max: 9999 },
    ],
    presets: [{ label: '200 points', values: { n: 200, seed: 2024 } }, { label: 'Another start', values: { n: 200, seed: 77 } }, { label: 'Few points', values: { n: 60, seed: 5 } }],
    counters: [{ id: 'done', label: 'reverse steps done' }],
    run(inp) {
      const d = P.diff2d(), s = d.sched, T = s.T, J = P.D().diff2d, X = d.data.X, F = H().frames(['done']), q = F.q;
      const out = d.sample({ n: inp.n, seed: inp.seed, frames: 10 }), ax = plane('ds', 40, 50, 340, 3);
      const trace = Object.fromEntries(out.trace.map((st) => [st.t, st]));
      const data = dots(ax, 'ds:d', X.slice(0, 300), 'p14real', 2.2), chain = out.frames.map((f) => f.t);
      const timeline = (now) => chain.map((t, k) => ({ key: 'tl:' + t, kind: 'cell', x: 470 + k * 28, y: 60, w: 24, h: 24, rx: 4, text: '', cls: 'p14t' + (t === now ? ' act' : ''), title: 't = ' + t }));
      const tlab = [H().text('tl:a', 470, 108, 't = ' + T, 'tick', 12), H().text('tl:b', 470 + (chain.length - 1) * 28 + 24, 108, 't = 0', 'tick', 12, 'end'), H().text('tl:h', 470, 44, 'The reverse chain', 'strong', 14)];
      out.frames.forEach((fr, k) => {
        F.set('done', T - fr.t);
        const items = [...ax.items, ...data, ...dots(ax, 'ds:s', fr.points, 'p14fake', 2.8), ...timeline(fr.t), ...tlab, H().text('ds:h', 40, 24, 'Samples at t = ' + fr.t, 'strong', 14)];
        const near = S.Gen.chamfer(fr.points, X.slice(0, 600));
        let say;
        if (k === 0) say = `Start: ${q.n(inp.n)} points of pure Gaussian noise, x_${q.n(T)} ~ N(0, I). They lie a mean ${q(near, 2)} from the nearest data point.`;
        else {
          const st = trace[fr.t + 1] || trace[fr.t];
          say = `t = ${q.n(fr.t)}. In the step from t = ${q.n(st.t)} the denoiser predicted ε̂ for every point; the chain removed ${q.sg(st.k2, 3)} × ε̂, rescaled by 1/√α = ${q.sg(st.k1, 4)}` + (st.sigma ? ` and added fresh noise with σ = ${q.sg(st.sigma, 3)}` : ' and, at this last step, added no noise') + `. The points now lie a mean ${q(near, 2)} from the data.`;
          Object.assign(fr, { st });
        }
        F.push(items, say, Object.assign({ phase: k ? 'step' : 'start', t: fr.t, n: inp.n, near, T }, fr.st ? { k1: fr.st.k1, k2: fr.st.k2, sigma: fr.st.sigma, at: fr.st.t } : {}));
      });
      // how sampling quality grew during training (the stored snapshots), against plain noise
      const base = S.T.rng(inp.seed), noise = Array.from({ length: inp.n }, () => [base.normal(), base.normal()]), swdNoise = S.Gen.swd(noise, X.slice(0, inp.n)), swdNow = S.Gen.swd(out.final, X.slice(0, inp.n));
      const snaps = J.snaps.map((sn) => ({ step: sn.step, swd: sn.swd, pts: P.pairs(P.i16(sn.samples)) }));
      const items = [...ax.items, ...data, ...dots(ax, 'ds:s', out.final, 'p14fake', 2.8), H().text('ds:h', 40, 24, 'Samples at t = 0', 'strong', 14), H().text('sm:h', 470, 44, 'Samples during training', 'strong', 14)];
      snaps.forEach((sn, k) => { const a2 = plane('sm' + k, 470 + (k % 3) * 104, 70 + Math.floor(k / 3) * 128, 90, 3, { xticks: 0, yticks: 0, xname: '', yname: '' }); const inside = sn.pts.filter((p) => a2.inside(p[0], p[1])).length; if (!inside) items.push(H().text('sm' + k + ':o', 470 + (k % 3) * 104 + 45, 70 + Math.floor(k / 3) * 128 + 50, 'off the chart', 'tick', 12, 'middle')); items.push(...a2.items, ...dots(a2, 'sm' + k + ':p', sn.pts.slice(0, 120), 'p14fake', 1.6), H().text('sm' + k + ':t', 470 + (k % 3) * 104 + 45, 70 + Math.floor(k / 3) * 128 + 108, 'step ' + sn.step, 'tick', 12, 'middle')); });
      F.push(items, `Done. The sliced Wasserstein distance from these samples to the data is ${q(swdNow, 3)}, against ${q(swdNoise, 3)} for plain noise. During training the same distance for ${q.n(256)} samples went ${H().list(snaps.map((sn) => q.sg(sn.swd, 3)))} at steps ${H().list(snaps.map((sn) => q.n(sn.step)))}.`, { phase: 'end', swdNow, swdNoise, swds: snaps.map((sn) => sn.swd), steps: snaps.map((sn) => sn.step), n256: 256 });
      for (const f of F.list) f.narrowView = { x: 10, y: 0, w: 400, h: 480 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'One reverse step', text: ['ε̂ = denoiser(x_t, t)', 'x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε̂) / √α_t + σ_t · z', 'σ_t² = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t),  z ~ N(0, I)', 'no fresh noise at the last step'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will the finished samples be closer to the data than plain noise is (by sliced Wasserstein distance)?', options: ['yes', 'no'], answer: (fr) => { const m = fr[fr.length - 1].meta; return m.swdNow < m.swdNoise ? 'yes' : 'no'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'How many network evaluations does one sample cost with T = 100?', options: ['100, one per reverse step', '1', '2', '10 000'], answer: 0, why: 'Each step calls the denoiser once; this is why diffusion sampling is slow.' },
      { q: 'Why is fresh noise added at each reverse step except the last?', options: ['the reverse steps are draws from a distribution, not a deterministic map', 'to slow sampling down', 'to undo the denoiser', 'it is a bug'], answer: 0, why: 'Adding σ_t z keeps the samples spread like the data instead of collapsing to averages.' },
      { q: 'Where does sampling start?', options: ['pure noise from N(0, I)', 'a training point', 'the origin', 'the denoiser’s weights'], answer: 0, why: 'After T forward steps the data look like N(0, I), so that is where the reverse chain begins.' },
      { q: 'Early in sampling (t near T) the points move towards…', options: ['the middle of the data, since the denoiser cannot tell which part they belong to', 'the exact training points', 'infinity', 'the origin only'], answer: 0, why: 'Fine detail is decided only when little noise is left.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  P.diffImg = () => P.once('diffImg', () => { const J = P.D().diffImg; return { o: J.o, steps: J.steps, loss: J.loss, sample: J.sample || null, final64: J.final64 ? P.imgs(J.final64) : null, frames: J.frames.map((f) => ({ t: f.t, images: P.imgs(f.images) })), sched: S.Gen.schedule({ steps: J.o.steps, kind: J.o.kind || 'cosine' }) }; });
  P.stats = (px) => { let m = 0; for (const v of px) m += v; m /= px.length; let s = 0; for (const v of px) s += (v - m) ** 2; return { mean: m, sd: Math.sqrt(s / px.length) }; };

  S.topic({
    id: 'gen-diff-images', family: 'p14-diffusion', order: 40, title: 'Sampling 16×16 images',
    blurb: 'The same reverse chain on images: every one of the 256 pixels is a coordinate, the denoiser is a small convolutional network, and 60 steps with a cosine schedule try to turn eight squares of static into eight circles. Top: a real circle dissolved by the forward process, for comparison.',
    inputs: [{ id: 'pick', kind: 'number', label: 'Sample to follow (1–8)', def: 1, min: 1, max: 8 }],
    presets: [{ label: 'Sample 1', values: { pick: 1 } }, { label: 'Sample 4', values: { pick: 4 } }, { label: 'Sample 7', values: { pick: 7 } }],
    counters: [{ id: 'done', label: 'reverse steps done' }],
    run(inp) {
      const D = P.diffImg(), s = D.sched, T = s.T, F = H().frames(['done']), q = F.q, j = inp.pick - 1;
      const x0 = P.shapeImg(0), g = S.T.rng(31), e = x0.map(() => g.normal()), fwdT = [0, 10, 20, 30, 40, 50, 60].filter((t) => t <= T);
      const top = [H().text('fw:h', 40, 24, 'Forward: a real circle dissolving', 'strong', 14), ...fwdT.flatMap((t, k) => [P.img('fw:' + t, 40 + k * 70, 36, 56, P.toPix(S.Gen.noise(x0, t, s, e))), H().text('fw:t' + t, 68 + k * 70, 108, 't = ' + t, 'tick', 12, 'middle')])];
      const timeline = (now) => D.frames.map((f, k) => ({ key: 'tl:' + f.t, kind: 'cell', x: 560 + k * 21, y: 160, w: 18, h: 18, rx: 3, text: '', cls: 'p14t' + (f.t === now ? ' act' : '') }));
      D.frames.forEach((fr, k) => {
        F.set('done', T - fr.t);
        const items = [...top, H().text('rv:h', 40, 150, 'Reverse: eight samples at t = ' + fr.t, 'strong', 14), ...timeline(fr.t), H().text('tl:a', 560, 196, 't = ' + T, 'tick', 12), H().text('tl:b', 560 + (D.frames.length - 1) * 21 + 18, 196, 't = 0', 'tick', 12, 'end')];
        fr.images.forEach((im, i) => items.push(P.img('rv:' + i, 40 + (i % 4) * 110, 170 + Math.floor(i / 4) * 110, 96, im, i === j ? 'act' : '')));
        const st = P.stats(fr.images[j]), ab = s.alphaBar[fr.t], like = P.corr(fr.images[j], D.frames[D.frames.length - 1].images[j]);
        const say = k === 0
          ? `Start: eight images of pure noise, ${q.n(256)} pixels each. Sample ${q.n(inp.pick)} has mean brightness ${q(st.mean, 2)} and spread ${q(st.sd, 2)}.`
          : `t = ${q.n(fr.t)}: ᾱ = ${q.sg(ab, 3)}, so at this step the chain still treats the images as ${q.pct(Math.sqrt(1 - ab))} noise by scale. Sample ${q.n(inp.pick)} now has mean brightness ${q(st.mean, 2)} and spread ${q(st.sd, 2)}` + (fr.t === 0 ? ': a clean shape on a plain ground.' : '.');
        F.push(items, say, { phase: k ? 'step' : 'start', t: fr.t, T, ab, noise: Math.sqrt(1 - ab), mean: st.mean, sd: st.sd, like, P: 256, pick: inp.pick });
      });
      // the dark area of each finished sample, against the range of the training circles
      const area = (px) => Array.from(px).filter((v) => v > 0.5).length, circles = P.once('circles64', () => { const set = S.Gen.images('shapes', { n: 64, size: 16, seed: 1, classes: [0] }); return [...Array(set.n).keys()].map((i) => area(set.X.subarray(i * 256, (i + 1) * 256))); });
      const lo = Math.min(...circles), hi = Math.max(...circles), areas = D.frames[D.frames.length - 1].images.map(area), good = areas.filter((a) => a >= lo && a <= hi).length;
      const small = areas.filter((a) => a < lo).length, big = areas.filter((a) => a > hi).length, more = D.final64 ? D.final64.map(area) : null, good64 = more ? more.filter((a) => a >= lo && a <= hi).length : null;
      // (the quoted numbers are made in the order they are read)
      const blank = areas.filter((a) => a < 5).length, faint = small - blank;
      const why = () => { const w = [blank ? q.n(blank) + ' faded to a blank ground' : '', faint ? q.n(faint) + ' came out too faint or too small' : '', big ? q.n(big) + ' too large or ran together' : ''].filter(Boolean); return w.length ? '; ' + H().list(w) : ''; };
      const temp = D.sample && D.sample.temp && D.sample.temp !== 1 ? D.sample.temp : null;
      const endItems = [...F.last().items];
      if (more) { endItems.push(H().text('m64:h', 560, 236, 'Another 64 samples', 'strong', 14)); D.final64.forEach((im, i) => endItems.push(P.img('m64:' + i, 560 + (i % 8) * 29, 248 + Math.floor(i / 8) * 29, 26, im, 'tile'))); }
      F.push(endItems, `After ${q.n(T)} reverse steps the samples have dark areas of ${H().list(areas.map((a) => q.n(a)))} pixels. The ${q.n(circles.length)} training circles cover from ${q.n(lo)} to ${q.n(hi)}, and ${q.n(good)} of the ${q.n(8)} samples fall in that range` + why() + '.' + (more ? ` Of ${q.n(more.length)} more samples from fresh noise, ${q.n(good64)} do.` : '') + ` The denoiser was trained for ${q.n(D.steps)} steps on ${q.n(D.o.n || 256)} circles; its loss ended near ${q.sg(D.loss[D.loss.length - 1], 3)}.` + (temp ? ` The fresh noise of each reverse step was scaled by ${q(temp, 1)}, a lower temperature than the formula for σ gives.` : ''),
        { phase: 'end', T, areas, lo, hi, nc: circles.length, good, small, blank, faint, big, eight: 8, steps: D.steps, nTrain: D.o.n || 256, loss: D.loss[D.loss.length - 1], n64: more ? more.length : null, good64, temp });
      for (const f of F.list) f.narrowView = { x: 20, y: 130, w: 470, h: 360 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The image denoiser', text: ['input: x_t (16×16) and t (as 4 extra channels)', 'convolutions 3×3, dilated 1, 2 and 4, with a skip', 'output: ε̂, 16×16', D.sample && D.sample.temp ? 'sampling: fresh noise scaled by ' + D.sample.temp : '', D.o.ema ? 'sampling uses a moving average of the weights (' + D.o.ema + ' per step)' : '', 'the reverse step goes through x̂₀, clipped to [−1, 1]'].filter(Boolean).join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'When will the followed sample first look like its finished self (correlation 0.8 or more)?', options: ['in the first third of the chain', 'in the middle third', 'in the last third'], answer: (fr) => { const T = fr[0].meta.T, f = fr.find((x) => x.meta.like >= 0.8), t = f ? f.meta.t : 0; return t > (2 * T) / 3 ? 'in the first third of the chain' : t > T / 3 ? 'in the middle third' : 'in the last third'; }, explain: (fr) => { const f = fr.find((x) => x.meta.like >= 0.8); return f ? f.say : fr[fr.length - 1].say; } },
    ],
    quiz: [
      { q: 'How many numbers does the reverse chain update at each step for one 16×16 image?', options: ['256', '16', '1', '60'], answer: 0, why: 'Every pixel is a coordinate of x.' },
      { q: 'Why is a convolutional denoiser better than a fully connected one here?', options: ['the same local pattern (an edge, a curve) is handled the same way everywhere in the image', 'it has more parameters', 'it is faster to sample', 'it needs no t'], answer: 0, why: 'Convolutions share weights across positions, which suits images.' },
      { q: 'Why clip the predicted clean image to [−1, 1] in each step?', options: ['the pixels of real images lie there, so wild guesses are reined in', 'to add noise', 'to speed up training', 'to change the schedule'], answer: 0, why: 'It keeps the chain inside the range of the data.' },
      { q: 'With T = 60, how many denoiser calls make 8 images?', options: ['60, each on a batch of 8', '8', '480 separate calls are needed', '1'], answer: 0, why: 'All eight move together, one batched call per step.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
