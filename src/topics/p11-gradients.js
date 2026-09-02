// Part 11, vanishing and exploding gradients: a recurrent cell of eight numbers whose state-to-state matrix is a
// rotation scaled by a factor, run forward over a stream of random inputs; the loss is read at the last step and
// the engine's backward pass gives the gradient reaching every earlier state. On a log scale the gradients fall or
// rise in a straight line: each step back multiplies them by about the same factor.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml;
  const HID = 8;
  const SUP = (v) => [...String(v)].map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)] || c).join('');

  /** a rotation (orthogonal matrix) from a seeded Gram–Schmidt */
  const rotation = (n, g) => {
    const Q = [];
    while (Q.length < n) {
      let v = Array.from({ length: n }, () => g.normal());
      for (const u of Q) { const p = u.reduce((s, x, j) => s + x * v[j], 0); v = v.map((x, j) => x - p * u[j]); }
      const L = Math.hypot(...v);
      if (L > 1e-6) Q.push(v.map((x) => x / L));
    }
    return Q;
  };
  /** forward and backward through n steps; the gradient of L = Σ h_n with respect to every state h_t */
  P.gradRun = (o) => P.once('grad:' + JSON.stringify(o), () => {
    const T = S.T, g = T.rng(o.seed), Q = rotation(HID, g), Whm = Q.map((r) => r.map((v) => v * o.scale));
    const Wx = T.tensor([Array.from({ length: HID }, () => 0.5 * g.normal())]), xs = Array.from({ length: o.steps }, () => g.normal());
    const Wh = T.param(T.tensor(Whm));
    let h = T.param(T.zeros([1, HID]));
    const hs = [];
    T.enableGrad(() => {
      for (let t = 0; t < o.steps; t++) {
        const pre = T.add(T.scale(Wx, xs[t]), T.matmul(h, Wh));
        h = o.act === 'tanh' ? T.tanh(pre) : pre;
        hs.push(h);
      }
      T.sum(h).backward();
    });
    const norm = (t) => Math.hypot(...Array.from(t.values()));
    const hn = hs.map(norm), gn = hs.map((t) => (t.grad ? norm(t.grad) : 0));
    return { Wh: Whm, xs, hn, gn, Wx: Array.from(Wx.values()) };
  });

  S.topic({
    id: 'rnn-gradients', family: 'p11-rnn', order: 20, title: 'Vanishing and exploding gradients',
    blurb: 'To learn from something that happened many steps ago, the error signal must travel back through every step in between. Each step back multiplies it by the state-to-state weights and by the slope of the activation. If those shrink it, a little each time, it fades away to nothing; if they stretch it, it blows up. Here the stretch of the weights is a dial you can turn, and the engine’s backward pass shows what reaches each step.',
    inputs: [
      { id: 'scale', kind: 'slider', label: 'Stretch of W_h', def: 0.9, min: 0.5, max: 1.5, step: 0.05, fmt: (v) => H().fx(v, 2) },
      { id: 'act', kind: 'choice', label: 'Activation', def: 'tanh', options: [{ value: 'tanh', label: 'tanh' }, { value: 'linear', label: 'none (linear)' }] },
      { id: 'steps', kind: 'number', label: 'Steps', def: 30, min: 10, max: 40 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'tanh, stretch 0.9: fades', values: { scale: 0.9, act: 'tanh', steps: 30, seed: 1 } },
      { label: 'tanh, stretch 1.5', values: { scale: 1.5, act: 'tanh', steps: 30, seed: 1 } },
      { label: 'linear, stretch 0.8', values: { scale: 0.8, act: 'linear', steps: 30, seed: 2 } },
      { label: 'linear, stretch 1.2: explodes', values: { scale: 1.2, act: 'linear', steps: 30, seed: 2 } },
      { label: 'linear, stretch 1: steady', values: { scale: 1, act: 'linear', steps: 20, seed: 3 } },
    ],
    counters: [{ id: 'back', label: 'steps back' }],
    run(inp) {
      const F = H().frames(['back']), q = F.q, n = inp.steps, r = P.gradRun({ scale: inp.scale, act: inp.act, steps: n, seed: inp.seed });
      const back = (k) => r.gn[n - 1 - k];
      const logs = r.gn.map((v) => Math.log10(Math.max(v, 1e-30)));
      const refs = (inp.act === 'tanh' ? [0.5, 1, 1.5] : [0.8, 1, 1.2]).map((s) => ({ s, run: P.gradRun({ scale: s, act: inp.act, steps: n, seed: inp.seed }) }));
      const allLogs = logs.concat(...refs.map((x) => x.run.gn.map((v) => Math.log10(Math.max(v, 1e-30)))));
      let lo = Math.floor(Math.min(...allLogs)), hi = Math.ceil(Math.max(...allLogs));
      if (hi - lo < 2) { lo -= 1; hi += 1; }
      const ax = S.Kit.curve.axes({ prefix: 'gr:ax', x: 380, y: 60, w: 390, h: 250, xr: [0, n - 1], yr: [lo, hi], xticks: n > 20 ? 10 : 5, yticks: hi - lo > 12 ? 4 : hi - lo > 6 ? 2 : 1, fmtY: (v) => '10' + SUP(v), xname: 'steps back from the end', yname: 'size of the gradient |∂L/∂h|', grid: true });
      const curve = (upto) => S.Kit.curve.line(ax, 'gr:c', logs.slice(n - 1 - upto).reverse().map((v, k) => [k, v]), 'p11a');
      const whGrid = P.grid('gr:wh', 10, 84, r.Wh, { cw: 36, ch: 26, scheme: 'div', max: Math.max(1, inp.scale), fmt: (v) => H().fx(v, 1), size: 12, rows: null, cols: null, title: 'W_h: a rotation × ' + H().fx(inp.scale, 2) });
      const panel = (rows) => H().lines('gr:p', 0, 380, rows, 13, 21);
      const rule = inp.act === 'tanh' ? 'each step back: × W_h, then × tanh′ (at most 1)' : 'each step back: × W_h, and nothing else';
      F.push([...whGrid, ...ax.items, ...panel([['h_t = ' + (inp.act === 'tanh' ? 'tanh(' : '(') + 'x_t W_x + h_{t−1} W_h)', 'math'], ['loss L = sum of the last state h_' + n, 'legend'], [rule, 'legend']])],
        `The cell has ${P.qpl(q, HID, 'hidden number')} and runs for ${P.qpl(q, n, 'step')} over random inputs. Its matrix W_h is a rotation scaled by ${q(inp.scale, 2)}: it turns the state and stretches every direction by exactly that much. The loss is read at the last step only.`, { phase: 'setup', H: HID, n, scale: inp.scale });
      // backward, a chunk at a time
      const chunks = [0, Math.round(n / 4), Math.round(n / 2), Math.round((3 * n) / 4), n - 1];
      chunks.forEach((k, ci) => {
        F.set('back', k);
        const items = [...whGrid, ...ax.items, curve(k), S.Kit.curve.dot(ax, 'gr:d', k, logs[n - 1 - k], 'p11a')];
        if (ci === 0) F.push(items.concat(panel([['∂L/∂h_' + n + ' = [1, 1, …, 1]', 'math'], ['size √' + HID + ' = ' + H().fx(back(0), 2), 'legend']])),
          `Backpropagation starts at the end: the loss is the sum of the last state, so its gradient there is a vector of ones, of size ${q(back(0), 2)}.`, { phase: 'back', k, g: back(0) });
        else {
          const ratio = Math.pow(back(k) / back(0), 1 / k);
          F.push(items.concat(panel([[k + ' steps back: |∂L/∂h| = ' + H().sg(back(k), 2), 'strong'], ['on average × ' + H().fx(ratio, 3) + ' per step', 'legend']])),
            `${q.n(k)} steps back the gradient has size ${q.sg(back(k), 2)}: on average each step multiplied it by ${q(ratio, 3)}.` + (ci === chunks.length - 1 ? (ratio < 0.97 ? ' It has all but vanished: a mistake at the end cannot teach the first steps anything.' : ratio > 1.03 ? ' It has exploded: one update this size would wreck the weights.' : ' It stays about the same size all the way back.') : ''), { phase: 'back', k, g: back(k), g0: back(0), ratio });
        }
      });
      // three stretches side by side
      // a reference stretch equal to the user's own is already drawn as theirs: no second line or legend entry for it
      const dup = (x) => Math.abs(x.s - inp.scale) < 1e-9;
      const lines = refs.map((x, i) => (dup(x) ? null : S.Kit.curve.line(ax, 'gr:r' + i, x.run.gn.slice().reverse().map((v, k) => [k, Math.log10(Math.max(v, 1e-30))]), ['p11c', 'p11b', 'p11d'][i]))).filter(Boolean);
      const ends = refs.map((x) => x.run.gn[0]);
      F.push([...whGrid, ...ax.items, ...lines, curve(n - 1), ...S.Kit.curve.legend('gr:lg', 10, 350, [{ text: 'stretch ' + H().fx(inp.scale, 2) + ' (yours)', cls: 'p11a' }].concat(refs.map((x, i) => (dup(x) ? null : { text: 'stretch ' + H().fx(x.s, 2), cls: ['p11c', 'p11b', 'p11d'][i] })).filter(Boolean)), { size: 12, dir: 'col' })],
        `The same cell with stretches ${H().list(refs.map((x) => q(x.s, 2)))}: after ${P.qpl(q, n - 1, 'step')} back the gradient is ${H().list(ends.map((v) => q.sg(v, 2)))}. On a log scale each is close to a straight line: the size changes by about the same factor every step, so over many steps it fades or explodes.` + (inp.act === 'tanh' ? ' With tanh even a stretch of one fades, because its slope is below one wherever the state is not zero.' : ''),
        { phase: 'compare', ends, scales: refs.map((x) => x.s), n1: n - 1 });
      F.push([...whGrid, ...ax.items, ...lines, curve(n - 1), ...panel([['exploding: clip the gradient’s size before each update', 'legend'], ['vanishing: gated cells (LSTM, GRU) keep a path where the', 'legend'], ['signal is carried along without being squashed', 'legend']])],
        `The two cures: exploding gradients are tamed by clipping, scaling the whole gradient down whenever its size passes a limit. Vanishing ones need a different cell: [[lstm-gates|the LSTM]] carries a separate memory that is updated by adding, so its gradient can pass back many steps.`, { phase: 'cures' });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The gradient through time', text: ['h_t = f(x_t W_x + h_{t−1} W_h)', '∂L/∂h_{t−1} = (∂L/∂h_t ⊙ f′(·)) W_hᵀ', 'k steps back: about (stretch × slope)^k', 'below 1: vanishes;  above 1: explodes'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'back' && f.meta.k === 0, once: true, pick: 'option',
      ask: (fr) => 'Going back to the first step, will the gradient be smaller, about the same, or larger than at the end?',
      options: ['much smaller', 'about the same', 'much larger'],
      answer: (fr) => { const b = fr.filter((f) => f.meta.phase === 'back'), last = b[b.length - 1].meta; return last.g < last.g0 / 3 ? 'much smaller' : last.g > last.g0 * 3 ? 'much larger' : 'about the same'; },
      explain: (fr) => { const b = fr.filter((f) => f.meta.phase === 'back'); return b[b.length - 1].say; },
    }],
    quiz: [
      { q: 'A gradient is multiplied by 0.9 at each step back. After 50 steps it is about…', options: ['0.005 of its size', '0.9 of its size', '45% of its size', 'unchanged'], answer: 0, why: '0.9⁵⁰ ≈ 0.005.' },
      { q: 'Which fixes exploding gradients?', options: ['clipping the gradient’s size', 'a longer sequence', 'removing the bias', 'a larger learning rate'], answer: 0, why: 'Clipping caps the size of each update.' },
      { q: 'Why does tanh make gradients fade even when W_h stretches by exactly 1?', options: ['its slope is at most 1 and below 1 away from zero', 'it is not differentiable', 'it adds noise', 'it doubles the state'], answer: 0, why: 'Every step back multiplies by tanh′ as well.' },
      { q: 'What does a vanishing gradient mean for learning?', options: ['early steps barely learn from errors made later', 'the model trains faster', 'the loss becomes negative', 'the weights explode'], answer: 0, why: 'Almost no signal reaches them.' },
      { q: 'On a log scale, a gradient that shrinks by the same factor every step back looks like…', options: ['a straight line sloping down', 'a parabola', 'a flat line', 'a step'], answer: 0, why: 'log(c^k) = k log c is linear in k.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
