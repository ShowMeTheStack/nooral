// Part 11, a recurrent network unrolled through time: one small cell (4 hidden numbers) trained to keep a running
// count of the ones in a stream of bits, drawn as a row of copies of itself, one per step, each passing its hidden
// state to the next. Every step's arithmetic is shown with the trained weights.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11;
  const CH = { x: 40, y: 170, bw: 66, bh: 70, gap: 20 };

  /** the trained counting RNN run on a bit string: its weights, every step's pieces and outputs */
  P.rnnTrace = (bits) => {
    const net = P.net('rnnCount'), T = S.T, n = bits.length;
    const logits = net.predict(T.tensor([bits.map((b) => [b])])), L = logits.values(), C = logits.shape[2];
    const Wx = Array.from(net.cell.Wx.values()), Wh = net.cell.Wh.toArray(), b = Array.from(net.cell.b.values()), Hn = b.length;
    const hs = net.cell.steps.map((s) => Array.from(s.h.values()));
    const steps = bits.map((x, t) => {
      const prev = t ? hs[t - 1] : new Array(Hn).fill(0), xw = Wx.map((w) => w * x), hw = b.map((_, j) => prev.reduce((s, v, i) => s + v * Wh[i][j], 0));
      const pre = b.map((bj, j) => xw[j] + hw[j] + bj), row = Array.from(L.subarray(t * C, (t + 1) * C)), p = P.softmax(row), k = P.argmax(p);
      return { x, prev, xw, hw, pre, h: hs[t], logits: row, p, count: k, pk: p[k] };
    });
    return { net, Wx, Wh, b, steps, C };
  };

  S.topic({
    id: 'rnn-unrolled', family: 'p11-rnn', order: 10, title: 'A recurrent network unrolled through time',
    blurb: 'A recurrent network reads a sequence one item at a time and keeps a hidden state: a few numbers that summarise what it has seen so far. At every step the same cell, with the same weights, mixes the new input with the old state. Drawn out step by step, it is a deep network whose layers all share one set of weights. This one has learned to count the ones in a stream of bits.',
    inputs: [{ id: 'bits', kind: 'string', label: 'Bits (3 to 8)', def: '10110110', alphabet: '01', min: 3, max: 8 }],
    presets: [
      { label: 'Mixed bits', values: { bits: '10110110' } },
      { label: 'All ones', values: { bits: '11111111' } },
      { label: 'Zeros, then ones', values: { bits: '00001111' } },
      { label: 'A short one', values: { bits: '101' } },
    ],
    counters: [{ id: 'steps', label: 'steps read' }],
    run(inp) {
      const F = H().frames(['steps']), q = F.q, bits = [...inp.bits].map(Number), n = bits.length, tr = P.rnnTrace(bits), job = P.job('rnnCount');
      const ev = job.evals[job.evals.length - 1], nPar = tr.net.numParams();
      const truth = bits.map((_, t) => bits.slice(0, t + 1).reduce((a, b) => a + b, 0));
      const strip = (prefix, x, y, v, name) => K().strip({ prefix, x, y, vals: v, cw: 46, ch: 24, scheme: 'div', max: 1, fmt: (u) => H().fx(u, 2), name, size: 11 }).items;
      const inner = (t, b) => K().strip({ prefix: 'rn:hv' + t, x: b.x + 7, y: b.y + 34, vals: tr.steps[t].h, cw: 11.5, ch: 22, scheme: 'div', max: 1, fmt: null, gap: 1.5 }).items;
      const chain = (upto, outCls) => K().chain(Object.assign({ prefix: 'rn:c', n, upto, inputs: bits.map(String), outputs: tr.steps.map((s) => String(s.count)), inner, cls: (t) => (t === upto - 1 ? 'act' : t < upto ? 'done' : ''), inCls: (t) => (t === upto - 1 ? 'new' : ''), outCls, label: (t) => 'step ' + (t + 1) }, CH)).items;
      const panel = (rows) => H().lines('rn:p', 0, 352, rows, 13, 21);
      // the weights
      const wy = 150;
      F.push([...P.grid('rn:wx', 60, wy, [tr.Wx], { cw: 46, ch: 26, scheme: 'div', max: 2, rows: ['x'], cols: ['1', '2', '3', '4'], title: 'W_x (input → state)' }),
        ...P.grid('rn:wh', 300, wy, tr.Wh, { cw: 46, ch: 26, scheme: 'div', max: 2, rows: ['h1', 'h2', 'h3', 'h4'], cols: ['1', '2', '3', '4'], title: 'W_h (state → state)' }),
        ...P.grid('rn:b', 540, wy, [tr.b], { cw: 46, ch: 26, scheme: 'div', max: 2, rows: ['b'], cols: ['1', '2', '3', '4'], title: 'bias b' }),
        ...panel([['h_t = tanh(x_t · W_x + h_{t−1} · W_h + b)', 'math'], ['the same W_x, W_h and b at every step', 'legend'], ['an output layer turns h_t into a count, 0 to ' + (tr.C - 1), 'legend']])],
        `The whole network: a hidden state of ${P.qpl(q, tr.b.length, 'number')}, an input weight row W_x, a square state-to-state matrix W_h and a bias, plus an output layer, ${P.qpl(q, nPar, 'parameter')} in all. It was trained on random bits for ${P.qpl(q, job.steps, 'step')} and counts ${q.pct(ev.accuracy)} of the steps of new streams right.`, { phase: 'weights', H: tr.b.length, nPar, steps: job.steps, acc: ev.accuracy });
      tr.steps.forEach((s, t) => {
        F.count('steps');
        const rows = [['step ' + (t + 1) + ':  x = ' + s.x, 'strong'],
          ['x · W_x        = ' + P.vec(s.xw), 'math'], ['h_prev · W_h   = ' + P.vec(s.hw), 'math'], ['+ b → tanh  h  = ' + P.vec(s.h), 'math'],
          ['count so far: ' + s.count + '  (p = ' + H().fx(s.pk, 2) + ')', 'strong']];
        const first = t === 0;
        F.push([...chain(t + 1), ...panel(rows)],
          `Step ${q.n(t + 1)} reads x = ${q.n(s.x)}. ` + (first ? `The state starts at zero, so only the input and the bias count: ` : `The old state through W_h gives ${P.qvec(q, s.hw)}; with the input and the bias that makes `) + `${P.qvec(q, s.pre)}, and tanh squashes it to h = ${P.qvec(q, s.h)}. The output layer reads h and gives the count so far as ${q.n(s.count)} (probability ${q(s.pk, 2)}).`,
          { phase: 'step', t: t + 1, x: s.x, hw: s.hw, pre: s.pre, h: s.h, count: s.count, pk: s.pk });
      });
      const right = tr.steps.filter((s, t) => s.count === truth[t]).length;
      F.push([...chain(n, (t) => (tr.steps[t].count === truth[t] ? 'p11right' : 'p11wrong')), ...panel([['unrolled: ' + n + ' copies of one cell', 'strong'], ['a deep network whose layers share weights', 'legend'], ['training sends the error back through every step', 'legend']])],
        `All ${P.qpl(q, n, 'step')} read: ${q.n(right)} of the ${q.n(n)} running counts are right (green). Unrolled, the network is ${P.qpl(q, n, 'layer')} deep, every layer the same cell; training sends the error back through all of them, which is where [[rnn-gradients|gradients fade or explode]].`, { phase: 'done', right, n });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A simple recurrent cell', text: ['h_0 = 0', 'h_t = tanh(x_t W_x + h_{t−1} W_h + b)', 'y_t = softmax(h_t W_o + b_o)', 'the same W_x, W_h, b at every step t'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => f.meta.phase === 'step' && f.meta.t === 2 && fr[i + 1] && fr[i + 1].meta.phase === 'step', once: true, pick: 'option',
      ask: (fr, i) => 'The next bit is ' + fr[i + 1].meta.x + '. What count will the network say after it?',
      options: (fr, i) => { const c = fr[i + 1].meta.count; return [...new Set([c - 1, c, c + 1].filter((v) => v >= 0))].map(String); },
      answer: (fr, i) => String(fr[i + 1].meta.count),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'What does an RNN carry from one step to the next?', options: ['its hidden state h', 'the whole input so far', 'new weights', 'the loss'], answer: 0, why: 'h summarises everything read so far in a fixed number of values.' },
      { q: 'Unrolled over 8 steps, how many different weight matrices W_h does the network use?', options: ['one, shared by all steps', 'eight', 'sixteen', 'none'], answer: 0, why: 'The same cell is applied at every step.' },
      { q: 'Why can one small cell read sequences of any length?', options: ['it is applied again at every step with the same weights', 'it grows a new layer per step', 'it reads the whole sequence at once', 'it ignores long sequences'], answer: 0, why: 'Sharing the weights over time makes the length free.' },
      { q: 'At the first step, h_{t−1} is…', options: ['all zeros', 'random', 'the last state of the previous sequence', 'the input'], answer: 0, why: 'The state starts empty.' },
      { q: 'Training an unrolled RNN of 50 steps is like training…', options: ['a 50-layer network whose layers share weights', 'a single layer', '50 separate networks', 'a decision tree'], answer: 0, why: 'Backpropagation goes back through every step: backpropagation through time.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
