// Part 11, the LSTM cell step by step: the same counting task as the plain recurrent network, now with a cell of
// four memory numbers and four gates. Every step shows the gates' values (input i, forget f, candidate g, output o)
// and the memory update c = f·c + i·g for the memory number that does the counting.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11;
  const CH = { x: 40, y: 92, bw: 66, bh: 76, gap: 20 };

  /** the trained counting LSTM on a bit string: gates and states per step, as plain arrays [t][unit] */
  P.lstmTrace = (bits) => {
    const net = P.net('lstmCount'), T = S.T, n = bits.length;
    const logits = net.predict(T.tensor([bits.map((b) => [b])])), L = logits.values(), C = logits.shape[2], G = net.gates();
    const take = (name) => { const t = G[name], Hn = t.shape[2], v = t.values(); return Array.from({ length: n }, (_, k) => Array.from(v.subarray(k * Hn, (k + 1) * Hn))); };
    const i = take('gate.i'), f = take('gate.f'), g = take('gate.g'), o = take('gate.o'), c = take('cell'), h = net.cell.steps.map((s) => Array.from(s.h.values()));
    const out = bits.map((_, t) => { const p = P.softmax(Array.from(L.subarray(t * C, (t + 1) * C))), k = P.argmax(p); return { count: k, pk: p[k] }; });
    // the memory number that moves most over the stream: the counter
    const Hn = c[0].length, range = Array.from({ length: Hn }, (_, u) => Math.max(...c.map((r) => r[u])) - Math.min(0, ...c.map((r) => r[u])));
    return { net, i, f, g, o, c, h, out, unit: P.argmax(range), Hn };
  };
  /** a small table of gate rows: a name, then one heat cell per hidden unit */
  P.gateTable = (prefix, x, y, rows, o) => {
    o = o || {};
    const items = [], cw = o.cw || 50, ch = o.ch || 24;
    rows.forEach((r, k) => items.push(...K().strip({ prefix: prefix + k, x, y: y + k * (ch + 4), vals: r.v, cw, ch, scheme: r.scheme || 'seq', max: r.max || 1, fmt: (v) => H().fx(v, 2), name: r.name, size: 12, cls: (j) => (j === o.unit ? 'act' : '') }).items));
    for (let j = 0; j < rows[0].v.length; j++) items.push(H().text(prefix + ':u' + j, x + j * (cw + 2) + cw / 2, y - 8, (o.head || 'slot ') + (j + 1), 'tick', 12, 'middle'));
    return items;
  };

  S.topic({
    id: 'lstm-gates', family: 'p11-rnn', order: 30, title: 'LSTM: gates that decide what to keep',
    blurb: 'An LSTM keeps two things from step to step: a memory c and an output h. Three gates, each a number between 0 and 1 per memory slot, decide what happens: the forget gate f how much of the old memory to keep, the input gate i how much of the new candidate g to write in, and the output gate o how much of the memory to show. Because the memory is updated by adding, it can hold a value for a long time. Watch a trained cell count ones with it.',
    inputs: [{ id: 'bits', kind: 'string', label: 'Bits (3 to 8)', def: '10011101', alphabet: '01', min: 3, max: 8 }],
    presets: [
      { label: 'Mixed bits', values: { bits: '10011101' } },
      { label: 'All ones', values: { bits: '11111111' } },
      { label: 'Ones after a gap', values: { bits: '10000001' } },
      { label: 'Three bits', values: { bits: '011' } },
    ],
    counters: [{ id: 'steps', label: 'steps read' }],
    run(inp) {
      const F = H().frames(['steps']), q = F.q, bits = [...inp.bits].map(Number), n = bits.length, tr = P.lstmTrace(bits), u = tr.unit, job = P.job('lstmCount');
      const ev = job.evals[job.evals.length - 1], truth = bits.map((_, t) => bits.slice(0, t + 1).reduce((a, b) => a + b, 0));
      const cmax = Math.max(1, ...tr.c.flat().map(Math.abs));
      const inner = (t, b) => [...K().strip({ prefix: 'ls:cv' + t, x: b.x + 7, y: b.y + 28, vals: tr.c[t], cw: 11.5, ch: 16, scheme: 'div', max: cmax, fmt: null, gap: 1.5 }).items, H().text('ls:cn' + t, b.x + 7, b.y + 56, 'c', 'tick', 11), ...K().strip({ prefix: 'ls:hv' + t, x: b.x + 7, y: b.y + 50, vals: tr.h[t], cw: 11.5, ch: 16, scheme: 'div', max: 1, fmt: null, gap: 1.5 }).items].filter((it) => it.key !== 'ls:cn' + t);
      const chain = (upto, outCls) => K().chain(Object.assign({ prefix: 'ls:c', n, upto, inputs: bits.map(String), outputs: tr.out.map((s) => String(s.count)), inner, label: (t) => 'step ' + (t + 1), cls: (t) => (t === upto - 1 ? 'act' : t < upto ? 'done' : ''), inCls: (t) => (t === upto - 1 ? 'new' : ''), outCls }, CH)).items;
      const panel = (rows) => H().lines('ls:p', 330, 290, rows, 13, 21);
      F.push([...chain(0), ...panel([['f = σ(·)  forget: keep the old memory', 'math'], ['i = σ(·)  input: write the candidate', 'math'], ['g = tanh(·)  the candidate value', 'math'], ['o = σ(·)  output: show the memory', 'math'], ['c = f ⊙ c_prev + i ⊙ g', 'strong'], ['h = o ⊙ tanh(c)', 'strong']]), H().text('ls:k', 40, 262, 'in each box: memory c (top row), output h (bottom row)', 'legend', 12)],
        `The LSTM reads the same kind of bit stream as the plain network, with ${P.qpl(q, tr.Hn, 'memory slot')}. Each gate is a sigmoid of the input and the previous output, a value between 0 (shut) and 1 (open), one per slot. Trained for ${P.qpl(q, job.steps, 'step')}, it counts ${q.pct(ev.accuracy)} of the steps of new streams right.`, { phase: 'intro', Hn: tr.Hn, steps: job.steps, acc: ev.accuracy });
      for (let t = 0; t < n; t++) {
        F.count('steps');
        const cp = t ? tr.c[t - 1][u] : 0, fv = tr.f[t][u], iv = tr.i[t][u], gv = tr.g[t][u], cv = tr.c[t][u];
        const table = P.gateTable('ls:g', 70, 300, [{ name: 'f', v: tr.f[t] }, { name: 'i', v: tr.i[t] }, { name: 'g', v: tr.g[t], scheme: 'div' }, { name: 'o', v: tr.o[t] }, { name: 'c', v: tr.c[t], scheme: 'div', max: cmax }], { unit: u, cw: 52, ch: 22 });
        const rows = [['step ' + (t + 1) + ', x = ' + bits[t] + ', slot ' + (u + 1) + ' (the counter):', 'strong'], ['c = f·c_prev + i·g', 'math'], ['  = ' + H().fx(fv, 2) + ' × ' + H().fx(cp, 2) + ' + ' + H().fx(iv, 2) + ' × ' + H().fx(gv, 2), 'math'], ['  = ' + H().fx(cv, 2), 'math'], ['count so far: ' + tr.out[t].count + ' (p = ' + H().fx(tr.out[t].pk, 2) + ')', 'strong']];
        const what = bits[t] ? (iv > 0.5 ? 'The input gate is open' : 'The input gate is mostly shut') : iv < 0.5 ? 'The input gate stays mostly shut' : 'The input gate is open';
        F.push([...chain(t + 1), ...table, ...panel(rows), H().text('ls:k', 40, 262, 'in each box: memory c (top row), output h (bottom row)', 'legend', 12)],
          `Step ${q.n(t + 1)} reads ${q.n(bits[t])}. In slot ${q.n(u + 1)} the forget gate is ${q(fv, 2)}, so the memory ${q(cp, 2)} is ${fv > 0.9 ? 'kept almost whole' : 'partly forgotten'}. ${what} (${q(iv, 2)}) and the candidate is ${q(gv, 2)}, so the memory becomes ${q(cv, 2)}. The count so far is ${q.n(tr.out[t].count)}.`,
          { phase: 'step', t: t + 1, x: bits[t], unit: u, slot: u + 1, f: fv, i: iv, g: gv, cPrev: cp, c: cv, count: tr.out[t].count, pk: tr.out[t].pk });
      }
      // the counter slot over time against the true count
      const cs = tr.c.map((r) => r[u]), ax = S.Kit.curve.axes({ prefix: 'ls:ax', x: 360, y: 290, w: 400, h: 140, xr: [1, n], yr: [Math.min(0, ...cs), Math.max(...cs, ...truth, 0.5) * 1.1], xticks: 1, yticks: 3, xname: 'step', yname: 'memory slot ' + (u + 1), grid: false });
      const right = tr.out.filter((s, t) => s.count === truth[t]).length;
      const per = [];
      for (let t = 0; t < n; t++) per.push(cs[t] - (t ? cs[t - 1] : 0));
      const up1 = per.filter((_, t) => bits[t] === 1), up0 = per.filter((_, t) => bits[t] === 0), mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
      F.push([...chain(n, (t) => (tr.out[t].count === truth[t] ? 'p11right' : 'p11wrong')), ...ax.items, S.Kit.curve.line(ax, 'ls:tc', truth.map((v, t) => [t + 1, v]), 'p11c'), S.Kit.curve.line(ax, 'ls:cl', cs.map((v, t) => [t + 1, v]), 'p11a'), ...cs.map((v, t) => S.Kit.curve.dot(ax, 'ls:cd' + t, t + 1, v, bits[t] ? 'p11b' : 'p11a')), ...H().lines('ls:p', 0, 300, [['memory slot ' + (u + 1) + ' step by step', 'strong'], ['orange dots: a 1 was read', 'legend'], ['dashed: the true count', 'legend'], [right + ' of ' + n + ' counts right', 'legend']], 13, 21)],
        `Memory slot ${q.n(u + 1)} works as the counter: it rises by ${q(mean(up1), 2)} on average when a 1 arrives` + (up0.length ? ` and moves by ${q(mean(up0.map(Math.abs)), 2)} on average, up or down, on a 0` + (up0.length > 1 ? ` (from ${q(Math.min(...up0), 2)} to ${q(Math.max(...up0), 2)})` : '') : '') + `. The forget gate holds it, the input gate adds to it. ${q.n(right)} of the ${q.n(n)} counts are right.`, { phase: 'done', unit: u, slot: u + 1, up1: mean(up1), up0: up0.length ? mean(up0) : null, up0abs: up0.length ? mean(up0.map(Math.abs)) : null, up0range: up0.length ? [Math.min(...up0), Math.max(...up0)] : null, right, n });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The LSTM cell', text: ['z = x_t W_x + h_{t−1} W_h + b', 'i, f, o = σ(parts of z)   g = tanh(part of z)', 'c_t = f ⊙ c_{t−1} + i ⊙ g', 'h_t = o ⊙ tanh(c_t)', 'gates: 0 shut, 1 open, one per slot'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => f.meta.phase === 'step' && fr[i + 1] && fr[i + 1].meta.phase === 'step' && fr[i + 1].meta.x === 1, once: true, pick: 'option',
      ask: (fr, i) => 'The next bit is a 1. Will the counter slot’s memory go up, down or stay?',
      options: ['up', 'down', 'stay about the same'],
      answer: (fr, i) => { const m = fr[i + 1].meta, d = m.c - m.cPrev; return Math.abs(d) < 0.05 ? 'stay about the same' : d > 0 ? 'up' : 'down'; },
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'The forget gate of a slot is 1 and its input gate is 0. The memory…', options: ['stays exactly as it was', 'is wiped to zero', 'becomes the candidate', 'doubles'], answer: 0, why: 'c = 1·c_prev + 0·g.' },
      { q: 'Why do LSTM gradients fade less than a plain RNN’s?', options: ['the memory is updated by adding, so with f near 1 the signal passes back almost unchanged', 'LSTMs have no weights', 'they use ReLU', 'they read fewer steps'], answer: 0, why: '∂c_t/∂c_{t−1} = f, which can stay close to 1.' },
      { q: 'What range does a gate value have?', options: ['0 to 1', '−1 to 1', 'any number', '0 or 1 only'], answer: 0, why: 'Gates are sigmoids.' },
      { q: 'Which gate controls how much of the memory is shown as the output h?', options: ['the output gate o', 'the forget gate f', 'the input gate i', 'the candidate g'], answer: 0, why: 'h = o ⊙ tanh(c).' },
      { q: 'An LSTM layer with 4 slots and 1 input has how many weights in W_x and W_h together?', options: ['4·(1 + 4)·4 = 80', '16', '32', '4'], answer: 0, why: 'Four gates, each with 4 outputs from 1 input and 4 state values.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
