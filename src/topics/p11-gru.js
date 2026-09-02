// Part 11, the GRU step by step: a cell of four numbers trained to remember the last letter it saw in a stream of
// letters and blanks. Its update gate z decides, per number, how much of the old state to keep; when a letter
// arrives the gate drops and the state is rewritten, and over blanks it stays near 1 and the state is held.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11;
  const SYM = ['-', 'A', 'B', 'C'], CH = { x: 30, y: 92, bw: 58, bh: 62, gap: 18 };
  const show = (k) => (k === 0 ? '–' : SYM[k]);

  /** the trained GRU on a stream: gates and states per step, as plain arrays [t][unit] */
  P.gruTrace = (ids) => {
    const net = P.net('gruLast'), T = S.T, n = ids.length;
    const logits = net.predict(T.tensor([ids])), L = logits.values(), C = logits.shape[2], G = net.gates();
    const take = (name) => { const t = G[name], Hn = t.shape[2], v = t.values(); return Array.from({ length: n }, (_, k) => Array.from(v.subarray(k * Hn, (k + 1) * Hn))); };
    const r = take('gate.r'), z = take('gate.z'), nn = take('gate.n'), h = net.cell.steps.map((s) => Array.from(s.h.values()));
    const out = ids.map((_, t) => { const p = P.softmax(Array.from(L.subarray(t * C, (t + 1) * C))), k = P.argmax(p); return { last: k, pk: p[k] }; });
    // the number to follow: the one whose update gate best tells letters (rewrite) from blanks (hold) in this stream
    const Hn = h[0].length, mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const score = Array.from({ length: Hn }, (_, u) => { const L = z.filter((_, t) => ids[t]).map((row) => row[u]), B = z.filter((_, t) => !ids[t]).map((row) => row[u]); return L.length && B.length ? mean(B) - mean(L) : Math.max(...z.map((row) => row[u])) - Math.min(...z.map((row) => row[u])); });
    return { net, r, z, n: nn, h, out, unit: P.argmax(score), Hn };
  };

  S.topic({
    id: 'gru-gates', family: 'p11-rnn', order: 40, title: 'GRU: one gate to keep or rewrite',
    blurb: 'The GRU is a lighter gated cell. It has no separate memory: its state h is the memory. An update gate z chooses, number by number, between keeping the old state (z near 1) and taking a new candidate n (z near 0); a reset gate r chooses how much of the old state the candidate may look at. This cell has learned to remember the last letter in a stream of letters and blanks.',
    inputs: [{ id: 'stream', kind: 'string', label: 'Stream (letters A, B, C and blanks -)', def: 'A--B---C-A', alphabet: '-ABC', min: 4, max: 10 }],
    presets: [
      { label: 'Letters and gaps', values: { stream: 'A--B---C-A' } },
      { label: 'Long hold', values: { stream: 'B--------C' } },
      { label: 'Letter after letter', values: { stream: 'ABCABC' } },
      { label: 'Blanks first', values: { stream: '----C---' } },
    ],
    counters: [{ id: 'steps', label: 'steps read' }],
    run(inp) {
      const F = H().frames(['steps']), q = F.q, ids = [...inp.stream].map((c) => SYM.indexOf(c)), n = ids.length, tr = P.gruTrace(ids), u = tr.unit, job = P.job('gruLast');
      const ev = job.evals[job.evals.length - 1], truth = []; let last = 0; for (const k of ids) { if (k) last = k; truth.push(last); }
      const inner = (t, b) => K().strip({ prefix: 'gu:hv' + t, x: b.x + 4, y: b.y + 30, vals: tr.h[t], cw: 11.5, ch: 20, scheme: 'div', max: 1, fmt: null, gap: 1.5 }).items;
      const chain = (upto, outCls) => K().chain(Object.assign({ prefix: 'gu:c', n, upto, inputs: ids.map(show), outputs: tr.out.map((s) => show(s.last)), inner, label: (t) => 't' + (t + 1), cls: (t) => (t === upto - 1 ? 'act' : t < upto ? 'done' : ''), inCls: (t) => (t === upto - 1 ? 'new' : ''), outCls }, CH)).items;
      const panel = (rows) => H().lines('gu:p', 330, 270, rows, 13, 21);
      F.push([...chain(0), ...panel([['r = σ(·)  reset: how much old state the candidate sees', 'math'], ['z = σ(·)  update: how much old state to keep', 'math'], ['n = tanh(x W + r ⊙ (h_prev U))  the candidate', 'math'], ['h = (1 − z) ⊙ n + z ⊙ h_prev', 'strong']]), H().text('gu:k', 30, 240, 'in each box: the state h (four numbers); above, the letter it reports', 'legend', 12)],
        `The cell has ${P.qpl(q, tr.Hn, 'number')} of state and two gates. At every step its output says which letter came last (– while none has). Trained for ${P.qpl(q, job.steps, 'step')} on random streams, it is right at ${q.pct(ev.accuracy)} of the steps of new ones.`, { phase: 'intro', Hn: tr.Hn, steps: job.steps, acc: ev.accuracy });
      for (let t = 0; t < n; t++) {
        F.count('steps');
        const hp = t ? tr.h[t - 1][u] : 0, zv = tr.z[t][u], nv = tr.n[t][u], hv = tr.h[t][u];
        const table = P.gateTable('gu:g', 60, 280, [{ name: 'r', v: tr.r[t] }, { name: 'z', v: tr.z[t] }, { name: 'n', v: tr.n[t], scheme: 'div' }, { name: 'h', v: tr.h[t], scheme: 'div' }], { unit: u, cw: 52, ch: 22 });
        const rows = [['t' + (t + 1) + ', input ' + show(ids[t]) + ', number ' + (u + 1) + ':', 'strong'], ['h = (1 − z)·n + z·h_prev', 'math'], ['  = ' + H().fx(1 - zv, 2) + ' × ' + H().fx(nv, 2) + ' + ' + H().fx(zv, 2) + ' × ' + H().fx(hp, 2), 'math'], ['  = ' + H().fx(hv, 2), 'math'], ['last letter: ' + show(tr.out[t].last) + ' (p = ' + H().fx(tr.out[t].pk, 2) + ')', 'strong']];
        const kind = ids[t] ? `the letter ${show(ids[t])}` : 'a blank';
        F.push([...chain(t + 1), ...table, ...panel(rows)],
          `Step ${q.n(t + 1)} reads ${kind}. For number ${q.n(u + 1)} the update gate is ${q(zv, 2)}: it keeps that share of the old ${q(hp, 2)} and takes the rest from the candidate ${q(nv, 2)}, giving ${q(hv, 2)}. ` + (zv > 0.9 ? 'The state is held.' : Math.abs(nv - hp) < 0.1 ? 'The candidate repeats the old value, so the number barely changes.' : zv < 0.5 ? 'The state is being rewritten.' : 'The state shifts part of the way.') + ` The output says the last letter is ${show(tr.out[t].last)}.`,
          { phase: 'step', t: t + 1, x: ids[t], unit: u, num: u + 1, z: zv, n: nv, hPrev: hp, h: hv, last: tr.out[t].last, pk: tr.out[t].pk });
      }
      const zs = tr.z.map((row) => row[u]), atL = zs.filter((_, t) => ids[t]), atB = zs.filter((_, t) => !ids[t]), mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
      const ax = S.Kit.curve.axes({ prefix: 'gu:ax', x: 360, y: 280, w: 400, h: 140, xr: [1, n], yr: [0, 1], xticks: 1, yticks: 0.5, xname: 'step', yname: 'update gate z, number ' + (u + 1), grid: false });
      const right = tr.out.filter((s, t) => s.last === truth[t]).length;
      F.push([...chain(n, (t) => (tr.out[t].last === truth[t] ? 'p11right' : 'p11wrong')), ...ax.items, S.Kit.curve.line(ax, 'gu:zl', zs.map((v, t) => [t + 1, v]), 'p11a'), ...zs.map((v, t) => S.Kit.curve.dot(ax, 'gu:zd' + t, t + 1, v, ids[t] ? 'p11b' : 'p11a')), ...H().lines('gu:p', 0, 290, [['the update gate over time', 'strong'], ['orange dots: a letter arrived', 'legend'], [right + ' of ' + n + ' answers right', 'legend']], 13, 21)],
        (atL.length && atB.length ? `Number ${q.n(u + 1)}’s update gate averages ${q(mean(atL), 2)} when a letter arrives and ${q(mean(atB), 2)} over blanks: ` + (mean(atL) < mean(atB) ? 'it opens to write a letter in and closes to hold it.' : 'here it does not split cleanly between letters and blanks; other numbers share the work.') : `Number ${q.n(u + 1)}’s update gate moves between ${q(Math.min(...zs), 2)} and ${q(Math.max(...zs), 2)}.`) + ` ${q.n(right)} of the ${q.n(n)} answers are right.`,
        { phase: 'done', unit: u, num: u + 1, zL: mean(atL), zB: mean(atB), zMin: Math.min(...zs), zMax: Math.max(...zs), right, n });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The GRU cell', text: ['r = σ(x W_r + h_{t−1} U_r + b_r)', 'z = σ(x W_z + h_{t−1} U_z + b_z)', 'n = tanh(x W_n + r ⊙ (h_{t−1} U_n) + b_n)', 'h_t = (1 − z) ⊙ n + z ⊙ h_{t−1}'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => f.meta.phase === 'step' && f.meta.t >= 2 && fr[i + 1] && fr[i + 1].meta.phase === 'step' && fr[i + 1].meta.x === 0, once: true, pick: 'option',
      ask: (fr, i) => 'The next input is a blank. Will the update gate of number ' + (fr[i].meta.unit + 1) + ' be nearer 0 (rewrite) or nearer 1 (hold)?',
      options: ['nearer 0: rewrite', 'nearer 1: hold'],
      answer: (fr, i) => (fr[i + 1].meta.z >= 0.5 ? 'nearer 1: hold' : 'nearer 0: rewrite'),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'In a GRU, an update gate of 1 means the number…', options: ['keeps its old value', 'takes the candidate', 'is reset to zero', 'doubles'], answer: 0, why: 'h = (1 − 1)·n + 1·h_prev = h_prev.' },
      { q: 'How does a GRU differ from an LSTM?', options: ['it has no separate memory and fewer gates', 'it has no gates', 'it reads sequences backwards', 'it cannot be trained'], answer: 0, why: 'Two gates, and the state is the memory.' },
      { q: 'What does the reset gate do?', options: ['controls how much of the old state the candidate is built from', 'clears the output', 'restarts the sequence', 'sets the learning rate'], answer: 0, why: 'n = tanh(x W + r ⊙ (h_prev U)).' },
      { q: 'To hold a letter over a long run of blanks, the update gate should be…', options: ['close to 1', 'close to 0', 'exactly 0.5', 'negative'], answer: 0, why: 'Keeping the old state is z near 1.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
