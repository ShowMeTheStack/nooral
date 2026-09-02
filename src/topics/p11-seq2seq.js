// Part 11, sequence to sequence: an encoder GRU reads number words ("three one four") and hands its last state, one
// vector of 32 numbers, to a decoder GRU that writes the digits one at a time, each fed back in as its next input.
// The model was trained on inputs of one to five words (weights shipped in p11-0-data.js); longer inputs show the
// limit of squeezing everything through that one vector.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11, C = () => P.S2S;
  const ENC = { x: 92, y: 64, bw: 56, bh: 46, gap: 16 }, DEC = { x: 92, y: 300, bw: 56, bh: 46, gap: 16 };
  /** a vector of 32 as a small 8 × 4 block of heat cells */
  const block = (prefix, x, y, v, cell) => S.Kit.heat.grid({ prefix, x, y, cw: cell || 5.5, ch: cell || 5.5, vals: [0, 1, 2, 3].map((r) => Array.from(v).slice(r * 8, r * 8 + 8)), scheme: 'div', max: 1, fmt: null, gap: 0.6, rx: 1 }).items;

  S.topic({
    id: 'seq2seq', family: 'p11-seq', order: 20, title: 'Sequence to sequence: encode, then decode',
    blurb: 'Some tasks turn one sequence into another of a different length or vocabulary: translation is the classic case. An encoder reads the whole input and ends with one vector; a decoder starts from that vector and writes the output one token at a time, feeding each token it writes back in. Here the “translation” is from number words to digits. Everything the decoder knows must pass through that single vector, and longer inputs show where that breaks.',
    inputs: [{ id: 'digits', kind: 'string', label: 'Number to say (1 to 8 digits)', def: '3141', alphabet: '0123456789', min: 1, max: 8 }],
    presets: [
      { label: 'Four words', values: { digits: '3141' } },
      { label: 'Two words', values: { digits: '72' } },
      { label: 'Five: the longest trained', values: { digits: '90215' } },
      { label: 'Seven: longer than any seen', values: { digits: '5830271' } },
    ],
    counters: [{ id: 'tok', label: 'tokens processed' }],
    run(inp) {
      const F = H().frames(['tok']), q = F.q, net = P.net('seq2seq'), job = P.job('seq2seq'), src = [...inp.digits].map(Number), L = src.length;
      const r = net.translate(src, L + 2), words = src.map((d) => C().WORDS[d]), outIds = r.steps.map((s) => s.id), m = outIds.length;
      const ok = r.out.join() === src.join();
      const enc = (upto) => K().chain(Object.assign({ prefix: 's2:e', n: L, upto, inputs: words, outputs: null, inW: 50, inSize: 12, inGap: 26, label: (t) => 'enc ' + (t + 1), inner: (t, b) => block('s2:eh' + t, b.x + 6, b.y + 22, r.encH[t]), cls: (t) => (t === upto - 1 ? 'act' : t < upto ? 'done' : '') }, ENC)).items;
      const dec = (upto) => K().chain(Object.assign({ prefix: 's2:d', n: m, upto, inputs: [C().START].concat(outIds.slice(0, -1)).map((id, t) => (t < upto ? C().show(id) : '')), outputs: outIds.map(C().show), inW: 40, outW: 40, inGap: 26, outGap: 50, label: (t) => 'dec ' + (t + 1), inner: (t, b) => block('s2:dh' + t, b.x + 6, b.y + 22, r.steps[t].h), cls: (t) => (t === upto - 1 ? 'act' : t < upto ? 'done' : ''), outCls: (t) => (t === m - 1 ? '' : outIds[t] === src[t] ? 'p11right' : 'p11wrong') }, DEC)).items;
      const ctx = (lit) => [{ key: 's2:cx', kind: 'cell', x: 0, y: DEC.y, w: 66, h: DEC.bh, rx: 8, text: '', cls: 'p11box ' + (lit ? 'act' : ''), audit: 'group' }, H().text('s2:cxl', 33, DEC.y + 17, 'context', 'strong', 12, 'middle'), ...block('s2:cxb', 11, DEC.y + 22, r.context), { key: 's2:cxa', kind: 'edge', from: { x: 68, y: DEC.y + DEC.bh / 2 }, to: { x: DEC.x - 2, y: DEC.y + DEC.bh / 2 }, head: true, cls: lit ? 'act' : '', audit: 'edge-x' }];
      const names = [H().text('s2:en', 0, ENC.y + 28, 'encoder', 'strong', 13), H().text('s2:dn', 0, DEC.y - 70, 'decoder', 'strong', 13)];
      const note = (rows) => H().lines('s2:p', 0, 470, rows, 13, 21);
      F.push([...names, ...enc(0), ...ctx(false), ...dec(0)],
        `The input is ${P.qpl(q, L, 'word')}: “${words.join(', ')}”. The encoder and the decoder are each a GRU with ${P.qpl(q, r.context.length, 'number')} of state (the small blocks). They were trained together for ${P.qpl(q, job.steps, 'step')} on inputs of one to ${P.qpl(q, C().MAXLEN, 'word')}.`, { phase: 'task', L, H: r.context.length, steps: job.steps, maxTrained: C().MAXLEN });
      for (let t = 0; t < L; t++) {
        F.count('tok');
        F.push([...names, ...enc(t + 1), ...ctx(false), ...dec(0)],
          t === 0 ? `The encoder reads “${words[0]}” and sets its state.` : `It reads “${words[t]}” and mixes it into the state, which now has to stand for ${P.qpl(q, t + 1, 'word')}.`, { phase: 'enc', t: t + 1, n: t + 1 });
      }
      F.push([...names, ...enc(L), ...ctx(true), ...dec(0)],
        `The encoder’s last state becomes the context: ${P.qpl(q, r.context.length, 'number')} that must hold all ${P.qpl(q, L, 'word')} and their order. Nothing else passes to the decoder.`, { phase: 'context', H: r.context.length, L });
      r.steps.forEach((s, t) => {
        F.count('tok');
        const top = s.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]), p0 = top[0], p1 = top[1];
        const isEnd = s.id === C().END, want = t < L ? src[t] : C().END, right = s.id === want;
        const tq = (id) => (id >= 10 ? C().show(id) : q.n(id));
        const fed = t === 0 ? '' : tq(s.input), wrote = isEnd ? 'the end mark ' + C().show(s.id) : '“' + tq(s.id) + '”', p0s = q(p0[0], 2), alt = tq(p1[1]), p1s = q(p1[0], 2), wantS = right ? '' : tq(want);
        F.push([...names, ...enc(L), ...ctx(true), ...dec(t + 1), ...note([[`step ${t + 1}: input ${C().show(s.input)} → ${C().show(s.id)} (p = ${H().fx(p0[0], 2)}); next best ${C().show(p1[1])} (${H().fx(p1[0], 2)})`, 'math']])],
          (t === 0 ? `The decoder starts from the context with the start mark ${C().show(C().START)} as input` : `It is fed the ${fed} it just wrote`) + ` and writes ${wrote} with probability ${p0s} (next best “${alt}”, ${p1s}).` + (right ? '' : ` The right token was “${wantS}”.`),
          { phase: 'dec', t: t + 1, id: s.id, input: s.input, p: p0[0], second: p1[1], p2: p1[0], right, want });
      });
      // the bottleneck: exact matches by input length
      const lens = job.lengths, ax = S.Kit.curve.axes({ prefix: 's2:ax', x: 120, y: 80, w: 560, h: 260, xr: [0.5, lens.length + 0.5], yr: [0, 1], xticks: lens.map((x) => x.L), yticks: 0.25, fmtY: (v) => H().pct(v), xname: 'input length (words)', yname: 'outputs exactly right', grid: true });
      const bw = (ax.X(1) - ax.X(0)) * 0.6;
      const bars = lens.map((x, k) => ({ key: 's2:lb' + k, kind: 'cell', x: P.r2(ax.X(x.L) - bw / 2), y: ax.Y(x.exact), w: P.r2(bw), h: P.r2(Math.max(1, ax.Y(0) - ax.Y(x.exact))), rx: 2, text: '', cls: 'bar p11bar ' + (x.L <= C().MAXLEN ? '' : 'hot') + (x.L === L ? ' good' : ''), audit: 'ignore' }));
      const here = lens.find((x) => x.L === L), last = lens[lens.length - 1];
      F.push([...ax.items, ...bars, ...S.Kit.curve.legend('s2:lg', 130, 410, [{ text: 'lengths seen in training', cls: 'p11a' }, { text: 'longer than any seen', cls: 'p11b' }, { text: 'this input’s length', cls: 'p11g' }], { size: 12 })],
        `This input came out ${ok ? 'exactly right' : 'wrong'}. On ${P.qpl(q, 100, 'new input')} of each length the model is ${q.pct(here.exact)} right at ${P.qpl(q, L, 'word')}, but only ${q.pct(last.exact)} at ${P.qpl(q, last.L, 'word')}: past the lengths it trained on, one fixed vector cannot hold the whole input. Attention fixes this by letting the decoder look back at every encoder state.`, { phase: 'lengths', ok, exact: here.exact, lastExact: last.exact, lastL: last.L, L, nTest: 100 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Encoder–decoder', text: ['encoder: h_t = GRU(e(x_t), h_{t−1})', 'context c = h_n (the last state)', 'decoder: s_0 = c', 's_k = GRU(e(y_{k−1}), s_{k−1})', 'y_k = argmax softmax(s_k W + b)', 'stop at the end mark </s>'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'context', once: true, pick: 'option',
      ask: 'Will the decoder write every digit right?',
      options: ['yes, all of them', 'no, at least one is wrong'],
      answer: (fr) => (fr.find((f) => f.meta.phase === 'lengths').meta.ok ? 'yes, all of them' : 'no, at least one is wrong'),
      explain: (fr) => fr.find((f) => f.meta.phase === 'lengths').say,
    }],
    quiz: [
      { q: 'What does the decoder receive from the encoder in this model?', options: ['only the encoder’s last state', 'every encoder state', 'the input words themselves', 'nothing'], answer: 0, why: 'The single context vector is the only link.' },
      { q: 'What is the decoder’s input at each step after the first?', options: ['the token it wrote at the step before', 'the next input word', 'the context vector again', 'a random token'], answer: 0, why: 'It feeds its own output back in.' },
      { q: 'Why does accuracy fall for inputs longer than any seen in training?', options: ['one fixed-size vector must hold the whole input, and it never learned to hold that much', 'the vocabulary runs out', 'the decoder stops early on purpose', 'longer inputs are always wrong'], answer: 0, why: 'This bottleneck is what attention removes.' },
      { q: 'How does the decoder know when to stop?', options: ['it writes a special end mark', 'it counts the input words', 'after a fixed number of steps', 'when the probability drops below 0.5'], answer: 0, why: 'The end mark is a token like any other, learned in training.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
