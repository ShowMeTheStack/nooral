// Part 13, next-token prediction: the trained character GPT reading a prompt and giving a probability to every
// character that could come next, one step at a time (the most likely is appended and read again); then the same
// probabilities scored: cross-entropy as the mean of −ln p of the true next character, and perplexity as its
// exponential, against the untrained model and against guessing uniformly.
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml;
  /** the top k characters of a probability row: [{id, c, p}] */
  P.top = (probs, k) => probs.map((p, id) => ({ id, p, c: P.vocab()[id] })).sort((a, b) => b.p - a.p || a.id - b.id).slice(0, k);
  /** a probability for the narration: two decimals, or two significant digits when it is small */
  P.qp = (q, p) => (p >= 0.01 ? q(p, 2) : q.sg(p, 2));

  S.topic({
    id: 'lm-next-token', family: 'p13-lm', order: 10, title: 'Next-token prediction, one character at a time',
    blurb: 'A language model does one thing: given the text so far, it gives every token in its vocabulary a probability of coming next. Here the tokens are characters and the model is a tiny GPT trained on the harbour text. Take the most likely character, append it, and ask again: that loop is all there is to generating text.',
    inputs: [
      { id: 'prompt', kind: 'text', label: 'Prompt', def: 'the weaver said that ', check: P.checkText('Prompt', 3, 24) },
      { id: 'steps', kind: 'number', label: 'Characters to add', def: 6, min: 1, max: 8 },
    ],
    presets: [
      { label: 'The weaver said…', values: { prompt: 'the weaver said that ', steps: 6 } },
      { label: 'Middle of a word', values: { prompt: 'the lighth', steps: 5 } },
      { label: 'After a full stop', values: { prompt: 'the boats rock.', steps: 6 } },
      { label: 'An unusual start', values: { prompt: 'a cat ', steps: 8 } },
    ],
    counters: [{ id: 'added', label: 'characters added' }],
    run(inp) {
      const net = P.gpt(), V = P.vocab().length, F = H().frames(['added']), q = F.q, nParams = P.D().gpt.params;
      let text = inp.prompt;
      const n0 = text.length;
      const strip = (lit) => [H().text('nt:h', 0, 16, 'the text so far (· is a space)', 'strong', 13), ...P.cells('nt:c', 0, 30, Array.from(text), { w: 22, cls: (i) => (i >= n0 ? 'new' : '') + (i === lit ? ' p13cur' : '') })];
      const model = [
        { key: 'nt:m', kind: 'state', x: 110, y: 210, w: 170, h: 74, rx: 10, lines: ['tiny GPT', nParams.toLocaleString('en') + ' numbers'], size: 13, cls: 'block p13model' },
        { key: 'nt:in', kind: 'edge', from: { x: 110, y: 64 }, to: 'nt:m', head: true, cls: 'p13flow' },
        H().text('nt:inl', 122, 120, 'reads every character', 'legend', 12),
      ];
      const barsAt = (probs, pick) => {
        const top = P.top(probs, 8);
        return [
          ...P.panel('nt:p', 300, 96, 490, 300, 'probability of each next character (top 8 of ' + V + ')'),
          { key: 'nt:out', kind: 'edge', from: 'nt:m', to: { x: 300, y: 210 }, head: true, cls: 'p13flow' },
          ...P.bars('nt:b', 316, 130, top.map((t) => ({ label: P.show(t.c), v: t.p, cls: t.id === pick ? 'hot' : '', text: H().fx(t.p, 3) })), { lw: 30, bw: 330, rh: 32, max: 1 }),
        ];
      };
      F.push([...strip(-1), ...model, ...P.panel('nt:p', 300, 96, 490, 300, 'probability of each next character (top 8 of ' + V + ')'), { key: 'nt:out', kind: 'edge', from: 'nt:m', to: { x: 300, y: 210 }, head: true, cls: 'p13flow' }, H().text('nt:wait', 545, 250, 'run the model to see them', 'legend', 13, 'middle')],
        `The model reads the ${P.qpl(q, n0, 'character')} of the prompt, each turned into a vector and passed through its layers. Out of the last position comes one score for each of the ${q.n(V)} characters it knows, turned into probabilities that add up to one.`, { phase: 'start', n: n0, V });
      const steps = [];
      for (let k = 0; k < inp.steps; k++) {
        const r = P.next(net, text), top = P.top(r.probs, 3), pick = top[0];
        steps.push({ c: pick.c, p: pick.p });
        F.count('added');
        F.push([...strip(-1), ...model, ...barsAt(r.probs, pick.id)],
          `After “${P.show(text.slice(-12))}” the most likely next character is ${P.name(pick.c)} with p = ${P.qp(q, pick.p)}; then ${P.name(top[1].c)} (${P.qp(q, top[1].p)}) and ${P.name(top[2].c)} (${P.qp(q, top[2].p)}). The model takes the first, appends it, and reads the longer text again.`,
          { phase: 'step', k: k + 1, text, top: top.map((t) => ({ id: t.id, p: t.p })), pick: pick.id, p: pick.p, probs: r.probs });
        text += pick.c;
      }
      const sure = steps.reduce((a, s, i) => (s.p > steps[a].p ? i : a), 0), unsure = steps.reduce((a, s, i) => (s.p < steps[a].p ? i : a), 0);
      F.push([...strip(n0 + unsure), ...model, ...P.panel('nt:p', 300, 96, 490, 300, 'how sure it was of each character it added'), { key: 'nt:out', kind: 'edge', from: 'nt:m', to: { x: 300, y: 210 }, head: true, cls: 'p13flow' }, ...P.bars('nt:b', 316, 130, steps.map((st, i) => ({ label: (i + 1) + ': ' + P.show(st.c), v: st.p, cls: i === unsure ? 'hot' : '', text: H().fx(st.p, 3) })), { lw: 44, bw: 316, rh: 32, max: 1 })],
        `The model added “${P.show(text.slice(n0))}”. It was surest of character ${q.n(sure + 1)}, ${P.name(steps[sure].c)} (p = ${P.qp(q, steps[sure].p)}), and least sure of character ${q.n(unsure + 1)}, ${P.name(steps[unsure].c)} (p = ${P.qp(q, steps[unsure].p)}, ringed): inside a word the next letter is often clear, at the start of a word many are possible.`,
        { phase: 'end', added: text.slice(n0), sure: sure + 1, pSure: steps[sure].p, unsure: unsure + 1, pUnsure: steps[unsure].p });
      return { frames: F.list, def: { sections: [{ id: 'lm', kind: 'math', title: 'A language model', text: ['p(next | text so far)  for every token', 'scores z → softmax: pᵢ = e^zᵢ / Σⱼ e^zⱼ', 'generate: pick a token, append it,', '  and ask again'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start',
      once: true, pick: 'option',
      ask: 'Which character will the model put next?',
      options: (fr) => { const s = fr.find((f) => f.meta.phase === 'step').meta; return P.top(s.probs, 4).map((t) => P.show(t.c)).sort(); },
      answer: (fr) => P.show(P.vocab()[fr.find((f) => f.meta.phase === 'step').meta.pick]),
      explain: (fr) => fr.find((f) => f.meta.phase === 'step').say,
    }],
    quiz: [
      { q: 'What does a language model output at each position?', options: ['a probability for every token in its vocabulary', 'the next word only', 'a yes or no', 'the whole rest of the text'], answer: 0, why: 'A softmax over the vocabulary.' },
      { q: 'How is a longer text generated?', options: ['one token at a time: pick, append, run again', 'all tokens at once', 'by copying the training text', 'backwards from the end'], answer: 0, why: 'That loop is autoregressive generation.' },
      { q: 'The model’s probabilities for the next character add up to…', options: ['one', 'the number of characters', 'zero', 'whatever the model likes'], answer: 0, why: 'The softmax makes them a probability distribution.' },
      { q: 'Where is a character model usually most uncertain?', options: ['at the start of a new word', 'in the middle of a common word', 'right after “th”', 'never'], answer: 0, why: 'Many words can follow a space; few letters can finish “lighth”.' },
    ],
  });

  S.topic({
    id: 'lm-cross-entropy', family: 'p13-lm', order: 20, title: 'Cross-entropy and perplexity',
    blurb: 'How good is a language model? Run it along a text and, at every position, look at the probability it gave to the character that really came next. Cross-entropy is the average of −ln p over the text: a sure, correct model scores near zero, a clueless one scores high. Perplexity, e to that power, reads as “as unsure as choosing among this many characters”.',
    inputs: [
      { id: 'text', kind: 'text', label: 'Text to score', def: 'the sailor coils the ropes.', check: P.checkText('Text to score', 4, 32) },
      { id: 'model', kind: 'choice', label: 'Model', def: 'trained', options: [{ value: 'trained', label: 'trained (1,600 steps)' }, { value: 'untrained', label: 'untrained (random weights)' }] },
    ],
    presets: [
      { label: 'A familiar sentence', values: { text: 'the sailor coils the ropes.', model: 'trained' } },
      { label: 'Untrained model', values: { text: 'the sailor coils the ropes.', model: 'untrained' } },
      { label: 'Odd text', values: { text: 'ropes the coils sailor the.', model: 'trained' } },
      { label: 'Letters at random', values: { text: 'qw kpv bfltr mmo gy', model: 'trained' } },
    ],
    counters: [{ id: 'scored', label: 'characters scored' }],
    run(inp) {
      const net = inp.model === 'trained' ? P.gpt() : P.gpt0(), V = P.vocab().length, F = H().frames(['scored']), q = F.q;
      const chars = Array.from(inp.text), ids = net.tokenizer.encode(inp.text), L = P.logits(net, ids), n = chars.length;
      const probs = L.map((row) => P.softmax(row));
      // p[i]: the probability given, after characters 0..i−1, to character i
      const p = chars.map((c, i) => (i ? probs[i - 1][ids[i]] : null)), nll = p.map((x) => (x === null ? null : -Math.log(x)));
      const CW = Math.min(22, Math.floor(780 / n) - 2), BY = 200, BH = 110;
      const strip = (cur, done) => [
        H().text('ce:h', 0, 16, 'the text; under each character, the probability the model gave it', 'strong', 13),
        ...P.cells('ce:c', 0, 30, chars, { w: CW, cls: (i) => (i === cur ? 'p13cur' : i && i <= done ? '' : 'soft') }),
        { key: 'ce:base', kind: 'line', x1: 0, y1: BY, x2: n * (CW + 2), y2: BY, cls: 'faint', audit: 'ignore' },
        H().text('ce:one', n * (CW + 2) + 6, BY - BH + 4, 'p = 1', 'tick', 12),
        { key: 'ce:top', kind: 'line', x1: 0, y1: BY - BH, x2: n * (CW + 2), y2: BY - BH, cls: 'faint', audit: 'ignore' },
        ...chars.map((_, i) => (i && i <= done ? { key: 'ce:b' + i, kind: 'cell', x: i * (CW + 2) + 3, y: BY - Math.max(1.5, BH * p[i]), w: CW - 6, h: Math.max(1.5, BH * p[i]), rx: 2, text: '', cls: 'bar p13bar ' + (i === cur ? 'hot' : p[i] < 0.1 ? 'p13low' : ''), audit: 'ignore' } : null)).filter(Boolean),
      ];
      const panel = (lines) => [...P.panel('ce:p', 0, 228, 790, 210, null), ...H().lines('ce:l', 16, 258, lines, 13, 24)];
      const altPanel = (i) => {
        const top = P.top(probs[i - 1], 5), rows = top.map((t) => ({ label: P.show(t.c), v: t.p, cls: t.id === ids[i] ? 'hot' : 'soft', text: H().fx(t.p, 2) }));
        if (!top.some((t) => t.id === ids[i])) rows.push({ label: P.show(chars[i]), v: p[i], cls: 'hot', text: H().sg(p[i], 2) });
        return [H().text('ce:at', 530, 252, 'the model’s top guesses here', 'legend', 12), ...P.bars('ce:a', 520, 262, rows, { lw: 26, bw: 160, rh: 28, max: 1 })];
      };
      const say0 = inp.model === 'trained' ? 'the trained character GPT' : 'the same GPT before training, with its random first weights';
      F.push([...strip(-1, 0), ...panel([['cross-entropy  = mean over positions of −ln p(true next character)', 'math'], ['perplexity  = e^(cross-entropy)', 'math']])],
        `The text has ${P.qpl(q, n, 'character')}. ${say0[0].toUpperCase() + say0.slice(1)} predicts each of the last ${q.n(n - 1)} from the ones before it; the first has nothing before it and is not scored.`, { phase: 'start', n, scored: n - 1 });
      const single = Math.min(n - 1, 8);
      let sum = 0;
      for (let i = 1; i <= single; i++) {
        sum += nll[i];
        F.count('scored');
        F.push([...strip(i, i), ...panel([[`after “${P.show(inp.text.slice(Math.max(0, i - 14), i))}”:`, 'legend'], [`p(${P.show(chars[i])}) = ${H().sg(p[i], 3)}`, 'math'], [`−ln p = ${H().fx(nll[i], 3)}`, 'math'], [`mean so far: ${H().fx(sum / i, 3)} over ${H().pl(i, 'character')}`, 'strong']]), ...altPanel(i)],
          `Position ${q.n(i + 1)}: after “${P.show(inp.text.slice(Math.max(0, i - 14), i))}” the model gave the true next character, ${P.name(chars[i])}, p = ${P.qp(q, p[i])}, so it pays −ln p = ${q(nll[i], 2)}. ` + (p[i] > 0.5 ? 'A confident right guess costs little.' : p[i] < 0.1 ? 'A surprise costs a lot.' : 'A middling guess costs a middling amount.'),
          { phase: 'pos', i, pos: i + 1, c: chars[i], p: p[i], nll: nll[i], mean: sum / i });
      }
      if (n - 1 > single) {
        for (let i = single + 1; i < n; i++) sum += nll[i];
        F.set('scored', n - 1);
        F.push([...strip(-1, n - 1), ...panel([['every position scored', 'strong'], [`sum of −ln p: ${H().fx(sum, 2)}`, 'math']])],
          `The remaining ${P.qpl(q, n - 1 - single, 'position')} are scored the same way. Adding all ${q.n(n - 1)} costs gives ${q(sum, 2)}.`, { phase: 'rest', rest: n - 1 - single, sum, scored: n - 1 });
      }
      const ce = sum / (n - 1), ppl = Math.exp(ce), lnV = Math.log(V);
      const worst = p.reduce((a, x, i) => (x !== null && (a < 0 || x < p[a]) ? i : a), -1);
      F.push([...strip(worst, n - 1), ...panel([[`cross-entropy = ${H().fx(sum, 2)} / ${n - 1} = ${H().fx(ce, 3)}`, 'math'], [`perplexity = e^${H().fx(ce, 3)} = ${H().fx(ppl, 2)}`, 'math'], [`guessing uniformly: ln ${V} = ${H().fx(lnV, 3)}, perplexity ${V}`, 'legend'], [`hardest: ${P.show(chars[worst])} at position ${worst + 1}, p = ${H().sg(p[worst], 2)}`, 'legend']])],
        `Cross-entropy is the mean cost, ${q(sum, 2)} / ${q.n(n - 1)} = ${q(ce, 3)} nats per character, and perplexity is e to that power, ${q(ppl, 2)}: on this text the model is as unsure as picking among about ${q.n(Math.round(ppl))} characters. Guessing uniformly among all ${q.n(V)} would cost ln ${q.n(V)} = ${q(lnV, 3)}, perplexity ${q.n(V)}. The hardest character was ${P.name(chars[worst])} at position ${q.n(worst + 1)} (p = ${P.qp(q, p[worst])}).`,
        { phase: 'end', sum, ce, ppl, round: Math.round(ppl), V, lnV, worst, worstPos: worst + 1, pWorst: p[worst], scored: n - 1 });
      return { frames: F.list, def: { sections: [{ id: 'ce', kind: 'math', title: 'Scoring a language model', text: ['cost at position t:  −ln p(x_t | x_<t)', 'cross-entropy  H = (1/N) Σ_t −ln p(x_t | x_<t)', 'perplexity  = e^H', 'uniform over V tokens: H = ln V, perplexity V', 'training lowers H on the training text'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start',
      once: true, pick: 'item',
      ask: 'Click the character the model will find hardest to predict.',
      answer: (fr) => 'ce:c:' + fr[fr.length - 1].meta.worst,
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'The model gave the true next character p = 1. Its cost there is…', options: ['0', '1', 'ln 2', 'infinite'], answer: 0, why: '−ln 1 = 0.' },
      { q: 'The model gave the true next character p = 0.01. Its cost there is about…', options: ['4.6', '0.01', '1', '0.99'], answer: 0, why: '−ln 0.01 ≈ 4.6: surprises are expensive.' },
      { q: 'A model with perplexity 3 on a text is, on average…', options: ['as unsure as choosing among 3 equally likely tokens', 'wrong 3 times in a row', '3 times better than chance', 'right 3% of the time'], answer: 0, why: 'Perplexity is e to the cross-entropy: an effective number of choices.' },
      { q: 'A model that guesses uniformly over V tokens has cross-entropy…', options: ['ln V', 'V', '1 / V', '0'], answer: 0, why: 'Every token gets p = 1/V, and −ln(1/V) = ln V.' },
      { q: 'Training a language model lowers…', options: ['its cross-entropy on the training text', 'its vocabulary size', 'the length of its outputs', 'the number of its parameters'], answer: 0, why: 'Cross-entropy is the loss it is trained on.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
