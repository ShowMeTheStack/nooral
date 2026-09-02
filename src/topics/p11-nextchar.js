// Part 11, next-character prediction: an LSTM trained on the corpus one character at a time (weights shipped in
// p11-0-data.js). It reads a prompt letter by letter, showing after each one its five likeliest next characters and
// how surprised it was by the real one, then writes on by itself, greedily and by sampling.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11, TK = () => S.Kit.tokens;
  const sh = (c) => TK().show(c);

  S.topic({
    id: 'next-char', family: 'p11-seq', order: 10, title: 'Predicting the next character',
    blurb: 'Train a recurrent network to guess the next character of a text and it must learn spelling, common words and a little grammar on the way. This LSTM was trained on our corpus of short sentences. Give it the start of a sentence: after each character it lists its guesses for the next one with their probabilities, and the cost of its surprise at the real one. Then it writes on by itself.',
    inputs: [
      { id: 'prompt', kind: 'string', spaces: true, label: 'Prompt (lower-case letters, spaces, full stops)', def: 'the old king left', min: 2, max: 20, check: (v) => { const voc = P.net('charLM').tokenizer.vocab, bad = [...v].find((c) => !voc.includes(c)); return bad === undefined ? null : 'Prompt: “' + bad + '” is not a character the model knows'; } },
      { id: 'temp', kind: 'slider', label: 'Temperature for sampling', def: 0.8, min: 0.2, max: 2, step: 0.1, fmt: (v) => H().fx(v, 1) },
      { id: 'seed', kind: 'number', label: 'Sampling seed', def: 3, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A king', values: { prompt: 'the old king left', temp: 0.8, seed: 3 } },
      { label: 'At the market', values: { prompt: 'we bought three', temp: 0.8, seed: 5 } },
      { label: 'Animals', values: { prompt: 'the goat ran acr', temp: 0.5, seed: 1 } },
      { label: 'Hot sampling', values: { prompt: 'she is the', temp: 1.6, seed: 2 } },
    ],
    counters: [{ id: 'chars', label: 'characters read' }],
    run(inp) {
      const F = H().frames(['chars']), q = F.q, net = P.net('charLM'), tok = net.tokenizer, job = P.job('charLM'), T = S.T;
      const text = inp.prompt, ids = tok.encode(text), n = ids.length, V = tok.size;
      const logits = net.predict(T.tensor([ids])), L = logits.values();
      const probs = ids.map((_, t) => P.softmax(Array.from(L.subarray(t * V, (t + 1) * V))));
      const top = (p, k) => p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k);
      const strip = (upto, extra, cls) => TK().strip({ prefix: 'nc:s', x: 0, y: 30, width: 790, tokens: [...text].concat(extra || []), size: 14, gap: 3, minW: 20, cls: (i) => (i >= n ? 'p11gen ' + (cls || '') : i === upto ? 'new' : i < upto ? '' : 'p11in') }).items;
      const bars = (p, real, title) => [H().text('nc:bt', 0, 140, title, 'strong', 13), ...K().bars({ prefix: 'nc:b', x: 0, y: 150, w: 400, rows: top(p, 5).map(([v, i]) => ({ label: '“' + sh(tok.vocab[i]) + '”', v, value: H().fx(v, 2), cls: i === real ? 'good' : '' })), max: 1, labelW: 60, rowH: 26 })];
      const panel = (rows) => H().lines('nc:p', 450, 150, rows, 13, 21);
      // how it learned
      const ev = job.evals[job.evals.length - 1], smp = job.samples;
      F.push([...strip(-1), H().text('nc:st', 0, 120, 'what it wrote after “' + job.samples[0].greedy.slice(0, 4).trim() + '” during training (greedy)', 'strong', 13), ...smp.map((s, k) => H().text('nc:sm' + k, 0, 150 + k * 30, 'step ' + s.step + ':  ' + s.greedy.slice(0, 54), 'mono', 13))],
        `The model reads one character at a time from a vocabulary of ${P.qpl(q, V, 'character')}. It was trained for ${P.qpl(q, job.steps, 'step')} on ${P.qpl(q, P.charText().length, 'character')} of our sentences. Its greedy writing went from one repeated letter at step ${q.n(smp[0].step)} to whole sentences by step ${q.n(smp[smp.length - 1].step)}, when it guessed ${q.pct(ev.accuracy)} of held-out characters right.`, { phase: 'learned', V, steps: job.steps, nText: P.charText().length, s0: smp[0].step, s1: smp[smp.length - 1].step, acc: ev.accuracy });
      // reading the prompt
      let lossSum = 0, lossN = 0;
      for (let t = 0; t < n; t++) {
        F.count('chars');
        const p = probs[t], real = t + 1 < n ? ids[t + 1] : -1, [b0] = top(p, 1), guess = b0[1];
        let say = `After “${sh(text.slice(0, t + 1))}” its likeliest next character is “${sh(tok.vocab[guess])}” (${q(b0[0], 2)}).`;
        const meta = { phase: 'read', t, guess, pGuess: b0[0] };
        if (real >= 0) {
          const loss = -Math.log(p[real]);
          lossSum += loss; lossN++;
          say += ` The real one is “${sh(tok.vocab[real])}”, given ${q(p[real], 2)}: a loss of −ln ${q(p[real], 2)} = ${q(loss, 2)}.`;
          Object.assign(meta, { real, pReal: p[real], loss });
        } else say += ' That is where the prompt ends.';
        F.push([...strip(t), ...bars(p, real, 'next character after “' + sh(text.slice(Math.max(0, t - 9), t + 1)) + '”'), ...panel([['read so far: ' + (t + 1) + ' of ' + n, 'legend'], ...(real >= 0 ? [['real next: “' + sh(tok.vocab[real]) + '”', 'strong'], ['p = ' + H().fx(p[real], 2) + ',  loss = ' + H().fx(-Math.log(p[real]), 2), 'math']] : [['the prompt ends here', 'strong']])])], say, meta);
      }
      const mean = lossSum / lossN;
      F.push([...strip(n), ...panel([['mean loss over the prompt', 'strong'], [H().fx(mean, 2) + ' per character', 'math'], ['perplexity e^loss = ' + H().fx(Math.exp(mean), 2), 'math']])],
        `Over the ${P.qpl(q, lossN, 'character')} it could check, the mean loss is ${q(mean, 2)}: as unsure, on average, as choosing among e^${q(mean, 2)} = ${q(Math.exp(mean), 2)} equally likely characters (its perplexity), against ${q.n(V)} for blind guessing.`, { phase: 'loss', mean, ppl: Math.exp(mean), lossN, V });
      // writing on
      const g = net.generate(ids, 24, { method: 'greedy' }), gen = tok.decode(g.ids.slice(n));
      F.push([...strip(n, [...gen]), ...panel([['greedy: always the likeliest', 'strong'], ['each new character is fed back in', 'legend']])],
        `Writing on by itself, feeding each character back in and always taking the likeliest, it continues “${sh(text)}” with “${sh(gen)}”.`, { phase: 'greedy', gen });
      const s = net.generate(ids, 24, { temperature: inp.temp }, inp.seed), sgen = tok.decode(s.ids.slice(n));
      const mk = s.steps.map((st) => st.probs[st.id]), avgP = mk.reduce((a, b) => a + b, 0) / mk.length;
      F.push([...strip(n, [...sgen], 'p11samp'), ...panel([['sampled at temperature ' + H().fx(inp.temp, 1), 'strong'], ['each character drawn at random', 'legend'], ['by its probability', 'legend']])],
        `Sampling instead, at temperature ${q(inp.temp, 1)} with seed ${q.n(inp.seed)}, draws each character by its probability: “${sh(sgen)}”. The characters it drew had probability ${q(avgP, 2)} on average.` + (inp.temp > 1.2 ? ' So hot a temperature flattens the choices and the spelling falls apart.' : inp.temp < 0.5 ? ' So cold a temperature is close to greedy.' : ''), { phase: 'sample', gen: sgen, avgP, temp: inp.temp, seed: inp.seed });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Next-character prediction', text: ['p(next | so far) = softmax(h_t W + b)', 'loss at t = −ln p(real next character)', 'perplexity = e^(mean loss)', 'greedy: take the argmax; sampling: draw from p'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => f.meta.phase === 'read' && f.meta.t >= 2 && fr[i + 1] && fr[i + 1].meta.phase === 'read', once: true, pick: 'option',
      ask: (fr, i) => 'What will the model guess most likely after the next character?',
      options: (fr, i) => { const tok = P.net('charLM').tokenizer, g = fr[i + 1].meta.guess; return [...new Set([g, ...[...'e at.'].map((c) => tok.index.get(c))])].filter((x) => x !== undefined).slice(0, 4).map((k) => '“' + sh(tok.vocab[k]) + '”'); },
      answer: (fr, i) => '“' + sh(P.net('charLM').tokenizer.vocab[fr[i + 1].meta.guess]) + '”',
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'The model gave the real next character probability 0.5. Its loss for that step is…', options: ['ln 2 ≈ 0.69', '0.5', '2', '0'], answer: 0, why: '−ln 0.5 = ln 2.' },
      { q: 'A perplexity of 3 means the model is, on average, as unsure as…', options: ['choosing among 3 equally likely characters', 'being wrong 3 times', 'reading 3 characters at once', 'a loss of 3'], answer: 0, why: 'Perplexity is e to the mean loss.' },
      { q: 'Why does greedy writing often repeat itself?', options: ['it always takes the likeliest character, so the same loop wins again', 'it runs out of characters', 'the temperature is too high', 'the model forgets its weights'], answer: 0, why: 'There is no randomness to break the loop.' },
      { q: 'Raising the sampling temperature makes the text…', options: ['more varied and more error-prone', 'always the same', 'shorter', 'better spelled'], answer: 0, why: 'It flattens the probabilities.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
