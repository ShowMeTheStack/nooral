// Part 13, limits of language models, shown on the tiny GPT where it is honest to: asked about something its text
// never mentions it still writes fluent, confident text (a small hallucination), and it copies a skew planted in its
// training text (sailors mostly "he", weavers mostly "she") into its probabilities (a small bias). Each with what is
// done about it.
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml;
  const REAL = 'the sailor said that he would sail to the far rocks.';
  /** the probability the model gave each character of text after prompt (T = 1) */
  P.scoreAfter = (net, prompt, text) => { const ids = net.tokenizer.encode(prompt + text), L = P.logits(net, ids.slice(0, net.context)), out = []; for (let i = prompt.length; i < Math.min(ids.length, net.context); i++) out.push(P.softmax(L[i - 1])[ids[i]]); return out; };
  /** p(word | prompt) by the chain rule over its characters */
  P.chain = (net, prompt, word) => P.scoreAfter(net, prompt, word).reduce((a, b) => a * b, 1);
  P.countIn = (text, re) => (text.match(re) || []).length;

  S.topic({
    id: 'lm-limits', family: 'p13-beyond', order: 30, title: 'Limits: invented facts and copied bias',
    blurb: 'A language model learns what text is likely, not what is true. Ask it about something its training never covered and it still writes fluent, confident text: a hallucination. And whatever patterns its text holds, fair or not, it reproduces. The tiny GPT shows both in miniature, on a text where we know exactly what it saw.',
    inputs: [
      { id: 'prompt', kind: 'text', label: 'Prompt about something new', def: 'the king said that ', check: P.checkText('Prompt', 4, 20) },
      { id: 'role', kind: 'choice', label: 'Role for the bias test', def: 'weaver', options: [{ value: 'sailor', label: 'the sailor' }, { value: 'weaver', label: 'the weaver' }, { value: 'keeper', label: 'the keeper' }] },
      { id: 'seed', kind: 'number', label: 'Seed', def: 2, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A king', values: { prompt: 'the king said that ', role: 'weaver', seed: 2 } },
      { label: 'A dragon', values: { prompt: 'the dragon ', role: 'sailor', seed: 5 } },
      { label: 'The moon', values: { prompt: 'on the moon ', role: 'keeper', seed: 8 } },
    ],
    counters: [{ id: 'part', label: 'limit' }],
    run(inp) {
      const net = P.gpt(), F = H().frames(['part']), q = F.q, text = S.P13T.harbour, N = 40;
      const words = inp.prompt.split(/[ .,]+/).filter(Boolean), counts = words.map((w) => P.countIn(text, new RegExp('\\b' + w + '\\b', 'g')));
      const newWords = words.filter((w, i) => counts[i] === 0);
      const strip = (s, n0) => [H().text('li:h', 0, 16, n0 === undefined ? 'the prompt (· is a space)' : 'prompt and continuation (· is a space)', 'strong', 13), ...P.cells('li:c', 0, 30, Array.from(s).slice(0, 32), { w: 22, cls: (i) => (n0 !== undefined && i >= n0 ? 'new' : '') })];
      const card = (title, lines) => P.textBox('li:card', 0, 300, 790, [title].concat(lines), { cls: 'p13card', tcls: (i) => (i ? '' : 'strong'), lh: 22 }).items;
      F.set('part', 1);
      F.push([...strip(inp.prompt), ...P.panel('li:wp', 0, 80, 790, 190, 'how often each word of the prompt appears in the training text'), ...P.bars('li:w', 20, 116, words.map((w, i) => ({ label: w, v: counts[i], cls: counts[i] ? 'soft' : 'bad', text: '×' + counts[i] })), { lw: 90, bw: 420, rh: 24, max: Math.max(1, ...counts) })],
        newWords.length ? `The prompt mentions ${H().list(newWords.map((w) => '“' + w + '”'))}, which the training text never contains: ${newWords.length === 1 ? 'its count is' : 'their counts are'} ${q.n(0)}. The model has learned nothing about ${newWords.length === 1 ? 'it' : 'them'}.` : `Every word of this prompt appears in the training text, at least ${q.n(Math.min(...counts))} times, but never in this combination.`,
        { phase: 'words', counts, zero: 0, min: Math.min(...counts), newWords });
      const g = net.generate(net.tokenizer.encode(inp.prompt), N, { temperature: 0.8 }, inp.seed), cont = g.ids.slice(inp.prompt.length).map((i) => P.vocab()[i]).join('');
      const pc = P.scoreAfter(net, inp.prompt, cont), mean = pc.reduce((a, b) => a + b, 0) / pc.length;
      const rp = P.scoreAfter(net, REAL.slice(0, 21), REAL.slice(21)), rmean = rp.reduce((a, b) => a + b, 0) / rp.length;
      const box = P.textBox('li:out', 0, 90, 790, [inp.prompt + cont], { cls: 'p13mine' }).items;
      F.push([...strip(inp.prompt + cont, inp.prompt.length), ...box],
        `Asked anyway, it writes “${cont.slice(0, 30)}…” at temperature ${q(0.8, 1)}. Nothing stops it: there is no step where it checks what it knows. It averages p = ${q(mean, 2)} per character on its own words, the confidence of familiar text.`,
        { phase: 'gen', text: cont, mean, T: 0.8 });
      F.push([...strip(inp.prompt + cont, inp.prompt.length), ...box, ...P.panel('li:cp', 0, 150, 790, 130, 'average probability per character'), ...P.bars('li:cb', 20, 186, [{ label: 'invented text', v: mean, cls: 'bad', text: H().fx(mean, 2) }, { label: 'a true sentence', v: rmean, cls: 'good', text: H().fx(rmean, 2) }], { lw: 120, bw: 420, rh: 34, max: 1 }), ...card('why this happens, and what helps', ['the model scores text by how likely it sounds, not by whether it is true', 'helps: retrieving sources and quoting them, training it to say “I do not know”,', 'checking answers against a database, and asking for its sources'])],
        `A sentence it really read, “${REAL}”, scores ${q(rmean, 2)} per character; its invented text scores ${q(mean, 2)}. ` + (Math.abs(mean - rmean) < 0.15 ? 'The two are close: fluency gives no warning that the content is made up.' : mean > rmean ? 'The invented text even scores higher: likely-sounding is not the same as true.' : 'The invented text scores lower, but nothing in the output shows the reader that difference.'),
        { phase: 'conf', mean, rmean });
      // bias
      F.set('part', 2);
      const role = inp.role, pre = 'the ' + role + ' said that ';
      const nHe = P.countIn(text, new RegExp('the ' + role + ' said that he ', 'g')), nShe = P.countIn(text, new RegExp('the ' + role + ' said that she ', 'g'));
      const aHe = P.countIn(text, /\bhe\b/g), aShe = P.countIn(text, /\bshe\b/g);
      F.push([...strip(pre), ...P.panel('li:bp', 0, 80, 790, 190, 'in the training text, “' + pre + '” is followed by'), ...P.bars('li:bc', 20, 116, [{ label: 'he', v: nHe, cls: 'p13he', text: '×' + nHe }, { label: 'she', v: nShe, cls: 'p13she', text: '×' + nShe }], { lw: 60, bw: 300, rh: 34, max: Math.max(1, nHe, nShe, aHe, aShe) }), H().text('li:all', 20, 210, 'the words “he” and “she” anywhere in the text:', 'legend', 12), ...P.bars('li:ba', 20, 218, [{ label: 'he', v: aHe, cls: 'p13he', text: '×' + aHe }, { label: 'she', v: aShe, cls: 'p13she', text: '×' + aShe }], { lw: 60, bw: 300, rh: 24, max: Math.max(1, nHe, nShe, aHe, aShe) })],
        `Now bias. The harbour text was written with a skew: after “${pre}” comes “he” ${P.qpl(q, nHe, 'time')} and “she” ${P.qpl(q, nShe, 'time')}, and in the whole text “he” stands ${P.qpl(q, aHe, 'time')} and “she” ${P.qpl(q, aShe, 'time')}. Nothing about the jobs requires it; it is only what the text happened to say.`,
        { phase: 'counts', nHe, nShe, aHe, aShe });
      const pHe = P.chain(net, pre, 'he '), pShe = P.chain(net, pre, 'she ');
      F.push([...strip(pre), ...P.panel('li:bp', 0, 80, 790, 190, 'the model’s probability of the next word after “' + pre + '”'), ...P.bars('li:bm', 20, 120, [{ label: 'he', v: pHe, cls: 'p13he', text: H().fx(pHe, 3) }, { label: 'she', v: pShe, cls: 'p13she', text: H().fx(pShe, 3) }], { lw: 60, bw: 420, rh: 40, max: Math.max(pHe, pShe) })],
        `The model, multiplying its probabilities character by character, gives “he” ${q(pHe, 3)} and “she” ${q(pShe, 3)}` + (pHe > pShe === nHe > nShe ? `: it has copied the skew of its text, “${pHe > pShe ? 'he' : 'she'}” about ${q(Math.max(pHe, pShe) / Math.min(pHe, pShe), 1)} times as likely.` : `: it ignores the ${q.n(nHe + nShe)} sentences about this role and follows, even amplifies, the text’s overall habit, where “he” is the commoner word. A small model leans on the commonest pattern; the minority case is erased, a typical form of bias.`),
        { phase: 'model', pHe, pShe, ratio: Math.max(pHe, pShe) / Math.min(pHe, pShe), nHe, nShe, n: nHe + nShe });
      F.push([...strip(pre), ...card('bias in, bias out, and what helps', ['a model trained on text repeats that text’s patterns, including unfair ones', 'helps: measuring outputs across groups, balancing or filtering training data,', 'preference tuning against stereotyped answers, and testing before release'])],
        'A large model reads a vast amount of human writing, with every skew in it, and returns them unless something is done. The remedies are measurement first, then better data and tuning, then checking again; none of them is ever finished.', { phase: 'fix' });
      return { frames: F.list, def: { sections: [{ id: 'lim', kind: 'math', title: 'What the model optimises', text: ['minimise −ln p(next token | text)', 'likely ≠ true: no step checks facts', 'p(he | the sailor said that)', '  = p(h) · p(e | h) · p(space | he)', 'skews in the data become skews in p'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'counts',
      once: true, pick: 'option',
      ask: 'Which word will the model think more likely next, “he” or “she”?',
      options: ['he', 'she'],
      answer: (fr, i) => (fr[i + 1].meta.pHe > fr[i + 1].meta.pShe ? 'he' : 'she'),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'Why can a language model state something false with confidence?', options: ['it is trained to produce likely text, and nothing checks facts', 'it lies on purpose', 'its vocabulary is too small', 'temperature is always too high'], answer: 0, why: 'Likelihood under the training text is not truth.' },
      { q: 'Which of these reduces invented answers?', options: ['retrieving real sources and answering from them', 'raising the temperature', 'a longer prompt with no facts', 'removing the KV cache'], answer: 0, why: 'Grounding the answer in sources gives it something true to copy.' },
      { q: 'Where does the bias shown here come from?', options: ['the training text’s skew', 'the tokenizer', 'the optimiser', 'the random seed'], answer: 0, why: 'The model copies the statistics of what it read.' },
      { q: 'The probability of a word such as “she ” after a prompt is…', options: ['the product of the probabilities of its characters, each given the ones before', 'the probability of its first letter', 'the count of the word', 'always one half'], answer: 0, why: 'That is the chain rule of probability.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
