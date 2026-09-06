// Part 13, from a next-token predictor to an assistant: the three training stages as a diagram with an example of
// each stage's data, and preference tuning run for real on the tiny GPT: a copy of it is nudged, by the direct
// preference loss −ln σ(β·((log p_A − log p_B) − (the same under the frozen original))), to raise the chosen
// answer and lower the rejected one.
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml;
  const STAGES = [
    { id: 'pre', title: 'pretraining', sub: 'next token, huge text' },
    { id: 'it', title: 'instruction tuning', sub: 'fine-tuning on requests' },
    { id: 'pt', title: 'preference tuning', sub: 'better of two answers' },
  ];
  /** log p of answer after prompt, summed over the answer's characters, as a tensor in the graph */
  const logp = (net, prompt, answer) => {
    const T = S.T, tok = net.tokenizer, a = tok.encode(prompt), b = tok.encode(answer), ids = a.concat(b);
    const x = ids.slice(0, -1), y = ids.slice(1).map((v, i) => (i >= a.length - 1 ? v : -1));
    return T.neg(T.crossEntropy(net.forward(T.tensor([x])), [y], { ignoreIndex: -1, reduction: 'sum' }));
  };
  /** the preference run: a copy of the trained GPT tuned for `steps` Adam steps; log-probabilities before and after */
  P.prefer = (prompt, A, B, beta, steps, lr) => P.once('pref:' + [prompt, A, B, beta, steps, lr].join('|'), () => {
    const T = S.T, base = P.gpt(), J = P.D().gpt;
    const net = S.DL.build('gpt', Object.assign({}, J.config, { tokenizer: { kind: 'char', vocab: J.vocab } }));
    net.loadState(base.stateDict());
    net.eval();
    const ref = T.noGrad(() => [logp(base, prompt, A).item(), logp(base, prompt, B).item()]);
    const opt = T.optim.make('adam', net.parameters(), { lr });
    const hist = [];
    for (let s = 0; s <= steps; s++) {
      opt.zeroGrad();
      const la = logp(net, prompt, A), lb = logp(net, prompt, B);
      const margin = T.scale(T.sub(T.sub(la, lb), T.tensor(ref[0] - ref[1])), beta), loss = T.softplus(T.neg(margin));
      hist.push({ step: s, la: la.item(), lb: lb.item(), loss: loss.item() });
      if (s === steps) break;
      loss.backward();
      opt.step();
    }
    return { ref, hist, net };
  });

  S.topic({
    id: 'lm-training-stages', family: 'p13-beyond', order: 10, title: 'Pretraining, instruction tuning and preference tuning',
    blurb: 'An assistant is the same kind of next-token model, trained in stages. Pretraining on a vast amount of text teaches language and facts; instruction tuning, which is fine-tuning on many tasks written as requests, teaches it to answer; preference tuning on pairs of answers, one judged better, teaches which answers people want. The last stage is run here, for real, on the tiny GPT.',
    inputs: [
      { id: 'pair', kind: 'choice', label: 'Preference pair', def: 'sea', options: [{ value: 'sea', label: 'the sea is: calm and grey / made of bread' }, { value: 'bread', label: 'the baker sells: warm bread / old boots' }, { value: 'keeper', label: 'the keeper: lights the lamp / eats the boat' }] },
      { id: 'beta', kind: 'slider', label: 'β (strength)', def: 0.5, min: 0.1, max: 2, step: 0.1, fmt: (v) => (+v).toFixed(1) },
      { id: 'steps', kind: 'number', label: 'Tuning steps', def: 10, min: 1, max: 30 },
    ],
    presets: [
      { label: 'The sea', values: { pair: 'sea', beta: 0.5, steps: 10 } },
      { label: 'The baker', values: { pair: 'bread', beta: 0.5, steps: 10 } },
      { label: 'Strong push', values: { pair: 'keeper', beta: 2, steps: 30 } },
      { label: 'One step', values: { pair: 'sea', beta: 0.5, steps: 1 } },
    ],
    counters: [{ id: 'stage', label: 'stage' }],
    run(inp) {
      const PAIRS = { sea: ['the sea is ', 'calm and grey.', 'made of bread.'], bread: ['the baker sells ', 'warm bread.', 'old boots.'], keeper: ['the keeper ', 'lights the lamp.', 'eats the boat.'] };
      const [prompt, A, B] = PAIRS[inp.pair], F = H().frames(['stage']), q = F.q, J = P.D().gpt;
      const boxes = (lit) => {
        const items = [];
        STAGES.forEach((s, i) => {
          items.push({ key: 'ts:s' + i, kind: 'state', x: 135 + i * 260, y: 50, w: 176, h: 62, rx: 10, lines: [s.title, s.sub], size: 13, cls: 'block p13stage ' + (i === lit ? 'act' : lit >= 0 && i > lit ? 'p13later' : '') });
          if (i) items.push({ key: 'ts:a' + i, kind: 'edge', from: 'ts:s' + (i - 1), to: 'ts:s' + i, head: true, cls: 'p13flow' });
        });
        return items;
      };
      const card = (title, lines, cls) => P.textBox('ts:card', 0, 120 + 0, 790, [title].concat(lines), { cls: 'p13card ' + (cls || ''), tcls: (i) => (i ? '' : 'strong'), lh: 22 }).items;
      F.push([...boxes(-1), ...card('one model, three kinds of data', ['the weights carry over from stage to stage; only the data and the goal change', 'each stage uses far less data than the one before it, but shapes the behaviour more'])],
        'An assistant starts as a next-token predictor and is shaped in three stages, left to right. The same network passes through all of them: each stage starts from the weights the last one left.', { phase: 'all' });
      F.set('stage', 1);
      F.push([...boxes(0), ...card('pretraining data: ordinary text, read to predict the next token', ['“' + S.P13T.harbour.slice(0, 74) + '”', '“' + S.P13T.harbour.slice(102, 176) + '”', 'goal: lower the cross-entropy of every next token', 'our tiny GPT: ' + S.P13T.harbour.length.toLocaleString('en') + ' characters, ' + J.steps + ' steps; a large model: trillions of tokens'])],
        `Pretraining is what our tiny GPT did: read text and predict the next token. It saw ${q.n(S.P13T.harbour.length)} characters and ended with a held-out loss of ${q(J.val[J.val.length - 1][1], 3)}. A large model reads a large part of the public text in the world; from it come grammar, facts and style, but it only continues text: ask it a question and it may answer with more questions.`,
        { phase: 'pre', chars: S.P13T.harbour.length, loss: J.val[J.val.length - 1][1] });
      F.set('stage', 2);
      F.push([...boxes(1), ...card('fine-tuning data: examples of a task, input then output', ['input: the tide is high.        output: the boats go out.', 'input: the fog comes in.        output: the keeper rings the bell.', 'input: the storm is over.       output: the sailors mend the nets.', 'goal: the same next-token loss, counted on the outputs only'])],
        'The middle stage is fine-tuning: training continues on a small set of examples, with a smaller learning rate, so the model keeps what it knows and learns the new format. The loss is the same next-token cross-entropy, counted only on the answer part. Fine-tuned on one task, a model does that one job; an assistant is fine-tuned on many tasks at once.', { phase: 'ft' });
      F.push([...boxes(2), ...card('instruction data: many tasks, each written as a request and a good reply', ['request: name two things the baker sells.     reply: bread and rolls.', 'request: say it more simply: the vessel departs.     reply: the boat leaves.', 'request: is a gull a fish?     reply: no, a gull is a bird.', 'goal: answer the request, whatever the task'])],
        'Instruction tuning is that fine-tuning on thousands of different tasks, each phrased as a request with a good reply written by people. The model learns the general habit of answering what was asked, so it can follow requests it never saw in training.', { phase: 'it' });
      // preference tuning, for real
      F.set('stage', 3);
      const run = P.prefer(prompt, A, B, inp.beta, inp.steps, 1e-4), h0 = run.hist[0], hN = run.hist[run.hist.length - 1];
      const nA = A.length, nB = B.length, prefA = h0.la / nA > h0.lb / nB;
      const bars = (h, k) => P.bars('ts:pb' + k, 30, 290 + k * 0, [{ label: 'A', v: Math.exp(h.la / nA), cls: 'good', text: 'p per char ' + H().fx(Math.exp(h.la / nA), 3) }, { label: 'B', v: Math.exp(h.lb / nB), cls: 'bad', text: 'p per char ' + H().fx(Math.exp(h.lb / nB), 3) }], { lw: 20, bw: 300, rh: 30, max: 1 });
      const prefCard = (h, after) => [...P.textBox('ts:card', 0, 120, 790, ['preference data: a prompt and two answers, A judged better than B', 'prompt: “' + prompt + '”     A: “' + A + '”     B: “' + B + '”', 'loss = −ln σ(β·((log p_A − log p_B) − the same under the original))'], { cls: 'p13card', tcls: (i) => (i ? 'mono' : 'strong'), lh: 22 }).items,
        H().text('ts:ph', 0, 276, after ? 'after ' + inp.steps + ' tuning steps' : 'the tiny GPT before tuning', 'strong', 13), ...bars(h, 0),
        H().text('ts:loss', 420, 320, 'preference loss ' + H().fx(h.loss, 3), 'legend', 13)];
      F.push([...boxes(2), ...prefCard(h0, false)],
        `Preference tuning shows the model a prompt and two answers, one judged better. Before tuning, the tiny GPT gives answer A an average of ${q(Math.exp(h0.la / nA), 3)} per character and answer B ${q(Math.exp(h0.lb / nB), 3)}: it ${prefA ? 'already leans to A' : 'leans to the worse answer, B'}, since both are made of words it knows. The tuned copy starts equal to the original, so the loss starts at ln 2 = ${q(h0.loss, 3)}.`,
        { phase: 'pref0', pA: Math.exp(h0.la / nA), pB: Math.exp(h0.lb / nB), loss: h0.loss, la: h0.la, lb: h0.lb, nA, nB, prefA });
      F.push([...boxes(2), ...prefCard(hN, true)],
        `After ${P.qpl(q, inp.steps, 'step')} of gradient descent on that loss with β = ${q(inp.beta, 1)}, log p(A) went from ${q(h0.la, 2)} to ${q(hN.la, 2)} and log p(B) from ${q(h0.lb, 2)} to ${q(hN.lb, 2)}; the loss fell to ${q(hN.loss, 3)}. The model was not told why A is better, only that it is. The reference term keeps it from drifting far from what pretraining taught.`,
        { phase: 'pref', steps: inp.steps, beta: inp.beta, la0: h0.la, la: hN.la, lb0: h0.lb, lb: hN.lb, loss: hN.loss });
      F.push([...boxes(-1), ...card('what each stage gives', ['pretraining: language, facts, style (and the flaws of its text)', 'instruction tuning (fine-tuning on many tasks): the habit of answering requests', 'preference tuning: answers people judge helpful, honest and harmless'])],
        'Each stage changes the same weights with a different kind of data and a different goal. Most of what the model knows comes from pretraining; the later stages, with much less data, mostly change how it behaves.', { phase: 'end' });
      return { frames: F.list, def: { sections: [{ id: 'dpo', kind: 'math', title: 'Direct preference loss', text: ['Δ = log p(A|x) − log p(B|x)', 'Δ_ref = the same under the frozen original', 'loss = −ln σ(β (Δ − Δ_ref))', 'starts at ln 2 (the copy equals the original)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'pref0',
      once: true, pick: 'option',
      ask: 'After tuning, will log p(B), the worse answer, go up or down?',
      options: ['up', 'down'],
      answer: (fr, i) => (fr[i + 1].meta.lb < fr[i + 1].meta.lb0 ? 'down' : 'up'),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'Which stage uses by far the most data?', options: ['pretraining', 'instruction tuning', 'preference tuning'], answer: 0, why: 'It reads a huge amount of ordinary text; the later stages use small curated sets.' },
      { q: 'What does a pretrained-only model do with a question?', options: ['continues the text, which may not be an answer', 'always answers it', 'refuses it', 'translates it'], answer: 0, why: 'It predicts likely next text; answering is taught later.' },
      { q: 'Preference data consists of…', options: ['a prompt with two answers, one judged better', 'single correct answers', 'unlabelled text', 'images'], answer: 0, why: 'The model learns from comparisons.' },
      { q: 'At the first step of direct preference tuning the loss is ln 2 because…', options: ['the tuned copy still equals the reference, so Δ − Δ_ref = 0 and σ(0) = ½', 'the answers are equally long', 'β is one', 'the learning rate is zero'], answer: 0, why: '−ln σ(0) = ln 2.' },
      { q: 'Why keep a frozen reference model in preference tuning?', options: ['so the tuned model cannot drift far from what pretraining taught', 'to save memory', 'to choose the answers', 'to sample faster'], answer: 0, why: 'The loss rewards changes relative to the reference, which limits drift.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
