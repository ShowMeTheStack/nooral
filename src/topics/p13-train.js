// Part 13, training a tiny GPT: the stored run of tests/ref/p13_runs.mjs (the loss of every step, the validation
// loss every 50 steps and a sample at each checkpoint), drawn as it happened, then a fresh sample from the final
// weights; and scaling: five widths of the same model trained the same number of steps, their validation (or
// training) loss against their size, with a straight line fitted on log–log axes.
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml, C = () => S.Kit.curve;
  /** a moving average over the last w values */
  P.smooth = (xs, w) => { const out = []; let s = 0; xs.forEach((x, i) => { s += x; if (i >= w) s -= xs[i - w]; out.push(s / Math.min(i + 1, w)); }); return out; };
  P.valAt = (val, step) => { const v = val.find((x) => x[0] === step); return v ? v[1] : null; };

  S.topic({
    id: 'lm-gpt-training', family: 'p13-lm', order: 30, title: 'Training a tiny GPT',
    blurb: 'A GPT with 27 thousand numbers learns from 7,000 characters of harbour sentences, 16 windows of 32 characters a step. The loss falls from the cost of guessing to something far better, and samples drawn at checkpoints turn from noise into word-shaped text. Then draw your own sample from the finished model.',
    inputs: [
      { id: 'curve', kind: 'choice', label: 'Training loss', def: 'smooth', options: [{ value: 'smooth', label: 'averaged over 25 steps' }, { value: 'raw', label: 'every step (noisy)' }] },
      { id: 'temperature', kind: 'slider', label: 'Temperature of your sample', def: 0.8, min: 0.2, max: 1.5, step: 0.1, fmt: (v) => (+v).toFixed(1) },
      { id: 'seed', kind: 'number', label: 'Seed of your sample', def: 3, min: 1, max: 999 },
    ],
    presets: [
      { label: 'As trained', values: { curve: 'smooth', temperature: 0.8, seed: 3 } },
      { label: 'Raw loss', values: { curve: 'raw', temperature: 0.8, seed: 3 } },
      { label: 'Cautious sample', values: { curve: 'smooth', temperature: 0.4, seed: 11 } },
      { label: 'Wild sample', values: { curve: 'smooth', temperature: 1.4, seed: 5 } },
    ],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const J = P.D().gpt, F = H().frames(['step']), q = F.q, N = J.steps, lnV = Math.log(J.vocab.length);
      const tr = inp.curve === 'raw' ? J.hist : P.smooth(J.hist, 25), checks = J.samples.map((s) => s.step);
      const ax = C().axes({ prefix: 'gt:ax', x: 50, y: 30, w: 700, h: 200, xr: [0, N], yr: [1, 3.6], xticks: C().ticks(0, N, 8), yticks: [1, 1.5, 2, 2.5, 3, 3.5], xname: 'training step', yname: 'loss (nats per character)' });
      const plot = (upto) => {
        const items = [...ax.items];
        items.push(C().hline(ax, 'gt:uni', lnV, 'p13uni'), H().text('gt:unil', ax.X(N) - 4, ax.Y(lnV) - 6, 'guessing uniformly', 'tick', 12, 'end'));
        if (upto > 0) items.push(C().line(ax, 'gt:tr', tr.slice(0, upto).map((v, i) => [i + 1, v]), 'p13train'));
        const vs = J.val.filter((v) => v[0] <= upto);
        if (vs.length > 1) items.push(C().line(ax, 'gt:va', vs, 'p13valline'));
        vs.forEach((v) => items.push(C().dot(ax, 'gt:v' + v[0], v[0], v[1], 'p13val', 3.5)));
        items.push(...C().legend('gt:lg', 420, 285, [{ text: 'training batches', cls: 'p13train' }, { text: 'held-out text', cls: 'p13valline' }]));
        return items;
      };
      const samples = (upto, mine) => {
        const items = [H().text('gs:h', 0, 316, 'samples from the prompt “' + J.samples[0].text.slice(0, 11) + '” (temperature 0.8)', 'strong', 13)];
        J.samples.filter((s) => s.step <= upto).forEach((s, i) => {
          items.push(H().text('gs:s' + s.step, 66, 340 + i * 21, s.text, 'mono ' + (s.step === upto && !mine ? 'p13new' : ''), 12));
          items.push(H().text('gs:n' + s.step, 56, 340 + i * 21, 'step ' + s.step, 'tick', 12, 'end'));
        });
        return items;
      };
      checks.forEach((step, k) => {
        F.set('step', step);
        const v = P.valAt(J.val, step), s = J.samples[k];
        let say;
        if (step === 0) say = `Before training the weights are random, so every character is about equally likely: the loss on held-out text is ${q(v, 3)}, close to ln ${q.n(J.vocab.length)} = ${q(lnV, 3)}, the cost of guessing uniformly. The sample is noise.`;
        else {
          const prev = P.valAt(J.val, checks[k - 1]), tAvg = P.smooth(J.hist, 25)[step - 1];
          say = `Step ${q.n(step)}: the training loss (averaged over the last ${q.n(25)} steps) is ${q(tAvg, 2)} and the loss on held-out text ${q(v, 3)}, down from ${q(prev, 3)} at step ${q.n(checks[k - 1])}; perplexity e^${q(v, 3)} = ${q(Math.exp(v), 2)}. ` +
            (step <= 50 ? 'It has learned which characters are common: spaces and e, t, h.' : step <= 100 ? 'Common short words such as “the” and “he” appear.' : step <= 400 ? 'Words from the text begin to form, though most are misspelt.' : `Many words are spelt right and sentences end with full stops, but the sense is loose: the model sees only ${q.n(J.config.context)} characters at once.`);
        }
        F.push([...plot(step), ...samples(step)], say, { phase: 'check', step, val: v, prev: k ? P.valAt(J.val, checks[k - 1]) : null, prevStep: k ? checks[k - 1] : null, tAvg: step ? P.smooth(J.hist, 25)[step - 1] : null, avgW: 25, ctx: J.config.context, ppl: Math.exp(v), V: J.vocab.length, lnV, sample: s.text });
      });
      const out = P.gpt().complete(J.samples[0].text.slice(0, 11), 64, { temperature: inp.temperature }, inp.seed).text, end = P.valAt(J.val, N), gap = P.smooth(J.hist, 25)[N - 1];
      F.push([...plot(N), ...samples(N, true), ...P.textBox('gs:mine', 0, 482, 790, ['yours (T ' + H().fx(inp.temperature, 1) + ', seed ' + inp.seed + '): ' + out.slice(0, 60)], { cls: 'p13mine', size: 12 }).items],
        `The finished model, sampled with your temperature ${q(inp.temperature, 1)} and seed ${q.n(inp.seed)}, writes “${out.slice(11, 40)}…”. Its held-out loss ended at ${q(end, 3)} while the training batches averaged ${q(gap, 2)}: the gap is the part it memorised rather than learned. A real model reads billions of times more text and holds millions of times more numbers.`,
        { phase: 'mine', temperature: inp.temperature, seed: inp.seed, text: out, end, gap });
      return { frames: F.list, def: { sections: [{ id: 'gpt', kind: 'math', title: 'The tiny GPT', text: ['characters: ' + J.vocab.length + ' (letters, space, “.”, “,”)', 'width 32, 4 heads, 2 blocks, window 32', 'parameters: ' + J.params.toLocaleString('en'), 'loss: cross-entropy of the next character', 'AdamW, warm-up then cosine decay'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'check' && f.meta.step === 400,
      once: true, pick: 'option',
      ask: 'Will the loss on held-out text be lower at step 800 than now?',
      options: ['lower', 'about the same', 'higher'],
      answer: (fr, i) => { const a = fr[i].meta.val, b = fr[i + 1].meta.val; return Math.abs(b - a) < 0.01 ? 'about the same' : b < a ? 'lower' : 'higher'; },
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'Before training, the loss of a model over V tokens is close to…', options: ['ln V: it guesses about uniformly', '0', '1', 'V'], answer: 0, why: 'Random small weights give nearly equal probabilities.' },
      { q: 'Why is the loss on held-out text worth watching, not only the training loss?', options: ['the training loss also falls when the model memorises; held-out loss shows what generalises', 'it is cheaper', 'it is always lower', 'it decides the learning rate'], answer: 0, why: 'A growing gap between them means memorising.' },
      { q: 'The tiny model’s samples are word-like but make little sense mainly because…', options: ['it has few parameters, little text and sees only 32 characters at once', 'the sampling is broken', 'characters cannot form words', 'it was trained on noise'], answer: 0, why: 'Scale of model, data and context all limit it.' },
      { q: 'One training step here…', options: ['scores 16 windows of 32 characters and nudges every weight down the gradient', 'reads the whole text', 'adds a new word to the vocabulary', 'samples text'], answer: 0, why: 'A mini-batch of windows, one gradient step.' },
    ],
  });

  // ---- scaling ----
  const COLS = ['p13s0', 'p13s1', 'p13s2', 'p13s3', 'p13s4'];
  /** least squares of y on x: {a, b} with y ≈ a + b x */
  P.fitLine = (xs, ys) => { const n = xs.length, mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n; let sxy = 0, sxx = 0; xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; }); const b = sxy / sxx; return { a: my - b * mx, b }; };
  /** the final loss of a run: the last validation loss, or the training loss averaged over the last 50 steps */
  P.finalLoss = (run, measure) => (measure === 'val' ? run.val[run.val.length - 1][1] : run.hist.slice(-50).reduce((s, v) => s + v, 0) / 50);

  S.topic({
    id: 'lm-scaling', family: 'p13-lm', order: 40, title: 'Scaling: bigger models, lower loss',
    blurb: 'Train the same GPT at five widths, from about two thousand to sixty thousand numbers, for the same 400 steps on the same text. Plot the final loss against the number of parameters on a logarithmic axis and the points fall close to a smooth curve: loss falls roughly as a power of size. Large language models are built on that regularity, which holds only while there is enough data and training to feed the model.',
    inputs: [
      { id: 'measure', kind: 'choice', label: 'Loss measured on', def: 'val', options: [{ value: 'val', label: 'held-out text' }, { value: 'train', label: 'training batches (last 50 steps)' }] },
    ],
    presets: [
      { label: 'Held-out text', values: { measure: 'val' } },
      { label: 'Training batches', values: { measure: 'train' } },
    ],
    counters: [{ id: 'models', label: 'models' }],
    run(inp) {
      const runs = P.D().scale.runs, F = H().frames(['models']), q = F.q, steps = runs[0].hist.length;
      const lossOf = (r) => P.finalLoss(r, inp.measure), name = inp.measure === 'val' ? 'held-out loss' : 'training loss';
      const ys = runs.map(lossOf), lo = Math.floor(Math.min(...ys, ...runs.map((r) => Math.min(...r.val.map((v) => v[1])))) * 5) / 5 - 0.1, hi = 3.6;
      const A = C().axes({ prefix: 'sc:a', x: 50, y: 30, w: 300, h: 300, xr: [0, steps], yr: [lo, hi], xticks: C().ticks(0, steps, 4), yticks: C().ticks(lo, hi, 5), xname: 'step', yname: 'held-out loss' });
      const lx = runs.map((r) => Math.log10(r.params)), xr = [Math.floor(Math.min(...lx) * 2) / 2 - 0.1, Math.ceil(Math.max(...lx) * 2) / 2 + 0.1];
      const yr = [Math.floor(Math.min(...ys) * 10) / 10 - 0.1, Math.ceil(Math.max(...ys) * 10) / 10 + 0.1];
      const B = C().axes({ prefix: 'sc:b', x: 470, y: 30, w: 300, h: 300, xr, yr, xticks: [3, Math.log10(3000), 4, Math.log10(30000), 5].filter((v) => v >= xr[0] && v <= xr[1]), yticks: C().ticks(yr[0], yr[1], 5), fmtX: (v) => Math.round(Math.pow(10, v) / 1000) + 'k', xname: 'parameters (log scale)', yname: name });
      const pic = (k, fit) => {
        const items = [...A.items, ...B.items];
        runs.slice(0, k).forEach((r, i) => {
          items.push(C().line(A, 'sc:l' + i, r.val, COLS[i]));
          items.push(C().dot(B, 'sc:d' + i, Math.log10(r.params), ys[i], COLS[i], 5));
        });
        items.push(...C().legend('sc:lg', 0, 400, runs.map((r, i) => ({ text: 'width ' + r.d + ' (' + (r.params >= 1e4 ? (r.params / 1000).toFixed(0) + 'k' : r.params) + ')', cls: COLS[i], dot: true, tcls: i < k ? '' : 'p13faded' })), { size: 12 }));
        if (fit) items.push(C().line(B, 'sc:fit', Array.from({ length: 21 }, (_, j) => { const u = xr[0] + ((xr[1] - xr[0]) * j) / 20; return [u, Math.exp(fit.a + fit.b * u * Math.LN10)]; }), 'p13fit'));
        return items;
      };
      runs.forEach((r, k) => {
        F.count('models');
        const prev = k ? ys[k - 1] : null;
        F.push(pic(k + 1),
          `Width ${q.n(r.d)}: ${q.n(r.params)} parameters, trained ${q.n(steps)} steps. Its final ${name} is ${q(ys[k], 3)}` + (k ? (ys[k] < prev ? `, below the ${q(prev, 3)} of the model before it.` : `, no better than the ${q(prev, 3)} of the model before it.`) : '.'),
          { phase: 'model', k, d: r.d, params: r.params, loss: ys[k], prev, steps });
      });
      const fit = P.fitLine(runs.map((r) => Math.log(r.params)), ys.map(Math.log)), dbl = Math.pow(2, fit.b);
      F.push(pic(runs.length, fit),
        `A line fitted through the points as ln L = a + b ln N has slope b = ${q(fit.b, 3)}: each doubling of the parameters multiplies the loss by 2^b = ${q(dbl, 3)}, about ${q.pct(1 - dbl)} less.` + (inp.measure === 'val' ? ' With only a few thousand characters of text this cannot go on for long: a scaling law holds only while the data and the compute grow with the model.' : ' On the training batches the bigger models keep improving: more parameters can also memorise more.'),
        { phase: 'fit', a: fit.a, b: fit.b, dbl, less: 1 - dbl });
      return { frames: F.list, def: { sections: [{ id: 'sl', kind: 'math', title: 'A scaling law', text: ['L(N) ≈ c · N^b   (b < 0)', 'ln L ≈ ln c + b ln N: a line on log–log axes', 'doubling N multiplies L by 2^b', 'holds while data D and compute grow too'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => f.meta.phase === 'model' && f.meta.k === fr.filter((x) => x.meta.phase === 'model').length - 2,
      once: true, pick: 'option',
      ask: 'Will the widest model end with a lower loss than this one?',
      options: ['lower', 'higher'],
      answer: (fr, i) => (fr[i + 1].meta.loss < fr[i].meta.loss ? 'lower' : 'higher'),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'On log–log axes, a power law L = c·N^b looks like…', options: ['a straight line with slope b', 'a parabola', 'an exponential curve', 'a step'], answer: 0, why: 'ln L = ln c + b ln N.' },
      { q: 'If the text stayed at a few thousand characters while the model kept growing, the held-out loss would…', options: ['stop improving: the model would only memorise more', 'keep falling forever', 'fall faster', 'rise at once for every size'], answer: 0, why: 'Scaling needs data to grow along with parameters.' },
      { q: 'If b = −0.1, doubling the parameters multiplies the loss by about…', options: ['0.93', '0.5', '0.1', '2'], answer: 0, why: '2^−0.1 ≈ 0.933.' },
      { q: 'What did the training-batch loss show that the held-out loss did not?', options: ['bigger models also memorise more of their training text', 'smaller models are faster', 'nothing', 'that the data was shuffled'], answer: 0, why: 'Training loss keeps falling with size even when held-out loss stalls.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
