// Part 12, a tiny transformer learning a toy task, and the key–value cache. The training runs were made offline by
// tests/ref/p12t_runs.mjs (they take seconds, a topic must draw at once) and are kept in p12-0-data.js: the loss at
// every step and, at checkpoints, the attention on a fixed probe, the score on new sequences and the model's own
// answer. The cache topic runs the trained model live, step by step, with and without the cache.
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const WHAT = { reverse: 'reverses', copy: 'copies', sort: 'sorts' };

  S.topic({
    id: 'transformer-train', family: 'p12-train', order: 10, title: 'A tiny transformer learns to reverse',
    blurb: 'A decoder-only transformer with two blocks of two heads, 16 numbers wide (under five thousand weights), is trained on sequences like 2 7 4 0 5 = 5 0 4 7 2: five digits, a separator, and the digits reversed (or copied, or sorted). Only the answer is scored. Watch the loss fall, the answers on new sequences go from wrong to right, and one attention head form the pattern the task needs.',
    inputs: [{ id: 'task', kind: 'choice', label: 'Task', def: 'reverse', options: P.TASK_OPTS }],
    presets: [{ label: 'Reverse', values: { task: 'reverse' } }, { label: 'Copy', values: { task: 'copy' } }, { label: 'Sort', values: { task: 'sort' } }],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const F = H().frames(['step']), q = F.q, R = P.run(inp.task), probe = R.probe, full = R.probe.slice(0, 5).concat([P.SEP], P.answer(inp.task, R.probe.slice(0, 5))), toks = probe.map(P.tok);
      const lg = (v) => Math.log10(v);
      const A = C().axes({ prefix: 'tl', x: 70, y: 60, w: 290, h: 190, xr: [0, R.steps], yr: [-2, 0.5], yticks: [-2, -1, 0], fmtY: (v) => String(+Math.pow(10, v).toPrecision(1)), xname: 'training step', yname: 'loss (log scale)' });
      const src = (i) => P.sourceOf(full, 5, i);
      const hits = (w) => { let k = 0; for (let i = 5; i < probe.length; i++) if (P.argmax(w[i]) === src(i)) k++; return k; };
      const mapsAt = (c) => c.maps.map((m, k) => P.mat('tm' + k, 440 + (k % 2) * 180, 84 + Math.floor(k / 2) * 200, m.w || m.weights, { rows: false, cols: false, cw: 14, ch: 14, gap: 1, fmt: null, scheme: 'seq', max: 1, title: 'layer ' + m.layer + ' head ' + m.head + ': ' + hits(m.weights) + ' of 5' }).items).flat();
      const answerStrip = (c) => {
        const it = P.strip('ta', 70, 330, full.slice(0, 6).map(P.tok).concat(c.out.map(P.tok)), { size: 13, gap: 3, minW: 24, cls: (i) => (i < 6 ? '' : c.out[i - 6] === full[i] ? 'p12ans' : 'p12wrong') }).items;
        return [H().text('ta:h', 70, 318, 'the model’s own answer for the probe', 'strong', 13), ...it];
      };
      let best = { k: 0, n: -1 };
      R.ckpts.forEach((c, idx) => {
        F.set('step', c.step);
        const upto = R.loss.slice(0, c.step).map((v, i) => [i + 1, lg(v)]);
        const items = [...A.items, ...(upto.length > 1 ? [C().line(A, 'tl:l', upto, 'p12a')] : []), ...(c.step ? [C().dot(A, 'tl:d', c.step, lg(c.loss), 'p12c')] : []), ...mapsAt(c), ...answerStrip(c), H().text('tl:x', 70, 420, 'exact answers on 100 new sequences: ' + c.exact, 'legend', 13)];
        const right = c.out.filter((v, i) => v === full[6 + i]).length, hs = c.maps.map((m) => hits(m.weights)), top = hs.indexOf(Math.max(...hs));
        if (hs[top] > best.n) best = { k: top, n: hs[top] };
        let say;
        if (!c.step) say = `Before training the weights are random. The model answers the probe ${H().list(toks.slice(0, 5).map((t, i) => P.qtok(q, probe[i])))} with ${H().list(c.out.map((v) => P.qtok(q, v)))}, ${q.n(right)} of ${q.n(5)} digits right, and gets ${q.n(c.exact)} of ${q.n(c.n)} new sequences exactly. Its attention is a blur: no head points where it should.`;
        else say = `Step ${q.n(c.step)}: the loss is ${q.sg(c.loss, 3)}. On the probe it answers ${H().list(c.out.map((v) => P.qtok(q, v)))} (${q.n(right)} of ${q.n(5)} right) and ${q.n(c.exact)} of ${q.n(c.n)} new sequences come out exactly right. The sharpest head, layer ${q.n(c.maps[top].layer)} head ${q.n(c.maps[top].head)}, points at the needed digit for ${q.n(hs[top])} of the ${q.n(5)} answers.`;
        F.push(items, say, { phase: 'ckpt', idx, step: c.step, loss: c.loss, exact: c.exact, n: c.n, out: c.out, right, hits: hs, top, tl: c.maps[top].layer, th: c.maps[top].head, answers: 5, probe: probe.slice(0, 5) });
      });
      const last = R.ckpts[R.ckpts.length - 1], half = R.ckpts.find((c) => c.exact >= 50);
      F.push(F.last().items, `After ${q.n(last.step)} steps (about ${q.n(Math.round((last.step * 32) / 1000) * 1000)} sequences seen, in batches of ${q.n(32)}) the model ${WHAT[inp.task]} ${q.n(last.exact)} of ${q.n(last.n)} new sequences exactly.` + (half ? ` It first passed half at step ${q.n(half.step)}.` : '') + ` Nobody wrote the rule: the pointing head was found by gradient descent because it lowers the loss.`, { phase: 'end', step: last.step, seen: Math.round((last.step * 32) / 1000) * 1000, batch: 32, exact: last.exact, n: last.n, half: half ? half.step : -1 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The training set-up', text: ['sequence: x₁ … x₅ = y₁ … y₅', 'model: 2 causal blocks, 2 heads, d = 16', 'loss: cross-entropy on y₁ … y₅ only', 'optimiser: AdamW, cosine schedule'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'ckpt' && f.meta.idx === 2, once: true, pick: 'option',
      ask: 'At the last checkpoint, how many of the 100 new sequences will the model get exactly right?',
      options: ['fewer than 50', '50 to 89', '90 or more'],
      answer: (fr) => { const e = fr.find((f) => f.meta.phase === 'end').meta.exact; return e < 50 ? 'fewer than 50' : e < 90 ? '50 to 89' : '90 or more'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'end').say,
    }],
    quiz: [
      { q: 'Before training, the loss is about ln 9 ≈ 2.2. Why that number?', options: ['a random model spreads its chances over the 9 tokens about evenly', 'the model starts perfectly', 'the learning rate is 2.2', 'there are 2.2 heads'], answer: 0, why: 'Cross-entropy of a uniform guess over 9 tokens is ln 9.' },
      { q: 'To reverse five digits, the answer position just after “=” must read…', options: ['the last input digit', 'the first input digit', 'the separator', 'itself'], answer: 0, why: 'The first answer digit is the last input digit.' },
      { q: 'Why are the input digits not scored, only the answer?', options: ['the input digits are random and cannot be predicted', 'to save time', 'because they are padding', 'they are scored twice'], answer: 0, why: 'Predicting a random digit only adds noise to the loss.' },
      { q: 'Sorting is harder for this model than copying because…', options: ['each answer digit depends on comparing all the inputs, not on one fixed position', 'sorted lists are longer', 'it has a different vocabulary', 'the separator differs'], answer: 0, why: 'Copying and reversing are pure pointing; sorting needs comparison.' },
    ],
  });

  S.topic({
    id: 'kv-cache-attention', family: 'p12-train', order: 20, title: 'The key–value cache',
    blurb: 'A decoder generates one token at a time, and under the causal mask the keys and values of earlier positions never change. So instead of running the whole sequence again for every new token, the model keeps each layer’s keys and values in a cache and computes only the newest row. Here the trained tiny transformer writes its answer token by token, with its first layer’s cache (one head) drawn as it grows, and the result checked against full recomputation.',
    inputs: [
      { id: 'task', kind: 'choice', label: 'Trained model', def: 'reverse', options: P.TASK_OPTS },
      { id: 'digits', kind: 'string', label: 'Five digits (0 to 7)', def: '27405', alphabet: '01234567', min: 5, max: 5 },
      { id: 'head', kind: 'choice', label: 'Head shown', def: '1', options: [{ value: '0', label: 'head 0' }, { value: '1', label: 'head 1' }] },
    ],
    presets: [
      { label: 'Reverse', values: { task: 'reverse', digits: '27405', head: '1' } },
      { label: 'Sort', values: { task: 'sort', digits: '62517', head: '0' } },
      { label: 'Copy', values: { task: 'copy', digits: '04471', head: '1' } },
    ],
    counters: [{ id: 'rows', label: 'rows projected (with cache)' }],
    run(inp) {
      const T = S.T, F = H().frames(['rows']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), prompt = x.concat([P.SEP]), hd = +inp.head, L = net.blocks.length;
      const cache = net.newCache(), seq = prompt.slice(), steps = [];
      let z = T.noGrad(() => net.forward(T.tensor([prompt]), { cache }));
      const snap = () => { const c = cache.layers[0], k = c.k.toArray()[0][hd], v = c.v.toArray()[0][hd]; return { k, v, w: net.outputs['block0.attn'].toArray()[0][hd] }; };
      const first = snap();
      let withC = prompt.length, without = 0;
      for (let s = 0; s < 5; s++) {
        const row = Array.from(T.slice(z, 0, z.shape[1] - 1, null).values()), id = P.argmax(row);
        const fullZ = net.predict(T.tensor([seq])), fullRow = Array.from(T.slice(fullZ, 0, seq.length - 1, null).values());
        const diff = Math.max(...row.map((v, i) => Math.abs(v - fullRow[i])));
        without += seq.length;
        seq.push(id);
        const st = { id, t: seq.length - 1, diff };
        if (s < 4) { z = T.noGrad(() => net.forward(T.tensor([[id]]), { cache })); withC += 1; Object.assign(st, snap()); }
        steps.push(st);
      }
      const dk = first.k[0].length, want = P.answer(inp.task, x);
      const toks = (n) => seq.slice(0, n).map(P.tok);
      const fmt = (v) => H().fx(v, 1).replace(/^0\./, '.').replace(/^−0\./, '−.');
      const kmax = Math.max(...steps.filter((s) => s.k).flatMap((s) => s.k.flat().map(Math.abs)), ...first.k.flat().map(Math.abs)), vmax = Math.max(...steps.filter((s) => s.v).flatMap((s) => s.v.flat().map(Math.abs)), ...first.v.flat().map(Math.abs));
      const picture = (c, n, newRow, wRow) => {
        const rows = toks(n).map((t, i) => i + ' ' + t);
        const items = [
          ...P.mat('kc', 90, 92, c.k, { rows, cols: Array.from({ length: dk }, (_, j) => String(j)), cw: 30, ch: 22, max: kmax, fmt, size: 11, title: 'K cache (layer 0, head ' + hd + ')', cls: (i) => (i === newRow ? 'p12new' : '') }).items,
          ...P.mat('vc', 440, 92, c.v, { rows: false, cols: Array.from({ length: dk }, (_, j) => String(j)), cw: 30, ch: 22, max: vmax, fmt, size: 11, title: 'V cache', cls: (i) => (i === newRow ? 'p12new' : '') }).items,
        ];
        if (wRow) items.push(...P.mat('wc', 364, 92, wRow.map((v) => [v]), { rows: false, cols: ['w'], cw: 40, ch: 22, scheme: 'seq', max: 1, fmt: (v) => (v < 0.005 ? '' : H().fx(v, 2).replace(/^0\./, '.')), size: 11, title: 'weights', titleCls: 'strong' }).items);
        return items;
      };
      const gen = (n) => [H().text('gs:h', 90, 30, 'prompt and answer so far', 'strong', 13), ...P.strip('gs', 290, 14, toks(n), { size: 13, gap: 3, minW: 24, cls: (i) => (i < prompt.length ? '' : i === n - 1 ? 'p12ask' : 'p12ans') }).items];
      const cost = (a, b) => [H().text('co:h', 90, 410, 'rows of Q, K, V computed so far, per layer', 'strong', 13), { key: 'co:a', kind: 'cell', x: 250, y: 424, w: P.r2(Math.max(2, 9 * a)), h: 20, rx: 3, text: '', cls: 'p12bar', audit: 'ignore' }, H().text('co:al', 240, 439, 'with cache', 'legend', 13, 'end'), H().text('co:av', 258 + 9 * a, 439, String(a), 'tick', 12),
        { key: 'co:b', kind: 'cell', x: 250, y: 454, w: P.r2(Math.max(2, 9 * b)), h: 20, rx: 3, text: '', cls: 'p12bar hot', audit: 'ignore' }, H().text('co:bl', 240, 469, 'without', 'legend', 13, 'end'), H().text('co:bv', 258 + 9 * b, 469, String(b), 'tick', 12)];
      F.set('rows', prompt.length);
      F.push([...gen(prompt.length), ...picture(first, prompt.length, -1, first.w[prompt.length - 1]), ...cost(prompt.length, prompt.length)],
        `First the prompt, ${P.qpl(q, prompt.length, 'token')}, runs through the model once (the prefill). Every layer stores its keys and values: here layer ${q.n(0)}, head ${q.n(hd)} keeps a ${q.n(prompt.length)} × ${q.n(dk)} K and a V of the same shape. The last position’s output gives the first answer digit, “${P.qtok(q, steps[0].id)}”.`, { phase: 'prefill', n: prompt.length, dk, hd, l0: 0, id: steps[0].id });
      let wsum = prompt.length;
      steps.slice(0, 4).forEach((st, s) => {
        F.count('rows');
        const n = st.t + 1, wr = st.w[0], j = P.argmax(wr);
        wsum += n;
        F.push([...gen(n), ...picture(st, n, st.t, wr), ...cost(prompt.length + s + 1, wsum)],
          `“${P.qtok(q, st.id)}” is fed back in. Only its own row is new: one query, one key and one value per layer, the key and value appended to the cache (outlined); the rows above are reused as they are. Its query is scored against all ${P.qpl(q, n, 'cached key')}; head ${q.n(hd)} puts ${q(wr[j], 2)} on position ${q.n(j)}. The next digit is “${P.qtok(q, steps[s + 1].id)}”. Without the cache all ${P.qpl(q, n, 'row')} would be computed again.`, { phase: 'step', s, t: st.t, n, id: st.id, next: steps[s + 1].id, j, wj: wr[j], hd });
      });
      const ans = steps.map((s) => s.id), right = ans.filter((v, i) => v === want[i]).length, worst = Math.max(...steps.map((s) => s.diff));
      const total = withC, totalNo = without;
      F.push([...gen(seq.length), ...picture(steps[3], seq.length - 1, -1, null), ...cost(total, totalNo)],
        `The answer is ${H().list(ans.map((v) => P.qtok(q, v)))}, ${q.n(right)} of ${q.n(5)} digits right. At every step the cached logits matched a full recomputation${worst ? ' to within ' + q.sg(worst, 2) : ' exactly'}. With the cache the ${q.n(L)} layers each computed ${q.n(total)} rows of Q, K and V; recomputing everything would have cost ${q.n(totalNo)}. For a long text the saving grows with its length, since without a cache the work for the whole text grows with the square of it; the price is memory for the keys and values of every layer.`, { phase: 'end', ans, right, worst, total, totalNo, L });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Generating with a cache', text: ['prefill: run the prompt, keep K, V of every layer', 'each new token t:', '  q_t, k_t, v_t from its own row only', '  K ← [K; k_t],  V ← [V; v_t]', '  out_t = softmax(q_t Kᵀ / √dₖ) V'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'prefill', once: true, pick: 'option',
      ask: 'When the next token is fed in, how many new rows of keys does each layer compute?',
      options: ['one', 'as many as tokens so far', 'none'],
      answer: 'one',
      explain: 'The earlier keys cannot change under the causal mask, so they come from the cache.',
    }],
    quiz: [
      { q: 'Why can the keys and values of earlier tokens be cached in a decoder?', options: ['under the causal mask they never depend on later tokens', 'they are random', 'they are the same for every token', 'the cache stores the answers'], answer: 0, why: 'Position i sees only positions up to i, so its rows are final once computed.' },
      { q: 'What does the cache cost?', options: ['memory: keys and values of every layer for every token so far', 'accuracy', 'a larger vocabulary', 'nothing at all'], answer: 0, why: 'It trades memory for repeated computation.' },
      { q: 'Generating 100 tokens after a 10-token prompt, how many rows does each layer project with a cache?', options: ['109', '110', 'about 6,000', '10'], answer: 0, why: '10 in the prefill, then one for each new token fed back: every one except the last, which is never fed in.' },
      { q: 'Is the query of an earlier token ever needed again?', options: ['no: only keys and values are reused', 'yes, at every step', 'only in the last layer', 'only for the prompt'], answer: 0, why: 'The new token’s query is the only one scored at each step.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
