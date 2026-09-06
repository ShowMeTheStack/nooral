// Part 13, generating text from the trained character GPT: the next-character probabilities reshaped by
// temperature (sharper below one, flatter above) and the texts greedy choice and sampling give; top-k and top-p
// cutting the list before a draw (DL.sample); and the key–value cache, counting the positions that pass through the
// blocks at every step with and without it (net.generate both ways, the texts compared).
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml;
  const COLW = 265;
  /** does a text repeat a stretch of at least n characters? the stretch, or null */
  P.loop = (t, n) => { for (let i = 0; i + n <= t.length; i++) { const s = t.slice(i, i + n); if (t.indexOf(s, i + 1) >= 0) return s; } return null; };

  S.topic({
    id: 'sampling-temperature', family: 'p13-sampling', order: 10, title: 'Greedy choice and temperature',
    blurb: 'The model gives probabilities; something must still choose. Greedy decoding always takes the most likely token, which is safe but dull and often loops. Sampling draws a token at random by its probability. Temperature divides the scores before the softmax: below one the favourite grows stronger, above one the long tail gets a chance.',
    inputs: [
      { id: 'prompt', kind: 'text', label: 'Prompt', def: 'the sailor said that ', check: P.checkText('Prompt', 3, 24) },
      { id: 'cool', kind: 'slider', label: 'Low temperature', def: 0.5, min: 0.1, max: 0.9, step: 0.1, fmt: (v) => (+v).toFixed(1) },
      { id: 'hot', kind: 'slider', label: 'High temperature', def: 1.6, min: 1.1, max: 2.5, step: 0.1, fmt: (v) => (+v).toFixed(1) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 4, min: 1, max: 999 },
    ],
    presets: [
      { label: 'The sailor said…', values: { prompt: 'the sailor said that ', cool: 0.5, hot: 1.6, seed: 4 } },
      { label: 'Very cold, very hot', values: { prompt: 'the weaver ', cool: 0.2, hot: 2.4, seed: 9 } },
      { label: 'Mid-word', values: { prompt: 'the lightho', cool: 0.7, hot: 1.3, seed: 2 } },
      { label: 'After a full stop', values: { prompt: 'the tide turns.', cool: 0.5, hot: 1.8, seed: 17 } },
    ],
    counters: [{ id: 'texts', label: 'texts drawn' }],
    run(inp) {
      const net = P.gpt(), F = H().frames(['texts']), q = F.q, r = P.next(net, inp.prompt), N = 40;
      const temps = [inp.cool, 1, inp.hot], dist = temps.map((t) => P.softmax(r.logits, t)), top = P.top(dist[1], 8);
      const col = (j, on) => {
        const items = [H().text('st:h' + j, j * COLW + 10, 30, (j === 1 ? 'temperature 1 (as trained)' : 'temperature ' + H().fx(temps[j], 1)), 'strong', 13)];
        if (on) items.push(...P.bars('st:b' + j, j * COLW, 44, top.map((t) => ({ label: P.show(t.c), v: dist[j][t.id], cls: t.id === top[0].id ? 'hot' : '', text: H().fx(dist[j][t.id], 2) })), { lw: 24, bw: 170, rh: 25, max: 1 }));
        return items;
      };
      const texts = [];
      const boxes = () => texts.map((t, i) => P.textBox('st:t' + i, 0, 270 + i * 50, 790, [t.label + ': ' + inp.prompt + t.text], { cls: i === texts.length - 1 ? 'p13mine' : '' }).items).flat();
      const pT = (j) => dist[j][top[0].id];
      F.push([...col(1, true), ...col(0, false), ...col(2, false)],
        `After “${P.show(inp.prompt)}” the model gives ${P.name(top[0].c)} p = ${P.qp(q, pT(1))} and ${P.name(top[1].c)} p = ${P.qp(q, dist[1][top[1].id])}. These are the probabilities at temperature ${q.n(1)}: the scores z go into the softmax as they are.`, { phase: 'base', p1: pT(1), p2: dist[1][top[1].id], T: 1 });
      F.push([...col(1, true), ...col(0, true), ...col(2, false)],
        `Dividing the scores by ${q(inp.cool, 1)} before the softmax stretches the gaps between them: ${P.name(top[0].c)} rises to p = ${P.qp(q, pT(0))} and the rest shrink. As the temperature goes to zero this becomes greedy choice.`, { phase: 'cool', T: inp.cool, p: pT(0) });
      F.push([...col(1, true), ...col(0, true), ...col(2, true)],
        `Dividing by ${q(inp.hot, 1)} squeezes the gaps: ${P.name(top[0].c)} falls to p = ${P.qp(q, pT(2))}, and the eighth character in the list now has p = ${P.qp(q, dist[2][top[7].id])} instead of ${P.qp(q, dist[1][top[7].id])}. Very high temperatures approach a uniform draw.`, { phase: 'hot', T: inp.hot, p: pT(2), p8: dist[2][top[7].id], p8base: dist[1][top[7].id] });
      const gen = (o, seed) => net.generate(r.ids, N, o, seed).ids.slice(r.ids.length).map((i) => P.vocab()[i]).join('');
      const greedy = gen({ method: 'greedy' }, 1), loop = P.loop(greedy, 8);
      texts.push({ label: 'greedy', text: greedy }); F.count('texts');
      F.push([...col(1, true), ...col(0, true), ...col(2, true), ...boxes()],
        `Greedy decoding takes the most likely character every time, so the same prompt always gives the same ${P.qpl(q, N, 'character')}.` + (loop ? ` It falls into a loop: “${P.show(loop)}” comes round again, because once the text repeats, so do the probabilities.` : ' It plays safe: the commonest words of the text, in the commonest order.'), { phase: 'greedy', text: greedy, loop, n: N });
      const cool = gen({ temperature: inp.cool }, inp.seed);
      texts.push({ label: 'T ' + H().fx(inp.cool, 1), text: cool }); F.count('texts');
      F.push([...col(1, true), ...col(0, true), ...col(2, true), ...boxes()],
        `Sampling at temperature ${q(inp.cool, 1)} with seed ${q.n(inp.seed)} draws each character by its sharpened probability: ${cool === greedy ? 'here it picks exactly what greedy picked.' : 'mostly the favourites, with an occasional second choice that sends the text another way.'}`, { phase: 'coolText', T: inp.cool, seed: inp.seed, text: cool, same: cool === greedy });
      const hot = gen({ temperature: inp.hot }, inp.seed), words = hot.split(/[ .,]+/).filter(Boolean), known = words.filter((w) => new RegExp('\\b' + w + '\\b').test(S.P13T.harbour));
      texts.push({ label: 'T ' + H().fx(inp.hot, 1), text: hot }); F.count('texts');
      F.push([...col(1, true), ...col(0, true), ...col(2, true), ...boxes()],
        `At temperature ${q(inp.hot, 1)} with the same seed the rare characters get through: ${q.n(known.length)} of the ${P.qpl(q, words.length, 'word')} it writes are words of the training text. Low temperatures suit answers that must be right, higher ones variety; most systems sit a little below ${q.n(1)}.`, { phase: 'hotText', T: inp.hot, seed: inp.seed, text: hot, known: known.length, words: words.length, one: 1 });
      return { frames: F.list, def: { sections: [{ id: 'temp', kind: 'math', title: 'Temperature', text: ['pᵢ = e^(zᵢ / T) / Σⱼ e^(zⱼ / T)', 'T = 1: the model’s own probabilities', 'T → 0: all weight on the largest score (greedy)', 'T → ∞: uniform'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'base',
      once: true, pick: 'option',
      ask: (fr) => 'At the low temperature, will the favourite’s probability rise or fall?',
      options: ['rise', 'fall', 'stay the same'],
      answer: (fr) => { const a = fr[0].meta.p1, b = fr[1].meta.p; return Math.abs(a - b) < 5e-4 ? 'stay the same' : b > a ? 'rise' : 'fall'; },
      explain: (fr) => fr[1].say,
    }],
    quiz: [
      { q: 'Temperature T changes the probabilities by…', options: ['dividing the scores by T before the softmax', 'multiplying the probabilities by T', 'adding T to every score', 'removing T tokens'], answer: 0, why: 'pᵢ ∝ e^(zᵢ/T).' },
      { q: 'Greedy decoding with the same prompt gives…', options: ['the same text every time', 'a different text every time', 'random characters', 'the prompt reversed'], answer: 0, why: 'There is no randomness: it always takes the top token.' },
      { q: 'Why does greedy text often loop?', options: ['the same recent text gives the same top choice again', 'the model runs out of tokens', 'the seed repeats', 'the cache is full'], answer: 0, why: 'Deterministic choice plus repeated context gives repeated output.' },
      { q: 'A very high temperature makes the output…', options: ['close to random characters', 'more correct', 'shorter', 'identical to greedy'], answer: 0, why: 'All tokens approach equal probability.' },
    ],
  });

  S.topic({
    id: 'sampling-topk-topp', family: 'p13-sampling', order: 20, title: 'Top-k and top-p: cutting the tail',
    blurb: 'Sampling from every token lets a long tail of unlikely ones through, and one bad character can derail the rest. Top-k keeps only the k most likely tokens; top-p (nucleus sampling) keeps the smallest set whose probabilities add up to at least p, so it keeps few when the model is sure and many when it is not. The kept probabilities are scaled back up to add to one before the draw.',
    inputs: [
      { id: 'prompt', kind: 'text', label: 'Prompt', def: 'the boats ', check: P.checkText('Prompt', 3, 30) },
      { id: 'k', kind: 'number', label: 'k (top-k)', def: 5, min: 1, max: 26 },
      { id: 'p', kind: 'slider', label: 'p (top-p)', def: 0.8, min: 0.05, max: 1, step: 0.05, fmt: (v) => (+v).toFixed(2) },
      { id: 'temperature', kind: 'slider', label: 'Temperature', def: 1, min: 0.3, max: 2, step: 0.1, fmt: (v) => (+v).toFixed(1) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Start of a word', values: { prompt: 'the boats ', k: 5, p: 0.8, temperature: 1, seed: 1 } },
      { label: 'Model is sure', values: { prompt: 'the lighthou', k: 5, p: 0.9, temperature: 1, seed: 1 } },
      { label: 'Tight nucleus', values: { prompt: 'the weaver ', k: 10, p: 0.5, temperature: 1, seed: 3 } },
      { label: 'Hot and wide', values: { prompt: 'a storm comes ', k: 20, p: 0.95, temperature: 1.6, seed: 8 } },
    ],
    counters: [{ id: 'kept', label: 'characters kept' }],
    run(inp) {
      const net = P.gpt(), F = H().frames(['kept']), q = F.q, r = P.next(net, inp.prompt), V = P.vocab().length;
      const p0 = P.softmax(r.logits, inp.temperature), order = p0.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p || a.id - b.id);
      const k = Math.min(inp.k, V), zk = order.slice(0, k).reduce((s, o) => s + o.p, 0);
      let cum = 0, m = 0;
      while (m < V) { cum += order[m].p; m++; if (cum >= inp.p) break; }
      const zp = order.slice(0, m).reduce((s, o) => s + o.p, 0);
      const RH = 21, Y0 = 44, SHOW = 15;
      const col = (j, title, val, kept, cut) => {
        const items = [H().text('kp:h' + j, j * COLW + 8, 28, title, 'strong', 13)];
        const rows = order.slice(0, SHOW).map((o, i) => ({ label: P.show(P.vocab()[o.id]), v: val(o, i), cls: kept(i) ? (o.id === order[0].id ? 'hot' : '') : 'soft', text: kept(i) ? H().fx(val(o, i), 2) : '' }));
        const rest = order.slice(SHOW).reduce((s, o, i) => s + val(o, i + SHOW), 0);
        rows.push({ label: '+' + (V - SHOW), v: rest, cls: kept(SHOW) ? '' : 'soft', text: kept(SHOW) || j === 0 ? H().fx(rest, 2) : '', lcls: 'p13rest' });
        items.push(...P.bars('kp:b' + j, j * COLW, Y0, rows, { lw: 34, bw: 160, rh: RH, max: 1, size: 13, vsize: 13 }));
        if (cut !== undefined && cut < V) items.push({ key: 'kp:cut' + j, kind: 'line', x1: j * COLW + 40, y1: Y0 + Math.min(cut, SHOW) * RH, x2: j * COLW + 250, y2: Y0 + Math.min(cut, SHOW) * RH, cls: 'p13cut', audit: 'ignore' });
        return items;
      };
      const A = col(0, 'all ' + V + ' characters', (o) => o.p, () => true);
      const B = (on) => (on ? col(1, 'top-k, k = ' + k, (o, i) => (i < k ? o.p / zk : 0), (i) => i < k, k) : [H().text('kp:h1', COLW + 8, 28, 'top-k, k = ' + k, 'strong p13faded', 13)]);
      const Cc = (on) => (on ? col(2, 'top-p, p = ' + H().fx(inp.p, 2), (o, i) => (i < m ? o.p / zp : 0), (i) => i < m, m) : [H().text('kp:h2', 2 * COLW + 8, 28, 'top-p, p = ' + H().fx(inp.p, 2), 'strong p13faded', 13)]);
      F.set('kept', V);
      F.push([...A, ...B(false), ...Cc(false)], `After “${P.show(inp.prompt)}”, at temperature ${q(inp.temperature, 1)}, the ${q.n(V)} characters sorted by probability: ${P.name(P.vocab()[order[0].id])} leads with ${P.qp(q, order[0].p)}, and the ${q.n(V - 5)} least likely share ${P.qp(q, order.slice(5).reduce((s, o) => s + o.p, 0))}. Plain sampling could pick any of them.`,
        { phase: 'all', T: inp.temperature, V, p1: order[0].p, tail: order.slice(5).reduce((s, o) => s + o.p, 0), nTail: V - 5 });
      F.set('kept', k);
      F.push([...A, ...B(true), ...Cc(false)], `Top-k with k = ${q.n(k)} keeps the ${P.qpl(q, k, 'most likely character')} and drops the rest. The kept ones held ${P.qp(q, zk)} of the probability, so each is divided by ${P.qp(q, zk)}: ${P.name(P.vocab()[order[0].id])} goes from ${P.qp(q, order[0].p)} to ${P.qp(q, order[0].p / zk)}.`,
        { phase: 'topk', k, zk, p1: order[0].p, p1k: order[0].p / zk });
      F.set('kept', m);
      F.push([...A, ...B(true), ...Cc(true)], `Top-p with p = ${q(inp.p, 2)} walks down the list adding probabilities until the total reaches ${q(inp.p, 2)}: that takes ${P.qpl(q, m, 'character')} (total ${P.qp(q, zp)}). ` + (m < k ? `Here the model is fairly sure, so the nucleus is smaller than k.` : m > k ? `Here the model is unsure, so the nucleus is larger than k.` : `Here both rules keep the same number.`),
        { phase: 'topp', p: inp.p, m, zp, k });
      const s = S.DL.sample(r.logits, { temperature: inp.temperature, topK: k, topP: inp.p }, inp.seed), c = P.vocab()[s.id];
      F.set('kept', s.kept);
      F.push([...A, ...B(true), ...Cc(true), ...P.textBox('kp:draw', 0, Y0 + (SHOW + 1) * RH + 14, 790, ['drawn with seed ' + inp.seed + ': ' + P.show(c) + '   (from ' + s.kept + ' kept, p = ' + H().fx(s.probs[s.id], 3) + ')'], { cls: 'p13mine' }).items],
        `With both rules on, as generation usually runs, top-k is applied first and top-p inside it: ${P.qpl(q, s.kept, 'character')} remain. The draw with seed ${q.n(inp.seed)} gives ${P.name(c)}, whose renormalised probability is ${P.qp(q, s.probs[s.id])}.`, { phase: 'draw', kept: s.kept, id: s.id, p: s.probs[s.id], seed: inp.seed, probs: s.probs });
      return { frames: F.list, def: { sections: [{ id: 'kp', kind: 'math', title: 'Cutting the tail', text: ['sort tokens by p', 'top-k: keep the first k', 'top-p: keep the first m with Σ p ≥ p', 'renormalise: p′ᵢ = pᵢ / Σ kept p', 'then draw from p′'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'topk',
      once: true, pick: 'option',
      ask: 'How many characters will top-p keep?',
      options: (fr) => { const t = fr.find((f) => f.meta.phase === 'topp').meta, m = t.m; return [...new Set([m, Math.max(1, m - 2), m + 2, t.k === m ? m + 4 : t.k])].sort((x, y) => x - y).map(String); },
      answer: (fr) => String(fr.find((f) => f.meta.phase === 'topp').meta.m),
      explain: (fr) => fr.find((f) => f.meta.phase === 'topp').say,
    }],
    quiz: [
      { q: 'Top-k with k = 1 is the same as…', options: ['greedy decoding', 'uniform sampling', 'top-p with p = 1', 'temperature 2'], answer: 0, why: 'Only the most likely token is left.' },
      { q: 'Top-p adapts to the model’s confidence because…', options: ['a sure model reaches p with few tokens, an unsure one needs many', 'p changes during generation', 'it uses the seed', 'it sorts alphabetically'], answer: 0, why: 'The nucleus size depends on how the probability is spread.' },
      { q: 'After cutting, the kept probabilities are…', options: ['divided by their total so they add up to one', 'left as they are', 'set equal', 'squared'], answer: 0, why: 'Renormalising makes a proper distribution to draw from.' },
      { q: 'Top-p with p = 1 keeps…', options: ['every token', 'one token', 'half the tokens', 'none'], answer: 0, why: 'Only the whole list adds up to one.' },
    ],
  });

  S.topic({
    id: 'kv-cache-generation', family: 'p13-sampling', order: 30, title: 'The KV cache: generating without recomputing',
    blurb: 'Each new token needs the keys and values of every token before it. Without a cache the model runs the whole text through its blocks again at every step, so step t costs t positions. With a cache it keeps each layer’s keys and values and runs only the newest token: one position a step, the same text out, for the price of memory.',
    inputs: [
      { id: 'prompt', kind: 'text', label: 'Prompt', def: 'the keeper ', check: P.checkText('Prompt', 2, 14) },
      { id: 'n', kind: 'number', label: 'Characters to generate', def: 8, min: 2, max: 16 },
    ],
    presets: [
      { label: 'Eight characters', values: { prompt: 'the keeper ', n: 8 } },
      { label: 'Short prompt, long run', values: { prompt: 'a ', n: 16 } },
      { label: 'Long prompt', values: { prompt: 'the gulls cry ', n: 6 } },
      { label: 'Two characters', values: { prompt: 'the tide ', n: 2 } },
    ],
    counters: [{ id: 'nocache', label: 'positions without cache' }, { id: 'cache', label: 'positions with cache' }],
    run(inp) {
      const net = P.gpt(), F = H().frames(['nocache', 'cache']), q = F.q, cfg = P.D().gpt.config;
      const ids = net.tokenizer.encode(inp.prompt), L = ids.length, n = inp.n;
      // run the model's own generation both ways, counting the positions each forward pass takes
      const count = (cache) => { const log = []; net.forward = function (x, o) { log.push(x.shape[1]); return Object.getPrototypeOf(this).forward.call(this, x, o); }; try { return { r: net.generate(ids, n, { method: 'greedy' }, 1, { cache }), log }; } finally { delete net.forward; } };
      const A = count(false), B = count(true), same = A.r.ids.join() === B.r.ids.join(), all = Array.from(inp.prompt + A.r.ids.slice(L).map((i) => P.vocab()[i]).join(''));
      const rows = L + n - 1, CW = 15, RH = Math.min(14, Math.floor(400 / rows)), Y0 = 74;
      const grid = (gx, title, log, upto) => {
        const items = [H().text('kv:t' + gx, gx, 18, title, 'strong', 13), H().text('kv:s' + gx, gx + 20, 40, 'step → (one column per pass through the blocks)', 'tick', 12)];
        for (let r = 0; r < rows; r++) items.push(H().text('kv:r' + gx + '_' + r, gx + 10, Y0 + r * RH + RH / 2 + 4, P.show(all[r]), 'mono ' + (r >= L ? 'p13new' : ''), 12, 'end'));
        for (let s = 0; s < upto; s++) {
          const end = s === 0 ? L : L + s, start = end - log[s];
          for (let r = 0; r < end; r++) items.push({ key: 'kv:c' + gx + '_' + s + '_' + r, kind: 'cell', x: gx + 18 + s * CW, y: Y0 + r * RH, w: CW - 2, h: RH - 2, rx: 2, text: '', cls: 'p13kv ' + (r >= start ? 'p13run' : 'p13read'), audit: 'ignore' });
        }
        return items;
      };
      const legend = S.Kit.curve.legend('kv:lg', 0, Y0 + rows * RH + 30, [{ text: 'run through the blocks this step', cls: 'p13run', dot: true }, { text: 'keys and values read from the cache', cls: 'p13read', dot: true }], { size: 12 });
      const pic = (upto) => [...grid(20, 'without a cache', A.log, upto), ...grid(420, 'with a cache', B.log, upto), ...legend];
      F.push(pic(0), `The prompt has ${P.qpl(q, L, 'character')} and ${P.qpl(q, n, 'more character')} will be generated, greedily, by the trained GPT. Each new character attends to every character before it, so it needs their keys and values in both blocks.`, { phase: 'start', L, n });
      const sumA = (s) => A.log.slice(0, s).reduce((a, b) => a + b, 0), sumB = (s) => B.log.slice(0, s).reduce((a, b) => a + b, 0);
      F.set('nocache', sumA(1)).set('cache', sumB(1));
      F.push(pic(1), `Step one is the same both ways: the ${P.qpl(q, A.log[0], 'prompt position')} pass through the blocks together (the prefill), giving the first new character, ${P.name(all[L])}. With a cache, every position’s keys and values are kept.`, { phase: 'prefill', a: A.log[0], b: B.log[0], c: all[L] });
      const shown = Math.min(n, 12);
      for (let s = 1; s < shown; s++) {
        F.set('nocache', sumA(s + 1)).set('cache', sumB(s + 1));
        F.push(pic(s + 1), `Step ${q.n(s + 1)}: without a cache all ${P.qpl(q, A.log[s], 'position')} run again; with it only the newest character runs (${P.qpl(q, B.log[s], 'position')}) and reads the other ${q.n(L + s - 1)} positions’ keys and values from memory. So far: ${q.n(sumA(s + 1))} positions against ${q.n(sumB(s + 1))}.`,
          { phase: 'step', s: s + 1, a: A.log[s], b: B.log[s], read: L + s - 1, totA: sumA(s + 1), totB: sumB(s + 1) });
      }
      const tA = sumA(n), tB = sumB(n), mem = cfg.layers * 2 * rows * cfg.d;
      F.set('nocache', tA).set('cache', tB);
      F.push(pic(n), `Both runs write the same text, “${P.show(all.slice(L).join(''))}”` + (same ? '' : ' (they differ!)') + `. Without the cache ${q.n(tA)} positions went through the blocks, with it ${q.n(tB)}: ${q(tA / tB, 1)} times less work. The price is memory: ${q.n(cfg.layers)} blocks × keys and values × ${q.n(rows)} positions × ${q.n(cfg.d)} numbers = ${q.n(mem)} numbers kept. Without the cache the work grows with the square of the length, with it only linearly.`,
        { phase: 'end', tA, tB, ratio: tA / tB, same, mem, layers: cfg.layers, rows, d: cfg.d, text: all.slice(L).join('') });
      return { frames: F.list, def: { sections: [{ id: 'kv', kind: 'math', title: 'Cost of generating n tokens after a prompt of L', text: ['without cache: L + (L+1) + … + (L+n−1)', '  positions through the blocks', 'with cache: L + (n − 1)', 'memory: layers × 2 × positions × d'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'prefill',
      once: true, pick: 'option',
      ask: 'With the cache, how many positions go through the blocks at the next step?',
      options: (fr) => [...new Set(['1', String(fr[0].meta.L), String(fr[0].meta.L + 1)])],
      answer: '1',
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'What does the KV cache store?', options: ['the keys and values of earlier positions, per layer', 'the generated text', 'the gradients', 'the output probabilities'], answer: 0, why: 'Queries are needed only for the new token; keys and values of the past are reused.' },
      { q: 'Does the cache change the generated text?', options: ['no: it only avoids recomputing the same numbers', 'yes, it adds randomness', 'yes, it shortens it', 'only at high temperature'], answer: 0, why: 'Earlier positions cannot see later ones (causal attention), so their keys and values never change.' },
      { q: 'Why can earlier keys and values be reused at all?', options: ['causal masking means a position never depends on later tokens', 'the weights are frozen during training', 'the keys are random', 'the vocabulary is small'], answer: 0, why: 'Adding a token never changes what came before.' },
      { q: 'What does the cache cost?', options: ['memory that grows with the length of the text', 'accuracy', 'a larger vocabulary', 'nothing at all'], answer: 0, why: 'Layers × 2 × positions × width numbers must be kept.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
