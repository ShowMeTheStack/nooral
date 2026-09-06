// Part 13, in-context learning on a toy task: a tiny GPT trained (tests/ref/p13_runs.mjs) on sequences of key–value
// pairs whose meaning is drawn afresh for every sequence. The weights cannot hold the answer; the model must read it
// from earlier in the same sequence. A repeated key it answers from context (the last block's attention points at
// the earlier pair); a key seen for the first time it can only guess.
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml;
  /** the held-out sequences of the run, rebuilt with its generator and seed (the training ones are drawn first) */
  P.iclVal = () => P.once('iclVal', () => {
    const g = S.T.rng(5), K = 6, N = 8, one = () => { const map = Array.from({ length: K }, () => g.int(10)), s = []; for (let i = 0; i < N; i++) { const k = g.int(K); s.push(10 + k, map[k]); } return s; };
    for (let i = 0; i < 6000; i++) one();
    return Array.from({ length: 300 }, one);
  });
  /** accuracy on the held-out sequences at the value places, split by whether the key appeared before */
  P.iclAccuracy = () => P.once('iclAcc', () => {
    const net = P.icl(), val = P.iclVal(), z = net.predict(S.T.tensor(val.map((s) => s.slice(0, -1)))), V = z.shape[2], n = val[0].length - 1, v = z.values();
    const out = { seen: [0, 0], fresh: [0, 0] };
    val.forEach((s, b) => {
      const ks = new Set();
      for (let i = 0; i < s.length; i += 2) {
        const row = v.subarray((b * n + i) * V, (b * n + i + 1) * V);
        let best = 0; for (let d = 1; d < 10; d++) if (row[d] > row[best]) best = d;
        const c = ks.has(s[i]) ? out.seen : out.fresh;
        c[0] += best === s[i + 1] ? 1 : 0; c[1]++;
        ks.add(s[i]);
      }
    });
    return out;
  });
  const parse = (txt) => String(txt).trim().split(/\s+/).filter(Boolean);
  const checkSeq = (v) => {
    const items = parse(v);
    if (items.length < 2) return 'Sequence: at least two pairs, such as a3 a?';
    if (items.length > 8) return 'Sequence: at most eight pairs (the model’s window holds 16 tokens)';
    const bad = items.find((x) => !/^[a-f][0-9?]$/.test(x));
    if (bad) return 'Sequence: each pair is a key a–f and a digit or ?, such as c7 or c? (not “' + bad + '”)';
    if (!items.some((x) => x[1] === '?')) return 'Sequence: put a ? in at least one pair for the model to answer';
    return null;
  };

  S.topic({
    id: 'lm-in-context', family: 'p13-beyond', order: 20, title: 'In-context learning on a toy task',
    blurb: 'Large models can pick up a task from examples in the prompt, with no change to their weights. A tiny version: a GPT trained on sequences of pairs like “c7 a2 c7”, where in every sequence each letter means a different digit. Nothing in the weights says what c means; the only place to find it is earlier in the same sequence. Ask about a letter it has seen and it copies the answer from context; ask about a new one and it can only guess.',
    inputs: [{ id: 'seq', kind: 'text', label: 'Sequence (key a–f, digit or ?)', def: 'c7 a2 e5 a? d4 c? f? e?', check: checkSeq }],
    presets: [
      { label: 'Seen and unseen', values: { seq: 'c7 a2 e5 a? d4 c? f? e?' } },
      { label: 'One example', values: { seq: 'b9 b?' } },
      { label: 'Same digit twice', values: { seq: 'a4 b4 c1 b? a? c?' } },
      { label: 'Never seen', values: { seq: 'a1 b2 c3 d? e? f?' } },
    ],
    counters: [{ id: 'asked', label: 'questions answered' }],
    run(inp) {
      const net = P.icl(), F = H().frames(['asked']), q = F.q, items = parse(inp.seq), toks = [], shown = [];
      items.forEach((x) => { toks.push(10 + P.KEYS.indexOf(x[0])); toks.push(x[1] === '?' ? null : +x[1]); });
      const label = (t) => (t >= 10 ? P.KEYS[t - 10] : String(t));
      const CW = 42, strip = (cur, att, ask) => {
        const items2 = [H().text('ic:h', 0, 16, 'the sequence: keys and the digits they stand for in this sequence', 'strong', 13)];
        toks.forEach((t, i) => {
          const it = { key: 'ic:t' + i, kind: 'cell', x: i * (CW + 4), y: 30, w: CW, h: 30, rx: 5, text: t === null ? '?' : label(t), size: 15, cls: 'tok mono ' + (i % 2 ? 'p13val' : 'p13key') + (i === cur ? ' p13cur' : '') + (shown.includes(i) ? ' new' : '') + (att && att.top === i ? ' p13att' : '') };
          if (att && i <= att.from) it.sub = H().fx(att.w[i], 2);
          items2.push(it);
        });
        if (att) items2.push(H().text('ic:al', 0, 96, 'under each token: how much the asking key attends to it (last block, strongest head)', 'tick', 12));
        return items2;
      };
      const facts = new Map(), answers = [];
      F.push([...strip(-1), ...P.panel('ic:p', 0, 120, 400, 300, null), ...H().lines('ic:lg', 16, 150, [['blue cells: keys a–f', 'legend'], ['orange cells: the digit after each key', 'legend'], ['?: a digit for the model to supply', 'legend']], 13, 26)], `Each pair is a key, one of the letters a to f, then a digit. In every training sequence the letters were given new random digits, so the model could not learn what a letter means, only how to find it. There are ${P.qpl(q, items.filter((x) => x[1] === '?').length, 'question')} here, marked ?.`, { phase: 'start', asks: items.filter((x) => x[1] === '?').length });
      for (let i = 0; i < toks.length; i += 2) {
        const key = toks[i];
        if (toks[i + 1] !== null) { facts.set(key, toks[i + 1]); continue; }
        const ctx = toks.slice(0, i + 1), z = P.logits(net, ctx), row = z[i], pr = P.softmax(row).slice(0, 10);
        let best = 0; for (let d = 1; d < 10; d++) if (pr[d] > pr[best]) best = d;
        // the last block's attention from the asking key: the head that looks hardest at one earlier token
        const maps = S.DL.attentionMaps(net).filter((m) => m.layer === P.D().icl.config.layers - 1), rows = maps.map((m) => m.weights[i]);
        const head = rows.reduce((a, r, h) => (Math.max(...r.slice(0, i)) > Math.max(...rows[a].slice(0, i)) ? h : a), 0), w = rows[head], top = w.slice(0, i).indexOf(Math.max(...w.slice(0, i)));
        const known = facts.has(key), want = known ? facts.get(key) : null;
        toks[i + 1] = best; shown.push(i + 1);
        answers.push({ key, best, known, ok: known ? best === want : null });
        F.count('asked');
        const bars = P.bars('ic:b', 0, 150, pr.map((p, d) => ({ label: String(d), v: p, cls: d === best ? 'hot' : d === want ? 'good' : 'soft', text: H().fx(p, 2) })), { lw: 20, bw: 300, rh: 26, max: 1 });
        F.push([...strip(i, { from: i, w, top }), ...P.panel('ic:p', 0, 120, 400, 300, null), H().text('ic:ph', 12, 140, 'the model’s probability for each digit after “' + P.KEYS[key - 10] + '”', 'legend', 12), ...bars],
          (known ? `Key “${P.KEYS[key - 10]}” appeared earlier with ${q.n(want)}. ` : `Key “${P.KEYS[key - 10]}” has not appeared before in this sequence. `) +
          `The model gives ${q.n(best)} the highest probability, ${q(pr[best], 2)}` + (known ? (best === want ? `: it found the answer in its context. Its last block attends most (${q(w[top], 2)}) to position ${q.n(top + 1)}, ${top + 1 === i ? 'the key just before' : top % 2 ? 'the digit of an earlier pair' : 'an earlier key'}.` : `, not the ${q.n(want)} it should have copied: the lookup failed this time.`) : `, but it is guessing: no digit stands out much, and any answer is as good as another.`),
          { phase: 'ask', i, key: key - 10, best, p: pr[best], known, want, top, wTop: w[top], pos: top + 1, probs: pr });
      }
      const acc = P.iclAccuracy(), sa = acc.seen[0] / acc.seen[1], fa = acc.fresh[0] / acc.fresh[1];
      F.push([...strip(-1), ...P.panel('ic:p', 0, 120, 790, 200, 'on 300 held-out sequences'), ...P.bars('ic:acc', 20, 160, [{ label: 'key seen before', v: sa, cls: 'good', text: H().pct(sa) }, { label: 'key new', v: fa, cls: 'soft', text: H().pct(fa) }, { label: 'chance', v: 0.1, cls: 'soft', text: '10%' }], { lw: 130, bw: 420, rh: 40, max: 1 })],
        `Over ${q.n(acc.seen[1] + acc.fresh[1])} questions in ${q.n(P.iclVal().length)} held-out sequences, the model answers a key it has seen before correctly ${q.pct(sa)} of the time, and a new key ${q.pct(fa)} of the time, about the chance of guessing one digit in ${q.n(10)}. What it learned is a procedure, find the earlier pair and copy its digit, not any fact. Large models do the same with examples in a prompt, on a much richer scale.`,
        { phase: 'end', seen: sa, fresh: fa, nSeen: acc.seen[1], nFresh: acc.fresh[1], total: acc.seen[1] + acc.fresh[1], seqs: P.iclVal().length, ten: 10 });
      return { frames: F.list, def: { sections: [{ id: 'icl', kind: 'math', title: 'The toy task', text: ['tokens: keys a–f, digits 0–9', 'each sequence: a fresh random digit per key', '8 pairs; scored at every digit', 'seen key → copy from context', 'new key → only a guess (1 in 10)'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => f.meta.phase === 'start' && fr.some((x) => x.meta.phase === 'ask'),
      once: true, pick: 'option',
      ask: (fr) => { const a = fr.find((f) => f.meta.phase === 'ask').meta; return 'What digit will the model give for key “' + P.KEYS[a.key] + '”?'; },
      options: (fr) => { const a = fr.find((f) => f.meta.phase === 'ask').meta; return [...new Set([a.best, a.want === null ? (a.best + 3) % 10 : a.want, (a.best + 5) % 10, (a.best + 7) % 10])].sort((x, y) => x - y).map(String); },
      answer: (fr) => String(fr.find((f) => f.meta.phase === 'ask').meta.best),
      explain: (fr) => fr.find((f) => f.meta.phase === 'ask').say,
    }],
    quiz: [
      { q: 'In this task, where can the model find what key c means?', options: ['only earlier in the same sequence', 'in its weights', 'in the vocabulary', 'nowhere: it always guesses'], answer: 0, why: 'Each sequence assigns new digits, so only the context holds the answer.' },
      { q: 'For a key that has not appeared yet, the best possible accuracy is about…', options: ['10%: one digit in ten', '50%', '100%', '0%'], answer: 0, why: 'The digit was drawn at random for this sequence.' },
      { q: 'In-context learning changes the model’s weights…', options: ['not at all: only the prompt changes', 'a little each time', 'completely', 'only the last layer'], answer: 0, why: 'The learning happens in the forward pass, through attention over the prompt.' },
      { q: 'What did the model learn in training?', options: ['a procedure: find the earlier pair with the same key and copy its digit', 'the meaning of each letter', 'to always say 0', 'the order of the alphabet'], answer: 0, why: 'That procedure works for every sequence; facts would not.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
