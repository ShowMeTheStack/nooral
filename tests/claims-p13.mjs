// Part 13, language models: the narration of every topic read back against an independent working-out.
//  - BPE: the merges replayed by a separate trainer (pair counts over the chunks of the text), a new sentence
//    cut by the usual lowest-rank-first rule per chunk;
//  - probabilities: the model run one token at a time through its key–value cache (a different path from the
//    topics' full forward pass), softmax, temperature, top-k and top-p, the draw and the chain rule by formula;
//  - cross-entropy and perplexity by formula; the KV-cache work by formula and the texts compared;
//  - the stored training runs: the first steps of each re-run with the engine must give the stored losses, and
//    the stored final weights must give the stored held-out loss;
//  - in-context accuracy from sequences rebuilt with the training script's own generator.
// Usage: node tests/claims-p13.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { gptSetup, GPT, GPT_STEPS, SCALE, iclTask, PROMPT, SAMPLE } from './ref/p13_runs.mjs';

const P = S.P13, T = S.T;
const ALLOWED = [/ln 2\b/g, /\be\^/g, /2\^b/g];
/** generation with and without the cache agrees while the text fits the window: the first characters past a prompt of length L */
const fits = (t, L) => t.slice(0, GPT.context - 1 - L);
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-6) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const f32 = 2e-5;
const softmax = (row, t) => { const m = Math.max(...row), e = row.map((v) => Math.exp((v - m) / (t || 1))), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
/** the logits after each token, fed one at a time through the model's cache */
const stepLogits = (net, ids) => T.noGrad(() => { net.eval(); const cache = net.newCache(), out = []; for (const id of ids) { const z = net.forward(T.tensor([[id]]), { cache }); out.push(Array.from(z.values())); } return out; });
const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const vocab = () => P.vocab();
const enc = (s) => Array.from(s, (c) => vocab().indexOf(c));

// ---- an independent BPE ----
const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;
function bpeTrain(text, n) {
  const freq = {};
  for (const w of text.match(PRE)) freq[w] = (freq[w] || 0) + 1;
  const words = Object.keys(freq).map((w) => ({ s: [...w], n: freq[w] }));
  let tokens = words.reduce((a, w) => a + w.s.length * w.n, 0);
  const merges = [];
  for (let m = 0; m < n; m++) {
    const pc = {};
    for (const w of words) for (let i = 0; i + 1 < w.s.length; i++) { const k = w.s[i] + '\u0000' + w.s[i + 1]; pc[k] = (pc[k] || 0) + w.n; }
    const best = Object.entries(pc).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0];
    if (!best || best[1] < 2) break;
    const [a, b] = best[0].split('\u0000');
    for (const w of words) { const o = []; for (let i = 0; i < w.s.length; i++) { if (w.s[i] === a && w.s[i + 1] === b) { o.push(a + b); i++; tokens -= w.n; } else o.push(w.s[i]); } w.s = o; }
    merges.push({ pair: [a, b], count: best[1], tokens });
  }
  return merges;
}
function bpeCut(text, merges) {
  const rank = new Map(merges.map((m, i) => [m.pair[0] + '\u0000' + m.pair[1], i]));
  return (text.match(PRE) || []).flatMap((ch) => {
    let s = [...ch];
    for (;;) { let best = -1, at = -1; for (let i = 0; i + 1 < s.length; i++) { const r = rank.get(s[i] + '\u0000' + s[i + 1]); if (r !== undefined && (best < 0 || r < best)) { best = r; at = i; } } if (at < 0) break; s = s.slice(0, at).concat([s[at] + s[at + 1]], s.slice(at + 2)); }
    return s;
  });
}

const checks = {
  'bpe-build': {
    check(c) {
      const text = S.P13T[c.input.text], ms = bpeTrain(text, c.input.merges), st = phase(c.frames, 'start')[0].meta;
      c.eq(st.chars, [...text].length, 'characters'); c.eq(st.tokens, [...text].length, 'tokens at the start'); c.eq(st.base, new Set(text).size, 'distinct characters');
      c.eq(st.count, ms[0].count, 'first pair count'); c.eq(st.pair, ms[0].pair, 'first pair');
      let prev = [...text].length;
      phase(c.frames, 'merge').forEach((f) => {
        const m = ms[f.meta.k - 1], nx = ms[f.meta.k];
        c.eq(f.meta.pair, m.pair, 'merge ' + f.meta.k); c.eq(f.meta.count, m.count, 'count of merge ' + f.meta.k); c.eq(f.meta.tokens, m.tokens, 'tokens after merge ' + f.meta.k);
        c.eq(f.meta.saved, prev - m.tokens, 'saved'); prev = m.tokens;
        if (nx) { c.eq(f.meta.next, nx.pair, 'next pair'); c.eq(f.meta.nextCount, nx.count, 'next count'); }
      });
      const e = phase(c.frames, 'end')[0].meta;
      c.eq(e.vocab, new Set(text).size + ms.length, 'vocabulary'); near(c, e.ratio, [...text].length / ms[ms.length - 1].tokens, 'characters per token');
    },
  },
  'bpe-tokenise': {
    random: (r) => ({ sentence: ['the gulls follow the boats home.', 'a yellow boat rocks at the wall.', 'the sailor and the weaver meet.', 'quick wizards jump.'][Math.floor(r() * 4)], merges: 5 + Math.floor(r() * 140) }),
    randomRuns: 6,
    check(c) {
      const text = S.P13T.harbour, ms = bpeTrain(text, c.input.merges), s = c.input.sentence, st = phase(c.frames, 'start')[0].meta;
      c.eq(st.chars, [...s].length, 'characters'); c.eq(st.chunks, s.match(PRE).length, 'chunks');
      c.eq(st.unk, [...new Set([...s].filter((ch) => !text.includes(ch)))], 'unknown characters');
      phase(c.frames, 'merge').forEach((f) => { const r = f.meta.ranks[f.meta.ranks.length - 1]; c.eq(f.meta.tokens, bpeCut(s, ms.slice(0, r + 1)).length, 'tokens after merge ' + (r + 1)); });
      const e = phase(c.frames, 'end')[0].meta, cut = bpeCut(s, ms);
      c.eq(e.pieces, cut.map((p) => (p.length > 1 || text.includes(p) ? p : '<unk>')), 'pieces');
      c.eq(e.tokens, cut.length, 'token count'); c.eq(e.vocab, 1 + new Set(text).size + ms.length, 'vocabulary with the unknown token');
      near(c, e.ratio, [...s].length / cut.length, 'characters per token');
    },
  },
  'lm-next-token': {
    random: (r) => ({ prompt: ['the boats ', 'the keeper climbs ', 'a gull ', 'the market opens at '][Math.floor(r() * 4)], steps: 1 + Math.floor(r() * 8) }),
    check(c) {
      const net = P.gpt(), g = net.generate(enc(c.input.prompt), c.input.steps, { method: 'greedy' }, 1), sf = phase(c.frames, 'step');
      c.eq(sf.length, c.input.steps, 'one frame per character');
      sf.forEach((f, k) => {
        const s = g.steps[k], order = s.probs.map((p, id) => [p, id]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
        c.eq(f.meta.pick, s.id, 'greedy pick ' + k);
        f.meta.top.forEach((t, j) => { c.eq(t.id, order[j][1], 'rank ' + j); near(c, t.p, order[j][0], 'probability of rank ' + j, f32); });
      });
      const ps = g.steps.map((s) => s.probs[s.id]), e = phase(c.frames, 'end')[0].meta;
      c.eq(e.added, g.steps.map((s) => vocab()[s.id]).join(''), 'added text');
      near(c, e.pSure, Math.max(...ps), 'surest', f32); near(c, e.pUnsure, Math.min(...ps), 'least sure', f32);
    },
  },
  'lm-cross-entropy': {
    random: (r) => ({ text: ['the tide comes in.', 'the baker sells warm rolls.', 'gulls rope tide bread', 'and the and the and'][Math.floor(r() * 4)], model: r() < 0.5 ? 'trained' : 'untrained' }),
    check(c) {
      const net = c.input.model === 'trained' ? P.gpt() : S.DL.build('gpt', Object.assign({}, P.D().gpt.config, { seed: 1 }));
      const ids = enc(c.input.text), L = stepLogits(net, ids), p = ids.slice(1).map((id, i) => softmax(L[i])[id]), nll = p.map((x) => -Math.log(x));
      phase(c.frames, 'pos').forEach((f) => { near(c, f.meta.p, p[f.meta.i - 1], 'p at ' + f.meta.i, f32); near(c, f.meta.mean, nll.slice(0, f.meta.i).reduce((a, b) => a + b, 0) / f.meta.i, 'running mean', 1e-4); });
      const e = phase(c.frames, 'end')[0].meta, ce = nll.reduce((a, b) => a + b, 0) / nll.length;
      near(c, e.ce, ce, 'cross-entropy', 1e-4); near(c, e.ppl, Math.exp(ce), 'perplexity', 1e-4); near(c, e.lnV, Math.log(vocab().length), 'ln V');
      c.eq(e.worst, 1 + argmax(p.map((x) => -x)), 'hardest character'); c.eq(e.scored, ids.length - 1, 'scored');
    },
  },
  'lm-gpt-training': {
    check(c) {
      const J = P.D().gpt;
      phase(c.frames, 'check').forEach((f) => {
        c.eq(f.meta.val, J.val.find((v) => v[0] === f.meta.step)[1], 'held-out loss at ' + f.meta.step);
        if (f.meta.step) { const w = J.hist.slice(Math.max(0, f.meta.step - 25), f.meta.step); near(c, f.meta.tAvg, w.reduce((a, b) => a + b, 0) / w.length, 'average training loss', 1e-9); }
        near(c, f.meta.ppl, Math.exp(f.meta.val), 'perplexity');
      });
      const m = phase(c.frames, 'mine')[0].meta, net = P.gpt(), alt = net.generate(enc(PROMPT), 64, { temperature: c.input.temperature }, c.input.seed, { cache: false });
      c.eq(fits(m.text, 0), fits(alt.ids.map((i) => vocab()[i]).join(''), 0), 'your sample, regenerated without the cache');
      replay.gpt(c);
    },
  },
  'lm-scaling': {
    check(c) {
      const runs = P.D().scale.runs, V = vocab().length, ctx = GPT.context;
      // parameters by formula: embeddings, positions, per block (two layer norms, four d×d projections, the
      // feed-forward d×4d and 4d×d, all with biases), the final layer norm; the output layer is tied
      const count = (d) => V * d + ctx * d + SCALE.layers * (4 * d + 4 * (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d)) + 2 * d;
      const ys = runs.map((r) => (c.input.measure === 'val' ? r.val[r.val.length - 1][1] : r.hist.slice(-50).reduce((a, b) => a + b, 0) / 50));
      phase(c.frames, 'model').forEach((f) => { c.eq(f.meta.params, count(f.meta.d), 'parameters of width ' + f.meta.d); near(c, f.meta.loss, ys[f.meta.k], 'loss of width ' + f.meta.d, 1e-9); });
      // the fit by the normal equations of a 2×2 system
      const x = runs.map((r) => Math.log(r.params)), y = ys.map(Math.log), n = x.length, sx = x.reduce((a, b) => a + b), sy = y.reduce((a, b) => a + b), sxx = x.reduce((a, v) => a + v * v, 0), sxy = x.reduce((a, v, i) => a + v * y[i], 0);
      const b = (n * sxy - sx * sy) / (n * sxx - sx * sx), fit = phase(c.frames, 'fit')[0].meta;
      near(c, fit.b, b, 'slope', 1e-9); near(c, fit.dbl, Math.pow(2, b), 'factor per doubling', 1e-9);
      replay.scale(c);
    },
  },
  'sampling-temperature': {
    random: (r) => ({ prompt: ['the boats ', 'the weaver said that ', 'in the winter '][Math.floor(r() * 3)], cool: 0.1 + Math.floor(r() * 9) / 10, hot: 1.1 + Math.floor(r() * 14) / 10, seed: 1 + Math.floor(r() * 900) }),
    check(c) {
      const net = P.gpt(), ids = enc(c.input.prompt), L = stepLogits(net, ids), z = L[L.length - 1], p1 = softmax(z), top = argmax(p1);
      const b = phase(c.frames, 'base')[0].meta;
      near(c, b.p1, p1[top], 'top probability', f32);
      near(c, phase(c.frames, 'cool')[0].meta.p, softmax(z, c.input.cool)[top], 'at the low temperature', f32);
      const h = phase(c.frames, 'hot')[0].meta; near(c, h.p, softmax(z, c.input.hot)[top], 'at the high temperature', f32);
      // greedy by hand, one step at a time through the cache
      const g = T.noGrad(() => { const cache = net.newCache(); let zz = null; for (const id of ids) zz = Array.from(net.forward(T.tensor([[id]]), { cache }).values()); let out = ''; for (let k = 0; k < GPT.context - 1 - ids.length; k++) { const id = argmax(zz); out += vocab()[id]; zz = Array.from(net.forward(T.tensor([[id]]), { cache }).values()); } return out; });
      c.eq(fits(phase(c.frames, 'greedy')[0].meta.text, ids.length), g, 'greedy text');
      for (const [ph, t] of [['coolText', c.input.cool], ['hotText', c.input.hot]]) {
        const m = phase(c.frames, ph)[0].meta, alt = net.generate(ids, 40, { temperature: t }, c.input.seed, { cache: false });
        c.eq(fits(m.text, ids.length), fits(alt.ids.slice(ids.length).map((i) => vocab()[i]).join(''), ids.length), ph + ' regenerated without the cache');
      }
      const hm = phase(c.frames, 'hotText')[0].meta, words = hm.text.split(/[ .,]+/).filter(Boolean), known = new Set(S.P13T.harbour.split(/[ .,]+/));
      c.eq([hm.words, hm.known], [words.length, words.filter((w) => known.has(w)).length], 'words and known words');
    },
  },
  'sampling-topk-topp': {
    random: (r) => ({ prompt: ['the boats ', 'the lighthou', 'a storm comes ', 'the weaver '][Math.floor(r() * 4)], k: 1 + Math.floor(r() * 26), p: Math.round((0.05 + r() * 0.95) * 20) / 20, temperature: Math.round((0.3 + r() * 1.7) * 10) / 10, seed: 1 + Math.floor(r() * 900) }),
    randomRuns: 10,
    check(c) {
      const net = P.gpt(), ids = enc(c.input.prompt), L = stepLogits(net, ids), pr = softmax(L[L.length - 1], c.input.temperature);
      const ord = pr.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]), V = pr.length, k = Math.min(c.input.k, V);
      const zk = ord.slice(0, k).reduce((a, o) => a + o[0], 0);
      let m = 0, cum = 0; for (const o of ord) { m++; cum += o[0]; if (cum >= c.input.p) break; }
      const a = phase(c.frames, 'all')[0].meta, tk = phase(c.frames, 'topk')[0].meta, tp = phase(c.frames, 'topp')[0].meta, d = phase(c.frames, 'draw')[0].meta;
      near(c, a.p1, ord[0][0], 'top probability', f32); near(c, a.tail, ord.slice(5).reduce((s, o) => s + o[0], 0), 'tail', f32);
      near(c, tk.zk, zk, 'top-k mass', f32); near(c, tk.p1k, ord[0][0] / zk, 'renormalised', f32);
      c.eq(tp.m, m, 'top-p count'); near(c, tp.zp, ord.slice(0, m).reduce((s, o) => s + o[0], 0), 'top-p mass', f32);
      // both rules, then the draw, as the engine's rule says: top-p inside the top k
      let keep = 0; cum = 0; for (const o of ord.slice(0, k)) { keep++; cum += o[0]; if (cum >= c.input.p) break; }
      c.eq(d.kept, keep, 'kept by both rules');
      const z2 = ord.slice(0, keep).reduce((s, o) => s + o[0], 0), r = T.rng(c.input.seed).next();
      let acc = 0, id = ord[keep - 1][1]; for (const o of ord.slice(0, keep)) { acc += o[0] / z2; if (r < acc) { id = o[1]; break; } }
      c.eq(d.id, id, 'the drawn character'); near(c, d.p, pr[id] / z2, 'its renormalised probability', f32);
    },
  },
  'kv-cache-generation': {
    random: (r) => ({ prompt: ['the keeper ', 'a ', 'the gulls cry ', 'at night '][Math.floor(r() * 4)], n: 2 + Math.floor(r() * 15) }),
    check(c) {
      const L = c.input.prompt.length, n = c.input.n, e = phase(c.frames, 'end')[0].meta;
      let tA = 0; for (let s = 0; s < n; s++) tA += L + s;
      c.eq([e.tA, e.tB], [tA, L + n - 1], 'positions through the blocks');
      c.ok(e.same, 'the same text both ways'); near(c, e.ratio, tA / (L + n - 1), 'ratio');
      c.eq(e.mem, GPT.layers * 2 * (L + n - 1) * GPT.d, 'numbers kept');
      phase(c.frames, 'step').forEach((f) => { let a = 0; for (let s = 0; s < f.meta.s; s++) a += L + s; c.eq([f.meta.totA, f.meta.totB, f.meta.b, f.meta.read], [a, L + f.meta.s - 1, 1, L + f.meta.s - 2], 'step ' + f.meta.s); });
      const g = P.gpt().generate(enc(c.input.prompt), n, { method: 'greedy' }, 1, { cache: false });
      c.eq(e.text, g.ids.slice(L).map((i) => vocab()[i]).join(''), 'the generated text');
    },
  },
  'lm-training-stages': {
    check(c) {
      const PAIRS = { sea: ['the sea is ', 'calm and grey.', 'made of bread.'], bread: ['the baker sells ', 'warm bread.', 'old boots.'], keeper: ['the keeper ', 'lights the lamp.', 'eats the boat.'] };
      const [pr, A, B] = PAIRS[c.input.pair], score = (net, ans) => { const ids = enc(pr + ans), L = stepLogits(net, ids); let s = 0; for (let i = pr.length; i < ids.length; i++) s += Math.log(softmax(L[i - 1])[ids[i]]); return s; };
      const p0 = phase(c.frames, 'pref0')[0].meta;
      near(c, p0.la, score(P.gpt(), A), 'log p(A)', 1e-4); near(c, p0.lb, score(P.gpt(), B), 'log p(B)', 1e-4); near(c, p0.loss, Math.log(2), 'loss starts at ln 2', 1e-5);
      near(c, p0.pA, Math.exp(p0.la / A.length), 'per character A'); near(c, p0.pB, Math.exp(p0.lb / B.length), 'per character B');
      const pf = phase(c.frames, 'pref')[0].meta, tuned = P.prefer(pr, A, B, c.input.beta, c.input.steps, 1e-4).net;
      const la = score(tuned, A), lb = score(tuned, B);
      near(c, pf.la, la, 'log p(A) after', 1e-4); near(c, pf.lb, lb, 'log p(B) after', 1e-4);
      near(c, pf.loss, Math.log(1 + Math.exp(-c.input.beta * ((la - lb) - (p0.la - p0.lb)))), 'preference loss by formula', 1e-4);
      const pre = phase(c.frames, 'pre')[0].meta; c.eq(pre.chars, S.P13T.harbour.length, 'characters'); c.eq(pre.loss, P.D().gpt.val[P.D().gpt.val.length - 1][1], 'final loss');
    },
  },
  'lm-in-context': {
    random: (r) => { const n = 2 + Math.floor(r() * 7), items = []; for (let i = 0; i < n; i++) items.push('abcdef'[Math.floor(r() * 6)] + (r() < 0.4 ? '?' : String(Math.floor(r() * 10)))); if (!items.some((x) => x[1] === '?')) items[n - 1] = items[n - 1][0] + '?'; return { seq: items.join(' ') }; },
    randomRuns: 10,
    check(c) {
      const net = P.icl(), items = c.input.seq.trim().split(/\s+/), toks = [], facts = new Map();
      let k = 0;
      const asks = phase(c.frames, 'ask');
      items.forEach((x) => {
        const key = 10 + 'abcdef'.indexOf(x[0]);
        toks.push(key);
        if (x[1] !== '?') { toks.push(+x[1]); facts.set(key, +x[1]); return; }
        const L = stepLogits(net, toks), pr = softmax(L[L.length - 1]).slice(0, 10), best = argmax(pr), f = asks[k++].meta;
        c.eq(f.best, best, 'answer'); near(c, f.p, pr[best], 'its probability', f32);
        c.eq([f.known, f.want], [facts.has(key), facts.has(key) ? facts.get(key) : null], 'seen before');
        if (f.known) { c.eq(f.pos, f.top + 1, 'position named'); c.ok(f.wTop > 0 && f.wTop <= 1, 'an attention weight'); }
        toks.push(best);
      });
      c.eq(k, asks.length, 'every question asked');
      const e = phase(c.frames, 'end')[0].meta, val = iclTask(S).examples, z = net.predict(T.tensor(val.map((s) => s.slice(0, -1)))), V = z.shape[2], n = val[0].length - 1, v = z.values();
      const acc = { seen: [0, 0], fresh: [0, 0] };
      val.forEach((s, b) => { const seen = new Set(); for (let i = 0; i < s.length; i += 2) { const row = Array.from(v.slice((b * n + i) * V, (b * n + i) * V + 10)), o = seen.has(s[i]) ? acc.seen : acc.fresh; o[0] += argmax(row) === s[i + 1]; o[1]++; seen.add(s[i]); } });
      near(c, e.seen, acc.seen[0] / acc.seen[1], 'accuracy on seen keys', 1e-12); near(c, e.fresh, acc.fresh[0] / acc.fresh[1], 'accuracy on new keys', 1e-12);
      c.eq(e.total, val.length * (n + 1) / 2, 'questions'); c.eq(e.seqs, val.length, 'sequences');
    },
  },
  'lm-limits': {
    check(c) {
      const net = P.gpt(), text = S.P13T.harbour, all = text.split(/[^a-z]+/);
      const w = phase(c.frames, 'words')[0].meta, words = c.input.prompt.split(/[ .,]+/).filter(Boolean);
      c.eq(w.counts, words.map((x) => all.filter((y) => y === x).length), 'word counts');
      const g = phase(c.frames, 'gen')[0].meta, alt = net.generate(enc(c.input.prompt), 40, { temperature: 0.8 }, c.input.seed, { cache: false });
      c.eq(fits(g.text, c.input.prompt.length), fits(alt.ids.slice(c.input.prompt.length).map((i) => vocab()[i]).join(''), c.input.prompt.length), 'continuation regenerated without the cache');
      const meanAfter = (pr, t) => { const ids = enc(pr + t).slice(0, 32), L = stepLogits(net, ids), ps = []; for (let i = pr.length; i < ids.length; i++) ps.push(softmax(L[i - 1])[ids[i]]); return ps; };
      const ps = meanAfter(c.input.prompt, g.text); near(c, g.mean, ps.reduce((a, b) => a + b, 0) / ps.length, 'mean probability', 1e-4);
      const real = 'the sailor said that he would sail to the far rocks.', rp = meanAfter(real.slice(0, 21), real.slice(21));
      near(c, phase(c.frames, 'conf')[0].meta.rmean, rp.reduce((a, b) => a + b, 0) / rp.length, 'true sentence', 1e-4);
      const role = c.input.role, cnt = (s) => text.split('the ' + role + ' said that ' + s + ' ').length - 1, ct = phase(c.frames, 'counts')[0].meta;
      c.eq([ct.nHe, ct.nShe, ct.aHe, ct.aShe], [cnt('he'), cnt('she'), all.filter((x) => x === 'he').length, all.filter((x) => x === 'she').length], 'counts');
      const chain = (s) => meanAfter('the ' + role + ' said that ', s).reduce((a, b) => a * b, 1), md = phase(c.frames, 'model')[0].meta;
      near(c, md.pHe, chain('he '), 'p(he)', 1e-4); near(c, md.pShe, chain('she '), 'p(she)', 1e-4);
    },
  },
};

// ---- replays of the stored runs (once per test run) ----
const done = new Set();
const replay = {
  gpt(c) {
    if (done.has('gpt')) return; done.add('gpt');
    const J = P.D().gpt, n = 20, { net, trainer } = gptSetup(S, GPT, {}, GPT_STEPS);
    c.eq(net.complete(PROMPT, SAMPLE.n, { temperature: SAMPLE.temperature }, SAMPLE.seed).text, J.samples[0].text, 'the step-0 sample replayed');
    for (let k = 0; k < n; k++) near(c, trainer.step().loss, J.hist[k], 'gpt step ' + (k + 1) + ' replayed', 1e-4);
    const { net: n2, trainer: t2 } = gptSetup(S, GPT, {}, GPT_STEPS);
    n2.loadState(P.gpt().stateDict());
    near(c, t2.evaluate('val', 10000).loss, J.val[J.val.length - 1][1], 'held-out loss of the stored weights', 1e-4);
  },
  scale(c) {
    if (done.has('scale')) return; done.add('scale');
    const r = P.D().scale.runs[0], { trainer } = gptSetup(S, { d: r.d, heads: SCALE.heads, layers: SCALE.layers, context: GPT.context, dropout: 0.1 }, {}, SCALE.steps);
    for (let k = 0; k < 10; k++) near(c, trainer.step().loss, r.hist[k], 'width ' + r.d + ' step ' + (k + 1) + ' replayed', 1e-4);
  },
};

run('claims-p13', /^p13-/, checks, ALLOWED);
