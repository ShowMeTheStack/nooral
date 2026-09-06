// Part 13, byte-pair encoding: the merges learned one by one from a text of our own (DL.bpe.train and its trace),
// with the commonest pairs counted at every step; then a new sentence cut with those merges, replayed in the order
// they were learned.
(function (S) {
  'use strict';
  const P = S.P13, H = () => S.Kit.ml, BPE = () => S.DL.bpe;
  const TEXTS = { harbour: 'the harbour town (7,000 characters)', garden: 'the garden (600 characters)' };
  const textOf = (id) => S.P13T[id];
  /** a passage shown on the stage: the first sentences, up to about n characters */
  const passage = (text, n) => { const s = text.split('. '); let out = s[0] + '.'; for (let i = 1; i < s.length && (out + ' ' + s[i]).length <= n; i++) out += ' ' + s[i] + '.'; return out; };
  /** the training, kept per text and number of merges */
  P.bpeRun = (id, m) => P.once('bpe:' + id + ':' + m, () => BPE().train(textOf(id), m, { sample: passage(textOf(id), 100), sampleEvery: 1 }));
  /** every pair of neighbouring tokens in the text cut by the first k merges, with its count, commonest first */
  P.pairCounts = (id, merges, k) => {
    const tok = BPE().fromMerges(P.bpeRun(id, merges).base, P.bpeRun(id, merges).merges.slice(0, k)), c = new Map();
    for (const ch of BPE().pretokenize(textOf(id))) { const sym = tok.split(ch); for (let i = 0; i + 1 < sym.length; i++) { const key = sym[i] + '\u0000' + sym[i + 1]; c.set(key, (c.get(key) || 0) + 1); } }
    return [...c.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(([key, n]) => ({ pair: key.split('\u0000'), n }));
  };
  const pl = (p) => '“' + P.show(p[0]) + '” + “' + P.show(p[1]) + '”';

  S.topic({
    id: 'bpe-build', family: 'p13-tokens', order: 10, title: 'Byte-pair encoding: learning merges from a text',
    blurb: 'A language model does not read letters or whole words but tokens, pieces learned from text. Byte-pair encoding starts with single characters, counts every pair of neighbours in the whole text, and merges the commonest pair into one new token. Then it counts again. Watch the vocabulary grow and the text shrink, merge by merge, on a text written for Nooral.',
    inputs: [
      { id: 'text', kind: 'choice', label: 'Text', def: 'harbour', options: Object.keys(TEXTS).map((k) => ({ value: k, label: TEXTS[k] })) },
      { id: 'merges', kind: 'number', label: 'Merges', def: 12, min: 3, max: 24 },
    ],
    presets: [
      { label: 'Harbour, 12 merges', values: { text: 'harbour', merges: 12 } },
      { label: 'Harbour, 24 merges', values: { text: 'harbour', merges: 24 } },
      { label: 'Garden, 10 merges', values: { text: 'garden', merges: 10 } },
      { label: 'Garden, 3 merges', values: { text: 'garden', merges: 3 } },
    ],
    counters: [{ id: 'merges', label: 'merges' }, { id: 'tokens', label: 'tokens in the text' }],
    run(inp) {
      const run = P.bpeRun(inp.text, inp.merges), tr = run.trace, F = H().frames(['merges', 'tokens']), q = F.q;
      const start = tr[0], steps = tr.filter((s) => s.kind === 'merge'), chars = textOf(inp.text).length;
      const strip = (pieces, lit) => {
        const st = S.Kit.tokens.strip({ prefix: 'bp', x: 0, y: 34, width: 790, tokens: pieces, size: 13, gap: 3, lineGap: 7, cls: (i) => (lit && pieces[i] === lit ? 'new' : '') });
        return [H().text('bp:t', 0, 16, 'the first lines of the text, cut into tokens (· is a space)', 'strong', 13)].concat(st.items);
      };
      const pairsPanel = (k, next) => {
        const rows = P.pairCounts(inp.text, inp.merges, k).slice(0, 6);
        const items = P.panel('pp', 0, 196, 380, 214, 'commonest neighbouring pairs now');
        return items.concat(P.bars('pp:b', 14, 236, rows.map((r) => ({ label: P.show(r.pair[0]) + ' + ' + P.show(r.pair[1]), v: r.n, text: '×' + r.n, cls: next && r.pair[0] === next[0] && r.pair[1] === next[1] ? 'hot' : 'soft' })), { lw: 96, bw: 210, rh: 28 }));
      };
      const vocabPanel = (k) => {
        const items = P.panel('vp', 410, 196, 380, 214, 'vocabulary: ' + start.base.length + ' characters + ' + H().pl(k, 'merge'));
        const show = steps.slice(Math.max(0, k - 6), k);
        show.forEach((s, i) => items.push(H().text('vp:' + (s.rank + 1), 426, 250 + i * 26, (s.rank + 1) + '. ' + P.show(s.pair[0]) + ' + ' + P.show(s.pair[1]) + ' → ' + P.show(s.token) + '   ×' + s.count, 'mono ' + (s.rank === k - 1 ? 'p13new' : ''), 13)));
        if (!k) items.push(H().text('vp:none', 426, 250, 'only single characters so far', 'legend', 13));
        return items;
      };
      const foot = (tokens) => [H().text('ft', 0, 440, 'the whole text: ' + tokens.toLocaleString('en') + ' tokens from ' + chars.toLocaleString('en') + ' characters', 'legend', 13)];
      const first = steps[0];
      F.set('tokens', start.tokens);
      F.push([...strip(Array.from(passage(textOf(inp.text), 100))), ...pairsPanel(0, first && first.pair), ...vocabPanel(0), ...foot(start.tokens)],
        `The text has ${q.n(chars)} characters, so it starts as ${q.n(start.tokens)} tokens, one per character, from a vocabulary of ${q.n(start.base.length)} different characters. Every pair of neighbours is counted, never across a word boundary: the commonest pair is ${pl(first.pair)}, seen ${q.n(first.count)} times.`, { phase: 'start', tokens: start.tokens, chars, base: start.base.length, count: first.count, pair: first.pair });
      let prev = start.tokens;
      steps.forEach((s, k) => {
        F.count('merges').set('tokens', s.tokens);
        const next = steps[k + 1];
        F.push([...strip(s.sample || [], s.token), ...pairsPanel(k + 1, next && next.pair), ...vocabPanel(k + 1), ...foot(s.tokens)],
          `Merge ${q.n(k + 1)}: ${pl(s.pair)} becomes the new token “${P.show(s.token)}”, wherever the pair stands: ${q.n(s.count)} places. The whole text is now ${q.n(s.tokens)} tokens, ${q.n(prev - s.tokens)} fewer.` + (next ? ` Counting again, the commonest pair is ${pl(next.pair)} (${q.n(next.count)} times).` : ''),
          { phase: 'merge', k: k + 1, pair: s.pair, token: s.token, count: s.count, tokens: s.tokens, saved: prev - s.tokens, next: next ? next.pair : null, nextCount: next ? next.count : null });
        prev = s.tokens;
      });
      const last = steps[steps.length - 1], words = steps.filter((s) => s.token.length >= 3).map((s) => '“' + P.show(s.token) + '”');
      F.push([...strip(last.sample || [], null), ...pairsPanel(steps.length, null), ...vocabPanel(steps.length), ...foot(last.tokens)],
        `After ${P.qpl(q, steps.length, 'merge')} the vocabulary holds ${q.n(start.base.length + steps.length)} tokens and the text needs ${q.n(last.tokens)} of them, ${q(chars / last.tokens, 2)} characters per token on average.` + (words.length ? ` Common pieces such as ${H().list(words.slice(0, 3))} are now single tokens.` : '') + ' Real tokenizers run tens of thousands of merges, so most common words become one token.',
        { phase: 'end', vocab: start.base.length + steps.length, tokens: last.tokens, ratio: chars / last.tokens, merges: steps.length });
      return { frames: F.list, def: { sections: [{ id: 'alg', kind: 'math', title: 'Byte-pair encoding (training)', text: ['start: one token per character', 'repeat:', '  count every pair of neighbours', '  (inside a word, with its leading space)', '  merge the commonest pair everywhere', '  add it to the vocabulary', 'ties: the pair that sorts first'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'merge' && f.meta.k === 2 && f.meta.next,
      once: true, pick: 'option',
      ask: 'Which pair will be merged next?',
      options: (fr, i, inp) => { const rows = P.pairCounts(inp.text, inp.merges, 2).slice(0, 4).map((r) => P.show(r.pair[0]) + ' + ' + P.show(r.pair[1])); return rows.slice().sort(); },
      answer: (fr, i) => P.show(fr[i].meta.next[0]) + ' + ' + P.show(fr[i].meta.next[1]),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'Byte-pair encoding chooses the next merge by…', options: ['counting every pair of neighbouring tokens and taking the commonest', 'taking the longest word', 'asking which pair helps the model most', 'alphabetical order'], answer: 0, why: 'It is a pure counting rule over the training text.' },
      { q: 'Each merge makes the text…', options: ['shorter in tokens and the vocabulary one larger', 'longer in tokens', 'shorter in characters', 'the same length in tokens'], answer: 0, why: 'Every place the pair stood now holds one token, and the new token joins the vocabulary.' },
      { q: 'Why is “·the” (with its space) a likely early merge in English-like text?', options: ['“the” is very common and usually follows a space', 'spaces are removed first', 'it is the shortest word', 'merges always include spaces'], answer: 0, why: 'Frequent words with their leading space become single tokens early.' },
      { q: 'A word never seen in the training text is…', options: ['still cut into known smaller pieces, down to single characters', 'dropped', 'always one unknown token', 'spelled backwards'], answer: 0, why: 'The single characters are always in the vocabulary, so any word made of them can be written.' },
      { q: 'More merges give…', options: ['a larger vocabulary and shorter token sequences', 'a smaller vocabulary', 'longer sequences', 'no change after the first merge'], answer: 0, why: 'That is the trade-off a tokenizer’s size sets.' },
    ],
  });

  // ---- cutting a sentence with the learned merges ----
  /** the merges that change a sentence, in rank order, with the pieces after each */
  P.tokenise = (sentence, run) => {
    const rank = new Map(run.merges.map((m, i) => [m[0] + '\u0000' + m[1], i]));
    let chunks = BPE().pretokenize(sentence).map((c) => Array.from(c));
    const steps = [];
    for (let r = 0; r < run.merges.length; r++) {
      const [a, b] = run.merges[r];
      let n = 0;
      chunks = chunks.map((sym) => { const out = []; for (let i = 0; i < sym.length; i++) { if (i + 1 < sym.length && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; n++; } else out.push(sym[i]); } return out; });
      if (n) steps.push({ rank: r, pair: [a, b], token: a + b, n, pieces: chunks.flat() });
    }
    return { chunks: BPE().pretokenize(sentence), steps, pieces: chunks.flat(), rank };
  };

  S.topic({
    id: 'bpe-tokenise', family: 'p13-tokens', order: 20, title: 'Cutting a sentence into tokens',
    blurb: 'Once the merges are learned, a new sentence is cut by replaying them in the order they were learned: first into chunks (a word with its leading space, a run of punctuation), then each chunk from single characters up. Common words end as one token; rare and new words fall apart into several pieces.',
    inputs: [
      { id: 'sentence', kind: 'text', label: 'Sentence', def: 'the sailor sells a blue sail at the market.', check: (v) => (typeof v !== 'string' || v.length < 3 ? 'Sentence: at least 3 characters' : v.length > 70 ? 'Sentence: at most 70 characters, so it fits the stage' : null) },
      { id: 'merges', kind: 'number', label: 'Merges learned (harbour text)', def: 60, min: 5, max: 150 },
    ],
    presets: [
      { label: 'Common words', values: { sentence: 'the sailor sells a blue sail at the market.', merges: 60 } },
      { label: 'Few merges', values: { sentence: 'the sailor sells a blue sail at the market.', merges: 10 } },
      { label: 'Many merges', values: { sentence: 'the weaver said that she would finish the cloth.', merges: 150 } },
      { label: 'New words', values: { sentence: 'a jazz quartet played in the lighthouse.', merges: 100 } },
      { label: 'Capitals', values: { sentence: 'The Keeper climbs.', merges: 60 } },
    ],
    counters: [{ id: 'tokens', label: 'tokens' }],
    run(inp) {
      const run = P.bpeRun('harbour', inp.merges), tok = run.tokenizer, t = P.tokenise(inp.sentence, run), F = H().frames(['tokens']), q = F.q;
      const chars = Array.from(inp.sentence), unk = [...new Set(chars.filter((c) => !run.base.includes(c)))];
      const strip = (pieces, lit, sub) => S.Kit.tokens.strip({ prefix: 'tk', x: 0, y: 40, width: 790, tokens: pieces, size: 14, gap: 4, lineGap: sub ? 22 : 8, cls: (i) => (lit && lit(i) ? 'new' : unk.includes(pieces[i]) ? 'p13unk' : ''), sub }).items;
      const head = (s) => [H().text('tk:h', 0, 18, s, 'strong', 13)];
      const applied = (k) => {
        const items = P.panel('ap', 0, 170, 790, 200, 'merges that change this sentence, in the order they were learned');
        const list = t.steps.slice(0, k), shown = list.slice(-10);
        shown.forEach((s, i) => items.push(H().text('ap:' + s.rank, 16 + (i % 2) * 390, 218 + Math.floor(i / 2) * 28, 'merge ' + (s.rank + 1) + ': ' + P.show(s.pair[0]) + ' + ' + P.show(s.pair[1]) + ' → ' + P.show(s.token), 'mono ' + (i === shown.length - 1 ? 'p13new' : ''), 13)));
        if (!k) items.push(H().text('ap:none', 16, 218, 'none yet: every character is its own token', 'legend', 13));
        return items;
      };
      F.set('tokens', chars.length);
      F.push([...head('the sentence as characters'), ...strip(chars), ...applied(0)],
        `The sentence has ${q.n(chars.length)} characters. It is first cut into ${P.qpl(q, t.chunks.length, 'chunk')} (each word keeps the space in front of it), and every chunk starts as single characters.` + (unk.length ? ` ${H().list(unk.map(P.name))} never appeared in the training text, so ${unk.length === 1 ? 'it has' : 'they have'} no token: it becomes the unknown token.` : ''), { phase: 'start', chars: chars.length, chunks: t.chunks.length, unk });
      // at most fourteen frames: when more merges change the sentence, each frame replays a few
      const per = Math.max(1, Math.ceil(t.steps.length / 14));
      for (let k = 0; k < t.steps.length; k += per) {
        const group = t.steps.slice(k, k + per), last = group[group.length - 1], made = new Set(group.map((s) => s.token));
        F.set('tokens', last.pieces.length);
        F.push([...head('after merge ' + (last.rank + 1) + ' of ' + run.merges.length), ...strip(last.pieces, (i) => made.has(last.pieces[i])), ...applied(k + group.length)],
          (group.length === 1 ? `Merge ${q.n(last.rank + 1)}, ${pl(last.pair)}, fits ${last.n === 1 ? 'one place' : q.n(last.n) + ' places'} here` : `Merges ${H().list(group.map((s) => q.n(s.rank + 1)))} fit here, making ${H().list(group.map((s) => '“' + P.show(s.token) + '”'))}`) + `: the sentence is now ${P.qpl(q, last.pieces.length, 'token')}.`,
          { phase: 'merge', ranks: group.map((s) => s.rank), nums: group.map((s) => s.rank + 1), tokens: last.pieces.length, n: last.n });
      }
      const ids = tok.encode(inp.sentence), pieces = ids.map((i) => tok.vocab[i]), multi = t.chunks.filter((c) => tok.split(c).length >= 3).map((c) => '“' + P.show(c) + '”');
      F.set('tokens', ids.length);
      F.push([...head('the tokens and their numbers (ids)'), ...strip(pieces.map((p, i) => (ids[i] === 0 ? '?' : p)), null, (i) => String(ids[i])), ...applied(t.steps.length)],
        `The model receives ${P.qpl(q, ids.length, 'token')}, each as its number in a vocabulary of ${q.n(tok.size)}: ${q(chars.length / ids.length, 2)} characters per token.` + (multi.length ? ` ${H().list(multi.slice(0, 3))} still fall${multi.length === 1 ? 's' : ''} into three or more pieces: rarer words cost more tokens.` : ' Every word here is common enough to need at most two tokens.'),
        { phase: 'end', tokens: ids.length, ids, vocab: tok.size, ratio: chars.length / ids.length, pieces });
      return { frames: F.list, def: { sections: [{ id: 'enc', kind: 'math', title: 'Encoding with learned merges', text: ['cut into chunks (a word with its space)', 'each chunk: single characters', 'apply the merges in learned order', 'look up each piece’s id', 'unknown character → id 0'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start',
      once: true, pick: 'option',
      ask: 'How many tokens will the sentence end as?',
      options: (fr) => { const e = fr[fr.length - 1].meta.tokens, s = fr[0].meta; return [...new Set([e, s.chars, s.chunks, e + 3])].sort((a, b) => a - b).map(String); },
      answer: (fr) => String(fr[fr.length - 1].meta.tokens),
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'In what order are the merges applied to a new sentence?', options: ['the order they were learned', 'longest first', 'left to right, any merge that fits', 'at random'], answer: 0, why: 'Replaying in learned order gives the same pieces the training text was cut into.' },
      { q: 'A word the tokenizer saw often during training usually becomes…', options: ['one token (with its leading space)', 'one token per letter', 'an unknown token', 'two tokens always'], answer: 0, why: 'Its pairs were common, so they were merged all the way up.' },
      { q: 'Why does a rare word cost more tokens?', options: ['few merges cover its letter pairs, so it stays in small pieces', 'rare words are padded', 'the model refuses them', 'it is split into syllables by a dictionary'], answer: 0, why: 'Merges follow frequency in the training text.' },
      { q: 'A character never seen in training (a capital here) becomes…', options: ['the unknown token', 'a space', 'two tokens', 'the commonest token'], answer: 0, why: 'Real tokenizers avoid this by starting from the 256 possible bytes, so nothing is ever unknown.' },
      { q: 'The model itself sees…', options: ['the list of token ids', 'the letters', 'the merge rules', 'the spaces only'], answer: 0, why: 'Each id selects one row of the embedding matrix.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
