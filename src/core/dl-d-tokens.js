// Tokenizers: characters, words, and byte-pair encoding (BPE) trained merge by merge with a trace.
//   DL.charTokenizer(text) -> tok      DL.wordTokenizer(texts, {minCount}) -> tok (ids 0 <pad>, 1 <unk>)
//   tok: {kind, vocab: [strings], size, encode(text) -> ids, decode(ids) -> text, toJSON()}   DL.tokenizer(json) rebuilds
//   DL.bpe.train(text, merges, {sample}) -> {tokenizer, trace}: each step {kind: 'merge', rank, pair, token, count,
//      tokens (the corpus length in tokens after it), sample (how opts.sample splits now, every few merges), say}
//   DL.bpe.pretokenize(text) -> chunks (words with their leading space, numbers, punctuation, runs of spaces)
//   BPE ties (pairs seen equally often) go to the pair that sorts first, so training is deterministic.
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});
  const show = (s) => JSON.stringify(s).slice(1, -1).replace(/ /g, '·');
  const num = (n) => n.toLocaleString('en');

  const finish = (tok) => {
    tok.size = tok.vocab.length;
    tok.index = new Map(tok.vocab.map((v, i) => [v, i]));
    return tok;
  };

  DL.charTokenizer = (text, extra) => {
    const vocab = Array.isArray(text) ? text.slice() : [...new Set(text)].sort();
    const tok = finish({ kind: 'char', vocab });
    tok.unk = tok.index.has(' ') ? tok.index.get(' ') : 0;
    tok.encode = (s) => Array.from(s, (ch) => (tok.index.has(ch) ? tok.index.get(ch) : tok.unk));
    tok.decode = (ids) => Array.from(ids, (i) => tok.vocab[i] || '').join('');
    tok.toJSON = () => ({ kind: 'char', vocab: tok.vocab });
    return Object.assign(tok, extra || {});
  };

  DL.words = (s) => s.toLowerCase().match(/[a-z]+(?:'[a-z]+)?|[0-9]+/g) || [];
  DL.wordTokenizer = (texts, o = {}) => {
    let vocab = o.vocab;
    if (!vocab) {
      const c = new Map();
      for (const t of texts) for (const w of DL.words(t)) c.set(w, (c.get(w) || 0) + 1);
      const words = [...c.entries()].filter(([, n]) => n >= (o.minCount || 1)).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map((x) => x[0]);
      vocab = ['<pad>', '<unk>'].concat(words);
    }
    const tok = finish({ kind: 'word', vocab, pad: 0, unk: 1 });
    tok.encode = (s) => DL.words(s).map((w) => (tok.index.has(w) ? tok.index.get(w) : 1));
    tok.decode = (ids) => Array.from(ids).filter((i) => i > 0).map((i) => tok.vocab[i]).join(' ');
    tok.toJSON = () => ({ kind: 'word', vocab: tok.vocab });
    return tok;
  };

  const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;
  const BPE = (DL.bpe = {});
  BPE.pretokenize = (text) => text.match(PRE) || [];

  /** a BPE tokenizer from its base symbols and ordered merges */
  BPE.fromMerges = (base, merges) => {
    const vocab = ['<unk>'].concat(base, merges.map(([a, b]) => a + b));
    const tok = finish({ kind: 'bpe', vocab, base: base.slice(), merges: merges.map((m) => m.slice()), unk: 0 });
    const rank = new Map(merges.map(([a, b], i) => [a + '\u0000' + b, i]));
    const cache = new Map();
    /** the symbols of one chunk after applying the merges, lowest rank first */
    tok.split = (chunk) => {
      if (cache.has(chunk)) return cache.get(chunk);
      let sym = Array.from(chunk);
      for (;;) {
        let best = Infinity, at = -1;
        for (let i = 0; i + 1 < sym.length; i++) { const r = rank.get(sym[i] + '\u0000' + sym[i + 1]); if (r !== undefined && r < best) { best = r; at = i; } }
        if (at < 0) break;
        const [a, b] = merges[best], out = [];
        for (let i = 0; i < sym.length; i++) { if (i + 1 < sym.length && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; } else out.push(sym[i]); }
        sym = out;
      }
      if (cache.size < 50000) cache.set(chunk, sym);
      return sym;
    };
    tok.encode = (text) => {
      const ids = [];
      for (const ch of BPE.pretokenize(text)) for (const s of tok.split(ch)) ids.push(tok.index.has(s) ? tok.index.get(s) : 0);
      return ids;
    };
    tok.decode = (ids) => Array.from(ids, (i) => (i === 0 ? '�' : tok.vocab[i] || '')).join('');
    tok.pieces = (text) => tok.encode(text).map((i) => tok.vocab[i]);
    tok.toJSON = () => ({ kind: 'bpe', base: tok.base, merges: tok.merges });
    return tok;
  };

  /** train numMerges merges on text; the trace has one step per merge */
  BPE.train = (text, numMerges = 200, o = {}) => {
    const freq = new Map();
    for (const ch of BPE.pretokenize(text)) freq.set(ch, (freq.get(ch) || 0) + 1);
    const base = [...new Set(text)].sort();
    const words = [...freq.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([w, n]) => ({ sym: Array.from(w), n }));
    let tokens = words.reduce((s, w) => s + w.n * w.sym.length, 0);
    const merges = [], trace = [];
    trace.push(DL.step('start', 'The text starts as ' + num(tokens) + ' tokens, one per character, from ' + base.length + ' distinct characters in ' + num(words.length) + ' distinct chunks.', { tokens, base: base.slice() }));
    const every = o.sampleEvery || Math.max(1, Math.round(numMerges / 20));
    for (let m = 0; m < numMerges; m++) {
      const pc = new Map();
      for (const w of words) for (let i = 0; i + 1 < w.sym.length; i++) { const k = w.sym[i] + '\u0000' + w.sym[i + 1]; pc.set(k, (pc.get(k) || 0) + w.n); }
      let bestK = null, bestN = 0;
      for (const [k, n] of pc) if (n > bestN || (n === bestN && k < bestK)) { bestK = k; bestN = n; }
      if (!bestK || bestN < (o.minCount || 2)) { trace.push(DL.step('stop', 'No pair appears ' + (o.minCount || 2) + ' times any more, so training stops after ' + m + ' merges.', { merges: m })); break; }
      const [a, b] = bestK.split('\u0000'), tokn = a + b;
      let saved = 0;
      for (const w of words) {
        if (w.sym.length < 2) continue;
        const out = [];
        for (let i = 0; i < w.sym.length; i++) { if (i + 1 < w.sym.length && w.sym[i] === a && w.sym[i + 1] === b) { out.push(tokn); i++; saved += w.n; } else out.push(w.sym[i]); }
        w.sym = out;
      }
      tokens -= saved;
      merges.push([a, b]);
      const st = DL.step('merge', 'Merge ' + (m + 1) + ': "' + show(a) + '" + "' + show(b) + '" → "' + show(tokn) + '", seen ' + num(bestN) + ' times; the text is now ' + num(tokens) + ' tokens.', { rank: m, pair: [a, b], token: tokn, count: bestN, tokens });
      if (o.sample && (m % every === 0 || m + 1 === numMerges)) st.sample = BPE.fromMerges(base, merges).pieces(o.sample);
      trace.push(st);
    }
    const tokenizer = BPE.fromMerges(base, merges);
    return { tokenizer, trace, merges, base };
  };

  DL.tokenizer = (j) => {
    if (!j) return null;
    if (j.encode) return j;
    if (j.kind === 'char') return DL.charTokenizer(j.vocab);
    if (j.kind === 'word') return DL.wordTokenizer(null, { vocab: j.vocab });
    if (j.kind === 'bpe') return BPE.fromMerges(j.base, j.merges);
    throw new Error('unknown tokenizer ' + j.kind);
  };
})(globalThis.Nooral = globalThis.Nooral || {});
