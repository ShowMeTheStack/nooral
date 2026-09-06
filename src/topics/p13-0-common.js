// Part 13, language models: the families, the trained models rebuilt from the stored runs (S.P13D, made by
// tests/ref/p13_runs.mjs: a character GPT on the harbour text, and a GPT that learned key–value lookups in context),
// the next-token probabilities read from them, and the pieces of picture the topics share (probability bars, a
// token strip, a text box, a stage panel).
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'llm', order, title, blurb });
  fam('p13-tokens', 10, 'Tokens', 'How a language model cuts text into pieces: byte-pair encoding learning its merges from a text, then cutting a new sentence with them.');
  fam('p13-lm', 20, 'Predicting the next token', 'A language model gives a probability to every possible next token. Cross-entropy and perplexity score those guesses, a tiny GPT learns to make them better, and larger models make them better still.');
  fam('p13-sampling', 30, 'Generating text', 'Turning the probabilities into text: greedy choice, temperature, top-k and top-p, and the key–value cache that keeps generation fast.');
  fam('p13-beyond', 40, 'From a tiny model to a large one', 'The stages that turn a next-token predictor into an assistant, learning from examples in the prompt, and the limits that remain.');

  const P = (S.P13 = {}), H = () => S.Kit.ml;
  const memo = new Map();
  P.once = (key, f) => { if (!memo.has(key)) memo.set(key, f()); return memo.get(key); };
  P.D = () => S.P13D;

  // ---- base64 by hand (the checks run without atob or Buffer) ----
  const AL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/', REV = new Int16Array(128).fill(-1);
  for (let i = 0; i < 64; i++) REV[AL.charCodeAt(i)] = i;
  P.bytes = (str) => {
    const clean = str.replace(/=+$/, ''), n = Math.floor((clean.length * 3) / 4), out = new Uint8Array(n);
    let buf = 0, bits = 0, k = 0;
    for (let i = 0; i < clean.length; i++) { buf = ((buf << 6) | REV[clean.charCodeAt(i)]) & 0xffffff; bits += 6; if (bits >= 8) { bits -= 8; out[k++] = (buf >> bits) & 255; } }
    return out;
  };
  P.f32 = (str) => { const b = P.bytes(str); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
  const stateOf = (w) => Object.fromEntries(Object.entries(w).map(([n, e]) => [n, { shape: e.shape, data: P.f32(e.d) }]));

  // ---- the models ----
  /** the trained character GPT, with its tokenizer */
  P.gpt = () => P.once('gpt', () => {
    const J = P.D().gpt;
    const net = S.DL.build('gpt', Object.assign({}, J.config, { tokenizer: { kind: 'char', vocab: J.vocab } }));
    net.loadState(stateOf(J.weights));
    net.eval();
    return net;
  });
  /** the same model before training: its first weights, from the same seed */
  P.gpt0 = () => P.once('gpt0', () => { const J = P.D().gpt, net = S.DL.build('gpt', Object.assign({}, J.config, { seed: 1, tokenizer: { kind: 'char', vocab: J.vocab } })); net.eval(); return net; });
  P.vocab = () => P.D().gpt.vocab;
  /** the in-context model: keys a–f are tokens 10–15, values 0–9 */
  P.icl = () => P.once('icl', () => { const J = P.D().icl, net = S.DL.build('gpt', J.config); net.loadState(stateOf(J.weights)); net.eval(); return net; });
  P.KEYS = 'abcdef';

  /** softmax of a row of logits, with an optional temperature (in double precision) */
  P.softmax = (row, temp) => { const t = temp || 1; let mx = -Infinity; for (const v of row) mx = Math.max(mx, v); const e = Array.from(row, (v) => Math.exp((v - mx) / t)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
  /** the logits the model gives after each position of ids: [[V numbers] per position] */
  P.logits = (net, ids) => {
    const z = net.predict(S.T.tensor([ids])), V = z.shape[2], v = z.values();
    return Array.from({ length: ids.length }, (_, i) => Array.from(v.subarray(i * V, (i + 1) * V)));
  };
  /** the next-token probabilities after a text */
  P.next = (net, text) => { const ids = net.tokenizer.encode(text), L = P.logits(net, ids.slice(-net.context)); return { ids, logits: L[L.length - 1], probs: P.softmax(L[L.length - 1]) }; };
  /** the characters the model does not know (it reads them as a space) */
  P.unknown = (text) => [...new Set([...text].filter((c) => !P.vocab().includes(c)))];
  P.checkText = (label, min, max) => (v) => {
    if (typeof v !== 'string') return label + ': some text';
    if (v.length < min) return label + ': at least ' + min + ' characters';
    if (v.length > max) return label + ': at most ' + max + ' characters, so it fits the model’s window';
    const bad = P.unknown(v);
    return bad.length ? label + ': the model only knows lower-case letters, spaces, full stops and commas (not “' + bad.join('') + '”)' : null;
  };

  // ---- text ----
  P.show = (t) => String(t).replace(/ /g, '·');
  /** a character as the narration names it */
  P.name = (c) => (c === ' ' ? 'a space' : c === '.' ? 'a full stop' : c === ',' ? 'a comma' : '“' + c + '”');
  /** "1 token" / "3 tokens" with the number quoted */
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');
  /** wrap text into lines of at most n characters, at spaces when it can */
  P.wrap = (text, n) => {
    const out = [];
    let s = text;
    while (s.length > n) { let k = s.lastIndexOf(' ', n); if (k < n / 2) k = n; out.push(s.slice(0, k)); s = s.slice(k).replace(/^ /, ''); }
    out.push(s);
    return out;
  };

  // ---- pictures ----
  const r2 = (v) => Math.round(v * 100) / 100;
  /**
   * probability bars, one row each: rows [{label, v, cls, text}], o {lw label width, bw bar width, rh row height,
   * max, size, fmt}; the label right-aligned before the bar, the value after it
   */
  P.bars = function (prefix, x, y, rows, o) {
    o = o || {};
    const lw = o.lw || 34, bw = o.bw || 160, rh = o.rh || 22, max = o.max || Math.max(1e-9, ...rows.map((r) => r.v)), fs = o.size || 13, items = [];
    rows.forEach((r, i) => {
      const yy = y + i * rh, w = Math.max(1.5, (bw * Math.max(0, r.v)) / max);
      items.push({ key: prefix + ':l' + i, kind: 'text', x: r2(x + lw), y: r2(yy + rh / 2 + fs * 0.35 - 1), text: r.label, anchor: 'end', size: fs, cls: 'mono p13lab ' + (r.lcls || r.cls || '') });
      items.push({ key: prefix + ':b' + i, kind: 'cell', x: r2(x + lw + 6), y: r2(yy + 3), w: r2(w), h: rh - 6, rx: 2, text: '', cls: 'bar p13bar ' + (r.cls || ''), audit: 'ignore' });
      if (r.text !== '') items.push({ key: prefix + ':v' + i, kind: 'text', x: r2(x + lw + 12 + w), y: r2(yy + rh / 2 + fs * 0.35 - 1), text: r.text === undefined ? (o.fmt ? o.fmt(r.v) : H().fx(r.v, 2)) : r.text, size: o.vsize || fs - 1, cls: 'tick p13num ' + (r.cls || '') });
    });
    return items;
  };
  /** a titled panel behind a group */
  P.panel = (key, x, y, w, h, title, cls) => {
    const it = [{ key, kind: 'region', x, y, w, h, rx: 8, cls: 'p13panel ' + (cls || ''), audit: 'group' }];
    if (title) it.push({ key: key + ':t', kind: 'text', x: x + 12, y: y + 20, text: title, size: 13, cls: 'strong' });
    return it;
  };
  /** lines of text in a box, mono */
  P.textBox = (key, x, y, w, lines, o) => {
    o = o || {};
    const lh = o.lh || 20, h = lines.length * lh + 16, items = [{ key, kind: 'region', x, y, w, h, rx: 6, cls: 'p13textbox ' + (o.cls || ''), audit: 'group' }];
    lines.forEach((l, i) => items.push({ key: key + ':' + i, kind: 'text', x: x + 10, y: y + 8 + lh * (i + 0.72), text: l, size: o.size || 13, cls: 'mono ' + (o.tcls ? o.tcls(i) : '') }));
    return { items, h };
  };
  /** characters as a row of cells: cells(prefix, x, y, chars, {w, h, gap, cls(i), sub(i)}) */
  P.cells = (prefix, x, y, chars, o) => {
    o = o || {};
    const w = o.w || 22, h = o.h || 26, gap = o.gap === undefined ? 2 : o.gap;
    return chars.map((c, i) => {
      const it = { key: prefix + ':' + i, kind: 'cell', x: r2(x + i * (w + gap)), y, w, h, rx: 4, text: P.show(c), size: o.size || 14, cls: 'tok mono ' + (o.cls ? o.cls(i) || '' : '') };
      if (o.sub) { const s = o.sub(i); if (s) it.sub = s; }
      return it;
    });
  };
  P.r2 = r2;
})(globalThis.Nooral = globalThis.Nooral || {});
