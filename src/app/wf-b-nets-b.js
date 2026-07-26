// Networks that read text on the workflow canvas: a recurrent classifier (a simple RNN, an LSTM or a GRU reading the
// tokens one at a time) and a small transformer classifier (attention between all the tokens at once), both on the
// tensor engine. Each splits its texts into words or characters itself, keeps a vocabulary from the training texts,
// and keeps for the pictures one probe text's hidden states (and an LSTM's gates) or attention weights per head.
(function (S) {
  'use strict';
  const WF = S.WF, NETS = WF.nets;
  const { r4 } = WF.mr;
  const T = () => S.T;

  /** words (lower case, letters and digits, apostrophes kept) or characters */
  const tokens = (text, level) => (level === 'char' ? [...String(text).toLowerCase()] : String(text).toLowerCase().match(/[a-z0-9']+|[.,!?]/g) || []);
  /** a vocabulary from texts: ids 0 (padding) and 1 (a word it never saw), then the words by count */
  NETS.vocab = function (texts, level, max) {
    const c = new Map();
    for (const t of texts) for (const w of tokens(t, level)) c.set(w, (c.get(w) || 0) + 1);
    const words = [...c.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, max || 2000).map((e) => e[0]);
    const id = new Map(words.map((w, i) => [w, i + 2]));
    return { words: ['·', '?'].concat(words), level, encode: (t, len) => { const ids = tokens(t, level).slice(0, len).map((w) => id.get(w) || 1); return { ids: new Array(len - ids.length).fill(0).concat(ids), toks: tokens(t, level).slice(0, len) }; } };
  };
  const textParams = (extra) => [
    { id: 'level', label: 'Tokens', kind: 'choice', def: 'word', options: [['word', 'words'], ['char', 'characters']] },
    { id: 'maxLen', label: 'Longest text read (tokens)', kind: 'int', def: 24, min: 4, max: 120 },
    { id: 'embed', label: 'Embedding size', kind: 'int', def: 16, min: 2, max: 128 },
  ].concat(extra, [{ id: 'weightDecay', label: 'Weight decay', kind: 'number', def: 0, min: 0, max: 0.1, step: 0.0001 }]);

  WF.model('rnn', {
    label: 'Recurrent network (RNN, LSTM, GRU)', task: 'classification', order: 64, icon: 'refresh', iterative: true, takes: ['text'],
    blurb: 'Reads a text one token at a time, carrying a hidden state forward; the last state decides. An LSTM or GRU adds gates that choose what to keep, so it remembers further back.',
    params: textParams([
      { id: 'cell', label: 'Cell', kind: 'choice', def: 'LSTM', options: [['RNN', 'simple RNN'], ['LSTM', 'LSTM'], ['GRU', 'GRU']] },
      { id: 'hiddenSize', label: 'Hidden size', kind: 'int', def: 16, min: 2, max: 128 },
    ]),
  });
  WF.model('transformer', {
    label: 'Transformer classifier', task: 'classification', order: 66, icon: 'layers', iterative: true, takes: ['text'],
    blurb: 'Every token looks at every other through attention, in several heads at once; positions are learned; the tokens are averaged and a dense layer decides.',
    params: textParams([
      { id: 'heads', label: 'Heads', kind: 'int', def: 2, min: 1, max: 8 },
      { id: 'blocks', label: 'Blocks', kind: 'int', def: 1, min: 1, max: 4 },
    ]),
  });

  const makeSeqNet = (kind, p, V, K, g) => {
    const Tn = T(), nn = Tn.nn, Net = S.DL.makeNet(), E = Math.max(2, +p.embed || 16);
    if (kind === 'rnn') {
      const H = Math.max(2, +p.hiddenSize || 16), cell = nn[p.cell] ? p.cell : 'LSTM';
      return new (class SeqRNN extends Net {
        constructor() { super(); this.emb = new nn.Embedding(V, E, { rng: g, std: 0.3 }); this.rnn = new nn[cell](E, H, { rng: g }); this.out = new nn.Dense(H, K, { rng: g }); this.cell = cell; }
        forward(ids) { this.begin(); const e = this.tap('emb', this.emb.forward(ids)); const hs = this.tap('hs', this.rnn.forward(e)); return this.tap('logits', this.out.forward(hs.h)); }
      })();
    }
    const heads = Math.max(1, +p.heads || 2), d = Math.ceil(E / heads) * heads, L = Math.max(4, +p.maxLen || 24);
    return new (class SeqTransformer extends Net {
      constructor() {
        super();
        this.emb = new nn.Embedding(V, d, { rng: g, std: 0.3 }); this.pos = new nn.PositionalEmbedding(L, d, { rng: g });
        this.blocks = Array.from({ length: Math.max(1, +p.blocks || 1) }, () => new nn.TransformerBlock(d, heads, { rng: g, ff: 2 * d }));
        this.ln = new nn.LayerNorm(d); this.out = new nn.Dense(d, K, { rng: g }); this.d = d; this.heads = heads;
      }
      forward(ids) {
        this.begin();
        const B = ids.length, n = ids[0].length;
        const keep = ids.map((r) => r.map((v) => (v ? 1 : 0)));
        // a text of nothing but padding still attends to itself, so nothing divides by zero
        keep.forEach((r) => { if (!r.some(Boolean)) r[r.length - 1] = 1; });
        const mask = Tn.tensor(keep.map((r) => [[r]]));
        let x = this.pos.forward(this.emb.forward(ids));
        this.blocks.forEach((b, i) => { x = this.tap('block' + i, b.forward(x, { mask })); });
        x = this.ln.forward(x);
        const m3 = Tn.tensor(keep.map((r) => r.map((v) => [v]))), cnt = Tn.tensor(keep.map((r) => [r.reduce((a, b) => a + b, 0)]));
        const pooled = Tn.div(Tn.sum(Tn.mul(x, m3), 1), cnt);
        return this.tap('logits', this.out.forward(pooled));
      }
    })();
  };
  const seqTrainer = (kind) => function (ctx, spec, data) {
    if (data.kind !== 'text') throw new Error(spec.label + ' reads texts');
    const p = spec.params, L = Math.max(4, +p.maxLen || 24), g = T().rng(ctx.seed || 1);
    const voc = NETS.vocab(data.texts, p.level, 2000), K = data.classes.length;
    const enc = data.texts.map((t) => voc.encode(t, L).ids);
    const net = makeSeqNet(kind, p, voc.words.length, K, g);
    const batchOf = (list) => ({ x: list.map((i) => enc[i]), y: list.map((i) => data.y[i]) });
    const forward = (texts) => net.predict(texts.map((t) => (Array.isArray(t) ? t : voc.encode(t, L).ids)));
    return NETS.trainer(ctx, spec, data, { net, batchOf, forward, clip: 5, internals: (m) => { m.readsText = true; return NETS.seqInside(m, net, data, voc, L, kind); } });
  };
  // a model that reads texts is handed the texts of a set, not its (empty) rows of numbers
  if (WF.predict && !WF.predict.__texts) {
    const orig = WF.predict;
    WF.predict = (model, data) => (model && model.readsText && data && data.kind === 'text' ? orig(model, Object.assign({}, data, { X: data.texts })) : orig(model, data));
    WF.predict.__texts = true;
  }
  WF.trainers.rnn = seqTrainer('rnn');
  WF.trainers.transformer = seqTrainer('transformer');
  /** one probe text through the net: its tokens and, per step, the hidden state (and gates), or attention per head */
  NETS.seqInside = function (m, net, data, voc, L, kind) {
    const probe = 0, e = voc.encode(data.texts[probe], L), lg = net.predict([e.ids]), first = e.ids.findIndex((v) => v !== 0);
    const lv = Array.from(lg.values()), mx = Math.max(...lv), ex = lv.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
    const out = { kind, probe, text: data.texts[probe], probeY: data.y[probe], toks: e.ids.slice(first < 0 ? L : first).map((id) => voc.words[id]), vocab: voc.words.length, level: voc.level, maxLen: L, probs: ex.map((v) => r4(v / s)), nParams: net.numParams() };
    const from = first < 0 ? L : first;
    if (kind === 'rnn') {
      const st = net.rnn.steps.slice(from), H = net.rnn.H;
      out.cell = net.cell; out.H = H;
      out.h = st.map((x) => Array.from(x.h.values(), r4).slice(0, H));
      if (net.cell === 'LSTM') { out.gates = { i: st.map((x) => Array.from(x.i.values(), r4)), f: st.map((x) => Array.from(x.f.values(), r4)), o: st.map((x) => Array.from(x.o.values(), r4)) }; }
      if (net.cell === 'GRU') out.gates = { r: st.map((x) => Array.from(x.r.values(), r4)), z: st.map((x) => Array.from(x.z.values(), r4)) };
    } else {
      const w = net.blocks[0].attn.weights, [, Hh, n] = w.shape, v = w.values();
      out.heads = Array.from({ length: Hh }, (_, h) => Array.from({ length: n - from }, (_, i) => Array.from({ length: n - from }, (_, j) => r4(v[(h * n + i + from) * n + j + from]))));
      out.d = net.d; out.blocks = net.blocks.length;
    }
    return out;
  };
  for (const id of ['rnn', 'transformer']) WF.families[id].view = () => ({});
})(globalThis.Nooral = globalThis.Nooral || {});
