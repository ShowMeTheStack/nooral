// Recurrent sequence models (plain RNN, LSTM, GRU) and their tasks.
//   DL.build('rnn', {cell: 'rnn' | 'lstm' | 'gru', vocab (token ids in; 0 for numbers in), inputs, embed, hidden,
//       outputs, mode: 'each' (an output per step) | 'last' (one per sequence), lm (a next-token model), seed})
//   named outputs embed, hidden, logits; net.extras() adds the gates per step [B, T, H] (gate.i gate.f gate.g gate.o
//   cell for an LSTM; gate.r gate.z gate.n for a GRU) and the embedding matrix
//   net.generate(prompt ids, n, sampling, rng) -> {ids, steps}   for a next-character model
//   tasks: 'chars' {len, stride} next character on the prose corpus; 'reviews' {set: 'reviews' | 'opinions'} sentiment;
//          'counting' {len} the running count of ones in a 0/1 sequence
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});

  DL.register('rnn', (c) => {
    const T = S.T, nn = T.nn, Net = DL.makeNet(), g = T.rng(c.seed || 1);
    const cell = (c.cell || 'lstm').toLowerCase(), Cell = { rnn: nn.RNN, lstm: nn.LSTM, gru: nn.GRU }[cell];
    if (!Cell) throw new Error('unknown recurrent cell ' + c.cell);
    const H = c.hidden || 32, E = c.vocab ? c.embed || 16 : c.inputs || 1;
    class Recurrent extends Net {
      constructor() {
        super();
        if (c.vocab) this.emb = new nn.Embedding(c.vocab, E, { rng: g, std: 0.3 });
        this.cell = new Cell(E, H, { rng: g });
        this.head = new nn.Dense(H, c.outputs || c.vocab || 2, { rng: g });
        this.isLM = !!c.lm;
        this.cellKind = cell;
        this.tokenizer = c.tokenizer ? DL.tokenizer(c.tokenizer) : null;
        this.snapshotNames = ['hidden', 'logits'].concat(cell === 'lstm' ? ['gate.i', 'gate.f', 'gate.o', 'cell'] : cell === 'gru' ? ['gate.r', 'gate.z'] : []);
      }
      /**
       * The recurrence, step by step with the cell's own weights and gate order (as in S.T.nn). Each step's input
       * is made on its own (an embedding lookup, or a slice of the small input), so the backward pass never has
       * to scatter into the whole sequence once per step.
       */
      forward(x, state) {
        this.begin();
        const [B, n] = x.shape, cl = this.cell, H = cl.H, part = (t, j) => T.slice(t, null, [j * H, (j + 1) * H]);
        let xs;
        if (this.emb) {
          const iv = x.values();
          xs = Array.from({ length: n }, (_, k) => T.embedding(this.emb.W, Array.from({ length: B }, (_, b) => iv[b * n + k])));
          this.tap('embed', T.stack(xs, 1));
        } else xs = Array.from({ length: n }, (_, k) => T.slice(x, null, k, null));
        let h = state ? (cell === 'lstm' ? state.h : state) : T.zeros([B, H]), cc = cell === 'lstm' ? (state ? state.c : T.zeros([B, H])) : null;
        const hs = [];
        cl.steps = [];
        for (let k = 0; k < n; k++) {
          const xw = T.add(T.matmul(xs[k], cl.Wx), cl.b);
          if (cell === 'rnn') {
            h = T[cl.act](T.add(xw, T.matmul(h, cl.Wh)));
            cl.steps.push({ h });
          } else if (cell === 'lstm') {
            const z = T.add(xw, T.matmul(h, cl.Wh));
            const i = T.sigmoid(part(z, 0)), f = T.sigmoid(part(z, 1)), gg = T.tanh(part(z, 2)), o = T.sigmoid(part(z, 3));
            cc = T.add(T.mul(f, cc), T.mul(i, gg));
            h = T.mul(o, T.tanh(cc));
            cl.steps.push({ i, f, g: gg, o, c: cc, h });
          } else {
            const hh = T.add(T.matmul(h, cl.Wh), cl.bh);
            const r = T.sigmoid(T.add(part(xw, 0), part(hh, 0))), u = T.sigmoid(T.add(part(xw, 1), part(hh, 1)));
            const cand = T.tanh(T.add(part(xw, 2), T.mul(r, part(hh, 2))));
            h = T.add(T.mul(T.sub(1, u), cand), T.mul(u, h));
            cl.steps.push({ r, z: u, n: cand, h });
          }
          hs.push(h);
        }
        this.state = cell === 'lstm' ? { h, c: cc } : h;
        if (c.mode === 'last') { this.tap('hidden', T.stack(hs, 1)); return this.tap('logits', this.head.forward(h)); }
        return this.tap('logits', this.head.forward(this.tap('hidden', T.stack(hs, 1))));
      }
      /** the gate values of the last forward pass, stacked over time: [B, T, H] each */
      gates() {
        const st = this.cell.steps || [], o = {};
        if (!st.length) return o;
        const names = cell === 'lstm' ? { i: 'gate.i', f: 'gate.f', g: 'gate.g', o: 'gate.o', c: 'cell' } : cell === 'gru' ? { r: 'gate.r', z: 'gate.z', n: 'gate.n' } : {};
        T.noGrad(() => { for (const k of Object.keys(names)) o[names[k]] = T.stack(st.map((s) => s[k]), 1); });
        return o;
      }
      extras() { const o = this.gates(); if (this.emb) o.embeddings = this.emb.W; return o; }
      /** continue a prompt n tokens, carrying the recurrent state from step to step */
      generate(prompt, n, sampling = {}, rng) {
        rng = T.asRng(rng === undefined ? 1 : rng);
        const ids = prompt.slice(), steps = [], was = this.training;
        this.eval();
        T.noGrad(() => {
          let z = this.forward(T.tensor([prompt]));
          for (let k = 0; k < n; k++) {
            const V = z.shape[2], row = T.slice(z, 0, z.shape[1] - 1, null).values();
            const s = DL.sample(row, sampling, rng);
            ids.push(s.id); steps.push(s);
            z = this.forward(T.tensor([[s.id]]), this.state);
            if (V !== z.shape[2]) throw new Error('vocabulary changed');
          }
        });
        this.setMode(was);
        return { ids, steps, text: this.tokenizer ? this.tokenizer.decode(ids) : null };
      }
      /** continue a text prompt (needs the tokenizer the model was trained with) */
      complete(text, n, sampling, rng) { return this.generate(this.tokenizer.encode(text), n, sampling, rng); }
    }
    return new Recurrent();
  });

  /** next character on the prose corpus: windows of len characters; the last tenth of the text validates */
  DL.tasks.register('chars', (o) => {
    const text = o.text || S.Data.proseText(), tok = DL.charTokenizer(text), ids = tok.encode(text), len = o.len || 32, stride = o.stride || 8;
    const cut = Math.floor(ids.length * 0.9), win = (a, b) => { const r = []; for (let i = a; i + len + 1 <= b; i += stride) r.push(i); return r; };
    const batchOf = (list) => ({ x: S.T.tensor(list.map((i) => ids.slice(i, i + len))), y: list.map((i) => ids.slice(i + 1, i + len + 1)) });
    const val = DL.split(win(cut, ids.length), batchOf);
    return { kind: 'rnn', train: DL.split(win(0, cut), batchOf), val, probe: val.get([0]), tokenizer: tok,
      model: { cell: 'lstm', vocab: tok.size, embed: 24, hidden: 64, lm: true, tokenizer: tok.toJSON() }, fit: { batch: 16, lr: 5e-3, clip: 1 } };
  });

  /** sentiment of the short reviews (set 'reviews', 60 hand-written, or 'opinions', 600 with turns and negations),
   *  words left-padded so the last step sees the whole review */
  DL.tasks.register('reviews', (o) => {
    const d = S.Data.load(o.set || 'reviews', { seed: o.seed || 1 }), tok = DL.wordTokenizer(d.texts), seqs = d.texts.map((t) => tok.encode(t));
    const L = Math.max(...seqs.map((s) => s.length)), nTrain = Math.round(d.n * (o.trainFrac || 0.8));
    const ex = seqs.map((s, i) => ({ ids: new Array(L - s.length).fill(0).concat(s), y: d.y[i], text: d.texts[i] }));
    const batchOf = (list) => ({ x: S.T.tensor(list.map((e) => e.ids)), y: list.map((e) => e.y) });
    const val = DL.split(ex.slice(nTrain), batchOf);
    return { kind: 'rnn', train: DL.split(ex.slice(0, nTrain), batchOf), val, probe: val.get([0]), tokenizer: tok, classes: d.classes,
      model: { cell: 'gru', vocab: tok.size, embed: 16, hidden: 24, outputs: 2, mode: 'last', tokenizer: tok.toJSON() }, fit: { batch: 16, lr: 1e-2, clip: 1 } };
  });

  /** count the ones seen so far: x is 0/1 per step, the target at step t is the count up to t */
  DL.tasks.register('counting', (o) => {
    const len = o.len || 8, g = S.T.rng(o.seed || 1), p = o.p || 0.4;
    const make = (n) => Array.from({ length: n }, () => { let c = 0; const x = [], y = []; for (let k = 0; k < len; k++) { const b = g.next() < p ? 1 : 0; c += b; x.push(b); y.push(c); } return { x, y }; });
    const batchOf = (list) => ({ x: S.T.tensor(list.map((e) => e.x.map((b) => [b]))), y: list.map((e) => e.y) });
    const val = DL.split(make(o.val || 200), batchOf);
    return { kind: 'rnn', train: DL.split(make(o.n || 512), batchOf), val, probe: val.get([0]),
      model: { cell: 'lstm', vocab: 0, inputs: 1, hidden: 24, outputs: len + 1 }, fit: { batch: 32, lr: 1e-2 } };
  });
})(globalThis.Nooral = globalThis.Nooral || {});
