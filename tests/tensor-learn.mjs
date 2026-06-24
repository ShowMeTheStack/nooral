// Learning with the tensor engine: an MLP on XOR and moons, a small CNN on a bar-pattern task, an LSTM that counts,
// and a 2-layer transformer that reverses short sequences. Each must reach its score within a time budget.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T, nn = T.nn;
const t = harness('tensor-learn');
const timed = (name, budgetMs, fn) => {
  const t0 = performance.now(), r = fn(), ms = performance.now() - t0;
  console.log(`  ${name.padEnd(34)} ${r.label}  in ${(ms / 1000).toFixed(2)} s`);
  t.ok(ms < budgetMs, name + ' took ' + ms.toFixed(0) + ' ms (budget ' + budgetMs + ')');
  return r;
};
function fit(model, opt, batches, steps, lossFn) {
  let last = 0;
  for (let i = 0; i < steps; i++) {
    const [x, y] = batches(i);
    opt.zeroGrad();
    const loss = lossFn(model.forward(x), y);
    loss.backward();
    opt.step();
    last = loss.item();
  }
  return last;
}

// ---- XOR ----
timed('MLP on XOR', 3000, () => {
  const X = T.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]), y = [0, 1, 1, 0];
  const m = new nn.MLP([2, 8, 2], { rng: 2, act: 'tanh' }), opt = new T.optim.Adam(m.parameters(), { lr: 0.05 });
  const loss = fit(m, opt, () => [X, y], 300, (z, yy) => T.crossEntropy(z, yy));
  const acc = T.accuracy(m.forward(X), y);
  t.ok(acc === 1 && loss < 0.01, 'XOR: accuracy ' + acc + ', loss ' + loss.toFixed(4));
  return { label: 'accuracy ' + acc + ', loss ' + loss.toFixed(4) };
});

// ---- moons ----
function moons(n, noise, rng) {
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, a = Math.PI * rng.next();
    const px = c ? 1 - Math.cos(a) : Math.cos(a), py = c ? 0.5 - Math.sin(a) : Math.sin(a);
    X.push([px + rng.normal(0, noise), py + rng.normal(0, noise)]); y.push(c);
  }
  return { X: T.tensor(X), y };
}
timed('MLP on moons', 6000, () => {
  const g = T.rng(11), tr = moons(400, 0.12, g), te = moons(400, 0.12, g);
  const m = new nn.MLP([2, 16, 16, 2], { rng: 3 }), opt = new T.optim.Adam(m.parameters(), { lr: 0.02 });
  const order = () => g.permutation(400);
  let perm = order();
  fit(m, opt, (i) => { const k = i % 10; if (!k) perm = order(); const idx = perm.slice(k * 40, k * 40 + 40); return [T.take(tr.X, idx), idx.map((j) => tr.y[j])]; }, 600, (z, yy) => T.crossEntropy(z, yy));
  const acc = T.accuracy(m.forward(te.X), te.y);
  t.ok(acc >= 0.97, 'moons: test accuracy ' + acc);
  return { label: 'test accuracy ' + acc.toFixed(3) };
});

// ---- a CNN: is the bar in an 8×8 image horizontal, vertical or diagonal? ----
function bars(n, rng) {
  const x = T.zeros([n, 1, 8, 8]), y = [];
  for (let i = 0; i < n; i++) {
    const c = rng.int(3), p = 1 + rng.int(6), len = 4 + rng.int(3), s = rng.int(8 - len + 1), base = i * 64;
    for (let k = 0; k < 64; k++) x.data[base + k] = rng.normal(0, 0.25);
    for (let k = 0; k < len; k++) {
      const [r, cc] = c === 0 ? [p, s + k] : c === 1 ? [s + k, p] : [s + k, Math.min(7, (s + k + p) % 8)];
      x.data[base + r * 8 + cc] += 1;
    }
    y.push(c);
  }
  return { x, y };
}
timed('CNN on bar patterns', 15000, () => {
  const g = T.rng(5), tr = bars(512, g), te = bars(300, g);
  const m = new nn.Sequential(
    new nn.Conv2d(1, 6, 3, { rng: 1, padding: 1 }), new nn.Activation('relu'), new nn.MaxPool(2),
    new nn.Conv2d(6, 8, 3, { rng: 2, padding: 1 }), new nn.BatchNorm(8), new nn.Activation('relu'), new nn.MaxPool(2),
    new nn.Flatten(), new nn.Dense(32, 3, { rng: 3 }));
  const opt = new T.optim.Adam(m.parameters(), { lr: 0.01 });
  fit(m, opt, (i) => { const idx = Array.from({ length: 32 }, (_, j) => (i * 32 + j) % 512); return [T.take(tr.x, idx), idx.map((j) => tr.y[j])]; }, 240, (z, yy) => T.crossEntropy(z, yy));
  m.eval();
  const acc = T.accuracy(m.forward(te.x), te.y);
  t.ok(acc >= 0.95, 'CNN: test accuracy ' + acc);
  return { label: 'test accuracy ' + acc.toFixed(3) };
});

// ---- an LSTM that keeps a running count of the ones it has seen ----
function counting(n, len, rng) {
  const x = T.zeros([n, len, 1]), y = [];
  for (let i = 0; i < n; i++) {
    let c = 0; const row = [];
    for (let k = 0; k < len; k++) { const b = rng.next() < 0.4 ? 1 : 0; c += b; x.data[i * len + k] = b; row.push(c); }
    y.push(row);
  }
  return { x, y };
}
timed('LSTM counting ones', 20000, () => {
  const g = T.rng(8), len = 8, tr = counting(512, len, g), te = counting(200, len, g);
  const lstm = new nn.LSTM(1, 24, { rng: 4 }), head = new nn.Dense(24, len + 1, { rng: 5 });
  const model = { forward: (x) => head.forward(lstm.forward(x)) };
  const opt = new T.optim.Adam(lstm.parameters().concat(head.parameters()), { lr: 0.01 });
  fit(model, opt, (i) => { const idx = Array.from({ length: 32 }, (_, j) => (i * 32 + j) % 512); return [T.take(tr.x, idx), idx.map((j) => tr.y[j])]; }, 400, (z, yy) => T.crossEntropy(z, yy));
  const pred = T.argmax(model.forward(te.x), -1).values(), flat = te.y.flat();
  const acc = flat.filter((v, i) => v === pred[i]).length / flat.length;
  const final = te.y.filter((row, i) => row[len - 1] === pred[i * len + len - 1]).length / te.y.length;
  t.ok(acc >= 0.97 && final >= 0.95, 'LSTM counting: per-step accuracy ' + acc + ', final count ' + final);
  return { label: 'per-step accuracy ' + acc.toFixed(3) + ', final count ' + final.toFixed(3) };
});

// ---- a 2-layer causal transformer that reverses 4 digits: x1 x2 x3 x4 | x4 x3 x2 x1 ----
timed('Transformer reversing sequences', 60000, () => {
  const V = 6, SEP = 5, L = 4, d = 32, g = T.rng(21);
  const emb = new nn.Embedding(V, d, { rng: 1, std: 0.3 }), pos = new nn.PositionalEmbedding(2 * L + 1, d, { rng: 2 });
  const blocks = [new nn.TransformerBlock(d, 2, { rng: 3, ff: 64, causal: true }), new nn.TransformerBlock(d, 2, { rng: 4, ff: 64, causal: true })];
  const lnf = new nn.LayerNorm(d), head = new nn.Dense(d, V, { rng: 5 });
  const forward = (ids) => { let h = pos.forward(emb.forward(ids)); for (const b of blocks) h = b.forward(h); return head.forward(lnf.forward(h)); };
  const params = [emb, pos, ...blocks, lnf, head].flatMap((m) => m.parameters());
  const steps = 500, opt = new T.optim.AdamW(params, { lr: 3e-3, weightDecay: 0.01, schedule: T.sched.warmup({ steps: 30, then: T.sched.cosine({ lr: 3e-3, min: 3e-4, total: steps - 30 }) }) });
  const sample = () => { const x = Array.from({ length: L }, () => g.int(5)); return x.concat([SEP], x.slice().reverse()); };
  for (let i = 0; i < steps; i++) {
    const seqs = Array.from({ length: 32 }, sample);
    const inp = T.tensor(seqs.map((s) => s.slice(0, -1))), tgt = seqs.map((s) => s.slice(1).map((v, k) => (k >= L ? v : -1)));
    opt.zeroGrad();
    const loss = T.crossEntropy(forward(inp), tgt, { ignoreIndex: -1 });
    loss.backward();
    T.clipGradNorm(params, 1);
    opt.step();
  }
  // greedy decoding of the four outputs, one token at a time
  let exact = 0;
  const tests = 200, gt = T.rng(99);
  T.noGrad(() => {
    for (let n = 0; n < tests; n++) {
      const x = Array.from({ length: L }, () => gt.int(5)), seq = x.concat([SEP]);
      for (let k = 0; k < L; k++) { const z = forward(T.tensor([seq])); seq.push(T.argmax(T.slice(z, 0, -1), -1).item()); }
      if (seq.slice(L + 1).join() === x.slice().reverse().join()) exact++;
    }
  });
  const w = blocks[1].attn.weights;
  t.ok(w.get(0, 0, 0, 1) === 0, 'the causal mask keeps attention off later positions');
  t.ok(exact / tests >= 0.95, 'transformer reversal: exact ' + exact + ' of ' + tests);
  return { label: 'exact reversals ' + exact + '/' + tests };
});
t.done();
