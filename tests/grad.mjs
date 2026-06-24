// Every differentiable op's gradient against central finite differences, in float64, on random shapes and seeds.
// The loss is Σ out ⊙ R for a fixed random R, so every output element contributes with its own weight.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T;
const t = harness('grad');
let checks = 0, worst = 0;

/** shapes: arrays (random normal inputs) or {shape, lo, hi} (uniform) or a ready tensor; f returns a tensor */
function gradcheck(name, f, inputs, seed = 1, opts = {}) {
  T.float64(() => {
    const g = T.rng(seed * 7919 + 13);
    const xs = inputs.map((s) => {
      if (s instanceof T.Tensor) return s.isParam ? s : T.param(T.tensor(s));
      if (Array.isArray(s)) return T.param(T.normal(s, 0, 1, g));
      return T.param(T.uniform(s.shape, s.lo, s.hi, g));
    });
    let out;
    try { out = f(...xs); } catch (e) { t.ok(false, name + ' (seed ' + seed + ') threw ' + e.message); return; }
    const R = T.normal(out.shape, 0, 1, g);
    const lossOf = () => { const o = T.noGrad(() => f(...xs)).values(), r = R.values(); let s = 0; for (let i = 0; i < o.length; i++) s += o[i] * r[i]; return s; };
    T.sum(T.mul(out, R)).backward();
    const eps = opts.eps || 1e-6, tol = opts.tol || 1e-6;
    xs.forEach((x, xi) => {
      if (opts.skip && opts.skip.includes(xi)) return;
      const n = x.size, picks = n <= 48 ? [...Array(n).keys()] : Array.from({ length: 48 }, () => g.int(n));
      let bad = null;
      for (const i of picks) {
        const old = x.data[i];
        x.data[i] = old + eps; const lp = lossOf();
        x.data[i] = old - eps; const lm = lossOf();
        x.data[i] = old;
        const num = (lp - lm) / (2 * eps), ana = x.grad ? x.grad.data[i] : 0;
        const err = Math.abs(num - ana) / Math.max(1, Math.abs(num), Math.abs(ana));
        if (err > worst) worst = err;
        if (!(err <= tol) && !bad) bad = `input ${xi} [${x.shape}] element ${i}: analytic ${ana}, numeric ${num}`;
      }
      checks++;
      t.ok(!bad, name + ' (seed ' + seed + '): ' + bad);
    });
  });
}
const pos = (shape) => ({ shape, lo: 0.3, hi: 2.5 });
const seeds = [1, 2, 3];
const dims = (g, k, lo = 1, hi = 5) => Array.from({ length: k }, () => lo + g.int(hi - lo + 1));

for (const sd of seeds) {
  const g = T.rng(sd), [a, b, c] = dims(g, 3, 2, 5);
  // elementwise, with broadcasting
  gradcheck('add', T.add, [[a, b, c], [b, 1]], sd);
  gradcheck('sub', T.sub, [[a, 1, c], [b, c]], sd);
  gradcheck('mul', T.mul, [[a, b], [a, b]], sd);
  gradcheck('mul broadcast', T.mul, [[c], [a, b, c]], sd);
  gradcheck('div', T.div, [[a, b], pos([b])], sd);
  gradcheck('pow', T.pow, [pos([a, b]), [a, b]], sd);
  gradcheck('pow scalar exponent', (x) => T.pow(x, 3), [[a, b]], sd);
  gradcheck('maximum', T.maximum, [[a, b], [a, b]], sd);
  gradcheck('minimum', T.minimum, [[a, b], [b]], sd);
  for (const op of ['neg', 'exp', 'abs', 'square', 'sin', 'cos', 'relu', 'sigmoid', 'tanh', 'gelu', 'softplus', 'elu']) gradcheck(op, T[op], [[a, b, c]], sd);
  gradcheck('log', T.log, [pos([a, b])], sd);
  gradcheck('sqrt', T.sqrt, [pos([a, b])], sd);
  gradcheck('leakyRelu', (x) => T.leakyRelu(x, 0.1), [[a, b]], sd);
  gradcheck('clamp', (x) => T.clamp(x, -0.5, 0.5), [[a, b]], sd);
  gradcheck('where', (x, y) => T.where(T.gt(T.arange(b), 1), x, y), [[a, b], [b]], sd);
  gradcheck('maskedFill', (x) => T.maskedFill(x, T.eq(T.arange(b), 0), -3), [[a, b]], sd);
  gradcheck('broadcastTo', (x) => T.broadcastTo(x, [a, b, c]), [[b, 1]], sd);
  // reductions and softmax
  gradcheck('sum all', (x) => T.sum(x), [[a, b, c]], sd);
  gradcheck('sum axis 1', (x) => T.sum(x, 1), [[a, b, c]], sd);
  gradcheck('sum axes 0,2 keepdims', (x) => T.sum(x, [0, 2], true), [[a, b, c]], sd);
  gradcheck('mean axis -1', (x) => T.mean(x, -1), [[a, b, c]], sd);
  gradcheck('max axis 1', (x) => T.max(x, 1), [[a, b, c]], sd);
  gradcheck('min all', (x) => T.min(x), [[a, b]], sd);
  gradcheck('variance', (x) => T.variance(x, 1), [[a, b + 1]], sd);
  gradcheck('logsumexp', (x) => T.logsumexp(x, -1), [[a, b]], sd);
  gradcheck('softmax last', (x) => T.softmax(x, -1), [[a, b, c]], sd);
  gradcheck('softmax axis 1', (x) => T.softmax(x, 1), [[a, b, c]], sd);
  gradcheck('logSoftmax', (x) => T.logSoftmax(x, -1), [[a, b]], sd);
  gradcheck('logSoftmax axis 0', (x) => T.logSoftmax(x, 0), [[a, b]], sd);
  // products and shapes
  gradcheck('matmul', T.matmul, [[a, b], [b, c]], sd);
  gradcheck('matmul batched', T.matmul, [[2, a, b], [2, b, c]], sd);
  gradcheck('matmul broadcast batch', T.matmul, [[3, 1, a, b], [2, b, c]], sd);
  gradcheck('matmul ND @ 2D', T.matmul, [[2, a, b], [b, c]], sd);
  gradcheck('matmul vector', T.matmul, [[b], [b, c]], sd);
  gradcheck('matmul matrix-vector', T.matmul, [[a, b], [b]], sd);
  gradcheck('matmul of transposed views', (x, y) => T.matmul(T.transpose(x), T.transpose(y)), [[b, a], [c, b]], sd);
  gradcheck('transpose', (x) => T.transpose(x, 0, 2), [[a, b, c]], sd);
  gradcheck('permute', (x) => T.permute(x, [2, 0, 1]), [[a, b, c]], sd);
  gradcheck('reshape of a view', (x) => T.reshape(T.transpose(x), [-1]), [[a, b]], sd);
  gradcheck('flatten', (x) => T.flatten(x), [[a, b, c]], sd);
  gradcheck('slice', (x) => T.slice(x, [1, null], [null, null, 2], -1), [[a + 1, b + 2, c]], sd);
  gradcheck('slice reversed', (x) => T.slice(x, null, [null, null, -1]), [[a, b]], sd);
  gradcheck('squeeze/unsqueeze', (x) => T.unsqueeze(T.squeeze(x, 1), 0), [[a, 1, c]], sd);
  gradcheck('expand', (x) => T.mul(T.expand(x, [a, b, c]), T.arange(c)), [[1, b, 1]], sd);
  gradcheck('concat', (x, y) => T.concat([x, y], 1), [[a, b, c], [a, 2, c]], sd);
  gradcheck('stack', (x, y) => T.stack([x, y], 2), [[a, b], [a, b]], sd);
  gradcheck('split', (x) => { const [p, q] = T.split(x, [1, b], 1); return T.mul(p, T.sum(q, 1, true)); }, [[a, b + 1]], sd);
  gradcheck('take', (x) => T.take(x, [0, b - 1, 0, 1 % b], 1), [[a, b, c]], sd);
  gradcheck('gather', (x) => T.gather(x, 1, T.tensor(Array.from({ length: a }, (_, i) => [i % b, (i + 1) % b]))), [[a, b]], sd);
  gradcheck('embedding', (w) => T.embedding(w, T.tensor([[1, 0, 2], [2, 2, 1]])), [[3, c]], sd);
  // normalisation and dropout
  gradcheck('layerNorm', (x, w, bb) => T.layerNorm(x, w, bb), [[a, b, c + 2], [c + 2], [c + 2]], sd);
  gradcheck('layerNorm plain', (x) => T.layerNorm(x), [[a, c + 2]], sd);
  gradcheck('batchNorm 2d train', (x, w, bb) => T.batchNorm(x, w, bb), [[a + 2, c], [c], [c]], sd);
  gradcheck('batchNorm 4d train', (x, w, bb) => T.batchNorm(x, w, bb), [[2, c, 3, 2], [c], [c]], sd);
  gradcheck('batchNorm eval', (x, w, bb) => T.batchNorm(x, w, bb, { training: false, running: { mean: T.normal([c], 0, 1, sd), var: T.uniform([c], 0.5, 2, sd) } }), [[a, c], [c], [c]], sd);
  gradcheck('dropout', (x) => T.dropout(x, 0.4, sd, true), [[a, b, c]], sd);
  // convolution and pooling
  gradcheck('conv2d', (x, w, bb) => T.conv2d(x, w, bb), [[2, 2, 5, 4], [3, 2, 3, 2], [3]], sd);
  gradcheck('conv2d stride 2 padding 1', (x, w) => T.conv2d(x, w, null, { stride: 2, padding: 1 }), [[1, 3, 6, 5], [2, 3, 3, 3]], sd);
  gradcheck('conv2d dilation 2', (x, w) => T.conv2d(x, w, null, { dilation: 2, padding: [2, 1] }), [[2, 1, 6, 6], [2, 1, 3, 3]], sd);
  gradcheck('conv2d groups 2', (x, w, bb) => T.conv2d(x, w, bb, { groups: 2, padding: 1 }), [[2, 4, 4, 4], [4, 2, 3, 3], [4]], sd);
  gradcheck('conv2d depthwise', (x, w) => T.conv2d(x, w, null, { groups: 3, stride: [1, 2] }), [[1, 3, 5, 5], [3, 1, 2, 2]], sd);
  gradcheck('im2col', (x) => T.im2col(x, 2, 3, { padding: 1, stride: 2 }), [[2, 2, 4, 5]], sd);
  gradcheck('maxPool2d', (x) => T.maxPool2d(x, 2), [[2, 2, 4, 6]], sd);
  gradcheck('maxPool2d overlapping, padded', (x) => T.maxPool2d(x, 3, 2, 1), [[1, 2, 5, 5]], sd);
  gradcheck('avgPool2d', (x) => T.avgPool2d(x, 2, 1, 1), [[2, 1, 4, 3]], sd);
  // attention
  gradcheck('attention', (q, k, v) => T.attention(q, k, v), [[2, a, 4], [2, b + 1, 4], [2, b + 1, 3]], sd);
  gradcheck('attention causal', (q, k, v) => T.attention(q, k, v, { causal: true }), [[2, 2, a + 1, 4], [2, 2, a + 1, 4], [2, 2, a + 1, 4]], sd);
  gradcheck('attention padding mask', (q, k, v) => T.attention(q, k, v, { mask: T.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 0]]) }), [[3, 4], [3, 4], [3, 2]], sd);
  // losses
  const ids = Array.from({ length: a + 1 }, (_, i) => (i * 7 + sd) % c);
  gradcheck('mseLoss', (p, y) => T.mseLoss(p, y), [[a, b], [a, b]], sd);
  gradcheck('l1Loss', (p) => T.l1Loss(p, T.zeros([a, b])), [[a, b]], sd);
  gradcheck('huberLoss', (p) => T.huberLoss(p, T.zeros([a, b]), 0.7), [[a, b]], sd);
  gradcheck('crossEntropy ids', (z) => T.crossEntropy(z, ids), [[a + 1, c]], sd);
  gradcheck('crossEntropy label smoothing', (z) => T.crossEntropy(z, ids, { labelSmoothing: 0.1 }), [[a + 1, c]], sd);
  gradcheck('crossEntropy ignoreIndex', (z) => T.crossEntropy(z, ids.map((x, i) => (i === 0 ? -1 : x)), { ignoreIndex: -1 }), [[a + 1, c]], sd);
  gradcheck('crossEntropy probabilities', (z, p) => T.crossEntropy(z, T.softmax(p)), [[a, c], [a, c]], sd, { skip: [1] });
  gradcheck('crossEntropy on sequences', (z) => T.crossEntropy(z, [[0, 1, 2 % c], [1 % c, 0, 0]]), [[2, 3, c]], sd);
  gradcheck('nllLoss', (z) => T.nllLoss(T.logSoftmax(z), ids), [[a + 1, c]], sd);
  gradcheck('bceLoss', (z) => T.bceLoss(T.sigmoid(z), T.tensor(Array.from({ length: a * b }, (_, i) => (i % 3 ? 1 : 0)), [a, b])), [[a, b]], sd);
  gradcheck('bceWithLogits', (z) => T.bceWithLogits(z, T.tensor(Array.from({ length: a * b }, (_, i) => (i % 2)), [a, b])), [[a, b]], sd);
  gradcheck('hingeLoss', (s) => T.hingeLoss(s, T.tensor(Array.from({ length: a }, (_, i) => (i % 2 ? 1 : -1)))), [[a]], sd);
  gradcheck('multiHinge', (s) => T.multiHinge(s, ids), [[a + 1, c]], sd);
  gradcheck('klDiv', (z) => T.klDiv(T.logSoftmax(z), T.softmax(T.normal([a, c], 0, 1, sd))), [[a, c]], sd);
}
// whole layers, with respect to their input and their parameters
for (const sd of seeds) {
  T.float64(() => {
    const mods = [
      ['Dense', new T.nn.Dense(3, 4, { rng: sd }), [2, 3]],
      ['Conv2d+BatchNorm', new T.nn.Sequential(new T.nn.Conv2d(2, 3, 3, { rng: sd, padding: 1 }), new T.nn.BatchNorm(3), new T.nn.Activation('tanh')), [2, 2, 4, 4]],
      ['RNN', new T.nn.RNN(3, 4, { rng: sd }), [2, 4, 3]],
      ['LSTM', new T.nn.LSTM(3, 4, { rng: sd }), [2, 4, 3]],
      ['GRU', new T.nn.GRU(3, 4, { rng: sd }), [2, 4, 3]],
      ['MultiHeadAttention causal', new T.nn.MultiHeadAttention(6, 2, { rng: sd, causal: true }), [2, 4, 6]],
      ['TransformerBlock', new T.nn.TransformerBlock(4, 2, { rng: sd, ff: 8, causal: true }), [2, 3, 4]],
    ];
    for (const [name, m, shape] of mods) {
      gradcheck(name, (x) => m.forward(x), [shape].concat(m.parameters()), sd, { tol: 1e-5 });
    }
  });
}
// no_grad and detach cut the graph
{
  const x = T.param(T.tensor([1, 2, 3]));
  const y = T.noGrad(() => T.mul(x, x));
  t.ok(!y.requiresGrad && !y.ctx, 'noGrad records nothing');
  const z = T.sum(T.mul(T.mul(x, x).detach(), x));
  z.backward();
  t.eq(Array.from(x.grad.values()), [1, 4, 9], 'detach: d/dx Σ stop(x²)·x = x²');
  const w = T.param(T.tensor([2])), u = T.mul(w, w), v = T.add(u, u);
  T.sum(v).backward();
  t.eq(w.grad.values()[0], 8, 'a node used twice accumulates its gradient (d/dw 2w² = 4w)');
  T.sum(v).backward();
  t.eq(w.grad.values()[0], 16, 'leaf gradients accumulate across backward passes');
}
console.log(`  ${checks} gradient checks, worst relative error ${worst.toExponential(2)}`);
t.done();
