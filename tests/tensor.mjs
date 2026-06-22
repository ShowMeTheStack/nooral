// The tensor engine: hand-computed values, views and broadcasting, seeds, initialisers, modules (save and load),
// optimisers on test functions, schedules, the training runner, traces and the CPU backend's speed.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T;
const t = harness('tensor');
const close = (a, b, tol = 1e-4) => {
  const flat = (v) => (v instanceof T.Tensor ? Array.from(v.values()) : [].concat(v).flat(Infinity)), x = flat(a), y = flat(b);
  return x.length === y.length && x.every((v, i) => Math.abs(v - y[i]) <= tol * Math.max(1, Math.abs(y[i])));
};
const near = (a, b, msg, tol) => t.ok(close(a, b, tol), msg + ': got ' + JSON.stringify(a instanceof T.Tensor ? Array.from(a.values()).map((v) => +v.toFixed(5)) : a) + ', want ' + JSON.stringify(b));

// ---- construction, views, broadcasting ----
{
  const a = T.tensor([[1, 2, 3], [4, 5, 6]]);
  t.eq(a.shape, [2, 3], 'shape of nested arrays'); t.eq(a.strides, [3, 1], 'row-major strides'); t.eq(a.dtype, 'float32', 'float32 by default');
  const at = T.transpose(a);
  t.eq([at.shape, at.strides, at.data === a.data], [[3, 2], [1, 3], true], 'transpose is a view with swapped strides');
  t.eq(at.toArray(), [[1, 4], [2, 5], [3, 6]], 'transpose values');
  t.ok(!at.isContiguous() && T.reshape(at, [6]).data !== a.data, 'reshape of a non-contiguous view copies');
  t.ok(T.reshape(a, [3, 2]).data === a.data, 'reshape of contiguous data is a view');
  t.eq(T.reshape(a, [-1, 2]).shape, [3, 2], 'reshape infers -1');
  const s = T.slice(a, null, [null, null, 2]);
  t.eq([s.toArray(), s.data === a.data], [[[1, 3], [4, 6]], true], 'slice with a step is a view');
  t.eq(T.slice(a, 1).toArray(), [4, 5, 6], 'an integer index drops the axis');
  t.eq(T.slice(a, null, [null, null, -1]).toArray(), [[3, 2, 1], [6, 5, 4]], 'negative step reverses');
  t.eq(T.slice(a, [-1, null], [-2, null]).toArray(), [[5, 6]], 'negative starts count from the end');
  t.eq(T.broadcastShapes([4, 1, 3], [5, 1]), [4, 5, 3], 'broadcast shapes');
  let threw = false; try { T.add(T.zeros([2, 3]), T.zeros([4])); } catch (e) { threw = /broadcast/.test(e.message); }
  t.ok(threw, 'incompatible shapes are refused');
  t.eq(T.add(a, T.tensor([10, 20, 30])).toArray(), [[11, 22, 33], [14, 25, 36]], 'row broadcast');
  t.eq(T.add(a, T.tensor([[100], [200]])).toArray(), [[101, 102, 103], [204, 205, 206]], 'column broadcast');
  t.eq(T.mul(T.tensor([[1], [2]]), T.tensor([1, 2, 3])).toArray(), [[1, 2, 3], [2, 4, 6]], 'outer broadcast');
  t.eq(T.sub(10, a).toArray(), [[9, 8, 7], [6, 5, 4]], 'a number on the left');
  t.eq(T.expand(T.tensor([[1], [2]]), [2, 3]).strides, [1, 0], 'expand is a stride-0 view');
  t.eq(T.concat([a, T.tensor([[7], [8]])], 1).toArray(), [[1, 2, 3, 7], [4, 5, 6, 8]], 'concat along columns');
  t.eq(T.stack([T.tensor([1, 2]), T.tensor([3, 4])], 1).toArray(), [[1, 3], [2, 4]], 'stack on a new axis');
  t.eq(T.split(T.arange(5), [2, 3]).map((x) => x.toArray()), [[0, 1], [2, 3, 4]], 'split');
  t.eq(T.take(a, [2, 0], 1).toArray(), [[3, 1], [6, 4]], 'take columns');
  t.eq(T.gather(a, 1, T.tensor([[2], [0]])).toArray(), [[3], [4]], 'gather');
  t.eq(T.embedding(T.tensor([[0, 0], [1, 1], [2, 2]]), [[2, 0]]).toArray(), [[[2, 2], [0, 0]]], 'embedding lookup keeps the id shape');
  t.eq(T.permute(T.zeros([2, 3, 4]), [2, 0, 1]).shape, [4, 2, 3], 'permute');
  t.eq(T.squeeze(T.zeros([1, 3, 1])).shape, [3], 'squeeze');
  t.eq(T.unsqueeze(T.zeros([3]), -1).shape, [3, 1], 'unsqueeze at the end');
  t.eq(T.float64(() => T.zeros([2]).dtype), 'float64', 'float64 mode');
  t.eq(T.eye(2).toArray(), [[1, 0], [0, 1]], 'eye'); t.eq(T.arange(1, 7, 2).toArray(), [1, 3, 5], 'arange');
  t.eq(T.oneHot([2, 0], 3).toArray(), [[0, 0, 1], [1, 0, 0]], 'one-hot');
}
// ---- hand-computed values ----
{
  const A = T.tensor([[1, 2], [3, 4]]), B = T.tensor([[5, 6], [7, 8]]);
  t.eq(T.matmul(A, B).toArray(), [[19, 22], [43, 50]], 'matmul 2×2');
  t.eq(T.matmul(T.stack([A, B]), A).toArray(), [[[7, 10], [15, 22]], [[23, 34], [31, 46]]], 'batched matmul against a shared matrix');
  t.eq(T.matmul(A, T.tensor([1, 1])).toArray(), [3, 7], 'matrix-vector');
  near(T.softmax(T.tensor([1, 2, 3])), [0.09003, 0.24473, 0.66524], 'softmax [1, 2, 3]');
  near(T.softmax(T.tensor([1000, 1001])), [0.26894, 0.73106], 'softmax is stable for large scores');
  near(T.logSoftmax(T.tensor([1000, 1001])), [-1.31326, -0.31326], 'log-softmax is stable');
  near(T.softmax(T.tensor([[1, 0], [0, Math.log(3)]]), 0), [0.731059, 0.25, 0.268941, 0.75], 'softmax down columns');
  t.eq(T.sum(A).item(), 10, 'sum'); t.eq(T.sum(A, 0).toArray(), [4, 6], 'sum axis 0'); t.eq(T.mean(A, 1, true).toArray(), [[1.5], [3.5]], 'mean keepdims');
  t.eq(T.max(T.tensor([[1, 5], [7, 2]]), 1).toArray(), [5, 7], 'max per row'); t.eq(T.argmax(T.tensor([[1, 5], [7, 2]]), 1).toArray(), [1, 0], 'argmax per row');
  t.eq(T.argmax(T.tensor([[1, 5], [7, 2]]), 0).toArray(), [1, 0], 'argmax down columns');
  near(T.variance(T.tensor([1, 2, 3, 4]), 0), 1.25, 'variance');
  near(T.relu(T.tensor([-1, 0, 2])), [0, 0, 2], 'relu'); near(T.sigmoid(T.tensor([0, 2])), [0.5, 0.880797], 'sigmoid');
  near(T.tanh(T.tensor([0.5])), [0.462117], 'tanh'); near(T.gelu(T.tensor([1, -1])), [0.841192, -0.158808], 'gelu (tanh form)');
  near(T.leakyRelu(T.tensor([-2, 3]), 0.1), [-0.2, 3], 'leaky relu'); near(T.pow(T.tensor([2, 3]), 2), [4, 9], 'pow');
  near(T.sqrt(T.tensor([9])), [3], 'sqrt'); near(T.abs(T.tensor([-2])), [2], 'abs'); near(T.log(T.exp(T.tensor([1.5]))), [1.5], 'log ∘ exp');
  near(T.div(T.tensor([1, 2]), T.tensor([4, 8])), [0.25, 0.25], 'div');
  const x = T.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]);
  t.eq(T.conv2d(x, T.ones([1, 1, 2, 2])).toArray(), [[[[12, 16], [24, 28]]]], 'conv2d 3×3 image, 2×2 ones');
  t.eq(T.conv2d(x, T.ones([1, 1, 3, 3]), T.tensor([1]), { padding: 1, stride: 2 }).toArray(), [[[[13, 17], [25, 29]]]], 'conv2d with padding, stride and bias');
  t.eq(T.conv2d(x, T.ones([1, 1, 2, 2]), null, { dilation: 2 }).toArray(), [[[[20]]]], 'dilated conv2d: corners 1+3+7+9');
  t.eq(T.conv2d(T.tensor([[[[1]], [[2]]]]), T.tensor([[[[3]]], [[[4]]]]), null, { groups: 2 }).toArray(), [[[[3]], [[8]]]], 'grouped conv2d keeps channels apart');
  t.eq(T.im2col(x, 2, 2).toArray(), [[[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]]], 'im2col columns are the windows');
  const p = T.tensor([[[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]]]);
  t.eq(T.maxPool2d(p, 2).toArray(), [[[[4, 8], [12, 16]]]], 'max pooling'); t.eq(T.avgPool2d(p, 2).toArray(), [[[[2.5, 6.5], [10.5, 14.5]]]], 'average pooling');
  near(T.layerNorm(T.tensor([[1, 2, 3]])), [-1.224740, 0, 1.224740], 'layer norm of 1, 2, 3');
  const rm = T.tensor([1, 2]), rv = T.tensor([4, 9]);
  near(T.batchNorm(T.tensor([[3, 5]]), null, null, { training: false, running: { mean: rm, var: rv }, eps: 0 }), [1, 1], 'batch norm in eval uses running statistics');
  const bx = T.tensor([[1, 10], [3, 30]]), run = { mean: T.zeros([2]), var: T.ones([2]) };
  near(T.batchNorm(bx, null, null, { running: run, eps: 0 }), [-1, -1, 1, 1], 'batch norm in training uses batch statistics');
  near(run.mean, [0.2, 2], 'running mean moves 10% toward the batch mean'); near(run.var, [1.1, 0.9 + 0.1 * 200], 'running variance uses the unbiased batch variance');
  const q = T.tensor([[1, 0]]), k = T.tensor([[1, 0], [0, 1]]), v = T.tensor([[1, 2], [3, 4]]), at = T.attention(q, k, v);
  near(at.attn.weights, [0.669762, 0.330238], 'attention weights: softmax of [1, 0] / √2'); near(at, [1.660477, 2.660477], 'attention output: the weighted values');
  near(T.attention(T.ones([2, 2]), T.ones([2, 2]), v, { causal: true }), [1, 2, 2, 3], 'causal attention: the first query sees only the first key');
  t.eq(T.causalMask(3).toArray(), [[1, 0, 0], [1, 1, 0], [1, 1, 1]], 'causal mask');
  near(T.crossEntropy(T.tensor([[0, 0]]), [0]), Math.LN2, 'cross-entropy of even logits is ln 2');
  near(T.crossEntropy(T.tensor([[2, 1, 0.1]]), [0]), 0.417030, 'cross-entropy, three classes');
  near(T.crossEntropy(T.tensor([[0, 0]]), T.tensor([[0.25, 0.75]])), Math.LN2, 'cross-entropy against a probability row');
  near(T.bceLoss(T.tensor([0.5]), [1]), Math.LN2, 'binary cross-entropy at 0.5'); near(T.bceWithLogits(T.tensor([0, 200]), T.tensor([1, 1])), Math.LN2 / 2, 'BCE with logits is stable at 200');
  near(T.hingeLoss(T.tensor([0.5, -2]), T.tensor([1, -1])), 0.25, 'hinge loss'); near(T.multiHinge(T.tensor([[1, 2, 0.5]]), [0]), 2.5, 'multi-class hinge');
  near(T.klDiv(T.log(T.tensor([[0.25, 0.75]])), T.tensor([[0.5, 0.5]])), 0.5 * Math.log(2) + 0.5 * Math.log(2 / 3), 'KL divergence');
  near(T.mseLoss(T.tensor([1, 2]), T.tensor([0, 0])), 2.5, 'MSE'); near(T.huberLoss(T.tensor([0.5, 3]), T.zeros([2])), (0.125 + 2.5) / 2, 'Huber');
  const d = T.dropout(T.ones([1000]), 0.3, 5), kept = Array.from(d.values()).filter((z) => z > 0);
  t.ok(Math.abs(kept.length / 1000 - 0.7) < 0.05 && close(kept[0], 1 / 0.7), 'dropout keeps about 70% and scales them by 1 / 0.7');
  t.eq(Array.from(T.dropout(T.ones([20]), 0.5, 9).values()), Array.from(T.dropout(T.ones([20]), 0.5, 9).values()), 'dropout masks repeat for a seed');
  t.ok(T.dropout(T.ones([5]), 0.5, 1, false).values().every((z) => z === 1), 'dropout is the identity in eval');
}
// ---- seeds and initialisers ----
{
  const r1 = T.rng(42), r2 = T.rng(42), a = [r1.next(), r1.next(), r1.normal()], b = [r2.next(), r2.next(), r2.normal()];
  t.eq(a, b, 'the same seed gives the same stream'); t.ok(T.rng(43).next() !== a[0], 'another seed differs');
  const n = T.normal([20000], 1, 2, 3), v = n.values(), mean = v.reduce((p, q) => p + q, 0) / v.length, sd = Math.sqrt(v.reduce((p, q) => p + (q - mean) ** 2, 0) / v.length);
  t.ok(Math.abs(mean - 1) < 0.05 && Math.abs(sd - 2) < 0.05, 'normal(1, 2): mean ' + mean.toFixed(3) + ', sd ' + sd.toFixed(3));
  const std = (t0) => { const w = t0.values(); let s = 0; for (const z of w) s += z * z; return Math.sqrt(s / w.length); };
  t.ok(Math.abs(std(T.xavier([300, 200], 1)) - Math.sqrt(2 / 500)) < 0.003, 'Xavier uniform has variance 2 / (in + out)');
  t.ok(Math.abs(std(T.xavier([300, 200], 1, { dist: 'normal' })) - Math.sqrt(2 / 500)) < 0.003, 'Xavier normal');
  t.ok(Math.abs(std(T.he([400, 100], 2)) - Math.sqrt(2 / 400)) < 0.003, 'He normal has variance 2 / in');
  t.ok(Math.abs(std(T.he([64, 16, 3, 3], 2)) - Math.sqrt(2 / 144)) < 0.005, 'He for a conv filter uses in·kh·kw');
  const u = T.uniform([1000], -1, 1, 4).values(); t.ok(Math.min(...u) >= -1 && Math.max(...u) < 1, 'uniform stays in range');
  t.eq(Array.from(T.randn([5], 8).values()), Array.from(T.randn([5], 8).values()), 'randn repeats for a seed');
  const r = T.rng(5); r.next(); const st = r.state(), x1 = r.next(); r.setState(st); t.eq(r.next(), x1, 'a generator state can be saved and restored');
}
// ---- graph inspection and traces ----
{
  const x = T.tensor([[1, 2]]).named('x'), W = T.param(T.tensor([[1], [-1]]), 'W'), h = T.matmul(x, W).named('h'), L = T.sum(T.square(h)).named('L');
  L.backward();
  const gr = T.graph(L);
  t.eq(gr.nodes.map((n) => n.name), ['x', 'W', 'h', 'square#' + gr.nodes[3].id, 'L'], 'graph nodes in topological order, by name');
  t.eq([gr.find('h').value.values, gr.find('h').grad.values, gr.find('W').grad.values], [[-1], [-2], [-2, -4]], 'each node keeps its value and gradient');
  t.eq(gr.edges.length, 4, 'graph edges');
  const fw = T.traceForward(L), bw = T.traceBackward(L);
  t.ok(fw.some((s) => s.kind === 'op' && /h = matmul\(x, W\)/.test(s.say)) && bw.length >= 3 && /norm/.test(bw[0].say), 'forward and backward traces read as sentences');
  const em = T.explain.matmul([[1, 2]], [[3], [4]]);
  t.ok(em.trace[1].say === 'C[0][0] = 1·3 + 2·4 = 11.', 'matmul trace: ' + em.trace[1].say);
  const es = T.explain.softmax([1, 2]); t.ok(/0\.2689, 0\.7311/.test(es.trace[3].say), 'softmax trace: ' + es.trace[3].say);
  const ea = T.explain.attention(T.tensor([[1, 0], [0, 1]]), T.tensor([[1, 0], [0, 1]]), T.tensor([[1, 2], [3, 4]]), { causal: true });
  t.ok(close(ea.out, T.attention(T.tensor([[1, 0], [0, 1]]), T.tensor([[1, 0], [0, 1]]), T.tensor([[1, 2], [3, 4]]), { causal: true })) && ea.trace.some((s) => s.kind === 'mask'), 'attention trace matches the op and shows the mask');
  const ec = T.explain.conv2d(T.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), T.ones([1, 1, 2, 2]));
  t.ok(ec.trace.length === 5 && ec.trace[4].value === 28, 'conv2d trace: one step per output position');
}
// ---- modules ----
{
  const m = new T.nn.Sequential(new T.nn.Conv2d(1, 2, 3, { rng: 1, padding: 1 }), new T.nn.BatchNorm(2), new T.nn.Activation('relu'), new T.nn.MaxPool(2), new T.nn.Flatten(), new T.nn.Dropout(0.2), new T.nn.Dense(8, 3, { rng: 1 }));
  t.eq(m.namedParameters().map((x) => x[0]), ['layers.0.W', 'layers.0.b', 'layers.1.gamma', 'layers.1.beta', 'layers.6.W', 'layers.6.b'], 'parameter paths');
  t.eq(m.numParams(), 18 + 2 + 2 + 2 + 24 + 3, 'parameter count');
  const x = T.randn([2, 1, 4, 4], 3);
  m.forward(x); m.eval();
  const y1 = m.forward(x).toArray(), saved = JSON.parse(JSON.stringify(m.stateDict())), typed = m.toTyped();
  t.ok(!m.layers[5].training && !m.layers[1].training, 'eval() reaches every layer');
  const m2 = new T.nn.Sequential(new T.nn.Conv2d(1, 2, 3, { rng: 9, padding: 1 }), new T.nn.BatchNorm(2), new T.nn.Activation('relu'), new T.nn.MaxPool(2), new T.nn.Flatten(), new T.nn.Dropout(0.2), new T.nn.Dense(8, 3, { rng: 9 }));
  m2.eval();
  t.ok(!close(m2.forward(x), y1), 'another seed gives another model');
  m2.loadState(saved);
  t.ok(close(m2.forward(x), y1, 1e-6), 'load from JSON reproduces the outputs (with the batch-norm running statistics)');
  const m3 = new T.nn.Sequential(new T.nn.Conv2d(1, 2, 3, { rng: 5, padding: 1 }), new T.nn.BatchNorm(2), new T.nn.Activation('relu'), new T.nn.MaxPool(2), new T.nn.Flatten(), new T.nn.Dropout(0.2), new T.nn.Dense(8, 3, { rng: 5 }));
  m3.eval(); m3.fromTyped(typed);
  t.ok(close(m3.forward(x), y1, 1e-6) && typed.buffer instanceof Float32Array, 'load from a typed array');
  t.ok(m.describe().length === 8, 'describe lists the layers');
  const mha = new T.nn.MultiHeadAttention(8, 2, { rng: 2, causal: true }), o = mha.forward(T.randn([1, 4, 8], 1));
  t.eq([o.shape, mha.weights.shape], [[1, 4, 8], [1, 2, 4, 4]], 'multi-head attention keeps its weights per head');
  t.ok(mha.weights.get(0, 0, 0, 1) === 0 && Math.abs(mha.weights.get(0, 1, 3, 0) + mha.weights.get(0, 1, 3, 1) + mha.weights.get(0, 1, 3, 2) + mha.weights.get(0, 1, 3, 3) - 1) < 1e-5, 'causal weights: none on the future, rows sum to 1');
  const lstm = new T.nn.LSTM(2, 3, { rng: 1 }), lo = lstm.forward(T.randn([2, 5, 2], 1));
  t.eq([lo.shape, lo.c.shape, lstm.steps.length, Object.keys(lstm.steps[0])], [[2, 5, 3], [2, 3], 5, ['i', 'f', 'g', 'o', 'c', 'h']], 'LSTM keeps its gates per step');
}
// ---- optimisers on test functions ----
function minimise(make, f, x0, steps) {
  const x = T.param(T.tensor(x0, undefined), 'x'), opt = make([x]);
  for (let i = 0; i < steps; i++) { opt.zeroGrad(); const L = f(x); L.backward(); opt.step(); }
  return Array.from(x.values());
}
const rosen = (x) => { const a = T.slice(x, 0), b = T.slice(x, 1); return T.add(T.square(T.sub(1, a)), T.scale(T.square(T.sub(b, T.square(a))), 100)); };
const quad = (x) => T.sum(T.mul(T.tensor([1, 10, 0.5]), T.square(T.sub(x, T.tensor([3, -2, 1])))));
const O = T.optim;
const cases = [
  ['SGD', (p) => new O.SGD(p, { lr: 0.04 }), (p) => new O.SGD(p, { lr: 0.001 }), 30000],
  ['Momentum', (p) => new O.Momentum(p, { lr: 0.02 }), (p) => new O.Momentum(p, { lr: 0.0005 }), 12000],
  ['Nesterov', (p) => new O.Nesterov(p, { lr: 0.02 }), (p) => new O.Nesterov(p, { lr: 0.0005 }), 12000],
  ['AdaGrad', (p) => new O.AdaGrad(p, { lr: 0.5 }), (p) => new O.AdaGrad(p, { lr: 0.3 }), 30000],
  ['RMSProp', (p) => new O.RMSProp(p, { lr: 0.01, schedule: T.sched.cosine({ lr: 0.01, min: 1e-5, total: 3000 }) }), (p) => new O.RMSProp(p, { schedule: T.sched.cosine({ lr: 0.003, min: 1e-6, total: 20000 }) }), 20000],
  ['Adam', (p) => new O.Adam(p, { lr: 0.05 }), (p) => new O.Adam(p, { lr: 0.01 }), 12000],
  ['AdamW', (p) => new O.AdamW(p, { lr: 0.05, weightDecay: 0 }), (p) => new O.AdamW(p, { lr: 0.01, weightDecay: 0 }), 12000],
];
for (const [name, mq, mr, rsteps] of cases) {
  const q = minimise(mq, quad, [0, 0, 0], 3000);
  t.ok(close(q, [3, -2, 1], 1e-3), name + ' finds the quadratic minimum (3, −2, 1): ' + q.map((z) => z.toFixed(4)));
  const r = minimise(mr, rosen, [-1.2, 1], rsteps);
  t.ok(Math.hypot(r[0] - 1, r[1] - 1) < 0.02, name + ' reaches the Rosenbrock minimum (1, 1): ' + r.map((z) => z.toFixed(4)) + ' after ' + rsteps + ' steps');
}
{
  const wd = minimise((p) => new O.AdamW(p, { lr: 0.01, weightDecay: 0.1 }), (x) => T.scale(T.sum(x), 0), [1, -2], 2000);
  t.ok(close(wd, [1 * (1 - 0.001) ** 2000, -2 * (1 - 0.001) ** 2000], 1e-3), 'AdamW decay shrinks weights by (1 − lr·λ) per step with no gradient');
  const l2 = minimise((p) => new O.SGD(p, { lr: 0.1, weightDecay: 0.5 }), (x) => T.scale(T.sum(x), 0), [1], 10);
  t.ok(close(l2, [0.95 ** 10]), 'SGD weight decay is an L2 gradient');
  const p = T.param(T.tensor([0, 0])); p.grad = T.tensor([3, 4]);
  t.ok(T.clipGradNorm([p], 1) === 5 && close(p.grad, [0.6, 0.8]), 'clip by norm'); p.grad = T.tensor([3, -4]); T.clipGradValue([p], 1); t.ok(close(p.grad, [1, -1]), 'clip by value');
  const sc = T.sched;
  t.ok(close(sc.curve(sc.step({ lr: 1, every: 2, gamma: 0.5 }), 5), [1, 1, 0.5, 0.5, 0.25]), 'step schedule');
  t.ok(close([sc.cosine({ lr: 1, total: 10 })(0), sc.cosine({ lr: 1, total: 10 })(5), sc.cosine({ lr: 1, total: 10 })(10)], [1, 0.5, 0]), 'cosine schedule');
  t.ok(close(sc.curve(sc.warmup({ steps: 4, lr: 1 }), 6), [0.25, 0.5, 0.75, 1, 1, 1]), 'warm-up');
  t.ok(close(sc.curve(sc.warmup({ steps: 2, then: sc.cosine({ lr: 1, total: 2 }) }), 5), [0.5, 1, 1, 0.5, 0]), 'warm-up then cosine');
  const opt = new O.Momentum([T.param(T.tensor([1]))]); opt.params[0].grad = T.tensor([2]); opt.step();
  t.ok(opt.inspect()[0].v === 2, 'optimiser state can be inspected');
}
// ---- the training runner (on the page; the worker side through a stand-in) ----
{
  const g = T.rng(1), X = [], y = [];
  for (let i = 0; i < 64; i++) { const a = g.next() > 0.5 ? 1 : 0, b = g.next() > 0.5 ? 1 : 0; X.push([a + g.normal(0, 0.1), b + g.normal(0, 0.1)]); y.push(a ^ b); }
  const msgs = [], r = new T.Runner(T.jobs.make('mlp', { X, y, sizes: [8], act: 'tanh', seed: 3, lr: 0.05 }), { snapshotEvery: 5, snapshot: ['layers.0.W'], checkpointEvery: 10 }, (m) => msgs.push(m));
  r.step(10);
  t.eq([r.stepNo, r.state, msgs.filter((m) => m.type === 'progress').length, msgs.filter((m) => m.type === 'snapshot').length, msgs.filter((m) => m.type === 'checkpoint').length], [10, 'paused', 10, 2, 1], 'step(10): ten progress messages, two snapshots, one checkpoint');
  t.eq(Object.keys(msgs.find((m) => m.type === 'snapshot').tensors), ['layers.0.W'], 'snapshots carry the chosen tensors');
  const ck = msgs.find((m) => m.type === 'checkpoint');
  // a second runner restored from the checkpoint continues exactly like the first
  const r2 = new T.Runner(T.jobs.make('mlp', { X, y, sizes: [8], act: 'tanh', seed: 3, lr: 0.05 }), {}, () => {});
  r2.restore(ck.state);
  t.ok(close(r2.job.model.layers[0].W, r.job.model.layers[0].W, 0), 'restoring a checkpoint restores the weights');
  const done = await new Promise((res) => {
    const r3 = new T.Runner(T.jobs.make('mlp', { X, y, sizes: [8], act: 'tanh', seed: 3, lr: 0.05 }), { maxSteps: 300, chunkMs: 5 }, (m) => { if (m.type === 'done') res({ m, r3 }); });
    r3.start();
    setTimeout(() => r3.pause(), 1);
    setTimeout(() => { t.ok(r3.state === 'paused' && r3.stepNo < 300, 'pause stops between slices at step ' + r3.stepNo); r3.resume(); }, 20);
  });
  t.ok(done.m.step === 300 && done.r3.history[299].loss < 0.1, 'resumed to the end: XOR loss ' + done.r3.history[299].loss.toFixed(4));
  const sent = [], fake = { postMessage: (m) => sent.push(m) };
  T.worker.serve(fake);
  fake.onmessage({ data: { cmd: 'start', job: 'mlp', params: { X, y, sizes: [8], act: 'tanh', seed: 3 }, opts: { paused: true, snapshotEvery: 2 } } });
  fake.onmessage({ data: { cmd: 'step', arg: 4 } });
  fake.onmessage({ data: { cmd: 'nonsense' } });
  t.eq([sent.filter((m) => m.type === 'progress').length, sent.filter((m) => m.type === 'snapshot').length, sent.at(-1).state], [4, 2, 'paused'], 'the worker side answers start, step and ignores unknown commands');
  fake.onmessage({ data: { cmd: 'start', job: 'nope' } });
  t.ok(sent.at(-1).type === 'error' && /no training job/.test(sent.at(-1).message), 'an unknown job reports an error');
  const pageMsgs = [];
  const ctl = T.train('mlp', { X, y, sizes: [8], act: 'tanh', seed: 3 }, { maxSteps: 20 }, (m) => pageMsgs.push(m));
  await new Promise((res) => setTimeout(res, 200));
  t.ok(ctl.mode === 'page' && pageMsgs.some((m) => m.type === 'done'), 'with no Worker (node), train() runs on the page in slices');
}
// ---- the worker code cut from a built page (every core file, as build.mjs concatenates them) ----
{
  const { readFileSync } = await import('node:fs');
  const { files } = await import('./load.mjs');
  const vm = await import('node:vm');
  const built = files(['core', 'app']).map((f) => `// ==== ${f} ====\n${readFileSync(new URL('../src/' + f, import.meta.url), 'utf8')}`).join('\n');
  const ctx = vm.createContext({ console, performance, setTimeout, clearTimeout, TextEncoder, TextDecoder, Nooral: {} });
  ctx.document = { scripts: [{ textContent: built }], querySelectorAll: () => [] };
  vm.runInContext(readFileSync(new URL('../src/core/t-15-worker.js', import.meta.url), 'utf8').replace('globalThis.Nooral = globalThis.Nooral || {}', 'globalThis.Nooral = Object.assign(globalThis.Nooral, {T: {}})'), ctx);
  const code = ctx.Nooral.T.worker.source();
  const cores = files(['core']).length;
  t.ok(code && (code.match(/(^|\n)\/\/ ==== core\//g) || []).length === cores && !/\n\/\/ ==== app\//.test(code), 'the worker gets all ' + cores + ' core files and nothing else from the built page');
  const posted = [], wctx = vm.createContext({ console, performance, setTimeout, clearTimeout, TextEncoder, TextDecoder });
  wctx.self = wctx; wctx.postMessage = (m) => posted.push(m);
  vm.runInContext(code, wctx);
  wctx.onmessage({ data: { cmd: 'start', job: 'mlp', params: { X: [[0, 0], [1, 1]], y: [0, 1], seed: 1 }, opts: { paused: true } } });
  wctx.onmessage({ data: { cmd: 'step', arg: 3 } });
  t.ok(posted.filter((m) => m.type === 'progress').length === 3, 'that code runs as a worker and trains');
}
// ---- determinism of a whole training run ----
{
  const train = (seed) => {
    const job = T.jobs.make('mlp', { X: [[0, 0], [0, 1], [1, 0], [1, 1]], y: [0, 1, 1, 0], sizes: [6], seed, lr: 0.05, batch: 2 });
    for (let i = 0; i < 50; i++) job.step();
    return Array.from(job.model.toTyped().buffer);
  };
  t.eq(train(7), train(7), 'the same seed trains to bit-identical weights');
  t.ok(JSON.stringify(train(7)) !== JSON.stringify(train(8)), 'another seed trains differently');
}
// ---- the matrix kernels: the WebAssembly ones against the JS one ----
{
  const K = T.kernels, g = T.rng(12);
  t.ok(K.wasm === true, 'WebAssembly SIMD kernels are in use here');
  let exactBad = 0, fastWorst = 0, cases = 0;
  for (const [M, Kk, N] of [[1, 300, 1], [3, 7, 5], [5, 64, 9], [64, 3, 17], [33, 129, 65], [128, 256, 96], [7, 1, 300], [130, 70, 257]]) {
    for (const ta of [false, true]) for (const tb of [false, true]) for (const F of [Float32Array, Float64Array]) {
      const A = new F(M * Kk + 5).map(() => g.normal()), B = new F(Kk * N + 3).map(() => g.normal()), o1 = new F(M * N + 2), o2 = new F(M * N + 2);
      K.gemmJS(A, B, M, Kk, N, o1, 5, 3, 2, ta, tb); K.gemm(A, B, M, Kk, N, o2, 5, 3, 2, ta, tb);
      for (let i = 0; i < o1.length; i++) if (!Object.is(o1[i], o2[i])) exactBad++;
      if (F === Float32Array) {
        const o3 = new F(M * N + 2); K.gemmFast(A, B, M, Kk, N, o3, 5, 3, 2, ta, tb);
        for (let i = 2; i < o1.length; i++) fastWorst = Math.max(fastWorst, Math.abs(o1[i] - o3[i]) / Math.max(1, Math.abs(o1[i])));
      }
      cases++;
    }
  }
  t.ok(exactBad === 0, 'the exact kernel gives the JS kernel\'s results to the last bit (' + cases + ' cases: odd sizes, transposed sides, offsets, float32 and float64)');
  t.ok(fastWorst < 2e-5, 'the fast float32 kernel is within ' + fastWorst.toExponential(1) + ' of them');
  const a = T.normal([40, 30], 0, 1, g), b = T.normal([30, 20], 0, 1, g), e = T.matmul(a, b).values(), f = T.fast(() => T.matmul(a, b)).values();
  t.ok(K.mode === 'exact' && e.some((x, i) => x !== f[i]) && e.every((x, i) => Math.abs(x - f[i]) < 1e-4), 'S.T.fast(fn) switches to the fast kernel only inside fn');
}
// ---- slices pass their gradient back without a full-size temporary each ----
{
  const g = T.rng(13), x = T.param(T.normal([4, 9, 3], 0, 1, g));
  let s = T.scalar(0);
  for (let i = 0; i < 9; i++) s = T.add(s, T.sum(T.mul(T.slice(x, null, i, null), i + 1)));
  s.backward();
  t.ok(x.grad.values().every((v, k) => v === Math.floor(k / 3) % 9 + 1), 'the gradients of nine slices land in the right places');
  const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
  const rec = (L) => { const y = T.param(T.normal([16, L, 32], 0, 1, g)), W = T.param(T.normal([32, 32], 0, 0.1, g)), c0 = cpu();
    let h = T.zeros([16, 32]); for (let k = 0; k < L; k++) h = T.tanh(T.add(T.matmul(h, W), T.slice(y, null, k, null))); T.sum(h).backward(); return cpu() - c0; };
  rec(50);
  const t1 = Math.min(rec(100), rec(100)), t4 = Math.min(rec(400), rec(400));
  t.ok(t4 < 9 * t1, 'a recurrence over 400 slices costs about four times one over 100, not sixteen (' + t1.toFixed(1) + ' ms and ' + t4.toFixed(1) + ' ms)');
}
// ---- the CPU backend: the async interface and its speed ----
{
  const cmp = await T.backends.compare(T.backends.cpu);
  t.ok(cmp.every((c) => c.ok && c.maxErr === 0), 'the CPU backend agrees with itself on all ' + cmp.length + ' comparison cases');
  t.ok((await T.backends.select({ prefer: 'webgpu' })).name === 'cpu' && T.backends.log.some((l) => /webgpu/.test(l)), 'select() falls back to the CPU without WebGPU: ' + T.backends.log.join('; '));
  if (!process.argv.includes('--quick')) {
    const rows = await T.backends.bench(T.backends.cpu, { sizes: [128, 256, 512], ms: 400 });
    for (const r of rows) console.log('  cpu (node) ' + r.op.padEnd(36) + r.ms.toFixed(1).padStart(8) + ' ms ' + r.gflops.toFixed(2).padStart(7) + ' GFLOP/s');
  }
}
t.done();
