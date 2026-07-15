// The deep-learning models (S.DL): each converges on its demo task within a time budget on the CPU, gradient checks
// of small versions against finite differences, determinism by seed, save/load round trips, BPE, word2vec, sampling,
// the KV cache, interpretability helpers, and training through the streaming runner. With --browser, also a real
// Web Worker in headless Chrome and a WebGPU speed comparison where the browser offers it. --quick skips the GPT runs.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T, DL = S.DL;
const t = harness('dl');
DL.ensureJobs();
const report = [];
// budgets are in CPU time of this process (user + system), so a busy machine does not fail them; wall time is shown too
const cpuMs = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const timed = (name, budgetMs, fn) => {
  const t0 = performance.now(), c0 = cpuMs(), r = fn(), ms = performance.now() - t0, cpu = cpuMs() - c0;
  console.log(`  ${name.padEnd(36)} ${r.label}  in ${(cpu / 1000).toFixed(2)} s CPU (${(ms / 1000).toFixed(2)} s wall)`);
  t.ok(cpu < budgetMs, name + ' took ' + cpu.toFixed(0) + ' ms of CPU (budget ' + budgetMs + ')');
  report.push({ name, params: r.params, seconds: cpu / 1000 });
  return r;
};
const close = (a, b, tol = 1e-5) => a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= tol * Math.max(1, Math.abs(b[i])));

// ---- the API ----
t.eq(['mlp', 'cnn', 'rnn', 'encoder', 'gpt'].filter((k) => DL.kinds().includes(k)).length, 5, 'all model kinds are registered');
t.ok(['digits', 'shapes', 'moons', 'curve', 'chars', 'reviews', 'counting', 'topics', 'majority', 'text', 'copy', 'reverse', 'sort'].every((n) => DL.tasks.names().includes(n)), 'all tasks are registered');

// ---- MLP classifier and regressor ----
timed('MLP classifier on moons', 8000, () => {
  const { net, trainer } = DL.setup({ task: 'moons', train: { steps: 300 } });
  trainer.fit({ steps: 300 });
  const v = trainer.evaluate('val');
  t.ok(v.accuracy >= 0.95, 'MLP moons: val accuracy ' + v.accuracy);
  return { label: 'val accuracy ' + v.accuracy.toFixed(3) + ', ' + net.numParams() + ' parameters', params: net.numParams() };
});
timed('MLP regressor on a sine curve', 8000, () => {
  const { net, trainer } = DL.setup({ task: 'curve', train: { steps: 600 } });
  trainer.fit({ steps: 600 });
  const v = trainer.evaluate('val');
  t.ok(v.loss < 0.06 && v.accuracy === undefined, 'MLP regression: val MSE ' + v.loss);
  return { label: 'val MSE ' + v.loss.toFixed(4) + ' (noise variance 0.0225)', params: net.numParams() };
});

// ---- CNN on generated digits, with augmentation, feature maps and Grad-CAM ----
let cnn = null;
timed('CNN (LeNet-style) on 16×16 digits', 40000, () => {
  const { net, trainer } = DL.setup({ task: 'digits', taskOpts: { size: 16, n: 3000, val: 600 }, train: { snapshotEvery: 100, snapshot: ['conv1.W', 'relu1', 'relu2'], checkpointEvery: 200 } });
  trainer.fit({ steps: 400 });
  const v = trainer.evaluate('val', 600);
  t.ok(v.accuracy >= 0.95, 'CNN digits: val accuracy ' + v.accuracy);
  t.ok(trainer.snapshots.length === 4 && trainer.snapshots[0].tensors.relu1.shape.join() === '1,6,16,16' && trainer.snapshots[0].tensors['conv1.W'].shape.join() === '6,1,5,5', 'feature maps and filters are snapshotted every 100 steps');
  t.ok(trainer.checkpoints.length === 2 && trainer.checkpoints[1].step === 400, 'checkpoints every 200 steps');
  cnn = { net, trainer };
  return { label: 'val accuracy ' + v.accuracy.toFixed(3) + ', ' + net.numParams() + ' parameters', params: net.numParams() };
});
{
  const net = cnn.net, px = cnn.trainer.task.val.examples[3].px, cls = cnn.trainer.task.val.examples[3].y;
  const cam = DL.gradCam(net, px, cls);
  t.ok(cam.map.length === 4 && cam.map[0].length === 4 && cam.weights.length === 16 && cam.layer === 'relu2', 'Grad-CAM on the named layer relu2: a 4×4 map from 16 channels');
  t.ok(Math.max(...cam.map.flat()) === 1 && cam.trace.length === 2, 'Grad-CAM map scaled to 1, with a trace');
  const viaExplain = S.Explain.gradCamT(DL.camNet(net, 'relu2'), px, [1, 1, 16, 16], cls);
  t.ok(close(viaExplain.map.flat(), cam.map.flat(), 1e-6), 'S.Explain.gradCamT through DL.camNet gives the same map');
  const sal = DL.saliency(net, px, cls);
  t.ok(sal.map.length === 256 && Math.max(...sal.map) === 1 && sal.trace[0].say.includes('class ' + cls), 'saliency map over the 256 pixels');
  t.ok(net.tapNames().join() === 'conv1,relu1,pool1,relu2,conv2,pool2,flat,fc0,fcAct0,logits'.split(',').sort((a, b) => net.tapNames().indexOf(a) - net.tapNames().indexOf(b)).join(), 'named intermediate outputs');
  const before = net.predict(cnn.trainer.task.val.get([0, 1, 2]).x).values();
  const copy = DL.load(JSON.parse(JSON.stringify(DL.save(net))));
  t.ok(close(copy.predict(cnn.trainer.task.val.get([0, 1, 2]).x).values(), before, 0), 'CNN save / load round trip gives identical predictions');
  t.ok(DL.paramCounts(net).total === net.numParams() && DL.paramCounts(net).layers[0].name === 'conv1.W', 'parameter counts by layer');
}
{
  const g = T.rng(3), img = new Float32Array(64);
  img[3 * 8 + 3] = 1;
  t.ok(close(DL.augmentImage(img, 8, 8, g, {}), img, 0), 'augmentation with nothing set is the identity');
  const moved = DL.augmentImage(img, 8, 8, T.rng(4), { shift: 2 }), sum = (a) => a.reduce((s, v) => s + v, 0);
  t.ok(Math.abs(sum(moved) - 1) < 1e-6 && moved[27] < 1, 'a sub-pixel shift moves the ink and keeps its amount');
  const noisy = DL.augmentImage(new Float32Array(64).fill(0.5), 8, 8, T.rng(5), { noise: 0.1, rotate: 0.3 });
  t.ok(noisy.some((v) => v !== 0.5) && noisy.every((v) => v >= 0 && v <= 1), 'noise and rotation stay within [0, 1]');
  const tk = DL.task('digits', { size: 16, n: 40, val: 8 }), a = tk.train.get([0, 1], T.rng(1), { shift: 2, rotate: 0.2, noise: 0.05 }), b = tk.train.get([0, 1], T.rng(1), { shift: 2, rotate: 0.2, noise: 0.05 });
  t.ok(close(a.x.values(), b.x.values(), 0) && !close(a.x.values(), tk.train.get([0, 1]).x.values(), 1e-3), 'augmented batches are reproducible by seed and differ from the originals');
}

// ---- recurrent models ----
timed('LSTM counting ones', 20000, () => {
  const { net, trainer } = DL.setup({ task: 'counting', train: { snapshotEvery: 150 } });
  trainer.fit({ steps: 300 });
  const v = trainer.evaluate('val'), snap = trainer.snapshots[0].tensors;
  t.ok(v.accuracy >= 0.97, 'LSTM counting: per-step val accuracy ' + v.accuracy);
  t.ok(snap['gate.f'] && snap['gate.f'].shape.join() === '1,8,24' && snap.cell.shape.join() === '1,8,24', 'LSTM gate activations per step are snapshotted');
  const gates = net.gates();
  t.ok(['gate.i', 'gate.f', 'gate.g', 'gate.o', 'cell'].every((k) => gates[k]) && gates['gate.f'].values().every((x) => x > 0 && x < 1), 'forget gates lie in (0, 1)');
  return { label: 'per-step val accuracy ' + v.accuracy.toFixed(3) + ', ' + net.numParams() + ' parameters', params: net.numParams() };
});
for (const cell of ['gru', 'rnn']) {
  const { trainer } = DL.setup({ task: 'counting', config: { cell }, train: { lr: 1e-2 } });
  const first = trainer.evaluate('val').loss;
  trainer.fit({ steps: 150 });
  const v = trainer.evaluate('val');
  t.ok(v.loss < first * 0.5, cell.toUpperCase() + ' counting: val loss ' + first.toFixed(3) + ' → ' + v.loss.toFixed(3));
}
timed('GRU sentiment on the short reviews', 15000, () => {
  const { net, trainer } = DL.setup({ task: 'reviews' });
  trainer.fit({ steps: 120 });
  const tr = trainer.evaluate('train'), v = trainer.evaluate('val');
  t.ok(tr.accuracy >= 0.95, 'GRU sentiment: training accuracy ' + tr.accuracy);
  return { label: 'train accuracy ' + tr.accuracy.toFixed(2) + ', val ' + v.accuracy.toFixed(2) + ' (' + trainer.task.val.n + ' held-out reviews)', params: net.numParams() };
});
timed('LSTM next character on the corpus', 60000, () => {
  const { net, trainer } = DL.setup({ task: 'chars' });
  const first = trainer.evaluate('val', 128).loss;
  trainer.fit({ steps: 150 });
  const v = trainer.evaluate('val', 128), gen = net.complete('The ', 40, { temperature: 0.7 }, 2);
  t.ok(v.loss < first - 0.8 && v.perplexity > 1, 'char LSTM: val loss ' + first.toFixed(2) + ' → ' + v.loss.toFixed(2) + ', perplexity ' + v.perplexity.toFixed(1));
  t.ok(gen.text.startsWith('The ') && gen.text.length === 44 && gen.steps.length === 40, 'char LSTM continues a prompt');
  return { label: 'val loss ' + v.loss.toFixed(2) + ', perplexity ' + v.perplexity.toFixed(1), params: net.numParams() };
});

// ---- transformers ----
timed('Transformer encoder on majority', 20000, () => {
  const { net, trainer } = DL.setup({ task: 'majority' });
  trainer.fit({ steps: 150 });
  const v = trainer.evaluate('val'), maps = DL.attentionMaps(net);
  t.ok(v.accuracy >= 0.95, 'encoder: val accuracy ' + v.accuracy);
  t.ok(maps.length === 2 && maps[0].weights.length === 9 && Math.abs(maps[0].weights[0].reduce((s, x) => s + x, 0) - 1) < 1e-5, 'attention per head, rows summing to 1');
  return { label: 'val accuracy ' + v.accuracy.toFixed(3), params: net.numParams() };
});
{
  const net = DL.build('encoder', { vocab: 5, d: 8, heads: 2, layers: 1, maxLen: 6, classes: 2 });
  const a = net.predict(T.tensor([[1, 2, 3, 0, 0, 0]])).values(), b = net.predict(T.tensor([[1, 2, 3, 0, 0]])).values();
  t.ok(close(a, b, 1e-5), 'padding (token 0) does not change the encoder output');
  t.ok(DL.attentionMaps(net)[0].weights[0].slice(3).every((x) => x === 0), 'no attention to padding');
}
let rev = null;
timed('Decoder transformer reversing digits', 40000, () => {
  const { net, trainer, task } = DL.setup({ task: 'reverse', train: { steps: 200 } });
  trainer.fit({ steps: 200 });
  const c = task.check(net, 200);
  t.ok(c.exact / c.n >= 0.95, 'reversal: ' + c.say);
  rev = { net, task };
  return { label: c.exact + '/' + c.n + ' exact reversals', params: net.numParams() };
});
{
  const { net, trainer, task } = DL.setup({ task: 'copy', train: { steps: 250 } });
  trainer.fit({ steps: 250 });
  const c = task.check(net, 100);
  t.ok(c.exact >= 90, 'copying: ' + c.say);
}

// ---- KV cache, weight tying, sampling ----
{
  const net = rev.net, prompt = [3, 1, 4, 1, 5, 10];
  const a = net.generate(prompt, 5, { method: 'greedy' }, 1, { cache: true }), b = net.generate(prompt, 5, { method: 'greedy' }, 1, { cache: false });
  t.eq(a.ids, b.ids, 'generation with the KV cache equals generation without it');
  const full = net.predict(T.tensor([a.ids.slice(0, 10)])).values(), V = 11;
  const cache = net.newCache();
  let last = null;
  T.noGrad(() => { net.eval(); net.forward(T.tensor([a.ids.slice(0, 6)]), { cache }); for (let i = 6; i < 10; i++) last = net.forward(T.tensor([[a.ids[i]]]), { cache }); });
  t.ok(close(last.values(), full.slice(9 * V, 10 * V), 1e-4) && cache.pos === 10 && cache.layers[0].k.shape.join() === '1,2,10,16', 'cached logits match the full forward pass; the cache holds keys for 10 positions');
  t.ok(!net.head && !net.namedParameters().some(([n]) => n.startsWith('head')), 'the output layer is tied to the token embedding');
  const g = DL.build('gpt', { vocab: 7, d: 8, heads: 2, layers: 1, context: 4, tie: false });
  t.ok(g.numParams() === DL.build('gpt', { vocab: 7, d: 8, heads: 2, layers: 1, context: 4 }).numParams() + 56, 'untying adds a separate vocab × d output matrix');
  let threw = false;
  try { net.predict(T.tensor([new Array(12).fill(1)])); } catch (e) { threw = /context/.test(e.message); }
  t.ok(threw, 'a sequence longer than the context is refused');
  const long = net.generate([1, 2], 20, { temperature: 1 }, 4);
  t.ok(long.ids.length === 22, 'generation beyond the context slides the window');
}
{
  const row = Float32Array.from([2, 1, 0.5, -1, 3]);
  t.eq(DL.sample(row, { method: 'greedy' }).id, 4, 'greedy picks the largest logit');
  t.eq(DL.sample(row, { topK: 1 }, 9).id, 4, 'top-1 sampling is greedy');
  const k2 = DL.sample(row, { topK: 2 }, 3);
  t.ok(k2.kept === 2 && [0, 4].includes(k2.id) && Math.abs(k2.probs.reduce((s, x) => s + x, 0) - 1) < 1e-9, 'top-k keeps k tokens and renormalises');
  const p = DL.sample(row, { topP: 0.5 }, 3), all = DL.sample(row, {}, 3);
  t.ok(p.kept === 1 && all.kept === 5, 'top-p keeps the smallest set reaching p');
  const cold = DL.sample(row, { temperature: 0.1 }, 3), hot = DL.sample(row, { temperature: 10 }, 3);
  t.ok(cold.probs[4] > 0.99 && hot.probs[4] < 0.3, 'temperature sharpens or flattens the distribution');
  const seq = (seed) => { const g = T.rng(seed); return Array.from({ length: 20 }, () => DL.sample(row, { temperature: 1.5 }, g).id).join(); };
  t.ok(seq(5) === seq(5) && seq(5) !== seq(6), 'sampling is reproducible by seed');
  t.ok(/^Token 4 \(p = /.test(DL.sample(row, { method: 'greedy' }).say), 'each sampled token has a sentence');
}

// ---- BPE ----
{
  const text = S.Data.proseText();
  const a = DL.bpe.train(text, 200, { sample: 'The night market opens at six.' }), b = DL.bpe.train(text, 200);
  t.eq(a.merges, b.merges, 'BPE merges are deterministic');
  t.ok(a.trace[0].kind === 'start' && a.trace.filter((s) => s.kind === 'merge').length === 200, 'one trace step per merge');
  const m1 = a.trace[1];
  t.ok(/^Merge 1: "/.test(m1.say) && m1.count > 100 && m1.tokens < a.trace[0].tokens, 'the first merge is the most frequent pair: ' + m1.say);
  t.ok(a.trace.filter((s) => s.sample).length >= 10 && a.trace.at(-1).sample.join('') === 'The night market opens at six.', 'the sample sentence is split along the way');
  const tok = a.tokenizer, ids = tok.encode(text);
  t.ok(tok.decode(ids) === text, 'BPE encode then decode is the identity on the corpus');
  t.eq(ids.length, a.trace.at(-1).tokens, 'encoding reproduces the training segmentation (' + ids.length + ' tokens)');
  const fresh = 'A quiet sailor counted 27 lanterns, then left.'.split('').filter((c) => a.base.includes(c)).join('');
  t.ok(fresh.length > 40 && tok.decode(tok.encode(fresh)) === fresh, 'and on new text');
  const re = DL.tokenizer(JSON.parse(JSON.stringify(tok.toJSON())));
  t.eq(re.encode(fresh), tok.encode(fresh), 'a BPE tokenizer survives JSON');
  t.ok(tok.decode(tok.encode('naïve ☃')).includes('�') || tok.decode(tok.encode('naïve ☃')) === 'naïve ☃', 'unknown characters become <unk>');
  const ch = DL.charTokenizer(text), w = DL.wordTokenizer(['Good food, good staff.', 'Bad food.']);
  t.ok(ch.decode(ch.encode(text)) === text && w.encode('good food')[0] === w.index.get('good') && w.vocab[0] === '<pad>', 'character and word tokenizers');
  console.log('  BPE: ' + a.trace.at(-1).say);
}

// ---- word2vec ----
{
  const t0 = performance.now(), w = new DL.Word2Vec(null, { seed: 2, snapshotWords: 30 }).fit(), ms = performance.now() - t0;
  t.ok(w.losses.length === 15 && w.losses[14] < w.losses[0] * 0.8, 'word2vec loss falls: ' + w.losses.map((x) => x.toFixed(3)).join(' → '));
  const w2 = new DL.Word2Vec(null, { epochs: 1, seed: 2 }).fit(), w3 = new DL.Word2Vec(null, { epochs: 1, seed: 2 }).fit();
  t.ok(close(w2.W, w3.W, 0), 'word2vec is deterministic by seed');
  const nn = w.nearest('tea', 5), an = w.analogy('he', 'his', 'she');
  t.ok(nn.length === 5 && nn.every((x) => x.label !== 'tea') && /^Nearest to "tea"/.test(nn.say), 'nearest neighbours: ' + nn.say);
  t.ok(an.answers.length === 3 && !['he', 'his', 'she'].includes(an.answers[0].label), 'analogy: ' + an.say);
  const pr = w.project(['he', 'she', 'his', 'her', 'the']);
  t.ok(pr.points.length === 5 && pr.points[0].length === 2 && pr.explained[0] >= pr.explained[1], '2-D projection by principal components');
  t.ok(w.snapshots.length === 15 && w.snapshots[0].points.length === 30, 'a 2-D snapshot of the frequent words each epoch');
  // the gradient of one pair against finite differences
  const g = T.rng(4), vec = (d) => Float64Array.from({ length: d }, () => g.normal(0, 0.5));
  const v = vec(6), u = vec(6), negs = [vec(6), vec(6)], r = DL.w2vPair(v, u, negs);
  let worst = 0;
  for (const [arr, grad] of [[v, r.gv], [u, r.gu], [negs[0], r.gneg[0]], [negs[1], r.gneg[1]]]) for (let j = 0; j < 6; j++) {
    const o = arr[j]; arr[j] = o + 1e-6; const lp = DL.w2vPair(v, u, negs).loss; arr[j] = o - 1e-6; const lm = DL.w2vPair(v, u, negs).loss; arr[j] = o;
    worst = Math.max(worst, Math.abs((lp - lm) / 2e-6 - grad[j]));
  }
  t.ok(worst < 1e-6, 'word2vec pair gradients match finite differences (worst ' + worst.toExponential(2) + ')');
  console.log(`  word2vec: ${w.vocab.length} words × 16 in ${(ms / 1000).toFixed(2)} s; ${nn.say} ${an.say}`);
}

// ---- gradient checks on small versions of every model ----
{
  const ids = (B, n, V, seed) => { const g = T.rng(seed); return T.tensor(Array.from({ length: B }, () => Array.from({ length: n }, () => 1 + g.int(V - 1)))); };
  const checks = [
    ['MLP', () => { const net = DL.build('mlp', { inputs: 3, hidden: [4], outputs: 3, act: 'tanh' }), x = T.normal([5, 3], 0, 1, 2); return { net, loss: () => T.crossEntropy(net.forward(x), [0, 1, 2, 1, 0]) }; }],
    ['MLP regression', () => { const net = DL.build('mlp', { inputs: 2, hidden: [3], outputs: 1, regress: true, act: 'tanh' }), x = T.normal([4, 2], 0, 1, 3); return { net, loss: () => net.lossOn({ x, y: T.tensor([[1], [0], [-1], [2]]) }).loss }; }],
    ['CNN', () => { const net = DL.build('cnn', { size: 8, filters: [2, 3], kernel: 3, hidden: [4], classes: 3 }), x = T.normal([2, 1, 8, 8], 0, 1, 4); return { net, loss: () => T.crossEntropy(net.forward(x), [0, 2]) }; }],
    ['CNN with batch norm', () => { const net = DL.build('cnn', { size: 8, filters: [2, 2], kernel: 3, hidden: [3], classes: 2, batchNorm: true }), x = T.normal([3, 1, 8, 8], 0, 1, 5); return { net, loss: () => T.crossEntropy(net.forward(x), [0, 1, 1]) }; }],
    ['RNN', () => { const net = DL.build('rnn', { cell: 'rnn', vocab: 5, embed: 3, hidden: 4, lm: true }), x = ids(2, 4, 5, 6); return { net, loss: () => net.lossOn({ x, y: [[1, 2, 3, 4], [4, 3, 2, 1]] }).loss }; }],
    ['LSTM', () => { const net = DL.build('rnn', { cell: 'lstm', vocab: 5, embed: 3, hidden: 4, lm: true }), x = ids(2, 4, 5, 7); return { net, loss: () => net.lossOn({ x, y: [[1, 2, 3, 4], [4, 3, 2, 1]] }).loss }; }],
    ['GRU (last step)', () => { const net = DL.build('rnn', { cell: 'gru', vocab: 5, embed: 3, hidden: 4, outputs: 2, mode: 'last' }), x = ids(3, 4, 5, 8); return { net, loss: () => net.lossOn({ x, y: [0, 1, 1] }).loss }; }],
    ['transformer encoder', () => { const net = DL.build('encoder', { vocab: 6, d: 8, heads: 2, layers: 1, maxLen: 5, classes: 2 }), x = T.tensor([[1, 2, 3, 0, 0], [4, 5, 1, 2, 3]]); return { net, loss: () => net.lossOn({ x, y: [0, 1] }).loss }; }],
    ['GPT (tied)', () => { const net = DL.build('gpt', { vocab: 6, d: 8, heads: 2, layers: 2, context: 5, initStd: 0.5 }), x = ids(2, 5, 6, 9); return { net, loss: () => net.lossOn({ x, y: [[1, 2, 3, 4, 5], [5, 4, -1, 2, 1]] }).loss }; }],
  ];
  for (const [name, make] of checks) {
    const r = DL.gradCheck(make, { per: 8 });
    t.ok(r.ok, 'gradient check ' + name + ': worst relative error ' + r.worst.toExponential(2) + ' at ' + r.where);
  }
  console.log('  gradient checks: ' + checks.length + ' small models, every parameter tensor against central differences');
}

// ---- determinism and save / load for every kind ----
{
  const run = (task, cfg) => { const { trainer } = DL.setup({ task, seed: 4, config: cfg, taskOpts: { n: 200, val: 40 } }); trainer.fit({ steps: 6 }); return trainer.history.map((h) => h.loss); };
  for (const [task, cfg] of [['digits', { size: 16 }], ['counting', {}], ['reverse', {}], ['majority', {}]]) {
    const a = run(task, cfg), b = run(task, cfg);
    t.eq(a, b, 'training on ' + task + ' is deterministic by seed');
  }
  const { trainer: t1 } = DL.setup({ task: 'counting', seed: 1 }), { trainer: t2 } = DL.setup({ task: 'counting', seed: 2 });
  t1.fit({ steps: 3 }); t2.fit({ steps: 3 });
  t.ok(t1.history[2].loss !== t2.history[2].loss, 'a different seed gives a different run');
  const samples = {
    mlp: [{ inputs: 2, hidden: [5], outputs: 2 }, T.normal([3, 2], 0, 1, 1)],
    cnn: [{ size: 16, classes: 4, batchNorm: true }, T.uniform([2, 1, 16, 16], 0, 1, 2)],
    rnn: [{ cell: 'gru', vocab: 9, embed: 4, hidden: 6, lm: true, tokenizer: DL.charTokenizer('abcdefgh ').toJSON() }, T.tensor([[1, 2, 3]])],
    encoder: [{ vocab: 9, d: 8, heads: 2, layers: 2, maxLen: 4, classes: 3 }, T.tensor([[1, 2, 3, 0]])],
    gpt: [{ vocab: 9, d: 8, heads: 2, layers: 2, context: 6, tokenizer: DL.charTokenizer('abcdefgh ').toJSON() }, T.tensor([[1, 2, 3]])],
  };
  for (const [kind, [cfg, x]] of Object.entries(samples)) {
    const net = DL.build(kind, Object.assign({ seed: 3 }, cfg));
    for (const p of net.parameters()) for (let i = 0; i < p.size; i++) p.data[i] += 0.01 * Math.sin(i + p.size);
    const json = JSON.stringify(DL.save(net)), back = DL.load(json);
    t.ok(close(back.predict(x).values(), net.predict(x).values(), 0) && back.numParams() === net.numParams(), kind + ': save / load round trip (' + (json.length / 1024).toFixed(1) + ' kB)');
  }
  const gpt = DL.load(DL.save(DL.build('gpt', samples.gpt[0])));
  t.ok(gpt.tokenizer && gpt.tokenizer.decode(gpt.tokenizer.encode('bad cafe')) === 'bad cafe', 'the tokenizer travels with a saved model');
  const { net, trainer } = DL.setup({ task: 'counting', seed: 5, train: { checkpointEvery: 5 } });
  trainer.fit({ steps: 5 });
  const ck = JSON.parse(JSON.stringify(trainer.checkpoints[0])), x = trainer.task.val.get([0, 1]).x, at5 = net.predict(x).values();
  trainer.fit({ steps: 5 });
  trainer.restore(ck);
  t.ok(close(net.predict(x).values(), at5, 0) && trainer.stepNo === 5, 'a trainer restores a checkpoint');
}

// ---- the streaming runner (the same code that runs in a Web Worker) ----
await new Promise((done) => {
  const msgs = [];
  const ctl = DL.train({ task: 'counting', seed: 2, train: { evalEvery: 20 } }, { forcePage: true, maxSteps: 60, snapshotEvery: 30, snapshot: ['gate.f', 'hidden'], checkpointEvery: 60, chunkMs: 20 }, (m) => {
    msgs.push(m);
    if (m.type !== 'done' && m.type !== 'error') return;
    const prog = msgs.filter((x) => x.type === 'progress'), snaps = msgs.filter((x) => x.type === 'snapshot'), cks = msgs.filter((x) => x.type === 'checkpoint');
    t.ok(ctl.mode === 'page' && m.type === 'done' && prog.length === 60, 'DL.train streams one progress message per step (' + prog.length + ')');
    t.ok(prog[19].metrics.valLoss > 0 && prog[19].metrics.accuracy >= 0, 'progress carries training and validation metrics');
    t.ok(snaps.length === 2 && snaps[0].tensors['gate.f'].shape.join() === '1,8,24' && snaps[0].tensors.hidden.data instanceof Float32Array, 'snapshots of gates and hidden states by name');
    t.ok(cks.length === 1 && cks[0].state.format === 'nooral-dl/1' && DL.load(cks[0].state).numParams() === 2721, 'checkpoints are loadable saved models');
    done();
  });
});

await new Promise((done) => {
  const msgs = [];
  DL.train({ epochs: 2, seed: 3 }, { forcePage: true, snapshotEvery: 10, snapshot: ['embeddings'] }, (m) => {
    msgs.push(m);
    if (m.type !== 'done' && m.type !== 'error') return;
    const prog = msgs.filter((x) => x.type === 'progress'), snap = msgs.find((x) => x.type === 'snapshot');
    t.ok(m.reason === 'trained 2 epochs' && prog.length > 10 && prog.at(-1).loss > 0, 'the word2vec job streams its loss (' + prog.length + ' chunks, ' + (m.reason || m.message) + ')');
    t.ok(snap && snap.tensors.embeddings.shape[1] === 16, 'and snapshots its embedding matrix');
    done();
  }, 'word2vec');
});

// ---- sizes and speeds of the language models (skipped with --quick) ----
const quick = process.argv.includes('--quick');
if (!quick) timed('CNN on 28×28 digits, augmented', 90000, () => {
  const { net, trainer } = DL.setup({ task: 'digits', taskOpts: { size: 28, n: 3000, val: 600, augment: { shift: 1.5, rotate: 0.15, noise: 0.05 } } });
  trainer.fit({ steps: 150 });
  const v = trainer.evaluate('val', 600);
  t.ok(v.accuracy >= 0.95, 'CNN 28×28 digits with shift, rotation and noise: val accuracy ' + v.accuracy);
  return { label: 'val accuracy ' + v.accuracy.toFixed(3) + ', ' + net.numParams() + ' parameters', params: net.numParams() };
});
if (!quick) timed('CNN on 16×16 shapes', 30000, () => {
  const { net, trainer } = DL.setup({ task: 'shapes', taskOpts: { size: 16 } });
  trainer.fit({ steps: 400 });
  const v = trainer.evaluate('val');
  t.ok(v.accuracy >= 0.93, 'CNN shapes: val accuracy ' + v.accuracy);
  return { label: 'val accuracy ' + v.accuracy.toFixed(3), params: net.numParams() };
});
if (!quick) timed('GPT (BPE tokens) on the corpus', 150000, () => {
  const { net, trainer } = DL.setup({ task: 'text', taskOpts: { tokenizer: 'bpe', merges: 300, context: 24, corpus: 'short' }, config: { d: 64, heads: 4, layers: 2 }, train: { batch: 16, lr: 1e-2, steps: 400, warmup: 20 } });
  const first = trainer.evaluate('val', 128).loss;
  trainer.fit({ steps: 400 });
  const v = trainer.evaluate('val', 256), tr = trainer.evaluate('train', 256);
  const vocab = new Set(DL.words(S.Data.proseText()));
  const words = [], texts = [];
  for (let s = 1; s <= 3; s++) { const out = net.complete('The ', 160, { temperature: 0.7, topK: 10 }, s).text; texts.push(out); words.push(...DL.words(out)); }
  const real = words.filter((w) => vocab.has(w)).length / words.length;
  t.ok(tr.loss < 2.5, 'GPT: training loss ' + first.toFixed(2) + ' → ' + tr.loss.toFixed(2) + ' nats per token (target 2.5)');
  t.ok(real >= 0.6, 'GPT samples are word-like: ' + (100 * real).toFixed(0) + '% of generated words occur in the corpus');
  console.log('  sample: ' + JSON.stringify(texts[0].slice(0, 120)));
  const nb = DL.neighbours(net, net.tokenizer.vocab[net.tokenizer.encode(' the')[0]], 5);
  t.ok(nb.length === 5 && nb.say.startsWith('Nearest to'), 'embedding neighbours of a token: ' + nb.say);
  return { label: 'train loss ' + tr.loss.toFixed(2) + ', val ' + v.loss.toFixed(2) + ' (perplexity ' + v.perplexity.toFixed(1) + '), ' + (100 * real).toFixed(0) + '% real words', params: net.numParams() };
});
if (!quick) {
  const sizes = [];
  for (const [label, cfg, tokz] of [['GPT default, characters', {}, 'char'], ['GPT default, BPE 300 merges', {}, 'bpe'], ['GPT 6 layers, d 256', { d: 256, heads: 8, layers: 6 }, 'char']]) {
    const { net, trainer } = DL.setup({ task: 'text', taskOpts: { tokenizer: tokz, context: 128, corpus: 'short' }, config: cfg, train: { batch: 8 } });
    trainer.step();
    const c0 = cpuMs(); trainer.step(); const ms = cpuMs() - c0;
    sizes.push(label + ': ' + (net.numParams() / 1e6).toFixed(2) + 'M parameters, ' + (ms / 1000).toFixed(2) + ' s of CPU per step of 8×128 tokens');
    t.ok(net.numParams() >= 1e6 && net.numParams() <= 5e6, label + ' has 1–5M parameters (' + net.numParams() + ')');
  }
  for (const s of sizes) console.log('  ' + s);
}
console.log('  ' + report.map((r) => r.name + ': ' + r.params + ' params, ' + r.seconds.toFixed(1) + ' s').join('\n  '));

// ---- the text lab's sizes, the long corpus, and the larger review set ----
{
  const sizes = {};
  for (const k of ['webgpu', 'cpu']) {
    const task = DL.task('text', { preset: k }), net = DL.build('gpt', Object.assign({ seed: 1 }, task.model));
    sizes[k] = net.numParams();
    t.ok(task.model.context === DL.gptPresets[k].context && task.fit.steps === DL.gptPresets[k].fit.steps, 'preset ' + k + ' sets the context and the steps');
  }
  t.ok(sizes.webgpu >= 1e6 && sizes.webgpu <= 5e6, 'the WebGPU preset has 1–5M parameters (' + sizes.webgpu + ')');
  t.ok(sizes.cpu >= 2e5 && sizes.cpu <= 5e5, 'the CPU preset has 0.2–0.5M parameters (' + sizes.cpu + ')');
  const long = DL.task('text', {}), short = DL.task('text', { corpus: 'short' });
  t.ok(long.ids.length > 7 * short.ids.length, 'the text task trains on the long corpus (' + long.ids.length + ' characters, the short one ' + short.ids.length + ')');
  t.ok(new DL.Trainer(DL.build('gpt', Object.assign({ seed: 1 }, DL.task('text', { preset: 'cpu' }).model)), long).fast === true && new DL.Trainer(DL.build('mlp', { inputs: 2, hidden: [8], outputs: 2 }), DL.task('moons')).fast === false, 'big models use the fast kernel, small ones the exact one');
}
timed('GRU sentiment on 600 reviews with turns', 30000, () => {
  const { net, trainer } = DL.setup({ task: 'reviews', taskOpts: { set: 'opinions' }, train: { batch: 32 } });
  trainer.fit({ steps: 300 });
  const v = trainer.evaluate('val');
  t.ok(v.accuracy >= 0.8, 'GRU on the opinions set: held-out accuracy ' + v.accuracy.toFixed(3) + ' on ' + trainer.task.val.n + ' reviews');
  return { label: 'held-out accuracy ' + v.accuracy.toFixed(3) + ' (' + trainer.task.val.n + ' reviews)', params: net.numParams() };
});

// ---- in a browser: a real Web Worker, and WebGPU where offered ----
if (process.argv.includes('--browser')) {
  const { launch } = await import('../tools/cdp.mjs');
  const b = await launch();
  try {
    await Promise.race([b.goto('dev.html'), b.sleep(10000)]);
    await b.waitFor('window.Nooral && Nooral.DL && Nooral.T && Nooral.T.worker', 10000);
    const wk = await b.eval(`(async () => {
      const msgs = [], ctl = Nooral.DL.train({ task: 'reverse', train: { evalEvery: 50 } }, { maxSteps: 120, snapshotEvery: 60, snapshot: ['block0.attn'] }, (m) => msgs.push(m));
      const t0 = performance.now();
      await new Promise((res) => { const tick = () => (msgs.some((m) => m.type === 'done' || m.type === 'error') || performance.now() - t0 > 60000 ? res() : setTimeout(tick, 20)); tick(); });
      const prog = msgs.filter((m) => m.type === 'progress'), snaps = msgs.filter((m) => m.type === 'snapshot');
      return { mode: ctl.mode, steps: prog.length, last: prog.length && prog.at(-1).loss, valAcc: prog.length >= 100 && prog[99].metrics.valAccuracy, snap: snaps.length && snaps[0].tensors['block0.attn'].shape, error: (msgs.find((m) => m.type === 'error') || {}).message, s: (performance.now() - t0) / 1000 };
    })()`);
    t.ok(wk.mode === 'worker' && wk.steps === 120 && !wk.error, 'DL.train runs in a Web Worker (' + wk.mode + ', ' + wk.steps + ' steps' + (wk.error ? ', ' + wk.error : '') + ')');
    t.ok(wk.snap && wk.snap.join() === '1,2,10,10', 'attention snapshots come back from the worker');
    console.log('  worker: reversal, 120 steps in ' + wk.s.toFixed(1) + ' s, loss ' + (+wk.last).toFixed(3) + ', val token accuracy ' + wk.valAcc);
    const gpu = await b.eval(`(async () => { const B = Nooral.T.backends, be = await B.select(); if (be.name !== 'webgpu') return { name: be.name, log: B.log };
      const T = Nooral.T, g = T.rng(1), rows = [];
      for (const [m, k, n] of [[1024, 192, 768], [1024, 768, 192], [1024, 192, 58]]) { const a = T.normal([m, k], 0, 1, g), w = T.normal([k, n], 0, 1, g);
        const cpu = await B.time(async () => { await B.cpu.matmul(a, w); }, { ms: 300 }), wg = await B.time(async () => { await be.matmul(a, w); }, { ms: 300 });
        rows.push(m + '×' + k + ' @ ' + k + '×' + n + ': cpu ' + cpu.toFixed(1) + ' ms, webgpu ' + wg.toFixed(1) + ' ms (with transfers)'); }
      return { name: be.name, rows }; })()`);
    if (gpu.name !== 'webgpu') console.log('  SKIP WebGPU: ' + (gpu.log || []).join('; '));
    else for (const r of gpu.rows) console.log('  ' + r);
    // training on the GPU: the same losses and gradients as the CPU, the worker picks it, and its speed
    const gt = await b.eval(`(async () => {
      const DL = Nooral.DL, T = Nooral.T, out = {};
      if (!(await T.gpu.init())) return { skip: T.gpu.why };
      const mk = (task, o) => DL.setup(Object.assign({ task }, o));
      const small = () => mk('text', { taskOpts: { context: 32, corpus: 'short' }, config: { d: 64, heads: 4, layers: 2 }, train: { batch: 4, clip: 1, steps: 100, warmup: 2 } });
      const A = small(), B = small(), batch = A.task.train.get([0, 5, 9, 13], T.rng(3));
      A.net.zeroGrad(); const r = A.net.lossOn(batch); r.loss.backward();
      const s = await DL.gpuSession(B.net), g = await s.step(batch, { lr: 0, t: 1, weightDecay: 0 }), gg = await s.grads();
      let top = 0; for (const [, p] of A.net.namedParameters()) for (const v of p.grad.values()) top = Math.max(top, Math.abs(v));
      out.grad = 0; for (const [n, p] of A.net.namedParameters()) { const c = p.grad.values(), d = gg[n].values(); for (let i = 0; i < c.length; i++) out.grad = Math.max(out.grad, Math.abs(c[i] - d[i]) / top); }
      out.loss = Math.abs(r.loss.item() - g.loss) / r.loss.item(); out.correct = [r.correct, g.correct];
      out.attn = T.backends.maxErr(await s.read('block1.attn'), A.net.outputs['block1.attn']);
      out.params = r.loss.item() === g.loss || true; s.free();
      const C = small(), D = small(), lc = [], ld = [];
      for (let i = 0; i < 6; i++) lc.push(C.trainer.step().loss);
      for (let i = 0; i < 6; i++) ld.push((await D.trainer.stepAsync()).loss);
      out.train = Math.max(...lc.map((x, i) => Math.abs(x - ld[i]) / x)); out.first = lc[0]; out.last = [lc[5], ld[5]];
      const E = mk('reverse', { train: { batch: 16 } }), F = mk('reverse', { train: { batch: 16 } }), le = [], lf = [];
      for (let i = 0; i < 5; i++) { le.push(E.trainer.step().loss); lf.push((await F.trainer.stepAsync()).loss); }
      out.toy = Math.max(...le.map((x, i) => Math.abs(x - lf[i]) / x));
      const ve = E.trainer.evaluate('val', 64), vf = await F.trainer.evaluateAsync('val', 64); out.evalLoss = Math.abs(ve.loss - vf.loss) / ve.loss;
      const msgs = [], ctl = DL.train({ task: 'text', taskOpts: { preset: 'auto' }, train: { steps: 40 } }, { maxSteps: 40, snapshotEvery: 20, snapshot: ['block0.attn'] }, (m) => msgs.push(m));
      const t0 = performance.now();
      await new Promise((res) => { const tick = () => (msgs.some((m) => m.type === 'done' || m.type === 'error') || performance.now() - t0 > 90000 ? res() : setTimeout(tick, 20)); tick(); });
      const prog = msgs.filter((m) => m.type === 'progress'), snap = msgs.find((m) => m.type === 'snapshot');
      out.worker = { mode: ctl.mode, steps: prog.length, backend: prog.length && prog[0].backend, info: prog.length && prog[0].info, first: prog.length && prog[0].loss, last: prog.length && prog.at(-1).loss, snap: snap && snap.tensors['block0.attn'].shape, error: (msgs.find((m) => m.type === 'error') || {}).message, s: (performance.now() - t0) / 1000 };
      for (const [k, batch] of [['webgpu', 16], ['cpu', 16]]) {
        const P = mk('text', { taskOpts: { preset: k } }); await P.trainer.stepAsync(); await P.trainer.stepAsync();
        const t1 = performance.now(); for (let i = 0; i < 20; i++) await P.trainer.stepAsync(); out[k + 'Ms'] = (performance.now() - t1) / 20; out[k + 'Params'] = P.net.numParams();
      }
      return out; })()`);
    if (gt.skip) console.log('  SKIP training on WebGPU: ' + gt.skip);
    else {
      t.ok(gt.loss < 1e-5 && gt.correct[0] === gt.correct[1], 'GPT on WebGPU: the same loss as the CPU (relative difference ' + gt.loss.toExponential(1) + ', ' + gt.correct[0] + ' right on both)');
      t.ok(gt.grad < 1e-4, 'and the same gradients for every parameter (largest difference ' + gt.grad.toExponential(1) + ' of the largest gradient)');
      t.ok(gt.attn < 1e-5, 'and the same attention weights (' + gt.attn.toExponential(1) + ')');
      t.ok(gt.train < 1e-4, 'six AdamW steps with clipping and warm-up: the losses agree within ' + gt.train.toExponential(1) + ' (' + gt.first.toFixed(3) + ' → ' + gt.last[0].toFixed(4) + ' on the CPU, ' + gt.last[1].toFixed(4) + ' on the GPU)');
      t.ok(gt.toy < 1e-4 && gt.evalLoss < 1e-5, 'the reversal task with its ignored targets agrees too (' + gt.toy.toExponential(1) + ', evaluation ' + gt.evalLoss.toExponential(1) + ')');
      const w = gt.worker;
      t.ok(w.mode === 'worker' && w.backend === 'webgpu' && w.info && w.info.preset === 'webgpu' && w.steps === 40 && !w.error, 'DL.train picks WebGPU in its worker with the text preset ' + (w.info && w.info.preset) + ' (' + w.steps + ' steps' + (w.error ? ', ' + w.error : '') + ')');
      t.ok(w.last < w.first && w.snap && w.snap.join() === '1,6,128,128', 'its loss falls (' + (+w.first).toFixed(3) + ' → ' + (+w.last).toFixed(3) + ') and attention snapshots come back from the GPU');
      console.log('  WebGPU training steps: preset webgpu (' + gt.webgpuParams + ' parameters, 16×128) ' + gt.webgpuMs.toFixed(1) + ' ms, preset cpu (' + gt.cpuParams + ', 16×64) ' + gt.cpuMs.toFixed(1) + ' ms; worker 40 steps in ' + w.s.toFixed(1) + ' s');
    }
  } finally { await b.close(); }
}
t.done();
