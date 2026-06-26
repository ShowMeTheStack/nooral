// The tensor engine in headless Chrome: the WebGPU backend against the CPU on every kernel, automatic backend
// selection, the training runner in a real Web Worker (pause, step, resume, snapshots), and matmul / conv speed on
// the CPU and on WebGPU. Where the browser has no WebGPU the GPU checks are reported as skipped, not passed.
import { launch } from '../tools/cdp.mjs';
import { harness } from './load.mjs';
const t = harness('tensor-gpu');
const b = await launch();
const guard = setTimeout(async () => { console.log('FAIL the browser test ran out of time'); await b.close(); process.exit(1); }, 240000);
try {
  await Promise.race([b.goto('dev.html'), b.sleep(10000)]);
  await b.waitFor('window.Nooral && Nooral.T && Nooral.T.backends && Nooral.T.backends.webgpu', 10000);
  const sel = await b.eval(`(async () => { const be = await Nooral.T.backends.select(); return { name: be.name, info: be.info, log: Nooral.T.backends.log }; })()`);
  console.log('  backend: ' + sel.log.join('; '));
  if (sel.name !== 'webgpu') {
    console.log('  SKIP WebGPU is not available in this headless Chrome; only the fallback and the worker are checked.');
    t.ok(sel.name === 'cpu', 'select() falls back to the CPU');
  } else {
    const cmp = await b.eval(`Nooral.T.backends.compare(Nooral.T.backends.webgpu, { tol: 1e-4 })`);
    for (const c of cmp) t.ok(c.ok, 'WebGPU ' + c.op + ': max relative error ' + c.maxErr + (c.error ? ' (' + c.error + ')' : ''));
    const worst = Math.max(...cmp.map((c) => c.maxErr));
    console.log(`  WebGPU agrees with the CPU on ${cmp.filter((c) => c.ok).length} of ${cmp.length} kernels (worst relative error ${worst.toExponential(2)})`);
    // larger shapes than one tile, odd sizes, many rows for the softmax
    const big = await b.eval(`(async () => { const T = Nooral.T, W = T.backends.webgpu, C = T.backends.cpu, g = T.rng(5), r = (s) => T.normal(s, 0, 1, g), out = [];
      const a = r([130, 257]), bb = r([257, 70]); out.push(['matmul 130×257 @ 257×70', T.backends.maxErr(await W.matmul(a, bb), await C.matmul(a, bb))]);
      const s = r([70000, 3]); out.push(['softmax over 70000 rows', T.backends.maxErr(await W.softmax(s), await C.softmax(s))]);
      const u = r([300000]); out.push(['relu of 300000', T.backends.maxErr(await W.unary('relu', u), await C.unary('relu', u))]);
      const q = r([2, 4, 33, 16]), k = r([2, 4, 40, 16]), v = r([2, 4, 40, 8]); const wa = await W.attention(q, k, v, {}), ca = await C.attention(q, k, v, {});
      out.push(['attention 33 queries × 40 keys', Math.max(T.backends.maxErr(wa, ca), T.backends.maxErr(wa.weights, ca.weights))]);
      return out; })()`);
    for (const [name, err] of big) {
      t.ok(err < 1e-4, 'WebGPU ' + name + ': max relative error ' + err);
    }
    const conv = await b.eval(`(async () => { const T = Nooral.T, W = T.backends.webgpu, g = T.rng(6), r = (s) => T.normal(s, 0, 1, g);
      const x = r([3, 5, 17, 13]), w = r([7, 5, 5, 3]), bias = r([7]), ref = T.noGrad(() => T.conv2d(x, w, bias, { padding: [2, 1], stride: [2, 1], dilation: [1, 2] }));
      const res = []; for (const mode of ['im2col', 'direct']) res.push(T.backends.maxErr(await W.conv2d(x, w, bias, { padding: [2, 1], stride: [2, 1], dilation: [1, 2], mode }), ref)); return res; })()`);
    t.ok(conv.every((e) => e < 1e-4), 'WebGPU conv2d with a 5×3 kernel, stride, padding and dilation (im2col and direct): ' + conv.join(', '));
  }
  // the GPU autodiff tape: every training op and its gradients against the CPU engine
  const tape = await b.eval(`(async () => {
    const T = Nooral.T, G = T.gpu, out = [];
    if (!(await G.init())) return { skip: G.why };
    const g = T.rng(9), r = (s, sd = 1) => T.normal(s, 0, sd, g);
    // run f on the CPU (params) and on the GPU (G.param copies), seed both with the same upstream gradient
    const check = async (name, inputs, cpu, gpu) => {
      const ps = inputs.map((x) => T.param(x.clone())), y = cpu(...ps), R = r(y.shape);
      T.sum(T.mul(y, R)).backward();
      const gs = inputs.map((x) => G.param(x)), gy = gpu(...gs);
      G.backward(gy, G.upload(R.values(), y.shape));
      const read = await G.readMany([gy, ...gs.map((q) => q.grad)]);
      let e = T.backends.maxErr(read[0], y);
      ps.forEach((p, i) => { e = Math.max(e, T.backends.maxErr(T.make(read[i + 1].data, p.shape), p.grad)); });
      out.push([name, e]); G.endStep();
    };
    await check('matmul', [r([37, 53]), r([53, 29])], (a, b) => T.matmul(a, b), (a, b) => G.matmul(a, b));
    await check('matmul by a transpose', [r([37, 53]), r([29, 53])], (a, b) => T.matmul(a, T.transpose(b)), (a, b) => G.matmul(a, b, { tb: 1 }));
    await check('matmul of a transpose', [r([53, 37]), r([53, 29])], (a, b) => T.matmul(T.transpose(a), b), (a, b) => G.matmul(a, b, { ta: 1 }));
    await check('batched matmul', [r([6, 17, 40]), r([6, 40, 9])], (a, b) => T.matmul(a, b), (a, b) => G.matmul(a, b));
    await check('add with a repeating row', [r([7, 5, 33]), r([33])], (a, b) => T.add(a, b), (a, b) => G.add(a, b));
    await check('gelu', [r([3000], 2)], (a) => T.gelu(a), (a) => G.gelu(a));
    await check('layer norm', [r([70, 48]), T.add(r([48], 0.3), 1), r([48], 0.3)], (x, a, b) => T.layerNorm(x, a, b), (x, a, b) => G.layerNorm(x, a, b));
    await check('permute', [r([2, 5, 3, 7])], (x) => T.permute(x, [0, 2, 1, 3]), (x) => G.permute(x, [0, 2, 1, 3]));
    await check('causal softmax of scaled scores', [r([6, 11, 11], 3)], (x) => T.softmax(T.maskedFill(T.scale(x, 0.3), T.eq(T.causalMask(11), 0), -1e9), -1), (x) => G.softmaxRows(x, { causal: true, scale: 0.3 }));
    const ids = Array.from({ length: 40 }, () => g.int(13));
    await check('embedding', [r([13, 8])], (W) => T.embedding(W, ids), (W) => G.embedding(W, ids));
    const tg = Array.from({ length: 30 }, (_, i) => (i % 7 === 3 ? -1 : g.int(17)));
    await check('cross-entropy with ignored targets', [r([30, 17], 2)], (z) => T.crossEntropy(z, tg, { ignoreIndex: -1 }), (z) => G.crossEntropy(z, tg));
    // AdamW with clipping: three steps of the same gradients on both sides
    const w = r([300]), grads = [r([300], 3), r([300], 3), r([300], 3)], pc = T.param(w.clone()), pg = G.param(w);
    const opt = T.optim.make('adamw', [pc], { lr: 0.01, weightDecay: 0.1 }), ga = new G.Adam([pg], { decoupled: true });
    for (let k = 0; k < 3; k++) {
      pc.grad = grads[k].clone(); T.clipGradNorm([pc], 2); opt.step();
      pg.grad = G.upload(grads[k].values(), [300]); ga.step({ lr: 0.01, t: k + 1, weightDecay: 0.1, clip: 2 });
    }
    out.push(['AdamW with clipping, three steps', T.backends.maxErr(await G.read(pg), pc)]);
    G.endStep();
    return { rows: out };
  })()`);
  if (tape.skip) console.log('  SKIP the GPU tape: ' + tape.skip);
  else {
    for (const [name, e] of tape.rows) t.ok(e < 2e-5, 'GPU tape ' + name + ': value and gradients within ' + e.toExponential(1) + ' of the CPU');
    console.log('  the GPU tape agrees with the CPU on ' + tape.rows.length + ' ops and AdamW (worst relative error ' + Math.max(...tape.rows.map((x) => x[1])).toExponential(2) + ')');
  }
  // the training runner in a Web Worker
  const wk = await b.eval(`(async () => {
    const T = Nooral.T, msgs = [], g = T.rng(1), X = [], y = [];
    for (let i = 0; i < 64; i++) { const a = g.next() > 0.5 ? 1 : 0, c = g.next() > 0.5 ? 1 : 0; X.push([a + g.normal(0, 0.1), c + g.normal(0, 0.1)]); y.push(a ^ c); }
    const ctl = T.train('mlp', { X, y, sizes: [8], act: 'tanh', seed: 3, lr: 0.05 }, { snapshotEvery: 50, snapshot: ['layers.0.W'], checkpointEvery: 100, paused: true, chunkMs: 10 }, (m) => msgs.push(m));
    const wait = (f, ms = 8000) => new Promise((res, rej) => { const t0 = performance.now(); const tick = () => f() ? res() : performance.now() - t0 > ms ? rej(new Error('waited too long')) : setTimeout(tick, 10); tick(); });
    ctl.step(5);
    await wait(() => msgs.filter((m) => m.type === 'progress').length >= 5);
    const afterStep = msgs.filter((m) => m.type === 'progress').length;
    ctl.resume();
    await wait(() => msgs.filter((m) => m.type === 'progress').length >= 60);
    ctl.pause();
    await wait(() => msgs.some((m) => m.type === 'state' && m.state === 'paused' && m.step > 5));
    const atPause = msgs.filter((m) => m.type === 'progress').length;
    await new Promise((r) => setTimeout(r, 150));
    const stillPaused = msgs.filter((m) => m.type === 'progress').length;
    ctl.resume();
    await wait(() => msgs.filter((m) => m.type === 'progress').length >= 400);
    ctl.stop();
    await wait(() => msgs.some((m) => m.type === 'done'));
    const prog = msgs.filter((m) => m.type === 'progress'), snaps = msgs.filter((m) => m.type === 'snapshot'), cks = msgs.filter((m) => m.type === 'checkpoint');
    return { mode: ctl.mode, afterStep, atPause, stillPaused, steps: prog.length, last: prog.at(-1).loss, snaps: snaps.length, snapKeys: Object.keys(snaps[0].tensors), snapShape: snaps[0].tensors['layers.0.W'].shape, cks: cks.length, ckKeys: Object.keys(cks[0].state).length };
  })()`);
  t.ok(wk.mode === 'worker', 'train() runs in a Web Worker on the page (mode ' + wk.mode + ')');
  t.ok(wk.afterStep === 5, 'step(5) in the worker gives five progress messages');
  t.ok(wk.atPause === wk.stillPaused, 'pause holds the worker (' + wk.atPause + ' then ' + wk.stillPaused + ' steps)');
  t.ok(wk.steps >= 400 && wk.last < 0.05, 'resumed to ' + wk.steps + ' steps, XOR loss ' + (wk.last && wk.last.toFixed(4)));
  t.ok(wk.snaps >= 8 && wk.snapKeys.join() === 'layers.0.W' && wk.snapShape.join() === '2,8', 'snapshots of the chosen tensor every 50 steps (' + wk.snaps + ')');
  t.ok(wk.cks >= 4 && wk.ckKeys === 4, 'checkpoints every 100 steps carry the whole state');
  // speed: the CPU (in the browser) and WebGPU
  const speed = await b.eval(`(async () => { const B = Nooral.T.backends, rows = [];
    for (const r of await B.bench(B.cpu, { sizes: [128, 256, 512], ms: 400 })) rows.push(Object.assign({ be: 'cpu (chrome)' }, r));
    if (B.webgpu.device) {
      for (const r of await B.bench(B.webgpu, { sizes: [128, 256, 512, 1024], ms: 400 })) rows.push(Object.assign({ be: 'webgpu, with transfers' }, r));
      // the kernels alone: inputs already on the GPU, timed to queue completion
      const G = B.webgpu.gpu, T = Nooral.T, g = T.rng(4);
      for (const n of [512, 1024, 2048]) { const a = G.upload(T.normal([n, n], 0, 1, g)), b2 = G.upload(T.normal([n, n], 0, 1, g));
        const ms = await B.time(async () => { const c = G.matmul(a, b2); await G.done(); G.free(c); }, { ms: 400 }); rows.push({ be: 'webgpu, kernel only', op: 'matmul ' + n + '×' + n, ms, gflops: 2 * n * n * n / ms / 1e6 }); G.free(a, b2); }
      const x = G.upload(T.normal([8, 16, 32, 32], 0, 1, g)), w = G.upload(T.normal([32, 16, 3, 3], 0, 1, g));
      for (const mode of ['im2col', 'direct']) { const ms = await B.time(async () => { const c = G.conv2d(x, w, null, { padding: 1, mode }); await G.done(); G.free(c); }, { ms: 400 }); rows.push({ be: 'webgpu, kernel only', op: 'conv2d ' + mode + ' [8,16,32,32]→32 3×3', ms, gflops: 2 * 8 * 32 * 32 * 32 * 16 * 9 / ms / 1e6 }); }
      G.free(x, w);
    }
    return rows; })()`);
  for (const r of speed) console.log('  ' + r.be.padEnd(23) + r.op.padEnd(38) + r.ms.toFixed(2).padStart(9) + ' ms ' + r.gflops.toFixed(2).padStart(8) + ' GFLOP/s');
  const errs = b.logs.filter((l) => l.type === 'exception' || (l.type === 'error' && /core\/t-/.test(l.text)));
  t.ok(!errs.length, 'no exceptions on the page: ' + errs.map((e) => e.text).join(' | '));
} catch (e) {
  t.ok(false, 'browser test: ' + e.message);
} finally {
  clearTimeout(guard);
  await b.close();
}
t.done();
