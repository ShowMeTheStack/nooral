// DDPM-style diffusion at a small scale, on 2-D points and on tiny 16×16 images.
//   S.Gen.schedule({steps: 100, kind: 'linear' | 'cosine', b0, b1}) -> {T, beta, alpha, alphaBar, trace}
//   S.Gen.noise(x0, t, sched, eps) -> x_t = √ᾱ_t x0 + √(1 − ᾱ_t) ε  (arrays of numbers)
//   S.Gen.diffusion2d({dist: 'moons', steps: 100, hidden, lr, batch, seed})
//   S.Gen.diffusionImages({data: 'shapes', classes: [0], n, steps: 60, arch: 'mlp' | 'conv', hidden, channels, lr, batch,
//      total, ema, seed})  total: cosine decay of the learning rate over that many steps; ema: sampling uses a moving
//      average of the weights
//      trainers: the denoiser learns to predict the noise ε from (x_t, t) by mean squared error.
//      forward(points?, every) -> the data dissolving into noise, a snapshot list [{t, points}]
//      sample({n, seed, keep, temp}) -> {final, frames: [{t, points | images}], trace} running the reverse chain step by
//      step; temp scales the fresh noise of each step (1 by default)
(function (S) {
  'use strict';
  const G = S.Gen;
  const fmt = (x) => G.fmt(x);

  G.schedule = (o = {}) => {
    const n = o.steps || 100, kind = o.kind || 'linear', beta = new Float64Array(n + 1), alpha = new Float64Array(n + 1), alphaBar = new Float64Array(n + 1);
    alphaBar[0] = 1; alpha[0] = 1;
    if (kind === 'cosine') {
      const f = (t) => Math.cos(((t / n + 0.008) / 1.008) * Math.PI / 2) ** 2;
      for (let t = 1; t <= n; t++) { beta[t] = Math.min(0.999, 1 - f(t) / f(t - 1)); }
    } else {
      const b0 = o.b0 || 1e-4 * (1000 / n), b1 = o.b1 || Math.min(0.5, 0.02 * (1000 / n));
      for (let t = 1; t <= n; t++) beta[t] = b0 + ((b1 - b0) * (t - 1)) / Math.max(1, n - 1);
    }
    for (let t = 1; t <= n; t++) { alpha[t] = 1 - beta[t]; alphaBar[t] = alphaBar[t - 1] * alpha[t]; }
    const T = S.T, mid = Math.round(n / 2);
    const trace = [
      T.step('beta', 'A ' + kind + ' schedule of ' + n + ' steps: β runs from ' + fmt(beta[1]) + ' at t = 1 to ' + fmt(beta[n]) + ' at t = ' + n + '.', { beta: Array.from(beta) }),
      T.step('alphaBar', 'ᾱ_t = Π (1 − β_s) is the share of the signal left: ' + fmt(alphaBar[1]) + ' at t = 1, ' + fmt(alphaBar[mid]) + ' at t = ' + mid + ' and ' + fmt(alphaBar[n]) + ' at t = ' + n + '.', { alphaBar: Array.from(alphaBar) }),
      T.step('snr', 'So at t = ' + n + ' a point is √ᾱ = ' + fmt(Math.sqrt(alphaBar[n])) + ' of itself plus √(1 − ᾱ) = ' + fmt(Math.sqrt(1 - alphaBar[n])) + ' of fresh noise: almost pure noise.', {}),
    ];
    return { T: n, kind, beta, alpha, alphaBar, trace };
  };
  G.noise = (x0, t, s, eps) => { const a = Math.sqrt(s.alphaBar[t]), b = Math.sqrt(1 - s.alphaBar[t]); return x0.map((v, i) => a * v + b * eps[i]); };

  /** a training core shared by both data kinds: rows are flat vectors of width dim */
  const ddpm = (o, sched, dim, net, draw) => {
    const T = S.T, r = T.rng((o.seed || 1) * 53 + 11), opt = new T.optim.Adam(net.parameters(), { lr: o.lr });
    // o.total: the learning rate falls along a cosine to a twentieth of itself over that many steps
    if (o.total) opt.schedule = T.sched.cosine({ lr: o.lr, min: o.lr / 20, total: o.total });
    // o.ema: a slowly moving average of the weights (w̄ ← ema · w̄ + (1 − ema) · w), which sampling uses
    const params = net.parameters(), avg = o.ema ? params.map((p) => Float64Array.from(p.data)) : null;
    const withAvg = (f) => { if (!avg) return f(); const keep = params.map((p) => Float32Array.from(p.data)); params.forEach((p, i) => p.data.set(avg[i])); try { return f(); } finally { params.forEach((p, i) => p.data.set(keep[i])); } };
    return {
      snapEvery: o.snapEvery || 200,
      withAvg,
      step() {
        const b = o.batch, x0 = draw(b, r), ts = Array.from({ length: b }, () => 1 + r.int(sched.T)), eps = T.randn([b, dim], r), xt = T.zeros([b, dim]);
        const xv = x0.values(), ev = eps.values();
        for (let i = 0; i < b; i++) { const a = Math.sqrt(sched.alphaBar[ts[i]]), c = Math.sqrt(1 - sched.alphaBar[ts[i]]); for (let j = 0; j < dim; j++) xt.data[i * dim + j] = a * xv[i * dim + j] + c * ev[i * dim + j]; }
        opt.zeroGrad();
        const loss = T.mseLoss(net.forward(xt, ts), eps);
        loss.backward(); opt.step();
        if (avg) params.forEach((p, i) => { const a = avg[i], d = p.data; for (let k = 0; k < d.length; k++) a[k] = o.ema * a[k] + (1 - o.ema) * d[k]; });
        return { loss: loss.item() };
      },
    };
  };
  /** the reverse chain: x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε̂) / √α_t + σ_t z, with σ_t² = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t) */
  const reverse = (net, sched, n, dim, o, toFrame) => {
    const T = S.T, g = T.rng(o.seed || 2024), keep = o.keep || G.keepSteps(sched.T, o.frames || 10), frames = [], trace = [];
    let x = T.randn([n, dim], g);
    frames.push(Object.assign({ t: sched.T }, toFrame(x)));
    T.noGrad(() => {
      for (let t = sched.T; t >= 1; t--) {
        const e = net.forward(x, new Array(n).fill(t)), xv = x.values(), ev = e.values(), out = T.zeros([n, dim]);
        const k1 = 1 / Math.sqrt(sched.alpha[t]), k2 = sched.beta[t] / Math.sqrt(1 - sched.alphaBar[t]);
        // o.temp < 1 scales the fresh noise down: a lower temperature, trading a little variety for fewer failed samples
        const sig = t > 1 ? (o.temp || 1) * Math.sqrt((sched.beta[t] * (1 - sched.alphaBar[t - 1])) / (1 - sched.alphaBar[t])) : 0;
        if (o.clip) {
          // through the predicted clean image x̂0 = (x_t − √(1 − ᾱ_t) ε̂) / √ᾱ_t, kept inside [−1, 1]
          const ab = sched.alphaBar[t], abp = sched.alphaBar[t - 1], c1 = (Math.sqrt(abp) * sched.beta[t]) / (1 - ab), c2 = (Math.sqrt(sched.alpha[t]) * (1 - abp)) / (1 - ab);
          for (let i = 0; i < n * dim; i++) { const x0 = Math.max(-1, Math.min(1, (xv[i] - Math.sqrt(1 - ab) * ev[i]) / Math.sqrt(ab))); out.data[i] = c1 * x0 + c2 * xv[i] + (sig ? sig * g.normal() : 0); }
        } else for (let i = 0; i < n * dim; i++) out.data[i] = k1 * (xv[i] - k2 * ev[i]) + (sig ? sig * g.normal() : 0);
        if (t === sched.T || t === 1 || keep.includes(t - 1)) {
          const p = Array.from(xv.subarray(0, Math.min(dim, 2))), q = Array.from(out.data.subarray(0, Math.min(dim, 2)));
          trace.push(T.step('reverse', 'Step t = ' + t + ': the denoiser predicts the noise, x is scaled by 1/√α = ' + fmt(k1) + ' after removing ' + fmt(k2) + ' × ε̂, and ' + (sig ? 'fresh noise with σ = ' + fmt(sig) + ' is added' : 'no noise is added at the last step') + '. The first sample moves from ' + G.pt(p) + ' to ' + G.pt(q) + '.', { t, k1, k2, sigma: sig }));
        }
        x = out;
        if (keep.includes(t - 1)) frames.push(Object.assign({ t: t - 1 }, toFrame(x)));
      }
    });
    return { x, frames, trace };
  };
  G.keepSteps = (n, k) => { const s = new Set([0]); for (let i = 1; i < k; i++) s.add(Math.round((n * i) / k)); return [...s].sort((a, b) => b - a); };

  G.diffusion2d = (o0 = {}) => {
    const T = S.T, nn = T.nn, o = Object.assign({ steps: 100, hidden: 64, lr: 2e-3, batch: 128, temb: 16, kind: 'linear' }, o0);
    const sched = G.schedule({ steps: o.steps, kind: o.kind }), data = G.dist2d(o.dist || 'moons', { n: o.n || 2048, seed: o.dataSeed || 1 }), X = data.X;
    const g = T.rng(o.seed || 1), l1 = new nn.Dense(2 + o.temb, o.hidden, { rng: g, init: 'he' }), l2 = new nn.Dense(o.hidden, o.hidden, { rng: g, init: 'he' }), l3 = new nn.Dense(o.hidden, o.hidden, { rng: g, init: 'he' }), l4 = new nn.Dense(o.hidden, 2, { rng: g });
    const net = new nn.Sequential(l1, l2, l3, l4);
    net.forward = (x, ts) => { const h = T.concat([x, G.timeEmbed(ts, o.temb, sched.T)], 1); return l4.forward(T.relu(l3.forward(T.relu(l2.forward(T.relu(l1.forward(h))))))); };
    const toPts = (x) => { const v = x.values(), out = []; for (let i = 0; i < v.length; i += 2) out.push([v[i], v[i + 1]]); return { points: out }; };
    const core = ddpm(o, sched, 2, net, (b, r) => G.rows(X, Array.from({ length: b }, () => r.int(X.length))));
    const sample = (so = {}) => { const s = reverse(net, sched, so.n || 512, 2, so, toPts); return { final: toPts(s.x).points, frames: s.frames, trace: s.trace }; };
    core.snapshot = () => { const s = sample({ n: 256, frames: 1 }); return { samples: s.final, swd: G.swd(s.final, X.slice(0, 512)) }; };
    const tr = G.trainer(core, o);
    return Object.assign(tr, {
      model: net, net, sched, data, sample,
      /** the forward process on the first n data points with one fixed noise draw per point */
      forward(n = 256, every = Math.max(1, Math.round(sched.T / 10))) {
        const g2 = T.rng(31), pts = X.slice(0, n), eps = pts.map(() => [g2.normal(), g2.normal()]), frames = [];
        for (let t = 0; t <= sched.T; t += every) frames.push({ t, points: pts.map((p, i) => G.noise(p, t, sched, eps[i])), signal: Math.sqrt(sched.alphaBar[t]), noise: Math.sqrt(1 - sched.alphaBar[t]) });
        const q = frames[frames.length - 1];
        return { frames, trace: [T.step('forward', 'At t = ' + q.t + ' the first point ' + G.pt(pts[0]) + ' has become ' + G.pt(q.points[0]) + ': ' + fmt(q.signal) + ' of the point plus ' + fmt(q.noise) + ' of its noise ' + G.pt(eps[0]) + '.', {})] };
      },
      /** distance from samples to the data, against the same distance for plain Gaussian noise */
      quality(n = 512, seed = 2024) {
        const s = sample({ n, seed, frames: 1 }).final, g3 = T.rng(seed), base = Array.from({ length: n }, () => [g3.normal(), g3.normal()]), ref = X.slice(0, n);
        return { swd: G.swd(s, ref), noiseSwd: G.swd(base, ref), chamfer: G.chamfer(s, X), noiseChamfer: G.chamfer(base, X), samples: s };
      },
    });
  };

  G.diffusionImages = (o0 = {}) => {
    const T = S.T, nn = T.nn, o = Object.assign({ steps: 60, arch: 'conv', hidden: 256, channels: 12, lr: 2e-3, batch: 16, temb: 4, kind: 'cosine' }, o0);
    const sched = G.schedule({ steps: o.steps, kind: o.kind }), set = G.images(o.data || 'shapes', { n: o.n || 256, size: o.size || 16, seed: o.dataSeed || 1, classes: o.classes || [0] });
    const s = set.size, C = o.channels, g = T.rng(o.seed || 1), P = set.P;
    let net;
    if (o.arch !== 'mlp') {
      const c1 = new nn.Conv2d(1 + o.temb, C, 3, { rng: g, padding: 1 }), c2 = new nn.Conv2d(C, C, 3, { rng: g, padding: 2, dilation: 2 }), c3 = new nn.Conv2d(C, C, 3, { rng: g, padding: 4, dilation: 4 }), c4 = new nn.Conv2d(C, 1, 3, { rng: g, padding: 1, init: 'xavier' });
      net = new nn.Sequential(c1, c2, c3, c4);
      // the time features enter as constant extra channels; the network sees pixels scaled to [−1, 1]
      net.forward = (x, ts) => {
        const b = x.shape[0], e = G.timeEmbed(ts, o.temb, sched.T), te = T.expand(T.reshape(e, [b, o.temb, 1, 1]), [b, o.temb, s, s]);
        const h1 = T.relu(c1.forward(T.concat([T.reshape(x, [b, 1, s, s]), te], 1))), h2 = T.relu(c2.forward(h1)), h3 = T.relu(T.add(c3.forward(h2), h1));
        return T.reshape(c4.forward(h3), [b, P]);
      };
    } else {
      // a fully connected denoiser: faster per step but much weaker, kept to show why convolutions help
      const H = o.hidden, l1 = new nn.Dense(P + o.temb, H, { rng: g, init: 'he' }), l2 = new nn.Dense(H, H, { rng: g, init: 'he' }), l3 = new nn.Dense(H, P, { rng: g });
      net = new nn.Sequential(l1, l2, l3);
      net.forward = (x, ts) => l3.forward(T.relu(l2.forward(T.relu(l1.forward(T.concat([x, G.timeEmbed(ts, o.temb, sched.T)], 1))))));
    }
    const draw = (b, r) => T.addScalar(T.scale(G.batchOf(set, Array.from({ length: b }, () => r.int(set.n))), 2), -1);
    const toImgs = (x) => { const v = x.values(), out = []; for (let i = 0; i < v.length; i += P) out.push(Float32Array.from(v.subarray(i, i + P), (u) => Math.max(0, Math.min(1, (u + 1) / 2)))); return { images: out }; };
    const core = ddpm(o, sched, P, net, draw);
    const sample = (so = {}) => core.withAvg(() => { const r = reverse(net, sched, so.n || 8, P, Object.assign({ clip: true }, so), toImgs); return { final: toImgs(r.x).images, frames: r.frames, trace: r.trace }; });
    core.snapshot = () => ({ images: sample({ n: 8, frames: 1 }).final });
    const tr = G.trainer(core, o);
    return Object.assign(tr, {
      model: net, net, sched, set, sample,
      forward(i = 0, every = Math.max(1, Math.round(sched.T / 6))) {
        const g2 = T.rng(31), x0 = Array.from(set.X.subarray(i * P, (i + 1) * P), (v) => 2 * v - 1), eps = x0.map(() => g2.normal()), frames = [];
        for (let t = 0; t <= sched.T; t += every) frames.push({ t, image: Float32Array.from(G.noise(x0, t, sched, eps), (u) => Math.max(0, Math.min(1, (u + 1) / 2))) });
        return { frames };
      },
    });
  };

  S.lateJob('diffusion2d', G.asJob((p) => G.diffusion2d(p)));
  S.lateJob('diffusionImages', G.asJob((p) => G.diffusionImages(p)));
})(globalThis.Nooral = globalThis.Nooral || {});
