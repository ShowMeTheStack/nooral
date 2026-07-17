// Generative adversarial networks: a generator G(z) and a discriminator D(x) trained against each other with the
// non-saturating loss (D minimises BCE on real = 1, fake = 0; G minimises BCE of D(G(z)) against 1).
//   S.Gen.gan2d({dist: 'ring' | 'grid' | ..., preset: 'good' | 'collapse', steps, seed, ...overrides})
//      snapshots {samples: [[x, y]] for a fixed z batch, field: {k, lo, hi, d: D's probability on a k×k grid},
//                 dReal, dFake, coverage}; history rows {loss: lossD + lossG, lossD, lossG, dReal, dFake}
//   S.Gen.ganImages({data: 'digits', classes, preset: 'steady' | 'plain', steps, seed, show: 16})
//      snapshots {images: [Float32Array], dFake, dReal}; samples(n, seed) -> n images from fresh noise
//      the steadied duel adds one-sided label smoothing (real target 0.9), two learning rates (the discriminator's
//      four times the generator's), a minibatch-spread feature in the discriminator, and a spectral-norm cap on each
//      of the discriminator's weight matrices; the plain duel has none of them
//   S.Gen.ganImagePresets  the settings behind each, with a sentence saying why
//   S.Gen.specNorm(layers, state, cap)  one power-iteration step per layer; scales W down so its largest singular
//      value is at most cap -> the estimated values before scaling
//   S.Gen.ganPresets  the settings behind each preset, with a sentence saying why it behaves as it does
//   trainer.explain() -> one step of the duel with the actual numbers
(function (S) {
  'use strict';
  const G = S.Gen;
  const fmt = (x) => G.fmt(x);

  G.ganPresets = {
    good: { zDim: 4, hidden: 64, lrG: 1e-3, lrD: 1e-3, dSteps: 1, batch: 128, betas: [0.5, 0.999],
      why: 'Balanced learning rates and large batches: the discriminator keeps finding the modes the generator neglects, so the generator spreads out to cover them.' },
    collapse: { zDim: 4, hidden: 64, lrG: 4e-3, lrD: 1e-4, dSteps: 1, batch: 32, betas: [0.5, 0.999],
      why: 'The generator learns 40 times faster than the discriminator: it rushes to whatever region the slow discriminator currently likes, and its samples bunch up on a few modes instead of all of them.' },
  };

  const mlp = (sizes, g, outAct) => {
    const nn = S.T.nn, layers = [];
    for (let i = 0; i + 1 < sizes.length; i++) {
      layers.push(new nn.Dense(sizes[i], sizes[i + 1], { rng: g, init: 'xavier' }));
      if (i + 2 < sizes.length) layers.push(new nn.Activation('leakyRelu', 0.2));
    }
    if (outAct) layers.push(new nn.Activation(outAct));
    return new nn.Sequential(...layers);
  };

  G.ganImagePresets = {
    steady: { zDim: 16, layersG: [128, 256], layersD: [256, 128], mbstd: true, smooth: 0.9, sn: 1, lrG: 1e-4, lrD: 4e-4, dSteps: 1, batch: 64, n: 2048, betas: [0.5, 0.999],
      why: 'Four remedies at once: real images are labelled 0.9 instead of 1, so the discriminator never becomes certain; it learns four times faster than the generator, so it keeps up with every move; it sees how spread out each batch is, so a batch of near copies gives itself away; and each of its weight matrices is scaled so it can stretch no direction by more than 1, which keeps its gradients smooth.' },
    plain: { zDim: 16, hidden: 128, lrG: 2e-3, lrD: 2e-3, dSteps: 1, batch: 64, n: 512, betas: [0.5, 0.999],
      why: 'The plain duel: equal, fast learning rates, hard 0/1 labels, and a discriminator that judges each image alone. Nothing stops the generator from settling on one kind of digit that fools it.' },
  };

  /** spectral-norm cap: one power-iteration step per dense layer (state keeps each layer's vector u) */
  G.specNorm = (layers, state, cap) => layers.map((L) => {
    const W = L.W.data, n = L.inF, m = L.outF;
    let u = state.get(L);
    if (!u) { u = new Float64Array(m).fill(1 / Math.sqrt(m)); state.set(L, u); }
    const v = new Float64Array(n);
    let nv = 0;
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += W[i * m + j] * u[j]; v[i] = s; nv += s * s; }
    nv = Math.sqrt(nv) || 1;
    const u2 = new Float64Array(m);
    for (let i = 0; i < n; i++) { const vi = v[i] / nv; for (let j = 0; j < m; j++) u2[j] += W[i * m + j] * vi; }
    let sig = 0; for (let j = 0; j < m; j++) sig += u2[j] * u2[j];
    sig = Math.sqrt(sig) || 1;
    for (let j = 0; j < m; j++) u[j] = u2[j] / sig;
    if (sig > cap) for (let k = 0; k < W.length; k++) W[k] *= cap / sig;
    return sig;
  });

  /** a discriminator that also sees the batch's spread: after its first layer the mean standard deviation of the
   *  features over the batch is appended to every row, so a batch of near copies looks different from a real one */
  const critic = (dim, widths, g) => {
    const nn = S.T.nn, T = S.T, d1 = new nn.Dense(dim, widths[0], { rng: g, init: 'xavier' }), d2 = new nn.Dense(widths[0] + 1, widths[1], { rng: g, init: 'xavier' }), d3 = new nn.Dense(widths[1], 1, { rng: g, init: 'xavier' });
    const act = new nn.Activation('leakyRelu', 0.2), net = new nn.Sequential(d1, act, d2, new nn.Activation('leakyRelu', 0.2), d3);
    net.spread = (h) => { const b = h.shape[0], m = T.mean(h, 0, true), v = T.mean(T.square(T.sub(h, m)), 0, true), s = T.mean(T.sqrt(T.addScalar(v, 1e-6))); return T.concat([h, T.expand(T.reshape(s, [1, 1]), [b, 1])], 1); };
    net.forward = (x) => d3.forward(act.forward(d2.forward(net.spread(act.forward(d1.forward(x))))));
    net.dense = [d1, d2, d3];
    return net;
  };

  /** the shared training loop: sample(idx) gives a real batch, dim is the data width */
  const duel = (o, real, dim, outAct) => {
    const T = S.T, g = T.rng(o.seed || 1), r = T.rng((o.seed || 1) * 97 + 3);
    const Gn = mlp([o.zDim, ...(o.layersG || [o.hidden, o.hidden]), dim], g, outAct), Dn = o.mbstd ? critic(dim, o.layersD || [o.hidden, o.hidden], g) : mlp([dim, ...(o.layersD || [o.hidden, o.hidden]), 1], g);
    const dense = Dn.dense || Dn.layers.filter((L) => L instanceof S.T.nn.Dense), snState = new Map(), smooth = o.smooth || 1;
    if (o.sn) G.specNorm(dense, snState, o.sn);
    const optG = new T.optim.Adam(Gn.parameters(), { lr: o.lrG, betas: o.betas }), optD = new T.optim.Adam(Dn.parameters(), { lr: o.lrD, betas: o.betas });
    const ones = (n) => T.ones([n, 1]), zeros = (n) => T.zeros([n, 1]);
    const mean = (t) => { const v = t.values(); let s = 0; for (const x of v) s += 1 / (1 + Math.exp(-x)); return s / v.length; };
    const core = {
      Gn, Dn,
      step() {
        const b = o.batch;
        let lossD = 0, dReal = 0, dFake = 0;
        for (let k = 0; k < o.dSteps; k++) {
          const xr = real(b, r), xf = T.noGrad(() => Gn.forward(T.randn([b, o.zDim], r)));
          optD.zeroGrad();
          const lr = Dn.forward(xr), lf = Dn.forward(xf);
          const l = T.add(T.bceWithLogits(lr, smooth === 1 ? ones(b) : T.full([b, 1], smooth)), T.bceWithLogits(lf, zeros(b)));
          l.backward(); optD.step();
          if (o.sn) G.specNorm(dense, snState, o.sn);
          lossD = l.item(); dReal = mean(lr); dFake = mean(lf);
        }
        optG.zeroGrad();
        const lg = T.bceWithLogits(Dn.forward(Gn.forward(T.randn([b, o.zDim], r))), ones(b));
        lg.backward(); optG.step();
        // the discriminator's gradients from the generator's step are not used
        Dn.zeroGrad();
        return { loss: lossD + lg.item(), lossD, lossG: lg.item(), dReal, dFake };
      },
    };
    return core;
  };

  G.gan2d = (o0 = {}) => {
    const T = S.T, o = Object.assign({}, G.ganPresets[o0.preset || 'good'], o0), data = G.dist2d(o.dist || 'ring', { n: o.n || 2048, seed: o.dataSeed || 1 });
    const X = data.X, core = duel(o, (b, r) => G.rows(X, Array.from({ length: b }, () => r.int(X.length))), 2);
    const zFix = T.randn([o.show || 256, o.zDim], T.rng(999)), k = o.fieldK || 25, lo = -3, hi = 3;
    const gridPts = []; for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) gridPts.push([lo + ((hi - lo) * c) / (k - 1), hi - ((hi - lo) * a) / (k - 1)]);
    const gridT = T.tensor(gridPts);
    const samples = (n) => T.noGrad(() => { const v = core.Gn.forward(n ? T.randn([n, o.zDim], T.rng(1234)) : zFix).values(), out = []; for (let i = 0; i < v.length; i += 2) out.push([v[i], v[i + 1]]); return out; });
    const field = () => T.noGrad(() => ({ k, lo, hi, d: Array.from(T.sigmoid(core.Dn.forward(gridT)).values()) }));
    const coverage = (P) => (data.modes ? G.coverage(P, data.modes, 3 * data.sd) : null);
    const tr = G.trainer({
      snapEvery: o.snapEvery || 100,
      step: core.step,
      snapshot() { const s = samples(); return { samples: s, field: field(), coverage: coverage(s), swd: G.swd(s, X.slice(0, 512)) }; },
    }, o);
    return Object.assign(tr, {
      model: core.Gn, G: core.Gn, D: core.Dn, data, preset: o.preset || 'good', settings: o, samples, field, coverage: (n = 1024) => coverage(samples(n)),
      swd: (n = 512) => G.swd(samples(n), X.slice(0, n)),
      explain() {
        const z = T.randn([1, o.zDim], T.rng(77)), xr = [X[0]];
        return T.noGrad(() => {
          const xf = core.Gn.forward(z), f = Array.from(xf.values()), dR = T.sigmoid(core.Dn.forward(T.tensor(xr))).item(), dF = T.sigmoid(core.Dn.forward(xf)).item();
          return [
            T.step('noise', 'The generator gets random noise z = [' + Array.from(z.values()).map(fmt).join(', ') + '].', { z: Array.from(z.values()) }),
            T.step('fake', 'It turns z into a point ' + G.pt(f) + '.', { fake: f }),
            T.step('judge', 'The discriminator says a real point ' + G.pt(xr[0]) + ' is real with probability ' + fmt(dR) + ', and the generated one with probability ' + fmt(dF) + '.', { dReal: dR, dFake: dF }),
            T.step('lossD', 'The discriminator\'s loss −log D(real) − log(1 − D(fake)) = ' + fmt(-Math.log(dR) - Math.log(1 - dF)) + ': it wants D(real) near 1 and D(fake) near 0.', { lossD: -Math.log(dR) - Math.log(1 - dF) }),
            T.step('lossG', 'The generator\'s loss −log D(fake) = ' + fmt(-Math.log(dF)) + ': it moves its points to where the discriminator is fooled.', { lossG: -Math.log(dF) }),
          ];
        });
      },
      tensors: () => Object.fromEntries(core.Gn.namedParameters().map(([n, p]) => ['G.' + n, p])),
    });
  };

  G.ganImages = (o0 = {}) => {
    const T = S.T, o = Object.assign({}, G.ganImagePresets[o0.preset || 'steady'], o0);
    const set = G.images(o.data || 'digits', { n: o.n || 512, size: o.size || 16, seed: o.dataSeed || 1, classes: o.classes || [0, 1, 7] });
    const core = duel(o, (b, r) => G.batchOf(set, Array.from({ length: b }, () => r.int(set.n))), set.P, 'sigmoid');
    const zFix = T.randn([o.show || 16, o.zDim], T.rng(999));
    const images = () => T.noGrad(() => { const v = core.Gn.forward(zFix).values(), out = []; for (let i = 0; i < v.length; i += set.P) out.push(Float32Array.from(v.subarray(i, i + set.P))); return out; });
    const realShow = G.batchOf(set, Array.from({ length: o.show || 16 }, (_, i) => i));
    // the discriminator's view: its probability of "real" for the shown fakes and for as many real images
    const judge = () => T.noGrad(() => ({ dFake: Array.from(T.sigmoid(core.Dn.forward(core.Gn.forward(zFix))).values()), dReal: Array.from(T.sigmoid(core.Dn.forward(realShow)).values()) }));
    const tr = G.trainer({ snapEvery: o.snapEvery || 100, step: core.step, snapshot: () => Object.assign({ images: images() }, judge()) }, o);
    /** n images from fresh noise of a seed of its own (the same seed gives the same noise at every step) */
    const samples = (n = 64, seed = 4242) => T.noGrad(() => { const v = core.Gn.forward(T.randn([n, o.zDim], T.rng(seed))).values(), out = []; for (let i = 0; i < v.length; i += set.P) out.push(Float32Array.from(v.subarray(i, i + set.P))); return out; });
    return Object.assign(tr, { model: core.Gn, G: core.Gn, D: core.Dn, set, images, judge, samples, preset: o0.preset || 'steady', settings: o, tensors: () => Object.fromEntries(core.Gn.namedParameters().map(([n, p]) => ['G.' + n, p])) });
  };

  S.lateJob('gan2d', G.asJob((p) => G.gan2d(p)));
  S.lateJob('ganImages', G.asJob((p) => G.ganImages(p)));
})(globalThis.Nooral = globalThis.Nooral || {});
