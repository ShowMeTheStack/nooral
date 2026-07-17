// Autoencoders and the variational autoencoder, on small generated images (16×16 by default).
//   S.Gen.autoencoder({kind: 'mlp' | 'conv', data: 'digits', n, latent, hidden, lr, batch, seed, show})
//      -> trainer: step(), run(n), history [{step, loss}], snapshots [{step, loss, recon: [Float32Array]}],
//         encode(images) -> codes, decode(codes) -> images, reconstruct(i), explain(i) -> trace
//   S.Gen.vae({data, n, latent: 2, hidden, beta: 1, lr, batch, seed, grid: 9, range: 2.5})
//      -> trainer as above; history rows {loss, elbo, recon, kl}; snapshots add {map: [{z, y}], grid: k×k images};
//         elbo(set?) on held-out images with a fixed noise seed, latentMap(), decodeGrid(k, range), decode(z),
//         explain(i) -> the reparameterisation step by step (μ, log σ², σ, ε, z, KL per dimension, reconstruction)
// Images are grey in [0, 1]; the decoder gives logits and the reconstruction loss is binary cross-entropy summed
// over the pixels of an image (so the numbers are "nats per image").
(function (S) {
  'use strict';
  const G = S.Gen;
  const fmt = (x) => G.fmt(x);

  /** nearest-neighbour upsampling by 2 on [N, C, H, W], built from views so its gradient comes for free */
  G.up2 = (x) => {
    const T = S.T, [N, C, H, W] = x.shape;
    return T.reshape(T.expand(T.reshape(x, [N, C, H, 1, W, 1]), [N, C, H, 2, W, 2]), [N, C, 2 * H, 2 * W]);
  };
  const params = (...ms) => ms.flatMap((m) => m.parameters());

  /** the networks of an autoencoder: {enc, dec, encode(x [b, P]), decode(z) -> logits [b, P]} */
  G.aeNet = (o) => {
    const T = S.T, nn = T.nn, P = o.P, s = o.size, L = o.latent, h = o.hidden || 64, g = T.rng(o.seed || 1);
    if (o.kind === 'conv') {
      const enc = new nn.Sequential(new nn.Conv2d(1, 8, 3, { rng: g, stride: 2, padding: 1 }), new nn.Activation('relu'),
        new nn.Conv2d(8, 16, 3, { rng: g, stride: 2, padding: 1 }), new nn.Activation('relu'), new nn.Flatten(), new nn.Dense(16 * (s / 4) * (s / 4), L, { rng: g }));
      const d0 = new nn.Dense(L, 16 * (s / 4) * (s / 4), { rng: g, init: 'he' }), c1 = new nn.Conv2d(16, 8, 3, { rng: g, padding: 1 }), c2 = new nn.Conv2d(8, 1, 3, { rng: g, padding: 1, init: 'xavier' });
      const dec = new nn.Sequential(d0, c1, c2);
      return {
        kind: 'conv', enc, dec, parameters: () => params(enc, dec),
        encode: (x) => enc.forward(T.reshape(x, [x.shape[0], 1, s, s])),
        decode: (z) => {
          let a = T.reshape(T.relu(d0.forward(z)), [z.shape[0], 16, s / 4, s / 4]);
          a = T.relu(c1.forward(G.up2(a)));
          return T.reshape(c2.forward(G.up2(a)), [z.shape[0], P]);
        },
      };
    }
    const outL = o.variational ? 2 * L : L;
    const enc = new nn.Sequential(new nn.Dense(P, h, { rng: g, init: 'he' }), new nn.Activation('relu'), new nn.Dense(h, outL, { rng: g }));
    const dec = new nn.Sequential(new nn.Dense(L, h, { rng: g, init: 'he' }), new nn.Activation('relu'), new nn.Dense(h, P, { rng: g }));
    return { kind: 'mlp', enc, dec, parameters: () => params(enc, dec), encode: (x) => enc.forward(x), decode: (z) => dec.forward(z) };
  };

  const perImageBce = (logits, x) => { const T = S.T; return T.scale(T.sum(T.bceWithLogits(logits, x, { reduction: 'none' })), 1 / x.shape[0]); };
  const sig = (v) => 1 / (1 + Math.exp(-v));
  const imgs = (t, P) => { const v = t.values(), out = []; for (let i = 0; i < v.length; i += P) out.push(Float32Array.from(v.subarray(i, i + P), sig)); return out; };
  const setup = (o, latent) => {
    const set = G.images(o.data || 'digits', { n: o.n || 512, size: o.size || 16, seed: o.dataSeed || 1, classes: o.classes });
    const test = G.images(o.data || 'digits', { n: o.nTest || 128, size: o.size || 16, seed: (o.dataSeed || 1) + 1000, classes: o.classes });
    const show = Array.from({ length: o.show || 8 }, (_, i) => i);
    return { set, test, show, latent, r: S.T.rng((o.seed || 1) * 31 + 7), batch: o.batch || 32 };
  };

  G.autoencoder = (o = {}) => {
    const T = S.T, c = setup(o, o.latent || (o.kind === 'conv' ? 8 : 4)), { set, test, show, r, batch } = c;
    const net = G.aeNet({ kind: o.kind || 'mlp', P: set.P, size: set.size, latent: c.latent, hidden: o.hidden || 64, seed: o.seed || 1 });
    const opt = new T.optim.Adam(net.parameters(), { lr: o.lr || 3e-3 });
    const shape = (x) => x;
    const showX = G.batchOf(test, show);
    const tr = G.trainer({
      snapEvery: o.snapEvery || 50,
      step() {
        const idx = Array.from({ length: batch }, () => r.int(set.n)), x = G.batchOf(set, idx);
        opt.zeroGrad();
        const loss = perImageBce(net.decode(net.encode(shape(x))), x);
        loss.backward(); opt.step();
        return { loss: loss.item() };
      },
      snapshot: () => T.noGrad(() => { const lg = net.decode(net.encode(showX)); return { loss: perImageBce(lg, showX).item(), recon: imgs(lg, set.P) }; }),
    }, o);
    Object.assign(tr, {
      model: net, net, set, test, latent: c.latent, originals: show.map((i) => test.X.slice(i * set.P, (i + 1) * set.P)),
      encode: (x) => T.noGrad(() => net.encode(x)),
      decode: (z) => T.noGrad(() => imgs(net.decode(z instanceof T.Tensor ? z : T.tensor(z)), set.P)),
      reconstruct: (i, from = test) => T.noGrad(() => imgs(net.decode(net.encode(G.batchOf(from, [i]))), set.P)[0]),
      testLoss: () => T.noGrad(() => { const x = G.batchOf(test, [...Array(test.n).keys()]); return perImageBce(net.decode(net.encode(x)), x).item(); }),
      explain(i = 0) {
        const x = G.batchOf(test, [i]), code = T.noGrad(() => net.encode(x)), lg = T.noGrad(() => net.decode(code)), out = imgs(lg, set.P)[0];
        const xv = x.values(), cv = Array.from(code.values()), loss = T.noGrad(() => perImageBce(lg, x).item());
        let ink = 0; for (const v of xv) ink += v;
        let err = 0; for (let k = 0; k < out.length; k++) err += Math.abs(out[k] - xv[k]);
        return [
          T.step('input', 'The input is a ' + set.size + '×' + set.size + ' image of ' + set.P + ' pixels, with total ink ' + fmt(ink) + '.', { image: Float32Array.from(xv) }),
          T.step('encode', 'The encoder squeezes it into ' + cv.length + ' numbers: [' + cv.map(fmt).join(', ') + '].', { code: cv }),
          T.step('decode', 'The decoder rebuilds ' + set.P + ' pixels from those ' + cv.length + ' numbers; the mean absolute pixel error is ' + fmt(err / set.P) + '.', { image: out }),
          T.step('loss', 'The reconstruction loss (binary cross-entropy summed over pixels) is ' + fmt(loss) + ' nats.', { loss }),
        ];
      },
    });
    tr.tensors = () => Object.fromEntries(net.enc.namedParameters().map(([n, p]) => ['enc.' + n, p]));
    return tr;
  };

  G.vae = (o = {}) => {
    const T = S.T, c = setup(o, o.latent || 2), { set, test, show, r, batch } = c, L = c.latent, beta = o.beta == null ? 1 : o.beta;
    const net = G.aeNet({ kind: 'mlp', variational: true, P: set.P, size: set.size, latent: L, hidden: o.hidden || 128, seed: o.seed || 1 });
    const opt = new T.optim.Adam(net.parameters(), { lr: o.lr || 2e-3 });
    const showX = G.batchOf(test, show), mapIdx = [...Array(Math.min(test.n, o.mapN || 128)).keys()];
    /** the forward pass with the reparameterisation: z = μ + σ ⊙ ε, ε from the generator g */
    const pass = (x, g, b = beta) => {
      const h = net.encode(x), n = x.shape[0];
      const mu = T.slice(h, null, [0, L]), lv = T.clamp(T.slice(h, null, [L, 2 * L]), -8, 6);
      const sd = T.exp(T.scale(lv, 0.5)), eps = T.normal([n, L], 0, 1, g), z = T.add(mu, T.mul(sd, eps));
      const logits = net.decode(z), recon = perImageBce(logits, x);
      // KL(q(z|x) ‖ N(0, I)) = ½ Σ (μ² + σ² − 1 − log σ²), averaged over the batch
      const kl = T.scale(T.sum(T.sub(T.add(T.square(mu), T.exp(lv)), T.addScalar(lv, 1))), 0.5 / n);
      return { mu, lv, sd, eps, z, logits, recon, kl, loss: T.add(recon, T.scale(kl, b)) };
    };
    const latentMap = () => T.noGrad(() => { const mu = T.slice(net.encode(G.batchOf(test, mapIdx)), null, [0, L]).values(); return mapIdx.map((i, j) => ({ z: Array.from(mu.subarray(j * L, j * L + L)), y: test.y[i] })); });
    const decodeGrid = (k = o.grid || 9, range = o.range || 2.5) => {
      if (L !== 2) return null;
      const zs = []; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) zs.push([-range + (2 * range * b) / (k - 1), range - (2 * range * a) / (k - 1)]);
      return { k, range, z: zs, images: T.noGrad(() => imgs(net.decode(T.tensor(zs)), set.P)) };
    };
    const elbo = (from = test) => T.noGrad(() => { const x = G.batchOf(from, [...Array(from.n).keys()]), p = pass(x, T.rng(4242), 1); return { elbo: -p.loss.item(), recon: p.recon.item(), kl: p.kl.item() }; });
    const tr = G.trainer({
      snapEvery: o.snapEvery || 100,
      step() {
        const idx = Array.from({ length: batch }, () => r.int(set.n)), x = G.batchOf(set, idx);
        opt.zeroGrad();
        const p = pass(x, r);
        p.loss.backward(); opt.step();
        const rec = p.recon.item(), kl = p.kl.item();
        return { loss: p.loss.item(), elbo: -(rec + kl), recon: rec, kl };
      },
      snapshot: () => Object.assign(elbo(), T.noGrad(() => ({ recon: imgs(pass(showX, T.rng(7)).logits, set.P) })), { map: latentMap(), grid: o.gridEvery === false ? null : decodeGrid() }),
    }, o);
    return Object.assign(tr, {
      model: net, net, set, test, latent: L, elbo, latentMap, decodeGrid, pass, originals: show.map((i) => test.X.slice(i * set.P, (i + 1) * set.P)),
      decode: (z) => T.noGrad(() => imgs(net.decode(T.tensor(Array.isArray(z[0]) ? z : [z])), set.P)),
      tensors: () => Object.fromEntries(net.parameters().map((p, i) => ['p' + i, p])),
      explain(i = 0, seed = 5) {
        const x = G.batchOf(test, [i]), p = T.noGrad(() => pass(x, T.rng(seed), 1));
        const v = (t) => Array.from(t.values()), mu = v(p.mu), lv = v(p.lv), sd = v(p.sd), eps = v(p.eps), z = v(p.z);
        const klDim = mu.map((m, k) => 0.5 * (m * m + Math.exp(lv[k]) - 1 - lv[k])), list = (a) => '[' + a.map(fmt).join(', ') + ']';
        return [
          T.step('encode', 'The encoder maps the image (a ' + set.classes[test.y[i]] + ') to a mean μ = ' + list(mu) + ' and a log-variance log σ² = ' + list(lv) + '.', { mu, logvar: lv }),
          T.step('sigma', 'σ = exp(½ log σ²) = ' + list(sd) + ': how unsure the encoder is about where the image belongs.', { sigma: sd }),
          T.step('noise', 'Draw ε from a standard normal: ε = ' + list(eps) + '. The randomness lives in ε, outside the network.', { eps }),
          T.step('reparam', 'z = μ + σ ⊙ ε = ' + list(z) + '. Because z is a sum and a product of μ and σ, gradients flow back into the encoder.', { z }),
          T.step('kl', 'KL to the standard normal = ½ Σ (μ² + σ² − 1 − log σ²) = ' + klDim.map(fmt).join(' + ') + ' = ' + fmt(p.kl.item()) + ' nats.', { kl: klDim }),
          T.step('decode', 'The decoder turns z into an image; its reconstruction loss is ' + fmt(p.recon.item()) + ' nats.', { image: imgs(p.logits, set.P)[0], recon: p.recon.item() }),
          T.step('elbo', 'ELBO = −(reconstruction + KL) = −(' + fmt(p.recon.item()) + ' + ' + fmt(p.kl.item()) + ') = ' + fmt(-p.loss.item()) + '; training pushes it up.', { elbo: -p.loss.item() }),
        ];
      },
    });
  };

  S.lateJob('autoencoder', G.asJob((p) => G.autoencoder(p)));
  S.lateJob('vae', G.asJob((p) => G.vae(p)));
})(globalThis.Nooral = globalThis.Nooral || {});
