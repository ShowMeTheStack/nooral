// A digit reader for judging generated digits: a small classifier trained on our drawn digits, all ten of them, so
// that a generator's samples can be counted by what they look like, not by eye.
//   S.Gen.judgeNet({hidden: 32, n: 4000, steps: 3000, seed: 5}) -> {net, train(steps), accuracy(set), read(images)}
//      the network is 256 pixels -> hidden (ReLU) -> 10 logits; train() runs Adam on cross-entropy
//   S.Gen.readDigits(net, images, {sure: 0.9}) -> {digit, p (the top probability), probs, counts (10), sure (how many
//      have p >= sure), sureCounts, effective (e to the entropy of the counts), kinds (digits seen at least once)}
(function (S) {
  'use strict';
  const G = S.Gen;

  G.judgeNet = (o0 = {}) => {
    const T = S.T, nn = T.nn, o = Object.assign({ hidden: 32, n: 4000, seed: 5, lr: 2e-3, batch: 64, dataSeed: 77 }, o0);
    const g = T.rng(o.seed), r = T.rng(o.seed * 31 + 7);
    const net = new nn.Sequential(new nn.Dense(256, o.hidden, { rng: g, init: 'he' }), new nn.Activation('relu'), new nn.Dense(o.hidden, 10, { rng: g }));
    let set = null, opt = null;
    const train = (steps) => {
      set = set || G.images('digits', { n: o.n, size: 16, seed: o.dataSeed });
      opt = opt || new T.optim.Adam(net.parameters(), { lr: o.lr });
      let loss = 0;
      for (let s = 0; s < steps; s++) {
        const idx = Array.from({ length: o.batch }, () => r.int(set.n));
        opt.zeroGrad();
        const l = T.crossEntropy(net.forward(G.batchOf(set, idx)), idx.map((i) => set.y[i]));
        l.backward(); opt.step(); loss = l.item();
      }
      return loss;
    };
    const accuracy = (test) => { const rd = G.readDigits(net, [...Array(test.n).keys()].map((i) => test.X.subarray(i * 256, (i + 1) * 256))); return rd.digit.filter((d, i) => d === test.y[i]).length / test.n; };
    return { net, train, accuracy, read: (imgs, ro) => G.readDigits(net, imgs, ro), settings: o };
  };

  G.readDigits = (net, imgs, o = {}) => {
    const T = S.T, n = imgs.length, x = T.zeros([n, 256]), sure = o.sure || 0.9;
    imgs.forEach((im, i) => x.data.set(im, i * 256));
    const pr = T.noGrad(() => T.softmax(net.forward(x), -1).values());
    const digit = [], p = [], probs = [], counts = new Array(10).fill(0), sureCounts = new Array(10).fill(0);
    for (let i = 0; i < n; i++) {
      const row = Array.from(pr.subarray(i * 10, i * 10 + 10));
      let b = 0; for (let k = 1; k < 10; k++) if (row[k] > row[b]) b = k;
      digit.push(b); p.push(row[b]); probs.push(row); counts[b]++;
      if (row[b] >= sure) sureCounts[b]++;
    }
    const H = counts.reduce((h, c) => (c ? h - (c / n) * Math.log(c / n) : h), 0);
    return { digit, p, probs, counts, sure: sureCounts.reduce((a, b) => a + b, 0), sureCounts, effective: Math.exp(H), kinds: counts.filter((c) => c > 0).length, n, threshold: sure };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
