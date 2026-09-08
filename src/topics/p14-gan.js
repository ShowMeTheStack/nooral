// Part 14, generative adversarial networks: the duel on a ring of eight Gaussians with the discriminator's view of
// the plane, mode collapse against a balanced run side by side, and a GAN learning small digits. All from stored
// training runs of the engine (S.Gen.gan2d, S.Gen.ganImages); the coverage of the modes is measured live.
(function (S) {
  'use strict';
  const P = S.P14, H = () => S.Kit.ml, C = () => S.Kit.curve;

  P.ganRun = (id) => P.once('gan' + id, () => {
    const J = P.D()[id === 'good' ? 'ganGood' : 'ganCollapse'];
    return { id, steps: J.steps, hist: J.hist, o: J.o, snaps: J.snaps.map((s) => ({ step: s.step, samples: P.pairs(P.i16(s.samples)), field: P.u8(s.field), k: s.k, swd: s.swd, covered: s.covered, effective: s.effective })) };
  });
  P.ringData = () => P.once('ringData', () => S.Gen.dist2d('ring', { n: 2048, seed: 1 }));
  /** the training curves at a step (kept every 10 steps) */
  P.histAt = (h, step) => (step ? { dReal: P.at(h.dReal, step, 10), dFake: P.at(h.dFake, step, 10), lossD: P.at(h.lossD, step, 10), lossG: P.at(h.lossG, step, 10) } : null);

  /** a plane over [−3, 3]² with the discriminator's field, real points, the modes and the generated samples */
  P.ganPlane = (pre, x, y, w, snap, o) => {
    o = o || {};
    const ax = C().axes({ prefix: pre, x, y, w, h: w, xr: [-3, 3], yr: [-3, 3], xticks: 1.5, yticks: 1.5, grid: false, xname: o.names === false ? '' : 'x₁', yname: o.names === false ? '' : 'x₂' }), data = P.ringData(), items = [...ax.items];
    if (o.field !== false) {
      // the field is kept on a k×k grid of corner points; drawn as a raster of the same size
      const k = snap.k, vals = Array.from(snap.field);
      items.push({ key: pre + ':f', kind: 'raster', x: ax.o.x, y: ax.o.y, w: ax.o.w, h: ax.o.h, nx: k, ny: k, vals, scheme: 'prob', max: 1, smooth: true, layer: 'back', z: 0, alpha: 0.55 });
    }
    data.X.slice(0, 240).forEach((p, i) => items.push({ key: pre + ':r' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 2.4, cls: 'p14real', audit: 'overlap-ok', layer: 'items' }));
    const miss = o.miss || [];
    data.modes.forEach((m, i) => items.push({ key: pre + ':m' + i, kind: 'dot', x: ax.X(m[0]), y: ax.Y(m[1]), r: (3 * data.sd * w) / 6 + 4, cls: 'p14mode' + (miss.includes(i) ? ' miss' : ''), audit: 'overlap-ok', layer: 'items', z: 3 }));
    let out = 0;
    snap.samples.forEach((p, i) => { if (Math.abs(p[0]) <= 3 && Math.abs(p[1]) <= 3) items.push({ key: pre + ':s' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 2.6, cls: 'p14fake', audit: 'overlap-ok', layer: 'items', z: 2 }); else out++; });
    if (out) items.push(H().text(pre + ':out', x + w - 4, y + 16, out + ' outside the picture', 'p14fake', 12, 'end'));
    return { ax, items, out };
  };
  const RUNS = [{ value: 'good', label: 'balanced (equal learning rates, batches of 128)' }, { value: 'collapse', label: 'unbalanced (the generator learns 10 times faster)' }];

  S.topic({
    id: 'gen-gan-duel', family: 'p14-gan', order: 10, title: 'A GAN duel on a ring',
    blurb: 'A GAN trains two networks against each other. The generator turns random noise into points; the discriminator looks at a point and says how likely it is to be real. The discriminator learns to tell the two apart, and the generator learns to fool it. Here the real data are eight small clusters on a ring, the coloured background is the discriminator’s opinion of every place in the plane, and the blue dots are 256 points from the generator.',
    inputs: [
      { id: 'run', kind: 'choice', label: 'Training settings', def: 'good', options: RUNS },
      { id: 'field', kind: 'choice', label: 'Discriminator’s view', def: 'on', options: [{ value: 'on', label: 'shown' }, { value: 'off', label: 'hidden' }] },
    ],
    presets: [
      { label: 'A balanced duel', values: { run: 'good', field: 'on' } },
      { label: 'Just the samples', values: { run: 'good', field: 'off' } },
      { label: 'The generator runs ahead', values: { run: 'collapse', field: 'on' } },
    ],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const R0 = P.ganRun(inp.run), F = H().frames(['step']), q = F.q, h = R0.hist, n = h.dReal.length;
      const ap = C().axes({ prefix: 'gp', x: 470, y: 50, w: 300, h: 120, xr: [0, R0.steps], yr: [0, 1], xticks: P.ticksN(R0.steps), yticks: [0, 0.5, 1], yname: 'D’s average output' });
      const lmax = Math.max(...h.lossD, ...h.lossG), al = C().axes({ prefix: 'gl', x: 470, y: 240, w: 300, h: 120, xr: [0, R0.steps], yr: [0, lmax * 1.05], xticks: P.ticksN(R0.steps), yticks: C().ticks(0, lmax, 3), xname: 'step', yname: 'losses' });
      const curve = (ax, key, arr, cls, k) => C().line(ax, key, arr.slice(0, k).map((v, j) => [(j + 1) * 10, v]), cls);
      const lg = [...C().legend('lgp', 40, 470, [{ text: 'real points', cls: 'p14real', dot: true }, { text: 'generated', cls: 'p14fake', dot: true }], { dir: 'row', size: 13 })];
      for (const s of R0.snaps) {
        F.set('step', s.step);
        const k = s.step / 10, pl = P.ganPlane('gx', 40, 50, 360, s, { field: inp.field === 'on' }), items = [...pl.items, ...ap.items, ...al.items, ...lg];
        if (k) {
          items.push(curve(ap, 'gp:r', h.dReal, 'p14d', k), curve(ap, 'gp:f', h.dFake, 'p14g', k), curve(al, 'gl:d', h.lossD, 'p14d', k), curve(al, 'gl:g', h.lossG, 'p14g', k));
          items.push(C().vline(ap, 'gp:now', s.step, 'act'), C().vline(al, 'gl:now', s.step, 'act'));
          const apart = (a, b) => { const m = (a + b) / 2; return Math.abs(a - b) >= 15 ? [a, b] : a <= b ? [m - 7.5, m + 7.5] : [m + 7.5, m - 7.5]; };
          const [y1, y2] = apart(ap.Y(P.at(h.dReal, s.step, 10)), ap.Y(P.at(h.dFake, s.step, 10))), [y3, y4] = apart(al.Y(P.at(h.lossD, s.step, 10)), al.Y(P.at(h.lossG, s.step, 10)));
          items.push(H().text('gp:lr', 776, y1 + 4, 'D(real)', 'p14real', 12), H().text('gp:lf', 776, y2 + 4, 'D(fake)', 'p14fake', 12));
          items.push(H().text('gl:ld', 776, y3 + 4, 'D', 'p14real', 12), H().text('gl:lg', 776, y4 + 4, 'G', 'p14fake', 12));
        }
        const at = P.histAt(h, s.step);
        let say;
        if (!s.step) say = `Before training the generator squeezes its noise into a blob near the middle and the discriminator has no opinion yet. Its samples reach ${q.n(s.covered)} of the ${q.n(8)} clusters.`;
        else say = `Step ${q.n(s.step)}: on its last batch the discriminator gave real points ${q(at.dReal, 2)} on average and generated ones ${q(at.dFake, 2)}; its loss is ${q(at.lossD, 2)}, the generator’s ${q(at.lossG, 2)}. The samples now cover ${q.n(s.covered)} of the ${q.n(8)} clusters, spread over the equivalent of ${q(s.effective, 1)}.`;
        F.push(items, say, Object.assign({ phase: s.step ? 'snap' : 'start', step: s.step, covered: s.covered, effective: s.effective, eight: 8 }, at || {}));
      }
      const last = R0.snaps[R0.snaps.length - 1], L = P.histAt(h, last.step);
      F.push(F.last().items, inp.run === 'good'
        ? `After ${q.n(last.step)} steps the duel has reached a standoff: D gives real and generated points about the same, ${q(L.dReal, 2)} and ${q(L.dFake, 2)}, near the ${q(0.5, 1)} of a coin toss, because the generator’s points now sit on the clusters. They cover ${q.n(last.covered)} of the ${q.n(8)}.`
        : `After ${q.n(last.step)} steps the generator, learning much faster than its judge, has chased the discriminator’s favourite places instead of spreading out: it covers ${q.n(last.covered)} of the ${q.n(8)} clusters (the equivalent of ${q(last.effective, 1)}). D still gives real points ${q(L.dReal, 2)} and generated ones ${q(L.dFake, 2)}.`, Object.assign({ phase: 'end', step: last.step, covered: last.covered, effective: last.effective, eight: 8, half: 0.5 }, L));
      for (const f of F.list) f.narrowView = { x: 10, y: 20, w: 410, h: 500 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The duel (non-saturating losses)', text: ['D(x): the probability that x is real', 'G(z): a point made from noise z ~ N(0, I)', 'D minimises −log D(real) − log(1 − D(G(z)))', 'G minimises −log D(G(z))', 'at the balance point D = ½ everywhere'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'How many of the 8 clusters will the generator cover at the end?', options: ['0–2', '3–5', '6–8'], answer: (fr) => { const c = fr[fr.length - 1].meta.covered; return c <= 2 ? '0–2' : c <= 5 ? '3–5' : '6–8'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'D(x) = 0.9 for a generated point. What is the generator’s loss −log D(G(z)) for it?', options: ['about 0.105', 'about 2.3', '0.9', '0.1'], answer: 0, why: '−ln 0.9 ≈ 0.105: the generator is already fooling the discriminator there.' },
      { q: 'What does the discriminator output at the ideal balance?', options: ['½ everywhere: it cannot tell real from generated', '1 for real and 0 for fake', '0 everywhere', 'it stops existing'], answer: 0, why: 'When the generator matches the data, no place is more likely real than fake.' },
      { q: 'The generator never sees a real point. How does it learn where they are?', options: ['through the discriminator’s gradient, which points towards places it thinks are real', 'it copies the data directly', 'from a reconstruction loss', 'from labels'], answer: 0, why: 'Its loss is computed by D, so the gradient flows through D into G.' },
      { q: 'Why is the generator’s loss −log D(G(z)) rather than log(1 − D(G(z)))?', options: ['early on D rejects fakes confidently and log(1 − D) would give almost no gradient', 'it is cheaper', 'it makes D learn faster', 'it removes the need for D'], answer: 0, why: 'The non-saturating form keeps the gradient large when the generator is losing.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'gen-mode-collapse', family: 'p14-gan', order: 20, title: 'Mode collapse',
    blurb: 'A generator can fool the discriminator without making the whole distribution: it is enough to make a few kinds of output well. If the duel is unbalanced, the generator keeps jumping to whatever the discriminator currently likes and its samples bunch up on a few modes. Two runs on the same ring, side by side: one balanced, one with a generator that learns 10 times faster than its discriminator. Under each, how many of its samples land near each of the eight clusters.',
    inputs: [{ id: 'r', kind: 'slider', label: 'Near a cluster: within r standard deviations', def: 3, min: 2, max: 6, step: 0.5, fmt: (v) => String(v) }],
    presets: [{ label: 'Within 3 standard deviations', values: { r: 3 } }, { label: 'A strict count: within 2', values: { r: 2 } }, { label: 'A loose count: within 5', values: { r: 5 } }],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const A = P.ganRun('good'), B = P.ganRun('collapse'), data = P.ringData(), F = H().frames(['step']), q = F.q, rad = inp.r * data.sd;
      const steps = A.snaps.map((s) => s.step);
      const bars = (pre, x, cov) => {
        const items = [], max = Math.max(32, ...cov.counts), bw = 30;
        cov.counts.forEach((c, i) => {
          const hgt = (c / max) * 70, miss = c < Math.max(1, 0.2 * (256 / 8));
          if (hgt > 0.5) items.push({ key: pre + ':b' + i, kind: 'cell', x: x + i * (bw + 8), y: 500 - hgt, w: bw, h: hgt, rx: 2, text: '', cls: 'p14bar' + (miss ? ' miss' : ''), audit: 'ignore' });
          items.push(H().text(pre + ':v' + i, x + i * (bw + 8) + bw / 2, 500 - hgt - 6, String(c), 'p14num' + (miss ? ' bad' : ''), 12, 'middle'));
        });
        items.push({ key: pre + ':base', kind: 'path', d: `M${x - 4},500 h${8 * (bw + 8)}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        items.push(H().text(pre + ':n', x, 522, 'samples near each cluster', 'axname', 12));
        return items;
      };
      steps.forEach((step, j) => {
        F.set('step', step);
        const sa = A.snaps[j], sb = B.snaps[j], ca = S.Gen.coverage(sa.samples, data.modes, rad), cb = S.Gen.coverage(sb.samples, data.modes, rad);
        const missA = ca.counts.map((c, i) => (c < Math.max(1, 0.2 * (256 / 8)) ? i : -1)).filter((i) => i >= 0), missB = cb.counts.map((c, i) => (c < Math.max(1, 0.2 * (256 / 8)) ? i : -1)).filter((i) => i >= 0);
        const pa = P.ganPlane('ma', 40, 50, 330, sa, { field: false, miss: missA }), pb = P.ganPlane('mb', 450, 50, 330, sb, { field: false, miss: missB });
        const items = [...pa.items, ...pb.items, H().text('ma:h', 40, 24, 'Balanced', 'strong', 14), H().text('mb:h', 450, 24, 'Generator 10 times faster', 'strong', 14), ...bars('ba', 50, ca), ...bars('bb', 460, cb)];
        const say = step === 0
          ? `Both generators start as a blob in the middle: within ${q(inp.r, 1)} standard deviations of a cluster lie ${q.n(ca.counts.reduce((s, c) => s + c, 0))} and ${q.n(cb.counts.reduce((s, c) => s + c, 0))} of their ${q.n(256)} samples. A cluster counts as covered when at least a fifth of an equal share of samples (${q(0.2 * (256 / 8), 1)}) lands near it.`
          : `Step ${q.n(step)}: the balanced run covers ${q.n(ca.covered)} of the ${q.n(8)} clusters (its samples spread over the equivalent of ${q(ca.effective, 1)}); the unbalanced one covers ${q.n(cb.covered)} (the equivalent of ${q(cb.effective, 1)}).`;
        F.push(items, say, { phase: step ? 'snap' : 'start', step, r: inp.r, a: ca.counts, b: cb.counts, coveredA: ca.covered, coveredB: cb.covered, effA: ca.effective, effB: cb.effective, n: 256, eight: 8, fair: 0.2 * (256 / 8), nearA: ca.counts.reduce((s, c) => s + c, 0), nearB: cb.counts.reduce((s, c) => s + c, 0) });
      });
      const m = F.last().meta;
      F.push(F.last().items, `The balanced generator ends covering ${q.n(m.coveredA)} clusters, the unbalanced one ${q.n(m.coveredB)}. ` + (m.coveredB < m.coveredA ? 'Fooling the discriminator on a few modes was enough for it: the samples look real, but the variety is gone. Remedies balance the duel: comparable learning rates, larger batches, or losses that reward variety.' : 'At this count both look alike; tighten r to see the difference in how the samples sit.'), Object.assign({}, m, { phase: 'end' }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Counting modes', text: ['near a cluster: within r × sd of its centre (sd = ' + data.sd + ')', 'covered: at least ⅕ of an equal share near it', 'effective modes: e to the entropy of the shares'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which run will cover fewer clusters at the end?', options: ['balanced', 'generator 10 times faster', 'the same'], answer: (fr) => { const m = fr[fr.length - 1].meta; return m.coveredB < m.coveredA ? 'generator 10 times faster' : m.coveredA < m.coveredB ? 'balanced' : 'the same'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What is mode collapse?', options: ['the generator produces only a few kinds of output, though each may look real', 'the discriminator stops learning', 'the loss becomes NaN', 'the generator copies the training data'], answer: 0, why: 'Each sample can fool the discriminator while the variety of the data is lost.' },
      { q: 'Why can a generator that learns much faster than its discriminator collapse?', options: ['it rushes to whatever region the slow discriminator likes now, then all its samples move together', 'it runs out of noise', 'its gradients vanish', 'the data are too simple'], answer: 0, why: 'Its samples chase a moving target as a group instead of spreading over the data.' },
      { q: 'Samples spread evenly over 8 modes. What is the effective number of modes, e^entropy?', options: ['8', '1', '3', 'log 8'], answer: 0, why: 'Entropy log 8 for equal shares, so e^entropy = 8.' },
      { q: 'Which is NOT a usual remedy for mode collapse?', options: ['training the generator much faster than the discriminator', 'balancing the learning rates', 'larger batches', 'losses that reward variety'], answer: 0, why: 'That is how the collapse in this picture was made.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  /** a stored image-GAN run: 'steady' (the steadied duel) or 'plain' (no remedies), with 64 samples at the end */
  P.ganImg = (id) => P.once('ganImg' + (id || 'steady'), () => { const J = P.D()[id === 'plain' ? 'ganImgPlain' : 'ganImg']; return { id: id || 'steady', steps: J.steps, hist: J.hist, o: J.o, final64: P.imgs(J.final64), snaps: J.snaps.map((s) => ({ step: s.step, images: P.imgs(s.images), dFake: s.dFake, dReal: s.dReal })) }; });
  P.ganReal = () => P.once('ganReal', () => { const J = P.D().ganImg, set = S.Gen.images('digits', { n: 512, size: 16, seed: 1, classes: J.o.classes }); return { images: [...Array(64).keys()].map((i) => set.X.subarray(i * 256, (i + 1) * 256)), y: set.y.slice(0, 64) }; });
  /** the digit reader: S.Gen.judgeNet as trained, its weights from the data, and its accuracy on 1000 new digits */
  P.judge = () => P.once('judge', () => {
    const J = P.D().judge, jn = S.Gen.judgeNet(J.o);
    S.Gen.setParams(jn.net, J.weights.map(P.i16));
    const test = S.Gen.images('digits', { n: 1000, size: 16, seed: 78 });
    jn.acc = jn.accuracy(test); jn.nTest = test.n; jn.hits = Math.round(jn.acc * test.n);
    return jn;
  });
  P.read = (imgs) => P.judge().read(imgs);
  P.TEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  /** "21 × 0, 20 × 1 and 23 × 7" for the digits read at least once */
  P.readList = (q, counts) => H().list(counts.map((c, d) => [c, d]).filter((x) => x[0] > 0).map((x) => q.n(x[0]) + ' × ' + q.n(x[1])));
  const GAN_RUNS = [{ value: 'steady', label: 'steadied: label smoothing, two learning rates, batch spread, spectral norm' }, { value: 'plain', label: 'plain: none of them (it collapses)' }];

  S.topic({
    id: 'gen-gan-digits', family: 'p14-gan', order: 30, title: 'A GAN on small digits',
    blurb: 'The same duel on images: the generator turns 16 random numbers into a 16×16 image, and the discriminator looks at an image and says how likely it is to be a real digit. Both were trained on 0s, 1s and 7s. Under each image is the discriminator’s probability that it is real; the same 16 noise vectors are used at every step, so you can watch each image grow. At the end a separate digit reader, a classifier trained on real digits, counts what 64 fresh samples look like. The plain duel collapses onto one digit; the steadied one keeps all three.',
    inputs: [
      { id: 'run', kind: 'choice', label: 'Training', def: 'steady', options: GAN_RUNS },
      { id: 'pick', kind: 'number', label: 'Image to follow (1–16)', def: 1, min: 1, max: 16 },
    ],
    presets: [{ label: 'The steadied duel', values: { run: 'steady', pick: 1 } }, { label: 'The plain duel collapses', values: { run: 'plain', pick: 1 } }, { label: 'Follow image 6', values: { run: 'steady', pick: 6 } }],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const R0 = P.ganImg(inp.run), other = P.ganImg(inp.run === 'plain' ? 'steady' : 'plain'), real = P.ganReal(), F = H().frames(['step']), q = F.q, j = inp.pick - 1, h = R0.hist, jn = P.judge();
      const grid = (pre, x, y, imgs, probs, lit) => {
        const out = [];
        imgs.forEach((im, i) => {
          const gx = x + (i % 4) * 78, gy = y + Math.floor(i / 4) * 92;
          out.push(P.img(pre + ':' + i, gx, gy, 62, im, i === lit ? 'act' : ''));
          if (probs) out.push(H().text(pre + ':p' + i, gx + 31, gy + 78, H().fx(probs[i], 2), 'p14num' + (probs[i] >= 0.5 ? ' good' : ''), 12, 'middle'));
        });
        return out;
      };
      const ax = C().axes({ prefix: 'gd', x: 380, y: 432, w: 400, h: 70, xr: [0, R0.steps], yr: [0, 1], xticks: P.ticksN(R0.steps), yticks: [0, 1], xname: 'step', yname: 'D’s average output on its batch' });
      for (const s of R0.snaps) {
        F.set('step', s.step);
        const k = s.step / 10, items = [H().text('h:g', 40, 30, 'Generated, step ' + s.step, 'strong', 14), H().text('h:r', 400, 30, 'Real training digits', 'strong', 14), ...grid('g', 40, 50, s.images, s.dFake, j), ...grid('rl', 420, 50, real.images.slice(0, 16), s.dReal), ...ax.items];
        if (k) items.push(C().line(ax, 'gd:r', h.dReal.slice(0, k).map((v, i) => [(i + 1) * 10, v]), 'p14d'), C().line(ax, 'gd:f', h.dFake.slice(0, k).map((v, i) => [(i + 1) * 10, v]), 'p14g'), C().vline(ax, 'gd:now', s.step, 'act'));
        const mf = s.dFake.reduce((a, b) => a + b, 0) / 16, mr = s.dReal.reduce((a, b) => a + b, 0) / 16, ink = P.ink(s.images[j]), fooled = s.dFake.filter((v) => v >= 0.5).length;
        const rd = P.read(s.images);
        const say = !s.step
          ? `Before training the generator turns each noise vector into a grey fog (image ${q.n(inp.pick)} has ink ${q(ink, 1)}), and the untrained discriminator gives the fakes ${q(mf, 2)} and the real digits ${q(mr, 2)} on average.`
          : `Step ${q.n(s.step)}: the discriminator gives these ${q.n(16)} fakes ${q(mf, 2)} on average and the ${q.n(16)} real digits ${q(mr, 2)}; ${q.n(fooled)} fakes pass as real (at least ${q(0.5, 1)}). The digit reader takes the fakes for ${P.readList(q, rd.counts)}, ${q.n(rd.sure)} of them with probability ${q(0.9, 1)} or more. Image ${q.n(inp.pick)} has ink ${q(ink, 1)}, and D gives it ${q(s.dFake[j], 2)}.`;
        F.push(items, say, { phase: s.step ? 'snap' : 'start', step: s.step, mf, mr, ink, d: s.dFake[j], fooled, sixteen: 16, half: 0.5, pick: inp.pick, counts: rd.counts, sure: rd.sure, nine: 0.9, classes: R0.o.classes, digits: P.TEN });
      }
      // 64 fresh samples, counted by the digit reader, against 64 real training digits
      const rd = P.read(R0.final64), rr = P.read(real.images), ro = P.read(other.final64), last = R0.snaps[R0.snaps.length - 1];
      const items = [H().text('h:g', 40, 30, '64 fresh samples after ' + R0.steps + ' steps', 'strong', 14), H().text('h:b', 430, 30, 'What the digit reader takes them for', 'strong', 14)];
      R0.final64.forEach((im, i) => items.push(P.img('t:' + i, 40 + (i % 8) * 40, 50 + Math.floor(i / 8) * 40, 34, im, 'tile')));
      const top = Math.max(32, ...rd.counts, ...rr.counts), sc = 230 / top;
      for (let d = 0; d < 10; d++) {
        const y = 60 + d * 30, c = rd.counts[d], cs = rd.sureCounts[d], cr = rr.counts[d];
        items.push(H().text('t:d' + d, 456, y + 13, String(d), 'axname', 13, 'end'));
        if (cs) items.push({ key: 't:bs' + d, kind: 'cell', x: 470, y, w: cs * sc, h: 12, rx: 1, text: '', cls: 'p14bar', audit: 'ignore' });
        if (c > cs) items.push({ key: 't:bu' + d, kind: 'cell', x: 470 + cs * sc, y, w: (c - cs) * sc, h: 12, rx: 1, text: '', cls: 'p14bar soft', audit: 'ignore' });
        if (cr) items.push({ key: 't:br' + d, kind: 'cell', x: 470, y: y + 14, w: cr * sc, h: 6, rx: 1, text: '', cls: 'p14bar real', audit: 'ignore' });
        if (c || cr) items.push(H().text('t:n' + d, 476 + Math.max(c, cr) * sc, y + 15, c + ' / ' + cr, 'p14num', 12));
      }
      items.push({ key: 't:base', kind: 'path', d: 'M470,54 V364', cls: 'axis', audit: 'ignore', layer: 'back' });
      items.push(...C().legend('t:lg', 430, 400, [{ text: 'generated, read with p ≥ 0.9', cls: 'p14fake', dot: true }, { text: 'generated, less sure', cls: 'p14soft', dot: true }, { text: 'real', cls: 'p14real', dot: true }], { dir: 'row', size: 12 }));
      items.push(H().text('t:k', 430, 380, 'each row: generated / real, of 64', 'axname', 12));
      const stray = rd.counts.reduce((a, c, d) => (R0.o.classes.includes(d) ? a : a + c), 0);
      F.push(items, `Sixty-four fresh samples, read by a digit reader: a small classifier trained on our drawn digits, which reads ${q.n(jn.hits)} of ${q.n(jn.nTest)} new real digits correctly. It takes the samples for ${P.readList(q, rd.counts)}` + (stray ? ` (${q.n(stray)} read as digits it was never shown)` : '') + `, and is sure of ${q.n(rd.sure)} of the ${q.n(64)} (probability ${q(0.9, 1)} or more). ${q.n(64)} real training digits read as ${P.readList(q, rr.counts)}. Measured as e to the entropy of the counts, the samples spread over the equivalent of ${q(rd.effective, 2)} kinds of digit, the real ones over ${q(rr.effective, 2)}.`,
        { phase: 'tally', step: R0.steps, counts: rd.counts, sureCounts: rd.sureCounts, sure: rd.sure, eff: rd.effective, realCounts: rr.counts, effReal: rr.effective, acc: jn.acc, hits: jn.hits, nTest: jn.nTest, n64: 64, nine: 0.9, stray, classes: R0.o.classes, digits: P.TEN });
      const mf = last.dFake.reduce((a, b) => a + b, 0) / 16, mr = last.dReal.reduce((a, b) => a + b, 0) / 16, topC = Math.max(...rd.counts), td = rd.counts.indexOf(topC), kinds = R0.o.classes.filter((d) => rd.counts[d] > 0).length;
      const meta = { phase: 'end', step: R0.steps, mf, mr, counts: rd.counts, sure: rd.sure, eff: rd.effective, top: topC, td, kinds, three: R0.o.classes.length, n64: 64, otherCounts: ro.counts, otherEff: ro.effective, otherSteps: other.steps, classes: R0.o.classes, digits: P.TEN };
      F.push(F.last().items, inp.run === 'plain'
        ? `The plain duel: equal, fast learning rates, hard labels, and a discriminator that judges each image alone. After ${q.n(R0.steps)} steps ${q.n(topC)} of ${q.n(64)} samples read as ${P.aQ(q, td)}, though the training digits are ${H().list(R0.o.classes.map((d) => q.n(d) + 's'))}: the generator found one kind of digit that fools its judge and stayed there, the mode collapse of [[gen-mode-collapse]] on images. The steadied duel, on the same digits, ends with ${P.readList(q, ro.counts)} (the equivalent of ${q(ro.effective, 2)} kinds).`
        : `After ${q.n(R0.steps)} steps the generator draws ${kinds === R0.o.classes.length ? 'all ' : ''}${q.n(kinds)} of the ${q.n(R0.o.classes.length)} digits it was shown: of ${q.n(64)} samples, ${H().list(R0.o.classes.map((d) => q.n(rd.counts[d]) + ' read as ' + q.n(d)))}, against ${H().list(R0.o.classes.map((d) => q.n(rr.counts[d])))} among ${q.n(64)} real digits. D gives its fakes ${q(mf, 2)} against ${q(mr, 2)} for real digits. The plain duel, with none of the four remedies, collapsed: after ${q.n(other.steps)} steps its ${q.n(64)} samples read as ${P.readList(q, ro.counts)}.`, Object.assign(meta, { realCounts: rr.counts }));
      for (const f of F.list) f.narrowView = { x: 20, y: 10, w: 340, h: 420 };
      const steady = inp.run !== 'plain';
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The networks', text: (steady
        ? ['G: 16 numbers of noise → 128 → 256 → 256 pixels (sigmoid)', 'D: 256 pixels → 256 → (+ the batch’s spread) → 128 → 1', 'D’s target for real images: 0.9, not 1', 'learning rates: D 4 × 10⁻⁴, G 10⁻⁴', 'each of D’s weight matrices scaled to a largest singular value of at most 1']
        : ['G: 16 numbers of noise → 128 → 128 → 256 pixels (sigmoid)', 'D: 256 pixels → 128 → 128 → 1 (the logit of "real")', 'learning rates: 2 × 10⁻³ for both', 'the same losses as on the ring']).join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Of the three digits it is shown (0, 1 and 7), how many will the generator still draw at the end?', options: ['1', '2', '3'], answer: (fr) => String(fr[fr.length - 1].meta.kinds), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Why are the same 16 noise vectors used at every snapshot?', options: ['so each image shows how one input’s output changes as the generator learns', 'the generator accepts only 16 inputs', 'to save memory', 'random noise would be illegal'], answer: 0, why: 'A fixed input isolates the change in the network.' },
      { q: 'The discriminator gives a fake 0.02. What does that tell the generator?', options: ['its gradient says strongly how to change that image towards looking real', 'the image is perfect', 'nothing', 'to stop training'], answer: 0, why: 'With the non-saturating loss −log 0.02 ≈ 3.9 the push is large.' },
      { q: 'Real digits score 0.95 and fakes 0.05 for many steps. What is wrong?', options: ['the discriminator has won; the generator is not learning from it', 'nothing, this is the goal', 'the generator has won', 'the data are wrong'], answer: 0, why: 'A duel where one side wins outright stops being informative.' },
      { q: 'What does the generator’s last layer use, so its outputs are pixels in [0, 1]?', options: ['a sigmoid', 'a ReLU', 'a softmax over pixels', 'nothing'], answer: 0, why: 'The sigmoid squashes each pixel into the range of the data.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
