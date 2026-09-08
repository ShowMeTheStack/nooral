// Part 14, the families compared: what an autoencoder, a VAE, a GAN and a diffusion model each make, how many
// network calls one new sample costs, and how varied their samples are, from the runs of the other topics.
(function (S) {
  'use strict';
  const P = S.P14, H = () => S.Kit.ml;

  /** which of a set of digits each image correlates with best, against the average images of those digits */
  P.classify = (imgs, means) => imgs.map((im) => { let b = -1, bc = -Infinity; for (const [d, m] of Object.entries(means)) { const c = P.corr(im, m); if (c > bc) { bc = c; b = +d; } } return b; });
  P.meansOf = (X, y, digits) => Object.fromEntries(digits.map((d) => { const m = new Float64Array(256); let n = 0; y.forEach((c, i) => { if (c !== d) return; n++; for (let k = 0; k < 256; k++) m[k] += X[i * 256 + k]; }); return [d, Float64Array.from(m, (v) => v / n)]; }));
  P.ganMeans = () => P.once('ganMeans', () => { const J = P.D().ganImg, set = S.Gen.images('digits', { n: 512, size: 16, seed: 1, classes: J.o.classes }); return P.meansOf(set.X, set.y, J.o.classes); });
  const tally = (cls, digits) => digits.map((d) => cls.filter((c) => c === d).length);

  const FAM = [
    { id: 'ae', name: 'Autoencoder', how: 'rebuilds its input', calls: 1 },
    { id: 'vae', name: 'VAE', how: 'decodes z ~ N(0, I)', calls: 1 },
    { id: 'gan', name: 'GAN', how: 'generates from noise', calls: 1 },
    { id: 'diff', name: 'Diffusion', how: 'denoises step by step', calls: null },
  ];

  S.topic({
    id: 'gen-compare', family: 'p14-families', order: 10, title: 'Four generative families side by side',
    blurb: 'Each family learns the data in a different way. An autoencoder learns to copy through a bottleneck; a VAE adds a prior so that its code space can be sampled; a GAN learns from a critic that judges its samples; a diffusion model learns to undo noise, one small step at a time. Here are samples from the models trained in the other topics, what one sample costs, and how varied the samples are.',
    inputs: [{ id: 'seed', kind: 'number', label: 'Seed for the VAE’s draws', def: 3, min: 1, max: 999 }],
    presets: [{ label: 'Seed 3', values: { seed: 3 } }, { label: 'Seed 40', values: { seed: 40 } }, { label: 'Seed 500', values: { seed: 500 } }],
    counters: [{ id: 'fam', label: 'families seen' }],
    run(inp) {
      const F = H().frames(['fam']), q = F.q, v = P.vae(), g = S.T.rng(inp.seed);
      const zs = [...Array(8).keys()].map(() => [g.normal(), g.normal()]), vaeImgs = v.decode(zs), vaeCls = P.classify(vaeImgs, P.vaeMeans());
      const gan = P.ganImg('steady'), ganImgs = gan.final64, ganRd = P.read(ganImgs), plainRd = P.read(P.ganImg('plain').final64);
      const ae = P.aeRun(4), aeLast = ae.snaps[ae.snaps.length - 1], test = P.aeTest(), di = P.diffImg(), diffImgs = di.frames[di.frames.length - 1].images, T = di.sched.T;
      const cards = (on) => FAM.flatMap((f, i) => [{ key: 'fc:' + f.id, kind: 'cell', x: 40 + i * 190, y: 30, w: 176, h: 58, rx: 8, lines: [f.name, f.how], size: 13, cls: 'p14fam' + (i === on ? ' act' : '') }]);
      const grid = (pre, imgs, x, y, w, cols, lab) => imgs.flatMap((im, i) => [P.img(pre + i, x + (i % cols) * (w + 10), y + Math.floor(i / cols) * (w + (lab ? 28 : 10)), w, im)].concat(lab ? [H().text(pre + 't' + i, x + (i % cols) * (w + 10) + w / 2, y + Math.floor(i / cols) * (w + 28) + w + 16, lab[i], 'p14num', 12, 'middle')] : []));
      const side = (rows) => H().lines('sd', 560, 140, rows, 13, 22);
      F.set('fam', 1);
      const aeErr = aeLast.recon.map((im, j) => P.mae(im, test.X.subarray(j * 256, (j + 1) * 256)));
      F.push([...cards(0), H().text('h', 40, 122, 'Rebuilt test digits (code of 4)', 'strong', 14), ...grid('ae', [...Array(4).keys()].map((j) => test.X.subarray(j * 256, (j + 1) * 256)), 40, 140, 60, 4), ...grid('ar', aeLast.recon.slice(0, 4), 40, 216, 60, 4), ...side([['input → code → output', 'legend'], ['1 network call', 'legend'], ['makes no new samples', 'legend']])],
        `An autoencoder only rebuilds what it is given: after ${q.n(ae.steps)} steps its rebuilt digits (bottom) are off by ${q(aeErr.reduce((a, b) => a + b, 0) / aeErr.length, 3)} per pixel on average. Its code space has no known shape, so there is no safe place to draw a new code from.`, { phase: 'ae', steps: ae.steps, err: aeErr.reduce((a, b) => a + b, 0) / aeErr.length });
      F.set('fam', 2);
      const vt = tally(vaeCls, P.VAE_CLASSES);
      F.push([...cards(1), H().text('h', 40, 122, 'Eight draws z ~ N(0, I), decoded', 'strong', 14), ...grid('vs', vaeImgs, 40, 140, 60, 4, vaeCls.map((d) => 'like ' + d)), ...side([['z ~ N(0, I) → decoder', 'legend'], ['1 network call', 'legend'], ['blurred but varied', 'legend']])],
        `A VAE samples by drawing z from its prior and decoding: one network call. These eight look most like ${H().list(P.VAE_CLASSES.map((d, i) => q.n(vt[i]) + ' × ' + q.n(d)))}; its samples are varied but soft-edged, since the decoder is trained to hedge between nearby codes.`, { phase: 'vae', tally: vt, digits: P.VAE_CLASSES });
      F.set('fam', 3);
      const kinds = gan.o.classes.filter((d) => ganRd.counts[d] > 0).length;
      F.push([...cards(2), H().text('h', 40, 122, 'Eight of the GAN’s 64 samples', 'strong', 14), ...grid('gs', ganImgs.slice(0, 8), 40, 140, 60, 4, ganRd.digit.slice(0, 8).map((d) => 'like ' + d)), ...side([['noise → generator', 'legend'], ['1 network call', 'legend'], ['sharp, but may collapse', 'legend']])],
        `A GAN also needs one network call per sample, and its samples are sharp. A digit reader takes its ${q.n(ganImgs.length)} samples for ${P.readList(q, ganRd.counts)}: ` + (kinds === gan.o.classes.length ? `all ${q.n(kinds)} digits it was trained on, thanks to the steadied duel of [[gen-gan-digits]]. The plain duel collapsed: its ${q.n(64)} samples read as ${P.readList(q, plainRd.counts)}.` : `only ${q.n(kinds)} of the ${q.n(gan.o.classes.length)} digits it was trained on, a mode collapse.`),
        { phase: 'gan', counts: ganRd.counts, plainCounts: plainRd.counts, kinds, three: gan.o.classes.length, digits: gan.o.classes, ten: P.TEN, n: ganImgs.length, n64: 64 });
      F.set('fam', 4);
      F.push([...cards(3), H().text('h', 40, 122, 'Diffusion samples (circles)', 'strong', 14), ...grid('ds', diffImgs, 40, 140, 60, 4), ...side([['noise → ' + T + ' denoising steps', 'legend'], [T + ' network calls', 'legend'], ['stable to train, slow to sample', 'legend']])],
        `A diffusion model pays for its quality at sampling time: each image took ${q.n(T)} calls of the denoiser. Its training is the steadiest of the four, a plain squared error with no duel, and it has no prior to fit and no critic to balance.`, { phase: 'diff', T });
      const rows = [['', 'calls per sample', 'trained by', 'risk'], ['Autoencoder', 'none', 'reconstruction', 'no sampling'], ['VAE', '1', 'reconstruction + KL', 'blur'], ['GAN', '1', 'a duel with a critic', 'mode collapse'], ['Diffusion', String(T), 'predicting noise', 'slow sampling']];
      const tb = S.Kit.dt.table({ prefix: 'tb', x: 40, y: 130, cols: [{ name: 'family', w: 130 }, { name: 'calls per sample', w: 140 }, { name: 'trained by', w: 210 }, { name: 'the usual trouble', w: 170 }], rows: rows.slice(1), rowH: 30, size: 13 });
      F.push([...cards(-1), ...tb.items], `In short: the VAE and the GAN make a sample in ${q.n(1)} call, diffusion here in ${q.n(T)}. The VAE trades sharpness for a well-shaped code space, the GAN trades stability and variety for sharpness, and diffusion trades sampling time for both.`, { phase: 'end', T, one: 1 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'What each one optimises', text: ['autoencoder: ‖x − dec(enc(x))‖', 'VAE: reconstruction + KL(q(z|x) ‖ N(0, I))', 'GAN: min over G, max over D of the duel', 'diffusion: ‖ε − ε̂(x_t, t)‖²'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'vae', pick: 'option', ask: 'The GAN was trained on 0s, 1s and 7s. Will its 64 samples include all three?', options: ['yes', 'no'], answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'gan').meta; return m.kinds === m.three ? 'yes' : 'no'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'gan').say },
    ],
    quiz: [
      { q: 'Which family needs the most network calls to make one sample?', options: ['diffusion', 'GAN', 'VAE', 'autoencoder'], answer: 0, why: 'One call per reverse step, often tens to hundreds.' },
      { q: 'Which family’s samples tend to be blurred, and why?', options: ['the VAE’s: its decoder hedges between nearby codes', 'the GAN’s: the critic blurs them', 'diffusion’s: noise remains', 'none'], answer: 0, why: 'Averaging over the code’s noise favours soft, averaged images.' },
      { q: 'Which family has no loss of its own to watch going down, only a balance?', options: ['the GAN', 'the VAE', 'diffusion', 'the autoencoder'], answer: 0, why: 'The two losses of a GAN fight each other; neither falling means success.' },
      { q: 'Why is a plain autoencoder not a generator?', options: ['nothing tells you where in its code space new codes should come from', 'it cannot decode', 'it is too slow', 'it needs labels'], answer: 0, why: 'A VAE fixes this with the prior.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
