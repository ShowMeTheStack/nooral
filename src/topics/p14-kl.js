// Part 14, the KL term against reconstruction: four VAEs trained alike except for the weight β on the KL term
// (stored runs of the engine), their latent maps as training goes, their curves, and what each decodes across the plane.
(function (S) {
  'use strict';
  const P = S.P14, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const BETAS = [0, 1, 4, 16];

  P.vaeRun = (b) => P.once('vaeRun' + b, () => {
    const J = P.D()['vae' + b];
    return { beta: b, steps: J.steps, hist: J.hist, final: J.final, grid: P.imgs(J.grid), snaps: J.snaps.map((s) => ({ step: s.step, elbo: s.elbo, recon: s.recon, kl: s.kl, z: P.pairs(P.i16(s.map)), y: [...s.y].map(Number) })) };
  });
  /** the mean distance of the codes from the centre */
  P.spread = (z) => z.reduce((s, p) => s + Math.hypot(p[0], p[1]), 0) / z.length;

  S.topic({
    id: 'gen-kl', family: 'p14-ae', order: 40, title: 'The KL term against reconstruction',
    blurb: 'A VAE’s loss has two parts that pull against each other. The reconstruction term wants each digit’s code precise and far from the others, so the decoder can rebuild it exactly. The KL term wants every code’s distribution to look like the prior N(0, I), crowded round the centre. Weighting the KL term by β moves the balance: four VAEs here differ only in β.',
    inputs: [{ id: 'beta', kind: 'choice', label: 'Weight β on the KL term', def: '1', options: BETAS.map((b) => ({ value: String(b), label: 'β = ' + b + (b === 0 ? ' (no KL: a plain autoencoder with noise)' : b === 1 ? ' (the standard VAE)' : '') })) }],
    presets: BETAS.map((b) => ({ label: b === 0 ? 'β = 0: no KL term' : b === 1 ? 'β = 1: the standard VAE' : b === 4 ? 'β = 4: a stronger pull' : 'β = 16: the pull wins', values: { beta: String(b) } })),
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const b = +inp.beta, R0 = P.vaeRun(b), all = BETAS.map(P.vaeRun), F = H().frames(['step']), q = F.q;
      const zmax = Math.max(3, ...R0.snaps.flatMap((s) => s.z.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))));
      const Rg = Math.ceil(zmax), ax = P.latentAxes('kl', 50, 50, 300, Rg);
      const n = R0.hist.recon.length, rmax = Math.max(...R0.hist.recon), kmax = Math.max(...R0.hist.kl, 1e-3);
      const ar = C().axes({ prefix: 'kr', x: 450, y: 50, w: 320, h: 110, xr: [0, R0.steps], yr: [0, rmax * 1.05], xticks: P.ticksN(R0.steps), yticks: C().ticks(0, rmax, 2), yname: 'reconstruction (nats)' });
      const ak = C().axes({ prefix: 'kk', x: 450, y: 220, w: 320, h: 110, xr: [0, R0.steps], yr: [0, kmax * 1.05], xticks: P.ticksN(R0.steps), yticks: C().ticks(0, kmax, 2), xname: 'step', yname: 'KL (nats)' });
      const dots = (s) => s.z.map((p, i) => ({ key: 'kl:d' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 3.4, cls: 'sp ' + P.cOf(s.y[i]), audit: 'overlap-ok', layer: 'items' }));
      const lg = C().legend('lg', 50, 420, P.VAE_CLASSES.map((d) => ({ text: 'digit ' + d, cls: P.cOf(d), dot: true })), { dir: 'row', size: 13 });
      const head = H().text('kl:h', 50, 22, 'The codes μ of the test digits, β = ' + b, 'strong', 14);
      for (const s of R0.snaps) {
        F.set('step', s.step);
        const k = Math.round(s.step / 10), items = [...ax.items, ...P.rings(ax, 'kl:p'), ...dots(s), ...lg, head, ...ar.items, ...ak.items];
        if (k) {
          items.push(C().line(ar, 'kr:l', R0.hist.recon.slice(0, k).map((v, j) => [(j + 1) * 10, v]), 'p14rec'), C().line(ak, 'kk:l', R0.hist.kl.slice(0, k).map((v, j) => [(j + 1) * 10, v]), 'p14kl'));
          items.push(C().vline(ar, 'kr:now', s.step, 'act'), C().vline(ak, 'kk:now', s.step, 'act'));
        }
        const sp = P.spread(s.z), loss = s.recon + b * s.kl;
        const say = s.step === 0
          ? `Before training, with β = ${q.n(b)}: on the test digits the reconstruction loss is ${q.sg(s.recon, 3)} nats and the KL ${q.sg(s.kl, 3)}. The untrained encoder puts every code near the same place, a mean ${q(sp, 2)} from the centre.`
          : `Step ${q.n(s.step)}: reconstruction ${q.sg(s.recon, 3)} nats, KL ${q.sg(s.kl, 3)} nats, so the loss reconstruction + β × KL is ${q.sg(loss, 3)}. The codes lie a mean ${q(sp, 2)} from the centre.`;
        F.push(items, say, { phase: s.step ? 'snap' : 'start', step: s.step, beta: b, recon: s.recon, kl: s.kl, loss, spread: sp });
      }
      // the end: what the decoder draws across the plane, and all four βs side by side
      const last = R0.snaps[R0.snaps.length - 1], gw = 300 / 7, items = [...ax.items, ...P.rings(ax, 'kl:p'), ...dots(last), ...lg, head];
      R0.grid.forEach((im, k) => items.push(P.img('kg:' + k, 450 + (k % 7) * gw + 1, 50 + Math.floor(k / 7) * gw + 1, gw - 2, im, 'tile')));
      items.push(H().text('kg:h', 450, 22, 'Decoded across [−2.5, 2.5]²', 'strong', 14));
      const rec = all.map((r) => r.final.recon), kl = all.map((r) => r.final.kl);
      const bars = (pre, x, vals, title, cls) => [H().text(pre + ':h', x, 376, title, 'axname', 13), ...S.Kit.dt.hbars({ prefix: pre, x, y: 384, w: 170, rowH: 21, vals, names: BETAS.map((x) => 'β = ' + x), max: Math.max(...vals), fmt: (v) => H().sg(v, 3), cls: (j) => cls + (BETAS[j] === b ? ' act' : ''), nameCls: (j) => (BETAS[j] === b ? 'strong' : ''), nameW: 56, valW: 40, size: 12 })];
      items.push(...bars('br', 440, rec, 'reconstruction at the end', 'p14rec'), ...bars('bk', 630, kl, 'KL at the end', 'p14kl'));
      F.set('step', R0.steps);
      F.push(items, `After ${q.n(R0.steps)} steps, on the test digits: reconstruction ${H().list(BETAS.map((x, j) => q.sg(rec[j], 3) + ' at β = ' + q.n(x)))} nats; KL ${H().list(BETAS.map((x, j) => q.sg(kl[j], 3)))}. ` + (b === 0 ? 'Without the KL term the codes spread where they like, and the grid of decoded squares round the centre holds gaps the decoder never learned.' : b >= 16 ? 'With a heavy KL term the codes crowd round the centre and blur together: the decoder draws vague averages.' : 'The decoded grid changes smoothly from digit to digit across the region the prior covers.'), { phase: 'end', step: R0.steps, beta: b, betas: BETAS, rec, kl });
      for (const f of F.list) f.narrowView = { x: 10, y: 0, w: 380, h: 450 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The β-weighted loss', text: ['loss = reconstruction + β · KL', 'reconstruction = − log p(x | z), summed over pixels', 'KL = ½ Σ (μ² + σ² − 1 − log σ²)', 'β = ' + b].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which β will end with the smallest KL?', options: BETAS.map((b) => 'β = ' + b), answer: (fr) => { const k = fr[fr.length - 1].meta.kl; return 'β = ' + BETAS[k.indexOf(Math.min(...k))]; }, explain: (fr) => fr[fr.length - 1].say },
      { at: (f, i) => i === 2, pick: 'option', ask: 'Which β will rebuild the test digits best?', options: BETAS.map((b) => 'β = ' + b), answer: (fr) => { const r = fr[fr.length - 1].meta.rec; return 'β = ' + BETAS[r.indexOf(Math.min(...r))]; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What does raising β do to a VAE?', options: ['pulls the codes harder towards the prior, at the cost of sharper reconstructions', 'makes reconstructions sharper', 'makes training faster', 'removes the noise from z'], answer: 0, why: 'The KL term weighs more, so the encoder gives up detail to stay close to N(0, I).' },
      { q: 'With β = 0, why can sampling z ~ N(0, I) give poor images?', options: ['the codes need not fill the prior’s region, so many draws land where the decoder never learned', 'the decoder is untrained', 'the reconstruction loss is zero', 'z has no noise'], answer: 0, why: 'Nothing pulls the codes towards the prior, so it no longer describes where they are.' },
      { q: 'An encoder outputs μ = 0 and σ = 1 for every input. What is its KL, and what is wrong?', options: ['0, but the code then carries nothing about the input', 'large, and the images are sharp', '1, nothing is wrong', 'negative'], answer: 0, why: 'This is posterior collapse: the KL is as small as can be, and the decoder must ignore z.' },
      { q: 'In the β = 1 run, the loss is reconstruction 40 plus KL 4. What is the ELBO?', options: ['−44', '44', '−36', '36'], answer: 0, why: 'ELBO = −(reconstruction + KL).' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
