// Part 14, the autoencoder: a digit squeezed through a code of 2, 4 or 8 numbers and rebuilt, followed through a
// training run (kept snapshots of the engine's run: the codes and reconstructions of eight test digits, the loss).
(function (S) {
  'use strict';
  const P = S.P14, H = () => S.Kit.ml, C = () => S.Kit.curve;

  /** the stored run for a code of k numbers, read back */
  P.aeRun = (k) => P.once('ae' + k, () => {
    const J = P.D()['ae' + k];
    return { k, steps: J.steps, loss: J.loss, testLoss: J.testLoss, snaps: J.snaps.map((s) => ({ step: s.step, loss: s.loss, recon: P.imgs(s.recon), codes: P.split(Float32Array.from(s.codes), k).map((c) => Array.from(c)) })) };
  });
  /** mean absolute difference of two images */
  P.mae = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };

  /** the funnel: input image, encoder, code cells, decoder, rebuilt image */
  P.funnel = (pre, x, y, input, code, out, cmax, sizes) => {
    const items = [], mid = y + 92, iw = 104, cellH = Math.min(24, 176 / code.length), codeH = cellH * code.length;
    items.push(P.img(pre + ':in', x, mid - iw / 2, iw, input), P.img(pre + ':out', x + 400, mid - iw / 2, iw, out));
    const trap = (key, x0, h0, x1, h1, text) => {
      const pts = [[x0, mid - h0 / 2], [x1, mid - h1 / 2], [x1, mid + h1 / 2], [x0, mid + h0 / 2]];
      items.push({ key, kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + H().r2(p[0]) + ',' + H().r2(p[1])).join(' ') + ' Z', cls: 'p14trap', layer: 'back', z: 1, box: H().boxOf(pts) });
      items.push(H().text(key + ':t', (x0 + x1) / 2, mid + 5, text, 'p14traptext', 12, 'middle'));
    };
    trap(pre + ':enc', x + 118, 150, x + 200, codeH + 8, sizes[0] + '→' + sizes[1]);
    trap(pre + ':dec', x + 304, codeH + 8, x + 386, 150, sizes[1] + '→' + sizes[0]);
    code.forEach((v, i) => {
      const c = S.Kit.heat.colour(v, 'div', cmax);
      items.push({ key: pre + ':code' + i, kind: 'cell', x: x + 214, y: H().r2(mid - codeH / 2 + i * cellH), w: 76, h: cellH - 2, rx: 3, text: H().fx(v, 2), size: Math.min(13, cellH - 9), cls: 'hm p14code', paint: { fill: c.fill, text: c.text } });
    });
    const ly = mid + Math.max(96, codeH / 2 + 30);
    items.push(H().text(pre + ':l0', x + iw / 2, ly, 'input', 'axname', 13, 'middle'), H().text(pre + ':l1', x + 159, ly, 'encoder', 'axname', 13, 'middle'),
      H().text(pre + ':l2', x + 252, ly, 'code', 'axname', 13, 'middle'), H().text(pre + ':l3', x + 345, ly, 'decoder', 'axname', 13, 'middle'), H().text(pre + ':l4', x + 400 + iw / 2, ly, 'rebuilt', 'axname', 13, 'middle'));
    return items;
  };

  S.topic({
    id: 'gen-ae', family: 'p14-ae', order: 10, title: 'The autoencoder’s bottleneck',
    blurb: 'An autoencoder is two networks back to back: an encoder that squeezes an input into a short code, and a decoder that rebuilds the input from the code alone. Trained only to make the rebuilt image match the input, it has to decide what the few numbers of the code should keep. Here 16×16 digits pass through codes of 2, 4 or 8 numbers.',
    inputs: [
      { id: 'k', kind: 'choice', label: 'Code size', def: '4', options: [{ value: '2', label: '2 numbers' }, { value: '4', label: '4 numbers' }, { value: '8', label: '8 numbers' }] },
      { id: 'img', kind: 'number', label: 'Test image to follow (1–8)', def: 1, min: 1, max: 8 },
    ],
    presets: [
      { label: 'A code of 4 numbers', values: { k: '4', img: 1 } },
      { label: 'Squeezed through 2', values: { k: '2', img: 1 } },
      { label: 'Room for 8', values: { k: '8', img: 5 } },
      { label: 'A thin 7 through 2', values: { k: '2', img: 4 } },
    ],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const k = +inp.k, R = P.aeRun(k), test = P.aeTest(), F = H().frames(['step']), q = F.q, i = inp.img - 1;
      const orig = [...Array(8).keys()].map((j) => test.X.subarray(j * 256, (j + 1) * 256)), digit = test.y[i];
      const cmax = Math.max(1e-6, ...R.snaps.flatMap((s) => s.codes.flat().map(Math.abs)));
      const lmax = Math.max(...R.loss), ax = C().axes({ prefix: 'aeL', x: 600, y: 52, w: 190, h: 130, xr: [0, R.steps], yr: [0, lmax * 1.05], xticks: P.ticksN(R.steps), yticks: C().ticks(0, lmax, 3), xname: 'step', yname: 'training loss (nats)' });
      const all = [2, 4, 8].map((kk) => P.aeRun(kk));
      const show = (snap, extra) => {
        const items = [...P.funnel('ae', 40, 20, orig[i], snap.codes[i], snap.recon[i], cmax, [256, 64]), ...ax.items];
        if (snap.step) items.push(C().line(ax, 'aeL:l', R.loss.slice(0, snap.step / 5).map((v, j) => [(j + 1) * 5, v]), 'p14loss'), C().vline(ax, 'aeL:now', snap.step, 'act'));
        items.push(H().text('row:o', 128, 310, 'original', 'axname', 13, 'end'), H().text('row:r', 128, 382, 'rebuilt', 'axname', 13, 'end'), H().text('row:e', 128, 436, 'pixels off by', 'axname', 13, 'end'));
        items.push(...P.row('o', 140, 280, 52, orig, 8), ...P.row('r', 140, 352, 52, snap.recon, 8));
        snap.recon.forEach((im, j) => items.push(H().text('err:' + j, 166 + j * 60, 436, H().fx(P.mae(im, orig[j]), 3), 'p14err' + (j === i ? ' act' : ''), 12, 'middle')));
        items.push(P.ring('ring:o', 140 + i * 60, 280, 52, 'act'), P.ring('ring:r', 140 + i * 60, 352, 52, 'act'));
        return items.concat(extra || []);
      };
      const s0 = R.snaps[0], m0 = s0.recon.reduce((s, im) => s + P.ink(im), 0) / (8 * 256);
      // how much the untrained pixels scatter around that mean: a flat grey has none
      const sd0 = Math.sqrt(s0.recon.reduce((s, im) => s + Array.from(im).reduce((t, v) => t + (v - m0) ** 2, 0), 0) / (8 * 256));
      F.push(show(s0), `The encoder squeezes each ${q.n(16)}×${q.n(16)} digit (${q.n(256)} pixels) through a hidden layer of ${q.n(64)} into a code of ${q.n(k)} numbers; the decoder rebuilds all ${q.n(256)} pixels from the code alone. Before training every rebuilt image is ${sd0 < 0.02 ? 'a flat grey' : 'grey noise'} (mean pixel ${q(m0, 2)}${sd0 < 0.02 ? '' : `, standard deviation ${q(sd0, 2)}`}), and the loss on the eight test digits is ${q.sg(s0.loss, 3)} nats each.`, { phase: 'start', step: 0, size: 16, P: 256, hidden: 64, k, mean: m0, sd: sd0, loss: s0.loss });
      for (const s of R.snaps.slice(1)) {
        F.set('step', s.step);
        const tl = R.loss[s.step / 5 - 1], e = P.mae(s.recon[i], orig[i]);
        F.push(show(s), `Step ${q.n(s.step)}: the training loss is ${q.sg(tl, 3)} nats per image, and ${q.sg(s.loss, 3)} on the eight test digits. Digit ${q.n(inp.img)}, ${P.aQ(q, digit)}, is coded as [${s.codes[i].map((v) => q(v, 2)).join(', ')}]; its rebuilt pixels are off by ${q(e, 3)} on average.`, { phase: 'snap', step: s.step, trainLoss: tl, loss: s.loss, code: s.codes[i], mae: e, img: inp.img, digit });
      }
      const last = R.snaps[R.snaps.length - 1], errs = last.recon.map((im, j) => P.mae(im, orig[j])), worst = errs.indexOf(Math.max(...errs)), tl = all.map((r) => r.testLoss);
      const order = [0, 1, 2].sort((a, b) => tl[a] - tl[b]);
      F.push(show(last, [P.ring('ring:w', 140 + worst * 60, 352, 52, 'bad')]), `After ${q.n(R.steps)} steps the loss on all ${q.n(128)} test digits is ${q.sg(tl[0], 3)} nats with a code of ${q.n(2)} numbers, ${q.sg(tl[1], 3)} with ${q.n(4)} and ${q.sg(tl[2], 3)} with ${q.n(8)}. ` + (order[0] === 2 ? 'A wider bottleneck keeps more detail. ' : 'Here the wider code did not help within this short run. ') + `The hardest of these eight is digit ${q.n(worst + 1)} (pixels off by ${q(errs[worst], 3)}).`, { phase: 'end', step: R.steps, n: 128, testLoss: tl, ks: [2, 4, 8], worst: worst + 1, errs, k }, { slow: true });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'An autoencoder', text: ['code: c = enc(x), ' + k + ' numbers', 'rebuilt: x̂ = σ(dec(c))', 'loss = − Σ over pixels [x log x̂ + (1 − x) log(1 − x̂)]', 'enc: 256 → 64 → ' + k + ',  dec: ' + k + ' → 64 → 256'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which code size will rebuild the test digits best after the run?', options: ['2 numbers', '4 numbers', '8 numbers'], answer: (fr) => { const t = fr[fr.length - 1].meta.testLoss; return ['2 numbers', '4 numbers', '8 numbers'][t.indexOf(Math.min(...t))]; }, explain: (fr) => fr[fr.length - 1].say },
      { at: (f, i, fr) => i === fr.length - 2, pick: 'item', ask: 'Click the rebuilt digit that will end furthest from its original.', answer: (fr) => 'r:' + (fr[fr.length - 1].meta.worst - 1), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What is an autoencoder trained to do?', options: ['rebuild its own input from a short code', 'predict a label for each input', 'tell real inputs from fakes', 'remove noise added on purpose'], answer: 0, why: 'The target is the input itself; the bottleneck forces the code to keep what matters most.' },
      { q: 'A 16×16 image goes through a code of 4 numbers. By what factor is it squeezed?', options: ['64', '16', '4', '256'], answer: 0, why: '256 pixels in, 4 numbers in the code: 256 / 4 = 64.' },
      { q: 'Why does a code of 2 numbers rebuild digits worse than a code of 8?', options: ['fewer numbers can tell fewer shapes apart', 'it trains for fewer steps', 'its decoder is linear', 'its loss is measured differently'], answer: 0, why: 'Every digit must be written as a point in a small space; with 2 numbers many shapes have to share nearby codes.' },
      { q: 'Why is a plain autoencoder not a good way to make new digits?', options: ['codes between the training codes may decode to nonsense, since nothing shapes the code space', 'its decoder cannot take a code as input', 'it can only copy the training digits', 'it needs labels'], answer: 0, why: 'Nothing asks the codes to fill a known region; a VAE adds that.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
