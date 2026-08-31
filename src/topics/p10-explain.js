// Part 10, what the network looks at: a saliency map (the gradient of one class score with respect to every pixel,
// S.DL.saliency) tested by blanking the pixels it points at; and Grad-CAM (S.DL.gradCam): the last convolution's maps
// weighted by the average gradient of the class score on each, added and cut at zero.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const TARGETS = [{ value: 'pred', label: 'the class it chooses' }, { value: 'second', label: 'its second choice' }, { value: 'true', label: 'the true class' }];
  const target = (o, y, which) => { const order = o.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]); return which === 'second' ? order[1][1] : which === 'true' ? y : order[0][1]; };
  const ix = (p) => [Math.floor(p / 16) + 1, (p % 16) + 1];

  S.topic({
    id: 'cnn-saliency', family: 'p10-explain', order: 10, title: 'Saliency maps: which pixels move the score',
    blurb: 'Take the score the network gives one class and ask how it would change if one pixel were a little darker: that is the gradient of the score with respect to the pixel, and backpropagation gives it for every pixel at once. Drawn as a picture it shows what the decision leans on. Then test it: blank the pixels it marks, and the score should fall more than when blanking as many random ink pixels.',
    inputs: P.imageInputs('digits', 3, 1).concat([{ id: 'target', kind: 'choice', label: 'Explain the score of', def: 'pred', options: TARGETS }, { id: 'blank', kind: 'number', label: 'Pixels to blank', def: 12, min: 4, max: 30 }]),
    presets: [
      { label: 'A 3', values: { set: 'digits', cls: 3, k: 1, target: 'pred', blank: 12 } },
      { label: 'A 7', values: { set: 'digits', cls: 7, k: 0, target: 'pred', blank: 12 } },
      { label: 'The misread 5, its true class', values: { set: 'digits', cls: 5, k: 8, target: 'true', blank: 12 } },
      { label: 'A circle', values: { set: 'shapes', cls: 0, k: 0, target: 'pred', blank: 16 } },
      { label: 'A star', values: { set: 'shapes', cls: 3, k: 1, target: 'pred', blank: 12 } },
      { label: 'A square it doubts: the test fails', values: { set: 'shapes', cls: 1, k: 2, target: 'pred', blank: 16 } },
    ],
    counters: [{ id: 'blanked', label: 'pixels blanked' }],
    run(inp) {
      const run = inp.set, im = P.pick(run, inp.cls, inp.k), o = P.forward(run, im.px), c = target(o, im.y, inp.target);
      const sal = S.DL.saliency(P.net(run), im.px, c), raw = sal.raw, map = sal.map, F = H().frames(['blanked']), q = F.q;
      const S1 = 156, Y1 = 40, Y2 = 268, rmax = P.maxAbs(raw);
      const peak = map.indexOf(1), cn = (k) => P.qname(q, run, k);
      // the most salient ink pixels against random ink pixels (a fixed seed)
      const ink = Array.from(im.px.keys()).filter((i) => im.px[i] > 0.2), n = Math.min(inp.blank, ink.length);
      const top = ink.slice().sort((a, b) => raw[b] - raw[a]).slice(0, n), g = S.T.rng(17), shuf = ink.slice();
      g.shuffle(shuf);
      const rnd = shuf.slice(0, n);
      const blankOut = (list) => { const px = Float32Array.from(im.px); for (const i of list) px[i] = 0; return px; };
      const A = blankOut(top), B = blankOut(rnd), sA = P.forward(run, A).logits.v[c], sB = P.forward(run, B).logits.v[c], s0 = o.logits.v[c];
      const head = (k, x, y, t) => P.text('h:' + k, x, y, t, 'axname', 13);
      const ring = (list, key) => list.map((p) => P.box(key + p, (p % 16) * (S1 / 16) + 2 * S1 + 2 * 34, Y1 + Math.floor(p / 16) * (S1 / 16), S1 / 16, S1 / 16, 'hot'));
      const base = (show) => {
        const items = [head('in', 0, Y1 - 14, 'the image'), P.img('in', 0, Y1, S1, im.px), head('pr', 3 * S1 + 3 * 34 - 10, Y1 - 14, 'chances'), ...P.chances('pr', 3 * S1 + 3 * 34 - 10, Y1, run, o.probs, { rowH: run === 'digits' ? 17 : 34, w: 172, nameW: run === 'digits' ? 18 : 60 })];
        items.push(head('raw', S1 + 34, Y1 - 14, 'gradient of the score'), show >= 1 ? P.img('raw', S1 + 34, Y1, S1, raw, { scheme: 'div', max: rmax }) : P.box('raw:e', S1 + 34, Y1, S1, S1, 'empty', 'items'));
        items.push(head('sal', 2 * S1 + 2 * 34, Y1 - 14, 'saliency |gradient|'));
        if (show >= 2) items.push(P.img('sal:i', 2 * S1 + 68, Y1, S1, im.px), Object.assign(P.img('sal', 2 * S1 + 68, Y1, S1, map, { scheme: 'seq', max: 1, layer: 'items' }), { opacity: 0.72, z: 2 }));
        else items.push(P.box('sal:e', 2 * S1 + 68, Y1, S1, S1, 'empty', 'items'));
        if (show >= 3) {
          items.push(head('a', 0, Y2 - 14, 'the marked pixels blanked'), P.img('A', 0, Y2, 120, A, { cls: 'hot' }), head('b', 190, Y2 - 14, 'random ink pixels blanked'), P.img('B', 190, Y2, 120, B));
          items.push(...H().lines('sc', 380, Y2 + 20, [['score of ' + P.cname(run, c) + ': ' + P.num(s0, 2), 'strong'], ['marked pixels blanked: ' + P.num(sA, 2), 'strong'], ['random pixels blanked: ' + P.num(sB, 2), 'legend']], 13, 26));
        }
        if (show >= 2) items.push(...ring(top, 'rg'));
        return items;
      };
      F.push(base(0), `The network gives ${P.qwords(q, run, im.y)} these chances. We explain the score of class ${cn(c)} (its chance is ${q.pct(o.probs[c])}): the number before softmax, here ${q.sg(s0, 3)}.`, { phase: 'start', y: im.y, c, p: o.probs[c], s0 });
      const [pr, pc] = ix(sal.raw.map(Math.abs).indexOf(Math.max(...raw.map(Math.abs)))), pos = raw.filter((v) => v > 0).length;
      F.push(base(1), `Backpropagation from that score down to the input gives one number per pixel: blue where darkening the pixel would raise the score, vermilion where it would lower it. The largest, ${q.sg(rmax, 3)} per unit of ink, is at row ${q.n(pr)}, column ${q.n(pc)}; ${q.n(pos)} of the ${q.n(256)} pixels push the score up.`, { phase: 'raw', max: rmax, row: pr, col: pc, pos, all: 256 });
      F.push(base(2), `Taking the size and ignoring the sign gives the saliency map, laid over the image. The ${q.n(n)} ink pixels with the largest positive gradient are outlined: the ink the score of class ${cn(c)} leans on most.`, { phase: 'sal', n, c, peak });
      F.count('blanked', 2 * n);
      F.push(base(3), `The test: blanking the ${q.n(n)} marked pixels moves the score of class ${cn(c)} from ${q.sg(s0, 3)} to ${q.sg(sA, 3)}; blanking ${q.n(n)} random ink pixels moves it to ${q.sg(sB, 3)}.` + (s0 - sA <= 0 && s0 - sB <= 0 ? ' Neither lowers the score: taking ink away from this image makes it look more like the class, not less. A gradient only describes very small changes, and blanking is not small.' : s0 - sA > s0 - sB ? ' The marked pixels matter more, as the map said.' : ' Here the random pixels lower it at least as much: a gradient only describes very small changes, and blanking is not small.'), { phase: 'test', n, c, s0, sA, sB });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Saliency', text: ['score s_c: the class’s output before softmax', 'saliency(pixel) = |∂ s_c / ∂ pixel|', 'one backward pass gives all 256 at once', 'a local measure: it says what tiny changes do'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'sal', once: true, pick: 'option',
      ask: 'Will blanking the marked pixels lower the score more than blanking as many random ink pixels?',
      options: ['yes, the marked pixels hurt more', 'no, the random ones hurt as much or more'],
      answer: (fr) => { const m = fr[fr.length - 1].meta; return m.s0 - m.sA > m.s0 - m.sB && m.s0 - m.sA > 0 ? 'yes, the marked pixels hurt more' : 'no, the random ones hurt as much or more'; },
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'A saliency map is…', options: ['the size of the gradient of a class score with respect to each pixel', 'the image after the first convolution', 'the network’s weights', 'the loss at each pixel'], answer: 0, why: '|∂ score / ∂ pixel|.' },
      { q: 'How many backward passes does a saliency map for one class need?', options: ['one', 'one per pixel', 'one per layer', 'none'], answer: 0, why: 'Backpropagation gives every pixel’s gradient at once.' },
      { q: 'Why test a saliency map by blanking pixels?', options: ['to check that the pixels it marks really matter', 'to train the network', 'to make the map smoother', 'to remove noise'], answer: 0, why: 'An explanation should predict what changes the output.' },
      { q: 'A gradient-based map describes…', options: ['what very small changes to a pixel do', 'what removing an object does, exactly', 'the training data', 'the class chances'], answer: 0, why: 'A derivative is a local, first-order measure.' },
    ],
  });

  S.topic({
    id: 'cnn-gradcam', family: 'p10-explain', order: 20, title: 'Grad-CAM: where in the picture the evidence lies',
    blurb: 'The last convolution layer still knows where things are: each of its maps is a coarse picture. Grad-CAM asks how much the class score depends on each map (the average of its gradient), weighs the maps by that, adds them and keeps the positive part. Stretched over the image, it shows the region that speaks for the class, and a different class gets a different region.',
    inputs: P.imageInputs('digits', 8, 2).concat([{ id: 'target', kind: 'choice', label: 'Explain the class', def: 'pred', options: TARGETS }]),
    presets: [
      { label: 'An 8', values: { set: 'digits', cls: 8, k: 2, target: 'pred' } },
      { label: 'A 7: the empty corner counts', values: { set: 'digits', cls: 7, k: 1, target: 'pred' } },
      { label: 'A 9, its second choice', values: { set: 'digits', cls: 9, k: 1, target: 'second' } },
      { label: 'The misread 5', values: { set: 'digits', cls: 5, k: 8, target: 'pred' } },
      { label: 'A circle', values: { set: 'shapes', cls: 0, k: 0, target: 'pred' } },
      { label: 'A square taken for a triangle, true class', values: { set: 'shapes', cls: 1, k: 0, target: 'true' } },
    ],
    counters: [{ id: 'maps', label: 'maps weighed' }],
    run(inp) {
      const run = inp.set, im = P.pick(run, inp.cls, inp.k), o = P.forward(run, im.px), c = target(o, im.y, inp.target);
      const order = o.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]), other = order[0][1] === c ? order[1][1] : order[0][1];
      const cam = S.DL.gradCam(P.net(run), im.px, c), cam2 = S.DL.gradCam(P.net(run), im.px, other);
      const A = o.relu2.v, w = cam.weights, F = H().frames(['maps']), q = F.q, cn = (k) => P.qname(q, run, k);
      const IS = 150, MX = 190, MS = 44, MG = 10, Y3 = 238, amax = P.maxAbs(A), wmax = P.maxAbs(w);
      const head = (k, x, y, t) => P.text('h:' + k, x, y, t, 'axname', 13);
      const overlay = (key, x, y, m, label) => [head(key, x, y - 14, label), P.img(key + ':i', x, y, IS, im.px), Object.assign({ key, kind: 'raster', x, y, w: IS, h: IS, nx: 4, ny: 4, vals: m.flat(), scheme: 'seq', max: 1, smooth: true, cls: 'p10img', layer: 'items', z: 2 }, { opacity: 0.66 })];
      const base = (show) => {
        const items = [head('in', 0, 26, 'the image'), P.img('in', 0, 40, IS, im.px), ...P.chances('pr', 0, 214, run, o.probs, { rowH: run === 'digits' ? 19 : 40, w: 150, nameW: run === 'digits' ? 18 : 60 })];
        items.push(head('m', MX, 26, 'last convolution, after ReLU: 12 maps of 4 × 4'));
        for (let ch = 0; ch < 12; ch++) {
          const x = MX + (ch % 6) * (MS + MG), y = 40 + Math.floor(ch / 6) * (MS + 36);
          items.push(P.img('a' + ch, x, y, MS, P.map(A, ch, 4), { scheme: 'seq', max: amax, cls: show >= 2 && w[ch] === Math.max(...w) ? 'act' : '' }));
          if (show >= 2) items.push(P.text('w' + ch, x + MS / 2, y + MS + 16, P.num(w[ch], 2), w[ch] >= 0 ? 'pos' : 'neg', 12, 'middle'));
        }
        if (show >= 3) items.push(head('sum', MX, Y3 - 14, 'weighted sum, cut at 0'), ...P.cells('sum', MX, Y3, cam.raw, { cw: 36, scheme: 'seq', max: Math.max(...cam.raw.flat()) || 1, d: 2, size: 11 }).items);
        if (show >= 4) items.push(...overlay('cam', 380, Y3, cam.map, 'Grad-CAM for ' + P.cname(run, c)));
        if (show >= 5) items.push(...overlay('cam2', 560, Y3, cam2.map, 'for ' + P.cname(run, other)));
        return items;
      };
      F.push(base(1), `${P.qwords(q, run, im.y).replace(/^./, (s) => s.toUpperCase())} goes through the network; we explain class ${cn(c)} (chance ${q.pct(o.probs[c])}). Grad-CAM starts from the last convolution layer after ReLU: ${q.n(12)} maps of ${q.n(4)} × ${q.n(4)}, each a coarse picture of where its pattern was found.`, { phase: 'start', y: im.y, c, p: o.probs[c], n12: 12, side: 4 });
      F.count('maps', 12);
      const best = w.indexOf(Math.max(...w)), negs = w.filter((v) => v < 0).length;
      F.push(base(2), `Backpropagating the score of ${cn(c)} to those maps and averaging the gradient over each map gives one weight per map (under each). Map ${q.n(best + 1)} counts most, ${q.sg(w[best], 3)}; ${q.n(negs)} of the ${q.n(12)} weights are negative: those maps speak against the class.`, { phase: 'weights', c, best: best + 1, wBest: w[best], negs, n12: 12 });
      const raw = cam.raw.flat(), pk = raw.indexOf(Math.max(...raw)), zero = raw.filter((v) => v === 0).length;
      F.push(base(3), `Each map is multiplied by its weight and the twelve are added, cell by cell; negative sums are cut to zero. The largest cell, ${q.sg(raw[pk], 3)}, is at row ${q.n(Math.floor(pk / 4) + 1)}, column ${q.n((pk % 4) + 1)}; ${q.n(zero)} of the ${q.n(16)} cells are zero.`, { phase: 'sum', c, max: raw[pk], row: Math.floor(pk / 4) + 1, col: (pk % 4) + 1, zero, n16: 16 });
      const where = (r, cc) => (r < 2 ? 'upper' : 'lower') + ' ' + (cc < 2 ? 'left' : 'right');
      let onInk = 0, tot = 0;
      for (let i = 0; i < 256; i++) { const v = cam.map[Math.floor(i / 64)][Math.floor((i % 16) / 4)]; onInk += v * im.px[i]; tot += v; }
      const share = tot ? onInk / tot : 0;
      F.push(base(4), `Scaled to 0…1 and stretched over the image, it marks the ${where(Math.floor(pk / 4), pk % 4)} part as the evidence for ${cn(c)}. Each cell covers a patch of about ${q.n(4)} × ${q.n(4)} pixels, so the map is coarse: it says where, not which pixel. Only ${q.pct(share)} of its weight falls on ink` + (share < 0.35 ? ': for this network, where the strokes are not is as telling as where they are.' : ', the rest on the paper around the strokes.'), { phase: 'overlay', c, side: 4, share });
      const r2 = cam2.raw.flat(), pk2 = r2.indexOf(Math.max(...r2));
      F.push(base(5), `The same for class ${cn(other)} (chance ${q.pct(o.probs[other])}): its evidence lies in the ${where(Math.floor(pk2 / 4), pk2 % 4)} part` + (pk2 === pk ? ', the same place: both classes are read from the same strokes.' : ', a different place from the evidence for ' + cn(c) + '.'), { phase: 'other', other, p: o.probs[other], pk, pk2, c });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Grad-CAM', text: ['Aᵏ: map k of the last convolution (after ReLU)', 'αₖ = mean over the map of ∂ s_c / ∂ Aᵏ', 'CAM = max(0, Σₖ αₖ Aᵏ), then scaled to 0…1', 'stretched to the image’s size and laid over it'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'sum', once: true, pick: 'item',
      ask: 'Click the cell of the weighted sum you expect to be the largest.',
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'sum').meta; return 'sum:c' + (m.row - 1) + '_' + (m.col - 1); },
      explain: (fr) => fr.find((f) => f.meta.phase === 'sum').say,
    }],
    quiz: [
      { q: 'Grad-CAM weighs each map of the last convolution by…', options: ['the average gradient of the class score over that map', 'the map’s largest value', 'the filter’s bias', 'the class chance'], answer: 0, why: 'αₖ = mean of ∂ s_c / ∂ Aᵏ.' },
      { q: 'Why is the Grad-CAM map of this network so coarse?', options: ['the last convolution’s maps are only 4 × 4', 'the image is blurred', 'the gradient is rounded', 'ReLU removes detail'], answer: 0, why: 'Its resolution is that of the layer it reads.' },
      { q: 'Why cut the weighted sum at zero?', options: ['to keep only evidence for the class', 'to make it smaller', 'because maps cannot be negative', 'to normalise it'], answer: 0, why: 'Negative regions argue against the class.' },
      { q: 'Grad-CAM maps for two different classes of the same image…', options: ['can point to different regions', 'are always the same', 'are always empty', 'need two networks'], answer: 0, why: 'Each class has its own weights αₖ.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
