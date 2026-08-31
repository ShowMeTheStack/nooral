// Part 10, data augmentation: each training image is shifted, turned and rescaled a little at random every time it
// is used, so a small training set looks larger. The copies are made live by the engine (S.DL.augmentImage); the two
// training runs (40 digits, with and without augmentation) were made by tests/ref/p10_runs.mjs.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const TOP = 40, BIG = 104, SM = 70, CX0 = 140, AY = 252, AH = 170;
  const centre = (px) => { let s = 0, sx = 0, sy = 0; for (let i = 0; i < 256; i++) { s += px[i]; sx += px[i] * (i % 16); sy += px[i] * Math.floor(i / 16); } return [sx / s, sy / s]; };

  S.topic({
    id: 'cnn-augment', family: 'p10-train', order: 20, title: 'Data augmentation: more images from the same ones',
    blurb: 'A network trained on 40 images learns them by heart and stumbles on new ones. Augmentation shows it a slightly shifted, turned and resized copy of each image every time, so it must learn the digit, not the exact pixels. Make some copies yourself, then compare two training runs on the same 40 digits, one plain and one augmented.',
    inputs: [
      { id: 'cls', kind: 'number', label: 'Digit to copy', def: 2, min: 0, max: 9 },
      { id: 'shift', kind: 'slider', label: 'Shift (pixels)', def: 2, min: 0, max: 4, step: 0.5 },
      { id: 'rotate', kind: 'slider', label: 'Turn (degrees)', def: 17, min: 0, max: 45, step: 1 },
      { id: 'scale', kind: 'slider', label: 'Resize (± percent)', def: 10, min: 0, max: 30, step: 1 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 11, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'The settings of the run', values: { cls: 2, shift: 2, rotate: 17, scale: 10, seed: 11 } },
      { label: 'Shifts only', values: { cls: 4, shift: 3, rotate: 0, scale: 0, seed: 11 } },
      { label: 'Turns only', values: { cls: 7, shift: 0, rotate: 30, scale: 0, seed: 5 } },
      { label: 'Too much: the 6 may turn into a 9', values: { cls: 6, shift: 4, rotate: 45, scale: 30, seed: 3 } },
      { label: 'No augmentation at all', values: { cls: 3, shift: 0, rotate: 0, scale: 0, seed: 11 } },
    ],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const T = S.T, im = P.pick('digits', inp.cls, 0), g = T.rng(inp.seed), opts = { shift: inp.shift, rotate: (inp.rotate * Math.PI) / 180, scale: inp.scale / 100 };
      const copies = Array.from({ length: 8 }, () => S.DL.augmentImage(im.px, 16, 16, g, opts));
      const c0 = centre(im.px), moves = copies.map((c) => { const m = centre(c); return Math.hypot(m[0] - c0[0], m[1] - c0[1]); });
      const diff = copies.map((c) => c.reduce((s, v, i) => s + Math.abs(v - im.px[i]), 0) / 256);
      const Joff = P.D().augOff, Jon = P.D().augOn, F = H().frames(['step']), q = F.q;
      const ax = C().axes({ prefix: 'aa', x: 40, y: AY, w: 300, h: AH, xr: [0, Joff.steps], yr: [0, 1], xticks: 150, yticks: [0, 0.25, 0.5, 0.75, 1], yname: 'accuracy', xname: 'training step', fmtY: (v) => Math.round(v * 100) + '%' });
      const lmax = Math.ceil(Math.max(...Joff.val.map((v) => v.valLoss), ...Jon.val.map((v) => v.valLoss)));
      const bx = C().axes({ prefix: 'al', x: 440, y: AY, w: 280, h: AH, xr: [0, Joff.steps], yr: [0, lmax], xticks: 150, yticks: 1, yname: 'validation loss', xname: 'training step' });
      const pics = (lit) => {
        const items = [P.text('h:o', 0, TOP - 14, 'the image', 'axname', 13), P.img('orig', 0, TOP, BIG, im.px, { cls: lit ? '' : 'act' }), P.text('h:c', CX0, TOP - 14, 'eight augmented copies (seed ' + inp.seed + ')', 'axname', 13)];
        copies.forEach((c, i) => items.push(P.img('cp' + i, CX0 + (i % 4) * (SM + 12), TOP + Math.floor(i / 4) * (SM + 14), SM, c, { cls: lit ? 'act' : '' })));
        return items;
      };
      const curves = (step) => {
        const items = [...ax.items, ...bx.items];
        if (step !== null) {
          const up = (J) => J.val.filter((v) => v.step <= step);
          items.push(C().line(ax, 'off:t', up(Joff).map((v) => [v.step, v.trainAcc]), 'p10off dash'), C().line(ax, 'off:v', up(Joff).map((v) => [v.step, v.valAcc]), 'p10off'));
          items.push(C().line(ax, 'on:t', up(Jon).map((v) => [v.step, v.trainAcc]), 'p10on dash'), C().line(ax, 'on:v', up(Jon).map((v) => [v.step, v.valAcc]), 'p10on'));
          items.push(C().line(bx, 'off:l', up(Joff).map((v) => [v.step, v.valLoss]), 'p10off'), C().line(bx, 'on:l', up(Jon).map((v) => [v.step, v.valLoss]), 'p10on'));
        }
        items.push(...C().legend('lg', 40, AY + AH + 54, [{ text: 'plain', cls: 'p10off' }, { text: 'augmented', cls: 'p10on' }, { text: 'dashed: training images; solid: validation', cls: 'p10off dash' }]));
        return items;
      };
      F.push([...pics(false), ...curves(null)], `${P.qwords(q, 'digits', im.y)}, as it is. The training set of the two runs below has only ${q.n(Joff.nTrain)} images of digits, this one not among them.`, { phase: 'orig', y: im.y, nTrain: Joff.nTrain });
      const none = !inp.shift && !inp.rotate && !inp.scale, far = moves.indexOf(Math.max(...moves));
      F.push([...pics(true), ...curves(null)], none
        ? `With every setting at zero the copies are the image itself: the average pixel changes by ${q.sg(Math.max(...diff), 3)}. Nothing new is learned from them.`
        : `Eight copies, each with its own random shift of up to ${q.sg(inp.shift, 2)} pixels, turn of up to ${q.n(inp.rotate)}° and size change of up to ${q.n(inp.scale)}%. Copy ${q.n(far + 1)} moved most: its ink’s centre is ${q.sg(moves[far], 3)} pixels from the original’s. On average a pixel changes by ${q.sg(diff.reduce((a, b) => a + b, 0) / 8, 3)}. Each is still clearly the same digit` + (inp.rotate > 35 || inp.shift > 3 ? ', though copies pushed this far start to leave the picture or look like other digits.' : '.'), { phase: 'copies', far: far + 1, move: moves[far], meanDiff: diff.reduce((a, b) => a + b, 0) / 8, maxDiff: Math.max(...diff), shift: inp.shift, rotate: inp.rotate, scale: inp.scale });
      for (const step of [100, 200, 400, 600]) {
        F.set('step', step);
        const a = Joff.val.find((v) => v.step === step), b = Jon.val.find((v) => v.step === step);
        F.push([...pics(false), ...curves(step)], `Step ${q.n(step)}. Plain: ${q.pct(a.trainAcc)} of the training images right, ${q.pct(a.valAcc)} of the validation images. Augmented: ${q.pct(b.trainAcc)} and ${q.pct(b.valAcc)}.` + (step === 100 ? ` The plain run already knows its ${q.n(Joff.nTrain)} images; the augmented one is still learning, because its images keep changing.` : ''), { phase: 'curve', step, nTrain: Joff.nTrain, offTrain: a.trainAcc, offVal: a.valAcc, onTrain: b.trainAcc, onVal: b.valAcc });
      }
      const a = Joff.val[Joff.val.length - 1], b = Jon.val[Jon.val.length - 1];
      F.push([...pics(false), ...curves(600)], `At the end both runs get every training image right, but on new images the plain run is right ${q.pct(a.valAcc)} of the time and the augmented run ${q.pct(b.valAcc)}. The plain run’s validation loss climbs back to ${q.sg(a.valLoss, 3)} while it grows surer of its ${q.n(Joff.nTrain)} images; the augmented run’s falls to ${q.sg(b.valLoss, 3)}. Augmentation is a cheap way to say what should not matter: a digit is the same digit a pixel to the left.`, { phase: 'end', nTrain: Joff.nTrain, offVal: a.valAcc, onVal: b.valAcc, offLoss: a.valLoss, onLoss: b.valLoss });
      return { frames: F.list, def: { sections: [{ id: 'aug', kind: 'math', title: 'The two runs', text: ['40 training digits, 300 validation digits', 'same network, seed, Adam, 600 steps of 32', 'augmented: shift ≤ 2 px, turn ≤ 0.3 rad, size ± 10%', 'new random copies every time an image is used', 'validation images are never augmented'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'copies', once: true, pick: 'option',
      ask: 'Trained on the same 40 digits, which run will be right more often on new digits?',
      options: ['the augmented run', 'the plain run', 'about the same'],
      answer: (fr) => { const e = fr[fr.length - 1].meta; return Math.abs(e.onVal - e.offVal) < 0.02 ? 'about the same' : e.onVal > e.offVal ? 'the augmented run' : 'the plain run'; },
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'Augmentation is applied to…', options: ['training images only', 'validation images only', 'both', 'the network’s weights'], answer: 0, why: 'Validation must measure the real task.' },
      { q: 'Why might turning a digit by 180° be bad augmentation?', options: ['a 6 becomes a 9: the label is no longer true', 'it is too slow', 'it removes ink', 'networks cannot see turned images'], answer: 0, why: 'A change must keep the label right.' },
      { q: 'A model with 100% training accuracy and 78% validation accuracy benefits most from…', options: ['more varied training data, such as augmentation', 'more training steps', 'a larger learning rate', 'removing the validation set'], answer: 0, why: 'It is memorising; variety fights that.' },
      { q: 'Why does the augmented run’s training accuracy rise more slowly?', options: ['its training images change every time', 'it has fewer images', 'it uses a smaller network', 'its learning rate is lower'], answer: 0, why: 'It cannot memorise images that keep moving.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
