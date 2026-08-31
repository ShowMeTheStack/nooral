// Part 10, training a small CNN: the run stored by tests/ref/p10_runs.mjs replayed. Loss and accuracy curves for the
// training batches and the validation images, and the confusion matrix of the 300 validation images at four moments;
// every cell can be clicked to see its images.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const AX = { x: 40, w: 300 }, LY = 40, AY = 262, PH = 150, CX = 430, CY = 58;

  S.topic({
    id: 'cnn-train', family: 'p10-train', order: 10, title: 'Training a CNN on generated digits and shapes',
    blurb: 'The network of this part (two convolution layers, two pooling layers, 32 hidden units) trained by Adam on mini-batches of 32 generated images. Watch the loss fall and the accuracy rise, for the batches it learns from and for 300 validation images it never trains on, and see in the confusion matrix which classes it still mixes up. Click a cell to see its images.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Images', def: 'digits', options: Object.keys(P.SETS).map((v) => ({ value: v, label: P.SETS[v] + ' (' + (v === 'digits' ? '2000' : '1200') + ' to train on)' })) },
      { id: 'norm', kind: 'choice', label: 'Confusion cells', def: 'count', options: [{ value: 'count', label: 'counts' }, { value: 'row', label: 'share of each true class' }] },
    ],
    presets: [
      { label: 'Digits', values: { set: 'digits', norm: 'count' } },
      { label: 'Shapes', values: { set: 'shapes', norm: 'count' } },
      { label: 'Shapes, rows as shares', values: { set: 'shapes', norm: 'row' } },
    ],
    counters: [{ id: 'step', label: 'training steps' }, { id: 'seen', label: 'images seen' }],
    run(inp) {
      const run = inp.set, J = P.D()[run], K = J.classes.length, names = P.classes(run), imgs = P.images(run), y = imgs.map((e) => e.y);
      const F = H().frames(['step', 'seen']), q = F.q, nV = imgs.length;
      // the training loss, averaged over 25 steps at a time (one point per evaluation)
      const avg = (s) => { const h = J.hist.slice(Math.max(0, s - 25), s); return h.reduce((a, b) => a + b[0], 0) / h.length; };
      const lmax = Math.ceil(Math.max(...J.val.map((v) => v.valLoss), ...J.val.map((v) => v.trainLoss)) * 2) / 2;
      const ax1 = C().axes({ prefix: 'al', x: AX.x, y: LY, w: AX.w, h: PH, xr: [0, J.steps], yr: [0, lmax], xticks: J.steps / 4, yticks: lmax > 2 ? 1 : 0.5, yname: 'loss', xname: '' });
      const ax2 = C().axes({ prefix: 'aa', x: AX.x, y: AY, w: AX.w, h: PH, xr: [0, J.steps], yr: [0, 1], xticks: J.steps / 4, yticks: [0, 0.25, 0.5, 0.75, 1], yname: 'accuracy', xname: 'training step', fmtY: (v) => Math.round(v * 100) + '%' });
      const cw = K > 4 ? 28 : 60, gap = 2;
      const confusion = (preds) => { const M = Array.from({ length: K }, () => new Array(K).fill(0)); preds.forEach((p, i) => M[y[i]][p]++); return M; };
      const draw = (sn) => {
        const upto = J.val.filter((v) => v.step <= sn.step), items = [...ax1.items, ...ax2.items];
        items.push(C().line(ax1, 'l:tr', upto.filter((v) => v.step > 0).map((v) => [v.step, avg(v.step)]), 'p10train'), C().line(ax1, 'l:va', upto.map((v) => [v.step, v.valLoss]), 'p10val'));
        items.push(C().line(ax2, 'a:tr', upto.map((v) => [v.step, v.trainAcc]), 'p10train'), C().line(ax2, 'a:va', upto.map((v) => [v.step, v.valAcc]), 'p10val'));
        const last = upto[upto.length - 1];
        items.push(C().dot(ax1, 'd:l', last.step, last.valLoss, 'p10val'), C().dot(ax2, 'd:a', last.step, last.valAcc, 'p10val'));
        items.push(...C().legend('lg', AX.x, AY + PH + 54, [{ text: 'training images', cls: 'p10train' }, { text: 'validation images', cls: 'p10val' }]));
        const preds = [...sn.preds].map(Number), M = confusion(preds), V = inp.norm === 'row' ? M.map((r) => { const t = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / t); }) : M;
        const g = S.Kit.heat.grid({ prefix: 'cm', x: CX, y: CY, cw, ch: cw, gap, vals: V, scheme: 'seq', max: inp.norm === 'row' ? 1 : Math.max(...M.flat()), fmt: (v) => (inp.norm === 'row' ? (v ? String(Math.round(v * 100)) : '') : v ? String(v) : ''), rows: names, cols: names, size: 12, labelSize: 12, cls: (i, j) => (M[i][j] ? 'pickable clickable' : '') + (i === j ? ' diag' : '') });
        items.push(...g.items, P.text('cm:t', CX - 8, CY - 26, 'true ↓  predicted →', 'axname', 12, 'end'));
        items.push(P.text('cm:h', CX, CY + K * (cw + gap) + 22, inp.norm === 'row' ? 'percent of each true class' : 'validation images, ' + nV + ' in all', 'legend', 12));
        const picks = {};
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (M[i][j]) {
          const idx = preds.map((p, n) => n).filter((n) => y[n] === i && preds[n] === j).slice(0, 12);
          picks['cm:c' + i + '_' + j] = { title: 'True ' + names[i] + ', predicted ' + names[j] + ': ' + M[i][j] + ' image' + (M[i][j] > 1 ? 's' : ''), images: idx.map((n) => ({ px: Array.from(imgs[n].px, (v) => Math.round(v * 15)), w: 16, h: 16, label: '#' + n })) };
        }
        return { items, M, preds, picks };
      };
      let prev = 0;
      J.snaps.forEach((sn, s) => {
        F.set('step', sn.step).set('seen', sn.step * 32);
        const d = draw(sn), right = d.preds.filter((p, i) => p === y[i]).length, v = J.val.find((x) => x.step === sn.step), gain = right - prev;
        const say = sn.step === 0
          ? `Before training the weights are random. On the ${q.n(nV)} validation images the loss is ${q.sg(v.valLoss, 3)} and ${q.n(right)} are right, ${q.pct(right / nV)}: about what guessing among ${q.n(K)} classes gives. The predictions pile up in a few columns.`
          : `Step ${q.n(sn.step)}, ${q.n(sn.step * 32)} images seen: training loss ${q.sg(avg(sn.step), 3)} (the last ${q.n(25)} batches), validation loss ${q.sg(v.valLoss, 3)}. ${q.n(right)} of the ${q.n(nV)} validation images are right${right / nV >= 0.995 && right < nV ? '' : ` (${q.pct(right / nV)})`}, ${q.n(gain)} more than at the last snapshot; the diagonal fills.`;
        prev = right;
        F.add({ items: d.items, say, picks: d.picks, meta: { phase: 'snap', step: sn.step, seen: sn.step * 32, valLoss: v.valLoss, trainLoss: sn.step ? avg(sn.step) : null, right, n: nV, acc: right / nV, k: K, gain, twentyfive: 25, quotes: q.take() } });
      });
      const last = J.snaps[J.snaps.length - 1], d = draw(last);
      let worst = null;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (i !== j && d.M[i][j] && (!worst || d.M[i][j] > worst.n)) worst = { i, j, n: d.M[i][j] };
      const fv = J.val[J.val.length - 1], gap2 = fv.trainAcc - fv.valAcc;
      // a score that is not perfect never shows as 100%
      const pc = (v) => (v < 1 && Math.round(v * 100) >= 100 ? q(v * 100, 1) + '%' : q.pct(v));
      F.add({ items: d.items, picks: d.picks, say: (worst ? `The mistakes that are left: the largest cell off the diagonal says ${worst.n > 1 ? q.n(worst.n) + ' images' : 'one image'} of ${run === 'digits' ? 'the digit ' + q.n(worst.i) : P.an(names[worst.i])} ${worst.n > 1 ? 'were' : 'was'} taken for ${run === 'digits' ? P.an(q.n(worst.j)) : P.an(names[worst.j])}. Click it to see ${worst.n > 1 ? 'them' : 'it'}. ` : 'No validation image is wrong. ') + `Training accuracy ${pc(fv.trainAcc)} against validation ${pc(fv.valAcc)}: ` + (gap2 > 0.03 ? 'a gap that says the network is starting to memorise.' : 'close together, so it has learned the classes rather than the particular images.'), meta: { phase: 'end', worst, trainAcc: fv.trainAcc, valAcc: fv.valAcc, percent: [fv.trainAcc * 100, fv.valAcc * 100], gap: gap2, quotes: q.take() } });
      return { frames: F.list, def: { sections: [{ id: 'setup', kind: 'math', title: 'The training run', text: ['images 16 × 16, ' + J.nTrain + ' to train, ' + J.nVal + ' to validate', 'loss: cross-entropy; optimiser: Adam, learning rate 0.002', 'batches of 32, ' + J.steps + ' steps, seed fixed', 'accuracy = right / all; the confusion matrix: true × predicted'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'snap' && f.meta.step === 0, once: true, pick: 'option',
      ask: (fr) => 'How many of the validation images will be right at the end of training?',
      options: ['under half', 'half to 90%', 'over 90%'],
      answer: (fr) => { const a = fr.filter((f) => f.meta.phase === 'snap').slice(-1)[0].meta.acc; return a < 0.5 ? 'under half' : a <= 0.9 ? 'half to 90%' : 'over 90%'; },
      explain: (fr) => fr.filter((f) => f.meta.phase === 'snap').slice(-1)[0].say,
    }],
    quiz: [
      { q: 'Before any training, a 10-class network is right about…', options: ['one time in ten', 'half the time', 'never', 'always'], answer: 0, why: 'Random weights guess.' },
      { q: 'Why keep validation images the network never trains on?', options: ['to measure how it does on new images', 'to train faster', 'to make the loss smaller', 'to choose the batch size'], answer: 0, why: 'Training accuracy flatters a model that memorises.' },
      { q: 'A large number in row 5, column 8 of the confusion matrix means…', options: ['many true 5s were predicted as 8', 'many 8s were predicted as 5', 'the network is right on 5s', 'there are many 5s'], answer: 0, why: 'Rows are true classes, columns predictions.' },
      { q: 'Training accuracy 100%, validation 78%. The network is…', options: ['overfitting: memorising its training set', 'underfitting', 'perfect', 'broken'], answer: 0, why: 'The gap between the two is the sign.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
