// Part 9, overfitting: a network with far more weights than it has training points learns the 30 points by heart.
// Its boundary twists round every noisy point, its training loss falls towards zero, and its loss on fresh points
// from the same moons turns and climbs. A smaller network, or more data, shows the difference.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml;
  P.SIZES = { small: { label: 'small: 4 hidden neurons', hidden: [4] }, medium: { label: 'medium: 8 and 8', hidden: [8, 8] }, big: { label: 'big: 32 and 32', hidden: [32, 32] } };
  P.weightsOf = (hidden) => { const s = [2].concat(hidden, [2]); let n = 0; for (let i = 0; i + 1 < s.length; i++) n += s[i] * s[i + 1] + s[i + 1]; return n; };

  S.topic({
    id: 'p09-overfit', family: 'p09-reg', order: 10, title: 'Overfitting: memorising the training set',
    blurb: 'A network with a thousand weights and thirty training points can fit every point exactly, noise and all. Watch it do so: the boundary bends round each stray point, the training loss heads for zero, and the loss on fresh points from the same moons, which is what we care about, stops falling and climbs. Shrink the network or give it more data and the gap closes.',
    inputs: [
      { id: 'size', kind: 'choice', label: 'Network', def: 'big', options: Object.keys(P.SIZES).map((k) => ({ value: k, label: P.SIZES[k].label })) },
      { id: 'n', kind: 'choice', label: 'Training points', def: '30', options: ['30', '100', '300'].map((v) => ({ value: v, label: v + ' points' })) },
    ],
    presets: [
      { label: 'Big network, 30 points', values: { size: 'big', n: '30' } },
      { label: 'Small network, 30 points', values: { size: 'small', n: '30' } },
      { label: 'Big network, 300 points', values: { size: 'big', n: '300' } },
      { label: 'Medium network, 100 points', values: { size: 'medium', n: '100' } },
    ],
    counters: [{ id: 'step', label: 'training steps' }],
    run(inp) {
      const F = H().frames(['step']), q = F.q, n = +inp.n, hid = P.SIZES[inp.size].hidden, r = P.regRun({ hidden: hid, n }), d = r.data, nw = P.weightsOf(hid);
      const best = r.evals.reduce((a, b) => (b.valLoss < a.valLoss ? b : a)), last = r.evals[r.evals.length - 1];
      const box = { x: 40, y: 50, w: 340, h: 230 }, cbox = { x: 480, y: 50, w: 380, h: 230 };
      const read = (s) => H().lines('ov:r', 40, 330, [['step ' + s.step, 'strong'], ['training: loss ' + H().fx(s.trainLoss, 3) + ', ' + H().pct(s.trainAcc) + ' right', 'math'], ['fresh points: loss ' + H().fx(s.valLoss, 3) + ', ' + H().pct(s.valAcc) + ' right', 'math']], 14, 24);
      r.snaps.forEach((s, k) => {
        F.set('step', s.step);
        const items = [...P.regPanel('ov:p', box, s.field, d), ...P.regCurves('ov:c', cbox, r.evals, s.step).items, ...read(s)];
        let say;
        if (k === 0) say = `A network with ${P.qpl(q, nw, 'weight')} (hidden layers of ${H().list(hid.map((h) => q.n(h)))}) and ${q.n(n)} training points (the large dots), with ${q.n(d.val.X.length)} fresh points from the same moons (the small ones) held back to judge it. Before training it says about the same everywhere: loss ${q(s.trainLoss, 3)}.`;
        else if (s.step <= best.step || k === 1) say = `Step ${q.n(s.step)}: the boundary takes the shape of the moons. Training loss ${q(s.trainLoss, 3)}, loss on fresh points ${q(s.valLoss, 3)}; ${q.pct(s.valAcc)} of the fresh points are right.`;
        else say = `Step ${q.n(s.step)}: training loss ${q.sg(s.trainLoss, 2)} with ${q.pct(s.trainAcc)} of the training points right, but the loss on fresh points is ${q(s.valLoss, 3)}` + (s.valLoss > best.valLoss * 1.15 ? `, up from its lowest, ${q(best.valLoss, 3)} at step ${q.n(best.step)}. The boundary is bending round single noisy points: it is learning the noise.` : ', still close to its lowest.');
        F.push(items, say, { phase: 'snap', step: s.step, trainLoss: s.trainLoss, valLoss: s.valLoss, trainAcc: s.trainAcc, valAcc: s.valAcc, nw, n, nVal: d.val.X.length, hid, best: { step: best.step, valLoss: best.valLoss } }, { hl: { eq: [k === 0 ? 0 : 1] } });
      });
      const gap = last.valLoss - last.trainLoss, over = last.valLoss > best.valLoss * 1.15 && gap > 0.1;
      F.push([...P.regPanel('ov:p', box, r.snaps[r.snaps.length - 1].field, d), ...P.regCurves('ov:c', cbox, r.evals, last.step, { mark: best.step }).items, ...read(last)],
        over ? `The gap: after ${q.n(last.step)} steps the loss on fresh points is ${q(last.valLoss, 3)} against ${q.sg(last.trainLoss, 2)} on the training points. The best moment for fresh points was step ${q.n(best.step)} (the line), with loss ${q(best.valLoss, 3)}. With ${q.n(nw)} weights for ${q.n(n)} points the network can afford to memorise; the next topics are the usual cures.`
          : `Here the two curves stay together: after ${q.n(last.step)} steps the loss on fresh points is ${q(last.valLoss, 3)} against ${q(last.trainLoss, 3)} in training, with its lowest, ${q(best.valLoss, 3)}, at step ${q.n(best.step)} (the line). ` + (n > 30 ? 'With more data, the noise of single points averages out and memorising no longer pays.' : 'A small network cannot bend round single points, so it has to learn the overall shape.'),
        { phase: 'end', gap, over, last: { step: last.step, valLoss: last.valLoss, trainLoss: last.trainLoss }, best: { step: best.step, valLoss: best.valLoss }, nw, n }, { hl: { eq: [2] } });
      P.narrow(F.list, (m) => (m.phase === 'end' ? { x: 440, y: 20, w: 450, h: 340 } : { x: 10, y: 20, w: 400, h: 400 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Fitting and generalising', text: ['training loss: on the points used to fit', 'loss on fresh points: what we care about', 'overfitting: the second rises while the first falls'].join('\n') }] }, result: { evals: r.evals, state: r.net.stateDict() } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'snap' && f.meta.step === 50, once: true, pick: 'option',
      ask: 'By the last step, will the loss on fresh points be lower or higher than it is now?',
      options: ['lower', 'higher'],
      answer: (fr) => { const a = fr.find((f) => f.meta.phase === 'snap' && f.meta.step === 50).meta.valLoss, b = fr.find((f) => f.meta.phase === 'end').meta.last.valLoss; return b < a ? 'lower' : 'higher'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'end').say,
    }],
    quiz: [
      { q: 'The clearest sign of overfitting is…', options: ['training loss falling while the loss on held-out data rises', 'both losses falling', 'both losses rising', 'a high learning rate'], answer: 0, why: 'The model is getting better at the training points only.' },
      { q: 'Which change usually reduces overfitting?', options: ['more training data', 'more epochs', 'more hidden neurons', 'removing the validation set'], answer: 0, why: 'With more points the noise of each one matters less.' },
      { q: 'A network reaches 100% training accuracy on noisy data. This shows…', options: ['it has enough capacity to memorise, not that it will do well on new data', 'it will be 100% right on new data', 'the data had no noise', 'the learning rate was perfect'], answer: 0, why: 'Fitting noise perfectly is memorising.' },
      { q: 'Why keep a validation set apart from the training set?', options: ['to measure how the model does on data it never fitted', 'to train faster', 'to make the loss smaller', 'it is required by backpropagation'], answer: 0, why: 'Only unseen data shows generalisation.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
