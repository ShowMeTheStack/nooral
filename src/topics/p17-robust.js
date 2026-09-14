// Part 17, brittle models: a small network on two moons, the gradient of its loss with respect to one input point,
// the step of size ε along that gradient's sign that flips the answer, and the smallest such ε. Then the defences:
// training on nudged examples, and smoothing (a vote over noisy copies of the input).
(function (S) {
  'use strict';
  const P = S.P17, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  // a percentage with its half point kept: on 200 test points each one is worth 0.5%, so 199 of 200 is 99.5%, not
  // 100% (a frame's state keeps these values in percent too, under 'percent')
  const whole = (v) => Math.abs(v * 100 - Math.round(v * 100)) < 1e-6;
  const pq = (q, v) => (whole(v) ? q.pct(v) : q(v * 100, 1) + '%'), pt100 = (v) => (whole(v) ? H().pct(v) : H().fx(v * 100, 1) + '%');
  const POINTS = { near: 'the test point closest to the boundary', typical: 'a typical test point', far: 'the test point farthest from the boundary' };
  /** the chosen correctly classified test point: the least, the median or the most confident */
  P.pickPoint = (clf, which) => {
    const X = clf.data.Xte, y = clf.data.yte, pr = clf.probs(X);
    const ok = X.map((_, i) => i).filter((i) => pr[i].indexOf(Math.max(...pr[i])) === y[i]).sort((a, b) => pr[a][y[a]] - pr[b][y[b]] || a - b);
    return which === 'near' ? ok[0] : which === 'far' ? ok[ok.length - 1] : ok[Math.floor(ok.length / 2)];
  };
  const pt = (q, p) => '(' + q(p[0], 2) + ', ' + q(p[1], 2) + ')';
  const box = (ax, key, x, e, cls) => ({ key, kind: 'path', d: `M${ax.X(x[0] - e)},${ax.Y(x[1] - e)} H${ax.X(x[0] + e)} V${ax.Y(x[1] + e)} H${ax.X(x[0] - e)} Z`, cls: cls || 'p17box', audit: 'ignore', layer: 'lines', z: 3 });

  S.topic({
    id: 'robust-brittle', family: 'p17-robust', order: 10, title: 'Why models are brittle: a small nudge flips the answer',
    blurb: 'A classifier can be right on a point and wrong on another point a hair’s breadth away. The gradient of the loss with respect to the input shows which way to move to make the model most wrong; a step of size ε in each coordinate along its sign is often enough to cross the boundary. Seeing how small that step can be is the first step to defending against it.',
    inputs: [
      { id: 'which', kind: 'choice', label: 'Point', def: 'near', options: Object.keys(POINTS).map((k) => ({ value: k, label: POINTS[k] })) },
      { id: 'eps', kind: 'slider', label: 'Step size ε', def: 0.15, min: 0.02, max: 0.6, step: 0.01, fmt: (v) => v.toFixed(2) },
      { id: 'seed', kind: 'number', label: 'Seed (the network)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Near the boundary, ε 0.15', values: { which: 'near', eps: 0.15, seed: 1 } },
      { label: 'A typical point, ε 0.15', values: { which: 'typical', eps: 0.15, seed: 1 } },
      { label: 'A typical point, ε 0.4', values: { which: 'typical', eps: 0.4, seed: 1 } },
      { label: 'Far from it', values: { which: 'far', eps: 0.3, seed: 2 } },
    ],
    counters: [{ id: 'flips', label: 'answers flipped' }],
    run(inp) {
      const clf = P.moonsModel(inp.seed), F = H().frames(['flips']), q = F.q, d = clf.data;
      const ax = P.moonAxes('rb:ax', { x: 40, y: 40, w: 430, h: 277 });
      const pf = P.once('rbf:' + inp.seed, () => P.probField(ax, 'rb', (rows) => clf.probs(rows).map((p) => p[1])));
      const i = P.pickPoint(clf, inp.which), x = d.Xte[i], y = d.yte[i];
      const fg = S.Rob.fgsm(clf, x, y, inp.eps), g = fg.grad, gl = Math.hypot(...g), acc = clf.accuracy();
      const base = [pf.field, pf.line, ...ax.items, ...P.dots(ax, 'rb:p', d.Xte, d.yte)];
      const me = (cls, at) => ({ key: 'rb:me', kind: 'dot', x: ax.X((at || x)[0]), y: ax.Y((at || x)[1]), r: 6.5, cls: 'sp c' + y + ' p17pt ' + (cls || ''), layer: 'top', audit: 'overlap-ok' });
      const panel = (rows) => H().lines('rb:t', 510, 70, rows, 13, 22);
      const L = 70 / Math.hypot(ax.X(1) - ax.X(0), 0), tip = [x[0] + (g[0] / gl) * L, x[1] + (g[1] / gl) * L];
      const arrow = H().arrow('rb:g', [ax.X(x[0]), ax.Y(x[1])], [ax.X(tip[0]), ax.Y(tip[1])], 'p17grad');
      F.push([...base, me(), H().head('rb:h', 510, 44, 'One test point'), ...panel([['class ' + y + ' (' + (y ? 'orange' : 'blue') + ')', 'legend'], ['p(class ' + y + ') = ' + H().fx(fg.probsBefore[y], 3), 'mono'], ['shading: the model’s chance of orange', 'axname'], ['black line: its boundary, p = 0.5', 'axname']])],
        `A small network trained on ${P.qpl(q, d.Xtr.length, 'point')} of two moons gets ${pq(q, acc)} of the ${q.n(d.Xte.length)} test points right. Take ${POINTS[inp.which]}, at ${pt(q, x)}: the model says class ${q.n(fg.before)} with probability ${q(fg.probsBefore[fg.before], 3)}, which is right.`, { phase: 'start', acc, percent: [acc * 100], i, x, y, before: fg.before, pBefore: fg.probsBefore, nTrain: d.Xtr.length, nTest: d.Xte.length });
      F.push([...base, arrow, me(), H().head('rb:h', 510, 44, 'Where the loss rises fastest'), ...panel([['∇ₓ loss = (' + H().fx(g[0], 3) + ', ' + H().fx(g[1], 3) + ')', 'mono'], ['the arrow shows its direction', 'axname']])],
        `The gradient of the loss with respect to the input, not the weights, is (${q(g[0], 3)}, ${q(g[1], 3)}), of length ${q(gl, 3)}. It points the way the point would have to move to make the model most wrong: towards the boundary.`, { phase: 'grad', g, gl });
      const xa = fg.x, flipped = fg.after !== fg.before, dir = (v, a, b) => (v > 0 ? a : v < 0 ? b : 'not at all');
      F.push([...base, box(ax, 'rb:box', x, inp.eps), arrow, H().arrow('rb:s', [ax.X(x[0]), ax.Y(x[1])], [ax.X(xa[0]), ax.Y(xa[1])], 'p17step'), me(), H().head('rb:h', 510, 44, 'A step of ε = ' + H().fx(inp.eps, 2)), ...panel([['each coordinate moves by ε,', 'legend'], ['following the gradient’s sign', 'legend'], ['dashed square: every point within ε', 'axname']])],
        `Keep only the sign of each part of the gradient and move every coordinate by ε = ${q(inp.eps, 2)}: x₁ ${dir(g[0], 'right', 'left')}, x₂ ${dir(g[1], 'up', 'down')}. The point lands at ${pt(q, xa)}, a corner of the dashed square of all points no more than ε away in any coordinate.`, { phase: 'step', eps: inp.eps, xa });
      if (flipped) F.count('flips');
      F.push([...base, box(ax, 'rb:box', x, inp.eps), me('p17faded'), { key: 'rb:new', kind: 'dot', x: ax.X(xa[0]), y: ax.Y(xa[1]), r: 6.5, cls: 'sp c' + fg.after + ' p17moved', layer: 'top', audit: 'overlap-ok' }, H().head('rb:h', 510, 44, flipped ? 'The answer flipped' : 'Still right'), ...panel([['before: class ' + fg.before + ', p = ' + H().fx(fg.probsBefore[fg.before], 3), 'mono'], ['after:  class ' + fg.after + ', p = ' + H().fx(fg.probsAfter[fg.after], 3), 'mono']])],
        flipped ? `At the new point the model says class ${q.n(fg.after)} with probability ${q(fg.probsAfter[fg.after], 3)}. A change of ${q(inp.eps, 2)} in each coordinate, small beside the spread of the data, turned a right answer into a wrong one.`
          : `At the new point the model still says class ${q.n(fg.after)}, now with probability ${q(fg.probsAfter[fg.after], 3)}: this step was not big enough to cross the boundary.`, { phase: 'after', after: fg.after, pAfter: fg.probsAfter, flipped });
      const sf = S.Rob.smallestFlip(clf, x, y, { maxEps: 2 }), span = Math.max(...d.Xtr.map((r) => r[0])) - Math.min(...d.Xtr.map((r) => r[0]));
      const it = [...base, me()];
      if (sf) it.push(box(ax, 'rb:box', x, sf.eps, 'p17box'), { key: 'rb:new', kind: 'dot', x: ax.X(sf.x[0]), y: ax.Y(sf.x[1]), r: 6.5, cls: 'sp c' + sf.after + ' p17moved', layer: 'top', audit: 'overlap-ok' });
      it.push(H().head('rb:h', 510, 44, 'The smallest flipping step'), ...panel(sf ? [['ε* = ' + H().fx(sf.eps, 3), 'mono'], ['the data spans ' + H().fx(span, 2) + ' along x₁', 'legend']] : [['no step up to ε = 2 flips it', 'legend']]),
        ...H().lines('rb:d', 510, 190, [['Defences', 'strong'], ['train on nudged examples', 'legend'], ['vote over noisy copies', 'legend'], ['reject inputs that look unusual', 'legend']], 13, 22));
      F.push(it, sf ? `Searching along the same direction, the smallest step that flips the answer is ε = ${q(sf.eps, 3)}, while the training data spans ${q(span, 2)} along x₁. Defences make that step larger: training on nudged examples and voting over noisy copies are shown in [[robust-defend]].`
        : `Even a step of ${q(2, 0)} in each coordinate along this direction does not flip this point: it sits deep inside its class. The training data spans ${q(span, 2)} along x₁. Defences are shown in [[robust-defend]].`, { phase: 'smallest', epsStar: sf ? sf.eps : null, span, maxEps: 2 });
      for (const f of F.list) f.narrowView = { x: 0, y: 10, w: 490, h: 350 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A sign step (defensive view)', text: ['g = ∇ₓ loss(f(x), y)', 'x′ = x + ε · sign(g)', 'every coordinate moves by exactly ε', 'ε* = the smallest ε that changes the answer', 'larger ε* = a more robust model at x'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'grad', pick: 'option', once: true,
      ask: (fr, i, inp) => 'Will a step of ε = ' + inp.eps.toFixed(2) + ' flip the answer?',
      options: ['yes', 'no'],
      answer: (fr) => (fr.find((f) => f.meta.phase === 'after').meta.flipped ? 'yes' : 'no'),
      explain: (fr) => fr.find((f) => f.meta.phase === 'after').say,
    }],
    quiz: [
      { q: 'The gradient used here is taken with respect to…', options: ['the input point', 'the weights', 'the labels', 'the learning rate'], answer: 0, why: 'It says how the loss changes when the input moves.' },
      { q: 'With ε = 0.1 and gradient (0.3, −2), the sign step moves the point by…', options: ['(+0.1, −0.1)', '(+0.03, −0.2)', '(0, −0.1)', '(+0.1, +0.1)'], answer: 0, why: 'Each coordinate moves by ε in the direction of its sign.' },
      { q: 'Why are points near the boundary easiest to flip?', options: ['a short step is enough to cross it', 'their gradients are zero', 'they are mislabelled', 'they are never flipped'], answer: 0, why: 'The distance to the boundary is small.' },
      { q: 'A model has a larger smallest flipping ε on most points after a change. It has become…', options: ['more robust', 'less accurate for sure', 'more brittle', 'overfitted'], answer: 0, why: 'It takes a bigger nudge to change its answers.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const EPS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4];
  /** the noisy copies smoothing votes over, drawn the same way the engine draws them */
  P.copies = (x, sigma, k, seed) => { const g = S.T.rng(seed || 3); return Array.from({ length: k }, () => x.map((v) => v + g.normal(0, sigma))); };
  S.topic({
    id: 'robust-defend', family: 'p17-robust', order: 20, title: 'Hardening a model: adversarial training and smoothing',
    blurb: 'Two defences against small nudges. Adversarial training mixes nudged copies of the training points into every batch, so the model learns to keep its answer within ε of each one: its boundary moves away from the data. Smoothing asks the model about many noisy copies of an input and takes the majority vote, so a tiny nudge rarely changes the answer. Both cost a little clean accuracy and buy accuracy under nudges.',
    inputs: [
      { id: 'eps', kind: 'slider', label: 'Nudge size ε (training and test)', def: 0.2, min: 0.1, max: 0.3, step: 0.05, fmt: (v) => v.toFixed(2) },
      { id: 'sigma', kind: 'slider', label: 'Smoothing noise σ', def: 0.3, min: 0.1, max: 0.6, step: 0.05, fmt: (v) => v.toFixed(2) },
      { id: 'seed', kind: 'number', label: 'Seed (the networks)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'ε 0.2, σ 0.3', values: { eps: 0.2, sigma: 0.3, seed: 1 } },
      { label: 'Small nudges, ε 0.1', values: { eps: 0.1, sigma: 0.2, seed: 1 } },
      { label: 'Heavy smoothing, σ 0.6', values: { eps: 0.2, sigma: 0.6, seed: 1 } },
      { label: 'Another seed', values: { eps: 0.25, sigma: 0.3, seed: 4 } },
    ],
    counters: [{ id: 'models', label: 'models' }],
    run(inp) {
      const plain = P.moonsModel(inp.seed), adv = P.moonsModel(inp.seed, inp.eps), d = plain.data, F = H().frames(['models']), q = F.q;
      const ax = P.moonAxes('rd:ax', { x: 20, y: 60, w: 390, h: 251 });
      const fp = P.once('rdf:' + inp.seed, () => P.probField(ax, 'rd', (rows) => plain.probs(rows).map((p) => p[1])));
      const fa = P.once('rdfa:' + inp.seed + ':' + inp.eps, () => { const f = P.probField(ax, 'rd', (rows) => adv.probs(rows).map((p) => p[1])); f.line.cls += ' adv'; return f; });
      const cx = C().axes({ prefix: 'rd:cx', x: 480, y: 64, w: 250, h: 160, xr: [0, 0.4], yr: [0.2, 1], xticks: 0.1, yticks: 0.2, xname: 'ε of the nudge', yname: 'test accuracy', tickSize: 12 });
      const cp = P.once('rdc:' + inp.seed, () => S.Rob.robustCurve(plain, EPS, { n: 200 })), ca = P.once('rdca:' + inp.seed + ':' + inp.eps, () => S.Rob.robustCurve(adv, EPS, { n: 200 }));
      const X = d.Xte, y = d.yte, Xp = S.Rob.fgsmBatch(plain, X, y, inp.eps), Xa = S.Rob.fgsmBatch(adv, X, y, inp.eps);
      const pp = plain.predict(Xp), pa = adv.predict(Xa), accP = plain.accuracy(), accA = adv.accuracy();
      const attP = pp.filter((v, i) => v === y[i]).length / y.length, attA = pa.filter((v, i) => v === y[i]).length / y.length;
      const legend = C().legend('rd:lg', 480, 284, [{ text: 'plain training', cls: 'p17plain' }, { text: 'adversarial training', cls: 'p17adv' }], { dir: 'col' });
      const curve = (c, key, cls) => [C().line(cx, key, c.map((r) => [r.eps, r.accuracy]), cls)];
      const right = (which) => [...cx.items, ...legend, ...(which >= 1 ? curve(cp, 'rd:cp', 'p17plain') : []), ...(which >= 2 ? curve(ca, 'rd:ca', 'p17adv') : []), C().vline(cx, 'rd:ve', inp.eps, 'p17thr')];
      F.set('models', 1);
      F.push([fp.field, fp.line, ...ax.items, ...P.dots(ax, 'rd:p', X, y), ...right(0), H().head('rd:h', 20, 22, 'Plain training')],
        `A network trained the plain way gets ${pq(q, accP)} of the ${q.n(y.length)} test points right. Its boundary (black) passes close to many of them.`, { phase: 'plain', accP, percent: [accP * 100], n: y.length });
      F.push([fp.field, fp.line, ...ax.items, ...P.dots(ax, 'rd:p', Xp, y, { cls: (i) => (pp[i] !== y[i] ? 'p17wrong' : '') }), ...right(1), H().head('rd:h', 20, 22, 'Plain training, every test point nudged')],
        `Nudge every test point by ε = ${q(inp.eps, 2)} in the direction that hurts this model most: accuracy falls to ${pq(q, attP)} (outlined points are now wrong). The curve on the right repeats this for each ε.`, { phase: 'plainAtt', attP, percent: [attP * 100], eps: inp.eps });
      F.set('models', 2);
      F.push([fa.field, fa.line, ...ax.items, ...P.dots(ax, 'rd:p', X, y), ...right(1), H().head('rd:h', 20, 22, 'Adversarial training, ε = ' + H().fx(inp.eps, 2))],
        `Now train a second network the same way, but replace half of every batch by its own nudged copies at ε = ${q(inp.eps, 2)}. Its boundary (green) keeps more room around the points; clean accuracy is ${pq(q, accA)}, against ${pq(q, accP)} before.`, { phase: 'adv', accA, accP, percent: [accA * 100, accP * 100], eps: inp.eps });
      F.push([fa.field, fa.line, ...ax.items, ...P.dots(ax, 'rd:p', Xa, y, { cls: (i) => (pa[i] !== y[i] ? 'p17wrong' : '') }), ...right(2), H().head('rd:h', 20, 22, 'Adversarial training, points nudged')],
        `Nudged against this model by the same ε, the test points keep ${pq(q, attA)} accuracy, against ${pq(q, attP)} for the plain network. The green curve stays higher as ε grows: the model was trained for exactly this.`, { phase: 'advAtt', attA, attP, percent: [attA * 100, attP * 100] });
      // smoothing at one nudged point that fooled the plain model
      const fooled = X.map((_, i) => i).filter((i) => pp[i] !== y[i]);
      const k = 32, def = P.once('rdd:' + inp.seed + ':' + inp.eps + ':' + inp.sigma, () => S.Rob.defences(plain, inp.eps, { sigma: inp.sigma, samples: k, n: 200 }));
      const its = [fp.field, fp.line, ...ax.items, ...right(2), H().head('rd:h', 20, 22, 'Smoothing: a vote over noisy copies')];
      let say, meta = { phase: 'smooth' };
      if (fooled.length) {
        // the first fooled point that smoothing puts right, if there is one
        const fix = fooled.find((i) => S.Rob.smoothPredict(plain, Xp[i], { sigma: inp.sigma, samples: k }).label === y[i]);
        const j = fix === undefined ? fooled[0] : fix, xp = Xp[j], cps = P.copies(xp, inp.sigma, k, 3), cpred = plain.predict(cps), votes = [0, 0];
        cpred.forEach((c) => votes[c]++);
        its.push(...cps.map((c, t) => ({ key: 'rd:c' + t, kind: 'dot', x: ax.X(c[0]), y: ax.Y(c[1]), r: 3, cls: 'sp c' + cpred[t] + ' p17noisy', layer: 'items', audit: 'overlap-ok' })), { key: 'rd:me', kind: 'dot', x: ax.X(xp[0]), y: ax.Y(xp[1]), r: 6.5, cls: 'sp c' + y[j] + ' p17moved', layer: 'top', audit: 'overlap-ok' });
        const win = votes[1] > votes[0] ? 1 : 0;
        say = `Take one nudged point the plain model gets wrong (class ${q.n(y[j])}, ringed). Smoothing asks the same model about ${P.qpl(q, k, 'noisy copy', 'noisy copies')} with noise σ = ${q(inp.sigma, 2)}: ${q.n(votes[y[j]])} vote for class ${q.n(y[j])} and ${q.n(votes[1 - y[j]])} against, so the smoothed answer is class ${q.n(win)}` + (win === y[j] ? ', which is right again.' : ', still wrong: this nudge carried it well past the boundary.');
        Object.assign(meta, { j, votes, win, y: y[j], k, sigma: inp.sigma, xp });
      } else {
        say = `No nudged test point fools the plain model at ε = ${q(inp.eps, 2)}, so there is nothing for smoothing to repair here; smoothing with σ = ${q(inp.sigma, 2)} still costs a little on clean points.`;
        Object.assign(meta, { k, sigma: inp.sigma });
      }
      F.push(its, say, meta);
      const rows = [['no defence', def.rows[0].clean, def.rows[0].attacked], ['smoothing', def.rows[1].clean, def.rows[1].attacked], ['adversarial training', accA, attA]];
      const tab = S.Kit.dt.table({ prefix: 'rd:tab', x: 20, y: 70, cols: [{ name: 'defence', w: 190 }, { name: 'clean', w: 90 }, { name: 'nudged, ε = ' + H().fx(inp.eps, 2), w: 140 }], rows: rows.map((r) => [r[0], pt100(r[1]), pt100(r[2])]), rowH: 30, size: 13 });
      F.push([...tab.items, ...right(2), H().head('rd:h', 20, 40, 'Test accuracy, clean and nudged')],
        `On the same ${q.n(200)} test points: no defence ${pq(q, def.rows[0].clean)} clean and ${pq(q, def.rows[0].attacked)} nudged; smoothing with σ = ${q(inp.sigma, 2)} ${pq(q, def.rows[1].clean)} and ${pq(q, def.rows[1].attacked)}; adversarial training ${pq(q, accA)} and ${pq(q, attA)}. Each defence trades a little clean accuracy for steadier answers, and none is a guarantee.`, { phase: 'table', rows: def.rows, accA, attA, percent: def.rows.flatMap((r) => [r.clean * 100, r.attacked * 100]).concat([accA * 100, attA * 100]), n: 200, sigma: inp.sigma });
      for (const f of F.list) f.narrowView = { x: -20, y: 0, w: 450, h: 350 };
      F.last().narrowView = { x: 0, y: 20, w: 470, h: 200 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Two defences', text: ['adversarial training: each batch = half clean,', '  half x + ε · sign(∇ₓ loss), with true labels', 'smoothing: g(x) = majority of f(x + noise),', '  noise ~ N(0, σ²) per coordinate', 'cost: some clean accuracy; gain: steadier answers'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'plainAtt', pick: 'option', once: true,
      ask: 'Nudged by the same ε, will the adversarially trained network do better than the plain one?',
      options: ['better', 'about the same', 'worse'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'advAtt').meta, dd = m.attA - m.attP; return dd > 0.02 ? 'better' : dd < -0.02 ? 'worse' : 'about the same'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'advAtt').say,
    }],
    quiz: [
      { q: 'What does adversarial training add to each batch?', options: ['nudged copies of training points, with their true labels', 'random labels', 'more layers', 'test points'], answer: 0, why: 'The model learns to keep its answer near each training point.' },
      { q: 'Smoothing predicts by…', options: ['a majority vote over noisy copies of the input', 'averaging the weights', 'removing the boundary', 'training longer'], answer: 0, why: 'A small nudge rarely changes most of the votes.' },
      { q: 'A common cost of both defences is…', options: ['a little lower accuracy on clean inputs', 'no cost at all', 'a model that cannot be trained', 'larger inputs'], answer: 0, why: 'Robustness is bought with some clean accuracy.' },
      { q: 'Smoothing noise σ is made much larger. What tends to happen?', options: ['answers get steadier but clean accuracy drops', 'the model becomes brittle', 'nothing changes', 'accuracy rises everywhere'], answer: 0, why: 'Very noisy copies blur away the details the model needs.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
