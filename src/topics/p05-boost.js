// Part 5, boosting. AdaBoost: each round fits a stump to the points weighted by how often they were missed, gets a
// say alpha from its weighted error, and multiplies the weights of its mistakes by e^alpha (dot size = weight).
// Gradient boosting: each stage fits a small regression tree to the residuals of the model so far and adds it,
// shrunk by the learning rate; and three learning rates raced on the same data.
(function (S) {
  'use strict';
  const H = S.PL, T5 = S.PL5, ML = () => S.ML;
  const n$ = H.n;
  const SETS2 = T5.SETS.filter((s) => ['moons', 'circles', 'xor', 'exam'].includes(s.value));
  const leafV = (T, x) => T.nodes[ML().treeApply(T, x)].value;
  const SHAPES = [{ value: 'sine', label: 'sine' }, { value: 'step', label: 'steps' }, { value: 'poly', label: 'cubic' }, { value: 'abs', label: 'V shape' }];

  S.topic({
    id: 'adaboost', family: 'p05-boost', order: 10, title: 'AdaBoost: weights grow on mistakes',
    blurb: 'Every point starts with the same weight. Each round fits a weak learner, here a stump, to the weighted points and measures its weighted error ε. Its say in the final vote is α = ln((1 − ε) / ε): the better it is, the louder. Then every point it got wrong has its weight multiplied by e^α, and the weights are rescaled to sum to 1, so the next learner is pulled towards the points still being missed. The final model is the α-weighted vote of all rounds.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: SETS2 },
      { id: 'n', kind: 'number', label: 'points', def: 40, min: 16, max: 100 },
      { id: 'rounds', kind: 'number', label: 'rounds', def: 8, min: 2, max: 14 },
      { id: 'depth', kind: 'number', label: 'learner depth', def: 1, min: 1, max: 2 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Two moons with stumps', values: { set: 'moons', n: 40, rounds: 8, depth: 1, seed: 2 } },
      { label: 'Circles: many stumps make a box', values: { set: 'circles', n: 50, rounds: 12, depth: 1, seed: 3 } },
      { label: 'XOR with depth-2 learners', values: { set: 'xor', n: 40, rounds: 6, depth: 2, seed: 4 } },
      { label: 'Exam results', values: { set: 'exam', n: 60, rounds: 8, depth: 1, seed: 5 } },
    ],
    counters: [{ id: 'rounds', label: 'rounds' }],
    run(inp) {
      const d = T5.data2(inp.set, inp.n, inp.seed), n = d.X.length;
      const m = ML().AdaBoostClassifier({ nEstimators: inp.rounds, maxDepth: inp.depth, seed: inp.seed }).fit(d.X, d.y);
      const F = new S.Frames(['rounds']);
      const L = H.axes({ prefix: 'ab', x: 40, y: 50, w: 320, h: 290, xr: d.xr, yr: d.yr, xname: d.names[0], yname: d.names[1], grid: false });
      const R = H.axes({ prefix: 'ac', x: 430, y: 50, w: 320, h: 290, xr: d.xr, yr: d.yr, xname: d.names[0], grid: false });
      const stages = m.trace.filter((s) => s.kind === 'stage'), M = m.trees.length;
      const bax = H.axes({ prefix: 'aa', x: 40, y: 410, w: 710, h: 110, xr: [0.4, Math.max(M, 2) + 0.6], yr: [0, Math.max(...m.alphas, 0.5) * 1.1], xticks: 0, yticks: 0, grid: false, yname: 'the say α of each round' });
      const dots = (ax, key, w, wrong) => d.X.map((p, i) => ({ key: key + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: H.r2(Math.max(2, Math.min(13, 1.4 + 3.4 * Math.sqrt(w[i] * n)))), cls: 'sp c' + d.y[i] + (wrong && wrong[i] ? ' wrong' : ''), audit: 'overlap-ok', layer: 'items' }));
      const vote = (k) => (u, v) => { const s = m.decisionFunction([[u, v]], k)[0]; return [s > 0 ? 1 : 0, Math.max(0.2, Math.min(1, Math.abs(s) * 1.5))]; };
      const alphaBars = (upto) => {
        const items = [...bax.items];
        for (let k = 0; k < upto; k++) {
          const a = m.alphas[k], x = bax.X(k + 1), y0 = bax.Y(0), y1 = bax.Y(a);
          items.push({ key: 'aa:b' + k, kind: 'cell', x: H.r2(x - 14), y: y1, w: 28, h: H.r2(Math.max(1, y0 - y1)), rx: 2, text: '', cls: 'pl-pbar' + (k === upto - 1 ? '' : ' soft'), audit: 'ignore' });
          items.push(H.text('aa:t' + k, x, y0 + 16, n$(a, 2), 'tick', 12, 'middle'));
        }
        return items;
      };
      const head = (k) => [H.title('ab:t', 40, 12, k ? 'round ' + k + ': weighted points' : 'equal weights'), H.title('ac:t', 430, 12, k ? 'vote of rounds 1 to ' + k : 'the vote')];
      const w0 = new Array(n).fill(1 / n);
      F.add({ items: [...head(0), ...L.items, ...dots(L, 'ab:p', w0), ...R.items, ...S.Kit.scatter.points(R, 'ac:p', d.X, d.y, { r: 3.4 }), ...alphaBars(0)], say: n + ' points, each with weight 1/' + n + ' = ' + n$(1 / n) + '. The size of a dot will show its weight.', meta: { phase: 'start', n, w: 1 / n } });
      stages.forEach((st, k) => {
        const T = m.trees[k], wrong = st.wrong || d.X.map(() => false);
        F.set('rounds', k + 1);
        const nw = wrong.filter(Boolean).length, acc = T5.acc(d.y, m.decisionFunction(d.X, k + 1).map((v) => (v > 0 ? 1 : 0)));
        const field = T5.field(L, 'ab:f', (u, v) => [ML().argmax(leafV(T, [u, v])), 0.75], 60);
        const items = [...head(k + 1), ...L.items, field, T5.cuts(L, T, null, d.xr, d.yr, { key: 'ab:cut', cls: 'act' }), ...dots(L, 'ab:p', st.weights, wrong), ...R.items, T5.field(R, 'ac:f', vote(k + 1), 60), ...S.Kit.scatter.points(R, 'ac:p', d.X, d.y, { r: 3.4 }), ...alphaBars(k + 1)];
        const wsum = st.weights.reduce((s, v, i) => s + (wrong[i] ? v : 0), 0);
        let say = 'Round ' + (k + 1) + ': the learner misses ' + H.plural(nw, 'point') + ' (ringed) carrying ' + n$(wsum, 3) + ' of the weight, so ε = ' + n$(st.error, 3) + ' and α = ln((1 − ε)/ε) = ' + n$(st.alpha, 3) + '.';
        if (st.error > 0) say += (k < M - 1 ? ' Their weights are multiplied by e^α = ' + n$(Math.exp(st.alpha), 3) + ' for the next round.' : '') + ' The vote of ' + H.plural(k + 1, 'round') + ' gets ' + H.pct(acc) + ' of the points right.';
        F.add({ items, say, meta: { phase: 'round', k: k + 1, wrongN: nw, wsum, err: st.error, alpha: st.alpha, factor: Math.exp(st.alpha), acc, maxW: Math.max(...st.weights) } });
      });
      const stop = m.trace.find((s) => s.kind === 'stop');
      const accAll = T5.acc(d.y, m.predict(d.X)), heavy = ML().argmax(m.sampleWeight);
      F.add({ items: [...head(M), ...L.items, ...dots(L, 'ab:p', m.sampleWeight, null), ...R.items, T5.field(R, 'ac:f', vote(M), 60), ...S.Kit.scatter.points(R, 'ac:p', d.X, d.y, { r: 3.4 }), ...alphaBars(M)], say: (stop ? 'Round ' + (stop.round + 1) + ' was no better than a coin (ε = ' + n$(stop.error, 3) + '), so boosting stopped. ' : '') + 'The vote of ' + H.plural(M, 'learner') + ' gets ' + H.pct(accAll) + ' of the training points right. The heaviest point now carries ' + n$(m.sampleWeight[heavy], 3) + ' of the weight, ' + n$(m.sampleWeight[heavy] * n, 3) + ' times its start: the points near the boundary are the ones boosting fought over.', meta: { phase: 'done', M, acc: accAll, heavy: m.sampleWeight[heavy], ratio: m.sampleWeight[heavy] * n, stopErr: stop ? stop.error : null, stopRound: stop ? stop.round + 1 : null } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'AdaBoost', text: ['start: wᵢ = 1/n', 'ε = Σ wᵢ over mistakes / Σ wᵢ', 'α = ln((1 − ε) / ε)', 'mistakes: wᵢ ← wᵢ·e^α, then rescale', 'predict: sign of Σ α·hₘ(x)'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'round' && f.meta.k === 1 && fr[i + 1] && fr[i + 1].meta.phase === 'round', pick: 'option', ask: 'Next round, will the learner’s weighted error be computed with these ringed points heavier or lighter?', options: ['heavier', 'lighter', 'the same'], answer: 'heavier', explain: (fr, i) => 'They were multiplied by e^α = ' + n$(fr[i].meta.factor, 3) + ' and the weights rescaled.' },
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will the first learner get the largest say α of all rounds?', options: ['yes', 'no'], answer: (fr) => { const a = fr.filter((x) => x.meta.phase === 'round').map((x) => x.meta.alpha); return a[0] >= Math.max(...a) - 1e-12 ? 'yes' : 'no'; }, explain: 'The first stump sees easy, equally weighted points; later ones face the hard points and usually have larger errors, so smaller say.' },
    ],
    quiz: [
      { q: 'A learner has weighted error ε = 0.2. Its say α is…', options: ['ln 4 ≈ 1.39', '0.8', '0.2', 'ln 0.25'], answer: 0, why: 'α = ln((1 − 0.2)/0.2) = ln 4.' },
      { q: 'What happens to a point the current learner gets right?', options: ['its weight is unchanged before rescaling, so it shrinks relative to the mistakes', 'its weight doubles', 'it is removed', 'its label flips'], answer: 0, why: 'Only mistakes are multiplied by e^α; rescaling then lowers the rest.' },
      { q: 'A learner with ε = 0.5 gets α = …', options: ['0: it is no better than a coin', '1', 'infinite', '0.5'], answer: 0, why: 'ln(0.5/0.5) = 0.' },
      { q: 'Why can AdaBoost be thrown off by mislabelled points?', options: ['their weights keep growing since no learner gets them right', 'they are ignored', 'they reduce α to zero', 'stumps cannot see them'], answer: 0, why: 'A wrong label is a mistake every round, so its weight explodes.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Gradient boosting fits the residuals
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'gb-residuals', family: 'p05-boost', order: 20, title: 'Gradient boosting fits the residuals',
    blurb: 'Start from the mean. At each stage compute the residuals, what the model so far still gets wrong (for squared error they are the negative gradient of the loss), fit a small regression tree to them, and add that tree times the learning rate. Stage by stage the residuals shrink towards noise.',
    inputs: [
      { id: 'shape', kind: 'choice', label: 'curve', def: 'sine', options: SHAPES },
      { id: 'n', kind: 'number', label: 'points', def: 40, min: 12, max: 100 },
      { id: 'stages', kind: 'number', label: 'stages', def: 6, min: 1, max: 12 },
      { id: 'depth', kind: 'number', label: 'tree depth', def: 2, min: 1, max: 3 },
      { id: 'lr', kind: 'slider', label: 'learning rate', def: 0.5, min: 0.1, max: 1, step: 0.05 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.2, min: 0, max: 0.8, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 3, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Sine, depth-2 trees, rate 0.5', values: { shape: 'sine', n: 40, stages: 6, depth: 2, lr: 0.5, noise: 0.2, seed: 3 } },
      { label: 'Stumps at full rate', values: { shape: 'sine', n: 40, stages: 8, depth: 1, lr: 1, noise: 0.2, seed: 3 } },
      { label: 'Steps', values: { shape: 'step', n: 40, stages: 4, depth: 1, lr: 1, noise: 0.15, seed: 4 } },
      { label: 'A cubic, small rate', values: { shape: 'poly', n: 50, stages: 10, depth: 2, lr: 0.3, noise: 0.25, seed: 5 } },
    ],
    counters: [{ id: 'stage', label: 'stages' }],
    run(inp) {
      const d = S.Data.curve({ n: inp.n, shape: inp.shape, noise: inp.noise, seed: inp.seed }), X = d.X, y = d.y, n = X.length, lr = inp.lr;
      const m = ML().GradientBoostingRegressor({ nEstimators: inp.stages, learningRate: lr, maxDepth: inp.depth, seed: inp.seed }).fit(X, y);
      const st = m.trace.filter((s) => s.kind === 'stage'), F = new S.Frames(['stage']);
      const ax = H.axes({ prefix: 'gb', x: 46, y: 34, w: 700, h: 190, xr: [-3, 3], yr: H.nice(y, 0.08), xticks: 1, xname: 'x', yname: 'y and the model', grid: true });
      const res0 = y.map((v) => v - m.init[0]), rr = Math.max(...res0.map(Math.abs)) * 1.1;
      const rx = H.axes({ prefix: 'gr', x: 46, y: 290, w: 700, h: 160, xr: [-3, 3], yr: [-rr, rr], xticks: 1, yticks: S.Kit.curve.ticks(-rr, rr, 4), xname: 'x', yname: 'residuals y − model, and the tree fitted to them', grid: true });
      const grid = Array.from({ length: 301 }, (_, k) => -3 + (6 * k) / 300);
      const curve = (ax2, key, f, cls) => S.Kit.curve.line(ax2, key, grid.map((x) => [x, f(x)]), cls);
      const modelAt = (k) => (x) => m.raw([[x]], k)[0][0];
      const pts = () => S.Kit.scatter.points(ax, 'gb:p', X.map((r, i) => [r[0], y[i]]), X.map(() => 0), { r: 3.4 });
      const rpts = (res) => S.Kit.scatter.points(rx, 'gr:p', X.map((r, i) => [r[0], res[i]]), X.map(() => 1), { r: 3.4 });
      const lossAx = H.axes({ prefix: 'gl', x: 46, y: 500, w: 260, h: 50, xr: [0, inp.stages], yr: [0, ML().mse(y, y.map(() => m.init[0])) * 1.05], xticks: 0, yticks: 0, grid: false, yname: 'training squared error' });
      const losses = [[0, ML().mse(y, y.map(() => m.init[0]))]];
      F.add({ items: [...ax.items, curve(ax, 'gb:F', () => m.init[0], 'act'), ...pts(), ...rx.items, ...rpts(res0), ...lossAx.items, S.Kit.curve.line(lossAx, 'gl:l', losses.slice(), 'act')], say: 'Stage 0 predicts the mean, ' + n$(m.init[0]) + ', everywhere. The residuals below are what it gets wrong; their mean square is ' + n$(losses[0][1]) + '.', meta: { phase: 'init', init: m.init[0], mse: losses[0][1] } });
      let prev = X.map(() => m.init[0]);
      st.forEach((s, k) => {
        const T = m.stages[k][0], res = y.map((v, i) => v - prev[i]), now = s.raw.map((r) => r[0]);
        losses.push([k + 1, s.loss]);
        F.set('stage', k + 1);
        const leaves = T.nodes.filter((x) => x.left < 0).map((x) => x.value);
        const items = [...ax.items, curve(ax, 'gb:Fp', modelAt(k), 'faint'), curve(ax, 'gb:F', modelAt(k + 1), 'act'), ...pts(), ...rx.items, ...rpts(res), curve(rx, 'gr:h', (x) => leafV(T, [x]), 'c0'), curve(rx, 'gr:hs', (x) => lr * leafV(T, [x]), 'faint'), ...lossAx.items, S.Kit.curve.line(lossAx, 'gl:l', losses.slice(), 'act'), T5.dot(lossAx, 'gl:d', k + 1, s.loss, '')];
        const big = res.reduce((b, v, i) => (Math.abs(v) > Math.abs(res[b]) ? i : b), 0);
        F.add({ items, say: 'Stage ' + (k + 1) + ': a depth-' + inp.depth + ' tree is fitted to the residuals (largest ' + n$(res[big]) + ' at x = ' + n$(X[big][0]) + '); its ' + leaves.length + ' leaves predict from ' + n$(Math.min(...leaves)) + ' to ' + n$(Math.max(...leaves)) + '. Added at rate ' + n$(lr) + ' (dashed), it moves the model to the solid line, and the training squared error falls to ' + n$(s.loss) + '.', meta: { phase: 'stage', k: k + 1, depth: inp.depth, resBig: res[big], xBig: X[big][0], nLeaves: leaves.length, lo: Math.min(...leaves), hi: Math.max(...leaves), lr, mse: s.loss, prevMse: losses[k][1] } });
        prev = now;
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Gradient boosting', text: ['F₀(x) = mean of y', 'rᵢ = yᵢ − Fₘ₋₁(xᵢ)  (−gradient of ½(y − F)²)', 'hₘ = small tree fitted to (xᵢ, rᵢ)', 'Fₘ = Fₘ₋₁ + ν·hₘ   (ν: learning rate)'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'init' && fr.length > 2, pick: 'option', ask: 'Will the training error fall at every stage?', options: ['yes', 'no'], answer: (fr) => (fr.slice(1).every((x) => x.meta.mse <= x.meta.prevMse + 1e-12) ? 'yes' : 'no'), explain: 'Each tree is fitted to the residuals, so adding a fraction of it can only lower the squared error on the training points (or leave it).' },
    ],
    quiz: [
      { q: 'For squared error, the negative gradient of the loss with respect to the prediction is…', options: ['the residual y − F(x)', 'the prediction', 'the squared residual', 'zero'], answer: 0, why: '−∂/∂F ½(y − F)² = y − F.' },
      { q: 'What does each new tree in gradient boosting fit?', options: ['what the model so far gets wrong', 'the original targets', 'a bootstrap sample', 'the weights of the mistakes'], answer: 0, why: 'Each stage corrects the previous model.' },
      { q: 'The model after three stages with rate ν is…', options: ['F₀ + ν·(h₁ + h₂ + h₃)', 'the average of h₁, h₂, h₃', 'h₃ alone', 'F₀·ν³'], answer: 0, why: 'Stages add up.' },
      { q: 'Why are the trees in gradient boosting kept small?', options: ['each only needs to make a small correction; many small steps generalise better', 'large trees cannot fit residuals', 'to save memory only', 'the residuals are always constant'], answer: 0, why: 'Weak learners added slowly avoid fitting the noise in one go.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Learning rate and shrinkage
  // ---------------------------------------------------------------------------------------------------------------
  const RATES = [1, 0.3, 0.1];
  S.topic({
    id: 'gb-shrinkage', family: 'p05-boost', order: 30, title: 'Learning rate and shrinkage',
    blurb: 'Each stage adds only ν times its tree. With ν = 1 the model chases the residuals at full speed and soon starts fitting the noise; with a small ν it needs many more stages, but each correction is gentle and the error on fresh points usually ends lower. Three rates race here on the same points, with the error on training and on test points after every stage.',
    inputs: [
      { id: 'shape', kind: 'choice', label: 'curve', def: 'sine', options: SHAPES },
      { id: 'n', kind: 'number', label: 'points', def: 40, min: 15, max: 120 },
      { id: 'stages', kind: 'number', label: 'stages', def: 60, min: 10, max: 120 },
      { id: 'depth', kind: 'number', label: 'tree depth', def: 3, min: 1, max: 4 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.35, min: 0.05, max: 0.8, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 6, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Noisy sine, depth 3', values: { shape: 'sine', n: 40, stages: 60, depth: 3, noise: 0.35, seed: 6 } },
      { label: 'Stumps, 100 stages', values: { shape: 'sine', n: 60, stages: 100, depth: 1, noise: 0.3, seed: 2 } },
      { label: 'Clean steps', values: { shape: 'step', n: 40, stages: 30, depth: 2, noise: 0.05, seed: 3 } },
      { label: 'Few points, deep trees', values: { shape: 'poly', n: 20, stages: 40, depth: 4, noise: 0.4, seed: 7 } },
    ],
    counters: [{ id: 'stage', label: 'stages' }],
    run(inp) {
      const tr = S.Data.curve({ n: inp.n, shape: inp.shape, noise: inp.noise, seed: inp.seed }), te = S.Data.curve({ n: 200, shape: inp.shape, noise: inp.noise, seed: inp.seed + 1000 }), M = inp.stages;
      const models = RATES.map((lr) => ML().GradientBoostingRegressor({ nEstimators: M, learningRate: lr, maxDepth: inp.depth, seed: inp.seed }).fit(tr.X, tr.y));
      const testCurve = models.map((m) => m.stagedPredict(te.X).map((p) => ML().mse(te.y, p)));
      const trainCurve = models.map((m) => m.trainLoss);
      const F = new S.Frames(['stage']);
      const ax = H.axes({ prefix: 'gs', x: 46, y: 34, w: 700, h: 200, xr: [-3, 3], yr: H.nice(tr.y, 0.08), xticks: 1, xname: 'x', yname: 'y and the three models', grid: true });
      const top = Math.max(ML().mse(tr.y, tr.y.map(() => ML().mean(tr.y))), ML().mse(te.y, te.y.map(() => ML().mean(tr.y)))) * 1.05;
      const lx = H.axes({ prefix: 'gt', x: 46, y: 300, w: 300, h: 190, xr: [0, M], yr: [0, top], xname: 'stage', yname: 'training squared error', grid: true });
      const tx = H.axes({ prefix: 'gu', x: 446, y: 300, w: 300, h: 190, xr: [0, M], yr: [0, top], xname: 'stage', yname: 'test squared error (200 fresh points)', grid: true });
      const legend = S.Kit.curve.legend('gs:lg', 46, 552, RATES.map((r, k) => ({ text: 'ν = ' + r, cls: 'c' + k })));
      const grid = Array.from({ length: 241 }, (_, k) => -3 + (6 * k) / 240);
      const sched = [...new Set([1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 80, 100, 120, M].filter((s) => s <= M))].sort((a, b) => a - b);
      const best = models.map((_, r) => { const c = testCurve[r]; const k = c.indexOf(Math.min(...c)); return { k: k + 1, v: c[k] }; });
      sched.forEach((s) => {
        F.set('stage', s);
        const items = [...ax.items, ...S.Kit.scatter.points(ax, 'gs:p', tr.X.map((r, i) => [r[0], tr.y[i]]), tr.X.map(() => 0), { r: 3.2, cls: () => 'cn' }), ...lx.items, ...tx.items, ...legend];
        models.forEach((m, r) => {
          items.push(S.Kit.curve.line(ax, 'gs:f' + r, grid.map((x) => [x, m.raw([[x]], s)[0][0]]), 'c' + r));
          items.push(S.Kit.curve.line(lx, 'gt:l' + r, trainCurve[r].slice(0, s).map((v, k) => [k + 1, v]), 'c' + r), S.Kit.curve.line(tx, 'gu:l' + r, testCurve[r].slice(0, s).map((v, k) => [k + 1, v]), 'c' + r));
        });
        const say = 'After ' + H.plural(s, 'stage') + ': training error ' + RATES.map((r, k) => n$(trainCurve[k][s - 1]) + ' at ν = ' + r).join(', ') + '; test error ' + RATES.map((r, k) => n$(testCurve[k][s - 1])).join(', ') + ' in the same order.' + (s === M ? ' The lowest test error for each rate: ' + RATES.map((r, k) => n$(best[k].v) + ' at stage ' + best[k].k).join(', ') + '.' : '');
        F.add({ items, say, meta: { phase: s === M ? 'last' : 'step', s, train: RATES.map((_, k) => trainCurve[k][s - 1]), test: RATES.map((_, k) => testCurve[k][s - 1]), rates: RATES, best: best.map((b) => b.v), bestK: best.map((b) => b.k), nte: 200 } });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Shrinkage', text: ['Fₘ = Fₘ₋₁ + ν·hₘ,  0 < ν ≤ 1', 'small ν: more stages, smoother path', 'ν and the number of stages trade off:', '  pick both on held-out data'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.s === 1, pick: 'option', ask: 'Which rate will reach the lowest test error at some stage?', options: ['ν = 1', 'ν = 0.3', 'ν = 0.1'], answer: (fr) => { const b = fr[fr.length - 1].meta.best; return 'ν = ' + RATES[b.indexOf(Math.min(...b))]; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Halving the learning rate usually means you need about…', options: ['twice as many stages', 'half as many stages', 'the same number of stages', 'deeper trees only'], answer: 0, why: 'Each stage moves half as far.' },
      { q: 'Why does a small learning rate often generalise better?', options: ['many small corrections average out the noise each tree fits', 'it makes trees deeper', 'it uses fewer points', 'it removes the need for trees'], answer: 0, why: 'Shrinkage is a form of regularisation.' },
      { q: 'With ν = 1 and deep trees, training error after many stages is…', options: ['close to zero: the model fits the noise', 'stuck at the variance of y', 'growing', 'negative'], answer: 0, why: 'Nothing slows it down, so it memorises.' },
      { q: 'The number of stages in boosting acts like…', options: ['a complexity knob, to be chosen by held-out error (early stopping)', 'a random seed', 'the depth of each tree', 'a learning rate for the data'], answer: 0, why: 'More stages, more complex model.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
