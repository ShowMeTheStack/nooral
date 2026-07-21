// Robustness, part 2: distribution shift. A generator of training and test sets that differ in a controlled way,
// and the accuracy drop of a model trained on one and tested on the other.
//   S.Rob.shift({kind: 'covariate' | 'label' | 'concept', amount: 0..1, n, seed})
//      -> {train: {X, y}, test: {X, y}, kind, amount, say}
//      training inputs: x1 ~ N(−1.3, 0.5), x2 ~ N(0, 0.9), labelled by the rule y = [x2 > 0.8 sin(1.2 x1)], which is
//      nearly flat there, so a model learns "x2 above −0.8";
//      covariate: the inputs move (x1 drifts right by 2.6·amount, onto the crest of the wave) but the rule stays;
//      label:     the rule and the inputs per class stay, but the share of class 1 grows from 50% to 50 + 45·amount %;
//      concept:   the inputs stay but the rule itself moves: y = [x2 > 0.8 sin(1.2 x1 + π·amount)].
//   S.Rob.shiftCurve({kind, amounts: [0, 0.25, 0.5, 0.75, 1], seed, model: 'mlp' | 'logistic'})
//      -> {rows: [{amount, accuracy}], drop, trace}
(function (S) {
  'use strict';
  const R = S.Rob, fmt = (x) => R.fmt(x);
  const rule = (x1, x2, turn) => (x2 > 0.8 * Math.sin(1.2 * x1 + turn) ? 1 : 0);

  R.shiftKinds = ['covariate', 'label', 'concept'];
  R.shift = (o = {}) => {
    const kind = o.kind || 'covariate', a = o.amount == null ? 0.5 : o.amount, n = o.n || 400, g = S.Data.rng(o.seed || 1);
    const draw = (m, x1Mean, turn, pos) => {
      const X = [], y = [], quota = pos == null ? null : [m - Math.round(pos * m), Math.round(pos * m)];
      while (X.length < m) {
        const x1 = g.normal(x1Mean, 0.5), x2 = g.normal(0, 0.9), c = rule(x1, x2, turn);
        // label shift: fill a quota per class, so the class shares are exact while each class looks the same
        if (quota) { if (!quota[c]) continue; quota[c]--; }
        X.push([S.Data.round(x1, 4), S.Data.round(x2, 4)]); y.push(c);
      }
      return { X, y };
    };
    const train = draw(n, -1.3, 0, kind === 'label' ? 0.5 : null);
    const test = kind === 'covariate' ? draw(n, -1.3 + 2.6 * a, 0, null) : kind === 'label' ? draw(n, -1.3, 0, 0.5 + 0.45 * a) : draw(n, -1.3, Math.PI * a, null);
    const share = (s) => s.y.filter((v) => v === 1).length / s.y.length;
    const say = kind === 'covariate' ? 'Covariate shift: the test inputs are centred at x1 = ' + fmt(-1.3 + 2.6 * a) + ' instead of −1.3, but the labelling rule is unchanged.'
      : kind === 'label' ? 'Label shift: class 1 makes up ' + Math.round(100 * share(test)) + '% of the test set against ' + Math.round(100 * share(train)) + '% in training.'
      : 'Concept shift: the inputs look the same, but the boundary has moved by a phase of ' + fmt(Math.PI * a) + ' radians.';
    return { train, test, kind, amount: a, say };
  };

  R.shiftCurve = (o = {}) => {
    const T = S.T, amounts = o.amounts || [0, 0.25, 0.5, 0.75, 1], kind = o.kind || 'covariate', base = R.shift({ kind, amount: 0, seed: o.seed });
    let predict;
    if (o.model === 'logistic') { const m = R.logreg(base.train.X, base.train.y, { iters: 300 }); predict = (X) => X.map((x) => m.predict(x)); }
    else {
      const c = R.classifier({ dataset: { Xtr: base.train.X, ytr: base.train.y, Xte: base.test.X, yte: base.test.y, dim: 2, classes: [0, 1], image: false }, seed: o.seed || 1, steps: o.steps || 600, hidden: 16 });
      c.run(o.steps || 600); predict = (X) => c.predict(X);
    }
    const rows = amounts.map((a) => { const s = R.shift({ kind, amount: a, seed: o.seed }), p = predict(s.test.X); return { amount: a, accuracy: p.filter((v, i) => v === s.test.y[i]).length / p.length, say: s.say }; });
    const drop = rows[0].accuracy - Math.min(...rows.map((r) => r.accuracy));
    return { kind, rows, drop, trace: rows.map((r) => T.step('shift', r.say + ' Accuracy: ' + fmt(r.accuracy) + '.', r)).concat([T.step('drop', 'From no shift to the largest, accuracy falls by up to ' + fmt(drop) + ': the model only knows the world it was trained in.', { drop })]) };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
