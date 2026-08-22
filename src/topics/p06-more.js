// Part 6, beyond two classes: many classes from binary machines, one against the rest or one for every pair; SVR,
// which fits a tube of half-width ε and pays only for points outside it; and the one-class SVM, which draws a
// border round normal data with ν bounding the share of points left outside.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const NAMES = ['blue', 'orange', 'green'];

  /** three classes: three of our blobs, or three iris species by their petals (standardised) */
  const three = (inp) => {
    if (inp.set === 'iris') {
      const d = S.Data.load('iris'), X0 = d.X.map((x) => [x[2], x[3]]), m = ML().colMeans(X0), sd = ML().colVars(X0).map(Math.sqrt);
      return { X: X0.map((r) => r.map((v, j) => +((v - m[j]) / sd[j]).toFixed(4))), y: d.y.slice(), names: ['setosa', 'versicolor', 'virginica'] };
    }
    const d = S.Data.blobs({ n: 45, centers: [[-1.2, -0.6], [1.2, -0.6], [0, 1.2]], std: inp.set === 'close' ? 0.75 : 0.5, seed: inp.seed });
    return { X: d.X, y: d.y, names: NAMES };
  };

  S.topic({
    id: 'svm-multiclass', family: 'p06-more', order: 10, title: 'Many classes: one-vs-rest and one-vs-one',
    blurb: 'An SVM separates two classes. For k classes, one-vs-rest trains k machines, each class against all the others, and picks the class whose machine is most confident. One-vs-one trains a machine for every pair, k(k − 1)/2 of them, and lets them vote. With three classes both need three machines; with ten, one-vs-rest needs 10 and one-vs-one 45, though each of those sees only two classes’ points.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'apart', options: [{ value: 'apart', label: 'three groups' }, { value: 'close', label: 'three groups, close' }, { value: 'iris', label: 'iris petals: three species' }] },
      { id: 'kernel', kind: 'choice', label: 'Kernel', def: 'linear', options: [{ value: 'linear', label: 'linear' }, { value: 'rbf', label: 'RBF, γ = 1' }] },
      { id: 'C', kind: 'slider', label: 'C', def: 1, min: 0.1, max: 100, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Three groups, linear', values: { set: 'apart', kernel: 'linear', C: 1, seed: 1 } },
      { label: 'Close groups, RBF', values: { set: 'close', kernel: 'rbf', C: 1, seed: 2 } },
      { label: 'Iris petals (real data)', values: { set: 'iris', kernel: 'linear', C: 1, seed: 1 } },
    ],
    counters: [{ id: 'machines', label: 'machines' }],
    run(inp) {
      const d = three(inp), X = d.X, y = d.y, nm = d.names, Cv = +H().sg(inp.C, 2), F = H().frames(['machines']), q = F.q;
      const mid = ML().colMeans(X);
      const fit = (mc) => { const m = new (ML().SVC)({ kernel: inp.kernel, gamma: 1, C: Cv, tol: 1e-4, multiClass: mc }).fit(X, y); P.record('svc', X, y, m.params, m, [mid]); return m; };
      const ovr = fit('ovr'), ovo = fit('ovo');
      const ax = P.axes('mc', X, { x: 40, y: 24, w: 400, h: 400 }, inp.set === 'iris' ? { pad: 0.3, xname: 'petal length', yname: 'petal width' } : { pad: 0.3 });
      const SC = S.Kit.scatter, RX = 510;
      const pts = (fade) => SC.points(ax, 'mc:p', X, y, { r: 4.4 }).map((it, i) => (fade && fade(i) ? Object.assign(it, { opacity: 0.25 }) : it));
      const lines = (m, t, cls) => { const f = (u, v) => m.machineValues([[u, v]])[0][t], g = SC.grid(ax, f, 70); return [SC.level(ax, 'mc:b' + t, g, 0, cls), SC.level(ax, 'mc:m' + t + 'a', g, 1, 'margin soft'), SC.level(ax, 'mc:m' + t + 'b', g, -1, 'margin soft')]; };
      const regions = (m) => SC.field(ax, 'mc:reg', (u, v) => [m.predict([[u, v]])[0], 0.9], { n: 90, scheme: 'classes', smooth: false });
      const accOf = (m) => ML().accuracy(y, m.predict(X));
      const legend = C().legend('mc:lg', RX, 400, nm.map((t, k) => ({ text: t, cls: 'c' + k, dot: true })), { dir: 'col', size: 13 });
      const panel = (title, rows) => [H().head('mc:pt', RX, 50, title), ...H().lines('mc:pl', RX, 80, rows, 13, 24), ...legend];
      // the point we classify by hand: the middle of the data, where the classes meet
      const vo = ovo.machineValues([mid])[0], vr = ovr.machineValues([mid])[0];
      const midDot = { key: 'mc:mid', kind: 'state', x: ax.X(mid[0]), y: ax.Y(mid[1]), w: 14, h: 14, rx: 2, text: '', cls: 'centre', audit: 'overlap-ok' };

      F.push([...ax.items, ...pts(), ...panel('Three classes, binary machines', [['one-vs-rest: one machine per class', 'legend'], ['one-vs-one: one machine per pair', 'legend']])], `${q.n(X.length)} points in three classes (${nm.join(', ')}). One SVM can only split two classes, so we combine several: first one-vs-rest, then one-vs-one.`, { phase: 'data', n: X.length });
      ovr.machines.forEach((m, t) => {
        F.count('machines');
        const k = m.cls, own = X.filter((_, i) => ovr.machineValues([X[i]])[0][t] > 0 && y[i] === k).length;
        F.push([...ax.items, ...lines(ovr, t, 'c' + k), ...pts((i) => y[i] !== k), ...panel('One-vs-rest', [[`machine ${t + 1}: ${nm[k]} against the rest`, 'c' + k], [`${m.sv.length} support vectors`, 'legend']])],
          `One-vs-rest, machine ${q.n(t + 1)}: ${nm[k]} (+1) against both other classes together (−1). It has ${q.n(m.sv.length)} support vectors and puts ${q.n(own)} of the ${q.n(y.filter((v) => v === k).length)} ${nm[k]} points on its positive side.`, { phase: 'ovr', t, no: t + 1, cls: k, nsv: m.sv.length, own, count: y.filter((v) => v === k).length });
      });
      const ra = accOf(ovr), rk = ML().argmax(vr);
      F.push([...ax.items, regions(ovr), ...pts(), midDot, ...panel('One-vs-rest: the most confident', vr.map((v, k) => [`${nm[k]} machine: f = ${H().fx(v)}`, 'c' + k]))],
        `Each place goes to the class whose machine gives the largest value f. At the square in the middle the three machines say ${q(vr[0])}, ${q(vr[1])} and ${q(vr[2])}, so it is ${nm[rk]}. Training accuracy ${q.pct(ra)}. Where every machine says “not mine”, the largest of three negative values still decides.`, { phase: 'ovr-all', acc: ra, vals: vr, win: rk });
      F.set('machines', 0);
      ovo.machines.forEach((m, t) => {
        F.count('machines');
        const [a, b] = m.pair;
        F.push([...ax.items, ...lines(ovo, t, ''), ...pts((i) => y[i] !== a && y[i] !== b), ...panel('One-vs-one', [[`machine ${t + 1}: ${nm[a]} against ${nm[b]}`, 'strong'], [`${m.sv.length} support vectors`, 'legend'], [`the ${nm[3 - a - b]} points are not used`, 'legend']])],
          `One-vs-one, machine ${q.n(t + 1)}: ${nm[a]} against ${nm[b]} only, trained on their ${q.n(y.filter((v) => v === a || v === b).length)} points, with ${q.n(m.sv.length)} support vectors. The ${nm[3 - a - b]} points play no part in it.`, { phase: 'ovo', t, no: t + 1, a, b, nsv: m.sv.length, used: y.filter((v) => v === a || v === b).length });
      });
      const votes = [0, 0, 0];
      ovo.machines.forEach((m, t) => votes[vo[t] > 0 ? m.pair[0] : m.pair[1]]++);
      const oa = accOf(ovo), ok = ovo.predict([mid])[0];
      const agree = ML().accuracy(ovo.predict(X), ovr.predict(X));
      F.push([...ax.items, regions(ovo), ...pts(), midDot, ...panel('One-vs-one: the vote', ovo.machines.map((m, t) => [`${nm[m.pair[0]]} or ${nm[m.pair[1]]}: ${nm[vo[t] > 0 ? m.pair[0] : m.pair[1]]}`, 'c' + (vo[t] > 0 ? m.pair[0] : m.pair[1])]).concat([[`votes: ${votes.join(', ')}`, 'strong']]))],
        `Each pair’s machine votes for one of its two classes, and the class with most votes wins. At the square the votes are ${nm[0]} ${q.n(votes[0])}, ${nm[1]} ${q.n(votes[1])} and ${nm[2]} ${q.n(votes[2])}, so it is ${(() => { const top = votes.map((v, k) => k).filter((k) => votes[k] === Math.max(...votes)); return top.length === 1 ? nm[ok] : (top.length === 3 ? 'a three-way tie' : 'a tie between ' + nm[top[0]] + ' and ' + nm[top[1]]) + ', broken in favour of the first class in the list, ' + nm[ok]; })()}. Training accuracy ${q.pct(oa)}; the two schemes agree on ${q.pct(agree)} of the training points.`, { phase: 'ovo-all', acc: oa, votes, win: ok, agree, names: nm });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Two schemes', text: ['one-vs-rest: k machines, class j against the rest', '  predict argmax_j f_j(x)', 'one-vs-one: k(k − 1)/2 machines, one per pair', '  each votes; most votes wins', 'k = 3: three machines either way'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'How many machines does one-vs-one need for three classes?', options: ['3', '2', '6', '9'], answer: '3', explain: 'One per pair: 3·2/2 = 3.' },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'ovo-all', pick: 'option', ask: 'Which class will win the vote at the square in the middle?', options: (fr, i) => fr[i + 1].meta.names.slice(), answer: (fr, i) => fr[i + 1].meta.names[fr[i + 1].meta.win], explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'How many one-vs-one machines are needed for 10 classes?', options: ['45', '10', '90', '100'], answer: 0, why: '10·9/2 = 45.' },
      { q: 'How does one-vs-rest pick a class?', options: ['the class whose machine gives the largest decision value', 'a vote among pairs', 'the class with most training points', 'at random among the positive machines'], answer: 0, why: 'Each machine scores its own class; the highest score wins.' },
      { q: 'Why can one-vs-one be fast despite needing more machines?', options: ['each machine trains on only two classes’ points', 'its machines have no support vectors', 'it skips the kernel', 'it needs no C'], answer: 0, why: 'SVM training grows faster than linearly in the number of points, so many small problems can beat a few large ones.' },
      { q: 'What can go wrong with one-vs-rest’s machines?', options: ['each sees one class against a much larger rest, an imbalanced problem', 'they can never agree', 'they need three times as many points', 'nothing'], answer: 0, why: 'One class against all others is lopsided, and the machines’ scores are not on a common scale.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
