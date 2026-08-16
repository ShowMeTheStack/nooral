// Part 3, the perceptron. It visits the points in order; a point on the wrong side of the line (or on it) is a
// mistake, and the rule adds the point to the weight vector (times its label ±1 and the step η). Every update turns
// the line toward the point. Each update in ML.Perceptron's trace is a frame: the old line dashed, the new one solid,
// and the weight vector before and after, drawn from the origin on equal axes so it stands at right angles to the line.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, SC = () => S.Kit.scatter;

  const DATA = [{ value: 'easy', label: 'well apart' }, { value: 'narrow', label: 'a narrow gap' }, { value: 'overlap', label: 'overlapping' }];
  H.percData = (kind, n, seed) => {
    const r = S.Data.rng(seed), X = [], y = [], dir = [Math.cos(0.5), Math.sin(0.5)];
    const gap = kind === 'easy' ? 1.2 : kind === 'narrow' ? 0.35 : -0.6;
    while (X.length < n) {
      const k = X.length % 2, u = r.range(-2.6, 2.6), v = r.range(0.05, 2.3) * (k ? 1 : -1);
      const along = [-dir[1], dir[0]], off = v + (k ? 1 : -1) * Math.max(0, gap) * 0.5 + (gap < 0 ? r.normal(0, -gap) : 0);
      const p = [u * dir[0] + off * along[0] + 0.4, u * dir[1] + off * along[1] - 0.3];
      if (Math.abs(p[0]) > 3 || Math.abs(p[1]) > 3) continue;
      X.push(p.map((q) => S.Data.round(q, 2))); y.push(k);
    }
    return { X, y };
  };

  S.topic({
    id: 'perceptron', family: 'p03-perc', order: 10, title: 'The perceptron, one update at a time',
    blurb: 'The perceptron predicts the sign of w·x + b. It goes through the points in order and leaves the weights alone when a point is on its correct side. On a mistake it adds the point to the weights, w ← w + η·t·x and b ← b + η·t, with t = +1 or −1 the true label: the line turns toward getting that point right. If a straight line can separate the classes, the mistakes stop after a finite number of updates; if not, they never stop.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'data', def: 'easy', options: DATA },
      { id: 'n', kind: 'number', label: 'points', def: 24, min: 4, max: 60 },
      { id: 'eta', kind: 'slider', label: 'step η', def: 1, min: 0.1, max: 2, step: 0.1 },
      { id: 'epochs', kind: 'number', label: 'passes at most', def: 10, min: 1, max: 30 },
      { id: 'seed', kind: 'number', label: 'seed', def: 6, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Well apart: a few updates', values: { data: 'easy', n: 24, eta: 1, epochs: 10, seed: 6 } },
      { label: 'A narrow gap: more updates', values: { data: 'narrow', n: 30, eta: 1, epochs: 20, seed: 4 } },
      { label: 'Overlapping: it never settles', values: { data: 'overlap', n: 24, eta: 1, epochs: 6, seed: 2 } },
      { label: 'A small step changes little', values: { data: 'easy', n: 24, eta: 0.2, epochs: 10, seed: 6 } },
    ],
    counters: [{ id: 'upd', label: 'updates' }, { id: 'pass', label: 'pass' }],
    run(inp) {
      const { X, y } = H.percData(inp.data, inp.n, inp.seed), t = y.map((v) => (v ? 1 : -1));
      const m = ML().Perceptron({ eta: inp.eta, epochs: inp.epochs }).fit(X, y), tr = m.trace;
      const lim = [-3.2, 3.2];
      const ax = H.axes({ prefix: 'pc', x: 46, y: 40, w: 420, h: 420, xr: lim, yr: lim, xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], xname: 'x₁', yname: 'x₂', grid: true });
      const F = new S.Frames(['upd', 'pass']);
      const line = (w, b, key, cls) => { const seg = H.lineInBox(w[0], w[1], b, lim, lim); return seg ? H.seg(key, ax.X(seg[0][0]), ax.Y(seg[0][1]), ax.X(seg[1][0]), ax.Y(seg[1][1]), 'pl-cut ' + cls) : null; };
      // the weight vector, scaled to fit when long, from the origin
      const wArrow = (w, key, cls) => { const L = Math.hypot(w[0], w[1]); if (L < 1e-9) return null; const s = L > 2.8 ? 2.8 / L : 1; return H.arrow(key, ax.X(0), ax.Y(0), ax.X(w[0] * s), ax.Y(w[1] * s), cls, { head: 10 }); };
      const side = (w, b) => X.map((x) => w[0] * x[0] + w[1] * x[1] + b);
      const pts = (w, b, hot) => { const z = side(w, b); return SC().points(ax, 'pc:p', X, y, { r: 5, cls: (i) => (i === hot ? 'act' : (t[i] * z[i] <= 0 ? 'wrong' : '')) }); };
      const eqX = 520, head = () => [...ax.items, H.title('pc:t', 46, 14, 'The line w·x + b = 0')];
      const meta0 = { n: X.length, eta: inp.eta };
      F.add({ items: [...head(), ...pts([0, 0], 0, -1), ...H.eqs('pc:eq', eqX, 60, [['w = (0, 0),  b = 0', ''], ['every point scores 0:', 'muted'], ['all count as mistakes', 'muted']])], say: X.length + ' points, labels t = +1 (orange) and −1 (blue). The weights start at w = (0, 0) and b = 0, so every point scores 0 and counts as a mistake until the first update.', meta: Object.assign({ phase: 'start' }, meta0) });
      let w = [0, 0], b = 0, upd = 0, shown = 0;
      const CAP = 40;
      for (const s of tr) {
        if (s.kind === 'epoch') {
          F.set('pass', s.epoch + 1);
          const z = side(w, b), wrongNow = z.filter((v, i) => t[i] * v <= 0).length;
          const items = [...head(), line(w, b, 'pc:line', 'act'), wArrow(w, 'pc:w', 'act'), ...pts(w, b, -1), ...H.eqs('pc:eq', eqX, 60, [['pass ' + (s.epoch + 1) + ': ' + H.plural(s.mistakes, 'mistake'), s.mistakes ? 'hot' : 'good'], ['w = (' + n$(w[0]) + ', ' + n$(w[1]) + '),  b = ' + n$(b), ''], [wrongNow + ' on the wrong side now', 'muted']])].filter(Boolean);
          const say = s.mistakes ? 'Pass ' + (s.epoch + 1) + ' through the data made ' + H.plural(s.mistakes, 'mistake') + ' (' + upd + ' updates so far). With the current line ' + H.plural(wrongNow, 'point') + ' ' + (wrongNow === 1 ? 'is' : 'are') + ' still on the wrong side, so another pass follows.' + (s.epoch + 1 === inp.epochs ? ' That was the last pass allowed.' : '') : 'Pass ' + (s.epoch + 1) + ' made no mistakes: every point is on its own side, and the perceptron stops after ' + H.plural(upd, 'update') + '.';
          F.add({ items, say, meta: Object.assign({ phase: 'pass', pass: s.epoch + 1, mistakes: s.mistakes, upd, wrongNow, w: w.slice(), b }, meta0) });
          continue;
        }
        if (s.kind !== 'update') continue;
        const old = w.slice(), oldB = b;
        w = s.w.slice(); b = s.b; upd++;
        F.set('upd', upd).set('pass', s.epoch + 1);
        if (shown >= CAP) continue;
        shown++;
        const x = X[s.i], items = [...head()];
        items.push(line(old, oldB, 'pc:old', 'soft'), line(w, b, 'pc:line', 'act'));
        // the update as vectors: old w, plus η t x, gives the new w
        const ow = wArrow(old, 'pc:wold', 'muted'), nw = wArrow(w, 'pc:w', 'act');
        if (ow) items.push(ow);
        if (nw) items.push(nw);
        items.push(...pts(w, b, s.i), SC().ring(ax, 'pc:ring', x[0], x[1], 'hot', 10));
        items.push(...H.eqs('pc:eq', eqX, 60, [['update ' + upd + ' (pass ' + (s.epoch + 1) + ')', 'act'], ['point ' + s.i + ' = (' + n$(x[0]) + ', ' + n$(x[1]) + ')', 'hot'], ['label t = ' + (t[s.i] > 0 ? '+1' : '−1'), 'hot'], ['score w·x + b = ' + n$(s.z), 'muted'], ['w ← w + ' + n$(inp.eta) + '·(' + (t[s.i] > 0 ? '+1' : '−1') + ')·x', ''], ['w = (' + n$(w[0]) + ', ' + n$(w[1]) + ')', 'act'], ['b = ' + n$(b), 'act']]));
        const say = 'Point ' + s.i + ' at (' + n$(x[0]) + ', ' + n$(x[1]) + ') has label ' + (t[s.i] > 0 ? '+1' : '−1') + ' but scores ' + n$(s.z) + (s.z === 0 ? ', on the line' : ', the wrong sign') + ': a mistake. Add ' + n$(inp.eta) + '·(' + (t[s.i] > 0 ? '+1' : '−1') + ') times it: w goes from (' + n$(old[0]) + ', ' + n$(old[1]) + ') to (' + n$(w[0]) + ', ' + n$(w[1]) + ') and b from ' + n$(oldB) + ' to ' + n$(b) + '.';
        F.add({ items: items.filter(Boolean), say, meta: Object.assign({ phase: 'update', upd, i: s.i, x, t: t[s.i], z: s.z, old, oldB, w: w.slice(), b, pass: s.epoch + 1 }, meta0) });
      }
      const z = side(w, b), wrong = z.filter((v, i) => t[i] * v <= 0).length;
      const items = [...head(), line(w, b, 'pc:line', 'act'), wArrow(w, 'pc:w', 'act'), ...pts(w, b, -1), ...H.eqs('pc:eq', eqX, 60, [[H.plural(upd, 'update') + ' in all', 'act'], ['w = (' + n$(w[0]) + ', ' + n$(w[1]) + ')', ''], ['b = ' + n$(b), ''], [wrong ? wrong + ' still wrong' : 'every point right', wrong ? 'hot' : 'good']])].filter(Boolean);
      F.add({ items, say: wrong ? 'Stopped after ' + H.plural(upd, 'update') + ' with ' + H.plural(wrong, 'point') + ' still on the wrong side. ' + (inp.data === 'overlap' ? 'No straight line separates these classes, so the perceptron would keep updating for ever; its line keeps swinging between compromises.' : 'More passes would finish the job: the classes can be separated.') : 'Done: ' + H.plural(upd, 'update') + ' in all, and every point is on its own side. The line found is one of many separating lines, not the widest gap; that is what a support vector machine adds.', meta: Object.assign({ phase: 'end', upd, wrong, w: w.slice(), b }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The perceptron', text: ['predict: sign(w·x + b)', 'mistake when t·(w·x + b) ≤ 0', 'update: w ← w + η t x,  b ← b + η t', 'separable data: a finite number of updates', 'starting from 0, η only scales w and b'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'update' && f.meta.upd === 1 && fr[i + 1] && fr[i + 1].meta.phase === 'update', pick: 'option', ask: 'The next update adds a misclassified point to w. Which way does the new weight vector move?', options: ['toward the point if its label is +1, away if −1', 'always toward the origin', 'always away from the point', 'it does not move'], answer: 'toward the point if its label is +1, away if −1', explain: 'w + t·x leans toward x when t = +1 and away from it when t = −1.' },
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will the perceptron stop making mistakes on this data?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.wrong === 0 ? 'yes' : 'no'), explain: (fr) => (fr[fr.length - 1].meta.wrong === 0 ? 'A separating line exists, so the updates run out.' : 'Not within the passes allowed here.') },
    ],
    quiz: [
      { q: 'What does the perceptron do when a point is classified correctly?', options: ['nothing', 'adds it to w', 'subtracts it from w', 'halves η'], answer: 0, why: 'Only mistakes change the weights.' },
      { q: 'w = (1, 0), b = 0 and the point x = (−2, 1) has label +1. After one update with η = 1, w is…', options: ['(−1, 1)', '(3, −1)', '(1, 0)', '(−2, 1)'], answer: 0, why: 'w·x = −2 < 0, a mistake: w + x = (−1, 1).' },
      { q: 'On data no line can separate, the perceptron…', options: ['keeps updating for ever', 'stops with the best line', 'gives an error', 'finds a curve'], answer: 0, why: 'Some point is always wrong, so there is always another update.' },
      { q: 'Starting from w = 0, how does η affect the lines found?', options: ['not at all: it scales w and b together', 'a bigger η gives a better line', 'a smaller η gives fewer updates', 'it rotates the line'], answer: 0, why: 'Every update adds η times something, so w and b are η times what they would be with η = 1.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
