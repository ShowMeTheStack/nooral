// Part 17, robustness, fairness and privacy (taught defensively: each weakness on toy data, then its defence): the
// families, the trained models the topics share (cached by their settings), and pieces of picture: a model's
// probability over the plane with its p = 0.5 boundary, histograms on shared axes, and small tables.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'responsible', order, title, blurb });
  fam('p17-robust', 10, 'Brittle models and how to harden them', 'A tiny, deliberate nudge can flip a classifier’s answer. Why that happens, and two defences: training on nudged examples, and voting over noisy copies of the input.');
  fam('p17-shift', 20, 'Distribution shift', 'The world a model meets after training is not the one it learned from: the inputs move, the class shares change, or the rule itself drifts. Seeing each kind, and watching for it once the model is in use.');
  fam('p17-fair', 30, 'Fairness', 'Measuring how a model treats two groups (approval rates, true- and false-positive rates) and two ways to narrow the gaps: reweighing the training data and a cut-off for each group.');
  fam('p17-privacy', 40, 'Privacy and data threats', 'Differential privacy in training (clip each person’s gradient, add noise), what a model can leak about who was in its training set, and how corrupted training labels can be found and removed.');

  const P = (S.P17 = {});
  const H = () => S.Kit.ml;
  const cache = new Map();
  P.once = (k, f) => { if (!cache.has(k)) cache.set(k, f()); return cache.get(k); };
  P.r2 = (v) => Math.round(v * 100) / 100;
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');

  /** the moons classifier of the robustness topics: a small network, trained once per setting */
  P.moonsModel = (seed, eps) => P.once('mm:' + seed + ':' + (eps || 0), () => {
    const c = S.Rob.classifier({ data: 'moons', seed, hidden: 16, steps: 300, n: 200, nTest: 200, adv: eps ? { eps } : undefined });
    c.run(300);
    return c;
  });
  /** fixed axes over the moons, equal scale */
  P.moonAxes = (prefix, box) => S.Kit.curve.axes(Object.assign({ prefix, xr: [-1.6, 2.6], yr: [-1.15, 1.55], xticks: 1, yticks: 1, xname: 'x₁', yname: 'x₂', tickSize: 12 }, box));
  /**
   * a model's chance of class 1 over the plane and its p = 0.5 line, every place asked in one batch:
   * prob(rows) -> [p1 for each row]
   */
  P.probField = (ax, key, prob, n) => {
    const want = [], rec = (u, v) => { want.push([u, v]); return 0; };
    S.Kit.scatter.field(ax, 'dry', rec, { n: n || 56 }); S.Kit.scatter.grid(ax, rec, 50);
    const got = new Map(), vals = prob(want);
    want.forEach((p, i) => got.set(p[0] + ',' + p[1], vals[i]));
    const at = (u, v) => got.get(u + ',' + v);
    return { field: S.Kit.scatter.field(ax, key + ':f', at, { n: n || 56, scheme: 'prob', alpha: 0.55 }), line: S.Kit.scatter.level(ax, key + ':l', S.Kit.scatter.grid(ax, at, 50), 0.5, 'p17edge') };
  };
  /** points coloured by class, as dots */
  P.dots = (ax, key, X, y, o) => {
    o = o || {};
    return X.map((p, i) => ({ key: key + ':' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: o.r ? (typeof o.r === 'function' ? o.r(i) : o.r) : 3.6, cls: 'sp c' + y[i] + ' ' + (o.cls ? o.cls(i) || '' : ''), audit: 'overlap-ok', layer: 'items' }));
  };
  /** a key row: a swatch dot and its text */
  P.key = (key, x, y, cls, text) => [{ key: key + ':s', kind: 'dot', x: x + 6, y: y - 4.5, r: 5.5, cls: 'swatch ' + cls }, H().text(key + ':t', x + 18, y, text, 'legend', 13)];
  /** a histogram of values on axes ax: counts per bin of width w from lo, drawn as bars of class cls */
  P.hist = (ax, key, vals, lo, w, nb, cls, scale) => {
    const c = new Array(nb).fill(0);
    for (const v of vals) { const b = Math.min(nb - 1, Math.max(0, Math.floor((v - lo) / w + 1e-9))); c[b]++; }
    const edges = Array.from({ length: nb + 1 }, (_, i) => lo + i * w);
    return { counts: c, items: S.Kit.dt.hist(ax, key, c.map((x) => x * (scale || 1)), edges, cls) };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
