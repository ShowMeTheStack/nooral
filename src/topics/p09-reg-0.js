// Part 9, the shared experiment of the regularisation topics: a 2–32–32–2 ReLU network trained with Adam, full batch,
// on only 30 noisy points of the two moons, and judged on 400 fresh points from the same moons. Without help it
// memorises the 30 points; dropout, weight decay and early stopping each fight that. Runs are cached per setting, so
// topics that compare settings train each one once.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const G = P.SETS.moons;
  P.REG = { set: 'moons', nTrain: 30, noise: 0.3, seedTrain: 5, nVal: 400, seedVal: 77, steps: 500, evalEvery: 10, lr: 0.02, hidden: [32, 32], snaps: [0, 20, 50, 120, 250, 500] };
  P.regData = (n, seed) => ({ train: P.data('moons', n || P.REG.nTrain, P.REG.noise, seed || P.REG.seedTrain), val: P.data('moons', P.REG.nVal, P.REG.noise, P.REG.seedVal) });
  P.regGrid = { xr: G.xr, yr: G.yr, n: 48, w: 300, h: 200 };
  /** one cached run: o {hidden, dropout, weightDecay, optimizer, n (training points), seed, steps, best: true} */
  P.regRun = function (o) {
    o = Object.assign({ hidden: P.REG.hidden, dropout: 0, weightDecay: 0, optimizer: 'adam', n: P.REG.nTrain, seed: 1, steps: P.REG.steps }, o || {});
    return P.once('reg:' + JSON.stringify(o), () => {
      const d = P.regData(o.n), snaps = P.REG.snaps.filter((s) => s <= o.steps);
      let best = null;
      const bests = [];
      const r = P.train({ train: d.train, val: d.val, hidden: o.hidden, act: 'relu', dropout: o.dropout, lr: P.REG.lr, optimizer: o.optimizer, weightDecay: o.weightDecay, steps: o.steps, evalEvery: P.REG.evalEvery, seed: o.seed, snaps, grid: P.regGrid,
        onEval: (e, net) => {
          let s2 = 0;
          for (const l of net.layers) for (const v of l.W.values()) s2 += v * v;
          e.wnorm = Math.sqrt(s2);
          if (o.best && (!best || e.valLoss < best.valLoss)) { best = Object.assign({}, e, { field: P.fieldOf(net, P.regGrid), state: net.stateDict() }); bests.push(best); }
        } });
      r.best = best; r.bests = bests;
      r.o = o; r.data = d;
      return r;
    });
  };
  /** a boundary panel: the field, the 30 training points (filled) and the fresh points (small, faint) */
  P.regPanel = function (prefix, box, field, data, o) {
    o = o || {};
    const ax = P.ax(Object.assign({ prefix, xr: G.xr, yr: G.yr, xticks: o.ticks === false ? 0 : [-1, 0, 1, 2], yticks: o.ticks === false ? 0 : [-1, 0, 1], frame: 'box' }, box));
    const items = [...ax.items, ...P.boundary(ax, prefix + ':b', field)];
    const inside = (X) => X.map((p, i) => i).filter((i) => ax.inside(X[i][0], X[i][1]));
    if (o.val !== false) items.push(...S.Kit.scatter.points(ax, prefix + ':v', data.val.X, data.val.y, { r: 1.8, cls: () => 'faded', only: inside(data.val.X) }));
    items.push(...S.Kit.scatter.points(ax, prefix + ':t', data.train.X, data.train.y, { r: 4.4, only: inside(data.train.X) }));
    if (o.title) items.push(H().text(prefix + ':h', box.x, box.y - 12, o.title, 'strong', 14));
    return items;
  };
  /** loss curves (training and validation) up to a step, on log axes of the loss */
  P.regCurves = function (prefix, box, evals, upto, o) {
    o = o || {};
    const lg = (v) => Math.log10(Math.max(1e-4, v)), steps = o.steps || P.REG.steps;
    const ax = P.ax(Object.assign({ prefix, xr: [0, steps], yr: [-3, 1], xticks: Array.from({ length: 6 }, (_, k) => Math.round((steps * k) / 5)), yticks: [-3, -2, -1, 0, 1], fmtY: (v) => '10' + ['⁻³', '⁻²', '⁻¹', '⁰', '¹'][v + 3], xname: 'step', yname: o.yname || 'loss (log scale)' }, box));
    const use = evals.filter((e) => e.step <= upto);
    const items = [...ax.items, C().line(ax, prefix + ':tr', use.map((e) => [e.step, lg(e.trainLoss)]), 'p09train'), C().line(ax, prefix + ':va', use.map((e) => [e.step, lg(e.valLoss)]), 'p09val')];
    if (o.extra) for (const [k, ev, cls] of o.extra) items.push(C().line(ax, prefix + ':x' + k, ev.filter((e) => e.step <= upto).map((e) => [e.step, lg(e.valLoss)]), cls));
    if (o.mark !== undefined && o.mark !== null) items.push(C().vline(ax, prefix + ':mk', o.mark, 'act'));
    if (o.legend !== false) items.push(...C().legend(prefix + ':lg', box.x, box.y + box.h + 54, [{ text: 'training loss', cls: 'p09train' }, { text: 'loss on fresh points', cls: 'p09val' }].concat(o.legendExtra || []), { size: 13 }));
    return { items, ax, lg };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
