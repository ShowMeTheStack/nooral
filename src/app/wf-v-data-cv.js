// The viewer of Cross-validation: the folds as a grid (in round i the i-th fold is held out, the rest train), the score
// of every round beside it, the mean with the spread as a band, and the rows coloured by the fold that tested them.
(function (S) {
  'use strict';
  const V = S.WFV, r2 = (v) => Math.round(v * 100) / 100;

  V.register('cv', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, k = view.k, narrow = ctx.narrow;
    el.appendChild(V.facts([['folds', k], ['rows', view.n], [view.metricLabel + ' (mean)', V.num(view.mean, 3)], ['spread (sd)', V.num(view.sd, 2)], ['model', view.model]]));
    el.appendChild(V.title('Each round trains on the grey folds and tests on the coloured one'));
    const items = [], lw = narrow ? 64 : 84, sw = narrow ? 84 : 120, gw = W - 24 - lw - sw, rh = Math.min(30, Math.max(18, 260 / k)), gap = 6;
    const sizes = view.perFold.map((f) => f.test), tot = sizes.reduce((a, b) => a + b, 0);
    view.perFold.forEach((f, i) => {
      const y = 26 + i * (rh + gap);
      items.push(V.text('cv:r' + i, 12 + lw - 10, y + rh / 2 + 4.5, 'round ' + (i + 1), { anchor: 'end', size: 13 }));
      let x = 12 + lw;
      sizes.forEach((s, j) => {
        const w = (gw * s) / tot;
        items.push({ key: 'cv:c' + i + '_' + j, kind: 'cell', x: r2(x), y: r2(y), w: r2(Math.max(1, w - 2)), h: r2(rh), rx: 3, cls: 'wfd-fold ' + (i === j ? 'test' : 'train') });
        x += w;
      });
      items.push(V.text('cv:s' + i, 12 + lw + gw + 12, y + rh / 2 + 4.5, V.num(f.score, 3), { size: 13, cls: 'wfd-foldscore' }));
    });
    let H = 26 + k * (rh + gap) + 6;
    items.push(V.text('cv:sh', 12 + lw + gw + 12, 14, V.fit(view.metricLabel, sw - 12, 12), { size: 12, cls: 'tick' }));
    // the scores on one axis: a dot per round, the mean, and mean ± sd as a band
    const lo = Math.min(...view.perFold.map((f) => f.score), view.mean - view.sd), hi = Math.max(...view.perFold.map((f) => f.score), view.mean + view.sd);
    const pad = Math.max(0.01, (hi - lo) * 0.25), box = { x: 12 + lw, y: H + 50, w: gw, h: 36 };
    const ax = S.Kit.curve.axes({ prefix: 'cvs', x: box.x, y: box.y, w: box.w, h: box.h, xr: [lo - pad, hi + pad], yr: [0, 1], xticks: S.Kit.curve.ticks(lo - pad, hi + pad, Math.max(3, Math.round(gw / 110))), yticks: 0, grid: false });
    items.push(...ax.items);
    items.push(V.text('cvs:t', 12, box.y - 34, V.fit('The ' + k + ' scores, their mean and one sd either side', W - 24, 13), { size: 13, bold: true }));
    items.push({ key: 'cvs:band', kind: 'cell', x: r2(ax.X(view.mean - view.sd)), y: box.y + 4, w: r2(Math.max(2, ax.X(view.mean + view.sd) - ax.X(view.mean - view.sd))), h: box.h - 8, rx: 4, cls: 'bar soft', audit: 'ignore', layer: 'back', z: 4 });
    items.push({ key: 'cvs:m', kind: 'path', d: `M${ax.X(view.mean)},${box.y} V${box.y + box.h}`, cls: 'series act', audit: 'ignore', layer: 'lines' });
    // equal scores are stacked, so every round's dot shows
    const seen = new Map();
    view.perFold.forEach((f, i) => { const kx = Math.round(ax.X(f.score)), c = seen.get(kx) || 0; seen.set(kx, c + 1); items.push({ key: 'cvs:d' + i, kind: 'dot', x: ax.X(f.score), y: r2(box.y + box.h / 2 + (c % 2 ? 1 : -1) * Math.ceil(c / 2) * 9), r: 5, cls: 'mark c0', audit: 'overlap-ok', layer: 'top' }); });
    items.push(V.text('cvs:mv', ax.X(view.mean), box.y - 12, 'mean ' + V.num(view.mean, 3), { anchor: 'middle', size: 12, cls: 'strong' }));
    H = box.y + box.h + 32;
    V.stepper(el, [{ items, view: V.view(W, H) }], { label: 'The folds and their scores' });
    if (view.task === 'classification' && view.counts.length) el.appendChild(V.table(['fold'].concat(view.classes), view.counts.map((c, i) => ['fold ' + (i + 1)].concat(c))));
    if (view.points) {
      el.appendChild(V.title('Which fold tested each row'));
      const box2 = V.plotBox(Math.min(W, 720), narrow ? 280 : 340, 48), it = [], ax2 = V.axes('cvp', view.points.P, box2, view.points.names);
      it.push(...ax2.items, ...V.dots(ax2, 'cvp:p', view.points.P, view.points.fold));
      const lg = V.legend('cvp:lg', 12, box2.y + box2.h + 66, view.perFold.map((_, i) => ({ text: 'fold ' + (i + 1), c: i })), W - 24);
      it.push(...lg.items);
      V.stepper(el, [{ items: it, view: V.view(W, box2.y + box2.h + 66 + lg.h + 4) }], { label: 'Rows by fold' });
    }
    el.appendChild(V.note('Every row was predicted once, by the round that held it out: the out-of-fold predictions give ' + view.metricLabel + ' `' + V.num(view.overall[view.metric], 3) + '` over all `' + view.n + '` rows. A spread of `' + V.num(view.sd, 2) + '` between rounds is how much one train/test split could have misled.'));
  }));
})(globalThis.Nooral = globalThis.Nooral || {});
