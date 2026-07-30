// The viewers of the Prepare blocks: what a column looked like before and after (two small histograms with the
// change between them), the numbers each block learned (the fill values, the scaling, the limits), the score of
// every column for Select, the variance each principal component explains, a vocabulary for Vectorise, and the
// leakage note: was the block fitted on the training rows only?
(function (S) {
  'use strict';
  const { h } = S.UI;
  const V = S.WFV, r2 = (v) => Math.round(v * 100) / 100;

  const leakNote = (el, view) => { if (view.leak) el.appendChild(V.note(view.leak.note, view.leak.ok ? 'good' : 'bad')); };
  const head = (el, view, extra) => el.appendChild(V.facts([['rows in', view.n], view.nOut !== view.n ? ['rows out', view.nOut, 'bad'] : null, ['columns', view.featuresBefore.length + ' → ' + view.featuresAfter.length], view.test ? ['test rows done the same way', view.test] : null].concat(extra || [])));

  /** each column as two histograms, before → after, on one row */
  function beforeAfterFrame(cols, W) {
    const items = [], narrow = W < 560, rowH = narrow ? 128 : 92, nameW = narrow ? 0 : Math.min(170, W * 0.2);
    const hw = narrow ? (W - 24 - 34) / 2 : (W - nameW - 24 - 44) / 2;
    items.push(V.text('ba:tb', 12 + nameW, 16, 'before', { cls: 'title', size: 12 }));
    items.push(V.text('ba:ta', 12 + nameW + hw + 44, 16, 'after', { cls: 'title', size: 12 }));
    cols.forEach((c, i) => {
      const y = 30 + i * rowH, top = narrow ? y + 22 : y;
      items.push(V.text('ba:n' + i, 12, narrow ? y + 13 : y + 26, V.fit(c.name, narrow ? W - 24 : nameW - 10, 13), { bold: true }));
      const box = (x) => ({ x, y: top + 6, w: hw, h: 38 });
      const draw = (k, st, x) => {
        if (!st) { items.push(V.text('ba:' + k + 'x' + i, x, top + 30, 'new column', { size: 12, cls: 'tick' })); return; }
        if (st.hist) items.push(...V.hist('ba:' + k + i, box(x), st.hist, st.lo, st.hi, k === 'a' ? 'c0' : 'soft'));
        else if (st.counts) items.push(...V.hist('ba:' + k + i, box(x), st.counts, undefined, undefined, k === 'a' ? 'c0' : 'soft'));
        const sub = st.hist ? 'mean ' + V.num(st.mean) + ' · sd ' + V.num(st.sd) : st.categories ? st.categories.length + ' values' : '';
        items.push(V.text('ba:' + k + 's' + i, x, top + 76, V.fit(sub + (st.missing ? ' · ' + st.missing + ' missing' : ''), hw - 10, 12), { size: 12, cls: st.missing ? 'bad' : 'tick' }));
      };
      const x0 = 12 + nameW;
      draw('b', c.before, x0);
      items.push({ key: 'ba:ar' + i, kind: 'edge', from: { x: x0 + hw + 8, y: top + 26 }, to: { x: x0 + hw + 36, y: top + 26 }, cls: 'act' });
      draw('a', c.after, x0 + hw + 44);
    });
    return { items, view: V.view(W, 34 + cols.length * rowH) };
  }

  V.register(['clean', 'impute', 'scale'], V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, info = view.info || {};
    if (view.kind === 'clean') {
      head(el, view, [info.mode === 'rows' ? ['rows dropped', info.dropped, info.dropped ? 'bad' : ''] : info.mode === 'cols' ? ['columns dropped', info.dropped.length, info.dropped.length ? 'bad' : ''] : ['cells clipped', info.clipped.reduce((a, b) => a + b, 0)]]);
      if (info.mode === 'rows') el.appendChild(V.note(info.dropped ? '`' + info.dropped + '` of `' + view.n + '` rows had a missing cell and were dropped. A model cannot drop a row it is asked about, so a new row with a gap gets the training mean of that column instead.' : 'No row had a missing cell: nothing was dropped.'));
      if (info.mode === 'cols') el.appendChild(info.dropped.length ? V.table(['dropped column', 'share missing'], info.dropped.map((d) => [d.name, V.pct(d.share)])) : V.note('No column was missing more than the limit: every column stays.'));
      if (info.mode === 'clip') el.appendChild(V.table(['column', 'lower limit', 'upper limit', 'cells clipped'], view.featuresBefore.map((n, j) => [n, info.limits[j] ? V.num(info.limits[j][0]) : '—', info.limits[j] ? V.num(info.limits[j][1]) : '—', info.clipped[j]]).filter((r) => r[1] !== '—')));
    } else if (view.kind === 'impute') {
      head(el, view, [['cells filled', info.gaps.reduce((a, b) => a + b, 0)], ['filled with', info.strategy === 'mode' ? 'the most frequent value' : 'the ' + info.strategy]]);
      el.appendChild(V.table(['column', 'missing cells', 'filled with'], view.featuresBefore.map((n, j) => [n, info.gaps[j], V.num(info.fills[j], 4)]), { cell: (i, j) => (j === 1 && info.gaps[i] ? 'miss' : '') }));
      if (info.flagged.length) el.appendChild(V.note('New columns say which cells were filled: `' + info.flagged.map((n) => n + ' was missing').join('`, `') + '`. A model may learn that a gap itself means something.'));
    } else {
      head(el, view, [['scaling', info.how]]);
      el.appendChild(V.table(['column', info.how === 'minmax' ? 'minus (the minimum)' : info.how === 'robust' ? 'minus (the median)' : 'minus (the mean)', info.how === 'minmax' ? 'divided by (the range)' : info.how === 'robust' ? 'divided by (the IQR)' : 'divided by (the sd)'], info.params.map((q) => [q.name, V.num(q.shift, 4), V.num(q.by, 4)])));
    }
    leakNote(el, view);
    if (view.columns && view.columns.length) { el.appendChild(V.title('Before and after', view.columns.length < view.featuresAfter.length ? '(the first ' + view.columns.length + ' columns)' : '')); V.stepper(el, [beforeAfterFrame(view.columns, W)], { label: 'Each column before and after' }); }
  }));

  V.register('encode', V.viewer((view, ctx) => {
    const el = ctx.el, info = view.info;
    head(el, view, [['encoding', info.how === 'onehot' ? 'one-hot' : 'ordinal']]);
    if (!info.columns.length) { el.appendChild(V.note('This table has no category columns: there is nothing to encode.')); return; }
    el.appendChild(V.note(info.how === 'onehot' ? 'Each category column becomes one 0/1 column per value, so no value counts as bigger than another: `' + info.columns.map((c) => c.name + ' → ' + c.categories.length + ' columns').join('`, `') + '`.' : 'Each category is replaced by its position in the list, `' + info.columns.map((c) => c.name + ': ' + c.categories.map((x, i) => x + ' = ' + i).join(', ')).join('`; `') + '`. That invents an order, which only makes sense when the categories have one.'));
    el.appendChild(V.title('Before', '(some rows)'));
    el.appendChild(V.table(view.featuresBefore, view.sampleBefore.map((r) => r.map((x, j) => (view.catsBefore[j] ? view.catsBefore[j][x] : V.num(x, 4))))));
    el.appendChild(V.title('After', view.namesAfter.length < view.featuresAfter.length ? '(the first ' + view.namesAfter.length + ' columns)' : ''));
    el.appendChild(V.table(view.namesAfter, view.sampleAfter.map((r) => r.map((x) => V.num(x, 4))), { cell: (i, j) => (view.namesAfter[j].includes('=') ? (view.sampleAfter[i][j] ? 'hi' : 'lo') : '') }));
    leakNote(el, view);
  }));

  V.register('select', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, info = view.info, names = view.names, kept = new Set(info.kept);
    const by = { mi: 'mutual information with the target', corr: 'correlation with the target', variance: 'variance' }[info.by];
    head(el, view, [['kept', info.kept.length]]);
    const order = names.map((_, j) => j).sort((a, b) => info.scores[b] - info.scores[a]).slice(0, 24);
    const nameW = Math.min(180, Math.max(...order.map((j) => S.geom.textWidth(names[j], 13))) + 14), bh = 20, items = [], mx = Math.max(1e-9, ...info.scores);
    order.forEach((j, i) => {
      const y = 30 + i * (bh + 6), bw = Math.max(1, ((W - nameW - 100) * info.scores[j]) / mx);
      items.push(V.text('se:n' + j, 12 + nameW - 8, y + 14.5, V.fit(names[j], nameW - 10), { anchor: 'end', bold: kept.has(j) }));
      items.push({ key: 'se:b' + j, kind: 'cell', x: 12 + nameW, y, w: r2(bw), h: bh, rx: 3, cls: 'bar ' + (kept.has(j) ? 'act' : 'soft'), audit: 'overlap-ok' });
      items.push(V.text('se:v' + j, 12 + nameW + bw + 6, y + 14.5, V.num(info.scores[j], 3) + (kept.has(j) ? '  kept' : ''), { size: 12, cls: kept.has(j) ? 'strong' : 'tick' }));
    });
    for (const it of items) it.y -= 22;
    el.appendChild(V.title('Each column scored by ' + by));
    V.stepper(el, [{ items, view: V.view(W, 18 + order.length * (bh + 6)) }], { label: 'Column scores' });
    leakNote(el, view);
  }));

  V.register('poly', V.viewer((view, ctx) => {
    const el = ctx.el;
    head(el, view, [['degree', view.info.degree]]);
    el.appendChild(V.note('From `' + view.featuresBefore.length + '` columns, degree `' + view.info.degree + '` makes `' + view.info.count + '`: every product of up to ' + view.info.degree + ' of them. A straight boundary in these new columns is a curve in the old ones.'));
    el.appendChild(h('div.wfd-facts', ...view.names.map((n) => h('span.wfd-fact', n)), view.more ? h('span.wfd-fact', '… and ' + view.more + ' more') : null));
    el.appendChild(V.title('Some rows', '(the first ' + Math.min(10, view.names.length) + ' columns)'));
    el.appendChild(V.table(view.names.slice(0, 10), view.sample.map((r) => r.map((x) => V.num(x, 4)))));
  }));

  V.register('pca', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, info = view.info, narrow = ctx.narrow;
    head(el, view, [['components', info.k], ['variance kept', V.pct(info.kept)]]);
    // the share of variance of every component, and the running total
    const items = [], k = info.ratio.length, plot = { x: 48, y: 30, w: (narrow ? W : Math.min(W, 640)) - 66, h: 170 };
    const ax = S.Kit.curve.axes({ prefix: 'pv', x: plot.x, y: plot.y, w: plot.w, h: plot.h, xr: [0.4, k + 0.6], yr: [0, 1], xticks: k <= 12 ? Array.from({ length: k }, (_, i) => i + 1) : S.Kit.curve.ticks(1, k, 6), yticks: [0, 0.25, 0.5, 0.75, 1], xname: 'component', yname: 'share of the variance', fmtY: (v) => Math.round(v * 100) + '%' });
    items.push(...ax.items);
    const bw = Math.min(40, (plot.w / k) * 0.7);
    let cum = 0;
    const cumPts = [];
    info.ratio.forEach((r, i) => {
      cum += r; cumPts.push([i + 1, cum]);
      items.push({ key: 'pv:b' + i, kind: 'cell', x: r2(ax.X(i + 1) - bw / 2), y: r2(ax.Y(r)), w: r2(bw), h: r2(Math.max(1, ax.Y(0) - ax.Y(r))), rx: 2, cls: 'bar ' + (i < info.k ? 'act' : 'soft'), audit: 'overlap-ok' });
    });
    items.push(S.Kit.curve.line(ax, 'pv:cum', cumPts, 'c0'));
    cumPts.forEach((p, i) => items.push(S.Kit.curve.dot(ax, 'pv:cd' + i, p[0], p[1], 'c0', 3.5)));
    const lg = V.legend('pv:lg', plot.x, plot.y + plot.h + 50, [{ text: 'kept components', c: 0 }, { text: 'running total', c: 0 }], W - 60);
    lg.items[0].cls = 'swatch'; lg.items[0].paint = { fill: 'var(--accent)' };
    items.push(...lg.items);
    V.stepper(el, [{ items, view: V.view(W, plot.y + plot.h + 50 + lg.h + 6) }], { label: 'Variance explained' });
    // the components as a heat map: how much each column weighs in each kept component
    if (info.components.length && info.names.length) {
      el.appendChild(V.title('What each component is made of', '(blue: the column pushes the component up, vermilion: down)'));
      const nc = info.names.length, lw = Math.min(150, Math.max(...info.names.map((n) => S.geom.textWidth(n, 13))) + 12), cw = Math.max(34, Math.min(56, (W - lw - 30) / info.components.length)), ch = 24;
      const mx = Math.max(1e-9, ...info.components.flat().map(Math.abs));
      const g = S.Kit.heat.grid({ prefix: 'pc', x: 12 + lw, y: 30, cw, ch, vals: info.names.map((_, j) => info.components.map((c) => c[j])), scheme: 'div', max: mx, rows: info.names.map((n) => V.fit(n, lw - 10)), cols: info.components.map((_, i) => 'PC' + (i + 1)), size: 12 });
      V.stepper(el, [{ items: g.items, view: V.view(W, 40 + nc * (ch + 2)) }], { label: 'The components' });
    }
    if (view.points) {
      el.appendChild(V.title('The rows on the first two components'));
      const box = V.plotBox(Math.min(W, 720), narrow ? 280 : 340, 48), it = [], ax2 = V.axes('pp', view.points.P, box, ['PC1', 'PC2']);
      it.push(...ax2.items, ...V.dots(ax2, 'pp:p', view.points.P, view.task === 'classification' ? view.points.y : null));
      V.stepper(el, [{ items: it, view: V.view(W, box.y + box.h + 46) }], { label: 'Rows on the components' });
    }
    leakNote(el, view);
  }));

  V.register('vectorise', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W, info = view.info;
    head(el, view, [['vocabulary', info.size + ' words'], ['cells hold', { count: 'counts', binary: '0 or 1', tfidf: 'TF-IDF weights' }[info.weight]]]);
    const top = info.vocab.slice(0, 20), items = [], lw = Math.min(150, Math.max(...top.map((w) => S.geom.textWidth(w, 13))) + 12), bh = 18, mx = Math.max(...info.counts.slice(0, 20));
    top.forEach((w, i) => {
      const y = 30 + i * (bh + 5), bw = Math.max(1, ((W - lw - 150) * info.counts[i]) / mx);
      items.push(V.text('vo:n' + i, 12 + lw - 8, y + 13.5, V.fit(w, lw - 10), { anchor: 'end' }));
      items.push({ key: 'vo:b' + i, kind: 'cell', x: 12 + lw, y, w: r2(bw), h: bh, rx: 3, cls: 'bar soft', audit: 'overlap-ok' });
      items.push(V.text('vo:v' + i, 12 + lw + bw + 6, y + 13.5, info.counts[i] + ' times · ' + info.df[i] + ' texts', { size: 12, cls: 'tick' }));
    });
    for (const it of items) it.y -= 22;
    el.appendChild(V.title('The most frequent words, and in how many texts each appears'));
    V.stepper(el, [{ items, view: V.view(W, 18 + top.length * (bh + 5)) }], { label: 'The vocabulary' });
    el.appendChild(V.title('Texts as rows', '(the first ' + view.cols.length + ' word columns)'));
    el.appendChild(V.table(['text', 'class'].concat(view.cols), view.texts.map((t, i) => [V.fit(t, 220, 13), V.className(view, view.rowsY[i])].concat(view.cells[i].map((x) => (x ? V.num(x, 3) : '·')))), { cell: (i, j) => (j >= 2 && view.cells[i][j - 2] ? 'hi' : j >= 2 ? 'lo' : '') }));
    leakNote(el, view);
  }));
})(globalThis.Nooral = globalThis.Nooral || {});
