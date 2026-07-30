// The viewers of the Data blocks (a built-in set, a CSV file, an image folder): the set's story, the class balance,
// the rows as a scatter (on the two features, or the first two principal components), a small histogram of every
// column, some rows as a table; pictures as a grid, texts as a list.
(function (S) {
  'use strict';
  const { h } = S.UI;
  const V = S.WFV, r2 = (v) => Math.round(v * 100) / 100;

  /** the class balance as bars (or the target's histogram for a number), in a box */
  V.balance = function (key, view, box) {
    const items = [];
    if (view.task === 'classification' && view.counts) {
      const n = view.counts.reduce((a, b) => a + b, 0) || 1, mx = Math.max(...view.counts), rows = view.counts.length;
      const bh = Math.min(26, Math.max(14, (box.h - 26) / rows - 6)), lw = Math.min(box.w * 0.42, Math.max(...view.classes.map((c) => S.geom.textWidth(String(c), 13))) + 10);
      items.push(V.text(key + ':t', box.x, box.y + 12, 'Class balance', { cls: 'title', size: 12 }));
      view.counts.forEach((c, i) => {
        const y = box.y + 26 + i * (bh + 6), bw = Math.max(2, ((box.w - lw - 92) * c) / mx);
        items.push(V.text(key + ':n' + i, box.x + lw - 8, y + bh / 2 + 4.5, V.fit(view.classes[i], lw - 8), { anchor: 'end' }));
        items.push({ key: key + ':b' + i, kind: 'cell', x: r2(box.x + lw), y: r2(y), w: r2(bw), h: r2(bh), rx: 3, cls: 'bar c' + (i % 8) + ' wfd-cbar', audit: 'overlap-ok' });
        items.push(V.text(key + ':v' + i, box.x + lw + bw + 6, y + bh / 2 + 4.5, c + ' (' + V.num((100 * c) / n, 3) + '%)', { size: 12, cls: 'tick' }));
      });
      return { items, h: 26 + rows * (bh + 6) };
    }
    if (view.targetStats && view.targetStats.hist) {
      const t = view.targetStats;
      items.push(V.text(key + ':t', box.x, box.y + 12, 'The target: ' + V.fit(view.target ? view.target.name : 'y', box.w - 110, 12), { cls: 'title', size: 12 }));
      items.push(...V.hist(key + ':h', { x: box.x, y: box.y + 26, w: box.w, h: Math.min(110, box.h - 50) }, t.hist, t.lo, t.hi, 'c0'));
      return { items, h: Math.min(110, box.h - 50) + 50 };
    }
    return { items, h: 0 };
  };

  /** the scatter of the rows coloured by class (wide: beside the balance; narrow: above it) */
  function scatterFrame(view, W) {
    const pts = view.points, narrow = W < 640, items = [];
    const leftW = narrow ? W : Math.round(W * 0.6);
    const box = V.plotBox(leftW, narrow ? 300 : 360, 48);
    const ax = V.axes('dv', pts.P, box, pts.names);
    items.push(...ax.items);
    const lab = view.task === 'classification' ? pts.y : null;
    items.push(...V.dots(ax, 'dv:p', pts.P, lab));
    let H = box.y + box.h + 44;
    const bb = narrow ? { x: 12, y: H, w: W - 24, h: 200 } : { x: leftW + 24, y: 30, w: W - leftW - 40, h: box.h };
    const bal = V.balance('dv:bal', view, bb);
    items.push(...bal.items);
    H = narrow ? H + bal.h + 14 : Math.max(H, bb.y + bal.h + 14);
    return { items, view: V.view(W, H), say: pts.how && pts.how.startsWith('the first') ? 'Each dot is a row, coloured by its class, placed by ' + pts.how + '.' : '' };
  }

  /** a small card per column: its name, type, and a histogram (numbers) or bars (categories) */
  function columnsFrame(cols, W) {
    const per = W < 420 ? 1 : W < 760 ? 2 : W < 1100 ? 3 : 4, gap = 16, cw = (W - 24 - gap * (per - 1)) / per, ch = 124, items = [];
    cols.forEach((c, i) => {
      const x = 12 + (i % per) * (cw + gap), y = 10 + Math.floor(i / per) * (ch + gap);
      items.push({ key: 'dc:bg' + i, kind: 'cell', x: r2(x), y: r2(y), w: r2(cw), h: ch, rx: 8, cls: 'panel', audit: 'group' });
      items.push(V.text('dc:n' + i, x + 10, y + 20, V.fit(c.name, cw - 90, 13), { bold: true }));
      items.push(V.text('dc:t' + i, x + cw - 10, y + 20, c.type === 'category' ? 'category' : 'number', { anchor: 'end', size: 12, cls: 'tick' }));
      const hb = { x: x + 12, y: y + 32, w: cw - 24, h: 40 };
      if (c.hist) items.push(...V.hist('dc:h' + i, hb, c.hist, c.lo, c.hi, 'soft'));
      else if (c.counts) {
        const mx = Math.max(1, ...c.counts), bw = hb.w / c.counts.length;
        c.counts.forEach((k, j) => { if (!k) return; const hh = Math.max(1.5, (k / mx) * hb.h); items.push({ key: 'dc:k' + i + '_' + j, kind: 'cell', x: r2(hb.x + j * bw + 1), y: r2(hb.y + hb.h - hh), w: r2(Math.max(1, bw - 2)), h: r2(hh), rx: 1, cls: 'bar soft', audit: 'overlap-ok' }); });
        if (c.counts.length <= 5) c.categories.forEach((nm, j) => items.push(V.text('dc:kl' + i + '_' + j, hb.x + (j + 0.5) * bw, hb.y + hb.h + 15, V.fit(nm, bw - 4, 12), { size: 12, cls: 'tick', anchor: 'middle' })));
        else items.push(V.text('dc:kl' + i, hb.x, hb.y + hb.h + 15, V.fit(c.categories.slice(0, 4).join(', ') + ', …', hb.w, 12), { size: 12, cls: 'tick' }));
      }
      const sub = c.hist ? 'mean ' + V.num(c.mean) + ' · sd ' + V.num(c.sd) : c.counts ? c.categories.length + (c.more ? '+' : '') + ' values' : '';
      items.push(V.text('dc:s' + i, x + 10, y + ch - 10, V.fit(sub + (c.missing ? ' · ' + c.missing + ' missing' : ''), cw - 20, 12), { size: 12, cls: c.missing ? 'bad' : 'tick' }));
    });
    const rows = Math.ceil(cols.length / per);
    return { items, view: V.view(W, 20 + rows * (ch + gap)) };
  }

  /** a grid of small pictures with their classes */
  function imagesFrame(view, W) {
    const n = view.images.length, cell = W < 500 ? 52 : 64, gap = 10, per = Math.max(3, Math.floor((W - 20 + gap) / (cell + gap))), items = [];
    view.images.forEach((px, i) => {
      const x = 10 + (i % per) * (cell + gap), y = 8 + Math.floor(i / per) * (cell + 30);
      items.push({ key: 'di:r' + i, kind: 'raster', x, y, w: cell, h: cell, nx: view.w, ny: view.h, vals: px.map((v) => v / 15), scheme: 'ink', smooth: false, cls: 'feature' });
      items.push(V.text('di:t' + i, x + cell / 2, y + cell + 15, V.fit(V.className(view, view.imageY[i]), cell + gap - 2, 12), { anchor: 'middle', size: 12, cls: 'tick' }));
    });
    return { items, view: V.view(W, 16 + Math.ceil(n / per) * (cell + 30)) };
  }

  const dataViewer = V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W;
    if (view.doc) el.appendChild(h('div.wfd-story', V.title(view.doc.title || view.name), view.doc.story ? h('p', view.doc.story) : null, view.doc.rule ? h('p.muted.wfd-small', 'How it was made: ' + view.doc.rule) : null, view.doc.licence ? h('p.muted.wfd-small', 'Source: ' + (view.doc.source || '') + ' · licence: ' + view.doc.licence) : null));
    else el.appendChild(V.title(view.name, view.file ? '(your own file)' : ''));
    el.appendChild(V.facts([[view.kind === 'images' ? 'pictures' : view.kind === 'text' ? 'texts' : 'rows', view.n], view.features ? ['features', view.features.length] : view.kind === 'images' ? ['size', view.w + ' × ' + view.h] : null,
      ['task', view.task || 'none'], view.classes ? ['classes', view.classes.length] : null, view.missing ? ['missing cells', view.missing, 'bad'] : null, view.noTarget ? ['rows with no target, left out', view.noTarget, 'bad'] : null]));
    if (view.dropped && view.dropped.length) el.appendChild(V.note('Columns of free text were left out: `' + view.dropped.join(', ') + '`.'));
    for (const w of view.warnings || []) el.appendChild(V.note(w, 'bad'));
    if (view.kind === 'table') {
      if (view.points) V.stepper(el, [scatterFrame(view, W)], { label: 'The rows as points' });
      else { const b = V.balance('dv:bal', view, { x: 12, y: 8, w: W - 24, h: 220 }); if (b.items.length) V.stepper(el, [{ items: b.items, view: V.view(W, b.h + 20) }], { label: 'The class balance' }); }
      el.appendChild(V.title('The columns', view.columns.length < view.features.length ? '(the first ' + view.columns.length + ' of ' + view.features.length + ')' : ''));
      V.stepper(el, [columnsFrame(view.columns, W)], { label: 'Each column' });
      const fn = view.features.map((f) => f.name);
      el.appendChild(V.title('Some rows', '(' + view.rows.length + ' of ' + view.n + ')'));
      el.appendChild(V.table(['#'].concat(fn, [view.target ? view.target.name + ' (target)' : 'target']), view.rows.map((r, i) => [view.rowIds[i]].concat(r.map((x, j) => (x === null ? '—' : view.features[j].categories && view.features[j].type === 'category' ? view.features[j].categories[x] || x : V.num(x, 4))), [view.task === 'classification' ? V.className(view, view.rowY[i]) : V.num(view.rowY[i], 4)])),
        { cell: (i, j) => (view.rows[i][j - 1] === null ? 'miss' : '') }));
    } else if (view.kind === 'images') {
      const b = V.balance('dv:bal', view, { x: 12, y: 8, w: Math.min(W - 24, 560), h: 240 });
      V.stepper(el, [{ items: b.items, view: V.view(W, b.h + 20) }], { label: 'The class balance' });
      el.appendChild(V.title('Some pictures', '(' + view.images.length + ' of ' + view.n + ')'));
      V.stepper(el, [imagesFrame(view, W)], { label: 'Pictures with their classes' });
    } else if (view.kind === 'text') {
      const b = V.balance('dv:bal', view, { x: 12, y: 8, w: Math.min(W - 24, 560), h: 200 });
      V.stepper(el, [{ items: b.items, view: V.view(W, b.h + 20) }], { label: 'The class balance' });
      el.appendChild(V.facts([['words per text', view.words.mean], ['shortest', view.words.min], ['longest', view.words.max]]));
      el.appendChild(h('ul.wfd-texts', ...view.texts.map((t, i) => h('li', h('span.wfd-pill.c' + (view.textY[i] % 8), V.className(view, view.textY[i])), ' ', t))));
    }
    if (view.trace && view.trace.length) el.appendChild(h('details.wfd-more', h('summary', 'How the file was read'), h('ul', ...view.trace.map((t) => h('li', t)))));
  });
  V.register(['data', 'csv', 'imagefolder'], dataViewer);

  // ---- the split ---------------------------------------------------------------------------------------------------
  V.register('split', V.viewer((view, ctx) => {
    const el = ctx.el, W = ctx.W;
    el.appendChild(V.facts([['rows', view.n], ['train', view.train], ['test', view.test], ['test share', V.pct(view.test / view.n)], view.task === 'classification' ? ['stratified', view.stratified ? 'yes' : 'no'] : null]));
    const items = [];
    let H = 10;
    // one bar of rows: training on the left, test on the right, split by class
    const bar = { x: 12, y: 30, w: W - 24, h: 26 };
    items.push(V.text('sp:t', bar.x, 18, 'Every row goes to one side', { cls: 'title', size: 12 }));
    const tw = (bar.w * view.train) / view.n;
    const seg = (key, x, w, counts, side) => {
      const tot = counts.reduce((a, b) => a + b, 0) || 1;
      let cx = x;
      counts.forEach((c, i) => { const ww = (w * c) / tot; if (ww > 0.5) items.push({ key: key + i, kind: 'cell', x: r2(cx), y: bar.y, w: r2(Math.max(1, ww - 1)), h: bar.h, rx: 2, cls: 'bar c' + (i % 8) + (side === 'test' ? ' wfd-hatch' : ''), audit: 'overlap-ok' }); cx += ww; });
    };
    if (view.task === 'classification') { seg('sp:a', bar.x, tw - 3, view.countsTrain, 'train'); seg('sp:b', bar.x + tw + 3, bar.w - tw - 3, view.countsTest, 'test'); }
    else { items.push({ key: 'sp:a', kind: 'cell', x: bar.x, y: bar.y, w: r2(tw - 3), h: bar.h, rx: 2, cls: 'bar soft' }); items.push({ key: 'sp:b', kind: 'cell', x: r2(bar.x + tw + 3), y: bar.y, w: r2(bar.w - tw - 3), h: bar.h, rx: 2, cls: 'bar c1' }); }
    items.push(V.text('sp:lt', bar.x, bar.y + bar.h + 17, 'train: ' + view.train, { size: 12, cls: 'tick' }));
    items.push(V.text('sp:le', bar.x + bar.w, bar.y + bar.h + 17, 'test: ' + view.test, { size: 12, cls: 'tick', anchor: 'end' }));
    H = bar.y + bar.h + 30;
    if (view.task === 'classification') {
      // class shares on each side: stratification keeps them equal
      const rows = view.classes.map((c, i) => [c, view.countsTrain[i], view.countsTest[i]]);
      const tt = view.countsTrain.reduce((a, b) => a + b, 0) || 1, te = view.countsTest.reduce((a, b) => a + b, 0) || 1;
      el.appendChild(V.table(['class', 'train', 'share', 'test', 'share'], rows.map(([c, a, b]) => [c, a, V.pct(a / tt), b, V.pct(b / te)])));
    }
    if (view.points) {
      const box = V.plotBox(Math.min(W, 760), ctx.narrow ? 300 : 360, 48);
      box.y = H + 24;
      const ax = V.axes('sp:ax', view.points.P, box, view.points.names);
      items.push(...ax.items);
      items.push(...V.dots(ax, 'sp:p', view.points.P, view.task === 'classification' ? view.points.y : null, { cls: (i) => (view.points.test[i] ? 'test' : '') }));
      const lg = V.legend('sp:lg', 12, box.y + box.h + 40, [{ text: 'filled: training row', c: 0 }, { text: 'hollow ring: test row', c: 0 }], W - 24);
      lg.items[2].cls += ' wfd-ring';
      items.push(...lg.items);
      H = box.y + box.h + 40 + lg.h;
    }
    V.stepper(el, [{ items, view: V.view(W, H + 8) }], { label: 'The split' });
    el.appendChild(V.note('The test rows are put away now and used once, at the end, to judge the model: a score on rows the model learned from says little about new rows.'));
  }));
})(globalThis.Nooral = globalThis.Nooral || {});
