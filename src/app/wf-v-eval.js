// The Evaluate block's viewer. For classes: the confusion matrix (counts, or shares of each true class or of each
// prediction), right answers in green on the diagonal and mistakes in vermilion off it; click a cell and the rows in it
// are listed (their values, images or texts, with what the model said); beside it the scores, each worked out from the
// counts. For numbers: predicted against true with the line of perfect answers, and the errors' spread.
(function (S) {
  'use strict';
  const WF = S.WF, VK = WF.VK, h = (...a) => S.UI.h(...a);
  const C = () => S.Kit.curve, HM = () => S.Kit.heat, tw = (t, s) => S.geom.textWidth(String(t), s || 13);

  /** the confusion matrix as stage items: {items, W, H}; sel = [i, j] lights a cell */
  WF.VK.confusion = function (o) {
    const K = o.classes.length, counts = o.counts, norm = o.normalize || 'counts';
    const rows = counts.map((r) => r.reduce((a, b) => a + b, 0)), cols = o.classes.map((_, j) => counts.reduce((s, r) => s + r[j], 0));
    const vals = counts.map((r, i) => r.map((c, j) => (norm === 'true' ? (rows[i] ? c / rows[i] : 0) : norm === 'pred' ? (cols[j] ? c / cols[j] : 0) : c)));
    const short = (s) => VK.short(s, K > 6 ? 6 : 13);
    const left = Math.min(150, Math.max(56, 18 + Math.max(...o.classes.map((c) => tw(short(c), 13))))), top = 52;
    const cw = Math.max(24, Math.min(o.cell || 76, Math.floor((o.W - left - 12) / K) - 2)), ch = Math.max(26, Math.min(cw, 60));
    const colName = (s) => (tw(short(s), 13) > cw + 6 ? VK.short(s, Math.max(2, Math.floor(cw / 8))) : short(s));
    const fmt = norm === 'counts' ? (v) => (v === 0 && K > 3 ? '' : String(v)) : (v) => (v === 0 && K > 3 ? '' : v >= 0.995 ? '1' : String(+v.toFixed(2)).replace(/^0\./, '.'));
    const g = HM().grid({ prefix: o.prefix || 'cm', x: left, y: top, cw, ch, vals, scheme: 'seq', max: 1, fmt, rows: o.classes.map(short), cols: o.classes.map(colName), pick: !!o.pick,
      cls: (i, j) => (o.sel && o.sel[0] === i && o.sel[1] === j ? 'sel' : ''), rowCls: (i) => (o.sel && o.sel[0] === i ? 'act' : ''), colCls: (j) => (o.sel && o.sel[1] === j ? 'act' : '') });
    let dmax = 0, omax = 0;
    vals.forEach((r, i) => r.forEach((v, j) => { if (i === j) dmax = Math.max(dmax, v); else omax = Math.max(omax, v); }));
    for (const it of g.items) {
      const m = /:c(\d+)_(\d+)$/.exec(it.key);
      if (!m) continue;
      const i = +m[1], j = +m[2], v = vals[i][j];
      it.paint = HM().colour(v, i === j ? 'good' : 'bad', i === j ? dmax || 1 : omax || 1);
      it.title = 'true ' + o.classes[i] + ', predicted ' + o.classes[j] + ': ' + counts[i][j] + ' row' + (counts[i][j] === 1 ? '' : 's');
    }
    g.items.push({ key: (o.prefix || 'cm') + ':pn', kind: 'text', x: left, y: 16, text: 'predicted →', size: 13, cls: 'axname' });
    g.items.push({ key: (o.prefix || 'cm') + ':tn', kind: 'text', x: left - 8, y: 16, text: 'true ↓', size: 13, anchor: 'end', cls: 'axname' });
    return { items: g.items, W: left + g.w + 12, H: top + g.h + 10, vals };
  };

  function classification(box, v, ctx) {
    const st = ctx.state, wide = !ctx.narrow && ctx.W >= 700;
    st.norm = st.norm || v.normalize || 'counts';
    // at first the largest cell of mistakes is open, so the rows behind an error are in view at once
    if (st.sel === undefined) { let best = null, n0 = 0; v.counts.forEach((r, i) => r.forEach((c, j) => { if (i !== j && c > n0) { n0 = c; best = [i, j]; } })); st.sel = best; }
    const K = v.classes.length, n = v.n;
    const head = h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.model || 'The model'), ' on ', v.data || 'the test rows', ': ', VK.fmt(v.scores.accuracy * n, 0), ' of ', n, ' right'),
      VK.seg('Show', [['counts', 'counts'], ['true', 'share of each true class'], ['pred', 'share of each prediction']], st.norm, (x) => { st.norm = x; ctx.redraw(); }));
    const fig = VK.fig('Confusion matrix');
    const figW = wide ? Math.floor(ctx.W * 0.54) : ctx.W;
    const cm = WF.VK.confusion({ classes: v.classes, counts: v.counts, normalize: st.norm, W: figW, pick: true, sel: st.sel });
    fig.show(cm.items, cm.W, cm.H);
    fig.onPick = (key) => { const m = /cm:c(\d+)_(\d+)$/.exec(key); if (!m) return; st.sel = [+m[1], +m[2]]; ctx.redraw(); };
    const side = h('div.wf-ev-side', scoresBlock(v));
    box.append(head, h('div.wf-ev-row' + (wide ? '.wide' : ''), h('div.wf-ev-main', fig.el, VK.note(st.sel ? 'Rows ' + v.classes[st.sel[0]] + ', predicted ' + v.classes[st.sel[1]] + ': listed below.' : 'Click a cell to list the rows in it. The diagonal holds the right answers.', 'muted')), side));
    box.appendChild(examples(v, st.sel));
  }
  function scoresBlock(v) {
    const n = v.n, right = v.counts.reduce((s, r, i) => s + r[i], 0), per = v.perClass || [];
    const macro = per.length ? per.reduce((s, p) => s + p.f1, 0) / per.length : NaN;
    const lines = [
      ['accuracy', right + ' / ' + n, VK.fmt(right / n)],
      [v.classes.length === 2 ? 'F1 of ' + VK.short(v.classes[1], 14) : 'F1, macro average', v.classes.length === 2 ? '2PR / (P + R)' : 'mean of ' + per.length, VK.fmt(v.classes.length === 2 && per[1] ? per[1].f1 : macro)],
      ['MCC', 'from all four counts', VK.fmt(v.mcc)],
    ];
    const t1 = VK.table(['score', 'worked out', 'value'], lines, 'scores');
    const t2 = VK.table(['class', 'precision TP/(TP+FP)', 'recall TP/(TP+FN)', 'F1', 'rows'], per.map((p, k) => [VK.short(v.classes[k], 14), p.tp + '/' + (p.tp + p.fp) + ' = ' + VK.fmt(p.precision), p.tp + '/' + (p.tp + p.fn) + ' = ' + VK.fmt(p.recall), VK.fmt(p.f1), p.support]), 'per-class');
    return h('div.wf-scores', t1, t2);
  }
  function examples(v, sel) {
    const box = h('div.wf-examples', { 'aria-live': 'polite' });
    if (!sel) return box;
    const [i, j] = sel, list = v.cells[i][j], total = v.cellTotals[i][j];
    box.appendChild(h('h4', (i === j ? 'Right: ' : 'Wrong: ') + 'true ' + v.classes[i] + ', predicted ' + v.classes[j] + ' · ' + total + ' row' + (total === 1 ? '' : 's') + (total > list.length ? ' (the first ' + list.length + ' shown)' : '')));
    if (!list.length) { box.appendChild(VK.note('No rows fall in this cell.', 'muted')); return box; }
    const p = (e) => (e.proba ? VK.fmt(e.proba[e.pred], 2) : '–');
    if (v.kindOfRows === 'images') {
      box.appendChild(h('div.wf-thumbs', ...list.map((e) => h('figure.wf-th', VK.thumb(e.img, v.w, v.h, Math.max(2, Math.round(56 / v.w))), h('figcaption', 'row ' + e.row + ' · p ' + p(e))))));
    } else if (v.kindOfRows === 'text') {
      box.appendChild(h('ul.wf-texts', ...list.map((e) => h('li', h('q', e.text || ''), h('span.muted', ' row ' + e.row + ' · p ' + p(e))))));
    } else {
      const F = (v.features || []).slice(0, 6), more = (v.features || []).length > 6;
      box.appendChild(VK.table(['row'].concat(F, more ? ['…'] : [], ['p(predicted)']), list.map((e) => [e.row].concat(F.map((_, k) => (e.x ? VK.fmt(e.x[k], 3) : '')), more ? ['…'] : [], [p(e)])), 'examples'));
    }
    return box;
  }

  function regression(box, v, ctx) {
    const wide = !ctx.narrow && ctx.W >= 700;
    box.appendChild(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.model || 'The model'), ' on ', v.data || 'the test rows', ': ', v.n, ' predictions')));
    const W = wide ? Math.floor(ctx.W * 0.56) : ctx.W, side = Math.min(W - 90, 420), x0 = 62, y0 = 26;
    const pts = v.points, all = pts.flat(), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.05 || 1;
    const ax = C().axes({ prefix: 'pt', x: x0, y: y0, w: side, h: side, xr: [lo - pad, hi + pad], yr: [lo - pad, hi + pad], xname: 'true value', yname: 'predicted' });
    const items = ax.items.concat([C().line(ax, 'pt:ideal', [[lo - pad, lo - pad], [hi + pad, hi + pad]], 'faint')]);
    pts.forEach(([a, b], i) => items.push({ key: 'pt:' + i, kind: 'dot', x: ax.X(a), y: ax.Y(b), r: 3.6, cls: 'sp c0', audit: 'overlap-ok' }));
    const fig = VK.fig('Predicted against true');
    fig.show(items, x0 + side + 20, y0 + side + 48);
    const t = VK.table(['score', 'value'], [['MAE (mean |error|)', VK.fmt(v.mae)], ['RMSE', VK.fmt(v.rmse)], ['R²', VK.fmt(v.r2)]], 'scores');
    box.appendChild(h('div.wf-ev-row' + (wide ? '.wide' : ''), h('div.wf-ev-main', fig.el, VK.note('Each dot is a test row; on the dashed line the prediction would be exact. Dots above it were predicted too high.', 'muted')), h('div.wf-ev-side', t)));
  }

  WF.viewers.evaluate = VK.viewer((box, v, ctx) => (v.task === 'regression' ? regression(box, v, ctx) : classification(box, v, ctx)), 'evaluate');
})(globalThis.Nooral = globalThis.Nooral || {});
