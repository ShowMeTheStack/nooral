// Viewers of the curve blocks: ROC and PR with a threshold that moves a point along both curves and counts the 2 × 2
// confusion matrix again (click a curve to put the threshold there), the reliability diagram over the histogram of
// scores, residual plots, and learning and validation curves with their bands.
(function (S) {
  'use strict';
  const WF = S.WF, VK = WF.VK, EV = WF.EV, h = (...a) => S.UI.h(...a);
  const C = () => S.Kit.curve, HM = () => S.Kit.heat;
  const r2 = (v) => Math.round(v * 100) / 100;

  /** a plot frame: axes in a box, with a transparent cover that turns a click into plot units */
  const plot = (prefix, x, y, w, hh, xr, yr, o) => C().axes(Object.assign({ prefix, x, y, w, h: hh, xr, yr }, o || {}));
  const cover = (ax, key) => ({ key, kind: 'cell', x: ax.box.x, y: ax.box.y, w: ax.box.w, h: ax.box.h, rx: 0, cls: 'hit', pick: true, audit: 'ignore', layer: 'top' });
  /** where a click landed, in the stage's own units */
  const stagePoint = (fig, e) => { const svg = fig.stage.root, p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const m = svg.getScreenCTM(); return m ? p.matrixTransform(m.inverse()) : null; };
  const classChooser = (v, k, on) => (v.classes.length > 2 ? VK.select('Positive class', v.classes.map((c, i) => [i, c + ' (against the rest)']), k, (x) => on(+x)) : null);
  /** a band between two series [[u, lo, hi]..] as one closed path */
  const band = (ax, key, pts, cls) => ({ key, kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + ax.X(p[0]) + ',' + ax.Y(p[2])).join(' ') + ' ' + pts.slice().reverse().map((p) => 'L' + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' ') + ' Z', cls: 'band ' + cls, audit: 'ignore', layer: 'back', z: 4 });

  // ---- ROC and PR ----------------------------------------------------------------------------------------------------
  WF.VK.matrix2 = function (prefix, x, y, cell, at, posName) {
    const vals = [[at.tp, at.fn], [at.fp, at.tn]], max = Math.max(1, at.tp + at.fn, at.fp + at.tn);
    const pos = VK.short(posName, 10);
    const g = HM().grid({ prefix, x, y, cw: cell, ch: Math.round(cell * 0.62), vals, fmt: null, rows: ['is ' + pos, 'is not'], cols: ['says ' + pos, 'says not'] });
    const names = [['TP', 'FN'], ['FP', 'TN']];
    for (const it of g.items) {
      const m = /:c(\d)_(\d)$/.exec(it.key);
      if (!m) continue;
      const i = +m[1], j = +m[2], v = vals[i][j], good = i === j;
      it.paint = HM().colour(v, good ? 'good' : 'bad', max * 0.8);
      it.lines = [names[i][j], String(v)];
      it.text = undefined;
      it.size = 14;
    }
    return g;
  };
  function roc(box, v, ctx) {
    const st = ctx.state;
    if (st.k === undefined || st.k >= v.classes.length) st.k = v.cls;
    if (st.t === undefined) st.t = v.threshold;
    const k = st.k, cv = EV.curves(v, k), W = ctx.W, narrow = ctx.narrow;
    const out = h('output', VK.fmt(st.t, 2));
    const slider = h('input', { type: 'range', min: '0', max: '1', step: '0.005', value: String(st.t), 'aria-label': 'Threshold', id: 'wfThreshold' });
    const kpis = h('div.wf-kpis');
    const fig = VK.fig('ROC and precision–recall curves at a threshold');
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.model || 'The model'), ': "' + v.classes[k] + '" when its probability is at least the threshold'), classChooser(v, k, (x) => { st.k = x; ctx.redraw(); if (ctx.api.setParams) ctx.api.setParams({ cls: x }); })),
      h('label.wf-threshold', h('span', 'Threshold'), slider, out), fig.el, kpis,
      VK.note('Lower the threshold and more rows are called "' + v.classes[k] + '": the true-positive rate (recall) rises, and so do the false alarms. Click a curve to put the threshold at that point.', 'muted'));
    const draw = () => {
      const at = EV.at(v, k, st.t);
      const P = cv.roc.P, N = cv.roc.N;
      // three across when there is room, the two curves side by side over the matrix, or all stacked on a phone
      const mode = !narrow && W >= 900 ? 'row' : !narrow && W >= 540 ? 'pair' : 'stack';
      const side = mode === 'row' ? Math.max(200, Math.min(290, Math.floor((W - 330) / 2) - 70)) : mode === 'pair' ? Math.max(170, Math.min(280, Math.floor((W - 2 * 54 - 50) / 2))) : Math.min(W - 80, 300);
      const m = { l: 54, t: 32, b: 58 };
      const items = [];
      const ax1 = plot('roc', m.l, m.t, side, side, [0, 1], [0, 1], { xticks: 0.25, yticks: 0.25, xname: 'false-positive rate', yname: 'true-positive rate' });
      const ox2 = mode === 'stack' ? m.l : m.l + side + 30 + m.l, oy2 = mode === 'stack' ? m.t + side + m.b + m.t : m.t;
      const ax2 = plot('pr', ox2, oy2, side, side, [0, 1], [0, 1], { xticks: 0.25, yticks: 0.25, xname: 'recall', yname: 'precision' });
      items.push(...ax1.items, ...ax2.items);
      items.push(C().line(ax1, 'roc:chance', [[0, 0], [1, 1]], 'faint'), C().line(ax1, 'roc:curve', cv.roc.fpr.map((f, i) => [f, cv.roc.tpr[i]]), 'act'));
      const base = P / (P + N);
      items.push(C().line(ax2, 'pr:base', [[0, base], [1, base]], 'faint'), C().line(ax2, 'pr:curve', cv.pr.recall.map((r, i) => [r, cv.pr.precision[i]]), 'act'));
      items.push(C().dot(ax1, 'roc:pt', at.fpr, at.recall, 'thr', 6), C().dot(ax2, 'pr:pt', at.recall, at.tp + at.fp ? at.precision : 1, 'thr', 6));
      items.push({ key: 'roc:auc', kind: 'text', x: ax1.X(0.96), y: ax1.Y(0.06), text: 'AUC ' + VK.fmt(cv.roc.auc), anchor: 'end', size: 13, cls: 'cap' });
      items.push({ key: 'pr:ap', kind: 'text', x: ax2.X(0.04), y: ax2.Y(0.06), text: 'AP ' + VK.fmt(cv.pr.ap), size: 13, cls: 'cap' });
      items.push(cover(ax1, 'roc:hit'), cover(ax2, 'pr:hit'));
      const cell = mode === 'stack' ? Math.min(118, Math.floor((W - 150) / 2)) : 104, mx = mode === 'row' ? ox2 + side + 36 + 92 : 100, my = mode === 'row' ? m.t + 26 : oy2 + side + m.b + 40;
      const g = WF.VK.matrix2('mx', mx, my, cell, at, v.classes[k]);
      items.push(...g.items, { key: 'mx:title', kind: 'text', x: mx - 92, y: my - 34, text: 'at threshold ' + VK.fmt(st.t, 2), size: 13, cls: 'axname' });
      const Wf = mode === 'row' ? mx + g.w + 10 : Math.max(ox2 + side + 20, mx + g.w + 10), Hf = mode === 'row' ? Math.max(m.t + side + m.b, my + g.h + 14) : my + g.h + 14;
      fig.show(items, Wf, Hf);
      fig.onPick = (key, e) => {
        const p = stagePoint(fig, e), ax = /^roc/.test(key) ? ax1 : /^pr/.test(key) ? ax2 : null;
        if (!p || !ax) return;
        let best = -1, bd = Infinity;
        const pts = ax === ax1 ? cv.roc.fpr.map((f, i) => [f, cv.roc.tpr[i], cv.roc.thresholds[i]]) : cv.pr.recall.map((r, i) => [r, cv.pr.precision[i], cv.pr.thresholds[i]]);
        pts.forEach((q, i) => { if (!Number.isFinite(q[2])) return; const d = Math.hypot(ax.X(q[0]) - p.x, ax.Y(q[1]) - p.y); if (d < bd) { bd = d; best = i; } });
        if (best < 0) return;
        st.t = Math.max(0, Math.min(1, pts[best][2]));
        slider.value = String(st.t); out.textContent = VK.fmt(st.t, 2); draw(); keep();
      };
      kpis.innerHTML = '';
      for (const [n, x] of [['precision', at.precision], ['recall (TPR)', at.recall], ['false-positive rate', at.fpr], ['F1', at.f1], ['called "' + VK.short(v.classes[k], 10) + '"', (at.tp + at.fp) + ' of ' + (P + N)]]) kpis.appendChild(h('div.wf-kpi', n, h('b', typeof x === 'number' ? VK.fmt(x) : x)));
    };
    let timer = null;
    const keep = () => { clearTimeout(timer); timer = setTimeout(() => { if (ctx.api.setParams) ctx.api.setParams({ threshold: +(+st.t).toFixed(3) }); }, 600); };
    slider.addEventListener('input', () => { st.t = +slider.value; out.textContent = VK.fmt(st.t, 2); draw(); keep(); });
    draw();
  }
  WF.viewers.roc = VK.viewer(roc, 'roc');

  // ---- calibration ---------------------------------------------------------------------------------------------------
  function calibration(box, v, ctx) {
    const W = ctx.W, side = Math.min(W - 80, ctx.narrow ? 300 : 340), m = { l: 56, t: 32 };
    const ax = plot('cal', m.l, m.t, side, side, [0, 1], [0, 1], { xticks: 0.2, yticks: 0.2, xname: 'mean predicted probability', yname: 'share that really was "' + VK.short(v.classes[v.cls], 14) + '"' });
    const items = ax.items.slice();
    items.push(C().line(ax, 'cal:ideal', [[0, 0], [1, 1]], 'faint'));
    const used = v.bins.filter((b) => b.count > 0), maxc = Math.max(...used.map((b) => b.count));
    items.push(C().line(ax, 'cal:line', used.map((b) => [b.meanProb, b.fraction]), 'act'));
    used.forEach((b, i) => items.push(Object.assign(C().dot(ax, 'cal:b' + i, b.meanProb, b.fraction, '', r2(3.5 + 4.5 * Math.sqrt(b.count / maxc))), { audit: 'overlap-ok', point: false })));
    items.push({ key: 'cal:lab', kind: 'text', x: ax.X(0.03), y: ax.Y(0.92), text: 'dashed: perfectly calibrated', size: 12, cls: 'tick' });
    // the histogram of scores under the plot, by true class
    const hy = m.t + side + 58, hh = 96, hmax = Math.max(1, ...v.hist[0].map((c, i) => c + v.hist[1][i]));
    const hx = plot('hist', m.l, hy, side, hh, [0, 1], [0, hmax], { xticks: 0.2, yticks: [0, hmax], xname: 'predicted probability', yname: 'rows' });
    items.push(...hx.items);
    const bw = side / 20;
    v.hist[0].forEach((c0, i) => {
      const c1 = v.hist[1][i], x = r2(m.l + i * bw + 1);
      if (c0) items.push({ key: 'hist:n' + i, kind: 'cell', x, y: hx.Y(c0), w: r2(bw - 2), h: r2(hx.Y(0) - hx.Y(c0)), rx: 1, cls: 'ebar c0', audit: 'ignore' });
      if (c1) items.push({ key: 'hist:p' + i, kind: 'cell', x, y: hx.Y(c0 + c1), w: r2(bw - 2), h: r2(hx.Y(c0) - hx.Y(c0 + c1)), rx: 1, cls: 'ebar c1', audit: 'ignore' });
    });
    const lx = m.l + side + 18, legend = ctx.narrow ? [] : C().legend('cal:lg', lx, hy + 14, [{ text: 'is ' + VK.short(v.classes[v.cls], 12), cls: 'c1', dot: true }, { text: 'is not', cls: 'c0', dot: true }], { dir: 'col' });
    items.push(...legend);
    const fig = VK.fig('Reliability diagram');
    fig.show(items, ctx.narrow ? m.l + side + 16 : lx + 130, hy + hh + 56);
    const head = h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.model || 'The model'), ': are its probabilities for "' + v.classes[v.cls] + '" honest?'),
      v.classes.length > 2 ? classChooser(v, v.cls, (x) => VK.set(ctx, { cls: x })) : null,
      VK.select('Bins', [5, 8, 10, 15, 20].map((n) => [n, String(n)]), v.bins.length, (x) => VK.set(ctx, { bins: +x })));
    const kp = h('div.wf-kpis', ...[['Brier score', v.brier], ['log loss', v.logloss], ['expected calibration error', v.ece]].map(([n, x]) => h('div.wf-kpi', n, h('b', VK.fmt(x)))));
    const wide = !ctx.narrow && ctx.W >= 760;
    box.append(head, h('div.wf-ev-row' + (wide ? '.wide' : ''), h('div.wf-ev-main', fig.el), h('div.wf-ev-side', kp, VK.note('Each dot is a bin of rows (bigger: more rows): across, the probability the model gave them on average; up, the share that really was "' + v.classes[v.cls] + '". A dot below the dashed line is a bin where the model was too sure; above it, not sure enough.', 'muted'),
      VK.note('The bars under it count the rows at each probability: a model that is often near 0 or 1 is decisive, and the dots then say whether it has the right to be.', 'muted'))));
  }
  WF.viewers.calibration = VK.viewer(calibration, 'calibration');

  // ---- residuals -----------------------------------------------------------------------------------------------------
  function residuals(box, v, ctx) {
    const W = ctx.W, narrow = ctx.narrow, res = v.y.map((y, i) => y - v.pred[i]);
    const ext = (a, pad) => { const lo = Math.min(...a), hi = Math.max(...a), d = (hi - lo) * (pad || 0.06) || 1; return [lo - d, hi + d]; };
    const rr = ext(res), pw = narrow ? W - 76 : Math.min(400, Math.floor((W - 160) / 2)), ph = Math.round(pw * 0.68), m = { l: 60, t: 30, b: 56 };
    const items = [];
    const ax1 = plot('r1', m.l, m.t, pw, ph, ext(v.pred), rr, { xname: 'predicted', yname: 'error (true − predicted)' });
    items.push(...ax1.items, C().line(ax1, 'r1:zero', [[ax1.o.xr[0], 0], [ax1.o.xr[1], 0]], 'faint'));
    res.forEach((e, i) => items.push({ key: 'r1:' + i, kind: 'dot', x: ax1.X(v.pred[i]), y: ax1.Y(e), r: 3.3, cls: 'sp c0', audit: 'overlap-ok' }));
    let Wf = m.l + pw + 20, Hf = m.t + ph + m.b;
    if (v.fx) {
      const ox = narrow ? m.l : m.l + pw + 34 + m.l, oy = narrow ? m.t + ph + m.b + m.t : m.t;
      const ax2 = plot('r2', ox, oy, pw, ph, ext(v.fx), rr, { xname: v.features[v.feature] || 'feature', yname: 'error' });
      items.push(...ax2.items, C().line(ax2, 'r2:zero', [[ax2.o.xr[0], 0], [ax2.o.xr[1], 0]], 'faint'));
      res.forEach((e, i) => items.push({ key: 'r2:' + i, kind: 'dot', x: ax2.X(v.fx[i]), y: ax2.Y(e), r: 3.3, cls: 'sp c1', audit: 'overlap-ok' }));
      Wf = ox + pw + 20; Hf = oy + ph + m.b;
    }
    // the spread of the errors, as a histogram
    const nb = 16, lo = rr[0], step = (rr[1] - rr[0]) / nb, counts = new Array(nb).fill(0);
    for (const e of res) counts[Math.min(nb - 1, Math.max(0, Math.floor((e - lo) / step)))]++;
    const hy = Hf + 22, hh = 90, hw = narrow ? pw : Math.min(pw * 2 + 34 + m.l, W - m.l - 20), cmax = Math.max(...counts);
    const ax3 = plot('r3', m.l, hy, hw, hh, rr, [0, cmax], { yticks: [0, cmax], xname: 'error', yname: 'rows' });
    items.push(...ax3.items);
    counts.forEach((c, i) => { if (c) items.push({ key: 'r3:b' + i, kind: 'cell', x: r2(ax3.X(lo + i * step) + 1), y: ax3.Y(c), w: r2(ax3.X(lo + step) - ax3.X(lo) - 2), h: r2(ax3.Y(0) - ax3.Y(c)), rx: 1, cls: 'ebar soft', audit: 'ignore' }); });
    const fig = VK.fig('Residuals');
    fig.show(items, Math.max(Wf, m.l + hw + 20), hy + hh + m.b);
    const fsel = v.features.length ? VK.select('Errors against', v.features.map((f, j) => [j, f]), v.feature, (x) => VK.set(ctx, { feature: +x })) : null;
    const kp = h('div.wf-kpis', ...[['MAE', v.mae], ['RMSE', v.rmse], ['R²', v.r2], ['mean error', v.mean]].map(([n, x]) => h('div.wf-kpi', n, h('b', VK.fmt(x)))));
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.model || 'The model'), ': ', v.n, ' errors'), fsel), fig.el, kp,
      VK.note('A good model leaves a shapeless band around the zero line. A curve or a funnel in the dots is a pattern the model missed: a feature it treats as a straight line, or errors that grow with the size of the answer.', 'muted'));
  }
  WF.viewers.residuals = VK.viewer(residuals, 'residuals');

  // ---- learning and validation curves ----------------------------------------------------------------------------------
  function learning(box, v, ctx) {
    const W = ctx.W, narrow = ctx.narrow, pw = narrow ? W - 80 : Math.min(420, Math.floor((W - 170) / (v.sweep ? 2 : 1.4))), ph = Math.round(pw * 0.66), m = { l: 58, t: 34, b: 58 };
    const all = v.learn.flatMap((l) => [l.train[0] + l.train[1], l.val[0] - l.val[1]]).concat(v.sweep ? v.sweep.pts.flatMap((p) => [p.train, p.val]) : []).filter(Number.isFinite);
    const lo = Math.max(v.metric === 'accuracy' ? 0 : -1, Math.floor(Math.min(...all) * 10) / 10), hi = Math.min(1, Math.ceil(Math.max(...all) * 10) / 10 + 0.0001);
    const yr = [lo, Math.max(hi, lo + 0.1)];
    const items = [];
    const xs = v.sizes, ax = plot('lc', m.l, m.t, pw, ph, [0, xs[xs.length - 1] * 1.04], yr, { xname: 'training rows', yname: v.metricLabel });
    items.push(...ax.items);
    items.push(band(ax, 'lc:bt', v.learn.map((l, i) => [xs[i], l.train[0] - l.train[1], l.train[0] + l.train[1]]), 'c1'), band(ax, 'lc:bv', v.learn.map((l, i) => [xs[i], l.val[0] - l.val[1], l.val[0] + l.val[1]]), 'c0'));
    items.push(C().line(ax, 'lc:t', v.learn.map((l, i) => [xs[i], l.train[0]]), 'c1'), C().line(ax, 'lc:v', v.learn.map((l, i) => [xs[i], l.val[0]]), 'c0'));
    const dot = (a, k, u, w, c) => Object.assign(C().dot(a, k, u, w, c, 3.6), { audit: 'overlap-ok', point: false });
    v.learn.forEach((l, i) => { items.push(dot(ax, 'lc:dt' + i, xs[i], l.train[0], 'c1'), dot(ax, 'lc:dv' + i, xs[i], l.val[0], 'c0')); });
    let Wf = m.l + pw + 20, legY = m.t + ph + m.b - 4;
    if (v.sweep) {
      const ox = narrow ? m.l : m.l + pw + 40 + m.l, oy = narrow ? m.t + ph + m.b + m.t : m.t, sp = v.sweep.pts.filter((p) => Number.isFinite(p.val));
      const X = (p) => (v.sweep.log ? Math.log10(p.v) : p.v), xr = [Math.min(...sp.map(X)), Math.max(...sp.map(X))];
      const pad = (xr[1] - xr[0]) * 0.04 || 1;
      const ax2 = plot('vc', ox, oy, pw, ph, [xr[0] - pad, xr[1] + pad], yr, { xname: String(v.sweep.label || v.sweep.param).replace(/\s*\(.*\)\s*$/, '').toLowerCase(), yname: v.metricLabel, fmtX: v.sweep.log ? (u) => { const z = Math.pow(10, u); return String(+z.toPrecision(2)); } : undefined, xticks: v.sweep.log ? sp.map(X).filter((_, i, a) => a.length < 6 || i % 2 === 0) : undefined });
      items.push(...ax2.items, C().line(ax2, 'vc:t', sp.map((p) => [X(p), p.train]), 'c1'), C().line(ax2, 'vc:v', sp.map((p) => [X(p), p.val]), 'c0'));
      sp.forEach((p, i) => items.push(dot(ax2, 'vc:dt' + i, X(p), p.train, 'c1'), dot(ax2, 'vc:dv' + i, X(p), p.val, 'c0')));
      const best = sp.reduce((b, p) => (p.val > b.val ? p : b), sp[0]);
      items.push(C().vline(ax2, 'vc:best', X(best), 'act'));
      Wf = ox + pw + 20;
      if (narrow) legY = oy + ph + m.b - 4;
    }
    items.push(...C().legend('lc:lg', m.l, legY + 16, [{ text: 'on its training rows', cls: 'c1' }, { text: 'on validation rows', cls: 'c0' }], { dir: narrow ? 'col' : 'row' }));
    const fig = VK.fig('Learning curve');
    fig.show(items, Wf, legY + (narrow ? 50 : 28));
    const words = { under: 'Both scores are low and close: the model is too simple for this data (it underfits). More rows will not help much; a more flexible model will.', over: 'The training score is well above the validation score: the model learns its training rows by heart (it overfits). More rows, or a simpler or more regularised model, close the gap.', fit: 'The two scores meet at a good level: the model fits without memorising.' };
    const sel = (v.sweepable || []).length ? VK.select('Validation curve over', [['', 'none']].concat(v.sweepable), v.sweep ? v.sweep.param : '', (x) => VK.set(ctx, { sweep: x })) : null;
    const wide = !narrow && !v.sweep && W >= 760;
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.label), ': validation ' + v.metricLabel + ' ' + VK.fmt(v.final) + ', gap ' + VK.fmt(v.gap)), sel), h('div.wf-ev-row' + (wide ? '.wide' : ''), h('div.wf-ev-main', fig.el), h('div.wf-ev-side',
      h('div.wf-verdict', words[v.verdict]), VK.note('The bands are one standard deviation over the repeats, each with its own shuffle of the rows.' + (v.sweep ? ' On the right, the same model trained at each value of ' + v.sweep.param + '; the marked line is the best on validation.' : ''), 'muted'))));
  }
  WF.viewers.learningcurve = VK.viewer(learning, 'learning');
})(globalThis.Nooral = globalThis.Nooral || {});
