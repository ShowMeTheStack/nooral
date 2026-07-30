// Viewers of the explanation blocks: permutation importance as bars with their spread, partial dependence with every
// row's own curve behind the average, SHAP as a waterfall for one row and a swarm over many, LIME's local slopes, and
// saliency and Grad-CAM laid over the images.
(function (S) {
  'use strict';
  const WF = S.WF, VK = WF.VK, h = (...a) => S.UI.h(...a);
  const C = () => S.Kit.curve, tw = (t, s) => S.geom.textWidth(String(t), s || 13);
  const r2 = (v) => Math.round(v * 100) / 100;
  const sgn = (x) => (x >= 0 ? '+' : '−') + VK.fmt(Math.abs(x), 3);
  const clsName = (v) => (v.task === 'classification' ? 'P(' + VK.short(v.classes[v.cls], 14) + ')' : 'the prediction');
  const rowPicker = (ctx, v) => {
    const i = h('input', { type: 'number', min: '0', max: String((v.n || 1) - 1), value: String(v.row), 'aria-label': 'Row to explain', style: { width: '84px' }, id: 'wfRow' });
    i.addEventListener('change', () => { const x = Math.max(0, Math.round(+i.value) || 0); VK.set(ctx, { row: x }); });
    return h('label.wf-pick', h('span', 'Row'), i);
  };
  const classPicker = (ctx, v) => (v.task === 'classification' && v.classes.length > 2 ? VK.select('Class', v.classes.map((c, i) => [i, c]), v.cls, (x) => VK.set(ctx, { cls: +x })) : null);

  /** horizontal bars from a zero line: rows [{name, value, lo, hi, cls}] -> {items, W, H} */
  function hbars(prefix, rows, W, opts) {
    opts = opts || {};
    const left = Math.min(220, Math.max(70, 16 + Math.max(...rows.map((r) => tw(r.name, 13))))), right = 76, bw = Math.max(120, W - left - right), rh = 26, top = opts.top || 26;
    const all = rows.flatMap((r) => [r.value, r.lo === undefined ? r.value : r.lo, r.hi === undefined ? r.value : r.hi, 0]);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.04, X = (x) => r2(left + ((x - lo + pad) / (hi - lo + 2 * pad)) * bw);
    const items = [{ key: prefix + ':zero', kind: 'path', d: 'M' + X(0) + ',' + (top - 6) + ' V' + (top + rows.length * rh), cls: 'zero', audit: 'ignore', layer: 'back' }];
    rows.forEach((r, i) => {
      const y = top + i * rh, a = Math.min(X(0), X(r.value)), b = Math.max(X(0), X(r.value));
      items.push({ key: prefix + ':n' + i, kind: 'text', x: left - 10, y: y + 17, text: r.name, anchor: 'end', size: 13, cls: 'cap' });
      items.push({ key: prefix + ':b' + i, kind: 'cell', x: a, y: y + 5, w: Math.max(1.5, r2(b - a)), h: 16, rx: 2, cls: 'ebar ' + (r.cls || (r.value >= 0 ? 'pos' : 'neg')), audit: 'ignore', title: r.name + ': ' + VK.fmt(r.value, 4) });
      if (r.lo !== undefined && r.hi > r.lo) items.push({ key: prefix + ':w' + i, kind: 'path', d: 'M' + X(r.lo) + ',' + (y + 13) + ' H' + X(r.hi) + ' M' + X(r.lo) + ',' + (y + 8) + ' V' + (y + 18) + ' M' + X(r.hi) + ',' + (y + 8) + ' V' + (y + 18), cls: 'whisker', audit: 'ignore', layer: 'top' });
      items.push({ key: prefix + ':v' + i, kind: 'text', x: left + bw + 10, y: y + 17, text: r.label || VK.fmt(r.value, 3), size: 12.5, cls: 'tick', mono: true });
    });
    if (opts.axis) { const aw = tw(opts.axis, 13), far = aw > left + bw - 4; items.push({ key: prefix + ':ax', kind: 'text', x: far ? 4 : left + bw, y: top + rows.length * rh + 18, text: opts.axis, anchor: far ? 'start' : 'end', size: 13, cls: 'axname' }); }
    return { items, W: left + bw + right, H: top + rows.length * rh + (opts.axis ? 26 : 8), X };
  }

  // ---- permutation importance -------------------------------------------------------------------------------------------
  WF.viewers.importance = VK.viewer((box, v, ctx) => {
    const rows = v.order.map((j) => ({ name: v.features[j], value: v.mean[j], lo: v.mean[j] - v.std[j], hi: v.mean[j] + v.std[j], cls: v.mean[j] >= 0 ? '' : 'neg' }));
    const g = hbars('imp', rows, Math.min(ctx.W, 760), { axis: 'fall in ' + v.metric + ' when the column is shuffled' });
    const fig = VK.fig('Permutation importance');
    fig.show(g.items, g.W, g.H);
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.label), ': ', v.metric, ' with the data intact ', VK.fmt(v.baseline))), fig.el,
      VK.note('Each bar is how far the score falls, on average, when one column is shuffled and the others are kept; the whiskers span one standard deviation over the repeats. A bar near zero is a column the model does not use; one below zero did better shuffled, which is noise.', 'muted'));
  }, 'importance');

  // ---- partial dependence -----------------------------------------------------------------------------------------------
  WF.viewers.pdp = VK.viewer((box, v, ctx) => {
    const W = Math.min(ctx.W, 720), pw = W - 90, ph = Math.round(Math.min(340, pw * 0.62)), m = { l: 60, t: 30 };
    const ys = v.ice.flat().concat(v.average), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.06 || 0.05;
    const xr = [v.grid[0], v.grid[v.grid.length - 1]], xp = (xr[1] - xr[0]) * 0.02 || 1;
    const ax = C().axes({ prefix: 'pd', x: m.l, y: m.t, w: pw, h: ph, xr: [xr[0] - xp, xr[1] + xp], yr: [lo - pad, hi + pad], xname: v.features[v.feature], yname: clsName(v) });
    const items = ax.items.slice();
    v.ice.forEach((row, i) => items.push(Object.assign(C().line(ax, 'pd:i' + i, v.grid.map((g, k) => [g, row[k]]), 'ice'), { layer: 'back', pathAudit: 'ignore' })));
    items.push(C().line(ax, 'pd:avg', v.grid.map((g, k) => [g, v.average[k]]), 'act'));
    const rug = v.rug.filter((x) => x >= xr[0] - xp && x <= xr[1] + xp).map((x) => 'M' + ax.X(x) + ',' + (m.t + ph) + ' v-7').join(' ');
    if (rug) items.push({ key: 'pd:rug', kind: 'path', d: rug, cls: 'rug', audit: 'ignore', layer: 'back' });
    items.push(...C().legend('pd:lg', m.l, m.t + ph + 76, [{ text: 'average: partial dependence', cls: 'act' }, { text: 'one row each (ICE)', cls: 'ice' }], { dir: ctx.narrow ? 'col' : 'row' }));
    const fig = VK.fig('Partial dependence');
    fig.show(items, m.l + pw + 20, m.t + ph + (ctx.narrow ? 120 : 92));
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.label), ': ', clsName(v), ' as ', v.features[v.feature], ' moves'),
      h('div.wf-tools', VK.select('Feature', v.features.map((f, j) => [j, f]), v.feature, (x) => VK.set(ctx, { feature: +x })), classPicker(ctx, v))), fig.el,
      VK.note('Every row is set to each value of ' + v.features[v.feature] + ' in turn, with its other columns kept, and the model asked again: the thin lines are the rows, the thick one their average. Lines that run apart mean the effect depends on the other columns. The ticks along the bottom are where the rows really lie.', 'muted'));
  }, 'pdp');

  // ---- SHAP ---------------------------------------------------------------------------------------------------------------
  WF.viewers.shap = VK.viewer((box, v, ctx) => {
    const W = Math.min(ctx.W, 760), F = v.features, order = F.map((_, j) => j).sort((a, b) => Math.abs(v.phi[b]) - Math.abs(v.phi[a]));
    const top = order.slice(0, 10), rest = order.slice(10), restSum = rest.reduce((s, j) => s + v.phi[j], 0);
    // the waterfall: from the average prediction, one feature at a time, to this row's prediction
    const steps = top.map((j) => ({ name: F[j] + ' = ' + VK.fmt(v.x[j], 3), d: v.phi[j] })).concat(rest.length ? [{ name: rest.length + ' other columns', d: restSum }] : []);
    const left = Math.min(240, Math.max(90, 16 + Math.max(...steps.map((s) => tw(s.name, 13))))), right = 70, bw = Math.max(120, W - left - right), rh = 26, t0 = 34;
    let cum = v.base;
    const pts = [cum];
    for (const s of steps) { cum += s.d; pts.push(cum); }
    let lo = Math.min(...pts), hi = Math.max(...pts);
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.05, X = (x) => r2(left + ((x - lo + pad) / (hi - lo + 2 * pad)) * bw);
    const items = [];
    cum = v.base;
    steps.forEach((s, i) => {
      const y = t0 + i * rh, a = X(Math.min(cum, cum + s.d)), b = X(Math.max(cum, cum + s.d));
      items.push({ key: 'wf:n' + i, kind: 'text', x: left - 10, y: y + 17, text: s.name, anchor: 'end', size: 13, cls: 'cap' });
      items.push({ key: 'wf:b' + i, kind: 'cell', x: a, y: y + 5, w: Math.max(1.5, r2(b - a)), h: 16, rx: 2, cls: 'ebar ' + (s.d >= 0 ? 'pos' : 'neg'), audit: 'ignore' });
      items.push({ key: 'wf:v' + i, kind: 'text', x: left + bw + 10, y: y + 17, text: sgn(s.d), size: 12.5, cls: 'tick', mono: true });
      cum += s.d;
    });
    const yEnd = t0 + steps.length * rh;
    items.push({ key: 'wf:base', kind: 'path', d: 'M' + X(v.base) + ',' + (t0 - 6) + ' V' + yEnd, cls: 'zero', audit: 'ignore', layer: 'back' });
    items.push({ key: 'wf:fx', kind: 'path', d: 'M' + X(v.fx) + ',' + t0 + ' V' + (yEnd + 8), cls: 'zero', audit: 'ignore', layer: 'back' });
    items.push({ key: 'wf:bt', kind: 'text', x: X(v.base), y: t0 - 12, text: 'average ' + VK.fmt(v.base, 3), anchor: X(v.base) > left + bw * 0.6 ? 'end' : 'start', size: 12.5, cls: 'tick' });
    items.push({ key: 'wf:ft', kind: 'text', x: X(v.fx), y: yEnd + 24, text: 'this row ' + VK.fmt(v.fx, 3), anchor: X(v.fx) > left + bw * 0.6 ? 'end' : 'start', size: 13, cls: 'cap' });
    const fig = VK.fig('SHAP waterfall');
    fig.show(items, left + bw + right, yEnd + 36);
    // the swarm: every summarised row as a dot on each feature's line, coloured by how high the feature's value was
    const so = F.map((_, j) => j).sort((a, b) => v.meanAbs[b] - v.meanAbs[a]).slice(0, 10);
    const allPhi = v.summary.flatMap((r) => so.map((j) => r.phi[j])), plo = Math.min(0, ...allPhi), phi = Math.max(0, ...allPhi), pp = (phi - plo) * 0.05 || 0.05;
    const sl = Math.min(200, Math.max(70, 16 + Math.max(...so.map((j) => tw(F[j], 13))))), sw = Math.max(140, W - sl - 30), rh2 = 30, s0 = 28;
    const SX = (x) => r2(sl + ((x - plo + pp) / (phi - plo + 2 * pp)) * sw);
    const it2 = [{ key: 'bs:zero', kind: 'path', d: 'M' + SX(0) + ',' + (s0 - 4) + ' V' + (s0 + so.length * rh2), cls: 'zero', audit: 'ignore', layer: 'back' }];
    so.forEach((j, r) => {
      const y = s0 + r * rh2 + rh2 / 2;
      it2.push({ key: 'bs:n' + j, kind: 'text', x: sl - 10, y: y + 4.5, text: F[j], anchor: 'end', size: 13, cls: 'cap' });
      v.summary.forEach((row, k) => { const jit = (((k * 7919) % 13) / 12 - 0.5) * 14; it2.push({ key: 'bs:' + j + '_' + k, kind: 'dot', x: SX(row.phi[j]), y: r2(y + jit), r: 3.4, cls: 'bee', audit: 'overlap-ok', paint: { fill: 'color-mix(in srgb, var(--cls1) ' + Math.round(row.pos[j] * 100) + '%, var(--cls0))' } }); });
    });
    const ys = s0 + so.length * rh2;
    const ticks = [...new Set([plo, 0, phi].map((x) => +x.toPrecision(2)))].filter((x, i, a) => i === 0 || Math.abs(SX(x) - SX(a[i - 1])) > 40);
    ticks.forEach((x, i) => it2.push({ key: 'bs:t' + i, kind: 'text', x: SX(x), y: ys + 16, text: VK.fmt(x, 3), anchor: 'middle', size: 12, cls: 'tick' }));
    it2.push({ key: 'bs:ax', kind: 'text', x: sl + sw, y: ys + 38, text: 'SHAP value: push on ' + clsName(v), anchor: 'end', size: 13, cls: 'axname' });
    it2.push(...C().legend('bs:lg', sl, ys + (ctx.narrow ? 64 : 38), [{ text: 'low value', cls: 'c0', dot: true }, { text: 'high value', cls: 'c1', dot: true }], { dir: 'row' }));
    const fig2 = VK.fig('SHAP summary');
    fig2.show(it2, sl + sw + 20, ys + (ctx.narrow ? 78 : 52));
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.label), ': why row ' + v.row + ' got ' + VK.fmt(v.fx, 3) + ' (' + v.method + ')'), h('div.wf-tools', rowPicker(ctx, v), classPicker(ctx, v))), fig.el,
      VK.note('Starting from the average prediction over the data, each column pushes this row’s ' + (v.task === 'classification' ? clsName(v) : 'prediction') + ' up (blue) or down (vermilion); the pushes add up exactly to what the model said.', 'muted'),
      h('h4.wf-sub', 'Over ' + v.summary.length + ' rows'), fig2.el,
      VK.note('Each dot is one row: how far that column pushed it, coloured by whether the column’s value was low or high. Columns at the top matter most; a clean split of colours left and right shows which way a column works.', 'muted'));
  }, 'shap');

  // ---- LIME -----------------------------------------------------------------------------------------------------------------
  WF.viewers.lime = VK.viewer((box, v, ctx) => {
    const order = v.features.map((_, j) => j).sort((a, b) => Math.abs(v.scaled[b]) - Math.abs(v.scaled[a])).slice(0, 12);
    const rows = order.map((j) => ({ name: v.features[j] + ' = ' + VK.fmt(v.x[j], 3), value: v.scaled[j], label: sgn(v.scaled[j]) }));
    const g = hbars('lm', rows, Math.min(ctx.W, 760), { axis: ctx.narrow ? 'change for one sd more' : 'change in ' + clsName(v) + ' for one standard deviation more' });
    const fig = VK.fig('LIME weights');
    fig.show(g.items, g.W, g.H);
    const trust = v.score >= 0.8 ? 'The line fits the model closely here, so the slopes can be trusted near this row.' : v.score >= 0.5 ? 'The line fits only roughly: read the slopes as a sketch.' : 'The line fits poorly: the model bends sharply near this row, and no straight line describes it well.';
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.label), ': near row ' + v.row), h('div.wf-tools', rowPicker(ctx, v), classPicker(ctx, v))),
      h('div.wf-kpis', h('div.wf-kpi', 'the model says', h('b', VK.fmt(v.fx, 3))), h('div.wf-kpi', 'the line says', h('b', VK.fmt(v.intercept, 3))), h('div.wf-kpi', 'fit of the line (R²)', h('b', VK.fmt(v.score, 3))), h('div.wf-kpi', 'points drawn', h('b', String(v.samples)))),
      fig.el, h('div.wf-verdict', trust),
      VK.note(v.samples + ' points were drawn around the row, the model was asked about each, and a straight line was fitted with nearer points counting more. Each bar is that line’s slope for one column, scaled to one standard deviation of the column so the columns compare.', 'muted'));
  }, 'lime');

  // ---- saliency and Grad-CAM ---------------------------------------------------------------------------------------------------
  WF.viewers.saliency = VK.viewer((box, v, ctx) => {
    const n = v.items.length, hasCam = v.items.some((it) => it.cam), panels = hasCam ? 3 : 2;
    const cell = ctx.narrow ? Math.floor((ctx.W - 30) / panels) - 8 : 112, gapX = 8;
    const per = panels * (cell + gapX) + 22, cols = Math.max(1, Math.min(n, Math.floor((Math.min(ctx.W, 980) + 22) / per)));
    const items = [];
    v.items.forEach((it, k) => {
      const x0 = (k % cols) * per, y0 = Math.floor(k / cols) * (cell + 74) + 20;
      const say = 'true ' + v.classes[it.y] + ', says ' + v.classes[it.pred] + (it.proba ? ' (' + VK.pct(it.proba[it.pred]) + ')' : '');
      items.push({ key: 's' + k + ':t', kind: 'text', x: x0, y: y0 - 6, text: say, size: 13, cls: it.y === it.pred ? 'good' : 'bad' });
      const ras = (key, x, vals, scheme, max, name) => { items.push({ key, kind: 'raster', x, y: y0 + 4, w: cell, h: cell, nx: v.w, ny: v.h, vals, scheme, max, smooth: false }); items.push({ key: key + 'n', kind: 'text', x: x + cell / 2, y: y0 + cell + 22, text: name, anchor: 'middle', size: 12, cls: 'tick' }); };
      ras('s' + k + ':img', x0, it.img, 'ink', 15, 'image');
      ras('s' + k + ':sal', x0 + cell + gapX, it.sal.map((x) => Math.sqrt(x)), 'seq', 1, 'saliency');
      if (it.cam) {
        const hh = it.cam.length, ww = it.cam[0].length, up = [];
        for (let j = 0; j < v.h; j++) for (let i = 0; i < v.w; i++) up.push(it.cam[Math.min(hh - 1, Math.floor((j * hh) / v.h))][Math.min(ww - 1, Math.floor((i * ww) / v.w))] * 0.7 + (it.img[j * v.w + i] / 15) * 0.3);
        ras('s' + k + ':cam', x0 + 2 * (cell + gapX), up, 'seq', 1, 'Grad-CAM');
      }
    });
    const rowsN = Math.ceil(n / cols), fig = VK.fig('Saliency');
    fig.show(items, cols * per - 22, rowsN * (cell + 74) + 6);
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', v.label), ': where it looked'), h('div.wf-tools', h('label.wf-pick', h('span', 'First image'), (() => { const i = h('input', { type: 'number', min: '0', value: String(v.items[0] ? v.items[0].i : 0), style: { width: '84px' }, 'aria-label': 'First image' }); i.addEventListener('change', () => VK.set(ctx, { first: Math.max(0, Math.round(+i.value) || 0) })); return i; })()))),
      fig.el, VK.note('Saliency marks the pixels that would change the answer most if they changed (the gradient of the chosen class’s probability, worked out ' + (v.items[0] ? 'by ' + v.items[0].how : '') + ').' + (hasCam ? ' Grad-CAM weighs the last convolution’s feature maps by how much they push the class, and shows where their evidence lies.' : ' Grad-CAM needs a convolutional model: train a CNN to see it.'), 'muted'));
  }, 'saliency');
})(globalThis.Nooral = globalThis.Nooral || {});
