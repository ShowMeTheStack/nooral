// The Train block's viewer, and the pictures of the linear and geometric families: the plane of any 2-D classifier,
// logistic and linear coefficients as signed bars, the ridge and lasso paths, the perceptron's line moving mistake by
// mistake, k-NN's neighbourhood of a point you click, the class ellipses of naive Bayes, LDA and QDA, the SVM's margin
// with its support vectors and multipliers, and SVR's ε-tube.
(function (S) {
  'use strict';
  const { h } = S.UI;
  const WF = S.WF, M = S.WMV, P = M.pics;
  const r2 = (v) => Math.round(v * 100) / 100;
  const K = () => S.Kit;

  // ---- the Train viewer ---------------------------------------------------------------------------------------------
  WF.viewers.train = M.mount((view, ctx) => {
    const fam = WF.families[view.family] || {}, s = view.trainScores || {};
    const done = !fam.iterative || (view.epoch >= view.epochs && !view.live);
    ctx.el.appendChild(M.title((fam.iterative ? (done ? 'Trained: ' : 'Training: ') : 'Trained: ') + view.label, view.task ? 'for ' + view.task : ''));
    const facts = [];
    if (s.accuracy !== undefined) facts.push(['training accuracy', M.pct(s.accuracy)]);
    if (s.f1 !== undefined && view.classes && view.classes.length === 2) facts.push(['training F1', M.num(s.f1)]);
    if (s.r2 !== undefined) facts.push(['training R²', M.num(s.r2)]);
    if (s.rmse !== undefined) facts.push(['training RMSE', M.num(s.rmse)]);
    if (view.ms !== undefined && !fam.iterative) facts.push(['fitted in', view.ms + ' ms']);
    if (view.internals && view.internals.nParams) facts.push(['parameters', view.internals.nParams.toLocaleString('en')]);
    if (facts.length) ctx.el.appendChild(M.facts(facts));
    if (fam.iterative && M.trainPanel) M.trainPanel(view, ctx);
    if (view.internals && view.internals.error) ctx.el.appendChild(M.note('The insides could not be read: ' + view.internals.error, 'bad'));
    const pic = P[view.family] || P.generic;
    pic(view, ctx);
    if (!fam.iterative) ctx.el.appendChild(M.note('These are scores on the rows it learned from. Wire it into an Evaluate block with rows it has not seen to know how it really does.'));
  });

  /** wide: the plane on the left and a side panel; narrow: one under the other. -> {plotBox, side: {x, w}, W} */
  const split = (ctx, sideW) => {
    if (!ctx.narrow && ctx.W >= 720 && sideW) { const pw = Math.min(520, ctx.W - sideW - 40); return { plot: M.plotBox(pw, { ratio: 0.78, maxH: 420 }), side: { x: pw + 20, w: ctx.W - pw - 30 } }; }
    return { plot: M.plotBox(ctx.W - 8, { ratio: 0.78, maxH: 380 }), side: null };
  };
  M.split = split;
  /** the plane, or the curve, or nothing: the picture any family has */
  P.generic = function (view, ctx, extra) {
    if (view.plane) {
      const L = split(ctx, 0), pl = M.plane('pl', view.plane, L.plot, extra);
      ctx.el.appendChild(M.legend(view) || h('span'));
      M.still(ctx.el, pl.items, M.viewOf(L.plot), 'What the model says everywhere in the plane');
      ctx.el.appendChild(M.sub(view.plane.scheme === 'prob' ? 'The shading is the probability of ' + M.cname(view, 1) + '; the line is where it is one half.' : view.plane.scheme === 'classes' ? 'Each colour is the class the model gives that place; paler where it is less sure.' : 'The shading is the predicted value.'));
      return pl;
    }
    if (view.line) return P.line(view, ctx);
    return null;
  };
  /** a regression model with one feature: the data and the fitted curve */
  P.line = function (view, ctx, more) {
    const ln = view.line, box = M.plotBox(ctx.W - 8, { ratio: 0.6, maxH: 340 });
    const ax = M.axes('ln', box, ln.xr, ln.yr, [ln.name, ln.target]);
    const items = ax.items.slice();
    ln.points.X.forEach((p, k) => items.push({ key: 'ln:p' + k, kind: 'dot', x: ax.X(p[0]), y: ax.Y(ln.points.y[k]), r: 3.4, cls: 'sp c0' + (more && more.sv && more.sv.has(k) ? ' sv' : ''), audit: 'overlap-ok', layer: 'items' }));
    if (more && more.items) items.push(...more.items(ax));
    items.push(K().curve.line(ax, 'ln:fit', ln.xs.map((x, i) => [x, ln.ys[i]]), 'act'));
    M.still(ctx.el, items, M.viewOf(box), 'The fitted curve over the data');
    return { ax };
  };

  // ---- coefficients ---------------------------------------------------------------------------------------------
  const coefBars = (view, ctx, where) => {
    const I = view.internals;
    if (!I || !I.coef) return;
    const multi = I.coef.length > 1;
    const rows = multi ? I.coef : [I.coef[0]];
    rows.forEach((w, r) => {
      const names = I.features.slice(0, 24), vals = w.slice(0, 24);
      const x = where ? where.x : 8, wd = where ? where.w : ctx.W - 16;
      const b = M.bars('cf' + r, names, vals, { x: 0, y: 26, w: wd, size: 13 });
      const title = multi ? 'weights for ' + M.cname(view, r) : view.task === 'classification' ? 'weights (towards ' + M.cname(view, 1) + ')' : 'weights';
      b.items.push({ key: 'cf' + r + ':t', kind: 'text', x: 0, y: 14, text: title + ' · intercept ' + M.num(I.intercept[r] !== undefined ? I.intercept[r] : I.intercept[0]), size: 13, cls: 'axname' });
      M.still(ctx.el, b.items, { x: -4, y: 0, w: wd + 8, h: b.h + 36 }, 'The weight of each feature');
      void x;
    });
    if (I.features.length > 24) ctx.el.appendChild(M.sub('The first 24 of ' + I.features.length + ' weights.'));
  };
  P.logistic = function (view, ctx) { P.generic(view, ctx); coefBars(view, ctx); ctx.el.appendChild(M.sub('A positive weight pushes towards the class named; the bigger the bar, the stronger the push for each unit of the feature.')); };
  P.linreg = function (view, ctx) { const ln = view.internals && view.internals.leastNorm; if (ln) ctx.el.appendChild(M.note('There are ' + ln.cols + ' features but only ' + ln.rows + ' rows, so many sets of weights fit the training rows exactly. This is the one with the smallest weights (ridge with a vanishing penalty); expect it to fit the training rows perfectly and new rows poorly.')); if (view.line) P.line(view, ctx); coefBars(view, ctx); ctx.el.appendChild(M.sub('Each weight is what one more unit of the feature adds to the prediction, the others held still.')); };
  const withPath = (word) => function (view, ctx) {
    if (view.line) P.line(view, ctx);
    coefBars(view, ctx);
    const pa = view.path;
    if (!pa || !pa.alphas.length) return;
    const box = M.plotBox(ctx.W - 8, { ratio: 0.5, maxH: 300, right: ctx.narrow ? 20 : 130, top: 50 });
    const lx = pa.alphas.map((a) => -Math.log10(a));
    const series = view.internals.features.map((f, j) => ({ key: j, pts: lx.map((x, k) => [x, pa.coefs[k][j]]), cls: 'c' + (j % 4) + ' thin' }));
    const la = -Math.log10(pa.at), xr = [Math.min(la, ...lx), Math.max(la, ...lx)];
    const sp = M.series('pa', box, series, { xr, names: ['−log₁₀ α (weaker to the right)', 'weight'] });
    sp.items.push(K().curve.vline(sp.ax, 'pa:at', la, 'act'));
    if (!ctx.narrow) {
      // name each line at its weakest-penalty end, spread so the names never touch
      const ends = view.internals.features.map((f, j) => ({ j, f, y: sp.ax.Y(pa.coefs[lx.indexOf(Math.max(...lx))][j]) })).sort((a, b) => a.y - b.y);
      let last = -1e9;
      ends.slice(0, 10).forEach((e) => { const y = Math.max(e.y, last + 16); last = y; sp.items.push({ key: 'pa:n' + e.j, kind: 'text', x: box.x + box.w + 8, y: r2(y + 4), text: e.f, size: 12, cls: 'legend c' + (e.j % 4) }); });
    }
    sp.items.push({ key: 'pa:t', kind: 'text', x: box.x, y: 14, text: ctx.narrow ? 'the ' + word + ' path (the line is your α)' : 'the ' + word + ' path: every weight as the penalty changes (the line is your α)', size: 13, cls: 'axname' });
    M.still(ctx.el, sp.items, { x: 0, y: 0, w: box.x + box.w + (ctx.narrow ? 20 : 130), h: box.y + box.h + 46 }, 'Weights against the penalty');
    ctx.el.appendChild(M.sub(word === 'lasso' ? 'Read from left to right: as α weakens, weights leave zero one at a time. The lasso can set a weight to exactly zero; ridge only shrinks.' : 'Ridge shrinks every weight smoothly towards zero as α grows, but never to exactly zero.'));
  };
  P.ridge = withPath('ridge');
  P.lasso = withPath('lasso');

  // ---- the perceptron: the line after every mistake -----------------------------------------------------------------------
  P.perceptron = function (view, ctx) {
    const I = view.internals, pl = view.plane;
    if (!pl || !I.updates || !I.updates.length) { P.generic(view, ctx); coefBars(view, ctx); return; }
    const L = split(ctx, 0), box = L.plot;
    const base = M.plane('pc', pl, box, { noLevels: true, noPoints: false, vals: pl.vals.map(() => 0.5), scheme: 'prob' });
    const lineOf = (w, b) => {
      const [x0, x1] = pl.xr, [y0, y1] = pl.yr, pts = [];
      if (Math.abs(w[1]) > 1e-12) for (const x of [x0, x1]) { const y = -(w[0] * x + b) / w[1]; if (y >= y0 && y <= y1) pts.push([x, y]); }
      if (Math.abs(w[0]) > 1e-12) for (const y of [y0, y1]) { const x = -(w[1] * y + b) / w[0]; if (x >= x0 && x <= x1) pts.push([x, y]); }
      return pts.length >= 2 ? pts.slice(0, 2) : null;
    };
    const ups = I.updates.filter((u) => u.head === undefined || u.head === I.updates[0].head).slice(0, 40), pos = new Map(pl.points.idx.map((i, k) => [i, k]));
    const frames = ups.map((u, k) => {
      const items = base.items.slice(), seg = lineOf(u.w, u.b);
      if (seg) items.push({ key: 'pc:line', kind: 'path', d: `M${base.ax.X(seg[0][0])},${base.ax.Y(seg[0][1])} L${base.ax.X(seg[1][0])},${base.ax.Y(seg[1][1])}`, cls: 'level boundary', pathAudit: 'edge-x', layer: 'lines' });
      const pk = pos.get(u.i);
      if (pk !== undefined) items.push(K().scatter.ring(base.ax, 'pc:mis', pl.points.X[pk][0], pl.points.X[pk][1], 'hot', 9));
      return { items, view: M.viewOf(box), title: 'pass ' + (u.epoch + 1), say: 'Mistake ' + (k + 1) + ', on row `' + u.i + '` in pass ' + (u.epoch + 1) + ': the weights become `' + u.w.map(M.num).join(', ') + '` and the bias `' + M.num(u.b) + '`, turning the line towards it.' };
    });
    const fin = M.plane('pc', pl, box, {});
    frames.push({ items: fin.items, view: M.viewOf(box), title: 'the end', say: 'After ' + I.epochs.length + ' passes: ' + I.epochs.map((e) => e.mistakes).join(', ') + ' mistakes in each. The shading is the side each place falls on.' });
    ctx.el.appendChild(M.legend(view) || h('span'));
    M.stepper(ctx.el, frames, { label: 'The perceptron learning', key: 'pc', mem: ctx.mem, start: frames.length - 1 });
  };

  // ---- k nearest neighbours: the neighbourhood of a point ----------------------------------------------------------------
  P.knn = function (view, ctx) {
    const pl = view.plane;
    if (!pl) { ctx.el.appendChild(M.note('k-NN keeps the ' + (view.internals ? view.internals.n : '') + ' training rows and, for a new row, asks the ' + (view.params.k) + ' nearest. With two features the plane shows its regions.')); return P.generic(view, ctx); }
    const q = ctx.mem.query || pl.query, L = split(ctx, 0), box = L.plot;
    const draw = (qq) => {
      const near = new Set(qq.all), byRow = new Map(pl.points.idx.map((i, k) => [i, k]));
      const pic = M.plane('kn', pl, box, { pointCls: (k) => (near.has(pl.points.idx[k]) ? 'near' : '') });
      const qx = pic.ax.X(qq.x[0]), qy = pic.ax.Y(qq.x[1]);
      qq.all.forEach((i, j) => { const k = byRow.get(i); if (k === undefined) return; pic.items.push({ key: 'kn:l' + j, kind: 'path', d: `M${r2(qx)},${r2(qy)} L${pic.ax.X(pl.points.X[k][0])},${pic.ax.Y(pl.points.X[k][1])}`, cls: 'nbr', pathAudit: 'edge-x', layer: 'lines' }); });
      const rad = Math.max(...qq.all.map((i) => { const k = byRow.get(i); return k === undefined ? 0 : Math.hypot(pic.ax.X(pl.points.X[k][0]) - qx, pic.ax.Y(pl.points.X[k][1]) - qy); }));
      pic.items.push({ key: 'kn:circle', kind: 'path', d: `M${r2(qx - rad)},${r2(qy)} a${r2(rad)},${r2(rad)} 0 1 0 ${r2(2 * rad)},0 a${r2(rad)},${r2(rad)} 0 1 0 ${r2(-2 * rad)},0`, cls: 'nbr ring', pathAudit: 'edge-x', layer: 'lines' });
      pic.items.push({ key: 'kn:q', kind: 'dot', x: r2(qx), y: r2(qy), r: 7, cls: 'query', audit: 'overlap-ok', layer: 'top' });
      return pic;
    };
    ctx.el.appendChild(M.legend(view) || h('span'));
    const pic = draw(q);
    const st = M.still(ctx.el, pic.items, M.viewOf(box), 'k nearest neighbours of a point');
    const counts = (qq) => (view.classes || []).map((c, i) => qq.y.filter((y) => y === i).length);
    const say = h('div.say.wfm-say');
    const tell = (qq) => { say.innerHTML = ''; say.append(...M.say('The point (`' + qq.x.map(M.num).join(', ') + '`) asks its `' + qq.all.length + '` nearest neighbours: ' + counts(qq).map((n, i) => (n ? '`' + n + '` of ' + (/^\d+$/.test(M.cname(view, i)) ? 'class ' : '') + M.cname(view, i) : '')).filter(Boolean).join(', ') + ', so it is called `' + M.cname(view, qq.pred) + '`. Click anywhere in the plane to ask about another point.')); };
    tell(q);
    ctx.el.appendChild(say);
    st.wrap.classList.add('wfm-pick');
    st.stage.root.addEventListener('click', async (e) => {
      const pt = st.stage.root.createSVGPoint ? st.stage.root.createSVGPoint() : null;
      if (!pt || !ctx.api || !ctx.api.call) return;
      pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(st.stage.root.getScreenCTM().inverse()), x = [+pic.ax.U(p.x).toPrecision(4), +pic.ax.V(p.y).toPrecision(4)];
      try { const qq = await ctx.api.call('neighbours', { x }); ctx.mem.query = qq; st.stage.show({ items: draw(qq).items, view: M.viewOf(box) }, { duration: 300 }); tell(qq); } catch (err) { /* the model is gone */ }
    });
  };

  // ---- class ellipses: naive Bayes, LDA, QDA ---------------------------------------------------------------------
  const ellipses = (view, ctx, covOf, word) => {
    const I = view.internals, pl = view.plane;
    if (!pl || !I || !I.means) return P.generic(view, ctx);
    const L = split(ctx, 0), pic = M.plane('el', pl, L.plot, {});
    I.means.forEach((m, c) => {
      const cov = covOf(c);
      for (const k of [1, 2]) pic.items.push({ key: 'el:e' + c + '_' + k, kind: 'path', d: M.ellipse(pic.ax, m, cov, k), cls: 'cov c' + (c % 4) + (k === 2 ? ' far' : ''), pathAudit: 'edge-x', layer: 'lines' });
      pic.items.push({ key: 'el:m' + c, kind: 'dot', x: pic.ax.X(m[0]), y: pic.ax.Y(m[1]), r: 6, cls: 'mean c' + (c % 4), audit: 'overlap-ok', layer: 'top' });
    });
    ctx.el.appendChild(M.legend(view) || h('span'));
    M.still(ctx.el, pic.items, M.viewOf(L.plot), 'The class ellipses');
    ctx.el.appendChild(M.sub('Each class as the model sees it: its mean (the large dot) and the ellipses one and two standard deviations out. ' + word));
    const rows = I.means.map((m, c) => h('tr', h('th', M.swatch(c), ' ', M.cname(view, c)), h('td.mono', I.priors ? M.num(I.priors[c]) : ''), h('td.mono', m.map(M.num).join(', '))));
    ctx.el.appendChild(h('div.wfm-tablebox', h('table.wfm-table', h('thead', h('tr', h('th', 'class'), h('th', 'prior'), h('th', 'mean'))), h('tbody', ...rows))));
  };
  P.naivebayes = function (view, ctx) {
    const I = view.internals;
    if (I && I.variant === 'multinomial') {
      ctx.el.appendChild(M.sub('The words (or counts) each class makes most likely:'));
      const rows = (I.top || []).map((list, c) => h('tr', h('th', M.swatch(c), ' ', M.cname(view, c)), h('td', list.map(([w, p]) => w + ' ' + M.num(p, 2)).join(' · '))));
      ctx.el.appendChild(h('div.wfm-tablebox', h('table.wfm-table', h('tbody', ...rows))));
      return P.generic(view, ctx);
    }
    ellipses(view, ctx, (c) => [[I.vars[c][0], 0], [0, I.vars[c][1]]], 'Naive Bayes treats the features as independent within a class, so its ellipses never tilt.');
  };
  P.lda = (view, ctx) => ellipses(view, ctx, () => view.internals.shared, 'LDA gives every class the same ellipse, which is why its boundaries are straight.');
  P.qda = (view, ctx) => ellipses(view, ctx, (c) => view.internals.covs[c], 'QDA lets each class have its own ellipse, so the boundaries bend.');

  // ---- support vector machines ------------------------------------------------------------------------------------
  P.svm = function (view, ctx) {
    const I = view.internals, pl = view.plane;
    if (!pl || !pl.f) { P.generic(view, ctx); ctx.el.appendChild(M.facts([['support vectors', I.support.length + ' of ' + I.n], ['kernel', I.kernel], ['C', M.num(I.C)], ['γ', M.num(I.gamma)]])); return; }
    const L = split(ctx, 260), sv = pl.sv || [];
    const pic = M.plane('sv', pl, L.plot, { rings: sv.map(([k, a]) => [k, a < I.C - 1e-6 ? 'act' : '']) });
    pic.items = pic.items.filter((x) => x.kind !== 'raster').concat([Object.assign({}, pic.items.find((x) => x.kind === 'raster'), { scheme: 'prob', vals: pl.score.map((f) => 1 / (1 + Math.exp(-1.6 * f))), max: undefined })]);
    let view2 = M.viewOf(L.plot);
    // the multipliers α beside the plot (below it on a phone): one bar per support vector, against the cap C
    const A = L.side ? { x: L.side.x, y: L.plot.y + 60, w: L.side.w, h: 150 } : { x: 50, y: L.plot.y + L.plot.h + 104, w: ctx.W - 80, h: 110 };
    const alphas = sv.map(([, a]) => a).sort((a, b) => b - a), bw = Math.max(1, Math.min(14, (A.w - 4) / Math.max(1, alphas.length)));
    pic.items.push({ key: 'al:t', kind: 'text', x: A.x, y: A.y - 34, text: 'the multipliers α of the ' + alphas.length + ' support vectors', size: 13, cls: 'axname' });
    pic.items.push({ key: 'al:base', kind: 'path', d: `M${A.x},${A.y + A.h} h${A.w}`, cls: 'axis', audit: 'ignore', layer: 'back' });
    pic.items.push({ key: 'al:cap', kind: 'path', d: `M${A.x},${A.y} h${A.w}`, cls: 'guide act', audit: 'ignore', layer: 'back' });
    pic.items.push({ key: 'al:capn', kind: 'text', x: A.x + A.w, y: A.y - 8, text: 'cap C = ' + M.num(I.C), anchor: 'end', size: 12, cls: 'legend' });
    alphas.forEach((a, k) => { const hh = Math.max(1.5, (Math.min(a, I.C) / I.C) * A.h); pic.items.push({ key: 'al:b' + k, kind: 'cell', x: r2(A.x + 2 + k * bw), y: r2(A.y + A.h - hh), w: Math.max(1, bw - 1), h: r2(hh), rx: 1, text: '', cls: 'bar' + (a < I.C - 1e-6 ? '' : ' soft'), audit: 'ignore' }); });
    const atCap = alphas.filter((a) => a >= I.C - 1e-6).length;
    pic.items.push({ key: 'al:read', kind: 'text', x: A.x, y: A.y + A.h + 22, text: (alphas.length - atCap) + ' on the margin (0 < α < C), ' + atCap + ' at the cap', size: 13, cls: 'legend' });
    view2 = { x: 0, y: 0, w: Math.max(view2.w, A.x + A.w + 16), h: Math.max(view2.h, A.y + A.h + 40) };
    ctx.el.appendChild(M.facts([['support vectors', I.support.length + ' of ' + I.n], ['kernel', I.kernel], ['C', M.num(I.C)], I.kernel !== 'linear' ? ['γ', M.num(I.gamma)] : null, ['SMO steps', I.smoSteps]]));
    M.still(ctx.el, pic.items, view2, 'The margin and the support vectors');
    ctx.el.appendChild(M.sub('Shading: the decision value f(x), deeper the further from the boundary, in the colour of the class it gives. Solid line: f = 0, the boundary; dashed: f = ±1, the margins. Ringed: the support vectors (coloured rings sit on the margin, grey ones inside it or on the wrong side).'));
  };
  P.svr = function (view, ctx) {
    const I = view.internals, ln = view.line;
    if (!ln) return P.generic(view, ctx);
    const eps = ln.eps;
    P.line(view, ctx, { sv: new Set(ln.sv || []), items: (ax) => [K().curve.line(ax, 'ln:up', ln.xs.map((x, i) => [x, ln.ys[i] + eps]), 'tube'), K().curve.line(ax, 'ln:dn', ln.xs.map((x, i) => [x, ln.ys[i] - eps]), 'tube')] });
    ctx.el.appendChild(M.facts([['support vectors', I.support.length + ' of ' + I.n], ['ε', M.num(eps)], ['C', M.num(I.C)], ['kernel', I.kernel]]));
    ctx.el.appendChild(M.sub('The dashed lines are the ε-tube: points inside it cost nothing. The ringed points, on or outside it, are the support vectors that hold the curve in place.'));
  };
})(globalThis.Nooral = globalThis.Nooral || {});
