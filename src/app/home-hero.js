// The picture on the home page: a small workflow (dataset, split, train, evaluate) with two of its blocks opened
// up: what the trained model says at every place of the plane, and its confusion matrix on the test rows. It plays
// through k-nearest-neighbours with k from 1 to 61, so the boundary smooths and the matrix changes with it. Every
// value is computed here by the engines, on the two-moons set.
(function (S) {
  'use strict';
  const { h } = S.UI;
  const KS = [1, 3, 9, 25, 61];

  function compute() {
    const ds = S.Data.load('moons', { n: 220, noise: 0.28, seed: 7 }), sp = S.Data.split(ds.y, 0.3, 3, true);
    const tr = { X: sp.train.map((i) => ds.X[i]), y: sp.train.map((i) => ds.y[i]) }, te = { X: sp.test.map((i) => ds.X[i]), y: sp.test.map((i) => ds.y[i]) };
    return KS.map((k) => {
      const m = new S.ML.KNeighborsClassifier({ k }).fit(tr.X, tr.y);
      const pred = m.predict(te.X), cm = [[0, 0], [0, 0]];
      te.y.forEach((y, i) => { cm[y][pred[i]]++; });
      return { k, m, cm, acc: (cm[0][0] + cm[1][1]) / te.y.length, tr, te };
    });
  }

  /** the items of one frame: the workflow strip on top, the boundary under Train, the matrix under Evaluate */
  function frame(f, narrow) {
    const F = S.Kit.flow, C = S.Kit.curve, SC = S.Kit.scatter, HM = S.Kit.heat;
    const N = F.net('hh');
    const bw = narrow ? 128 : 112, bh = 40, y0 = 30;
    const xs = narrow ? [72, 236, 72, 236] : [62, 206, 350, 526], ys = narrow ? [y0, y0, y0 + 70, y0 + 70] : [y0, y0, y0, y0];
    const names = ['Dataset', 'Split', 'Train · k-NN', 'Evaluate'];
    names.forEach((n, i) => N.box('b' + i, xs[i], ys[i], bw, bh, n, i === 2 || i === 3 ? 'open' : ''));
    const wire = (a, b, pts) => N.add({ key: 'hh:w' + a + b, kind: 'wire', pts: [N.port('b' + a, pts ? 's' : 'e')].concat(pts || [], [N.port('b' + b, pts ? 'n' : 'w')]), head: true });
    if (narrow) { wire(0, 1); wire(1, 2, [[236, y0 + 47], [72, y0 + 47]]); wire(2, 3); }
    else { wire(0, 1); wire(1, 2); wire(2, 3); }
    const items = N.items.slice();
    // the boundary of the trained model, opened out of the Train block
    const box = narrow ? { x: 24, y: 150, w: 260, h: 190 } : { x: 40, y: 104, w: 300, h: 180 };
    const ax = C.axes(Object.assign({ prefix: 'hb', xr: [-1.6, 2.6], yr: [-1.1, 1.6], xticks: 0, yticks: 0, frame: true }, box));
    items.push(...ax.items.filter((it) => it.kind !== 'text'));
    items.push(SC.field(ax, 'hb:f', (u, v) => f.m.predictProba([[u, v]])[0][1], { n: narrow ? 46 : 64 }));
    items.push(SC.level(ax, 'hb:lv', SC.grid(ax, (u, v) => f.m.predictProba([[u, v]])[0][1], narrow ? 40 : 56), 0.5, ''));
    items.push(...SC.points(ax, 'hb:p', f.te.X, f.te.y, { r: 3.6 }));
    items.push({ key: 'hb:cap', kind: 'text', x: box.x + box.w / 2, y: box.y + box.h + 22, text: 'what the model says, k = ' + f.k, anchor: 'middle', size: 13, cls: 'strong' });
    if (!narrow) items.push({ key: 'hh:lens', kind: 'line', x1: xs[2], y1: y0 + bh / 2, x2: box.x + box.w - 10, y2: box.y, cls: 'lens', audit: 'ignore' });
    // the confusion matrix on the test rows, opened out of Evaluate
    const cw = 42, mx = narrow ? 112 : 486, my = narrow ? 392 : 130;
    const g = HM.grid({ prefix: 'hm', x: mx, y: my, cw, ch: cw, vals: f.cm, scheme: 'seq', max: Math.max(...f.cm.flat()), fmt: (v) => String(v), rows: ['blue', 'orange'], cols: ['blue', 'orange'], size: 14 });
    items.push(...g.items);
    items.push({ key: 'hm:t', kind: 'text', x: mx + cw + 1, y: my + 2 * cw + 2 + 26, text: Math.round(f.acc * 100) + '% of test rows right', anchor: 'middle', size: 13, cls: 'strong' });
    if (!narrow) items.push({ key: 'hh:lens2', kind: 'line', x1: xs[3], y1: y0 + bh / 2, x2: mx + cw + 1, y2: my - 26, cls: 'lens', audit: 'ignore' });
    return { items, view: narrow ? { x: 0, y: 0, w: 310, h: 525 } : { x: 0, y: 0, w: 610, h: 316 } };
  }

  /** S.HomeHero(host, caption element) -> {stop(), resized(), redraw()} */
  S.HomeHero = function (host, cap) {
    if (!S.Stage || !S.Kit || !S.Kit.flow || !S.ML || !S.ML.KNeighborsClassifier || !S.Data) return null;
    let frames;
    try { frames = compute(); } catch (e) { return null; }
    const wrap = h('div.hero-stage');
    host.appendChild(wrap);
    const stage = new S.Stage(wrap, { label: 'A workflow with its Train and Evaluate blocks opened up' });
    let i = 2, timer = null, narrow = null;
    const show = (instant) => {
      narrow = wrap.clientWidth < 480;
      stage.show(frame(frames[i], narrow), { instant, duration: 700 });
      if (cap) cap.textContent = 'k-nearest neighbours with k = ' + frames[i].k + ': ' + (frames[i].k === 1 ? 'every training point claims its own patch, so the boundary is ragged.' : frames[i].k >= 25 ? 'many neighbours vote, so the boundary is smooth.' : 'a few neighbours vote, so the boundary bends but stays calm.');
    };
    show(true);
    const reduced = (() => { try { return matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; } })();
    const tick = () => { i = (i + 1) % frames.length; show(false); timer = setTimeout(tick, 2600); };
    if (!reduced && !(S.app && S.app.settings.animate === false)) timer = setTimeout(tick, 2600);
    return {
      stop() { clearTimeout(timer); timer = null; },
      resized() { if ((wrap.clientWidth < 480) !== narrow) show(true); else stage.applyView && stage.applyView(); },
      redraw() { show(true); },
    };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
