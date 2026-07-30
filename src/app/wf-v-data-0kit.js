// What the viewers of the data, prepare and unsupervised blocks share (page side): a mount that builds the viewer for
// the width it actually has (stage units are pixels, so text is drawn at its true size), rebuilds when that width
// changes or the theme does, and a stepper (first, back, play, next, a slider and a sentence for each step) for the
// viewers that replay an algorithm. Also: facts as pills, small tables, notes, and the plot helpers on top of the kits.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const V = (S.WFV = S.WFV || {});
  const WF = S.WF;
  const r2 = (v) => Math.round(v * 100) / 100;
  V.PHONE = 600;
  V.num = (x, d) => (x === null || x === undefined || !isFinite(x) ? '—' : String(+(+x).toPrecision(d || 3)).replace(/^-/, '−'));
  /** a tick label: whole numbers with thousands marked when large */
  V.tick = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en') : V.num(v, 3));
  V.pct = (x) => V.num(100 * x, 3) + '%';
  V.className = (view, c) => (view.classes && view.classes[c] !== undefined ? String(view.classes[c]) : String(c));
  /** text with `values` in the monospace face */
  V.say = (txt) => String(txt || '').split('`').map((part, i) => (i % 2 ? h('span.v', part) : document.createTextNode(part)));

  /**
   * S.WFV.viewer(build): a viewer function for S.WF.viewers. build(view, ctx) fills ctx.el (a div) and may return
   * {dispose()}; ctx = {W (px), narrow, api, el, stage(opts) -> a stepper, live}
   */
  V.viewer = (build) => (host, view, api) => {
    const root = h('div.wfd');
    host.appendChild(root);
    let cur = view, built = null, lastW = 0, live = false;
    const make = () => {
      if (built && built.dispose) try { built.dispose(); } catch (e) { /* gone */ }
      root.innerHTML = '';
      const W = Math.max(280, Math.floor(root.clientWidth || host.clientWidth || 640));
      lastW = W;
      if (!cur) { root.appendChild(h('p.muted', 'Nothing to show yet.')); return; }
      if (cur.error) { root.appendChild(h('div.wfd-note.bad', icon('warn'), h('span', cur.error))); return; }
      const ctx = { W, narrow: W < V.PHONE, api, el: root, live };
      try { built = build(cur, ctx) || null; } catch (e) { root.innerHTML = ''; root.appendChild(h('div.wfd-note.bad', icon('warn'), h('span', 'This view could not be drawn: ' + (e.message || e)))); if (typeof console !== 'undefined') console.error(e); }
    };
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { const w = Math.floor(root.clientWidth); if (w && Math.abs(w - lastW) > 24) make(); }) : null;
    if (ro) ro.observe(root);
    requestAnimationFrame(make);
    make();
    return {
      update(v, isLive) { cur = v; live = !!isLive; make(); },
      redraw() { make(); },
      dispose() { if (ro) ro.disconnect(); if (built && built.dispose) built.dispose(); root.remove(); },
    };
  };

  // ---- pieces of HTML ---------------------------------------------------------------------------------------------
  /** facts as pills: [[label, value], ...] */
  V.facts = (list) => h('div.wfd-facts', ...list.filter(Boolean).map(([k, v, cls]) => h('span.wfd-fact' + (cls ? '.' + cls : ''), h('span.k', k), ' ', h('b', String(v)))));
  V.note = (text, cls) => h('div.wfd-note' + (cls ? '.' + cls : ''), icon(cls === 'bad' ? 'warn' : cls === 'good' ? 'check' : 'info'), h('span', ...V.say(text)));
  V.title = (text, sub) => h('div.wfd-title', h('b', text), sub ? h('span.muted', ' ' + sub) : null);
  /** a small table: head [..], rows [[..]..]; cls(i, j) adds classes to a cell */
  V.table = function (head, rows, opts) {
    opts = opts || {};
    const t = h('table.wfd-table' + (opts.cls ? '.' + opts.cls : ''), head ? h('thead', h('tr', ...head.map((x) => h('th', String(x))))) : null,
      h('tbody', ...rows.map((r, i) => h('tr', ...r.map((x, j) => { const c = opts.cell ? opts.cell(i, j) : ''; return h('td' + (c ? '.' + c : ''), x instanceof Node ? x : String(x === null || x === undefined ? '' : x)); })))));
    const box = h('div.wfd-tablebox', { 'data-scroll-affordance': '1' }, t);
    return box;
  };
  /** a swatch of class c, as HTML */
  V.swatch = (c) => h('span.wfd-sw.c' + (c < 0 ? 'n' : c % 8));

  // ---- the stepper: a stage with its controls --------------------------------------------------------------------
  /**
   * V.stepper(parent, frames, {label, start}) frames: [{items, view: {x, y, w, h}, say, title}] -> {go(i), stop(), stage}
   * One frame shows a still picture with no controls.
   */
  V.stepper = function (parent, frames, opts) {
    opts = opts || {};
    const wrap = h('div.stage-wrap.wfd-stage');
    parent.appendChild(wrap);
    const stage = new S.Stage(wrap, { label: opts.label || 'A picture of the block' });
    let H = 0;
    for (const f of frames) if (f.view) H = Math.max(H, f.view.h);
    wrap.style.height = Math.round(H || 300) + 'px';
    const sayEl = frames.some((f) => f.say) ? h('div.say.wfd-say', { 'aria-live': 'polite' }) : null;
    let i = 0, timer = null;
    const anim = () => !(S.app && S.app.settings.animate === false);
    const btn = (name, label, fn, cls) => { const b = h('button.icon-btn' + (cls ? '.' + cls : ''), { type: 'button', title: label, 'aria-label': label }, icon(name)); b.addEventListener('click', fn); return b; };
    const one = frames.length < 2;
    let bPrev, bNext, bPlay, bFirst, range, pos;
    const go = (k, instant) => {
      i = Math.max(0, Math.min(frames.length - 1, k));
      const f = frames[i];
      stage.show(f, { instant: instant || !anim(), duration: opts.duration || 550 });
      if (sayEl) { sayEl.innerHTML = ''; sayEl.append(...(frames.length > 1 ? [h('span.step', 'Step ' + (i + 1) + ' of ' + frames.length + (f.title ? ' · ' + f.title : ''))] : []), ...V.say(f.say)); }
      if (!one) { range.value = i; pos.textContent = i + 1 + ' / ' + frames.length; bPrev.disabled = bFirst.disabled = i === 0; bNext.disabled = i === frames.length - 1; }
      if (opts.onStep) opts.onStep(i);
    };
    const stop = () => { clearTimeout(timer); timer = null; if (bPlay) { bPlay.replaceChildren(icon('play')); bPlay.title = 'Play'; bPlay.setAttribute('aria-label', 'Play'); } };
    const play = () => {
      if (i >= frames.length - 1) go(0);
      const tick = () => { if (i >= frames.length - 1) { stop(); return; } go(i + 1); timer = setTimeout(tick, opts.hold || 900); };
      timer = setTimeout(tick, 350);
      bPlay.replaceChildren(icon('pause')); bPlay.title = 'Pause'; bPlay.setAttribute('aria-label', 'Pause');
    };
    if (!one) {
      bFirst = btn('first', 'Back to the start', () => { stop(); go(0); });
      bPrev = btn('prev', 'Step back', () => { stop(); go(i - 1); });
      bPlay = btn('play', 'Play', () => (timer ? stop() : play()), 'play');
      bNext = btn('next', 'Step forward', () => { stop(); go(i + 1); });
      range = h('input', { type: 'range', min: 0, max: frames.length - 1, value: 0, 'aria-label': 'Step' });
      range.addEventListener('input', () => { stop(); go(+range.value, true); });
      pos = h('span.pos');
      parent.appendChild(h('div.controls.wfd-controls', bFirst, bPrev, bPlay, bNext, range, pos));
    }
    if (sayEl) parent.appendChild(sayEl);
    go(opts.start === undefined ? frames.length - 1 : opts.start, true);
    return { go, stop, stage, get i() { return i; }, dispose: stop };
  };

  // ---- plots ------------------------------------------------------------------------------------------------------
  /** a padded range around values */
  V.range = function (vals, pad) {
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const p = (hi - lo) * (pad === undefined ? 0.06 : pad);
    return [lo - p, hi + p];
  };
  /** plot axes fitted to points P [[u, v]..] inside a box, with names */
  V.axes = function (prefix, P, box, names, extra) {
    const xr = V.range(P.map((p) => p[0])), yr = V.range(P.map((p) => p[1]));
    const ticksX = Math.max(2, Math.round(box.w / 90)), ticksY = Math.max(2, Math.round(box.h / 70));
    return S.Kit.curve.axes(Object.assign({ prefix, x: box.x, y: box.y, w: box.w, h: box.h, xr, yr, xticks: S.Kit.curve.ticks(xr[0], xr[1], ticksX), yticks: S.Kit.curve.ticks(yr[0], yr[1], ticksY), xname: names && names[0], yname: names && names[1] }, extra || {}));
  };
  /** the box of a scatter plot that fills a width, with room for the ticks and the axis names */
  V.plotBox = (W, Hmax, left) => { const x = left || 46, w = Math.max(160, W - x - 18), hh = Math.min(Hmax || 420, Math.max(200, w * 0.72)); return { x, y: 30, w, h: hh }; };
  /** class dots: c<k> for a class, cn for noise (k < 0) */
  V.dots = function (ax, key, P, labels, opts) {
    opts = opts || {};
    const r = opts.r || (P.length > 400 ? 3.2 : 4);
    return P.map((p, i) => ({ key: key + ':' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: opts.rOf ? opts.rOf(i) : r, cls: 'sp ' + (labels ? (labels[i] < 0 ? 'cn' : 'c' + (labels[i] % 8)) : 'c0') + (opts.cls ? ' ' + (opts.cls(i) || '') : ''), audit: 'overlap-ok', layer: 'items' }));
  };
  /** a legend of classes or clusters: entries [{text, c}] laid along a row that wraps at width W */
  V.legend = function (key, x, y, entries, W) {
    const items = [];
    let cx = x, cy = y;
    entries.forEach((e, i) => {
      const w = 24 + S.geom.textWidth(e.text, 13) + 16;
      if (cx + w > x + W && cx > x) { cx = x; cy += 22; }
      items.push({ key: key + ':s' + i, kind: 'dot', x: r2(cx + 6), y: r2(cy - 4.5), r: 5.5, cls: 'swatch ' + (e.c < 0 ? 'cn' : 'c' + (e.c % 8)) });
      items.push({ key: key + ':n' + i, kind: 'text', x: r2(cx + 17), y: r2(cy), text: e.text, size: 13, cls: 'legend' });
      cx += w;
    });
    return { items, h: cy - y + 10 };
  };
  /** a little histogram: bars along a box, the range named under it */
  V.hist = function (key, box, counts, lo, hi, cls) {
    const items = [], n = counts.length, mx = Math.max(1, ...counts), bw = box.w / n;
    counts.forEach((c, i) => { if (!c) return; const hh = Math.max(1.5, (c / mx) * box.h); items.push({ key: key + ':b' + i, kind: 'cell', x: r2(box.x + i * bw + 0.5), y: r2(box.y + box.h - hh), w: r2(Math.max(1, bw - 1)), h: r2(hh), rx: 1, cls: 'bar ' + (cls || 'soft'), audit: 'overlap-ok' }); });
    items.push({ key: key + ':base', kind: 'path', d: `M${r2(box.x)},${r2(box.y + box.h)} h${r2(box.w)}`, cls: 'axis', audit: 'ignore' });
    if (lo !== undefined) {
      items.push({ key: key + ':lo', kind: 'text', x: r2(box.x), y: r2(box.y + box.h + 15), text: V.num(lo), size: 12, cls: 'tick' });
      items.push({ key: key + ':hi', kind: 'text', x: r2(box.x + box.w), y: r2(box.y + box.h + 15), text: V.num(hi), size: 12, anchor: 'end', cls: 'tick' });
    }
    return items;
  };
  V.text = (key, x, y, text, o) => Object.assign({ key, kind: 'text', x: r2(x), y: r2(y), text, size: 13 }, o || {});
  /** fit a string to a width at a size, cutting it with an ellipsis */
  V.fit = (t, w, size, bold) => { t = String(t); const tw = (x) => S.geom.textWidth(x, size || 13, { bold: !!bold }); if (tw(t) <= w) return t; while (t.length > 1 && tw(t + '…') > w) t = t.slice(0, -1); return t + '…'; };
  /** a frame's view: the whole width, the given height */
  V.view = (W, H) => ({ x: 0, y: 0, w: W, h: H });

  /** register a viewer for several block types */
  V.register = (types, fn) => { for (const t of [].concat(types)) WF.viewers[t] = fn; };
})(globalThis.Nooral = globalThis.Nooral || {});
