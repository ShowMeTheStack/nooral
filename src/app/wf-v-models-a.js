// The viewers of the Models blocks and of Train, page side: what they share (S.WMV). A viewer is mounted in the
// canvas's panel and built for the width it really has (stage units are pixels, so text is drawn at its true size);
// it is built again when that width, the theme or the view changes, keeping its place (a stepper's step). The
// pictures of the model families register in S.WMV.pics[family](view, ctx) and draw with the kits: the plane of a
// 2-D classifier (its probability as shading, the boundary, the points), a regression curve, coefficient bars,
// class ellipses and a stepper that replays a fit step by step.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const WF = S.WF;
  const M = (S.WMV = S.WMV || {});
  const r2 = (v) => Math.round(v * 100) / 100;
  M.PHONE = 600;
  M.pics = M.pics || {};
  M.num = (x, d) => (x === null || x === undefined || !isFinite(x) ? '—' : String(+(+x).toPrecision(d || 3)).replace(/^-/, '−'));
  M.pct = (x) => M.num(100 * x, 3) + '%';
  M.cname = (view, c) => (view.classes && view.classes[c] !== undefined ? String(view.classes[c]) : String(c));
  M.say = (txt) => String(txt || '').split('`').map((part, i) => (i % 2 ? h('span.v', part) : document.createTextNode(part)));
  const K = () => S.Kit;

  /**
   * M.mount(build): a viewer for S.WF.viewers. build(view, ctx) fills ctx.el; ctx = {W, narrow, api, el, live, mem}
   * (mem lasts as long as the viewer: a stepper's place, the curve points that arrived since the last view)
   */
  M.mount = (build) => (host, view, api) => {
    const root = h('div.wfm');
    host.appendChild(root);
    const mem = { disposers: [] };
    let cur = view, lastW = 0, live = false, made = null;
    const make = () => {
      for (const d of mem.disposers.splice(0)) try { d(); } catch (e) { /* gone */ }
      root.innerHTML = '';
      const W = Math.max(280, Math.floor(root.clientWidth || host.clientWidth || 640));
      lastW = W;
      if (!cur) { root.appendChild(h('p.muted', 'Nothing to show yet: the block has not run.')); return; }
      if (cur.error) { root.appendChild(M.note(cur.error, 'bad')); return; }
      const ctx = { W, narrow: W < M.PHONE, api, el: root, live, mem, view: cur };
      try { made = build(cur, ctx) || null; } catch (e) { root.innerHTML = ''; root.appendChild(M.note('This picture could not be drawn: ' + (e.message || e), 'bad')); if (typeof console !== 'undefined') console.error(e); }
    };
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { const w = Math.floor(root.clientWidth); if (w && Math.abs(w - lastW) > 24) make(); }) : null;
    if (ro) ro.observe(root);
    make();
    if (!lastW || lastW === 640) requestAnimationFrame(() => { if (Math.abs(Math.floor(root.clientWidth) - lastW) > 24) make(); });
    return {
      update(v, isLive) { cur = v; live = !!isLive; if (made && made.update && made.update(v, isLive)) return; make(); },
      redraw() { make(); },
      dispose() { if (ro) ro.disconnect(); for (const d of mem.disposers.splice(0)) try { d(); } catch (e) { /* gone */ } if (mem.off) mem.off(); root.remove(); },
      get mem() { return mem; },
    };
  };

  // ---- HTML pieces ----------------------------------------------------------------------------------------------
  M.facts = (list) => h('div.wfm-facts', ...list.filter(Boolean).map(([k, v, cls]) => h('span.wfm-fact' + (cls ? '.' + cls : ''), h('span.k', k), ' ', h('b', String(v)))));
  M.note = (text, cls) => h('div.wfm-note' + (cls ? '.' + cls : ''), icon(cls === 'bad' ? 'warn' : cls === 'good' ? 'check' : 'info'), h('span', ...M.say(text)));
  M.title = (text, sub) => h('div.wfm-title', h('b', text), sub ? h('span.muted', ' ' + sub) : null);
  M.sub = (text) => h('div.wfm-sub', text);
  M.swatch = (c) => h('span.wfm-sw.c' + (c % 4));
  M.legend = (view) => (view.classes && view.classes.length <= 8 ? h('div.wfm-legend', ...view.classes.map((n, c) => h('span', M.swatch(c), String(n)))) : null);

  // ---- a stage, with or without steps ------------------------------------------------------------------------------
  /**
   * M.stepper(parent, frames, {label, start, key, mem, hold}) frames: [{items, view, say, title}]; one frame shows a still
   * picture. key and mem keep the step across rebuilds.
   */
  M.stepper = function (parent, frames, opts) {
    opts = opts || {};
    const wrap = h('div.stage-wrap.wfm-stage');
    parent.appendChild(wrap);
    const stage = new S.Stage(wrap, { label: opts.label || 'A picture of the model' });
    let H = 0, Wd = 0;
    for (const f of frames) if (f.view) { H = Math.max(H, f.view.y + f.view.h); Wd = Math.max(Wd, f.view.x + f.view.w); }
    // one camera for every step, so nothing jumps and a shorter step sits at the top
    for (const f of frames) if (f.view) f.view = { x: f.view.x, y: f.view.y, w: Wd - f.view.x, h: H - f.view.y };
    wrap.style.height = Math.round(H || 300) + 'px';
    const sayEl = frames.some((f) => f.say) ? h('div.say.wfm-say', { 'aria-live': 'polite' }) : null;
    let i = 0, timer = null;
    const anim = () => !(S.app && S.app.settings.animate === false);
    const btn = (name, label, fn, cls) => { const b = h('button.icon-btn' + (cls ? '.' + cls : ''), { type: 'button', title: label, 'aria-label': label }, icon(name)); b.addEventListener('click', fn); return b; };
    const one = frames.length < 2;
    let bPrev, bNext, bPlay, bFirst, range, pos;
    const go = (k, instant) => {
      i = Math.max(0, Math.min(frames.length - 1, k));
      if (opts.mem && opts.key) opts.mem[opts.key] = i;
      const f = frames[i];
      stage.show(f, { instant: instant || !anim(), duration: opts.duration || 500 });
      if (sayEl) { sayEl.innerHTML = ''; sayEl.append(...(frames.length > 1 ? [h('span.step', 'Step ' + (i + 1) + ' of ' + frames.length + (f.title ? ' · ' + f.title : ''))] : []), ...M.say(f.say)); }
      if (!one) { range.value = i; pos.textContent = i + 1 + ' / ' + frames.length; bPrev.disabled = bFirst.disabled = i === 0; bNext.disabled = i === frames.length - 1; }
    };
    const stop = () => { clearTimeout(timer); timer = null; if (bPlay) { bPlay.replaceChildren(icon('play')); bPlay.title = 'Play'; bPlay.setAttribute('aria-label', 'Play'); } };
    const play = () => {
      if (i >= frames.length - 1) go(0);
      const tick = () => { if (i >= frames.length - 1) { stop(); return; } go(i + 1); timer = setTimeout(tick, opts.hold || 1100); };
      timer = setTimeout(tick, 300);
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
      parent.appendChild(h('div.controls.wfm-controls', bFirst, bPrev, bPlay, bNext, range, pos));
    }
    if (sayEl) parent.appendChild(sayEl);
    const saved = opts.mem && opts.key && opts.mem[opts.key] !== undefined ? opts.mem[opts.key] : undefined;
    go(saved !== undefined ? saved : opts.start === undefined ? frames.length - 1 : opts.start, true);
    if (opts.mem) opts.mem.disposers.push(stop);
    return { go, stop, stage, wrap, get i() { return i; } };
  };
  /** one still picture */
  M.still = (parent, items, view, label) => M.stepper(parent, [{ items, view }], { label });

  // ---- plots ------------------------------------------------------------------------------------------------------
  /** the box of a plot that fills width W (room on the left for the ticks, below for the axis name) */
  M.plotBox = (W, opts) => {
    opts = opts || {};
    const x = opts.left || 50, w = Math.max(150, Math.min(opts.maxW || 1e9, W - x - (opts.right || 20)));
    const hh = Math.round(Math.min(opts.maxH || 380, Math.max(150, w * (opts.ratio || 0.72))));
    return { x, y: opts.top || 28, w, h: hh };
  };
  M.viewOf = (box, extra) => ({ x: 0, y: 0, w: box.x + box.w + (extra && extra.right !== undefined ? extra.right : 20), h: box.y + box.h + (extra && extra.bottom !== undefined ? extra.bottom : 46) });
  M.axes = (prefix, box, xr, yr, names, opts) => K().curve.axes(Object.assign({ prefix, x: box.x, y: box.y, w: box.w, h: box.h, xr, yr, xticks: K().curve.ticks(xr[0], xr[1], Math.max(2, Math.round(box.w / 90))), yticks: K().curve.ticks(yr[0], yr[1], Math.max(2, Math.round(box.h / 70))), xname: names && names[0], yname: names && names[1] }, opts || {}));

  /**
   * the plane of a trained classifier: shading, boundary (and margins), points. plane from the view (wf-b-models-b.js);
   * o: {vals (another sampling of the same plane: a partial tree, a stage), n, score, levels, rings: [[k, cls]], r,
   * pointCls(k), noLevels}
   */
  M.plane = function (prefix, plane, box, o) {
    o = o || {};
    const ax = M.axes(prefix, box, plane.xr, plane.yr, plane.names);
    const items = ax.items.slice(), n = o.n || plane.n, vals = o.vals || plane.vals;
    let scheme = o.scheme || plane.scheme, rv = vals, max;
    if (scheme === 'value') { const [lo, hi] = plane.vr || [Math.min(...vals), Math.max(...vals)]; rv = vals.map((v) => (v - lo) / (hi - lo || 1)); scheme = 'seq'; max = 1; }
    const ras = { key: prefix + ':ras', kind: 'raster', x: box.x, y: box.y, w: box.w, h: box.h, nx: n + 1, ny: n + 1, vals: rv, scheme, max, smooth: scheme !== 'classes', layer: 'back', z: 0, audit: 'ignore' };
    if (scheme === 'classes') ras.vals2 = o.vals2 || plane.vals2;
    items.push(ras);
    const score = o.score !== undefined ? o.score : o.vals ? null : plane.score;
    if (score && !o.noLevels) {
      const g = { nx: n + 1, ny: n + 1, vals: Float64Array.from(score), x: box.x, y: box.y, w: box.w, h: box.h };
      for (const L of o.levels || plane.levels || [0]) items.push(K().scatter.level(ax, prefix + ':lv' + String(L).replace('-', 'm'), g, L, L === 0 ? 'boundary' : 'margin'));
    }
    const P = plane.points;
    if (P && !o.noPoints) {
      const r = o.r || (P.X.length > 250 ? 3.2 : 4);
      P.X.forEach((p, k) => items.push({ key: prefix + ':p' + k, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: o.rOf ? o.rOf(k) : r, cls: 'sp c' + ((plane.scheme === 'value' ? 0 : P.y[k]) % 4) + (o.pointCls ? ' ' + (o.pointCls(k) || '') : ''), audit: 'overlap-ok', layer: 'items' }));
      for (const [k, cls, rr] of o.rings || []) if (P.X[k]) items.push(K().scatter.ring(ax, prefix + ':ring' + k, P.X[k][0], P.X[k][1], cls || '', rr || r + 4));
    }
    return { items, ax };
  };
  /** an ellipse of a 2×2 covariance at k standard deviations, as path data on axes */
  M.ellipse = function (ax, mean, cov, k) {
    const a = cov[0][0], b = cov[0][1], d = cov[1][1], tr = (a + d) / 2, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr - det));
    const l1 = Math.max(1e-12, tr + disc), l2 = Math.max(1e-12, tr - disc), th = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
    let s = '';
    for (let i = 0; i <= 64; i++) {
      const t = (2 * Math.PI * i) / 64, u = k * Math.sqrt(l1) * Math.cos(t), v = k * Math.sqrt(l2) * Math.sin(t);
      const x = mean[0] + u * Math.cos(th) - v * Math.sin(th), y = mean[1] + u * Math.sin(th) + v * Math.cos(th);
      s += (i ? 'L' : 'M') + r2(ax.X(x)) + ',' + r2(ax.Y(y)) + ' ';
    }
    return s.trim() + 'Z';
  };
  /** horizontal bars of signed values with their names: {items, h}; o {x, y, w, size, fmt, max} */
  M.bars = function (prefix, names, vals, o) {
    const fs = o.size || 13, rowH = Math.max(fs + 10, o.rowH || 22), items = [], max = o.max || Math.max(1e-12, ...vals.map((v) => Math.abs(v)));
    const nameW = Math.min(o.w * 0.4, Math.max(...names.map((n) => S.geom.textWidth(String(n), fs))) + 10), valW = 58;
    const x0 = o.x + nameW, span = Math.max(40, o.w - nameW - valW), mid = vals.some((v) => v < 0) ? x0 + span / 2 : x0, half = vals.some((v) => v < 0) ? span / 2 : span;
    names.forEach((n, i) => {
      const y = o.y + i * rowH, v = vals[i], len = (Math.abs(v) / max) * (half - 2);
      items.push({ key: prefix + ':n' + i, kind: 'text', x: r2(x0 - 8), y: r2(y + rowH / 2 + fs * 0.35), text: String(n), anchor: 'end', size: fs, cls: 'legend' });
      items.push({ key: prefix + ':b' + i, kind: 'cell', x: r2(v >= 0 ? mid : mid - len), y: r2(y + 4), w: r2(Math.max(1, len)), h: rowH - 8, rx: 2, text: '', cls: 'wbar ' + (v >= 0 ? 'pos' : 'neg') + (o.cls ? ' ' + (o.cls(i) || '') : ''), audit: 'ignore' });
      items.push({ key: prefix + ':v' + i, kind: 'text', x: r2(x0 + span + valW - 4), y: r2(y + rowH / 2 + fs * 0.35), text: (o.fmt || M.num)(v), anchor: 'end', size: fs, cls: 'legend mono' });
    });
    if (vals.some((v) => v < 0)) items.push({ key: prefix + ':axis', kind: 'path', d: `M${r2(mid)},${r2(o.y)} V${r2(o.y + names.length * rowH)}`, cls: 'axis', audit: 'ignore', layer: 'back' });
    return { items, h: names.length * rowH };
  };
  /** a series plot: {items, ax}; series [{pts, cls, key}], o {xr, yr, names, dots} */
  M.series = function (prefix, box, series, o) {
    const all = series.flatMap((s) => s.pts).filter((p) => p && isFinite(p[1]));
    const xr = o.xr || [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
    let yr = o.yr;
    if (!yr) { const ys = all.map((p) => p[1]); const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo || 1) * 0.08; yr = [lo - pad, hi + pad]; }
    if (xr[0] === xr[1]) xr[1] = xr[0] + 1;
    const ax = M.axes(prefix, box, xr, yr, o.names, o.axes);
    const items = ax.items.slice();
    series.forEach((s, i) => { items.push(K().curve.line(ax, prefix + ':s' + (s.key || i), s.pts, s.cls)); });
    return { items, ax };
  };

  // ---- the untrained model: what it is and what its settings do --------------------------------------------------
  /** a model block before training: its idea, its settings (and for networks, its layers) */
  M.modelViewer = M.mount((view, ctx) => {
    const fam = WF.families[view.family] || {}, def = WF.def(view.family) || {};
    ctx.el.append(M.title(view.label, 'not trained yet'), h('p.wfm-blurb', def.blurb || ''));
    const rows = (def.params || []).filter((p) => !p.show || p.show(view.params)).map((p) => {
      let v = view.params[p.id];
      if (p.kind === 'choice') { const opts = typeof p.options === 'function' ? p.options(view.params) : p.options || []; const o = opts.find((x) => String(x[0]) === String(v)); if (o) v = o[1]; }
      if (Array.isArray(v)) v = v.join(' → ');
      if (typeof v === 'boolean') v = v ? 'yes' : 'no';
      return h('tr', h('th', p.label), h('td.mono', String(v)), h('td.muted', p.help || ''));
    });
    if (rows.length) ctx.el.appendChild(h('div.wfm-tablebox', h('table.wfm-table', h('tbody', ...rows))));
    if (M.architecture[view.family]) M.architecture[view.family](view, ctx);
    ctx.el.appendChild(M.note(fam.iterative ? 'Wire it into a Train block with some data: it learns there, epoch by epoch, and you can pause and step it.' : 'Wire it into a Train block with some data: it is fitted there, and Train shows its insides.'));
  });
  M.architecture = M.architecture || {};
  const register = () => { for (const id of Object.keys(WF.families || {})) if (!WF.viewers[id]) WF.viewers[id] = M.modelViewer; };
  register();
  // families defined by files that load later (the networks) are registered when the page starts
  if (typeof document !== 'undefined' && document.addEventListener) document.addEventListener('DOMContentLoaded', register);
})(globalThis.Nooral = globalThis.Nooral || {});
