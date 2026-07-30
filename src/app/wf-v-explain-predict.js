// The Predict block's viewer: an inference panel. For a table, a field for every raw column (a slider over the range
// the model saw, or a choice); on 2-D data the plane as well, where a click asks about that point. For an image model,
// a pad to draw on; for a text model, a sentence to type. Each change asks the worker (the block's predict method) and
// shows the answer with its probability bars and the reason in brief.
(function (S) {
  'use strict';
  const WF = S.WF, VK = WF.VK, h = (...a) => S.UI.h(...a);
  const C = () => S.Kit.curve;
  const fmtIn = (x, step) => { const d = step >= 1 ? 0 : Math.min(4, Math.max(0, -Math.floor(Math.log10(step || 0.01)))); return (+x).toFixed(d); };

  function answerBox(v, a, truthIdx) {
    const box = h('div.wf-answer', { id: 'wfAnswer' });
    if (!a) { box.appendChild(VK.note(v.kind === 'images' ? 'Draw a digit in the box: the answer appears when you lift the pen.' : 'Change a value to ask the model.', 'muted')); return box; }
    if (v.task === 'regression') box.appendChild(h('div.wf-answer-head', 'The model predicts ', h('b', a.label)));
    else {
      const p = a.proba ? a.proba[a.pred] : null;
      box.appendChild(h('div.wf-answer-head', 'The model says ', h('b', a.label), p !== null ? ', with probability ' + VK.pct(p) : ' (it gives no probabilities)'));
      if (a.proba) box.appendChild(VK.bars(v.classes, a.proba, truthIdx));
    }
    const w = a.why;
    if (w && w.kind === 'linear') {
      const parts = w.parts.slice().sort((x, y) => Math.abs(y.part) - Math.abs(x.part));
      const total = w.intercept + w.parts.reduce((s, x) => s + x.part, 0);
      box.appendChild(h('div', h('b', 'Why: '), 'the score is a weighted sum of the values' + (v.task === 'classification' ? ', and the probability of ' + v.classes[w.cls] + ' grows with it.' : '.')));
      box.appendChild(VK.table(['column', 'value', '× weight', '= part'], parts.map((x) => [x.name, VK.fmt(x.value, 3), VK.fmt(x.weight, 3), h('span' + (x.part >= 0 ? '.pos' : '.neg'), (x.part >= 0 ? '+' : '') + VK.fmt(x.part, 3))]).concat([['intercept', '', '', VK.fmt(w.intercept, 3)], ['score', '', '', h('b', VK.fmt(total, 3))]]), 'why'));
    } else if (w && w.kind === 'near') {
      box.appendChild(h('div', h('b', 'The rows it learned from most like this one: ')));
      box.appendChild(VK.table(v.features.map((f) => f.name).concat(['was']), w.rows.map((r) => r.x.map((x, j) => (v.features[j].categories ? v.features[j].categories[x] : VK.fmt(x, 3))).concat([r.y === null ? '' : v.task === 'classification' ? v.classes[r.y] : VK.fmt(r.y, 3)])), 'near'));
    }
    return box;
  }

  // ---- a table ----------------------------------------------------------------------------------------------------------
  function table(box, v, ctx) {
    const st = ctx.state, F = v.features;
    if (!st.row) st.row = F.map((f) => (f.categories ? Math.round(f.start || 0) : +f.start));
    const wide = !ctx.narrow && ctx.W >= 760 && v.plane;
    const ansHost = h('div.wf-ans-host');
    const fields = h('div.wf-ask', { id: 'wfAsk' });
    const setters = [];
    F.forEach((f, j) => {
      if (f.categories) {
        const sel = h('select', { 'aria-label': f.name }, ...f.categories.map((c, k) => h('option', { value: String(k), selected: k === st.row[j] }, c)));
        sel.addEventListener('change', () => { st.row[j] = +sel.value; ask(); });
        setters.push(() => { sel.value = String(st.row[j]); });
        fields.appendChild(h('label.wf-field', h('span', f.name), sel));
        return;
      }
      const lo = Number.isFinite(f.min) ? f.min : 0, hi = Number.isFinite(f.max) ? f.max : 1, step = f.step || 0.01;
      const r = h('input', { type: 'range', min: String(lo), max: String(hi), step: String(step), value: String(st.row[j]), 'aria-label': f.name });
      const t = h('input', { type: 'text', value: fmtIn(st.row[j], step), 'aria-label': f.name + ' value', inputmode: 'decimal', spellcheck: 'false' });
      r.addEventListener('input', () => { st.row[j] = +r.value; t.value = fmtIn(r.value, step); ask(); });
      t.addEventListener('change', () => { const x = Number(String(t.value).replace(/−/g, '-')); if (Number.isFinite(x)) { st.row[j] = x; r.value = String(x); ask(); } });
      setters.push(() => { r.value = String(st.row[j]); t.value = fmtIn(st.row[j], step); });
      fields.appendChild(h('label.wf-field', h('span', f.name + (f.unit ? ' (' + f.unit + ')' : '') + (Number.isFinite(f.min) ? ' · seen ' + VK.fmt(f.min, 3) + ' to ' + VK.fmt(f.max, 3) : '')), h('div.wf-field-row', r, t)));
    });
    let fig = null;
    const drawPlane = () => {
      if (!v.plane) return;
      const p = v.plane, side = Math.min(wide ? 380 : ctx.W - 70, 420), ax = C().axes({ prefix: 'pl', x: 54, y: 26, w: side, h: side, xr: p.xr, yr: p.yr, xname: F[0].name, yname: F[1].name, grid: false });
      const items = ax.items.slice();
      items.push({ key: 'pl:field', kind: 'raster', x: 54, y: 26, w: side, h: side, nx: p.nx, ny: p.ny, vals: p.vals, vals2: p.conf, scheme: v.task === 'classification' ? 'classes' : 'seq', max: v.task === 'classification' ? 1 : Math.max(...p.vals.map(Math.abs)) || 1, alpha: 0.4, smooth: false, layer: 'back', z: 0 });
      if (v.points) v.points.X.forEach((q, k) => { if (ax.inside(q[0], q[1])) items.push({ key: 'pl:p' + k, kind: 'dot', x: ax.X(q[0]), y: ax.Y(q[1]), r: 3, cls: 'sp c' + Math.min(3, v.points.y ? v.points.y[k] : 0), audit: 'overlap-ok' }); });
      const u = Math.max(p.xr[0], Math.min(p.xr[1], st.row[0])), w = Math.max(p.yr[0], Math.min(p.yr[1], st.row[1]));
      items.push(Object.assign(C().dot(ax, 'pl:me', u, w, 'thr', 7), { audit: 'overlap-ok' }));
      items.push({ key: 'pl:hit', kind: 'cell', x: 54, y: 26, w: side, h: side, rx: 0, cls: 'hit', pick: true, audit: 'ignore', layer: 'top' });
      fig.show(items, 54 + side + 16, 26 + side + 52);
      fig.onPick = (key, e) => {
        const svg = fig.stage.root, pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const m = svg.getScreenCTM(); if (!m) return;
        const q = pt.matrixTransform(m.inverse());
        st.row[0] = +ax.U(q.x).toPrecision(4); st.row[1] = +ax.V(q.y).toPrecision(4);
        setters.forEach((f) => f()); ask();
      };
    };
    const ask = S.UI.debounce(() => {
      ctx.api.call('predict', { rows: [st.row.slice()] }).then((ans) => { st.answer = ans[0]; ansHost.innerHTML = ''; ansHost.appendChild(answerBox(v, st.answer)); if (v.plane) drawPlane(); S.UI.fitTables(ansHost); }, (e) => { ansHost.innerHTML = ''; ansHost.appendChild(VK.note('The model could not answer: ' + (e.message || e), 'muted')); });
    }, 120);
    box.appendChild(h('div.wf-ev-head', h('div.wf-ev-title', h('b', 'Ask ' + v.label), v.plane ? ': type values, or click anywhere on the plane' : ': set each value')));
    if (v.plane) { fig = VK.fig('The plane the model divides'); box.appendChild(h('div.wf-ev-row' + (wide ? '.wide' : ''), h('div.wf-ev-main', fig.el), h('div.wf-ev-side', fields, ansHost))); drawPlane(); }
    else box.append(fields, ansHost);
    ansHost.appendChild(answerBox(v, st.answer));
    ask();
  }

  // ---- an image: a pad to draw on --------------------------------------------------------------------------------------
  function image(box, v, ctx) {
    const st = ctx.state, w = v.w || 28, hh = v.h || 28, scale = Math.max(8, Math.floor(240 / w));
    if (!st.px || st.px.length !== w * hh) st.px = new Float32Array(w * hh);
    const pad = h('canvas.wf-pad', { width: String(w), height: String(hh), 'aria-label': 'Draw here', id: 'wfPad' });
    pad.style.width = w * scale + 'px'; pad.style.height = hh * scale + 'px';
    const ansHost = h('div.wf-ans-host');
    const paint = () => {
      const c = pad.getContext && pad.getContext('2d'); if (!c) return;
      const im = c.createImageData(w, hh), ink = S.themeRgb('--text'), paper = S.themeRgb('--node');
      for (let i = 0; i < w * hh; i++) { for (let k = 0; k < 3; k++) im.data[i * 4 + k] = paper[k] + (ink[k] - paper[k]) * st.px[i]; im.data[i * 4 + 3] = 255; }
      c.putImageData(im, 0, 0);
    };
    // a soft round brush about a tenth of the image wide
    const brush = (x, y) => { const r = Math.max(1.1, w / 14); for (let j = Math.floor(y - r - 1); j <= y + r + 1; j++) for (let i = Math.floor(x - r - 1); i <= x + r + 1; i++) { if (i < 0 || j < 0 || i >= w || j >= hh) continue; const d = Math.hypot(i + 0.5 - x, j + 0.5 - y), a = Math.max(0, Math.min(1, (r + 0.6 - d) / 1.2)); st.px[j * w + i] = Math.max(st.px[j * w + i], a); } };
    let down = false, last = null;
    const at = (e) => { const r = pad.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * hh]; };
    const stroke = (e) => { const p = at(e); if (last) { const n = Math.ceil(Math.hypot(p[0] - last[0], p[1] - last[1]) * 3) + 1; for (let k = 1; k <= n; k++) brush(last[0] + ((p[0] - last[0]) * k) / n, last[1] + ((p[1] - last[1]) * k) / n); } else brush(p[0], p[1]); last = p; paint(); };
    pad.addEventListener('pointerdown', (e) => { down = true; last = null; try { pad.setPointerCapture(e.pointerId); } catch (x) { /* synthetic */ } stroke(e); });
    pad.addEventListener('pointermove', (e) => { if (down) stroke(e); });
    const up = () => { if (!down) return; down = false; last = null; ask(); };
    pad.addEventListener('pointerup', up); pad.addEventListener('pointercancel', up);
    const ask = () => ctx.api.call('predict', { images: [Array.from(st.px)] }).then((a) => { st.answer = a[0]; ansHost.innerHTML = ''; ansHost.appendChild(answerBox(v, st.answer)); }, (e) => { ansHost.innerHTML = ''; ansHost.appendChild(VK.note('The model could not answer: ' + (e.message || e), 'muted')); });
    const clear = h('button.btn.small', { type: 'button', id: 'wfPadClear' }, 'Clear');
    clear.addEventListener('click', () => { st.px.fill(0); st.answer = null; paint(); ansHost.innerHTML = ''; ansHost.appendChild(answerBox(v, null)); });
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', 'Ask ' + v.label), ': draw in the box (' + w + ' × ' + hh + ' pixels, as the model sees them)')),
      h('div.wf-pad-wrap', h('div', pad, h('div.wf-pad-tools', clear)), ansHost));
    paint();
    ansHost.appendChild(answerBox(v, st.answer));
    // put an image on the pad (a test, a lesson): any square image is resampled to the pad's own size
    WF.VK.padOf = (px) => {
      const n = Math.round(Math.sqrt(px.length)) || 1;
      for (let j = 0; j < hh; j++) for (let i = 0; i < w; i++) st.px[j * w + i] = +px[Math.min(n - 1, Math.floor((j * n) / hh)) * n + Math.min(n - 1, Math.floor((i * n) / w))] || 0;
      paint();
      return ask();
    };
  }

  // ---- text ------------------------------------------------------------------------------------------------------------
  function text(box, v, ctx) {
    const st = ctx.state;
    const ta = h('input', { type: 'text', value: st.text || '', placeholder: 'Type a sentence', 'aria-label': 'A sentence to classify', id: 'wfAskText', style: { flex: '1 1 260px' } });
    const go = h('button.btn.primary.small', { type: 'button', id: 'wfAskGo' }, 'Ask');
    const ansHost = h('div.wf-ans-host');
    const ask = () => { st.text = ta.value; if (!ta.value.trim()) return; ctx.api.call('predict', { texts: [ta.value] }).then((a) => { st.answer = a[0]; ansHost.innerHTML = ''; ansHost.appendChild(answerBox(v, st.answer)); }, (e) => { ansHost.innerHTML = ''; ansHost.appendChild(VK.note('The model could not answer: ' + (e.message || e), 'muted')); }); };
    go.addEventListener('click', ask);
    ta.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); ask(); } });
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', 'Ask ' + v.label), ': type a sentence')), h('div.wf-tools', ta, go), ansHost);
    ansHost.appendChild(answerBox(v, st.answer));
  }

  WF.viewers.predict = VK.viewer((box, v, ctx) => (v.kind === 'images' ? image(box, v, ctx) : v.kind === 'text' ? text(box, v, ctx) : table(box, v, ctx)), 'predict');
})(globalThis.Nooral = globalThis.Nooral || {});
