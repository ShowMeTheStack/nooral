// The answer areas of the machine-learning exercise kinds (checked by check-ml.js and check-workflow.js):
//   matrix         a grid of number boxes laid out as the matrix, with its row and column names; wrong cells turn red
//   point          a plot drawn on the stage; a click puts a marker (a point, a threshold line, or picks a point)
//   predict-curve  a plot on which the reader draws, dragging across it; the known part of the curve is drawn faint
//   workflow       the task (data, goal, limits) and a canvas to build the workflow in
// Each is S.ExerciseViews.kind = (x, host, draft, page) -> {get(), mark(result)?, after(result, box)?}.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const V = (S.ExerciseViews = S.ExerciseViews || {});
  const f = (v) => String(+(+v).toFixed(3)).replace(/^-/, '−');

  // ---- matrix --------------------------------------------------------------------------------------------------------
  V.matrix = function (x, host, draft) {
    const given = new Set((x.given || []).map(([i, j]) => i + ',' + j));
    const inputs = {}, sumCells = [];
    const head = h('tr', h('th.mx-corner', x.corner || ''), ...x.cols.map((c) => h('th.mx-col', c)));
    const body = h('tbody');
    x.answer.forEach((row, i) => {
      const tr = h('tr', h('th.mx-row', x.rows[i]));
      row.forEach((v, j) => {
        if (given.has(i + ',' + j)) { tr.appendChild(h('td.mx-given.mono', f(v))); return; }
        const inp = h('input.mono', { type: 'text', inputmode: 'decimal', autocomplete: 'off', spellcheck: 'false', 'aria-label': x.rows[i] + ', ' + x.cols[j], dataset: { i: String(i), j: String(j) }, placeholder: '' });
        if (draft && draft[i] && draft[i][j] !== undefined && draft[i][j] !== null) inp.value = draft[i][j];
        inp.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') { e.preventDefault(); const all = [...body.querySelectorAll('input')]; const n = all[all.indexOf(inp) + 1]; if (n) n.focus(); else document.getElementById('btnCheck').click(); }
        });
        inputs[i + ',' + j] = inp;
        tr.appendChild(h('td', inp));
      });
      if (x.rowsSumTo !== undefined) tr.appendChild(sumCells[i] = h('td.mx-sum.mono', { dataset: { row: String(i) }, 'aria-live': 'polite', 'aria-label': 'Sum of row ' + x.rows[i] }));
      body.appendChild(tr);
    });
    if (x.rowsSumTo !== undefined) head.appendChild(h('th.mx-col.mx-sumh', 'row sum'));
    // diag: true tints the diagonal (a confusion matrix's correct answers); other grids stay plain
    const tbl = h('table.mx-table' + (x.rowsSumTo !== undefined ? '.mx-attn' : '.mx-conf') + (x.diag ? '.mx-diag' : ''), h('thead', head), body);
    const sums = () => {
      if (x.rowsSumTo === undefined) return;
      // the live sum of what is written so far (a blank cell counts 0): red once it passes the target, or when the
      // row is full and misses it; marked good when it lands on the target
      x.answer.forEach((row, i) => {
        let s = 0, full = true;
        row.forEach((v, j) => { const k = i + ',' + j; const t = given.has(k) ? v : S.Check.readNumber(inputs[k].value); if (t !== null && Number.isFinite(t)) s += t; else full = false; });
        const c = sumCells[i], tol = Math.max(0.011, (x.tolerance || 0) * row.length), hit = Math.abs(s - x.rowsSumTo) <= tol;
        c.textContent = f(s);
        c.classList.toggle('off', (full && !hit) || s > x.rowsSumTo + tol);
        c.classList.toggle('hit', full && hit);
      });
    };
    body.addEventListener('input', sums);
    body.addEventListener('change', sums);
    host.appendChild(h('div.fit-table.mx-wrap', tbl));
    const ints = x.answer.every((r) => r.every((v) => Number.isInteger(v)));
    // a square table whose rows and columns carry the same names reads as a confusion matrix
    const confusionLike = !!(x.rows && x.cols && x.rows.length === x.cols.length && x.rows.every((r, i) => String(r) === String(x.cols[i])));
    host.appendChild(h('p.muted.ans-note', x.note || (x.rowsSumTo !== undefined ? 'Write a weight in each cell, to two places. The last column adds up each row: a row of weights adds up to ' + x.rowsSumTo + '.' : ints && (x.counts || x.rowLabel || x.colLabel || confusionLike) ? 'Write a count in each cell: rows are ' + (x.rowLabel || 'the true classes') + ', columns ' + (x.colLabel || 'what the model predicted') + '.' : ints ? 'Write a whole number in each cell.' : 'Write a number in each cell' + (x.tolerance ? ', to within ' + x.tolerance : '') + '.')));
    sums();
    return {
      get: () => x.answer.map((row, i) => row.map((_, j) => (inputs[i + ',' + j] ? inputs[i + ',' + j].value : ''))),
      mark(r) { sums(); for (const [k, inp] of Object.entries(inputs)) { const ok = r.cells && r.cells[k]; inp.classList.toggle('right', !!ok); inp.classList.toggle('wrong', !!(r.cells && !ok)); } },
    };
  };

  // ---- a plot on the stage, shared by point and predict-curve ---------------------------------------------------------
  // a narrower plot on a phone, so its numbers stay readable at the phone's width
  const boxFor = (strip) => { const phone = typeof innerWidth !== 'undefined' && innerWidth < 600; return phone ? { x: 30, y: 30, w: 310, h: strip ? 110 : 250 } : { x: 66, y: 34, w: 520, h: strip ? 130 : 300 }; };
  function plot(x, host, P, opts) {
    const BOX = boxFor(P.strip);
    if (BOX.x === 30 && !P.strip) BOX.x = 48, BOX.w = 290;
    // a plot where distances matter (nearest neighbours, clusters) keeps one unit the same length across and up
    if (P.equal && !P.strip) {
      const sx = P.xr[1] - P.xr[0], sy = P.yr[1] - P.yr[0], k = Math.min(BOX.w / sx, BOX.h / sy);
      BOX.w = Math.round(sx * k); BOX.h = Math.round(sy * k);
    }
    const wrap = h('div.stage-wrap.ex-plot' + (opts.cls ? '.' + opts.cls : ''), { id: 'exPlot' });
    host.appendChild(wrap);
    const stage = new S.Stage(wrap, { label: opts.label || 'A plot to answer on' });
    const K = S.Kit;
    const ax = K.curve.axes({ prefix: 'ex', x: BOX.x, y: BOX.y, w: BOX.w, h: BOX.h, xr: P.xr, yr: P.yr, xname: P.xname, yname: P.yname, yticks: P.strip ? 0 : undefined, frame: P.strip ? false : undefined });
    if (P.strip) ax.items.push({ key: 'ex:axis', kind: 'path', d: 'M' + BOX.x + ',' + (BOX.y + BOX.h) + ' H' + (BOX.x + BOX.w), cls: 'axis', audit: 'ignore', layer: 'back', z: 2 });
    if (P.strip) wrap.classList.add('strip');
    const view = { x: 0, y: 0, w: BOX.x + BOX.w + 24, h: BOX.y + BOX.h + 54 };
    const base = () => {
      const items = ax.items.slice();
      for (const [k, l] of (P.lines || []).entries()) {
        if (l.x !== undefined) items.push(K.curve.vline(ax, 'ex:l' + k, l.x, 'given'));
        else if (l.y !== undefined) items.push(K.curve.hline(ax, 'ex:l' + k, l.y, 'given'));
        else if (l.pts) items.push(K.curve.line(ax, 'ex:l' + k, l.pts, 'faint', { layer: 'back' }));
      }
      if (P.given && P.given.length > 1) items.push(K.curve.line(ax, 'ex:given', P.given, 'faint', { layer: 'back' }));
      if (P.given) P.given.forEach((g, k) => items.push(Object.assign(K.curve.dot(ax, 'ex:g' + k, g[0], g[1], 'given', 5.5), { audit: 'point', layer: 'items', z: 6 })));
      if (P.points) items.push(...K.scatter.points(ax, 'ex:p', P.points, P.classes || P.points.map(() => 0), { r: 5.5 }));
      for (const [k, c] of (P.centroids || []).entries()) items.push({ key: 'ex:c' + k, kind: 'path', d: 'M' + (ax.X(c[0]) - 7) + ',' + (ax.Y(c[1]) - 7) + ' l14,14 M' + (ax.X(c[0]) + 7) + ',' + (ax.Y(c[1]) - 7) + ' l-14,14', cls: 'centroid c' + k, audit: 'overlap-ok', layer: 'items', z: 3 });
      return items;
    };
    const draw = (extra) => stage.show({ items: base().concat(extra || []), view }, { instant: true });
    /** a pointer event in the plot's own units, or null outside the plot */
    const at = (e) => {
      const svg = stage.root, m = svg.getScreenCTM();
      if (!m) return null;
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const q = pt.matrixTransform(m.inverse());
      if (q.x < BOX.x - 6 || q.x > BOX.x + BOX.w + 6 || q.y < BOX.y - 6 || q.y > BOX.y + BOX.h + 6) return null;
      const cl = (v, a, b) => Math.max(Math.min(a, b), Math.min(Math.max(a, b), v));
      return { x: cl(ax.U(q.x), P.xr[0], P.xr[1]), y: cl(ax.V(q.y), P.yr[0], P.yr[1]), px: q.x, py: q.y };
    };
    draw();
    return { wrap, stage, ax, draw, at, box: BOX };
  }
  const round = (v, span) => { const d = Math.max(0, 2 - Math.floor(Math.log10(span || 1))); return +v.toFixed(d); };

  // ---- point ---------------------------------------------------------------------------------------------------------
  V.point = function (x, host, draft) {
    const P = S.Check.plotOf(x), mode = x.mode || 'xy', K = S.Kit;
    let ans = draft && typeof draft === 'object' ? draft : null;
    const pl = plot(x, host, P, { label: 'Click on the plot to answer', cls: 'pickable' });
    const readout = h('p.muted.ans-note', { id: 'pointReadout', 'aria-live': 'polite' });
    host.appendChild(readout);
    const show = (res) => {
      const ex = [];
      if (ans && mode === 'pick' && ans.i !== undefined) ex.push({ key: 'ex:pick', kind: 'dot', x: pl.ax.X(P.points[ans.i][0]), y: pl.ax.Y(P.points[ans.i][1]), r: 11, cls: 'ring answer' + (res ? (res.ok ? ' good' : ' bad') : ''), audit: 'overlap-ok', layer: 'items', z: 4 });
      else if (ans && mode === 'x') ex.push(Object.assign(K.curve.vline(pl.ax, 'ex:ans', ans.x, 'ansline'), { layer: 'items' }));
      else if (ans && mode === 'y') ex.push(Object.assign(K.curve.hline(pl.ax, 'ex:ans', ans.y, 'ansline'), { layer: 'items' }));
      else if (ans && mode === 'xy') ex.push({ key: 'ex:ans', kind: 'dot', x: pl.ax.X(ans.x), y: pl.ax.Y(ans.y), r: 8, cls: 'mark answer' + (res ? (res.ok ? ' good' : ' bad') : ''), audit: 'overlap-ok', layer: 'items', z: 5 });
      if (res && res.ok === false && x.showAnswerAfter) { /* never shown: the hints give it */ }
      pl.draw(ex);
      const sx = P.xr[1] - P.xr[0], sy = P.yr[1] - P.yr[0];
      readout.textContent = !ans ? (mode === 'pick' ? 'Click the point you choose.' : mode === 'x' ? 'Click where the line goes: only ' + (P.xname || 'x') + ' counts.' : mode === 'y' ? 'Click where the line goes: only ' + (P.yname || 'y') + ' counts.' : 'Click the place you choose.')
        : mode === 'pick' ? 'You chose ' + ((P.labels && P.labels[ans.i]) || (P.strip ? 'the point at ' + (P.xname || 'x') + ' = ' + f(P.points[ans.i][0]) : 'the point at (' + f(P.points[ans.i][0]) + ', ' + f(P.points[ans.i][1]) + ')')) + '.'
        : mode === 'x' ? 'Your line: ' + (P.xname || 'x') + ' = ' + round(ans.x, sx) + '.' : mode === 'y' ? 'Your line: ' + (P.yname || 'y') + ' = ' + round(ans.y, sy) + '.'
        : 'Your point: (' + round(ans.x, sx) + ', ' + round(ans.y, sy) + ').';
    };
    pl.stage.root.addEventListener('click', (e) => {
      const q = pl.at(e);
      if (!q) return;
      if (mode === 'pick') {
        let best = -1, bd = Infinity;
        P.points.forEach((p, i) => { const d = Math.hypot(pl.ax.X(p[0]) - q.px, pl.ax.Y(p[1]) - q.py); if (d < bd) { bd = d; best = i; } });
        if (best < 0 || bd > 24) return;
        ans = { i: best };
      } else ans = mode === 'x' ? { x: q.x } : mode === 'y' ? { y: q.y } : { x: q.x, y: q.y };
      show();
      host.dispatchEvent(new Event('input', { bubbles: true }));
    });
    show();
    return { get: () => ans, mark: (r) => show(r), redraw: () => show() };
  };

  // ---- predict-curve -------------------------------------------------------------------------------------------------
  V['predict-curve'] = function (x, host, draft) {
    const P = x.plot, K = S.Kit;
    let pts = Array.isArray(draft) ? draft : [];
    const pl = plot(x, host, P, { label: 'Draw the curve: drag across the plot', cls: 'drawable' });
    const show = (res) => {
      const ex = [];
      const sk = S.Check.cleanSketch(pts);
      if (sk.length > 1) ex.push(Object.assign(K.curve.line(pl.ax, 'ex:sketch', sk, 'act sketch' + (res ? (res.ok ? ' good' : ' bad') : '')), { layer: 'items' }));
      if (res && res.ok) ex.push(Object.assign(K.curve.line(pl.ax, 'ex:truth', x.answer, 'thin truth'), { layer: 'items' }));
      pl.draw(ex);
    };
    let drawing = false;
    const root = pl.stage.root;
    root.style.touchAction = 'none';
    root.addEventListener('pointerdown', (e) => { const q = pl.at(e); if (!q) return; drawing = true; pts = [[q.x, q.y]]; try { root.setPointerCapture(e.pointerId); } catch (err) { /* synthetic */ } show(); e.preventDefault(); });
    root.addEventListener('pointermove', (e) => {
      if (!drawing) return;
      const q = pl.at(e);
      if (!q) return;
      // drawing back over the curve replaces what was there
      if (pts.length && q.x < pts[pts.length - 1][0]) pts = pts.filter((p) => p[0] < q.x);
      pts.push([q.x, q.y]);
      show();
    });
    const end = () => { if (!drawing) return; drawing = false; host.dispatchEvent(new Event('input', { bubbles: true })); };
    root.addEventListener('pointerup', end);
    root.addEventListener('pointercancel', end);
    const clear = h('button.btn.small', { type: 'button', id: 'btnClearSketch' }, icon('clear'), 'Clear the sketch');
    clear.addEventListener('click', () => { pts = []; show(); host.dispatchEvent(new Event('input', { bubbles: true })); });
    host.appendChild(h('div.ans-row', h('p.muted.ans-note', P.note || 'Press and drag across the plot to draw the curve, from left to right. The faint line is what is known already.'), clear));
    show();
    return { get: () => S.Check.cleanSketch(pts).map(([a, b]) => [+a.toFixed(4), +b.toFixed(4)]), mark: (r) => show(r), redraw: () => show(), set(p) { pts = p; show(); } };
  };

  // ---- workflow ------------------------------------------------------------------------------------------------------
  /** the task of a workflow exercise or a challenge, as a card: the data, the goal and the limits */
  V.taskCard = function (x) {
    const C = S.Challenges, doc = S.Data && S.Data.doc(x.data.set);
    return h('div.wf-task', { id: 'wfTask' },
      h('div.wt-row', h('span.wt-k', 'Data'), h('span', h('b', doc ? doc.title : x.data.set), ', ' + (x.data.n || (doc && doc.size) || '') + (x.data.n ? ' training rows' : ''), doc && doc.story ? h('span.muted', ' · ' + doc.story) : null)),
      h('div.wt-row', h('span.wt-k', 'Goal'), h('span', h('b', C.goal(x)), ' on rows the workflow never sees')),
      h('div.wt-row', h('span.wt-k', 'Limits'), h('ul.wt-limits', ...C.limitSentences(x).map((s) => h('li', s)))));
  };
  V.workflow = function (x, host, draft, page) {
    host.appendChild(V.taskCard(x));
    const box = h('div.ex-canvas', { id: 'exCanvas' });
    host.appendChild(box);
    let doc = null;
    try { doc = draft && draft.blocks ? S.WF.fromJSON(draft) : S.Challenges.starter(x); } catch (e) { doc = S.Challenges.starter(x); }
    let ed = null;
    const allowed = x.limits && x.limits.blocks ? ['challenge-data'].concat(x.limits.blocks) : null;
    if (S.WF.embed) {
      try {
        ed = S.WF.embed(box, doc, { allowed, height: 460, onChange: () => host.dispatchEvent(new Event('input', { bubbles: true })) });
        if (page) (page.editors = page.editors || []).push(ed);
      } catch (e) { box.appendChild(S.UI.empty('warn', 'The canvas could not open', String(e.message || e))); }
    } else box.appendChild(S.UI.empty('layout', 'The canvas is on its way', 'This task is built on the workflow canvas, which is not in this build yet.'));
    host.appendChild(h('p.muted.ans-note', 'Build the workflow on the canvas, then check it: it is trained here and scored on hidden rows.'));
    return {
      get: () => (ed ? ed.doc() : doc),
      set(d) { doc = S.WF.fromJSON(d); if (ed && ed.loadDoc) ed.loadDoc(doc); },
      after(r, out) { if (r.score !== undefined && r.score !== null) out.appendChild(h('div.ex-d.wf-score', icon('trophy'), ' ' + S.Challenges.metricLabel(x) + ' ' + S.Challenges.fmt(r.score) + ' · target ' + S.Challenges.goal(x))); },
      dispose() { if (ed && ed.dispose) ed.dispose(); },
      redraw() { if (ed && ed.redraw) ed.redraw(); },
    };
  };

  // ---- what the diagram checker looks at: a page of each kind, and one answered ---------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.group === 'exercises')) {
    const ready = "document.querySelector('#exerciseView')";
    for (const [id, name] of [['ev-x-cm-fill', 'a confusion matrix to fill'], ['tf-x-weights', 'attention weights to fill'], ['un-x-update', 'a point to click'], ['tr-x-best-threshold', 'a threshold to click'], ['nn-x-nearest-pick', 'a point to pick'], ['li-x-gd-curve', 'a curve to sketch'], ['tr-x-wf-moons', 'a workflow to build']]) {
      S.Checks.push({ group: 'exercises', name, hash: 'exercise=' + id, ready, sel: '#main' });
    }
    S.Checks.push({ group: 'exercises', name: 'a matrix checked', hash: 'exercise=ev-x-cm-fill', fresh: true, setup: "document.querySelectorAll('#answer input').forEach((i, k) => { i.value = String(k + 1); }); document.getElementById('btnCheck').click();", ready: "document.querySelector('#exResult.bad')", sel: '#main' });
    S.Checks.push({ group: 'exercises', name: 'a sketch checked', hash: 'exercise=li-x-gd-curve', fresh: true, setup: "const x = Nooral.Content.exercises.get('li-x-gd-curve'); Nooral.app.page.answer.set(x.answer.map(([a, b]) => [a, b + 0.3])); document.getElementById('btnCheck').click();", ready: "document.querySelector('#exResult.ok')", sel: '#main' });
    S.Checks.push({ group: 'exercises', name: 'a point checked', hash: 'exercise=un-x-update', fresh: true, setup: "document.getElementById('btnCheck').click();", ready: "document.querySelector('#exResult.bad')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
