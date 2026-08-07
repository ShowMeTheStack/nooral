// The machine-learning answer kinds checked at once: a matrix to fill in (a confusion matrix, attention weights), a
// point to click on a plot (the next centroid, a split threshold, the point a model picks) and a curve to sketch
// (where a learning curve or a loss will go), checked approximately. Fields of each kind: src/app/check-index.js.
(function (S) {
  'use strict';
  const K = (S.Check = S.Check || {});
  K.kinds = K.kinds || {};
  const right = (x, extra) => ({ ok: true, message: 'Right.' + (extra ? ' ' + extra : '') + (x.why ? ' ' + x.why : '') });
  const f = (v) => String(+(+v).toFixed(3)).replace(/^-/, '−');
  const num = (t) => { const s = String(t === undefined || t === null ? '' : t).trim().replace(/[−–]/g, '-').replace(/,/g, ''); if (!s) return null; const v = /^-?\d+\s*\/\s*\d+$/.test(s) ? s.split('/').reduce((a, b) => +a / +b) : Number(s); return Number.isFinite(v) ? v : NaN; };
  K.readNumber = num;

  // ---- matrix ---------------------------------------------------------------------------------------------------
  /** the cells a reader fills: every cell but the given ones */
  K.matrixCells = (x) => { const given = new Set((x.given || []).map(([i, j]) => i + ',' + j)); const out = []; x.answer.forEach((r, i) => r.forEach((_, j) => { if (!given.has(i + ',' + j)) out.push([i, j]); })); return out; };
  K.kinds.matrix = {
    check(x, a) {
      a = a || [];
      const tol = x.tolerance || 0, cells = {};
      let wrong = 0, empty = 0, bad = 0;
      for (const [i, j] of K.matrixCells(x)) {
        const g = num(a[i] && a[i][j]);
        const ok = g !== null && !Number.isNaN(g) && Math.abs(g - x.answer[i][j]) <= tol + 1e-9;
        cells[i + ',' + j] = ok;
        if (g === null) empty++; else if (Number.isNaN(g)) bad++;
        if (!ok) wrong++;
      }
      const n = K.matrixCells(x).length;
      if (!wrong) return Object.assign(right(x, 'All ' + n + ' cells are right.'), { cells });
      const details = [];
      if (empty) details.push(K.plural(empty, 'cell is', 'cells are') + ' still empty.');
      if (bad) details.push(K.plural(bad, 'cell does', 'cells do') + ' not hold a number.');
      if (x.rowsSumTo !== undefined) {
        const off = x.answer.map((r, i) => i).filter((i) => { const s = x.answer[i].reduce((t, _, j) => { const g = K.matrixCells(x).some(([p, q]) => p === i && q === j) ? num(a[i] && a[i][j]) : x.answer[i][j]; return t + (g === null || Number.isNaN(g) ? NaN : g); }, 0); return Number.isFinite(s) && Math.abs(s - x.rowsSumTo) > Math.max(tol * x.answer[i].length, 0.011); });
        if (off.length) details.push('Row' + (off.length > 1 ? 's ' : ' ') + off.map((i) => x.rows[i]).join(', ') + (off.length > 1 ? ' do' : ' does') + ' not add up to ' + x.rowsSumTo + ': each row of weights must.');
      } else if (x.total !== undefined) {
        let s = 0; x.answer.forEach((r, i) => r.forEach((v, j) => { const g = K.matrixCells(x).some(([p, q]) => p === i && q === j) ? num(a[i] && a[i][j]) : v; s += g === null || Number.isNaN(g) ? 0 : g; }));
        if (!empty && Math.abs(s - x.total) > 1e-9) details.push('Your cells add up to ' + f(s) + ', but there are ' + x.total + ' examples in all.');
      }
      return { ok: false, cells, message: K.plural(wrong, 'cell is', 'cells are') + ' not right yet (marked in red).', details };
    },
    lint(x) {
      const p = [];
      if (!Array.isArray(x.answer) || !x.answer.length || !x.answer.every((r) => Array.isArray(r) && r.length === x.answer[0].length && r.every((v) => Number.isFinite(v)))) p.push('answer must be a rectangular matrix of numbers');
      else {
        if (!x.rows || x.rows.length !== x.answer.length) p.push('rows must name each row of the answer');
        if (!x.cols || x.cols.length !== x.answer[0].length) p.push('cols must name each column of the answer');
        if (x.rowsSumTo !== undefined) x.answer.forEach((r, i) => { const s = r.reduce((a, b) => a + b, 0); if (Math.abs(s - x.rowsSumTo) > Math.max((x.tolerance || 0) * r.length, 0.011)) p.push('row ' + i + ' of the answer adds up to ' + f(s) + ', not ' + x.rowsSumTo); });
      }
      return p;
    },
    /** a wrong answer made from the solution when none is listed: one cell changed */
    wrongFrom(x) { const a = x.answer.map((r) => r.map(String)); const [i, j] = K.matrixCells(x)[0]; a[i][j] = String(+x.answer[i][j] + Math.max(1, 3 * (x.tolerance || 0))); return [a]; },
  };

  /**
   * a plot as drawn: one with a single feature (no yr, or points given as numbers or one-number rows) becomes a strip,
   * its points spread a little up and down where they would overlap, with no y axis
   */
  K.plotOf = function (x) {
    const P = Object.assign({}, x.plot || {});
    const one = P.strip || !P.yr || (P.points && P.points.length && P.points.every((p) => typeof p === 'number' || (Array.isArray(p) && p.length === 1)));
    if (!one) return P;
    const xs = (P.points || []).map((p) => (typeof p === 'number' ? p : p[0])), span = (P.xr[1] - P.xr[0]) || 1;
    const order = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), ys = xs.map(() => 0.5);
    let run = 0;
    order.forEach(([v, i], k) => { run = k && v - order[k - 1][0] < 0.03 * span ? run + 1 : 0; ys[i] = run ? 0.5 + (run % 2 ? 1 : -1) * 0.18 * Math.ceil(run / 2) : 0.5; });
    return Object.assign(P, { strip: true, yr: [0, 1], yname: '', points: P.points ? xs.map((v, i) => [v, Math.max(0.08, Math.min(0.92, ys[i]))]) : undefined });
  };

  // ---- a point on a plot -----------------------------------------------------------------------------------------
  /** the reader's click as {x, y} (xy), {x} or {y}, or {i} (pick) */
  K.kinds.point = {
    check(x, a) {
      const mode = x.mode || 'xy', P = K.plotOf(x);
      if (!a || (mode === 'pick' ? a.i === undefined || a.i === null : (mode !== 'y' && !Number.isFinite(a.x)) || (mode !== 'x' && !Number.isFinite(a.y)))) return { ok: false, message: mode === 'pick' ? 'Click a point on the plot first.' : 'Click on the plot first.' };
      if (mode === 'pick') {
        const want = [].concat(x.answer);
        if (want.includes(a.i)) return right(x);
        const why = x.whyNot && x.whyNot[a.i];
        const name = (P.labels && P.labels[a.i]) || (P.strip ? 'the point at ' + (P.xname || 'x') + ' = ' + f(P.points[a.i][0]) : 'the point at (' + f(P.points[a.i][0]) + ', ' + f(P.points[a.i][1]) + ')');
        return { ok: false, message: 'Not quite: you picked ' + name + '.' + (why ? ' ' + why : '') };
      }
      const tol = x.tolerance !== undefined ? x.tolerance : 0.1;
      const w = x.answer;
      if (mode === 'x' || mode === 'y') {
        const g = mode === 'x' ? a.x : a.y, t = typeof w === 'number' ? w : w[mode === 'x' ? 0 : 1], d = g - t;
        if (Math.abs(d) <= tol) return right(x, 'You put it at ' + f(g) + '; it is at ' + f(t) + '.');
        return { ok: false, message: 'Not quite: you put it at ' + (P[mode + 'name'] || mode) + ' = ' + f(g) + ', which is ' + (d > 0 ? 'too high' : 'too low') + ' by ' + f(Math.abs(d)) + '.' + (x.whyNot ? ' ' + x.whyNot : '') };
      }
      const dx = a.x - w[0], dy = a.y - w[1], dist = Math.hypot(dx, dy);
      if (dist <= tol) return right(x, 'You clicked (' + f(a.x) + ', ' + f(a.y) + '); it is at (' + f(w[0]) + ', ' + f(w[1]) + ').');
      const dir = [Math.abs(dy) > tol / 2 ? (dy > 0 ? 'too high' : 'too low') : '', Math.abs(dx) > tol / 2 ? (dx > 0 ? 'too far right' : 'too far left') : ''].filter(Boolean).join(' and ');
      return { ok: false, message: 'Not quite: your point (' + f(a.x) + ', ' + f(a.y) + ') is ' + f(dist) + ' away' + (dir ? ', ' + dir : '') + '.' + (x.whyNot ? ' ' + x.whyNot : '') };
    },
    lint(x) {
      const p = [], mode = x.mode || 'xy';
      if (!x.plot || !x.plot.xr) return ['plot needs xr (and yr, unless it is a one-feature strip)'];
      const P = K.plotOf(x);
      if (P.strip && (mode === 'xy' || mode === 'y')) p.push('a one-feature strip takes mode x or pick');
      if (!['xy', 'x', 'y', 'pick'].includes(mode)) p.push('mode must be xy, x, y or pick');
      if (mode === 'pick' && !(P && P.points && [].concat(x.answer).every((i) => i >= 0 && i < P.points.length))) p.push('the answer must index a point of plot.points');
      if (mode !== 'pick' && P && P.xr) { const w = typeof x.answer === 'number' ? (mode === 'y' ? [P.xr[0], x.answer] : [x.answer, P.yr[0]]) : x.answer; if (!w || w[0] < P.xr[0] || w[0] > P.xr[1] || w[1] < P.yr[0] || w[1] > P.yr[1]) p.push('the answer lies outside the plot'); }
      return p;
    },
  };

  // ---- a sketched curve ------------------------------------------------------------------------------------------
  /** a curve [[x, y]..] read at x (straight lines between its points; null outside it) */
  K.curveAt = (c, xv) => {
    if (!c.length || xv < c[0][0] - 1e-9 || xv > c[c.length - 1][0] + 1e-9) return null;
    for (let i = 1; i < c.length; i++) if (xv <= c[i][0] + 1e-12) { const [x0, y0] = c[i - 1], [x1, y1] = c[i]; return x1 === x0 ? y1 : y0 + ((y1 - y0) * (xv - x0)) / (x1 - x0); }
    return c[c.length - 1][1];
  };
  /** a sketch as the reader drew it: sorted by x, one y per x */
  K.cleanSketch = (pts) => { const s = (pts || []).filter((p) => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1])).slice().sort((a, b) => a[0] - b[0]); const out = []; for (const p of s) if (!out.length || p[0] > out[out.length - 1][0]) out.push([p[0], p[1]]); return out; };
  K.kinds['predict-curve'] = {
    check(x, a) {
      const sk = K.cleanSketch(a), w = x.answer, P = x.plot || {};
      if (sk.length < 2) return { ok: false, message: 'Draw the curve first: drag across the plot.' };
      const x0 = w[0][0], x1 = w[w.length - 1][0], span = x1 - x0;
      const cover = (Math.min(x1, sk[sk.length - 1][0]) - Math.max(x0, sk[0][0])) / span;
      if (cover < 0.8) return { ok: false, message: 'Draw across the whole range, from ' + (P.xname || 'x') + ' = ' + f(x0) + ' to ' + f(x1) + ': your sketch covers ' + Math.round(Math.max(0, cover) * 100) + '% of it.' };
      // compare in thirds, so the message can say where the sketch is off
      const N = 30, gaps = [];
      for (let k = 0; k <= N; k++) { const xv = x0 + (span * k) / N, g = K.curveAt(sk, xv), t = K.curveAt(w, xv); if (g !== null && t !== null) gaps.push([xv, g - t]); }
      const mean = gaps.reduce((s, [, d]) => s + Math.abs(d), 0) / gaps.length;
      const tol = x.tolerance !== undefined ? x.tolerance : 0.1 * ((P.yr && P.yr[1] - P.yr[0]) || 1);
      const shape = [];
      for (const c of x.checks || []) {
        const ys = sk.map((p) => p[1]);
        if (c === 'rising' && !(ys[ys.length - 1] > ys[0])) shape.push('it should rise from left to right');
        if (c === 'falling' && !(ys[ys.length - 1] < ys[0])) shape.push('it should fall from left to right');
        if (c === 'peak') { const m = ys.indexOf(Math.max(...ys)); if (m === 0 || m === ys.length - 1) shape.push('it should rise to a peak and come down again'); }
        if (c === 'valley') { const m = ys.indexOf(Math.min(...ys)); if (m === 0 || m === ys.length - 1) shape.push('it should fall to a lowest point and rise again'); }
      }
      if (mean <= tol && !shape.length) return right(x, 'Your sketch is within ' + f(mean) + ' of the curve on average.');
      const thirds = ['On the left', 'In the middle', 'On the right'].map((name, k) => { const part = gaps.filter(([xv]) => xv >= x0 + (span * k) / 3 - 1e-9 && xv <= x0 + (span * (k + 1)) / 3 + 1e-9); const d = part.reduce((s, [, v]) => s + v, 0) / Math.max(1, part.length); return [name, d]; });
      const details = thirds.filter(([, d]) => Math.abs(d) > tol).map(([name, d]) => name + ', your sketch is ' + (d > 0 ? 'too high' : 'too low') + ' by about ' + f(Math.abs(d)) + '.');
      return { ok: false, message: 'Not quite: ' + (shape.length ? shape.join(', and ') + '; and your sketch' : 'your sketch') + ' is ' + f(mean) + ' away from the curve on average; within ' + f(tol) + ' counts.', details: details.concat(x.whyNot ? [x.whyNot] : []) };
    },
    lint(x) {
      const p = [];
      if (!x.plot || !x.plot.xr || !x.plot.yr) p.push('plot needs xr and yr');
      if (!Array.isArray(x.answer) || x.answer.length < 2) p.push('answer must be a curve of at least two points');
      else if (K.cleanSketch(x.answer).length !== x.answer.length) p.push('the answer curve must have increasing x');
      return p;
    },
  };
})(globalThis.Nooral = globalThis.Nooral || {});
