// Part 3, least squares as a projection. With three observations the targets y are one point of 3-D space, one
// axis per observation. Every prediction b·1 + w·x lies in the plane spanned by the column of ones and the column
// x; least squares picks the point of that plane nearest y, so the residual meets the plane at a right angle.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n;
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const add = (a, b, s) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
  const len = (a) => Math.sqrt(dot(a, a));
  const num = (v) => +String(v).replace('−', '-');
  const listCheck = (what) => (v) => (!Array.isArray(v) || v.length !== 3 ? what + ': exactly three numbers' : v.some((x) => !isFinite(num(x)) || x === '') ? what + ': numbers only' : v.some((x) => Math.abs(num(x)) > 6) ? what + ': each between −6 and 6, so the picture stays readable' : null);

  S.topic({
    id: 'ls-projection', family: 'p03-ls', order: 20, title: 'Least squares as a projection',
    blurb: 'Write the three targets as one vector y in 3-D, one axis per observation. A prediction b·1 + w·x is a mix of two vectors, the column of ones and the column of inputs, so all the predictions a line can make fill a plane: the column space. The best one is the foot of the perpendicular from y to that plane. The residual is perpendicular to both columns, and those two right angles are the normal equations.',
    inputs: [
      { id: 'x', kind: 'list', label: 'inputs x (three)', def: ['0', '1', '2'], check: listCheck('x') },
      { id: 'y', kind: 'list', label: 'targets y (three)', def: ['1', '3', '2'], check: listCheck('y') },
      { id: 'az', kind: 'slider', label: 'turn', def: 0, min: -90, max: 90, step: 5, fmt: (v) => (v > 0 ? '+' : '') + v + '°' },
      { id: 'el', kind: 'slider', label: 'tilt', def: 22, min: 5, max: 60, step: 1, fmt: (v) => v + '°' },
    ],
    check: (inp) => { const x = inp.x.map(num); return x[0] === x[1] && x[1] === x[2] ? 'x: the three inputs must not all be equal, or the columns 1 and x point the same way' : null; },
    presets: [
      { label: 'Three points off a line', values: { x: ['0', '1', '2'], y: ['1', '3', '2'], az: 0, el: 22 } },
      { label: 'Already on a line: y in the plane', values: { x: ['0', '1', '2'], y: ['1', '2', '3'], az: 0, el: 22 } },
      { label: 'A big residual', values: { x: ['0', '1', '2'], y: ['3', '−1', '3'], az: 0, el: 25 } },
      { label: 'Uneven inputs', values: { x: ['−1', '0', '3'], y: ['0', '2', '1'], az: 0, el: 20 } },
      { label: 'Seen from above', values: { x: ['0', '1', '2'], y: ['1', '3', '2'], az: -30, el: 58 } },
    ],
    counters: [{ id: 'dist', label: '‖residual‖' }],
    run(inp) {
      const xs = inp.x.map(num), ys = inp.y.map(num);
      const m = ML().LinearRegression().fit(xs.map((v) => [v]), ys), solve = m.trace.find((s) => s.kind === 'solve');
      const w = solve.coef[0], b = solve.intercept, one = [1, 1, 1], x = xs, y = ys;
      const yh = add(one.map((v) => v * b), x, w), r = [y[0] - yh[0], y[1] - yh[1], y[2] - yh[2]];
      const ybar = (y[0] + y[1] + y[2]) / 3, flat = [ybar, ybar, ybar], dFlat = len([y[0] - ybar, y[1] - ybar, y[2] - ybar]), dist = len(r);
      const r1 = dot(r, one), rx = dot(r, x);
      const F = new S.Frames(['dist']);

      // the camera: fit everything drawn into a box about 440 wide
      // the turn is measured from a view chosen for these columns: the plane seen at a slant (neither edge-on nor
      // face-on) and the three axes well spread on the screen
      const nrm = [one[1] * x[2] - one[2] * x[1], one[2] * x[0] - one[0] * x[2], one[0] * x[1] - one[1] * x[0]], nl = len(nrm);
      let bestAz = 35, best = -Infinity;
      for (let a = -180; a < 180; a += 5) {
        const ar = (a * Math.PI) / 180, er = (inp.el * Math.PI) / 180, d = [Math.sin(ar) * Math.cos(er), Math.cos(ar) * Math.cos(er), Math.sin(er)];
        const Pa = H.view3(a, inp.el, 1, 0, 0), tips = [0, 1, 2].map((k) => { const e = [0, 0, 0]; e[k] = 1; return Pa(e); }), ang = tips.map((q) => Math.atan2(q[1], q[0]));
        let spread = Math.PI;
        for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) { let g = Math.abs(ang[i] - ang[j]); if (g > Math.PI) g = 2 * Math.PI - g; spread = Math.min(spread, g); }
        const short = Math.min(...tips.map((q) => Math.hypot(q[0], q[1])));
        const score = -2 * Math.abs(Math.abs(dot(d, nrm)) / nl - 0.6) + spread / Math.PI - (short < 0.6 ? 3 * (0.6 - short) : 0);
        if (score > best + 1e-9) { best = score; bestAz = a; }
      }
      const az = bestAz + inp.az;
      const P0 = H.view3(az, inp.el, 1, 0, 0);
      const AX = Math.max(2, ...y.map(Math.abs), ...x.map(Math.abs), ...yh.map(Math.abs)) + 0.9;
      const span = [[0, 0, 0], y, yh, one, x, flat, [AX, 0, 0], [0, AX, 0], [0, 0, AX]];
      // the plane patch: s·1 + t·x over a range that covers the origin, ŷ and the flat guess
      // drawn on an orthonormal pair of the same plane (the ones, and x minus its mean), so the patch is square
      const u1 = one.map((v) => v / Math.sqrt(3)), xm = (x[0] + x[1] + x[2]) / 3, xc = x.map((v) => v - xm), u2 = xc.map((v) => v / len(xc));
      const inPl = [[0, 0, 0], yh, one, x, [ybar, ybar, ybar]].map((v) => [dot(v, u1), dot(v, u2)]);
      const sR = H.span(inPl.map((q) => q[0]), 0), tR = H.span(inPl.map((q) => q[1]), 0), pad = 0.35 * Math.max(sR[1] - sR[0], tR[1] - tR[0], 1.5);
      const corners = [[sR[0] - pad, tR[0] - pad], [sR[1] + pad, tR[0] - pad], [sR[1] + pad, tR[1] + pad], [sR[0] - pad, tR[1] + pad]].map(([s, t]) => add(u1.map((v) => v * s), u2, t));
      const all = span.concat(corners).map(P0), bx = H.span(all.map((p) => p[0]), 0), by = H.span(all.map((p) => p[1]), 0);
      const sc = Math.min(390 / (bx[1] - bx[0]), 440 / (by[1] - by[0]));
      const P = H.view3(az, inp.el, sc, 40 - bx[0] * sc, 60 - by[0] * sc);
      const O = P([0, 0, 0]);
      const axes = [], axisLines = [];
      ['observation 1', 'observation 2', 'observation 3'].forEach((t, k) => {
        const e = [0, 0, 0]; e[k] = AX;
        const q = P(e);
        axes.push({ key: 'pj:ax' + k, kind: 'path', d: `M${O[0]},${O[1]} L${q[0]},${q[1]}`, cls: 'pl-axis3', pathAudit: 'edge-x', layer: 'back', z: 2 });
        axisLines.push([O, q]);
      });
      const plane = { key: 'pj:plane', kind: 'path', d: corners.map(P).map((p, i) => (i ? 'L' : 'M') + p[0] + ',' + p[1]).join(' ') + ' Z', cls: 'pl-plane', audit: 'ignore', layer: 'back', z: 3 };
      const arrow = (key, from, to, cls) => { const a = P(from), c = P(to); return H.arrow(key, a[0], a[1], c[0], c[1], cls, { head: 11 }); };
      // spots for a label at the tip of an arrow from the origin: beyond the tip along the arrow first, then around it
      const tipSpots = (v, text, size) => {
        const q = P(v), d = Math.hypot(q[0] - O[0], q[1] - O[1]) || 1, ux = (q[0] - O[0]) / d, uy = (q[1] - O[1]) / d, hw = S.geom.textWidth(text, size) / 2 + 6, hh = size * 0.6 + 4;
        const out = [];
        for (const r of [1, 1.6, 2.3]) {
          const cx = q[0] + ux * (hw * Math.abs(ux) + hh * Math.abs(uy) + 4) * r, cy = q[1] + uy * (hw * Math.abs(ux) + hh * Math.abs(uy) + 4) * r;
          out.push([cx, cy + size * 0.35, 'middle']);
        }
        for (const [dx, dy] of [[1, -1], [-1, -1], [1, 1], [-1, 1], [0, -1.4], [0, 1.6]]) out.push([q[0] + dx * (hw + 4), q[1] + dy * (hh + 2) + size * 0.35, 'middle']);
        return out;
      };
      const rightAngle = () => {
        if (dist < 0.08) return [];
        const u = r.map((v) => (v / dist) * 0.28), inPlane = len(yh) > 0.2 ? yh.map((v) => (-v / len(yh)) * 0.28) : one.map((v) => v * 0.16);
        const a = P(add(yh, u, 1)), c = P(add(add(yh, u, 1), inPlane, 1)), d = P(add(yh, inPlane, 1));
        return [{ key: 'pj:ra', kind: 'path', d: `M${a[0]},${a[1]} L${c[0]},${c[1]} L${d[0]},${d[1]}`, cls: 'pl-right', audit: 'ignore', layer: 'lines' }];
      };
      // the line through the three points, small, at the right
      const inset = () => {
        const xr = H.nice(xs.concat([Math.min(...xs) - 0.5, Math.max(...xs) + 0.5]), 0), yr = H.nice(ys.concat(yh), 0.15);
        const ax = H.axes({ prefix: 'pi', x: 530, y: 60, w: 220, h: 170, xr, yr, xticks: 1, yticks: yr[1] - yr[0] > 6 ? 2 : 1, xname: 'x', yname: 'y' });
        const items = [...ax.items, H.title('pi:t', 500, 30, 'The same fit as a line')];
        items.push(H.seg('pi:line', ax.X(xr[0]), ax.Y(w * xr[0] + b), ax.X(xr[1]), ax.Y(w * xr[1] + b), 'pl-cut act'));
        xs.forEach((v, i) => { if (Math.abs(ax.Y(y[i]) - ax.Y(yh[i])) > 0.6) items.push(H.seg('pi:r' + i, ax.X(v), ax.Y(yh[i]), ax.X(v), ax.Y(y[i]), 'pl-seg res ' + (r[i] >= 0 ? 'pos' : 'neg'))); });
        items.push(...S.Kit.scatter.points(ax, 'pi:p', xs.map((v, i) => [v, y[i]]), [0, 0, 0], { r: 5 }));
        return items;
      };
      const eqX = 500, eq = (lines, y0) => H.eqs('pj:eq', eqX, y0 || 300, lines, 15, 28);
      const meta = { x: xs, y: ys, w, b, yh, r, dist, dFlat, ybar, r1, rx, sse: dist * dist };
      const vecs = (which) => {
        const items = [], labels = [], lines = axisLines.slice();
        const v = (key, from, to, cls, text, lcls) => { items.push(arrow(key, from, to, cls)); lines.push([P(from), P(to)]); if (text) labels.push({ key: key + ':l', text, cls: 'pl-lab ' + (lcls || ''), size: 14, at: tipSpots(to, text, 14), need: key === 'pj:y' }); };
        if (which.cols) { v('pj:one', [0, 0, 0], one, 'muted', '1', 'muted'); v('pj:x', [0, 0, 0], x, 'muted', 'x', 'muted'); }
        if (which.flat) v('pj:flat', [0, 0, 0], flat, 'c2', 'ȳ·1', 'c2');
        if (which.yh) v('pj:yh', [0, 0, 0], yh, 'act', 'ŷ', 'act');
        v('pj:y', [0, 0, 0], y, '', 'y', '');
        if (which.r && dist > 0.05) { const a = P(yh), c = P(y); items.push(H.arrow('pj:r', a[0], a[1], c[0], c[1], 'hot', { head: 9 })); lines.push([a, c]); }
        if (which.flatR && dFlat > 0.05) { const a = P(flat), c = P(y); items.push({ key: 'pj:fr', kind: 'path', d: `M${a[0]},${a[1]} L${c[0]},${c[1]}`, cls: 'pl-seg dash c2', pathAudit: 'edge-x', layer: 'lines' }); lines.push([a, c]); }
        const axLab = ['obs 1', 'obs 2', 'obs 3'].map((t, k) => {
          const e = [0, 0, 0]; e[k] = AX;
          const q = P(e), d = Math.hypot(q[0] - O[0], q[1] - O[1]) || 1, px = -(q[1] - O[1]) / d, py = (q[0] - O[0]) / d, hw = S.geom.textWidth(t, 12) / 2 + 8;
          const beside = [0.85, 0.65, 0.45].flatMap((f) => [1, -1].map((sg) => [O[0] + (q[0] - O[0]) * f + sg * px * (hw * Math.abs(py) + 10 * Math.abs(px) + 4), O[1] + (q[1] - O[1]) * f + sg * py * (hw * Math.abs(py) + 10 * Math.abs(px) + 4) + 4, 'middle']));
          return { key: 'pj:axl' + k, text: t, cls: 'tick', size: 12, at: tipSpots(e, t, 12).slice(0, 2).concat(beside, tipSpots(e, t, 12).slice(3)) };
        });
        const order = (L) => (L.key === 'pj:y:l' ? 0 : L.key === 'pj:yh:l' ? 1 : 2);
        labels.sort((a, b) => order(a) - order(b));
        return items.concat(H.place(labels.concat(axLab), [{ x: 490, y: -400, w: 600, h: 1400 }, { x: -600, y: -600, w: 2000, h: 600 }, { x: -600, y: -400, w: 605, h: 1400 }, { x: -600, y: 545, w: 2000, h: 600 }], lines));
      };
      const base = () => [...axes, H.title('pj:t', 20, 14, 'Three observations, one point in 3-D')];
      const dot0 = { key: 'pj:o', kind: 'dot', x: O[0], y: O[1], r: 3, cls: 'pl-pt muted', audit: 'overlap-ok' };

      F.add({ items: [...base(), dot0, ...vecs({}), ...inset(), ...eq([['y = ' + H.vec(y), '']])], say: 'The three targets y = ' + H.vec(y) + ' are a single arrow in 3-D: its first coordinate is observation 1 (the axis obs 1), its second observation 2, its third observation 3.', meta: Object.assign({ phase: 'y' }, meta) });
      F.add({ items: [...base(), dot0, ...vecs({ cols: true }), ...inset(), ...eq([['y = ' + H.vec(y), ''], ['1 = (1, 1, 1)', 'muted'], ['x = ' + H.vec(x), 'muted']])], say: 'The columns of the design matrix are arrows too: the ones, 1 = (1, 1, 1), which carries the intercept, and the inputs x = ' + H.vec(x) + ', which carries the slope.', meta: Object.assign({ phase: 'cols' }, meta) });
      F.add({ items: [...base(), plane, dot0, ...vecs({ cols: true, flat: true, flatR: true }), ...inset(), ...eq([['b·1 + w·x: the plane', 'act'], ['flat guess ȳ·1 = ' + H.vec(flat), 'c2'], ['‖y − ȳ·1‖ = ' + n$(dFlat), 'c2']])], say: 'Every prediction b·1 + w·x lies in the shaded plane they span. The flat guess ȳ·1 = ' + H.vec(flat) + ' is one point of it, at distance ' + n$(dFlat) + ' from y. Least squares looks for the nearest point.', meta: Object.assign({ phase: 'plane' }, meta) });
      F.set('dist', +dist.toFixed(3));
      F.add({ items: [...base(), plane, dot0, ...vecs({ cols: true, yh: true, r: true }), ...rightAngle(), ...inset(), ...eq([['ŷ = ' + n$(b) + '·1' + H.term(w, '·x', false), 'act'], ['  = ' + H.vec(yh), 'act'], ['r = y − ŷ = ' + H.vec(r), 'hot']])], say: 'The engine solves for b = ' + n$(b) + ' and w = ' + n$(w) + ', so ŷ = ' + H.vec(yh) + '. The residual r = ' + H.vec(r) + ' drops straight down to the plane; its length is ' + n$(dist) + ', shorter than the ' + n$(dFlat) + ' of the flat guess.', meta: Object.assign({ phase: 'project' }, meta) });
      F.add({ items: [...base(), plane, dot0, ...vecs({ cols: true, yh: true, r: true }), ...rightAngle(), ...inset(), ...eq([['r · 1 = ' + n$(r1), 'good'], ['r · x = ' + n$(rx), 'good'], ['‖r‖² = ' + n$(dist * dist) + ' = Σ r²', 'hot']])], say: 'The right angle is the normal equations: r · 1 = ' + n$(r1) + ' (the residuals sum to zero) and r · x = ' + n$(rx) + '. And ‖r‖² = ' + n$(dist * dist) + ' is exactly the sum of squared residuals of the line at the right.', meta: Object.assign({ phase: 'normal' }, meta) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The projection', text: ['column space: { b·1 + w·x }', 'ŷ = X (XᵀX)⁻¹ Xᵀ y', 'r = y − ŷ ⟂ every column:', '  Xᵀ r = 0  (the normal equations)', '‖r‖² = Σ rᵢ² (the least squares)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'plane', pick: 'option', ask: 'At what angle will the residual y − ŷ meet the plane?', options: ['a right angle', '45°', 'it depends on the data', 'it lies in the plane'], answer: 'a right angle', explain: 'The nearest point of a plane is the foot of the perpendicular.' },
      { at: (f) => f.meta.phase === 'cols', pick: 'option', ask: 'Will the best prediction ŷ be nearer to y than the flat guess ȳ·1?', options: (fr) => ['yes, or equally near', 'no, farther'], answer: 'yes, or equally near', explain: (fr) => 'ŷ is the nearest point of the plane; ȳ·1 is in the plane too. Here ' + n$(fr[0].meta.dist) + ' against ' + n$(fr[0].meta.dFlat) + '.' },
    ],
    quiz: [
      { q: 'In the projection picture, how many axes does the space have?', options: ['one per observation', 'one per feature', 'always two', 'one per class'], answer: 0, why: 'y is a vector with one entry per observation.' },
      { q: 'What is the column space of X?', options: ['every vector Xθ the model can produce', 'the set of data points', 'the rows of X', 'the residuals'], answer: 0, why: 'All predictions are combinations of the columns.' },
      { q: 'Xᵀr = 0 says that the residual is…', options: ['perpendicular to every column', 'zero', 'parallel to y', 'as long as y'], answer: 0, why: 'Each entry of Xᵀr is the dot product of one column with r.' },
      { q: 'If y already lies in the plane, the residual is…', options: ['zero: the line fits every point', 'perpendicular but long', 'equal to y', 'undefined'], answer: 0, why: 'The nearest point of the plane is y itself.' },
      { q: 'Why can a 3-point fit never be drawn this way with 4 points?', options: ['y would live in 4-D', 'least squares needs 3 points', 'the plane becomes a line', 'it can'], answer: 0, why: 'The picture has one axis per observation; the algebra is the same in any dimension.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
