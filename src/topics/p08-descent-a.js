// Part 8, gradient descent (1): a loss surface seen from the side as a landscape whose camera turns, then from
// above as a contour map with the downhill arrows; and gradient descent one step at a time, each step the
// learning rate times the gradient, run by the engine's optimiser.
(function (S) {
  'use strict';
  const P = S.P08, H = () => S.Kit.ml;
  const deg = Math.PI / 180;

  /** the lowest point near p: Newton's method on the gradient, from the best grid point near p */
  P.newtonMin = function (surf, p) {
    let [u, v] = p;
    for (let it = 0; it < 60; it++) {
      const g = surf.g(u, v), h = surf.h(u, v), det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
      if (Math.abs(det) < 1e-14) break;
      const du = (h[1][1] * g[0] - h[0][1] * g[1]) / det, dv = (h[0][0] * g[1] - h[1][0] * g[0]) / det;
      u -= du; v -= dv;
      if (Math.hypot(du, dv) < 1e-13) break;
    }
    return [u, v];
  };
  P.globalMin = function (surf) {
    if (surf.min.length) return surf.min.map((m) => m.slice()).sort((a, b) => surf.f(...a) - surf.f(...b))[0];
    let best = null;
    for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const u = surf.xr[0] + ((surf.xr[1] - surf.xr[0]) * i) / 60, v = surf.yr[0] + ((surf.yr[1] - surf.yr[0]) * j) / 60, f = surf.f(u, v); if (!best || f < best[2]) best = [u, v, f]; }
    return P.newtonMin(surf, [best[0], best[1]]);
  };
  /** a contour box of width w whose height keeps the surface's aspect */
  const boxFor = (surf, x, y, w, hmax) => { const r = (surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0]); const h = Math.min(hmax, w * r); return { x, y, w: h / r, h }; };

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'opt-surface', family: 'p08-descent', order: 10, title: 'Loss surfaces, from the side and from above',
    blurb: 'A model with two parameters has a loss for every pair of values: a landscape over the plane. Training is a walk downhill on it. Seen from the side the landscape has bowls, ravines, saddles and pits; seen from above it becomes a contour map, where each line joins points of equal loss and the gradient points straight across the lines, uphill.',
    inputs: [{ id: 'surf', kind: 'choice', label: 'Surface', def: 'ravine', options: P.surfOptions(['bowl', 'ravine', 'tilted', 'saddle', 'rosen', 'bumpy', 'plateau']) }],
    presets: [
      { label: 'A round bowl', values: { surf: 'bowl' } },
      { label: 'A long ravine', values: { surf: 'ravine' } },
      { label: 'A saddle', values: { surf: 'saddle' } },
      { label: 'Rosenbrock’s banana', values: { surf: 'rosen' } },
      { label: 'Bumpy, many pits', values: { surf: 'bumpy' } },
      { label: 'A plateau', values: { surf: 'plateau' } },
    ],
    counters: [{ id: 'turn', label: 'camera turns' }],
    run(inp) {
      const surf = P.SURF[inp.surf], F = H().frames(['turn']), q = F.q;
      const lo = P.gridMin(surf);
      let hi = -Infinity;
      for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) hi = Math.max(hi, surf.f(surf.xr[0] + ((surf.xr[1] - surf.xr[0]) * i) / 40, surf.yr[0] + ((surf.yr[1] - surf.yr[0]) * j) / 40));
      // heights squeezed on a log scale, as the shading of the map is, so the floor shows as well as the walls
      const lg = (f) => Math.log(1 + Math.max(0, f - lo)), LH = lg(hi);
      const nz = (u, v) => lg(surf.f(u, v)) / LH;
      const toU = (x) => surf.xr[0] + ((x + 1) / 2) * (surf.xr[1] - surf.xr[0]), toV = (y) => surf.yr[0] + ((y + 1) / 2) * (surf.yr[1] - surf.yr[0]);
      const fromU = (u) => (2 * (u - surf.xr[0])) / (surf.xr[1] - surf.xr[0]) - 1, fromV = (v) => (2 * (v - surf.yr[0])) / (surf.yr[1] - surf.yr[0]) - 1;
      const land = (az, mark) => {
        const V = H().view3({ cx: 205, cy: 262, s: 130, az: az * deg, el: 32 * deg }), items = [H().head('t3', 20, 14, 'From the side')];
        items.push(...H().surface3('s3', V, (x, y) => nz(toU(x), toV(y)), { xr: [-1, 1], yr: [-1, 1], n: 20 }));
        items.push(...P.box3('a3', V, 1.15));
        for (const [k, p, cls] of mark || []) { const pp = V([fromU(p[0]), fromV(p[1]), nz(p[0], p[1])]); items.push({ key: 'm3' + k, kind: 'dot', x: pp.x, y: pp.y, r: 6, cls, audit: 'overlap-ok', layer: 'top' }); }
        return items;
      };
      const bx = boxFor(surf, 470, 52, 300, 300), ax = P.axes('sf', surf, bx, { xticks: 1, yticks: 1 });
      const map = () => [H().head('t2', 470, 14, 'From above'), ...ax.items, ...P.contour(ax, 'sf:c', surf)];
      const AZ = [-60, -35, -10, 15, 40];
      AZ.forEach((az, k) => {
        if (k) F.count('turn');
        const say = k === 0
          ? `${surf.label[0].toUpperCase() + surf.label.slice(1)}: the loss L(u, v) drawn as a height over the plane of the two parameters u and v. Over this box it runs from ${q.sg(lo, 3)} to ${q.sg(hi, 3)}; the heights are drawn on a log scale, so the floor shows as well as the walls.`
          : k === 1 ? 'The camera turns. A surface is easier to read from several sides: the deep parts, the steep walls and the flat stretches move against each other.'
            : k === AZ.length - 1 ? 'Beside it, the same surface seen from above: a contour map. Each line joins points of equal loss, and the shading is darker where the loss is lower. A walk downhill follows the steepest slope under its feet, across the lines, not a straight line to the lowest point.' : 'Still turning.';
        F.push([...land(az), ...map()], say, { phase: k === AZ.length - 1 ? 'map' : 'turn', lo, hi, az });
      });
      // the downhill arrows over the map
      const arrows = [];
      let gmax = 0;
      const G = [];
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { const u = surf.xr[0] + ((i + 0.5) * (surf.xr[1] - surf.xr[0])) / 5, v = surf.yr[0] + ((j + 0.5) * (surf.yr[1] - surf.yr[0])) / 5, g = surf.g(u, v); G.push([u, v, g]); gmax = Math.max(gmax, Math.hypot(g[0], g[1])); }
      G.forEach(([u, v, g], k) => {
        const L = Math.hypot(g[0], g[1]); if (L < 1e-12) return;
        const len = 8 + 18 * Math.sqrt(L / gmax), a = [ax.X(u), ax.Y(v)], sx = (-g[0] / L) * (bx.w / (surf.xr[1] - surf.xr[0])), sy = (g[1] / L) * (bx.h / (surf.yr[1] - surf.yr[0])), sl = Math.hypot(sx, sy);
        const it = P.arrow('ga' + k, a, [a[0] + (sx / sl) * len, a[1] + (sy / sl) * len], 'down thin', { head: 7 });
        if (it) arrows.push(it);
      });
      const last = AZ[AZ.length - 1];
      F.push([...land(last), ...map(), ...arrows], `Each green arrow points downhill: the negative gradient, −∇L, the direction in which the loss falls fastest. Longer arrows mark steeper ground; the steepest arrow here is ${q.sg(gmax, 3)} long in loss per unit step.`, { phase: 'arrows', gmax });
      // one point: its gradient, at right angles to its level line
      const p = surf.start, g = surf.g(p[0], p[1]), gl = Math.hypot(g[0], g[1]), fp = surf.f(p[0], p[1]);
      const ga = [ax.X(p[0]), ax.Y(p[1])], gs = [(-g[0] / gl) * 46, (g[1] / gl) * 46 * (bx.h / (surf.yr[1] - surf.yr[0])) / (bx.w / (surf.xr[1] - surf.xr[0]))];
      const ptItems = [...land(last, [['p', p, 'opthead o-gd']]), ...map(), ...arrows.map((a) => Object.assign({}, a, { opacity: 0.35 })), P.arrow('gp', ga, [ga[0] + gs[0], ga[1] + gs[1]], 'grad', { head: 11 }), { key: 'sf:p', kind: 'dot', x: ga[0], y: ga[1], r: 6, cls: 'opthead o-gd', audit: 'overlap-ok', layer: 'top' }];
      const rd = (rows) => H().lines('rd', 470, bx.y + bx.h + 52, rows, 13, 21);
      F.push([...ptItems, ...rd([['at (' + H().fx(p[0]) + ', ' + H().fx(p[1]) + ')', 'strong'], ['L = ' + H().sg(fp, 4) + ',  ∇L = (' + H().sg(g[0], 3) + ', ' + H().sg(g[1], 3) + ')', 'legend']])],
        `Take the point (${q(p[0])}, ${q(p[1])}), where the loss is ${q.sg(fp, 4)}. Its gradient is ∇L = (${q.sg(g[0], 3)}, ${q.sg(g[1], 3)}), of length ${q.sg(gl, 3)}. The red arrow, −∇L, crosses the level line through the point at a right angle: along the line the loss does not change, so all of the slope lies across it.`, { phase: 'point', p, f: fp, g, gl });
      // the lowest point (or, on the saddle, the flat point that is not a minimum)
      const m = P.globalMin(surf), fm = surf.f(m[0], m[1]), ev = P.eig2(surf.h(m[0], m[1]));
      const minItems = [...land(last, [['p', p, 'opthead o-gd'], ['m', m, 'p8-min']]), ...map(), ...arrows.map((a) => Object.assign({}, a, { opacity: 0.35 })), { key: 'sf:m', kind: 'dot', x: ax.X(m[0]), y: ax.Y(m[1]), r: 7, cls: 'p8-min', audit: 'overlap-ok', layer: 'top' }];
      let say = `The lowest point is at (${q(m[0])}, ${q(m[1])}), with loss ${q.sg(fm, 4)}. There the gradient is zero, and the surface curves up in every direction: the Hessian’s two eigenvalues, ${q.sg(ev[0], 3)} and ${q.sg(ev[1], 3)}, are both positive.`;
      if (ev[0] / ev[1] > 8) say += ` One is ${q.sg(ev[0] / ev[1], 3)} times the other: steep across, flat along, the shape of a ravine that makes descent zigzag.`;
      const meta = { phase: 'min', m, f: fm, ev, ratio: ev[0] / ev[1] };
      if (surf.saddle) {
        const sp = surf.saddle, se = P.eig2(surf.h(sp[0], sp[1]));
        minItems.push({ key: 'sf:sd', kind: 'dot', x: ax.X(sp[0]), y: ax.Y(sp[1]), r: 7, cls: 'p8-sad', audit: 'overlap-ok', layer: 'top' });
        say += ` The ringed point (${q(sp[0])}, ${q(sp[1])}) is flat too, but it is a saddle: its eigenvalues are ${q.sg(se[0], 3)} and ${q.sg(se[1], 3)}, up one way and down the other.`;
        Object.assign(meta, { sp, se });
      }
      F.push([...minItems, ...rd([['lowest at (' + H().fx(m[0]) + ', ' + H().fx(m[1]) + ')', 'strong'], ['L = ' + H().sg(fm, 4) + ',  curvatures ' + H().sg(ev[0], 3) + ' and ' + H().sg(ev[1], 3), 'legend']])], say, meta);
      for (const f of F.list) f.narrowView = f.meta.phase === 'turn' ? { x: 0, y: 10, w: 420, h: 440 } : { x: 440, y: 10, w: 350, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: surf.label, text: [surf.formula, '∇L = (∂L/∂u, ∂L/∂v): steepest uphill', '−∇L: steepest downhill', 'level line: L = constant; ∇L is at right angles to it', 'at a minimum: ∇L = 0 and the Hessian’s eigenvalues are all > 0', 'at a saddle: ∇L = 0, eigenvalues of both signs'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'point', pick: 'option', ask: 'At a point on the map, which way does the downhill arrow −∇L point?', options: ['straight across the level line, at a right angle', 'along the level line', 'always straight at the lowest point', 'it depends on the learning rate'], answer: 'straight across the level line, at a right angle', explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'min', pick: 'option', ask: 'At the lowest point of the surface, how long is the gradient?', options: ['zero', 'as long as it is anywhere else', 'longest of anywhere', 'it has no gradient at all'], answer: 'zero', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'On a contour map of a loss, where is the ground steepest?', options: ['where the level lines are packed closest', 'where they are furthest apart', 'on the lowest line', 'it cannot be read from the map'], answer: 0, why: 'Equal steps in loss crowd together where the loss changes fastest.' },
      { q: 'The gradient of L at a point is (3, −4). Which way is steepest downhill?', options: ['(−3, 4)', '(3, −4)', '(4, 3)', '(0, 0)'], answer: 0, why: 'Downhill is −∇L.' },
      { q: 'How long is the gradient (3, −4)?', options: ['5', '7', '1', '−1'], answer: 0, why: '√(9 + 16) = 5.' },
      { q: 'At a saddle point, the gradient is…', options: ['zero, but the point is not a minimum', 'never zero', 'zero, and the point is the minimum', 'infinite'], answer: 0, why: 'The surface rises one way and falls the other: flat, but not a bottom.' },
      { q: 'A ravine has curvatures 0.2 along and 4 across. What makes it hard for gradient descent?', options: ['a step small enough for the steep direction crawls along the flat one', 'the gradient is zero everywhere', 'there is no minimum', 'it has many local minima'], answer: 0, why: 'One learning rate must suit both directions; the ratio of curvatures, 20, is what slows it.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'opt-gd', family: 'p08-descent', order: 20, title: 'Gradient descent, one step at a time',
    blurb: 'Gradient descent repeats one move: work out the gradient where you stand, and step against it, by the learning rate η times the gradient: θ ← θ − η∇L(θ). Steep ground gives long steps and flat ground short ones, so the walk slows as it nears the bottom, and in a ravine it zigzags from wall to wall.',
    inputs: [
      { id: 'surf', kind: 'choice', label: 'Surface', def: 'tilted', options: P.surfOptions(['bowl', 'ravine', 'tilted', 'rosen', 'saddle']) },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.3, min: 0.001, max: 2, step: 0.001, log: true, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 25, min: 3, max: 60 },
    ],
    presets: [
      { label: 'Tilted valley, η = 0.3', values: { surf: 'tilted', lr: 0.3, steps: 25 } },
      { label: 'Round bowl, η = 0.5', values: { surf: 'bowl', lr: 0.5, steps: 15 } },
      { label: 'Ravine: zigzag', values: { surf: 'ravine', lr: 0.45, steps: 40 } },
      { label: 'Banana, small steps', values: { surf: 'rosen', lr: 0.008, steps: 60 } },
      { label: 'Saddle: almost stuck', values: { surf: 'saddle', lr: 0.2, steps: 60 } },
      { label: 'Too big a step', values: { surf: 'tilted', lr: 1.05, steps: 20 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF[inp.surf], N = inp.steps, lr = inp.lr, F = H().frames(['step']), q = F.q;
      const r = P.run(surf, 'sgd', { lr }, N), pts = r.pts, n = pts.length - 1;
      const bx = boxFor(surf, 40, 30, 440, 430), ax = P.axes('gd', surf, bx, { xticks: 1, yticks: 1 });
      const off = P.gridMin(surf) < 0 ? -P.gridMin(surf) + 1e-3 : 0;
      const lv = r.loss.map((v) => v + off);
      const lax = P.logAxes('gl', { x: 560, y: 50, w: 210, h: 150 }, n, Math.max(1e-8, Math.min(...lv.filter(Number.isFinite))), Math.max(...lv.slice(0, 1).concat(lv.filter((v) => Number.isFinite(v) && v < 1e6))), { yname: off ? 'loss + ' + H().sg(off, 2) + ' (log scale)' : 'loss (log scale)' });
      const base = () => [...ax.items, ...P.contour(ax, 'gd:c', surf), P.start(ax, 'gd:s', pts[0]), ...lax.items];
      const show = [0, 1, 2, 3, 4, 5, 6];
      for (let k = 8; k <= n; k += Math.max(2, Math.round(n / 10))) show.push(k);
      if (!show.includes(n)) show.push(n);
      const shown = [...new Set(show.filter((k) => k <= n))];
      shown.forEach((k) => {
        F.set('step', k);
        const items = [...base(), ...P.path(ax, 'gd:r', pts, 'o-gd', k, { dots: 1 }), P.logLine(lax, 'gl:l', lv, 'o-gd', k)];
        const p = pts[k], out = P.left(ax, pts, k);
        const rows = [['step ' + k + ' of ' + n, 'strong'], ['(u, v) = (' + H().sg(p[0], 3) + ', ' + H().sg(p[1], 3) + ')', 'legend'], ['L = ' + H().sg(r.loss[k], 4), 'legend']];
        let say;
        if (k === 0) {
          say = `Start at (${q(p[0])}, ${q(p[1])}), where the loss is ${q.sg(r.loss[0], 4)}. With learning rate η = ${q.sg(lr, 3)}, each step moves by −η times the gradient.`;
          F.push([...items, ...H().lines('rd', 560, 250, rows, 13, 21)], say, { phase: 'start', k, p, loss: r.loss[0], lr });
          return;
        }
        const g = r.grads[k - 1], a = pts[k - 1], d = [p[0] - a[0], p[1] - a[1]];
        if (!out && k <= 6) {
          // the gradient (red, scaled to show its direction) and the step taken (accent)
          const A = [ax.X(a[0]), ax.Y(a[1])], B = [ax.X(p[0]), ax.Y(p[1])];
          const st = P.arrow('gd:st', A, B, 'step', { head: 9 });
          if (st) items.push(st);
        }
        rows.push(['∇L = (' + H().sg(g[0], 3) + ', ' + H().sg(g[1], 3) + ')', 'legend']);
        if (out || r.diverged && k === n) {
          say = `Step ${q.n(k)}: the steps have grown instead of shrinking, and the walk has left the map; the loss is now ${q.sg(r.loss[k], 3)}. The learning rate ${q.sg(lr, 3)} is too large for the steepest direction of this surface: every step overshoots the valley floor by more than it started.`;
          F.push([...items, ...H().lines('rd', 560, 250, rows.slice(0, 1).concat([['diverging: L = ' + H().sg(r.loss[k], 3), 'bad']]), 13, 21)], say, { phase: 'out', k, loss: r.loss[k], lr });
          return;
        }
        if (k <= 6) say = `Step ${q.n(k)}: at (${q.sg(a[0], 3)}, ${q.sg(a[1], 3)}) the gradient is (${q.sg(g[0], 3)}, ${q.sg(g[1], 3)}). Times η = ${q.sg(lr, 3)}, the step is (${q.sg(d[0], 3)}, ${q.sg(d[1], 3)}), and the loss goes from ${q.sg(r.loss[k - 1], 4)} to ${q.sg(r.loss[k], 4)}.`;
        else say = `After ${q.n(k)} steps the walk is at (${q.sg(p[0], 3)}, ${q.sg(p[1], 3)}) with loss ${q.sg(r.loss[k], 4)}; the last step was ${q.sg(Math.hypot(d[0], d[1]), 3)} long, because the gradient there is only ${q.sg(Math.hypot(g[0], g[1]), 3)}.`;
        if (k === n && !out) say += r.loss[k] < r.loss[0] * 1e-3 ? ' It has all but reached the bottom.' : ' It is still on its way: on flat ground the steps get short.';
        F.push([...items, ...H().lines('rd', 560, 250, rows, 13, 21)], say, { phase: 'step', k, p, a, g, d, step: Math.hypot(d[0], d[1]), gl: Math.hypot(g[0], g[1]), loss: r.loss[k], prev: r.loss[k - 1], lr });
      });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: bx.x + bx.w + 20, h: bx.y + bx.h + 40 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Gradient descent', text: [surf.formula, 'θ ← θ − η ∇L(θ)', 'the step is η × the gradient: long on steep ground, short on flat', 'too large an η overshoots and can diverge'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'step' && f.meta.k === 2 && fr[i + 1] && fr[i + 1].meta.phase === 'step', pick: 'option', ask: 'Will the next step be longer or shorter than this one?', options: ['longer', 'shorter'], answer: (fr, i) => (fr[i + 1].meta.step > fr[i].meta.step ? 'longer' : 'shorter'), explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which way will the first step go?', options: ['straight down the slope, against the gradient', 'straight towards the lowest point', 'along the level line', 'in a random direction'], answer: 'straight down the slope, against the gradient', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'With η = 0.1 and gradient (2, −1), what is the step?', options: ['(−0.2, 0.1)', '(0.2, −0.1)', '(−2, 1)', '(−0.1, 0.2)'], answer: 0, why: 'The step is −η∇L = −0.1 × (2, −1).' },
      { q: 'Why do the steps of gradient descent get shorter near a minimum, with η fixed?', options: ['the gradient shrinks towards zero there', 'η is lowered automatically', 'the loss is negative', 'they do not'], answer: 0, why: 'Each step is η times the gradient, and the gradient vanishes at the bottom.' },
      { q: 'Why does gradient descent zigzag in a ravine?', options: ['the steepest direction points mostly across the ravine, not along it', 'the learning rate is too small', 'the gradient is random', 'the ravine has no bottom'], answer: 0, why: 'Each step overshoots the steep walls while making little progress along the flat floor.' },
      { q: 'Gradient descent starts exactly on a saddle point. What happens?', options: ['it stays put: the gradient there is zero', 'it rolls to the minimum', 'it diverges', 'it moves sideways'], answer: 0, why: 'A zero gradient gives a zero step; a tiny push off the saddle is what lets it escape.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
