// Plain gradient descent, momentum and Adam racing down a curved valley from the same start: their paths over the
// contours of the loss, and the loss of each against the step.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, CT = () => S.Kit.contour, box = D.box;
  const RUN = [['gd', 'gradient descent', 'o-gd'], ['mom', 'momentum', 'o-mom'], ['adam', 'Adam', 'o-adam']];
  const AT = [0, 4, 10, 20, 35, 55, 80, 110, 150];
  D.add({
    id: 'optim-race', order: 50, tag: 'Optimisation', short: 'Optimiser race', title: 'Gradient descent, momentum and Adam racing on a loss surface',
    blurb: 'The loss L(u, v) = ½(1 − u)² + 4(v − u²)² has steep walls and a long, bending floor, with its lowest point at (1, 1). All three start on the wall at (−1.5, 0.5) and take 150 steps.',
    frames({ narrow }) {
      const T = S.Toy, surf = T.valley, start = [-1.5, 0.5];
      const race = D.once('race', () => T.race(surf, start, { steps: 150, lrGd: 0.02, lrMom: 0.008, beta: 0.9, lrAdam: 0.3 }));
      const L = narrow
        ? { view: box(-6, -16, 392, 740), sc: { x: 34, y: 20, w: 330, h: 300 }, lc: { x: 50, y: 530, w: 300, h: 140 }, lg: [34, 410] }
        : { view: box(-10, -16, 1180, 540), sc: { x: 50, y: 24, w: 560, h: 450 }, lc: { x: 730, y: 60, w: 400, h: 260 }, lg: [730, 400] };
      const ax = C().axes(Object.assign({ prefix: 'op', xr: [-2.2, 2.2], yr: [-1.2, 3.4], xticks: [-2, -1, 0, 1, 2], yticks: [-1, 0, 1, 2, 3], xname: 'u', yname: 'v' }, L.sc));
      const surface = D.once('race-surf:' + narrow, () => CT().surface(ax, 'op:s', surf.f, { levels: 8, n: narrow ? 70 : 110, depth: 2.4 }));
      const lmax = 40, lax = C().axes(Object.assign({ prefix: 'oc', xr: [0, 150], yr: [-4, Math.log10(lmax)], xticks: [0, 50, 100, 150], yticks: [-4, -3, -2, -1, 0, 1], fmtY: (v) => (v === 0 ? '1' : v === 1 ? '10' : '10' + ['⁻¹', '⁻²', '⁻³', '⁻⁴'][-v - 1]), xname: 'step', yname: 'loss (log scale)' }, L.lc));
      const lg = (v) => Math.log10(Math.max(1e-4, v));
      const fmt = (v) => (v < 1e-3 ? D.sci(v) : v.toFixed(3));
      return AT.map((n) => {
        const items = [...ax.items, ...surface];
        items.push({ key: 'op:goal', kind: 'dot', x: ax.X(1), y: ax.Y(1), r: 6, cls: 'goal', point: true });
        items.push({ key: 'op:start', kind: 'text', x: ax.X(start[0]) - 8, y: ax.Y(start[1]) + 20, text: 'start', anchor: 'end', size: 13, cls: 'strong' });
        for (const [k, , cls] of RUN) items.push(...CT().path(ax, 'op:' + k, race[k], cls, { upto: n, dots: 0 }));
        items.push(...lax.items);
        for (const [k, , cls] of RUN) items.push(C().line(lax, 'oc:' + k, race[k].slice(0, n + 1).map((p, i) => [i, lg(surf.f(p[0], p[1]))]), cls));
        items.push(...C().legend('op:lg', L.lg[0], L.lg[1], RUN.map(([k, name, cls]) => ({ text: name + ': ' + fmt(surf.f(...race[k][n])), cls })), { dir: narrow ? 'col' : 'col', size: 13 }));
        items.push({ key: 'op:lgg', kind: 'dot', x: L.lg[0] + 7, y: L.lg[1] + 3 * 22 - 4.5, r: 5.5, cls: 'goal', point: true }, { key: 'op:lggt', kind: 'text', x: L.lg[0] + 24, y: L.lg[1] + 3 * 22, text: 'the lowest point, (1, 1)', size: 13, cls: 'legend' });
        items.push({ key: 'op:lgt', kind: 'text', x: L.lg[0], y: L.lg[1] - 24, text: `after ${n} step${n === 1 ? '' : 's'}, the loss of each`, size: 13, cls: 'axname' });
        const f = (k) => surf.f(...race[k][n]), fm = fmt;
        const lead = RUN.map(([k, name]) => [f(k), name]).sort((a, b) => a[0] - b[0])[0][1];
        const now = `gradient descent \`${fm(f('gd'))}\`, momentum \`${fm(f('mom'))}\`, Adam \`${fm(f('adam'))}\``;
        const why = ['', 'Gradient descent steps by its learning rate (\`0.02\`) times the gradient: big steps on the steep wall, tiny ones on the flat floor.',
          'Momentum (rate \`0.008\`, β \`0.9\`) adds each gradient to a velocity that keeps nine tenths of itself, so it gathers speed along the floor and can swing past the bend.',
          'Adam (rate \`0.3\`) divides each coordinate’s step by the recent size of that coordinate’s gradient, so a small gradient still gives a full-size step.',
          'On the floor the gradient is small, which is exactly where plain gradient descent slows down.', '', '', '', ''][AT.indexOf(n)] || '';
        const say = n === 0 ? `All three start at \`(−1.5, 0.5)\`, up the wall, where the loss is \`${surf.f(...start).toFixed(2)}\`. The gradient there points mostly across the valley, not along it toward \`(1, 1)\`.`
          : `Step ${n}: ${now}; ${lead} is lowest. ` + why;
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
