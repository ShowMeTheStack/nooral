// A support vector machine on 60 points of the moons, as the kernel, C and γ change: the boundary f = 0, the
// margins f = ±1, the support vectors ringed, and the multipliers α beside the plot, capped at C.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, SC = () => S.Kit.scatter, box = D.box;
  const SWEEP = [
    { kernel: 'linear', C: 1, gamma: 0, say: 'A linear kernel can only draw a straight boundary. Points on or inside the dashed margins, or on the wrong side, are the support vectors: they alone fix the line.' },
    { kernel: 'rbf', C: 1, gamma: 0.5, say: 'An RBF kernel with a small γ: every point’s influence reaches far, so the boundary is a gentle curve.' },
    { kernel: 'rbf', C: 1, gamma: 2, say: 'A larger γ makes each point’s influence local, and the boundary starts to follow the moons.' },
    { kernel: 'rbf', C: 100, gamma: 2, say: 'Raising C makes mistakes costly: the margin narrows until nearly every point is on its own side, and fewer points need to be support vectors.' },
    { kernel: 'rbf', C: 1, gamma: 12, say: 'A very large γ: each support vector carves a small bubble round itself. The training points fit, but the boundary has stopped describing the moons.' },
  ];
  D.add({
    id: 'svm-margin', order: 30, tag: 'Support vector machine', short: 'SVM margin', title: 'The SVM margin and support vectors as C and γ change',
    blurb: 'Trained by SMO, which improves the multipliers α two at a time. The shading is the sign and size of the decision value f(x); solid is f = 0, dashed f = ±1. Move the sliders to try your own C and γ.',
    knobs: [{ id: 'C', label: 'C', min: 0.1, max: 1000, value: 10, log: true }, { id: 'gamma', label: 'γ (RBF)', min: 0.1, max: 30, value: 4, log: true }],
    knobsGo: 'last',
    frames({ narrow, knobs }) {
      const data = D.moonsSmall(), T = S.Toy, xr = [-1.6, 2.6], yr = [-1.15, 1.6];
      const L = narrow
        ? { view: box(-6, -14, 390, 560), sc: { x: 30, y: 20, w: 340, h: 240 }, al: { x: 30, y: 380, w: 330, h: 90 } }
        : { view: box(-10, -14, 1180, 500), sc: { x: 50, y: 24, w: 600, h: 400 }, al: { x: 740, y: 130, w: 400, h: 150 } };
      const ax = C().axes(Object.assign({ prefix: 'sv', xr, yr, xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1], xname: 'x', yname: 'y' }, L.sc));
      const list = SWEEP.concat([{ kernel: 'rbf', C: +knobs.C.toPrecision(3), gamma: +knobs.gamma.toPrecision(3), mine: true, say: 'Move the sliders to try others.' }]);
      return list.map((cfg) => {
        const m = D.once(`svm:${cfg.kernel}:${cfg.C}:${cfg.gamma}`, () => T.svm(data.X, data.y, { kernel: cfg.kernel, C: cfg.C, gamma: cfg.gamma }));
        const items = [...ax.items];
        items.push(SC().field(ax, 'sv:f', (u, v) => T.sigmoid(1.6 * m.f([u, v])), { n: narrow ? 60 : 90 }));
        const g = SC().grid(ax, (u, v) => m.f([u, v]), narrow ? 60 : 90);
        items.push(SC().level(ax, 'sv:m0', g, -1, 'margin'), SC().level(ax, 'sv:m1', g, 1, 'margin'), SC().level(ax, 'sv:b', g, 0, ''));
        items.push(...SC().points(ax, 'sv:p', data.X, data.y, { r: narrow ? 3.4 : 4.4 }));
        for (const i of m.sv) items.push(SC().ring(ax, 'sv:r' + i, data.X[i][0], data.X[i][1], m.alpha[i] < m.C - 1e-6 ? 'act' : '', narrow ? 6.5 : 8.5));
        // the multipliers: one bar per support vector, sorted, against the cap C
        const svs = m.sv.slice().sort((a, b) => m.alpha[b] - m.alpha[a]), n = svs.length, A = L.al, bw = Math.min(14, (A.w - 4) / Math.max(1, n));
        const amax = Math.max(m.C, 1e-9);
        items.push({ key: 'al:t', kind: 'text', x: A.x, y: A.y - 38, text: 'the multipliers α, one bar per support vector', size: 13, cls: 'axname' });
        items.push({ key: 'al:base', kind: 'path', d: `M${A.x},${A.y + A.h} h${A.w}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        items.push({ key: 'al:cap', kind: 'path', d: `M${A.x},${A.y} h${A.w}`, cls: 'guide act', audit: 'ignore', layer: 'back' });
        items.push({ key: 'al:capn', kind: 'text', x: A.x + A.w, y: A.y - 8, text: 'C = ' + D.num(m.C), anchor: 'end', size: 12, cls: 'act' });
        svs.forEach((i, k) => { const h = Math.max(1.5, (m.alpha[i] / amax) * A.h); items.push({ key: 'al:b' + k, kind: 'cell', x: A.x + 2 + k * bw, y: A.y + A.h - h, w: Math.max(1, bw - 2), h, rx: 1, text: '', cls: 'bar ' + (m.alpha[i] < m.C - 1e-6 ? '' : 'soft'), audit: 'ignore' }); });
        const cap = m.sv.length - m.margin.length;
        if (narrow) items.push({ key: 'al:read', kind: 'text', x: A.x, y: A.y + A.h + 24, text: `${m.sv.length} support vectors, ${cap} at the cap`, size: 13, cls: 'strong' }, { key: 'al:acc', kind: 'text', x: A.x, y: A.y + A.h + 44, text: `${D.pct(m.acc)} of the points right`, size: 13, cls: 'strong' });
        else items.push({ key: 'al:read', kind: 'text', x: A.x, y: A.y + A.h + 24, text: `${m.sv.length} support vectors · ${cap} at the cap · ${D.pct(m.acc)} of the points right`, size: 13, cls: 'strong' });
        items.push({ key: 'al:cfg', kind: 'text', x: A.x, y: A.y - 60, text: cfg.kernel === 'linear' ? `linear kernel, C = ${D.num(cfg.C)}` : `RBF kernel, C = ${D.num(cfg.C)}, γ = ${D.num(cfg.gamma)}`, size: 16, cls: 'strong' });
        if (!narrow) {
          const ly = A.y + A.h + 62, fake = (y) => ({ X: () => A.x + 8, Y: () => y });
          items.push(SC().ring(fake(ly - 4.5), 'sv:lr1', 0, 0, 'act', 7), { key: 'sv:lt1', kind: 'text', x: A.x + 24, y: ly, text: 'on the margin: 0 < α < C (coloured bars)', size: 12, cls: 'legend' });
          items.push(SC().ring(fake(ly + 19.5), 'sv:lr2', 0, 0, '', 7), { key: 'sv:lt2', kind: 'text', x: A.x + 24, y: ly + 24, text: 'α = C: inside the margin or wrong (grey bars)', size: 12, cls: 'legend' });
        }
        const say = (cfg.mine ? `Your setting, C = \`${D.num(cfg.C)}\` and γ = \`${D.num(cfg.gamma)}\`: ` : '') + cfg.say + ` Here \`${m.sv.length}\` of the 60 points are support vectors, \`${cap}\` of them with α at the cap C, and the training accuracy is \`${D.pct(m.acc)}\`.`;
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
