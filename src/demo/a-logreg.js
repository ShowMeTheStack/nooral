// Logistic regression learning the two moons: the boundary bends as the weights of the cubic features grow, the
// shading is the model's probability of orange at every place, and the loss falls beside it.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, SC = () => S.Kit.scatter, box = D.box;
  D.add({
    id: 'logreg-boundary', order: 10, tag: 'Logistic regression', short: 'Boundary forming', title: 'A decision boundary forming while logistic regression trains',
    blurb: 'Each point is described by the nine products of x and y up to the third power, so the boundary may bend. Full-batch gradient descent moves all nine weights a little each epoch; the shading is the model’s probability of orange.',
    frames({ narrow }) {
      const data = D.moons(), T = S.Toy;
      const fit = D.once('logreg3', () => T.logreg(data.X, data.y, { degree: 3, lr: 1, epochs: 300 }));
      const curve = D.once('logreg3c', () => T.logregCurve(data.X, data.y, { degree: 3, lr: 1, epochs: 300 }));
      const L = narrow
        ? { view: box(-6, -14, 384, 760), sc: { x: 34, y: 20, w: 330, h: 230 }, lc: { x: 46, y: 340, w: 300, h: 140 }, wb: { x: 20, y: 590, w: 350, h: 80 }, ts: 12 }
        : { view: box(-10, -16, 1180, 560), sc: { x: 50, y: 24, w: 580, h: 420 }, lc: { x: 730, y: 24, w: 400, h: 200 }, wb: { x: 700, y: 330, w: 440, h: 110 }, ts: 12 };
      const xr = [-1.6, 2.6], yr = [-1.15, 1.6];
      const ax = C().axes(Object.assign({ prefix: 'lr', xr, yr, xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1], xname: 'x', yname: 'y', tickSize: L.ts }, L.sc));
      const lax = C().axes(Object.assign({ prefix: 'lc', xr: [0, 300], yr: [0, 0.75], xticks: [0, 100, 200, 300], yticks: [0, 0.25, 0.5, 0.75], xname: 'epoch', yname: 'mean log-loss', tickSize: L.ts }, L.lc));
      return fit.snaps.map((sn, k) => {
        const p = (u, v) => fit.predict(sn.w, sn.b, [u, v]);
        const items = [...ax.items];
        items.push(SC().field(ax, 'lr:field', p, { n: narrow ? 60 : 90 }));
        if (sn.epoch > 0) items.push(SC().level(ax, 'lr:b', SC().grid(ax, p, narrow ? 50 : 80), 0.5, ''));
        const wrong = new Set();
        data.X.forEach((q, i) => { if ((p(q[0], q[1]) >= 0.5 ? 1 : 0) !== data.y[i]) wrong.add(i); });
        items.push(...SC().points(ax, 'lr:p', data.X, data.y, { r: narrow ? 3.4 : 4.4, cls: (i) => (sn.epoch > 0 && wrong.has(i) ? 'wrong' : '') }));
        items.push(...lax.items, C().line(lax, 'lc:loss', curve.slice(0, sn.epoch + 1).map((c) => [c[0], c[1]]), 'act'), C().dot(lax, 'lc:at', sn.epoch, sn.loss, ''));
        items.push({ key: 'lc:read', kind: 'text', x: L.lc.x + L.lc.w, y: narrow ? L.lc.y + L.lc.h + 58 : L.lc.y - 10, text: `epoch ${sn.epoch} · loss ${sn.loss.toFixed(3)} · ${D.pct(sn.acc)} right`, anchor: 'end', size: 13, cls: 'strong' });
        // the nine weights, as bars
        items.push({ key: 'wb:t', kind: 'text', x: L.wb.x, y: L.wb.y - 14, text: 'the weights', cls: 'title' });
        items.push(...D.bars('wb', L.wb.x, L.wb.y, L.wb.w, L.wb.h, sn.w, fit.names, { max: 4.5, size: 12 }));
        const big = sn.w.map((w, i) => [Math.abs(w), i]).sort((a, b) => b[0] - a[0])[0];
        const say = k === 0
          ? `Before training every weight is \`0\`, so the model says \`p = 0.5\` everywhere: no boundary yet, and the loss is \`ln 2 ≈ ${sn.loss.toFixed(3)}\`.`
          : k === 1 ? `One step of gradient descent (learning rate \`1\`) already tilts the weights toward the data: the loss drops to \`${sn.loss.toFixed(3)}\` and \`${D.pct(sn.acc)}\` of the 120 points fall on their own side.`
            : `Epoch ${sn.epoch}: loss \`${sn.loss.toFixed(3)}\`, accuracy \`${D.pct(sn.acc)}\`, ${wrong.size} point${wrong.size === 1 ? '' : 's'} (ringed) still on the wrong side. The largest weight is on \`${fit.names[big[1]]}\` (\`${D.fix(sn.w[big[1]])}\`)` + (k >= 6 ? ': the cubic terms bend the line into an S between the moons.' : '.');
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
