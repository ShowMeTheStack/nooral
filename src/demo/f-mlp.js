// A small network learning the moons: two inputs, six tanh neurons, one output. Every weight is a line whose
// colour is its sign and whose thickness is its size; each hidden neuron's map shows where in the input plane it is
// positive (blue) or negative (vermilion), with its zero line; the output combines them into a curved boundary.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, SC = () => S.Kit.scatter, NK = () => S.Kit.net, box = D.box;
  const KEEP = [0, 10, 30, 80, 200, 500];
  D.add({
    id: 'mlp-live', order: 60, tag: 'Neural network', short: 'MLP live weights', title: 'An MLP with live weights, and what each hidden neuron sees',
    blurb: 'A 2–6–1 network with tanh hidden neurons, trained by Adam on the log-loss. The neurons are filled with their activations for the ringed point; each small map is one hidden neuron over the whole input plane.',
    frames({ narrow }) {
      const T = S.Toy, data = D.moons(), xr = [-1.6, 2.6], yr = [-1.15, 1.6];
      const fit = D.once('mlp', () => T.trainMlp(data.X, data.y, { sizes: [2, 6, 1], lr: 0.03, keep: KEEP, seed: 4 }));
      const probe = data.X[3];
      const L = narrow
        ? { view: box(-6, -14, 392, 904), net: { x: 40, y: 34, w: 300, h: 230 }, r: 15, maps: { x: 14, y: 344, cw: 112, ch: 76, gx: 10, gy: 30, cols: 3 }, sc: { x: 34, y: 584, w: 330, h: 230 } }
        : { view: box(-10, -18, 1180, 500), net: { x: 60, y: 30, w: 330, h: 400 }, r: 18, maps: { x: 470, y: 46, cw: 126, ch: 88, gx: 14, gy: 46, cols: 2 }, sc: { x: 800, y: 50, w: 340, h: 240 } };
      const lay = NK().layout(Object.assign({ sizes: [2, 6, 1], r: L.r }, L.net));
      const ax = C().axes(Object.assign({ prefix: 'mo', xr, yr, xticks: [-1, 0, 1, 2], yticks: [-1, 0, 1], xname: 'x', yname: 'output p(orange)' }, L.sc));
      return fit.snaps.map((sn) => {
        const items = [];
        const A = fit.run(sn, probe), p = T.sigmoid(A[2][0]);
        const wmax = Math.max(2, ...sn.W.flat(2).map(Math.abs));
        items.push(...NK().draw(lay, { prefix: 'nn', weights: sn.W, acts: [A[0], A[1], [Math.tanh(A[2][0] / 2)]], amax: 1, wmax, wide: 6.5, names: [['x', 'y'], ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], ['p']], nodeCls: (l) => (l === 0 ? 'input' : '') }));
        items.push({ key: 'nn:t0', kind: 'text', x: lay.pos[0][0].x, y: L.net.y - 4, text: 'input', anchor: 'middle', size: 12, cls: 'axname' });
        items.push({ key: 'nn:t1', kind: 'text', x: lay.pos[1][0].x, y: L.net.y - 4, text: 'hidden (tanh)', anchor: 'middle', size: 12, cls: 'axname' });
        items.push({ key: 'nn:t2', kind: 'text', x: lay.pos[2][0].x, y: L.net.y - 4, text: 'output', anchor: 'middle', size: 12, cls: 'axname' });
        items.push({ key: 'nn:pv', kind: 'text', x: lay.pos[2][0].x, y: lay.pos[2][0].y + L.r + 20, text: 'p = ' + p.toFixed(2), anchor: 'middle', size: 13, cls: 'strong' });
        // each hidden neuron over the input plane
        const M = L.maps;
        for (let j = 0; j < 6; j++) {
          const cx = M.x + (j % M.cols) * (M.cw + M.gx), cy = M.y + Math.floor(j / M.cols) * (M.ch + M.gy);
          const hax = C().axes({ prefix: 'hm' + j, x: cx, y: cy, w: M.cw, h: M.ch, xr, yr, xticks: 0, yticks: 0, grid: false, frame: 'box' });
          const f = (u, v) => Math.tanh(sn.W[0][j][0] * u + sn.W[0][j][1] * v + sn.b[0][j]);
          items.push(...hax.items, SC().field(hax, 'hm' + j + ':f', f, { n: 40, scheme: 'div', max: 1.4 }));
          items.push(SC().level(hax, 'hm' + j + ':z', SC().grid(hax, f, 24), 0, 'soft', { minPts: 1 }));
          const wo = sn.W[1][0][j];
          items.push({ key: 'hm' + j + ':n', kind: 'text', x: cx, y: cy - 7, text: 'h' + (j + 1), size: 13, cls: 'strong' });
          items.push({ key: 'hm' + j + ':w', kind: 'text', x: cx + M.cw, y: cy - 7, text: 'to p: ' + D.fix(wo, 1), anchor: 'end', size: 12, cls: wo >= 0 ? 'pos' : 'neg' });
        }
        const my = M.y + Math.ceil(6 / M.cols) * (M.ch + M.gy) - M.gy + 22;
        items.push({ key: 'hm:lg', kind: 'text', x: M.x, y: my, text: 'blue: the neuron is positive there; vermilion: negative', size: 12, cls: 'axname' });
        // the output over the plane
        const out = (u, v) => T.sigmoid(fit.run(sn, [u, v])[2][0]);
        items.push(...ax.items, SC().field(ax, 'mo:f', out, { n: narrow ? 60 : 80 }));
        if (sn.epoch > 0) items.push(SC().level(ax, 'mo:b', SC().grid(ax, out, 60), 0.5, ''));
        items.push(...SC().points(ax, 'mo:p', data.X, data.y, { r: narrow ? 3 : 3.6 }), SC().ring(ax, 'mo:probe', probe[0], probe[1], 'act', 8));
        items.push({ key: 'mo:read', kind: 'text', x: L.sc.x, y: L.sc.y + L.sc.h + 58, text: `epoch ${sn.epoch} · loss ${sn.loss.toFixed(3)} · ${D.pct(sn.acc)} right`, size: 14, cls: 'strong' });
        const strong = sn.W[1][0].map((w, j) => [Math.abs(w), j]).sort((a, b) => b[0] - a[0]).slice(0, 2).map((x) => 'h' + (x[1] + 1));
        const say = sn.epoch === 0
          ? `Before training the weights are small random numbers (seed \`4\`): each hidden neuron is a soft step across the plane along some line, and the output is a muddle (loss \`${sn.loss.toFixed(3)}\`).`
          : `Epoch ${sn.epoch}: loss \`${sn.loss.toFixed(3)}\`, \`${D.pct(sn.acc)}\` right. The thickest lines into the output come from ${strong.join(' and ')}: the output adds their steps, each weighted, and the sum of straight steps bends into the boundary.` + (sn.epoch >= 200 ? ` For the ringed point the output says \`p = ${p.toFixed(2)}\`.` : '');
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
