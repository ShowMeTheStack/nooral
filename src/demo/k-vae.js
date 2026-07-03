// A tiny variational autoencoder trained on strokes of every angle and length (12 × 12 pictures). Left: where the
// encoder puts each training picture in its two-number latent space, coloured by the stroke's angle. Right: the
// decoder's picture for every point of a grid in that space. As training goes on, the map becomes smooth: nearby
// points decode to similar strokes.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, box = D.box;
  const BUCKETS = ['0–45°', '45–90°', '90–135°', '135–180°'];
  D.add({
    id: 'vae-latent', order: 110, tag: 'Generative · VAE', short: 'VAE latent map', title: 'A VAE latent map: a grid of latent points decoded into pictures',
    blurb: 'The encoder maps a 12 × 12 picture to a mean and a spread in two dimensions; the decoder maps any latent point back to a picture. The loss asks for good reconstructions and for codes that stay close to a standard normal.',
    frames({ narrow }) {
      const T = S.Toy, v = D.once('vae', () => T.vae({ keep: [0, 60, 250, 1000] })), size = v.size;
      const G = narrow ? 5 : 7;
      // the grid covers where the trained encoder puts the training pictures
      const fin = v.data.slice(0, 160).map((d) => v.encode(v.snaps[v.snaps.length - 1], d.img));
      const lo = [0, 1].map((k) => Math.min(...fin.map((z) => z[k]))), hi = [0, 1].map((k) => Math.max(...fin.map((z) => z[k])));
      const cz = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], span = Math.round(Math.max(hi[0] - lo[0], hi[1] - lo[1]) * 5) / 10 + 0.05;
      const L = narrow
        ? { view: box(-4, -14, 396, 820), sc: { x: 40, y: 24, w: 320, h: 300 }, grid: { x: 48, y: 470, cell: 58 }, lg: [6, 400] }
        : { view: box(-10, -14, 1180, 520), sc: { x: 50, y: 24, w: 440, h: 420 }, grid: { x: 690, y: 40, cell: 62 }, lg: [520, 120] };
      const ax = C().axes(Object.assign({ prefix: 'va', xr: [-3, 3], yr: [-3, 3], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], xname: 'z₁', yname: 'z₂' }, L.sc));
      const pts = v.data.slice(0, 160);
      return v.snaps.map((sn, k) => {
        const items = [...ax.items];
        pts.forEach((d, i) => {
          const z = v.encode(sn, d.img), b = Math.min(3, Math.floor(d.angle / (Math.PI / 4)));
          const zc = [Math.max(-2.95, Math.min(2.95, z[0])), Math.max(-2.95, Math.min(2.95, z[1]))];
          items.push({ key: 'va:p' + i, kind: 'dot', x: ax.X(zc[0]), y: ax.Y(zc[1]), r: narrow ? 2.8 : 3.4, cls: 'sp c' + b, audit: 'overlap-ok', layer: 'items' });
        });
        // the grid of decoded pictures, and where it lies in latent space
        const gw = G * L.grid.cell;
        items.push({ key: 'va:gbox', kind: 'path', d: `M${ax.X(cz[0] - span)},${ax.Y(cz[1] - span)} H${ax.X(cz[0] + span)} V${ax.Y(cz[1] + span)} H${ax.X(cz[0] - span)} Z`, cls: 'guide act', audit: 'ignore', layer: 'lines' });
        for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
          const z = [cz[0] - span + (2 * span * c) / (G - 1), cz[1] + span - (2 * span * r) / (G - 1)];
          const img = v.decode(sn, z), x = L.grid.x + c * L.grid.cell, y = L.grid.y + r * L.grid.cell;
          items.push({ key: 'vg' + r + '_' + c, kind: 'raster', x: x + 2, y: y + 2, w: L.grid.cell - 4, h: L.grid.cell - 4, nx: size, ny: size, vals: img, scheme: 'ink', max: 1, smooth: false, cls: 'feature', layer: 'items' });
        }
        items.push({ key: 'vg:t', kind: 'text', x: L.grid.x, y: L.grid.y - 12, text: `decoded at a ${G} × ${G} grid over the dashed square`, size: 13, cls: 'strong' });
        items.push({ key: 'vg:z1', kind: 'text', x: L.grid.x + gw, y: L.grid.y + gw + 20, text: 'z₁ →', anchor: 'end', size: 13, cls: 'axname' });
        items.push({ key: 'vg:z2', kind: 'text', x: L.grid.x - 8, y: L.grid.y + 12, text: 'z₂ ↑', anchor: 'end', size: 13, cls: 'axname' });
        items.push({ key: 'va:lgt', kind: 'text', x: L.lg[0], y: L.lg[1] - 22, text: 'stroke angle', size: 13, cls: 'strong' });
        items.push(...C().legend('va:lg', L.lg[0], L.lg[1], BUCKETS.map((b, i) => ({ text: b, cls: 'c' + i, dot: true })), { dir: narrow ? 'row' : 'col', size: 13 }));
        items.push({ key: 'va:read', kind: 'text', x: L.lg[0], y: narrow ? L.lg[1] + 26 : L.lg[1] + 4 * 22 + 20, text: `step ${sn.step} · loss ${sn.loss.toFixed(1)}`, size: 13, cls: 'strong' });
        const say = k === 0
          ? `Untrained, the encoder sends every picture to nearly the same place and the decoder draws the same grey smudge everywhere. Loss per picture: \`${sn.loss.toFixed(1)}\` (mostly the reconstruction’s cross-entropy over 144 pixels).`
          : k < v.snaps.length - 1 ? `Step ${sn.step} (batches of 16, Adam): loss \`${sn.loss.toFixed(1)}\`. Strokes of different angles start to separate in latent space, and the decoded grid starts to show strokes.`
            : `Step ${sn.step}: loss \`${sn.loss.toFixed(1)}\`. The angles lie round the latent space in order, like the hours on a clock, and walking along the ring of codes turns the decoded stroke smoothly. Points between training codes decode to strokes it never saw: that is what makes it generative. Where no code lies, the decoder can only guess, and draws a blur.`;
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
