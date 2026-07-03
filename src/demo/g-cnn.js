// A small convolutional network looking at a generated digit: four 3×3 filters slide over the 16×16 picture
// (the window and the sum shown at two places), ReLU keeps the positive part, 2×2 max pooling halves each map, and
// a second-layer filter adds a left edge to a top edge to find a corner.
(function (S) {
  'use strict';
  const D = S.Demo, HM = () => S.Kit.heat, F = () => S.Kit.flow, box = D.box;
  D.add({
    id: 'cnn-maps', order: 70, tag: 'Convolutional network', short: 'CNN feature maps', title: 'A CNN’s feature maps: a digit flowing through convolution and pooling',
    blurb: 'The digit 3 is drawn by our stroke model (seed 5). The filters are written out by hand so each has one job: a left edge, a right edge, a top edge, a diagonal. Every map is computed from the picture, number by number.',
    frames({ narrow }) {
      const T = S.Toy, img = D.once('digit3', () => T.digit(3, 16, 5)), net = D.once('cnn3', () => T.cnn(img));
      const L = narrow
        ? { view: box(-4, -12, 396, 560), inp: [6, 40, 112], spine: 128, fil: [138, 15], map: [204, 56], pool: [282, 42], c2: [336, 50], top: 214, row: 80, fs: 12.5, lab: false }
        : { view: box(-10, -14, 1180, 530), inp: [20, 70, 224], spine: 268, fil: [290, 24], map: [420, 98], pool: [600, 70], c2: [800, 150], top: 40, row: 118, fs: 12, lab: true };
      const [ix, iy, isz] = L.inp, cell = isz / 16, cmax = Math.max(...net.c1.flat().map(Math.abs)) * 0.9;
      const rowY = (k) => L.top + k * L.row;
      const fw = 3 * L.fil[1] + 4, msz = L.map[1], psz = L.pool[1], c2sz = L.c2[1];
      const midY = rowY(0) + (rowY(3) + msz - rowY(0)) / 2;
      const c2y = midY - c2sz / 2;
      // the wires: input to each filter, filter to map, map to pool, pools to the second layer
      const wires = () => {
        const N = F().net('cw'), out = [];
        const ys = [0, 1, 2, 3].map((k) => rowY(k) + msz / 2);
        N.wire('in', [[ix + isz, iy + isz / 2], [L.spine, iy + isz / 2]]);
        const iy2 = iy + isz / 2;
        N.wire('sp', [[L.spine, Math.min(ys[0], iy2)], [L.spine, Math.max(ys[3], iy2)]]);
        if (iy2 > ys[0] && iy2 < ys[3]) N.dot(L.spine, iy2);
        ys.forEach((y, k) => {
          N.wire('f' + k, [[L.spine, y], [L.fil[0] - 2, y]], { head: true });
          if (k === 1 || k === 2 || (k === 0 && iy2 < ys[0])) N.dot(L.spine, y);
          N.wire('m' + k, [[L.fil[0] + fw + 2, y], [L.map[0] - 2, y]], { head: true });
          N.wire('p' + k, [[L.map[0] + msz + 2, y], [L.pool[0] - 2, y]], { head: true });
          N.wire('q' + k, [[L.pool[0] + psz + 2, y], [L.c2[0] - 22, y]]);
        });
        N.wire('qs', [[L.c2[0] - 22, ys[0]], [L.c2[0] - 22, ys[3]]]);
        ys.forEach((y, k) => { if (k === 1 || k === 2) N.dot(L.c2[0] - 22, y); });
        N.wire('c2', [[L.c2[0] - 22, c2y + c2sz / 2], [L.c2[0] - 2, c2y + c2sz / 2]], { head: true });
        if (Math.abs(c2y + c2sz / 2 - ys[1]) > 0.5 && Math.abs(c2y + c2sz / 2 - ys[2]) > 0.5) N.dot(L.c2[0] - 22, c2y + c2sz / 2);
        out.push(...N.items);
        return out;
      };
      const raster = (key, x, y, size, vals, n, scheme, max, cls) => ({ key, kind: 'raster', x, y, w: size, h: size, nx: n, ny: n, vals, scheme, max, cls: 'feature ' + (cls || ''), layer: 'items' });
      const outline = (key, x, y, w, h, cls) => ({ key, kind: 'path', d: `M${x},${y} h${w} v${h} h${-w} Z`, cls: 'win ' + (cls || ''), audit: 'ignore', layer: 'top' });
      const label = (key, x, y, text, anchor, cls) => ({ key, kind: 'text', x, y, text, anchor: anchor || 'start', size: L.fs, cls: cls || 'axname' });
      const base = (stage, hl) => {
        const items = [...wires()];
        items.push(raster('cn:in', ix, iy, isz, img, 16, 'ink', 1, ''), label('cn:inl', ix, iy - 10, 'input, 16 × 16'));
        T.filters1.forEach((F1, k) => {
          const g = HM().grid({ prefix: 'cf' + k, x: L.fil[0], y: rowY(k) + msz / 2 - fw / 2 + 2, cw: L.fil[1], ch: L.fil[1], gap: 2, vals: [F1.f.slice(0, 3), F1.f.slice(3, 6), F1.f.slice(6)], scheme: 'div', max: 1.4, fmt: L.lab ? (v) => (v ? (v > 0 ? '1' : '−1') : '0') : null, size: 11, cls: () => (hl && hl.f === k ? 'act' : '') });
          items.push(...g.items);
          if (L.lab) items.push(label('cf' + k + ':n', L.fil[0], rowY(k) + msz / 2 - fw / 2 - 6, F1.name, 'start', hl && hl.f === k ? 'act' : 'axname'));
          const maps = stage === 'input' ? null : stage === 'conv' ? net.c1[k] : net.r1[k];
          if (maps) items.push(raster('cm' + k, L.map[0], rowY(k), msz, maps, net.m1, stage === 'conv' ? 'div' : 'seq', cmax, ''));
          else items.push(outline('cm' + k + ':e', L.map[0], rowY(k), msz, msz, 'empty'));
          if (stage === 'pool' || stage === 'layer2') items.push(raster('cp' + k, L.pool[0], rowY(k), psz, net.p1[k], net.m2, 'seq', cmax, ''));
          else items.push(outline('cp' + k + ':e', L.pool[0], rowY(k), psz, psz, 'empty'));
        });
        items.push(label('cm:l', L.map[0], L.top - 10, stage === 'relu' || stage === 'pool' || stage === 'layer2' ? (L.lab ? 'after ReLU, 14 × 14' : 'ReLU') : (L.lab ? 'convolved, 14 × 14' : 'conv')));
        items.push(label('cp:l', L.pool[0], L.top - 10, L.lab ? 'pooled, 7 × 7' : 'pool'));
        if (stage === 'layer2') items.push(raster('c2', L.c2[0], c2y, c2sz, net.c2, net.m3, 'seq', Math.max(...net.c2) * 0.95, 'act'));
        else items.push(outline('c2:e', L.c2[0], c2y, c2sz, c2sz, 'empty'));
        items.push(label('c2:l', L.c2[0], c2y - 10, L.lab ? 'layer 2: a corner, 5 × 5' : 'layer 2'));
        return items;
      };
      const frames = [];
      frames.push({ view: L.view, items: base('input'), say: 'The input is a 16 × 16 picture of a 3, each pixel between `0` (paper) and `1` (ink). The first layer has four 3 × 3 filters; blue weights add the pixel under them, vermilion ones subtract it.' });
      // the window at two places on the left-edge filter
      const m0 = net.c1[0], hi = m0.indexOf(Math.max(...m0)), lo = m0.indexOf(Math.min(...m0));
      for (const [i, j] of [[hi % 14, Math.floor(hi / 14)], [lo % 14, Math.floor(lo / 14)]]) {
        const items = base('conv');
        const k = 0, f = T.filters1[k].f;
        let sum = -0.4;
        const terms = [];
        for (let b = 0; b < 3; b++) for (let a = 0; a < 3; a++) { const v = img[(j + b) * 16 + i + a]; sum += v * f[b * 3 + a]; if (f[b * 3 + a] && v > 0.05) terms.push((f[b * 3 + a] > 0 ? '+' : '−') + v.toFixed(2)); }
        items.push(outline('cn:win', ix + i * cell, iy + j * cell, 3 * cell, 3 * cell, 'act'));
        const mc = msz / 14;
        items.push(outline('cm:px', L.map[0] + i * mc, rowY(0) + j * mc, mc, mc, 'act'));
        frames.push({ view: L.view, items, say: `The left-edge filter over the 3 × 3 window at column ${i + 1}, row ${j + 1}: the sum of pixel × weight is ${terms.length ? '`' + terms.join(' ') + '`' : '`0` (no ink)'}, plus the bias \`−0.4\`, giving \`${D.fix(net.c1[0][j * 14 + i])}\` at that place of the first map. ` + (net.c1[0][j * 14 + i] > 0 ? 'Ink on the right of the window and paper on the left: a left edge, the strongest on this map.' : 'Ink on the left and paper on the right is the opposite of a left edge: the most negative value on this map. Sliding the window to all 14 × 14 places gives the whole map.') });
      }
      frames.push({ view: L.view, items: base('conv'), say: `All four maps. Each is bright (blue) where its filter matches: the left-edge map lights the left side of every stroke, the top-edge map the tops of the three bars. The largest value on any map is \`${D.fix(Math.max(...net.c1.flat()))}\`.` });
      const neg = net.c1.flat().filter((v) => v < 0).length;
      frames.push({ view: L.view, items: base('relu'), say: `ReLU sets every negative value to \`0\`: \`${neg}\` of the \`${4 * 196}\` values were negative. What is left says only “this pattern is here, this strongly”.` });
      {
        const items = base('pool'), k = 2, p = net.p1[k], at = p.indexOf(Math.max(...p)), pi = at % 7, pj = Math.floor(at / 7), mc = msz / 14, pc = psz / 7;
        items.push(outline('cm:pw', L.map[0] + 2 * pi * mc, rowY(k) + 2 * pj * mc, 2 * mc, 2 * mc, 'act'), outline('cp:px', L.pool[0] + pi * pc, rowY(k) + pj * pc, pc, pc, 'act'));
        frames.push({ view: L.view, items, say: `Max pooling keeps the largest of each 2 × 2 block, so each map shrinks to 7 × 7 and a pattern may shift a pixel without changing much. The boxed block of the top-edge map holds \`${D.fix(p[at])}\`, its largest value.` });
      }
      const c2max = Math.max(...net.c2), c2at = net.c2.indexOf(c2max);
      frames.push({ view: L.view, items: base('layer2'), say: `The second layer’s filter reads all four pooled maps at once: it adds the left-edge and top-edge maps over a 2 × 2 corner and subtracts a little of the diagonal. Its map is strongest (\`${D.fix(c2max)}\`) at column ${c2at % net.m3 + 1}, row ${Math.floor(c2at / net.m3) + 1}` + (c2at % net.m3 < 2 && c2at < 2 * net.m3 ? ', at the top left, where the top bar of the 3 begins.' : '.') });
      return frames;
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
