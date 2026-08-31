// Part 10, multiple channels: a colour picture is three grids (red, green, blue), and a filter for it is three
// kernels, one per channel. At each place the three window sums are added, with one bias, into a single number: a
// filter always makes one map, however many channels it reads.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const NAMES = ['red', 'green', 'blue'];
  const fill = (f) => Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => f(i, j)));
  const PICS = {
    flag: { label: 'red on the left, blue on the right', ch: [fill((i, j) => (j < 2 ? 1 : 0)), fill(() => 0), fill((i, j) => (j >= 2 ? 1 : 0))] },
    dot: { label: 'a yellow dot on blue', ch: [fill((i, j) => (Math.abs(i - 2) + Math.abs(j - 2) <= 1 ? 1 : 0)), fill((i, j) => (Math.abs(i - 2) + Math.abs(j - 2) <= 1 ? 1 : 0)), fill((i, j) => (Math.abs(i - 2) + Math.abs(j - 2) <= 1 ? 0 : 1))] },
    stripes: { label: 'green and red stripes', ch: [fill((i) => (i % 2 ? 1 : 0)), fill((i) => (i % 2 ? 0 : 1)), fill(() => 0)] },
  };
  const k3 = (v) => [[v, v, v], [v, v, v], [v, v, v]], edge = [[1, 0, -1], [1, 0, -1], [1, 0, -1]], neg = (K) => K.map((r) => r.map((v) => -v)), z = k3(0);
  const FILTERS = {
    boundary: { label: 'red-to-blue boundary', w: [edge, z, neg(edge)], b: 0 },
    yellow: { label: 'yellow detector (red + green − blue)', w: [k3(1), k3(1), k3(-1)], b: -2 },
    redonly: { label: 'red edges only', w: [edge, z, z], b: 0 },
    stripe: { label: 'red rows minus green rows', w: [[[0, 0, 0], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]], z], b: 0.5 },
  };
  P.CHANNELS = { PICS, FILTERS };
  const C = 26, KC = 34, PC = 36, ROW = 172, TOP = 40, KX = 186, PX = 316, OX = 560;

  S.topic({
    id: 'conv-channels', family: 'p10-conv', order: 40, title: 'Many channels: one filter, three kernels',
    blurb: 'A colour picture has three channels, so a filter that reads it has a 3 × 3 kernel for each, 27 weights in all, and one bias. At every place each kernel makes its own window sum, and the three sums are added into one number. The same holds deeper in a network, where the channels are the maps of the layer before.',
    inputs: [
      { id: 'picture', kind: 'choice', label: 'Picture (5 × 5, three channels)', def: 'flag', options: P.options(PICS) },
      { id: 'filter', kind: 'choice', label: 'Filter', def: 'boundary', options: P.options(FILTERS) },
    ],
    presets: [
      { label: 'A colour boundary', values: { picture: 'flag', filter: 'boundary' } },
      { label: 'Finding yellow', values: { picture: 'dot', filter: 'yellow' } },
      { label: 'A filter blind to blue', values: { picture: 'flag', filter: 'redonly' } },
      { label: 'Stripes, with a bias', values: { picture: 'stripes', filter: 'stripe' } },
    ],
    counters: [{ id: 'places', label: 'places' }, { id: 'mults', label: 'multiplications' }],
    run(inp) {
      const T = S.T, pic = PICS[inp.picture].ch, fl = FILTERS[inp.filter], n = 5, k = 3, m = n - k + 1, nc = 3;
      const x = T.tensor(pic.flat(2), [1, nc, n, n]), w = T.tensor(fl.w.flat(2), [1, nc, k, k]), b = T.tensor([fl.b]);
      const out = Array.from(T.conv2d(x, w, b).values());
      const part = [0, 1, 2].map((c) => Array.from(T.conv2d(T.tensor(pic[c].flat(), [1, 1, n, n]), T.tensor(fl.w[c].flat(), [1, 1, k, k])).values()));
      const F = H().frames(['places', 'mults']), q = F.q;
      const wmax = P.maxAbs(fl.w.flat(2)), pmax = P.maxAbs(part.flat()), omax = P.maxAbs(out);
      const rowY = (c) => TOP + c * ROW, midY = TOP + ROW + (5 * (C + 2)) / 2 - (m * (40 + 2)) / 2;
      const opX = 512, opY = rowY(1) + (m * (PC + 2)) / 2 + 10;
      const wires = () => {
        const N = S.Kit.flow.net('cw'), ys = [0, 1, 2].map((c) => rowY(c) + 10 + (m * (PC + 2)) / 2);
        ys.forEach((y, c) => { if (c !== 1) N.wire('p' + c, [[PX + m * (PC + 2) + 6, y], [opX, y], [opX, opY + (c ? 13 : -13)]], { head: true }); });
        N.wire('p1', [[PX + m * (PC + 2) + 6, ys[1]], [opX - 13, opY]], { head: true });
        N.op('sum', opX, opY, 'add', '', 12);
        N.wire('o', [[opX + 13, opY], [OX - 6, opY]], { head: true });
        return N.items;
      };
      const base = (lit, done, cur) => {
        const items = [...wires()];
        for (let c = 0; c < nc; c++) {
          items.push(P.text('t:c' + c, 0, rowY(c) - 9, NAMES[c] + ' channel', 'axname', 13));
          items.push(...P.cells('in' + c, 0, rowY(c), pic[c], { cw: C, scheme: c === 0 ? 'c1' : c === 1 ? 'good' : 'c0', max: 1.3, d: 0, size: 12, cls: (i, j) => (lit && lit.c.includes(c) && i >= lit.i && i < lit.i + k && j >= lit.j && j < lit.j + k ? 'lit' : '') }).items);
          items.push(...P.cells('k' + c, KX, rowY(c) + 10, fl.w[c], { cw: KC, max: wmax, d: 2, size: 12 }).items);
          items.push(...P.cells('p' + c, PX, rowY(c) + 10, P.rows(part[c], m), { cw: PC, max: pmax, d: 2, size: 12, cls: (i, j) => (cur && cur[0] === i && cur[1] === j ? 'act' : '') }).items.map((it, idx) => (done(c, Math.floor(idx / m), idx % m) ? it : Object.assign({}, it, { text: '', cls: it.cls + ' wait', paint: null }))));
          if (lit && lit.c.includes(c)) items.push(P.box('win' + c, lit.j * (C + 2) - 3, rowY(c) + lit.i * (C + 2) - 3, k * (C + 2) + 4, k * (C + 2) + 4, 'act'));
        }
        items.push(P.text('t:k', KX, TOP - 26, 'kernels', 'axname', 13), P.text('t:p', PX, TOP - 26, 'window sums', 'axname', 13), P.text('t:o', OX, midY - 14, 'output map', 'axname', 13));
        items.push(...P.cells('out', OX, midY, P.rows(out, m), { cw: 40, max: omax, d: 2, size: 12, cls: (i, j) => (cur && cur[0] === i && cur[1] === j ? 'act' : '') }).items.map((it, idx) => (done(3, Math.floor(idx / m), idx % m) ? it : Object.assign({}, it, { text: '', cls: it.cls + ' wait', paint: null }))));
        if (fl.b) items.push(P.text('t:b', OX, midY + m * 42 + 22, 'bias ' + P.num(fl.b, 2), 'legend', 13));
        return items;
      };
      F.push(base(null, () => false),
        `The picture (${PICS[inp.picture].label}) is ${q.n(nc)} grids of ${q.n(n)} × ${q.n(n)}, one per colour. The filter has one ${q.n(k)} × ${q.n(k)} kernel per channel: ${q.n(nc * k * k)} weights and one bias. It makes one ${q.n(m)} × ${q.n(m)} map.`, { phase: 'start', nc, n, k, m, nw: nc * k * k });
      const seen = new Set();
      for (let c = 0; c < nc; c++) {
        F.count('mults', 9);
        seen.add(c + ',0,0');
        F.push(base({ i: 0, j: 0, c: [c] }, (cc, i, j) => seen.has(cc + ',' + i + ',' + j), [0, 0]),
          `At the first place, the ${NAMES[c]} window times the ${NAMES[c]} kernel adds up to ${q.sg(part[c][0], 3)}.` + (fl.w[c].flat().every((v) => v === 0) ? ' This kernel is all zeros, so the filter ignores this channel.' : ''), { phase: 'channel', c, val: part[c][0] });
      }
      F.count('places');
      seen.add('3,0,0');
      F.push(base({ i: 0, j: 0, c: [0, 1, 2] }, (cc, i, j) => seen.has(cc + ',' + i + ',' + j), [0, 0]),
        `The three sums are added: ${q.sg(part[0][0], 3, true)} + ${q.sg(part[1][0], 3)} + ${q.sg(part[2][0], 3)}` + (fl.b ? ` + bias ${q.sg(fl.b, 3)}` : '') + ` = ${q.sg(out[0], 3)}, one number of the output map.`, { phase: 'sum', parts: [part[0][0], part[1][0], part[2][0]], b: fl.b, val: out[0] });
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
        if (!i && !j) continue;
        F.count('places').count('mults', 27);
        for (let c = 0; c <= 3; c++) seen.add(c + ',' + i + ',' + j);
        const ix = i * m + j;
        F.push(base({ i, j, c: [0, 1, 2] }, (cc, a, bb) => seen.has(cc + ',' + a + ',' + bb), [i, j]),
          `Row ${q.n(i + 1)}, column ${q.n(j + 1)}: red ${q.sg(part[0][ix], 3)}, green ${q.sg(part[1][ix], 3)}, blue ${q.sg(part[2][ix], 3)}` + (fl.b ? `, bias ${q.sg(fl.b, 3)}` : '') + `; together ${q.sg(out[ix], 3)}.`, { phase: 'place', i, j, row: i + 1, col: j + 1, parts: [part[0][ix], part[1][ix], part[2][ix]], b: fl.b, val: out[ix] });
      }
      const hi = out.indexOf(Math.max(...out));
      F.push(base(null, () => true),
        `The map is strongest, ${q.sg(out[hi], 3)}, at row ${q.n(Math.floor(hi / m) + 1)}, column ${q.n((hi % m) + 1)}. A layer of ${q.n(16)} such filters would have ${q.n(16 * (nc * k * k + 1))} weights and biases and make ${q.n(16)} maps: the next layer’s filters then read ${q.n(16)} channels each.`, { phase: 'end', max: out[hi], row: Math.floor(hi / m) + 1, col: (hi % m) + 1, more: 16, params: 16 * (nc * k * k + 1) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A filter over C channels', text: ['y[i, j] = b + Σ_c Σᵤ Σᵥ x[c, i + u, j + v] · w[c, u, v]', 'weights per filter: C · k · k, plus one bias', 'a layer of O filters: O maps, O · (C · k · k + 1) numbers'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start', once: true, pick: 'option',
      ask: 'How many output maps does this one filter make from the three channels?',
      options: ['one', 'three', 'nine'], answer: 'one',
      explain: 'The three window sums are added into one number at every place, so one filter gives one map.',
    }],
    quiz: [
      { q: 'A layer reads 6 channels with 3 × 3 kernels. How many weights does each of its filters have, bias aside?', options: ['54', '9', '6', '18'], answer: 0, why: '6 · 3 · 3.' },
      { q: 'A layer with 12 filters, each 5 × 5 over 6 channels, has how many weights and biases?', options: ['1812', '300', '72', '1800'], answer: 0, why: '12 · (6 · 25 + 1) = 1812.' },
      { q: 'How many maps does a layer of 8 filters produce?', options: ['8', 'as many as its input channels', '8 times the input channels', 'one'], answer: 0, why: 'One map per filter.' },
      { q: 'A filter whose blue kernel is all zeros…', options: ['ignores the blue channel', 'only sees blue', 'cannot be trained', 'makes no map'], answer: 0, why: 'Blue pixels are multiplied by zero.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
