// Part 10, learned filters: the first layer's six 5 × 5 filters of the network trained here, drawn as small images
// (blue weights add the pixel, vermilion ones subtract it) at four moments of training, each with the map it makes
// on one image; then each trained filter in turn, and the second layer's 12 × 6 kernels.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const TOP = 40, FS = 96, GAP = 22, MY = TOP + FS + 64;
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

  S.topic({
    id: 'cnn-filters', family: 'p10-inside', order: 20, title: 'Learned filters, drawn as pictures',
    blurb: 'Nobody wrote the first layer’s filters: they start as random numbers and training shapes them. Drawn as tiny images they turn into stroke and edge detectors: a blue side and a vermilion side mean “ink here, paper there”. Under each filter is the map it makes from one image, so you can see what it picks out.',
    inputs: [P.imageInputs('digits', 4, 0)[0], P.imageInputs('digits', 4, 0)[1]],
    presets: [
      { label: 'Digits, shown on a 4', values: { set: 'digits', cls: 4 } },
      { label: 'Digits, shown on a 0', values: { set: 'digits', cls: 0 } },
      { label: 'Shapes, shown on a triangle', values: { set: 'shapes', cls: 2 } },
      { label: 'Shapes, shown on a circle', values: { set: 'shapes', cls: 0 } },
    ],
    counters: [{ id: 'steps', label: 'training steps' }],
    run(inp) {
      const T = S.T, run = inp.set, J = P.D()[run], im = P.pick(run, inp.cls, 0), x = T.tensor(Array.from(im.px), [1, 1, 16, 16]);
      const snaps = J.snaps.map((s) => ({ step: s.step, W: P.f32(s.conv1) })), W0 = snaps[0].W;
      const fin = P.param(run, 'conv1.W').v;
      const F = H().frames(['steps']), q = F.q;
      const respond = (W) => Array.from(T.conv2d(x, T.tensor(Array.from(W), [6, 1, 5, 5]), null, { padding: 2 }).values());
      const row = (W, maps, lit, step) => {
        const items = [P.text('h:f', 0, TOP - 14, step === null ? 'trained filters' : 'filters at step ' + step, 'axname', 13), P.text('h:m', 0, MY - 14, 'what each makes from the image (no bias)', 'axname', 13)];
        const wm = P.maxAbs(W), mm = P.maxAbs(maps);
        for (let f = 0; f < 6; f++) {
          const fx = f * (FS + GAP);
          items.push(P.img('f' + f, fx, TOP, FS, W.slice(f * 25, f * 25 + 25), { scheme: 'div', max: wm, cls: lit === f ? 'act' : '' }));
          items.push(P.img('m' + f, fx, MY, FS, maps.slice(f * 256, f * 256 + 256), { scheme: 'div', max: mm, cls: lit === f ? 'act' : '' }));
          items.push(P.text('n' + f, fx + FS / 2, TOP + FS + 18, 'filter ' + (f + 1), lit === f ? 'act' : 'legend', 12, 'middle'));
        }
        items.push(P.text('h:i', 0, MY + FS + 34, 'the image', 'axname', 13), P.img('img', 0, MY + FS + 44, 80, im.px));
        return items;
      };
      const blobs = [0, 1, 2, 3, 4, 5].filter((f) => { const w = fin.slice(f * 25, f * 25 + 25), p = w.filter((v) => v > 0).reduce((a, b) => a + b, 0), n = -w.filter((v) => v < 0).reduce((a, b) => a + b, 0); return p > 2 * n; }).length;
      snaps.forEach((s, i) => {
        F.set('steps', s.step);
        const change = norm(s.W.map((v, j) => v - W0[j])) / norm(W0), maps = respond(s.W);
        const say = i === 0
          ? `Before training (step ${q.n(s.step)}) the ${q.n(6)} filters are random numbers from the initialiser: speckled, with no shape. Their maps on ${P.qwords(q, run, im.y)} are speckled too, though each already echoes the strokes faintly.`
          : `Step ${q.n(s.step)}: the filters have moved ${q.pct(change)} of their starting size away from it.` + (i === snaps.length - 1 ? ` The speckle has given way to smooth patches: ${q.n(blobs)} of the ${q.n(6)} filters ${blobs === 1 ? 'is a mostly blue, stroke-shaped patch that adds' : 'are mostly blue, stroke-shaped patches that add'} up ink, and the other ${q.n(6 - blobs)} ${6 - blobs === 1 ? 'sets' : 'set'} blue against vermilion, the mark of an edge detector. Their maps light up along the strokes.` : ' Smooth patches of one sign are starting to appear.');
        F.push(row(s.W, maps, -1, s.step), say, { phase: 'snap', step: s.step, change, blobs, six: 6, rest: 6 - blobs });
      });
      // each trained filter in turn: how it is built and where it answers
      const maps = respond(fin);
      for (let f = 0; f < 6; f++) {
        const w = fin.slice(f * 25, f * 25 + 25), pos = w.filter((v) => v > 0).reduce((a, b) => a + b, 0), neg = w.filter((v) => v < 0).reduce((a, b) => a + b, 0);
        let ex = 0, ey = 0;
        for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) { ex += (w[i * 5 + j + 1] - w[i * 5 + j]) ** 2; ey += (w[j * 5 + 5 + i] - w[j * 5 + i]) ** 2; }
        const m = maps.slice(f * 256, f * 256 + 256), at = m.indexOf(Math.max(...m));
        const dir = ex > 1.5 * ey ? 'changes mostly from left to right, so it answers to upright strokes and edges' : ey > 1.5 * ex ? 'changes mostly from top to bottom, so it answers to level strokes and edges' : 'changes in both directions, so it answers to corners, slants or blobs';
        F.push(row(fin, maps, f, null),
          `Filter ${q.n(f + 1)}: its positive weights add up to ${q.sg(pos, 3)} and its negative ones to ${q.sg(neg, 3)}. It ${dir}. On this image it is strongest at row ${q.n(Math.floor(at / 16) + 1)}, column ${q.n((at % 16) + 1)}, with ${q.sg(m[at], 3)}.`, { phase: 'filter', f: f + 1, pos, neg, ex, ey, row: Math.floor(at / 16) + 1, col: (at % 16) + 1, max: m[at] });
      }
      // the second layer: 12 filters × 6 input maps
      const W2 = P.param(run, 'conv2.W').v, w2m = P.maxAbs(W2), items = [P.text('h2', 0, TOP - 14, 'layer 2: a row per filter, a column per map', 'axname', 13)];
      const strength = Array.from({ length: 12 }, (_, o) => Array.from({ length: 6 }, (_, c) => norm(W2.slice((o * 6 + c) * 25, (o * 6 + c + 1) * 25))));
      for (let o = 0; o < 12; o++) for (let c = 0; c < 6; c++) items.push(P.img('k2:' + o + '_' + c, 40 + c * 34, TOP + o * 34, 30, W2.slice((o * 6 + c) * 25, (o * 6 + c + 1) * 25), { scheme: 'div', max: w2m }));
      for (let o = 0; o < 12; o++) items.push(P.text('k2r' + o, 30, TOP + o * 34 + 20, String(o + 1), 'tick', 12, 'end'));
      for (let c = 0; c < 6; c++) items.push(P.img('k2f' + c, 40 + c * 34, TOP + 12 * 34 + 10, 30, fin.slice(c * 25, c * 25 + 25), { scheme: 'div', max: P.maxAbs(fin) }));
      items.push(P.text('k2fl', 40 + 6 * 34 + 10, TOP + 12 * 34 + 30, '← the first-layer filters whose maps each column reads', 'legend', 12));
      items.push(P.text('h2s', 300, TOP - 14, 'size of each kernel (√Σw²)', 'axname', 13));
      items.push(...S.Kit.heat.grid({ prefix: 'k2s', x: 300, y: TOP, cw: 52, ch: 30, gap: 4, vals: strength, scheme: 'seq', max: Math.max(...strength.flat()), fmt: (v) => P.num(v, 2), size: 12 }).items);
      const colSum = [0, 1, 2, 3, 4, 5].map((c) => strength.reduce((s, r) => s + r[c], 0)), most = colSum.indexOf(Math.max(...colSum)), least = colSum.indexOf(Math.min(...colSum));
      F.push(items, `The second layer’s ${q.n(12)} filters each have a 5 × 5 kernel for every one of the ${q.n(6)} first-layer maps: ${q.n(12 * 6)} small kernels, ${q.n(W2.length)} weights. They are harder to read as pictures, because they combine maps rather than pixels. Taken together they lean most on the map of filter ${q.n(most + 1)} and least on filter ${q.n(least + 1)}.`, { phase: 'layer2', n12: 12, n6: 6, count: 72, weights: W2.length, most: most + 1, least: least + 1 });
      return { frames: F.list, def: { sections: [{ id: 'read', kind: 'math', title: 'Reading a filter', text: ['blue: weight > 0, the pixel adds to the sum', 'vermilion: weight < 0, the pixel takes away', 'blue beside vermilion: an edge or stroke detector', 'the map: the filter slid over the image (padding 2)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'snap' && f.meta.step === 0, once: true, pick: 'option',
      ask: 'After training, will the filters look like random speckle, or like smooth patterns?',
      options: ['smooth patterns', 'random speckle'], answer: 'smooth patterns',
      explain: (fr) => fr.filter((f) => f.meta.phase === 'snap').slice(-1)[0].say,
    }],
    quiz: [
      { q: 'Where do a CNN’s filters come from?', options: ['they are learned by gradient descent from random starts', 'they are written by hand', 'they are copied from the images', 'they are fixed edge detectors'], answer: 0, why: 'Training adjusts every weight to lower the loss.' },
      { q: 'A filter with a blue left column and a vermilion right column answers most to…', options: ['ink on the left, paper on the right', 'ink on the right, paper on the left', 'a uniform patch', 'nothing'], answer: 0, why: 'Ink under positive weights raises the sum.' },
      { q: 'Why are second-layer filters harder to read as pictures?', options: ['they combine feature maps, not pixels', 'they are random', 'they are larger', 'they have no weights'], answer: 0, why: 'Their inputs are the first layer’s maps.' },
      { q: 'How many weights do the second layer’s twelve 5 × 5 filters over six maps have, biases aside?', options: ['1800', '300', '72', '150'], answer: 0, why: '12 · 6 · 25.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
