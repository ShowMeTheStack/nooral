// Part 10, feature maps: one validation image flowing through the CNN trained here (6 filters of 5 × 5, pool, 12
// filters of 5 × 5, pool, 32 hidden units, the classes), every named output of the engine drawn as it is reached.
(function (S) {
  'use strict';
  const P = S.P10, H = () => S.Kit.ml;
  const TOP = 40, RS = 72, IX = 0, IY = 168, IS = 128, C1 = 168, M1 = 60, P1 = 246, M2 = 40, C2 = 312, M3 = 30, P2 = 400, M4 = 16, DX = 462, BX = 498;
  const STAGES = ['input', 'conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2', 'dense', 'out'];

  S.topic({
    id: 'cnn-flow', family: 'p10-inside', order: 10, title: 'Feature maps: an image flowing through a trained CNN',
    blurb: 'A small network trained here on generated images: six 5 × 5 filters, ReLU and 2 × 2 max pooling, then twelve 5 × 5 filters, ReLU and pooling again, then 32 hidden units and one score per class. Follow one image through it. Each layer turns the maps of the layer before into new maps, smaller and more abstract, until a list of numbers is left for the dense layers to vote with.',
    inputs: P.imageInputs('digits', 7, 0),
    presets: [
      { label: 'A 7', values: { set: 'digits', cls: 7, k: 0 } },
      { label: 'A 3', values: { set: 'digits', cls: 3, k: 2 } },
      { label: 'A 5 the network gets wrong', values: { set: 'digits', cls: 5, k: 8 } },
      { label: 'A star', values: { set: 'shapes', cls: 3, k: 0 } },
      { label: 'A square taken for a triangle', values: { set: 'shapes', cls: 1, k: 0 } },
    ],
    counters: [{ id: 'layers', label: 'layers passed' }],
    run(inp) {
      const run = inp.set, im = P.pick(run, inp.cls, inp.k), o = P.forward(run, im.px), F = H().frames(['layers']), q = F.q;
      const nC = P.D()[run].classes.length, probs = o.probs, best = probs.indexOf(Math.max(...probs));
      const at = (s) => STAGES.indexOf(s);
      const draw = (stage) => {
        const si = at(stage), items = [];
        const head = (key, x, y, t) => items.push(P.text('h:' + key, x, y, t, 'axname', 13));
        head('in', IX, IY - 12, 'input, 16 × 16');
        items.push(P.img('in', IX, IY, IS, im.px, { cls: si === 0 ? 'act' : '' }));
        head('c1', C1, TOP - 14, si >= 2 ? 'ReLU 1' : 'conv 1');
        head('p1', P1, TOP - 14, 'pool');
        head('c2', C2, TOP - 14, si >= 5 ? 'ReLU 2' : 'conv 2');
        head('p2', P2, TOP - 14, 'pool');
        head('d', DX - 6, TOP - 14, 'dense');
        head('o', BX + 26, TOP - 14, 'chances');
        const m1 = si === 1 ? o.conv1.v : o.relu1.v, c1max = P.maxAbs(o.conv1.v), m3 = si === 4 ? o.conv2.v : o.relu2.v, c3max = P.maxAbs(o.conv2.v);
        for (let c = 0; c < 6; c++) {
          const y = TOP + c * RS;
          items.push(si >= 1 ? P.img('c1:' + c, C1, y, M1, P.map(m1, c, 16), { scheme: si === 1 ? 'div' : 'seq', max: c1max, cls: si <= 2 && si >= 1 ? 'act' : '' }) : P.box('c1e:' + c, C1, y, M1, M1, 'empty', 'items'));
          items.push(si >= 3 ? P.img('p1:' + c, P1, y + 10, M2, P.map(o.pool1.v, c, 8), { scheme: 'seq', max: c1max, cls: si === 3 ? 'act' : '' }) : P.box('p1e:' + c, P1, y + 10, M2, M2, 'empty', 'items'));
          for (let h = 0; h < 2; h++) {
            const ch = 2 * c + h, x = C2 + h * (M3 + 8), y2 = y + 15;
            items.push(si >= 4 ? P.img('c2:' + ch, x, y2, M3, P.map(m3, ch, 4), { scheme: si === 4 ? 'div' : 'seq', max: c3max, cls: si === 4 || si === 5 ? 'act' : '' }) : P.box('c2e:' + ch, x, y2, M3, M3, 'empty', 'items'));
            const x4 = P2 + h * (M4 + 8), y4 = y + 22;
            items.push(si >= 6 ? P.img('p2:' + ch, x4, y4, M4, P.map(o.pool2.v, ch, 2), { scheme: 'seq', max: c3max, cls: si === 6 ? 'act' : '' }) : P.box('p2e:' + ch, x4, y4, M4, M4, 'empty', 'items'));
          }
        }
        const dh = 6 * RS - 12;
        items.push(si >= 7 ? P.img('dense', DX, TOP, 14, o.fcAct0.v, { n: 1, m: 32, h: dh, scheme: 'seq', max: P.maxAbs(o.fcAct0.v), cls: si === 7 ? 'act' : '' }) : P.box('dense:e', DX, TOP, 14, dh, 'empty', 'items'));
        if (si >= 8) items.push(...P.chances('pr', BX, TOP, run, probs, { rowH: nC > 4 ? 40 : 60, w: 230 }));
        else items.push(P.box('pr:e', BX, TOP, 160, 6 * RS - 12, 'empty', 'items'));
        return items;
      };
      const name = P.qwords(q, run, im.y);
      F.push(draw('input'), `The input: ${name}, a validation image the network never trained on (example ${q.n(inp.k)} of its class). It is ${q.n(16)} × ${q.n(16)} grey levels, ${q.n(256)} numbers.`, { phase: 'input', y: im.y, k: inp.k, side: 16, px: 256 });
      const c1 = o.conv1.v, c1s = [0, 1, 2, 3, 4, 5].map((c) => Math.max(...P.map(c1, c, 16))), top1 = c1s.indexOf(Math.max(...c1s));
      F.count('layers');
      F.push(draw('conv1'), `Six 5 × 5 filters, each padded so the size stays ${q.n(16)} × ${q.n(16)}, give six maps (blue positive, vermilion negative). Values run from ${q.sg(Math.min(...c1), 3)} to ${q.sg(Math.max(...c1), 3)}; map ${q.n(top1 + 1)} answers most strongly.`, { phase: 'conv1', min: Math.min(...c1), max: Math.max(...c1), top: top1 + 1, side: 16 });
      const z1 = c1.filter((v) => v <= 0).length;
      F.push(draw('relu1'), `ReLU sets every negative value to zero: ${q.n(z1)} of the ${q.n(c1.length)} values (${q.pct(z1 / c1.length)}) become zero. What is left says where each filter’s pattern is present.`, { phase: 'relu1', zeroed: z1, total: c1.length, frac: z1 / c1.length });
      F.count('layers');
      F.push(draw('pool1'), `Max pooling keeps the largest of each 2 × 2 block: six ${q.n(8)} × ${q.n(8)} maps, ${q.n(o.pool1.v.length)} numbers instead of ${q.n(c1.length)}.`, { phase: 'pool1', side: 8, n: o.pool1.v.length, before: c1.length });
      const c2 = o.conv2.v, field = 5 + (2 - 1) + (5 - 1) * 2;
      F.count('layers');
      F.push(draw('conv2'), `Twelve filters, each with a 5 × 5 kernel for every one of the six pooled maps (no padding), give twelve ${q.n(4)} × ${q.n(4)} maps. Each value now depends on a ${q.n(field)} × ${q.n(field)} patch of the input, most of the picture.`, { phase: 'conv2', side: 4, field });
      const z2 = c2.filter((v) => v <= 0).length, s2 = Array.from({ length: 12 }, (_, c) => P.map(o.relu2.v, c, 4).reduce((a, b) => a + b, 0)), top2 = s2.indexOf(Math.max(...s2)), dead = s2.filter((v) => v === 0).length;
      F.push(draw('relu2'), `After ReLU, ${q.n(z2)} of the ${q.n(c2.length)} values are zero` + (dead ? ` and ${q.n(dead)} of the twelve maps ${dead === 1 ? 'is' : 'are'} dark everywhere for this image` : '') + `. Map ${q.n(top2 + 1)} is the busiest, with a total of ${q.sg(s2[top2], 3)}.`, { phase: 'relu2', zeroed: z2, total: c2.length, dead, top: top2 + 1, sum: s2[top2] });
      F.count('layers');
      F.push(draw('pool2'), `Pooling again leaves twelve ${q.n(2)} × ${q.n(2)} maps: ${q.n(o.pool2.v.length)} numbers, laid out in one list for the dense layers.`, { phase: 'pool2', side: 2, n: o.pool2.v.length });
      const d = o.fcAct0.v, on = d.filter((v) => v > 0).length, du = d.indexOf(Math.max(...d));
      F.count('layers');
      F.push(draw('dense'), `A dense layer of ${q.n(d.length)} units, each weighing all ${q.n(o.pool2.v.length)} numbers, then ReLU: ${q.n(on)} units are active, the strongest unit ${q.n(du + 1)} at ${q.sg(d[du], 3)}.`, { phase: 'dense', units: d.length, inputs: o.pool2.v.length, on, top: du + 1, max: d[du] });
      const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]), second = order[1][1];
      F.count('layers');
      F.push(draw('out'), `The last layer gives one score per class, and softmax turns the scores into chances: ${P.qname(q, run, best)} at ${q.pct(probs[best])}, ` + (probs[second] < 0.005 ? 'every other class under one percent. ' : `then ${P.qname(q, run, second)} at ${q.pct(probs[second])}. `) + (best === im.y ? 'Right.' : `Wrong: it was ${P.qname(q, run, im.y)}.`), { phase: 'out', best, second, pBest: probs[best], pSecond: probs[second], y: im.y, right: best === im.y });
      return { frames: F.list, def: { sections: [{ id: 'arch', kind: 'math', title: 'The network', text: ['input 1 × 16 × 16', 'conv 6 × 5 × 5, same padding → 6 × 16 × 16, ReLU', 'max pool 2 → 6 × 8 × 8', 'conv 12 × 5 × 5 → 12 × 4 × 4, ReLU', 'max pool 2 → 12 × 2 × 2 = 48 numbers', 'dense 48 → 32, ReLU; dense 32 → classes; softmax'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'input', once: true, pick: 'option',
      ask: 'Which class will the network choose?',
      options: (fr, i, inp) => P.D()[inp.set].classes.map(String),
      answer: (fr, i, inp) => P.cname(inp.set, fr[fr.length - 1].meta.best),
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'In this network, the maps get smaller from 16 × 16 to 8 × 8 because of…', options: ['2 × 2 max pooling', 'the 5 × 5 filters', 'ReLU', 'the dense layer'], answer: 0, why: 'The first convolution is padded and keeps the size; pooling halves it.' },
      { q: 'The second convolution’s filters each read…', options: ['all six pooled maps', 'one map each', 'the raw image only', 'the class scores'], answer: 0, why: 'A filter spans every channel of its input.' },
      { q: 'Why are many values exactly zero after ReLU?', options: ['ReLU replaces every negative value by zero', 'the filters are zero', 'pooling removes them', 'the image is mostly zero'], answer: 0, why: 'max(0, x).' },
      { q: 'Deeper maps are smaller, but each of their values depends on…', options: ['a larger patch of the input', 'a single pixel', 'only the image’s centre', 'nothing from the input'], answer: 0, why: 'Their receptive field grows with every layer.' },
      { q: 'Softmax turns the class scores into…', options: ['chances that add up to one', 'a single yes or no', 'pixel values', 'gradients'], answer: 0, why: 'eˢ / Σ eˢ for each class.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
