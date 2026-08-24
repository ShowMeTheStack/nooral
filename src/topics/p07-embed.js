// Part 7, maps of high-dimensional data: t-SNE running (each point's neighbourhood as probabilities set by the
// perplexity, then a layout that matches them, with early exaggeration first), and a UMAP-style layout (a fuzzy
// neighbour graph, pulled together along its edges and pushed apart at random pairs, epoch by epoch).
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const memo = new Map(), once = (k, f) => { if (!memo.has(k)) { if (memo.size > 24) memo.clear(); memo.set(k, f()); } return memo.get(k); };
  /** sets in many dimensions, with known classes for the colours */
  P.HIGH = {
    blobs: { label: 'four groups in 8 dimensions', make: () => { const r = S.Data.rng(17), cs = [0, 1, 2, 3].map(() => Array.from({ length: 8 }, () => r.range(-4, 4))), X = [], y = []; for (let i = 0; i < 48; i++) { const c = i % 4; X.push(cs[c].map((v) => +(v + r.normal(0, c === 3 ? 1.6 : 0.8)).toFixed(4))); y.push(c); } return { X, y, names: ['group A', 'group B', 'group C', 'group D (wider)'] }; } },
    iris: { label: 'iris: 4 measurements, every third flower', make: () => { const d = S.Data.load('iris'), X = [], y = []; d.X.forEach((x, i) => { if (i % 3 === 0) { X.push(x); y.push(d.y[i]); } }); return { X, y, names: ['setosa', 'versicolor', 'virginica'] }; } },
    digits: { label: 'digits 0, 1, 4 and 7 as 100 pixels', make: () => { const d = S.Data.digits({ n: 160, size: 10, seed: 5 }), X = [], y = []; d.X.forEach((x, i) => { const k = [0, 1, 4, 7].indexOf(d.y[i]); if (k >= 0 && X.length < 48) { X.push(Array.from(x)); y.push(k); } }); return { X, y, names: ['digit 0', 'digit 1', 'digit 4', 'digit 7'] }; } },
  };
  const setOpts = Object.keys(P.HIGH).map((k) => ({ value: k, label: P.HIGH[k].label }));
  const ITERS = 300, EXAG = 250;

  S.topic({
    id: 'tsne-run', family: 'p07-embed', order: 10, title: 't-SNE: a map that keeps neighbours together',
    blurb: 't-SNE turns distances in many dimensions into probabilities: point j is a likely neighbour of point i if it is close, with each point’s bandwidth set so that it has about “perplexity” effective neighbours. It then places points on a flat map where a heavier-tailed (Student t) similarity should give the same probabilities, moving them by gradient descent on the KL divergence between the two. For the first iterations the high-dimensional probabilities are exaggerated, so groups form early.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'blobs', options: setOpts },
      { id: 'perp', kind: 'slider', label: 'Perplexity', def: 12, min: 2, max: 30, step: 1 },
      { id: 'seed', kind: 'number', label: 'Seed (random start)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Four groups, perplexity 12', values: { set: 'blobs', perp: 12, seed: 1 } },
      { label: 'Perplexity 2: tiny neighbourhoods', values: { set: 'blobs', perp: 2, seed: 1 } },
      { label: 'Perplexity 30: wide neighbourhoods', values: { set: 'blobs', perp: 30, seed: 1 } },
      { label: 'Iris', values: { set: 'iris', perp: 15, seed: 2 } },
      { label: 'Digit images', values: { set: 'digits', perp: 10, seed: 3 } },
    ],
    counters: [{ id: 'iter', label: 'iterations' }],
    run(inp) {
      const d = P.HIGH[inp.set].make(), X = d.X, n = X.length, F = H().frames(['iter']), q = F.q;
      const ts = once(['t', inp.set, inp.perp, inp.seed].join(':'), () => { const t = ML().TSNE({ perplexity: inp.perp, maxIter: ITERS, init: 'random', seed: inp.seed, snapshotEvery: 1, method: 'exact' }); t.fitTransform(X); t.trust = ML().trustworthiness(X, t.embedding, 5); return t; });
      const snaps = ts.snapshots, last = snaps.length - 1;
      // shown: the layout before iteration t's step, with the KL computed on it
      const AT = [1, 10, 25, 50, 100, 175, 249, 250, 260, 275, last].filter((t, i, a) => t <= last && a.indexOf(t) === i);
      const Yof = (t) => snaps[t - 1].Y;
      const box = { x: 40, y: 20, w: 430, h: 430 };
      const KA = { x: 560, y: 50, w: 210, h: 170 }, kls = snaps.map((s) => s.kl), lg = (v) => Math.log10(Math.max(1e-3, v));
      const lo = Math.floor(lg(Math.min(...kls.slice(1)))), hi = Math.ceil(lg(Math.max(...kls.slice(1))));
      const kax = C().axes(Object.assign({ prefix: 'tk', xr: [0, ITERS], yr: [lo, hi], xticks: [0, 100, 200, 300], yticks: [...Array(hi - lo + 1).keys()].map((i) => lo + i), fmtY: (v) => String(+Math.pow(10, v).toPrecision(1)), xname: 'iteration', yname: 'KL divergence (log scale)' }, KA));
      const legend = d.names.flatMap((nm, k) => P.row('tl:' + k, 545, 300 + k * 24, k, nm));
      AT.forEach((t, u) => {
        F.set('iter', t);
        const Y = Yof(t), ax = P.axes('ts', Y, box, { xname: 'map x', yname: 'map y' });
        ax.items = ax.items.filter((it) => !/:t:|:grid/.test(it.key));
        const items = [...ax.items, ...P.points(ax, 'ts:p', Y, d.y, { r: 4.6 }), ...kax.items, { key: 'tk:ex', kind: 'cell', x: kax.X(0), y: KA.y, w: kax.X(EXAG) - kax.X(0), h: KA.h, rx: 0, text: '', cls: 'panel', opacity: 0.5, audit: 'ignore', layer: 'back' }, H().text('tk:ext', kax.X(EXAG / 2), KA.y + KA.h - 8, 'exaggerated', 'axname', 12, 'middle'), C().line(kax, 'tk:l', kls.slice(1, t + 1).map((v, i) => [i + 1, lg(v)]), 'act'), C().dot(kax, 'tk:d', t, lg(kls[t]), '', 5), ...legend];
        const ex = t < EXAG;
        let say;
        if (u === 0) say = `${P.qpl(q, n, 'point')} in ${q.n(X[0].length)} dimensions start at random, almost on top of one another (the map is rescaled in every picture). Each point’s neighbourhood is set by perplexity ${q.n(inp.perp)}: about that many effective neighbours. During the first ${q.n(EXAG)} iterations the attractions are multiplied by ${q.n(12)}, and the KL divergence measured is for those exaggerated probabilities: ${q(kls[t], 3)}.`;
        else if (t >= EXAG && AT[u - 1] < EXAG) say = `Iteration ${q.n(t)}: the exaggeration is switched off and the momentum raised. The KL divergence, now against the true probabilities, is ${q(kls[t], 3)}; the groups loosen and the points inside each spread out.`;
        else if (t === last) say = `Iteration ${q.n(t)}: KL divergence ${q(kls[t], 3)}. Points with the same colour, which were close in ${q.n(X[0].length)} dimensions, sit together. Trustworthiness, which is perfect when each point’s ${q.n(5)} nearest map neighbours are also its nearest in the data, is ${q(ts.trust, 3)} (at most one). Distances between groups on the map mean little.`;
        else say = `Iteration ${q.n(t)}${ex ? ' (exaggerated)' : ''}: KL divergence ${q(kls[t], 3)}. ` + (ex ? 'Attraction to true neighbours is so strong that groups pull into tight clumps.' : 'Each point settles among its neighbours.');
        F.push(items, say, { phase: ex ? 'exag' : 'free', t, kl: kls[t], Y, exag: ex ? 12 : 1, n, trust: ts.trust, perp: inp.perp, dim: X[0].length, iters: EXAG, factor: 12, k: 5, Yfinal: t === last ? ts.embedding : null });
      });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 't-SNE', text: ['pⱼ|ᵢ ∝ exp(−‖xᵢ − xⱼ‖² / 2σᵢ²), σᵢ set by the perplexity', 'pᵢⱼ = (pⱼ|ᵢ + pᵢ|ⱼ) / 2n', 'qᵢⱼ ∝ (1 + ‖yᵢ − yⱼ‖²)⁻¹ on the map', 'minimise KL(P ‖ Q) = Σ pᵢⱼ log(pᵢⱼ / qᵢⱼ)', 'early exaggeration: P × 12 for 250 iterations'].join('\n') }] } };
    },
    predict: [{ at: (f, i, fr) => f.meta.t === 1, pick: 'option', ask: 'At the end, where will points of the same colour be?', options: ['together, in their own patch', 'spread evenly over the map', 'in a single line', 'where they started'], answer: 'together, in their own patch', explain: (fr) => fr[fr.length - 1].say }],
    quiz: [
      { q: 'What does the perplexity set?', options: ['roughly how many neighbours each point pays attention to', 'the number of iterations', 'the number of dimensions of the map', 'the learning rate'], answer: 0, why: 'Each σᵢ is tuned so that the entropy of point i’s neighbour distribution equals log(perplexity).' },
      { q: 'Why use a heavy-tailed t similarity on the map?', options: ['so moderately far points can sit far apart, easing the crowding of many dimensions into two', 'it is faster', 'to make the map symmetric', 'to remove the need for perplexity'], answer: 0, why: 'There is far more room at a middle distance in many dimensions than on a plane.' },
      { q: 'Can you read the distance between two clusters on a t-SNE map?', options: ['not reliably: t-SNE keeps neighbourhoods, not large distances', 'yes, exactly', 'yes, after scaling', 'only with perplexity 30'], answer: 0, why: 'Only local structure is matched; cluster sizes and gaps can mislead.' },
      { q: 'What does early exaggeration do?', options: ['multiplies the data’s probabilities so groups form tight clumps early', 'slows the optimiser', 'adds noise to the start', 'doubles the dimensions'], answer: 0, why: 'Strong attraction early lets groups find each other before the fine layout.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'umap-run', family: 'p07-embed', order: 20, title: 'A UMAP-style map: a neighbour graph, pulled into shape',
    blurb: 'UMAP-style layouts first build a fuzzy graph: each point links to its k nearest neighbours, with weights that start at 1 for the nearest and fade with each point’s own distance scale, and the two directions of a link are combined. Then a layout is optimised epoch by epoch: each edge pulls its two ends together, and a few random pairs are pushed apart. min_dist sets how tightly points may pack on the map.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'blobs', options: setOpts },
      { id: 'k', kind: 'number', label: 'Neighbours k', def: 10, min: 2, max: 30 },
      { id: 'minDist', kind: 'slider', label: 'min_dist', def: 0.1, min: 0.01, max: 1, step: 0.01 },
    ],
    presets: [
      { label: 'Four groups', values: { set: 'blobs', k: 10, minDist: 0.1 } },
      { label: 'Few neighbours: k = 3', values: { set: 'blobs', k: 3, minDist: 0.1 } },
      { label: 'Loose packing: min_dist 0.8', values: { set: 'blobs', k: 10, minDist: 0.8 } },
      { label: 'Iris', values: { set: 'iris', k: 12, minDist: 0.1 } },
      { label: 'Digit images', values: { set: 'digits', k: 8, minDist: 0.1 } },
    ],
    counters: [{ id: 'epoch', label: 'epochs' }],
    run(inp) {
      const d = P.HIGH[inp.set].make(), X = d.X, n = X.length, F = H().frames(['epoch']), q = F.q, EP = 200;
      const um = once(['u', inp.set, inp.k, inp.minDist].join(':'), () => { const u = ML().UMAP({ nNeighbors: inp.k, minDist: inp.minDist, nEpochs: EP, snapshotEvery: 10, seed: 0 }); u.fitTransform(X); u.trust = ML().trustworthiness(X, u.embedding, 5); return u; });
      const G = um.graph, edges = G.edges, box = { x: 40, y: 20, w: 430, h: 430 };
      const show = [0, 1, 2, 4, 7, 10, 15, um.snapshots.length - 1].filter((i, j, a) => i < um.snapshots.length && a.indexOf(i) === j);
      const legend = d.names.flatMap((nm, k) => P.row('ul:' + k, 545, 300 + k * 24, k, nm));
      const wmax = Math.max(...edges.map((e) => e.w));
      const ex = 0, nb = edges.filter((e) => e.i === ex || e.j === ex).length;
      show.forEach((s, u) => {
        const sn = um.snapshots[s], Y = sn.Y, ep = sn.epoch, ax = P.axes('um', Y, box, { xname: 'map x', yname: 'map y' });
        F.set('epoch', ep);
        ax.items = ax.items.filter((it) => !/:t:|:grid/.test(it.key));
        const strong = edges.filter((e) => e.w > 0.5 * wmax), weak = edges.filter((e) => e.w <= 0.5 * wmax);
        const path = (es) => es.map((e) => `M${ax.X(Y[e.i][0])},${ax.Y(Y[e.i][1])} L${ax.X(Y[e.j][0])},${ax.Y(Y[e.j][1])}`).join(' ') || 'M0,0';
        const items = [...ax.items, { key: 'um:ew', kind: 'path', d: path(weak), cls: 'p7nb', opacity: 0.35, audit: 'ignore', layer: 'back', z: 3 }, { key: 'um:es', kind: 'path', d: path(strong), cls: 'p7nb', audit: 'ignore', layer: 'back', z: 4 }, ...P.points(ax, 'um:p', Y, d.y, { r: 4.6 }), ...legend];
        items.push(H().head('um:h', 545, 60, 'epoch ' + ep + ' of ' + EP), H().text('um:h2', 545, 84, H().pl(edges.length, 'edge') + ' in the graph', 'legend', 13), H().text('um:h3', 545, 106, 'dark: weight over half the largest', 'axname', 12));
        let say;
        const alpha = ep > 0 ? 1 - (ep - 1) / EP : 1;
        if (u === 0) say = `The fuzzy graph: each of the ${P.qpl(q, n, 'point')} links to its ${q.n(inp.k)} nearest neighbours. Point ${q.n(ex)} has its nearest neighbour at ρ = ${q(G.rhos[ex], 3)} and a scale σ = ${q(G.sigmas[ex], 3)} chosen so its ${q.n(inp.k)} weights add up to log₂ k = ${q(Math.log2(inp.k), 3)}; it takes part in ${P.qpl(q, nb, 'edge')}. Merging the two directions leaves ${P.qpl(q, edges.length, 'edge')}. The layout starts from a scaled PCA projection.`;
        else if (s === um.snapshots.length - 1) say = `Epoch ${q.n(ep)}: the layout has settled. Trustworthiness of the ${q.n(5)} nearest map neighbours is ${q(ts(um), 3)}. Edges are short inside the groups, while links between groups are few and weak.`;
        else say = `Epoch ${q.n(ep)}: every edge pulled its ends together (strong edges more often) and each pull came with random pairs pushed apart. The step size has fallen to ${q(alpha, 3)}.`;
        F.push(items, say, { phase: u === 0 ? 'graph' : 'epoch', ep, alpha, n, k: inp.k, rho: G.rhos[ex], sigma: G.sigmas[ex], ex, nb, edges: edges.length, trust: um.trust, log2k: Math.log2(inp.k), five: 5 });
      });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A UMAP-style layout', text: ['ρᵢ = distance to the nearest neighbour', 'wᵢⱼ = exp(−(dᵢⱼ − ρᵢ) / σᵢ) for the k nearest,', '  with Σⱼ wᵢⱼ = log₂ k', 'symmetrise: w = a + b − ab', 'map similarity: 1 / (1 + a·d^2b), a, b from min_dist', 'each epoch: pull along edges, push random pairs'].join('\n') }] } };
    },
    predict: [{ at: (f) => f.meta.phase === 'graph', pick: 'option', ask: 'With only 3 neighbours per point instead of 10, the groups on the map would be…', options: ['split into smaller pieces more easily', 'merged into one', 'exactly the same', 'arranged on a circle'], answer: 'split into smaller pieces more easily', explain: () => 'With few neighbours the graph only knows very local structure, so a group can fall apart into islands; try the preset with k = 3.' }],
    quiz: [
      { q: 'Why does each point get its own scale σᵢ?', options: ['so sparse and dense regions both give each point about the same total weight', 'to speed up the layout', 'to make the graph directed', 'it does not'], answer: 0, why: 'The weights of each point’s k links are tuned to add up to log₂ k.' },
      { q: 'Two points link with weights 0.6 one way and 0.5 the other. The merged weight?', options: ['0.8', '1.1', '0.55', '0.3'], answer: 0, why: 'a + b − ab = 0.6 + 0.5 − 0.3.' },
      { q: 'What does a larger min_dist do?', options: ['lets points spread out more within a group', 'removes edges', 'adds neighbours', 'shrinks the map to a point'], answer: 0, why: 'It sets the distance below which the map similarity is near 1, so points need not pack tightly.' },
      { q: 'Why push random pairs apart?', options: ['otherwise every edge’s pull would collapse the map', 'to add noise for privacy', 'to find the neighbours', 'to set k'], answer: 0, why: 'Negative sampling is a cheap stand-in for repulsion between all non-neighbours.' },
    ],
  });
  const ts = (u) => u.trust;
})(globalThis.Nooral = globalThis.Nooral || {});
