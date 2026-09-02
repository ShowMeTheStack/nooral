// Part 11, the trained word vectors as a map: the words in 2-D along their two main directions, a word's nearest
// neighbours by cosine in the full vectors, and an analogy done as arithmetic (b − a + c) with the words nearest to
// the result.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11;
  const MAP = { x: 0, y: 30, w: 480, h: 430 };
  const ANALOGIES = {
    royal: { abc: ['man', 'woman', 'king'], want: 'queen' },
    young: { abc: ['king', 'queen', 'prince'], want: 'princess' },
    folk: { abc: ['boy', 'girl', 'man'], want: 'woman' },
    age: { abc: ['old', 'young', 'king'], want: 'prince' },
    pronoun: { abc: ['he', 'she', 'his'], want: 'her' },
  };
  const label = (k) => { const a = ANALOGIES[k].abc; return a[0] + ' : ' + a[1] + ' = ' + a[2] + ' : ?'; };

  S.topic({
    id: 'word2vec-map', family: 'p11-embed', order: 30, title: 'The map of word vectors: neighbours and analogies',
    blurb: 'Once trained, every word is a point in a space of 16 numbers. Flattened to two dimensions it becomes a map where kinds of words form clumps. A word’s nearest neighbours by cosine are the words used like it, and differences between vectors can carry a relation: the step from “man” to “woman”, added to “king”, lands near “queen”.',
    inputs: [
      { id: 'probe', kind: 'choice', label: 'Word', def: 'horse', options: ['horse', 'queen', 'cheese', 'yellow', 'four', 'market', 'cart', 'milk'].map((w) => ({ value: w, label: w })) },
      { id: 'k', kind: 'number', label: 'Neighbours', def: 5, min: 3, max: 6 },
      { id: 'analogy', kind: 'choice', label: 'Analogy', def: 'royal', options: Object.keys(ANALOGIES).map((k) => ({ value: k, label: label(k) })) },
      { id: 'seed', kind: 'number', label: 'Training seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'man : woman = king : ?', values: { probe: 'horse', k: 5, analogy: 'royal', seed: 1 } },
      { label: 'king : queen = prince : ?', values: { probe: 'queen', k: 5, analogy: 'young', seed: 1 } },
      { label: 'boy : girl = man : ?', values: { probe: 'cheese', k: 4, analogy: 'folk', seed: 2 } },
      { label: 'old : young = king : ?', values: { probe: 'four', k: 6, analogy: 'age', seed: 1 } },
      { label: 'he : she = his : ?', values: { probe: 'market', k: 3, analogy: 'pronoun', seed: 3 } },
    ],
    counters: [{ id: 'cmp', label: 'cosines compared' }],
    run(inp) {
      const F = H().frames(['cmp']), q = F.q, run = P.w2v({ seed: inp.seed }), w = run.w, V = w.vocab.length, d = w.o.dim;
      const A = ANALOGIES[inp.analogy], [a, b, c] = A.abc;
      const nb = w.nearest(inp.probe, inp.k), an = w.analogy(a, b, c, 3).answers;
      const extra = [...nb.map((x) => x.label), a, b, c, ...an.map((x) => x.label)];
      const words = [...new Set(P.mapWords().concat(extra))];
      const pr = w.project(words), pts = pr.points, at = (x) => pts[words.indexOf(x)];
      const legend = P.kindLegend('wm:lg', 530, 60);
      const panel = (rows, y) => H().lines('wm:p', 530, y || 250, rows, 13, 22);
      const map = (o) => P.wordMap(words, pts, Object.assign({ prefix: 'wm:m', box: MAP }, o));
      let m = map({});
      F.push([...m.items, ...legend, ...panel([['each dot: one word', 'strong'], [d + ' numbers flattened to 2', 'legend'], ['the 2 axes keep ' + H().pct(pr.explained[0] + pr.explained[1]) + ' of the spread', 'legend']])],
        `The ${P.qpl(q, d, 'number')} of each word’s vector, flattened onto the two directions along which the words spread most. Those two directions keep ${q.pct(pr.explained[0] + pr.explained[1])} of the spread, so the map is a fair but imperfect picture: kinds of words sit in clumps.`, { phase: 'map', d, explained: pr.explained, share: pr.explained[0] + pr.explained[1] });
      // neighbours of the probe
      F.count('cmp', V - 1);
      const nbW = nb.map((x) => x.label);
      m = map({ act: inp.probe, lit: new Set(nbW), faded: true, first: [inp.probe].concat(nbW) });
      const lines = nbW.map((x) => { const p0 = at(inp.probe), p1 = at(x); return { key: 'wm:nl:' + x, kind: 'path', d: `M${m.X(p0[0])},${m.Y(p0[1])} L${m.X(p1[0])},${m.Y(p1[1])}`, cls: 'p11nb', audit: 'ignore', layer: 'lines' }; });
      const nbBars = K().bars({ prefix: 'wm:nb', x: 530, y: 262, w: 260, rows: nb.map((x) => ({ label: x.label, v: Math.max(0, x.score), value: H().fx(x.score, 2) })), max: 1, labelW: 74, rowH: 24 });
      const off = nbW.filter((x) => P.catOf(x) !== P.catOf(inp.probe));
      F.push([...m.items, ...lines, ...legend, H().text('wm:nt', 530, 250, 'nearest to “' + inp.probe + '” (cosine)', 'strong', 13), ...nbBars],
        `“${inp.probe}” is compared with all ${P.qpl(q, V - 1, 'other word')} by cosine, in the full ${P.qpl(q, d, 'number')}. Its nearest are ${H().list(nb.map((x) => x.label + ' (' + q(x.score, 2) + ')'))}.` + (off.length ? ` Not all are of its kind (${H().list(off)}): they share some of its company in the text.` : ' All are of its kind.'), { phase: 'probe', nb: nb.map((x) => ({ label: x.label, score: x.score })), others: V - 1, d });
      // the analogy, on a map of the words it involves (people and pronouns spread out along their own directions)
      const sub = [...new Set((inp.analogy === 'pronoun' ? ['he', 'she', 'his', 'her'] : []).concat(P.CATS.people, [a, b, c], an.map((x) => x.label)))];
      // the plane of the analogy: across, the step from a to b; up, the step from a to c (the part not along b − a)
      const rows = w.rows(), R = (x) => rows[w.index.get(x)], dot = (u, v) => u.reduce((t, x, j) => t + x * v[j], 0);
      const ab = R(b).map((x, j) => x - R(a)[j]), e1 = ab.map((x) => x / Math.sqrt(dot(ab, ab))), ac = R(c).map((x, j) => x - R(a)[j]), along = dot(ac, e1);
      const up = ac.map((x, j) => x - along * e1[j]), e2 = up.map((x) => x / Math.sqrt(dot(up, up)));
      const pr2 = { points: sub.map((x) => { const v = R(x).map((y, j) => y - R(a)[j]); return [dot(v, e1), dot(v, e2)]; }) }, at2 = (x) => pr2.points[sub.indexOf(x)];
      const map2 = (o) => P.wordMap(sub, pr2.points, Object.assign({ prefix: 'wm:m', box: MAP }, o));
      const pa = at2(a), pb = at2(b), pc = at2(c), pt = [pc[0] + pb[0] - pa[0], pc[1] + pb[1] - pa[1]];
      const arrow = (key, p0, p1, cls) => ({ key, kind: 'edge', from: { x: m.X(p0[0]), y: m.Y(p0[1]) }, to: { x: m.X(p1[0]), y: m.Y(p1[1]) }, cls: 'mlarrow ' + cls, head: true, audit: 'ignore' });
      const P2 = pr2.points, range = { u0: Math.min(...P2.map((p) => p[0]), pt[0]), u1: Math.max(...P2.map((p) => p[0]), pt[0]), v0: Math.min(...P2.map((p) => p[1]), pt[1]), v1: Math.max(...P2.map((p) => p[1]), pt[1]) };
      m = map2({ range, act: c, lit: new Set([a, b]), faded: true, first: [a, b, c] });
      F.push([...m.items, arrow('wm:ab', pa, pb, 'p11an'), ...legend, ...panel([['step: ' + b + ' − ' + a, 'math'], ['the difference of two vectors', 'legend'], ['a direction in the space', 'legend']])],
        `An analogy as arithmetic: “${a} is to ${b} as ${c} is to what?” This map shows the ${P.qpl(q, sub.length, 'word')} involved in the flat slice of the space that holds “${a}”, “${b}” and “${c}”: across is the step from “${a}” to “${b}”, up the step from “${a}” to “${c}”. The step from “${a}” to “${b}” is the difference of their vectors, ${b} − ${a}. If the vectors learned a relation, the same step should lead from “${c}” to the answer.`, { phase: 'arrow', abc: A.abc, nSub: sub.length });
      const top = an[0];
      m = map2({ range, act: c, lit: new Set([a, b, top.label]), faded: true, first: [c, top.label, a, b] });
      const target = { key: 'wm:tg', kind: 'dot', x: m.X(pt[0]), y: m.Y(pt[1]), r: 7, cls: 'ring', audit: 'overlap-ok', layer: 'items', z: 4 };
      F.push([...m.items, arrow('wm:ab', pa, pb, 'p11an'), arrow('wm:ct', pc, pt, 'p11an2'), target, ...legend, ...panel([[c + ' + (' + b + ' − ' + a + ')', 'math'], ['the ring: where it lands', 'legend']])],
        `Adding the same step to “${c}” gives a new vector, ${c} + ${b} − ${a}, drawn as the ring: in this slice, “${a}”, “${b}”, “${c}” and the ring make a parallelogram. The ring is not a word itself, so the answer is the word whose vector points most nearly the same way (leaving out the three words given).`, { phase: 'shift' });
      F.count('cmp', V - 3);
      const won = top.label === A.want, rank = an.findIndex((x) => x.label === A.want);
      F.push([...m.items, arrow('wm:ab', pa, pb, 'p11an'), arrow('wm:ct', pc, pt, 'p11an2'), target, ...legend, H().text('wm:at', 530, 250, 'nearest to ' + c + ' + ' + b + ' − ' + a, 'strong', 13), ...K().bars({ prefix: 'wm:an', x: 530, y: 262, w: 260, rows: an.map((x) => ({ label: x.label, v: Math.max(0, x.score), value: H().fx(x.score, 2), cls: x.label === A.want ? 'good' : '' })), max: 1, labelW: 74, rowH: 24 })],
        `The nearest words to ${c} + ${b} − ${a} are ${H().list(an.map((x) => x.label + ' (' + q(x.score, 2) + ')'))}.` + (won ? ` “${A.want}” comes first: the step from “${a}” to “${b}” carried the relation.` : rank >= 0 ? ` “${A.want}” comes ${['first', 'second', 'third'][rank]}: the relation is there, but this small corpus blurs it with others.` : ` “${A.want}” is not among the top three: with this little text the relation was not learned cleanly.`), { phase: 'answer', answers: an.map((x) => ({ label: x.label, score: x.score })), want: A.want, won, rank });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Neighbours and analogies', text: ['cos(x, y) = x·y / (|x| |y|)', 'neighbours: the words of highest cosine', 'a : b = c : ?', '  answer = the word nearest to', '  v(c) + v(b) − v(a)', '(the three given words left out)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'arrow', once: true, pick: 'option',
      ask: (fr, i, inp) => { const A = ANALOGIES[inp.analogy].abc; return 'Which word will be nearest to ' + A[2] + ' + ' + A[1] + ' − ' + A[0] + '?'; },
      options: (fr, i, inp) => { const ans = fr.find((f) => f.meta.phase === 'answer').meta.answers.map((x) => x.label); return [...new Set([ANALOGIES[inp.analogy].want].concat(ans))].slice(0, 4); },
      answer: (fr) => fr.find((f) => f.meta.phase === 'answer').meta.answers[0].label,
      explain: (fr) => fr.find((f) => f.meta.phase === 'answer').say,
    }],
    quiz: [
      { q: 'Why are neighbours found by cosine rather than by position on the 2-D map?', options: ['the map throws away most of the vector', 'cosine is faster to draw', 'the map is random', 'positions are not numbers'], answer: 0, why: 'The two map axes keep only part of the spread.' },
      { q: 'For “a is to b as c is to ?” the vector searched for is…', options: ['c + b − a', 'a + b + c', 'b − c + a', 'a − b − c'], answer: 0, why: 'The step b − a applied to c.' },
      { q: 'Why are a, b and c left out of the answers?', options: ['c + b − a is usually closest to c itself', 'they are not in the vocabulary', 'cosine cannot compare them', 'to save time'], answer: 0, why: 'The step is small, so the sum stays near c.' },
      { q: 'A word’s nearest neighbour is of a different kind. The likeliest reason is…', options: ['the two words often appear in the same places in the text', 'the vectors are broken', 'the map is upside down', 'they have the same number of letters'], answer: 0, why: 'word2vec only knows the company words keep.' },
      { q: 'With a larger, more varied corpus, analogies usually…', options: ['work more reliably', 'stop working', 'become exact', 'need no vectors'], answer: 0, why: 'More evidence separates the relations more cleanly.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
