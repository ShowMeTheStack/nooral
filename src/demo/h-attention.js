// Scaled dot-product attention on a short sentence of our own, with two heads whose weights are written out so
// each has one job: the features of each word, its query and key, the scores and their softmax as heat maps, the
// weights drawn as arcs from the asking word, and the weighted sum of the values.
(function (S) {
  'use strict';
  const D = S.Demo, HM = () => S.Kit.heat, TK = () => S.Kit.tokens, box = D.box;
  const tr = (M) => M[0].map((_, j) => M.map((row) => row[j]));
  D.add({
    id: 'attention-heads', order: 80, tag: 'Transformer · attention', short: 'Attention heads', title: 'Attention heads: queries, keys, values and the weighted sum',
    blurb: 'Each word asks a question (its query) and offers an answer (its key). How much word i attends to word j is a softmax, over all j, of the dot product of i’s query with j’s key, divided by √3. The word’s output is the attention-weighted sum of every word’s value.',
    frames({ narrow }) {
      const T = S.Toy, words = T.attnSentence, H = T.attnHeads.map((h) => T.attention(words, h)), n = words.length;
      const L = narrow
        ? { view: box(-4, -12, 396, 600), A: { x: 74, y: 40 }, B: { x: 74, y: 40 }, M: { x: 74, y: 40 }, cw: 44, ch: 26, strip: { x: 8, y: 480, w: 380 }, two: false }
        : { view: box(-10, -12, 1180, 520), A: { x: 116, y: 40 }, B: { x: 116, y: 180 }, M: { x: 760, y: 40 }, cw: 46, ch: 25, strip: { x: 116, y: 410, w: 1040 }, two: true };
      const fmt = (v) => (Math.abs(v) < 0.005 ? '0' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(2).replace(/^0\./, '.'));
      const grid = (prefix, M, at, rows, title, scheme, max, opts) => HM().grid(Object.assign({ prefix, x: at.x, y: at.y + 18, cw: L.cw, ch: L.ch, vals: M, scheme, max, fmt, rows, cols: words, size: narrow ? 12 : 11.5, labelSize: 12, title, titleCls: 'strong', titleSize: 13, gap: 2 }, opts || {})).items;
      // the sentence with arcs from one word to every word it attends to
      const arcs = (A, qi, cls) => {
        if (!A) A = words.map(() => words.map(() => null));
        const st = TK().strip({ prefix: 'as', x: L.strip.x, y: L.strip.y, width: L.strip.w, tokens: words.map((w) => w), size: narrow ? 13 : 14, gap: narrow ? 3 : 8, sub: (i) => (A[qi][i] === null ? '' : fmt(A[qi][i])), cls: (i) => (i === qi && A[qi][0] !== null ? 'new' : '') });
        // centre the strip
        const used = st.boxes[n - 1].x + st.boxes[n - 1].w - L.strip.x, dx = (L.strip.w - used) / 2;
        const items = st.items.map((it) => Object.assign({}, it, { x: Math.round((it.x + dx) * 100) / 100 }));
        const cx = (i) => st.boxes[i].x + dx + st.boxes[i].w / 2, top = L.strip.y - 2;
        A[qi].forEach((a, j) => {
          if (j === qi || a === null || a < 0.02) return;
          const x1 = cx(qi) + (j > qi ? 6 : -6), x2 = cx(j), hgt = 18 + Math.abs(j - qi) * (narrow ? 12 : 16);
          items.push({ key: 'as:a' + j, kind: 'path', d: `M${x1},${top} C${x1},${top - hgt} ${x2},${top - hgt} ${x2},${top}`, cls: 'arc ' + (cls || ''), paint: { sw: Math.round((1 + 9 * a) * 10) / 10 }, pathAudit: 'edge-x', layer: 'lines' });
        });
        if (A[qi][0] !== null) items.push({ key: 'as:t', kind: 'text', x: L.strip.x + L.strip.w / 2, y: L.strip.y + (narrow ? 72 : 80), text: `how much “${words[qi]}” attends to each word`, anchor: 'middle', size: 12, cls: 'axname' });
        return items;
      };
      const frames = [], h0 = H[0], h1 = H[1];
      const X = tr(h0.X);
      frames.push({ view: L.view, items: [...grid('ax', X, L.A, T.attnFeatures, 'each word’s features', 'div', 1.2), ...arcs(null, 1)],
        say: 'Each word is a column of six numbers: is it a thing, a describing word, an action, a word that points back; and its place in the sentence as an angle (cos and sin of half a radian per word).' });
      const qk = (h, hi) => [...grid('aq', tr(h.Q), L.two ? L.A : L.A, ['q₁', 'q₂', 'q₃'], 'queries Q = X·Wq', 'div', 3, { cls: (i, j) => (j === hi ? 'act' : '') }), ...grid('ak', tr(h.K), L.two ? L.B : { x: L.A.x, y: L.A.y + 140 }, ['k₁', 'k₂', 'k₃'], 'keys K = X·Wk', 'div', 3)];
      frames.push({ view: L.view, items: [...qk(h0, 1), ...arcs(null, 1)],
        say: `Head 1 asks “which thing does a describing word describe?”. Its query matrix gives a describing word a \`3\` in its first number (“I want a thing”) and turns its position one word forward; the keys give every thing a \`1\` there and keep the position as it is. So “small” has query \`(${h0.Q[1].map(D.fix).join(', ')})\`.` });
      const scores = (h, hi, title) => grid('am', h.scores, L.two ? L.M : L.A, words, title || 'scores q·k / √3', 'div', 4, { cls: (i) => (i === hi ? 'act' : '') });
      frames.push({ view: L.view, items: [...(L.two ? qk(h0, 1) : []), ...scores(h0, 1), ...arcs(null, 1)],
        say: `Every query meets every key: the score is their dot product divided by \`√3\`. In the row of “small” the score for “robot” is \`${D.fix(h0.scores[1][2])}\`, the highest: it is a thing, and it is the next word.` });
      const att = (h, hi) => grid('am', h.A, L.two ? L.M : L.A, words, 'attention: softmax of each row', 'seq', 1, { cls: (i) => (i === hi ? 'act' : '') });
      frames.push({ view: L.view, items: [...(L.two ? qk(h0, 1) : []), ...att(h0, 1), ...arcs(h0.A, 1, '')],
        say: `A softmax turns each row of scores into weights that add up to \`1\`. “small” gives \`${D.fix(h0.A[1][2])}\` of its attention to “robot”, and “red” gives \`${D.fix(h0.A[5][6])}\` to “cup”. The arcs below are the row of “small”, drawn as thick as their weights.` });
      const outT = tr(h0.out);
      frames.push({ view: L.view, items: [...grid('av', tr(h0.V), L.two ? L.A : L.A, T.attnValueNames, 'values V = X·Wv', 'seq', 1), ...grid('ao', outT, L.two ? L.B : { x: L.A.x, y: L.A.y + 140 }, T.attnValueNames, 'output: attention × V', 'seq', 1, { cls: (i, j) => (j === 1 ? 'act' : '') }), ...(L.two ? att(h0, 1) : []), ...arcs(h0.A, 1, '')],
        say: `The output for “small” is the weighted sum of the value columns: \`${h0.A[1].map((a) => a.toFixed(2)).join(' · ')}\` times each word’s value. Its “thing” number is \`${D.fix(h0.out[1][0])}\`: the describing word has gathered what it describes.` });
      frames.push({ view: L.view, items: [...(L.two ? qk(h1, 4) : []), ...att(h1, 4), ...arcs(h1.A, 4, 'alt')],
        say: `Head 2 asks a different question with different matrices: a word that points back (“its”) wants the thing two words before it. Its row gives \`${D.fix(h1.A[4][2])}\` to “robot” and only \`${D.fix(h1.A[4][6])}\` to “cup”. A transformer runs such heads side by side and joins their outputs.` });
      return frames;
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
