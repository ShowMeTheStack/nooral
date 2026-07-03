// Byte-pair encoding learning its merges on a short text of our own: count every pair of neighbouring tokens, merge
// the commonest everywhere it occurs, add it to the vocabulary, and count again. The text needs fewer tokens at
// every step.
(function (S) {
  'use strict';
  const D = S.Demo, TK = () => S.Kit.tokens, box = D.box;
  const TEXT = 'the slow snow fell low. the lowest glow grew slower. slow flows glow below the hollow.';
  const show = (t) => S.Kit.tokens.show(t);
  D.add({
    id: 'bpe-merges', order: 90, tag: 'Tokenisation · BPE', short: 'BPE merges', title: 'Byte-pair encoding, merge by merge',
    blurb: 'Start from single characters (a space is shown as ·). Each step merges the most frequent pair of neighbours into one new token, never across the end of a word. Language models read text as such tokens.',
    frames({ narrow }) {
      const T = S.Toy, run = D.once('bpe', () => T.bpe(TEXT, 10)), steps = run.steps;
      const L = narrow
        ? { view: box(-4, -12, 396, 690), strip: { x: 6, y: 34, w: 384 }, size: 13, top: { x: 6, y: 300, w: 300 }, voc: { x: 6, y: 490 } }
        : { view: box(-10, -12, 1180, 470), strip: { x: 10, y: 34, w: 720 }, size: 14, top: { x: 10, y: 250, w: 330 }, voc: { x: 800, y: 34 } };
      const frames = [];
      steps.forEach((st, k) => {
        const items = [];
        const next = steps[k + 1];
        // tokens: the ones just made are lit; the pair about to be merged is marked
        const isPair = new Set();
        if (next) for (let i = 1; i < st.tokens.length; i++) if (st.tokens[i - 1] === next.pair[0] && st.tokens[i] === next.pair[1] && !isPair.has(i - 1)) { isPair.add(i - 1); isPair.add(i); }
        const strip = TK().strip({ prefix: 'bp', x: L.strip.x, y: L.strip.y, width: L.strip.w, tokens: st.tokens, size: L.size, gap: 3, lineGap: 8, cls: (i) => (st.merged && st.tokens[i] === st.merged ? 'new' : isPair.has(i) ? 'pair' : '') });
        items.push(...strip.items);
        items.push({ key: 'bp:t', kind: 'text', x: L.strip.x, y: L.strip.y - 12, text: `the text as ${st.tokens.length} tokens`, size: 13, cls: 'strong' });
        // the commonest pairs now
        const pairs = [...T.bpePairs(st.tokens)].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 5), pmax = pairs.length ? pairs[0][1] : 1;
        items.push({ key: 'bt:t', kind: 'text', x: L.top.x, y: L.top.y - 14, text: 'the commonest pairs now', size: 13, cls: 'strong' });
        pairs.forEach(([pk, c], i) => {
          const [a, b] = pk.split('\u0000'), y = L.top.y + i * 26, win = next && a === next.pair[0] && b === next.pair[1];
          items.push({ key: 'bt:n' + i, kind: 'text', x: L.top.x + 110, y: y + 14, text: show(a) + ' + ' + show(b), anchor: 'end', size: 13, cls: 'mono ' + (win ? 'hot' : '') });
          items.push({ key: 'bt:b' + i, kind: 'cell', x: L.top.x + 120, y: y + 2, w: Math.max(2, ((L.top.w - 160) * c) / pmax), h: 16, rx: 3, text: '', cls: 'bar ' + (win ? '' : 'soft'), audit: 'ignore' });
          items.push({ key: 'bt:c' + i, kind: 'text', x: L.top.x + 128 + ((L.top.w - 160) * c) / pmax, y: y + 14, text: '×' + c, size: 12, cls: 'tick' });
        });
        // the merges so far: the vocabulary beyond the characters
        items.push({ key: 'bv:t', kind: 'text', x: L.voc.x, y: L.voc.y - 12, text: `vocabulary: ${run.steps[0].vocab.length} characters + ${k} merge${k === 1 ? '' : 's'}`, size: 13, cls: 'strong' });
        for (let m = 1; m <= k; m++) {
          const s2 = steps[m], y = L.voc.y + (m - 1) * 26 + (narrow ? 0 : 6), col = narrow && m > 5 ? 1 : 0, yy = narrow ? L.voc.y + ((m - 1) % 5) * 26 : y;
          items.push({ key: 'bv:' + m, kind: 'cell', x: L.voc.x + col * 196, y: yy, w: narrow ? 186 : 300, h: 22, rx: 4, text: `${m}. ${show(s2.pair[0])} + ${show(s2.pair[1])} → ${show(s2.merged)}  ×${s2.count}`, size: 13, align: 'left', cls: 'sample mono ' + (m === k ? 'act' : '') });
        }
        let say;
        if (k === 0) say = `The text starts as \`${st.tokens.length}\` single characters from a vocabulary of \`${st.vocab.length}\`. The commonest neighbouring pair is \`${show(next.pair[0])} ${show(next.pair[1])}\`, ${next.count} times (marked).`;
        else say = `Merge ${k}: \`${show(st.pair[0])}\` + \`${show(st.pair[1])}\` → \`${show(st.merged)}\`, found ${st.count} times. The text is now \`${st.tokens.length}\` tokens.` + (next ? ` Next comes \`${show(next.pair[0])}\` + \`${show(next.pair[1])}\` (${next.count} times).` : ` After ${k} merges common pieces such as \`${show(steps[2].merged)}\` and \`${show(steps[5].merged)}\` are single tokens: a model sees fewer, more meaningful pieces.`);
        frames.push({ view: L.view, items, say });
      });
      return frames;
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
