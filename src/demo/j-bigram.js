// A tiny character-level language model: a table of logits, one row per character, whose softmax gives the next
// character. Trained by gradient descent on our short story of a lighthouse keeper, sampled with the same seed after
// each stretch of training, so the samples can be compared.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, HM = () => S.Kit.heat, box = D.box;
  const SHOWN = [' ', 'a', 'd', 'e', 'h', 'l', 'n', 'o', 'r', 's', 't'];
  const KEEP = [0, 10, 40, 150, 600];
  D.add({
    id: 'bigram-lm', order: 100, tag: 'Language model', short: 'Tiny LM samples', title: 'A tiny language model: samples improving as it trains',
    blurb: 'The model knows only the previous character: its table holds, for every character, a score for each one that might follow. The heat map is the softmax of a few rows, the chance of each next character; samples use seed 3.',
    frames({ narrow }) {
      const T = S.Toy, m = D.once('bigram', () => T.bigram(T.corpus, { keep: KEEP, lr: 40 }));
      const ix = SHOWN.map((c) => m.chars.indexOf(c)), nS = narrow ? 40 : 60;
      const L = narrow
        ? { view: box(-4, -12, 396, 900), hm: { x: 34, y: 40, cw: 28, ch: 22 }, lc: { x: 50, y: 370, w: 310, h: 120 }, sm: { x: 4, y: 580, w: 386 } }
        : { view: box(-10, -12, 1180, 470), hm: { x: 40, y: 52, cw: 32, ch: 27 }, lc: { x: 520, y: 40, w: 620, h: 150 }, sm: { x: 470, y: 270, w: 690 } };
      const lax = C().axes(Object.assign({ prefix: 'bl', xr: [0, 600], yr: [1.5, 3.5], xticks: [0, 150, 300, 450, 600], yticks: [1.5, 2, 2.5, 3, 3.5], xname: 'training step', yname: 'mean cross-entropy (nats per character)' }, L.lc));
      const curve = D.once('bigramCurve', () => T.bigram(T.corpus, { keep: Array.from({ length: 121 }, (_, i) => i * 5), lr: 40 }).snaps.map((s) => [s.step, s.loss]));
      const show = (c) => (c === ' ' ? '·' : c);
      return m.snaps.map((sn, k) => {
        const P = m.probs(sn.L), items = [];
        const vals = ix.map((a) => ix.map((b) => P[a][b]));
        items.push(...HM().grid({ prefix: 'bh', x: L.hm.x, y: L.hm.y + 20, cw: L.hm.cw, ch: L.hm.ch, vals, scheme: 'seq', max: 0.6, fmt: null, rows: SHOWN.map(show), cols: SHOWN.map(show), labelSize: 13, gap: 2 }).items);
        items.push({ key: 'bh:t', kind: 'text', x: L.hm.x - 24, y: L.hm.y - 14, text: 'chance of the next character (columns) after each (rows)', size: 12, cls: 'axname' });
        items.push(...lax.items, C().line(lax, 'bl:c', curve.filter((c) => c[0] <= sn.step), 'act'), C().dot(lax, 'bl:d', sn.step, sn.loss, ''));
        items.push({ key: 'bl:v', kind: 'text', x: L.lc.x + L.lc.w, y: L.lc.y + 18, text: `step ${sn.step}: ${sn.loss.toFixed(3)}`, anchor: 'end', size: 13, cls: 'strong' });
        items.push({ key: 'bs:t', kind: 'text', x: L.sm.x, y: L.sm.y - 12, text: 'samples of ' + nS + ' characters, the same seed each time', size: 13, cls: 'strong' });
        m.snaps.slice(0, k + 1).forEach((s2, j) => {
          const y = L.sm.y + j * (narrow ? 50 : 34);
          if (narrow) items.push({ key: 'bs:n' + j, kind: 'text', x: L.sm.x, y: y + 12, text: 'step ' + s2.step, size: 12, cls: 'axname' });
          items.push({ key: 'bs:s' + j, kind: 'cell', x: narrow ? L.sm.x : L.sm.x + 90, y: narrow ? y + 18 : y, w: narrow ? L.sm.w : L.sm.w - 90, h: 26, rx: 4, text: show(' ') + m.sample(s2.L, nS, 3, ' ').replace(/ /g, '·'), size: narrow ? 12.5 : 14, align: 'left', cls: 'sample ' + (j === k ? 'act' : '') });
          if (!narrow) items.push({ key: 'bs:n' + j, kind: 'text', x: L.sm.x + 80, y: y + 18, text: 'step ' + s2.step, anchor: 'end', size: 12, cls: 'axname' });
        });
        const words = new Set(T.corpus.split(/[^a-z]+/).filter((w) => w.length >= 3));
        const real = [...new Set(m.sample(sn.L, nS, 3, ' ').split(/[^a-z]+/))].filter((w) => words.has(w)).slice(0, 3);
        const after = (a, b) => P[m.chars.indexOf(a)][m.chars.indexOf(b)];
        const say = k === 0
          ? `Untrained, every score is \`0\`, so all \`${m.chars.length}\` characters are equally likely after any other: the sample is noise, and the loss is \`ln ${m.chars.length} = ${sn.loss.toFixed(3)}\`.`
          : `After ${sn.step} steps of gradient descent the loss is \`${sn.loss.toFixed(3)}\`. The model has learnt, for example, that after “t” comes “h” with chance \`${after('t', 'h').toFixed(2)}\`, and after “h” an “e” with \`${after('h', 'e').toFixed(2)}\`.` + (k === m.snaps.length - 1 ? ` Seeing only one character back, it makes word-like pieces${real.length ? ' (even “' + real.join('”, “') + '”)' : ''} but never sentences: that needs a longer context, the job of the tiny GPT later.` : '');
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
