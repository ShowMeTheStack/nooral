// Part 9, dropout: at every training step each hidden neuron is switched off with probability p and the survivors are
// scaled up by 1 / (1 − p), so no neuron can rely on any particular partner. The masks come from the engine's own
// dropout op; then the overfitting experiment is trained with and without it and the two compared.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml;
  const SHOW = [2, 10, 10, 2];

  S.topic({
    id: 'p09-dropout', family: 'p09-reg', order: 20, title: 'Dropout: training with neurons switched off',
    blurb: 'Dropout switches off a random share p of the hidden neurons at every training step, a different share each time, and scales the rest up so the total stays the same on average. A neuron can no longer lean on one particular partner to fix its mistakes, so the network learns sturdier, simpler features. At test time every neuron is used.',
    inputs: [
      { id: 'p', kind: 'choice', label: 'Dropout rate p', def: '0.5', options: ['0.2', '0.5', '0.7'].map((v) => ({ value: v, label: v })) },
      { id: 'seed', kind: 'number', label: 'Seed for the masks', def: 3, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Half switched off', values: { p: '0.5', seed: 3 } },
      { label: 'A light touch, p = 0.2', values: { p: '0.2', seed: 3 } },
      { label: 'Heavy, p = 0.7', values: { p: '0.7', seed: 8 } },
    ],
    counters: [{ id: 'mask', label: 'masks drawn' }],
    run(inp) {
      const T = S.T, F = H().frames(['mask']), q = F.q, p = +inp.p, scale = 1 / (1 - p), g = T.rng(inp.seed);
      const L = S.Kit.net.layout({ x: 60, y: 60, w: 420, h: 330, sizes: SHOW, r: 15 });
      const masks = [0, 1, 2].map(() => [1, 2].map(() => Array.from(T.dropout(T.ones([1, SHOW[1]]), p, g, true).values())));
      const net = (mask, title) => {
        const items = [];
        const on = (l, i) => !mask || l === 0 || l === SHOW.length - 1 || mask[l - 1][i] > 0;
        for (let l = 0; l + 1 < SHOW.length; l++) for (let i = 0; i < SHOW[l + 1]; i++) for (let j = 0; j < SHOW[l]; j++) {
          if (!on(l, j) || !on(l + 1, i)) continue;
          const a = L.pos[l][j], b = L.pos[l + 1][i], dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
          items.push({ key: `do:w${l}_${i}_${j}`, kind: 'path', d: `M${P.r2(a.x + ux * 16)},${P.r2(a.y + uy * 16)} L${P.r2(b.x - ux * 16)},${P.r2(b.y - uy * 16)}`, cls: 'weight', pathAudit: 'edge-x', paint: { stroke: 'var(--node-line)', sw: 0.9 }, layer: 'lines' });
        }
        L.pos.forEach((col, l) => col.forEach((pp, i) => {
          const off = !on(l, i);
          items.push({ key: `do:n${l}_${i}`, kind: 'state', x: pp.x, y: pp.y, r: 15, text: '', cls: 'neuron ' + (l === 0 ? 'input ' : '') + (off ? 'p09drop' : '') });
          if (off) items.push({ key: `do:x${l}_${i}`, kind: 'path', d: `M${pp.x - 8},${pp.y - 8} L${pp.x + 8},${pp.y + 8} M${pp.x + 8},${pp.y - 8} L${pp.x - 8},${pp.y + 8}`, cls: 'p09cross', audit: 'ignore', layer: 'top' });
        }));
        items.push(H().text('do:t', 60, 30, title, 'strong', 15));
        return items;
      };
      const side = (rows) => H().lines('do:s', 560, 120, rows, 15, 28).map((it) => Object.assign(it, { mono: it.cls === 'math' }));
      F.push([...net(null, 'The network as it is'), ...side([['p = ' + H().fx(p, 1), 'math'], ['survivors × 1 / (1 − p)', 'math'], ['  = × ' + H().fx(scale, 2), 'math']])],
        `Dropout with p = ${q(p, 1)}: at every training step each hidden neuron is switched off with probability ${q(p, 1)}, and every neuron that stays on has its output multiplied by 1 / (1 − p) = ${q(scale, 2)}, so a layer’s total output is unchanged on average. (Drawn here: ${q.n(SHOW[1])} neurons per hidden layer.)`, { phase: 'start', p, scale, width: SHOW[1] }, { hl: { eq: [0, 1] } });
      masks.forEach((m, k) => {
        F.count('mask');
        const off = m.map((row) => row.filter((v) => v === 0).length);
        F.push([...net(m, 'Training step ' + (k + 1) + ': a fresh random mask'), ...side([['switched off:', 'legend'], ['layer 1: ' + off[0] + ' of ' + SHOW[1], 'math'], ['layer 2: ' + off[1] + ' of ' + SHOW[2], 'math'], ['the rest × ' + H().fx(scale, 2), 'math']])],
          `Training step ${q.n(k + 1)}: the engine’s dropout switches off ${P.qpl(q, off[0], 'neuron')} of the first hidden layer and ${q.n(off[1])} of the second (crossed out); their wires carry nothing this step, and only the survivors’ weights are updated.` + (k === 2 ? ' Each step trains a different thinned network; the full network is in effect an average of all of them.' : ''), { phase: 'mask', k, step: k + 1, off, mask: m }, { hl: { eq: [0] } });
      });
      F.push([...net(null, 'At test time: every neuron on'), ...side([['no mask, no scaling', 'math']])],
        `At test time nothing is switched off and nothing is scaled: the scaling during training already made each neuron’s expected output match this.`, { phase: 'test' }, { hl: { eq: [2] } });
      // the experiment with and without dropout
      const a = P.regRun({}), b = P.regRun({ dropout: p }), la = a.evals[a.evals.length - 1], lb = b.evals[b.evals.length - 1], d = a.data;
      const pa = { x: 40, y: 50, w: 300, h: 200 }, pb = { x: 400, y: 50, w: 300, h: 200 };
      const cur = P.regCurves('do:c', { x: 40, y: 320, w: 660, h: 150 }, a.evals, P.REG.steps, { extra: [['d', b.evals, 'act']], legendExtra: [{ text: 'fresh, with dropout', cls: 'act' }] });
      F.push([...P.regPanel('do:a', pa, a.snaps[a.snaps.length - 1].field, d, { title: 'without dropout' }), ...P.regPanel('do:b', pb, b.snaps[b.snaps.length - 1].field, d, { title: 'with dropout, p = ' + H().fx(p, 1) }), ...cur.items],
        `The overfitting experiment again (${q.n(d.train.X.length)} training points, hidden layers of ${H().list(P.REG.hidden.map((h) => q.n(h)))}), trained for ${q.n(la.step)} steps without and with dropout. Without: training loss ${q.sg(la.trainLoss, 2)}, loss on fresh points ${q(la.valLoss, 3)}, ${q.pct(la.valAcc)} right. With dropout: training loss ${q.sg(lb.trainLoss, 2)}, fresh loss ${q(lb.valLoss, 3)}, ${q.pct(lb.valAcc)} right.` + (lb.valLoss < la.valLoss ? ' The boundary is smoother and the gap between the curves is smaller.' : ' Here dropout did not lower the fresh loss: with so few points it is a weak cure on its own.'),
        { phase: 'compare', without: { trainLoss: la.trainLoss, valLoss: la.valLoss, valAcc: la.valAcc }, with: { trainLoss: lb.trainLoss, valLoss: lb.valLoss, valAcc: lb.valAcc }, n: d.train.X.length, hid: P.REG.hidden, steps: la.step }, { hl: { eq: [3] } });
      P.narrow(F.list, (m) => (m.phase === 'compare' ? { x: 380, y: 20, w: 340, h: 260 } : { x: 20, y: 0, w: 520, h: 420 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Inverted dropout', text: ['training: mᵢ = 0 with probability p, else 1 / (1 − p)', 'hᵢ ← mᵢ · hᵢ  (a new mask every step)', 'test: no mask, no scaling', 'E[mᵢ] = 1, so the expected output is unchanged'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'test', once: true, pick: 'option',
      ask: 'With dropout, will the loss on fresh points after training be lower or higher than without?',
      options: ['lower', 'higher'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'compare').meta; return m.with.valLoss < m.without.valLoss ? 'lower' : 'higher'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'compare').say,
    }],
    quiz: [
      { q: 'With p = 0.5, a neuron that stays on during training has its output…', options: ['doubled', 'halved', 'unchanged', 'set to 0.5'], answer: 0, why: 'Inverted dropout scales survivors by 1 / (1 − p) = 2.' },
      { q: 'At test time, dropout…', options: ['is switched off: every neuron is used, unscaled', 'drops the same neurons as the last training step', 'drops neurons with probability p', 'doubles every output'], answer: 0, why: 'The training-time scaling already matched the expected outputs.' },
      { q: 'Why does dropout reduce overfitting?', options: ['no neuron can depend on particular partners, so features must be useful on their own', 'it makes the network bigger', 'it removes training points', 'it raises the learning rate'], answer: 0, why: 'Co-adapted features that only work together are what memorising tends to build.' },
      { q: 'How often is a new dropout mask drawn?', options: ['at every training step', 'once, before training', 'once per epoch for the whole run', 'only at test time'], answer: 0, why: 'A fresh mask each step trains a different thinned network each time.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
