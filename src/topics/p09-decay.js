// Part 9, weight decay: every step each weight is also shrunk towards zero by a fixed fraction, so only weights the
// data keeps pushing on stay large. Shown on one real update of AdamW (decay decoupled from the gradient step), then
// the overfitting experiment trained with and without it: smaller weights, a smoother boundary.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;

  S.topic({
    id: 'p09-weight-decay', family: 'p09-reg', order: 30, title: 'Weight decay: keeping the weights small',
    blurb: 'Weight decay adds one more pull to every update: each weight shrinks by a small fixed fraction of itself, whatever the gradient says. A weight survives only if the data keeps pushing it back up. Small weights make a smooth function, and smooth functions cannot bend round every noisy point.',
    inputs: [{ id: 'lam', kind: 'choice', label: 'Decay λ', def: '2', options: ['0.5', '1', '2', '4'].map((v) => ({ value: v, label: 'λ = ' + v })) }],
    presets: [
      { label: 'λ = 2', values: { lam: '2' } },
      { label: 'Gentle, λ = 0.5', values: { lam: '0.5' } },
      { label: 'Strong, λ = 4', values: { lam: '4' } },
    ],
    counters: [],
    run(inp) {
      const T = S.T, F = H().frames([]), q = F.q, lam = +inp.lam, lr = P.REG.lr;
      // one real update: the network as the experiment builds it, one AdamW step on the 30 points
      const d = P.regData(), net = S.DL.build('mlp', { inputs: 2, hidden: P.REG.hidden, outputs: 2, act: 'relu', seed: 1 }), W = net.layers[0].W;
      const k = (() => { const v = W.values(); let b = 0; for (let i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[b])) b = i; return b; })();
      const before = W.values()[k], opt = T.optim.make('adamw', net.parameters(), { lr, weightDecay: lam });
      opt.zeroGrad(); net.lossOn({ x: T.tensor(d.train.X), y: d.train.y }).loss.backward();
      const grad = W.grad.values()[k];
      opt.step();
      const after = W.values()[k], decay = lr * lam * before, rest = before - decay - after;
      const rule = (rows) => H().lines('wd:r', 60, 80, rows, 16, 32);
      F.push(rule([['AdamW, one step for every weight w:', 'strong'], ['w ← w − η·λ·w − η·(Adam step from the gradient)', 'math'], ['η = ' + H().fx(lr, 2) + ', λ = ' + H().sg(lam, 2) + ': each step keeps 1 − η·λ = ' + H().fx(1 - lr * lam, 2) + ' of w', 'math']]),
        `Weight decay with λ = ${q.sg(lam, 2)} and learning rate η = ${q(lr, 2)}: before its gradient step, every weight is multiplied by 1 − η·λ = ${q(1 - lr * lam, 2)}. Over a hundred steps with no push from the data, that alone shrinks a weight to ${q.sg(Math.pow(1 - lr * lam, 100), 2)} of its size.`, { phase: 'rule', lam, lr, keep: 1 - lr * lam, hundred: Math.pow(1 - lr * lam, 100) }, { hl: { eq: [0] } });
      F.push(rule([['The largest first-layer weight, one step:', 'strong'], ['before: ' + H().fx(before, 4) + ',  gradient ' + H().fx(grad, 4), 'math'], ['decay: η·λ·w = ' + H().fx(decay, 4), 'math'], ['Adam step: ' + H().fx(rest, 4), 'math'], ['after: ' + H().fx(after, 4), 'math']]),
        `One real step on the largest weight of the first layer, ${q(before, 4)}. The decay takes η·λ·w = ${q(decay, 4)}; the Adam step from its gradient (${q(grad, 4)}) takes ${q(rest, 4)}; it ends at ${q(after, 4)}. The decay does not look at the data at all: it pulls every weight in proportion to its size.`, { phase: 'step', before, after, decay, rest, grad }, { hl: { eq: [0, 1] } });
      // the experiment
      const a = P.regRun({}), b = P.regRun({ optimizer: 'adamw', weightDecay: lam });
      const pa = { x: 40, y: 50, w: 300, h: 200 }, pb = { x: 400, y: 50, w: 300, h: 200 };
      const ws = (r) => r.net.layers.flatMap((l) => Array.from(l.W.values()));
      const both = (k2, extra) => [...P.regPanel('wd:a', pa, a.snaps[k2].field, a.data, { title: 'λ = 0' }), ...P.regPanel('wd:b', pb, b.snaps[k2].field, b.data, { title: 'λ = ' + H().sg(lam, 2) }), ...(extra || [])];
      const mid = a.snaps.findIndex((s) => s.step === 120), sa = a.snaps[mid], sb = b.snaps[mid];
      F.push(both(mid), `The overfitting experiment with AdamW, without decay and with λ = ${q.sg(lam, 2)}. At step ${q.n(sa.step)} the weights have total size ‖W‖ = ${q(sa.wnorm, 2)} without and ${q(sb.wnorm, 2)} with decay; the fresh-point loss is ${q(sa.valLoss, 3)} against ${q(sb.valLoss, 3)}.`, { phase: 'mid', step: sa.step, wa: sa.wnorm, wb: sb.wnorm, va: sa.valLoss, vb: sb.valLoss }, { hl: { eq: [2] } });
      // the weights at the end as two histograms on one axis
      const la = a.evals[a.evals.length - 1], lb = b.evals[b.evals.length - 1], A = ws(a), B = ws(b), m = Math.max(...A.map(Math.abs), ...B.map(Math.abs)), nb = 30;
      const counts = (v) => { const c = new Array(nb).fill(0); for (const x of v) c[Math.max(0, Math.min(nb - 1, Math.floor(((x + m) / (2 * m)) * nb)))]++; return c.map((x) => x / v.length); };
      const ca = counts(A), cb = counts(B), edges = Array.from({ length: nb + 1 }, (_, i) => -m + (2 * m * i) / nb), ymax = Math.max(...ca, ...cb) * 1.08;
      const hx = P.ax({ prefix: 'wd:h', x: 40, y: 330, w: 300, h: 130, xr: [-m, m], yr: [0, ymax], xticks: [-m, 0, m].map((v) => +v.toFixed(1)), yticks: 0, grid: false, xname: 'weight' });
      const hist = [...hx.items, ...S.Kit.dt.hist(hx, 'wd:ha', ca, edges, 'p09ha'), ...S.Kit.dt.hist(hx, 'wd:hb', cb, edges, 'p09hb'), ...C().legend('wd:hl', 400, 360, [{ text: 'weights, λ = 0', cls: 'c0', dot: true }, { text: 'weights, λ = ' + H().sg(lam, 2), cls: 'c1', dot: true }], { dir: 'col', size: 14 })];
      hist.push(H().text('wd:n', 400, 430, '‖W‖: ' + H().fx(la.wnorm, 1) + ' against ' + H().fx(lb.wnorm, 1), 'math', 14));
      const sd = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
      F.push(both(a.snaps.length - 1, hist), `After ${q.n(la.step)} steps. Without decay the weights have spread out (total size ${q(la.wnorm, 2)}, typical weight ${q(sd(A), 3)}); with λ = ${q.sg(lam, 2)} they are held near zero (total ${q(lb.wnorm, 2)}, typical ${q(sd(B), 3)}). Loss on fresh points: ${q(la.valLoss, 3)} without, ${q(lb.valLoss, 3)} with; ${q.pct(la.valAcc)} against ${q.pct(lb.valAcc)} right.` + (lb.valLoss < la.valLoss ? ' The smaller weights give the smoother boundary on the right.' : ''),
        { phase: 'end', wa: la.wnorm, wb: lb.wnorm, sa: sd(A), sb: sd(B), va: la.valLoss, vb: lb.valLoss, aa: la.valAcc, ab: lb.valAcc, steps: la.step }, { hl: { eq: [2] } });
      const lg = (v) => Math.log10(Math.max(1e-4, v));
      const cx = P.ax({ prefix: 'wd:c', x: 40, y: 330, w: 300, h: 130, xr: [0, P.REG.steps], yr: [-1.5, 1], xticks: [0, 250, 500], yticks: [-1, 0, 1], fmtY: (v) => '10' + ['⁻¹', '⁰', '¹'][v + 1], xname: 'step', yname: 'fresh-point loss' });
      const nx = P.ax({ prefix: 'wd:w', x: 420, y: 330, w: 280, h: 130, xr: [0, P.REG.steps], yr: [0, Math.max(...a.evals.map((e) => e.wnorm)) * 1.1], xticks: [0, 250, 500], yticks: 3, xname: 'step', yname: 'total weight size ‖W‖' });
      const curves = [...cx.items, C().line(cx, 'wd:ca', a.evals.map((e) => [e.step, lg(e.valLoss)]), 'p09train'), C().line(cx, 'wd:cb', b.evals.map((e) => [e.step, lg(e.valLoss)]), 'p09val'), ...nx.items, C().line(nx, 'wd:na', a.evals.map((e) => [e.step, e.wnorm]), 'p09train'), C().line(nx, 'wd:nb', b.evals.map((e) => [e.step, e.wnorm]), 'p09val')];
      curves.push(...C().legend('wd:cl', 40, 535, [{ text: 'λ = 0', cls: 'p09train' }, { text: 'λ = ' + H().sg(lam, 2), cls: 'p09val' }], { size: 14 }));
      const peak = Math.max(...b.evals.map((e) => e.wnorm));
      F.push(both(a.snaps.length - 1, curves), `Over the whole run: without decay the total weight size keeps growing, to ${q(la.wnorm, 2)}, as the network sharpens its boundary round each point. With decay it levels off (its highest was ${q(peak, 2)}): there the pull of the data and the pull towards zero balance.`, { phase: 'curves', wa: la.wnorm, peak }, { hl: { eq: [3] } });
      P.narrow(F.list, (m) => (m.phase === 'rule' || m.phase === 'step' ? { x: 40, y: 40, w: 560, h: 220 } : m.phase === 'curves' ? { x: 400, y: 310, w: 320, h: 250 } : m.phase === 'end' ? { x: 0, y: 310, w: 720, h: 200 } : { x: 380, y: 20, w: 340, h: 260 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Weight decay', text: ['AdamW: w ← w − η·λ·w − η·adam(g)', 'SGD + L2 penalty ½λ‖w‖²: w ← w − η(g + λw)', 'small weights ⇒ a smooth function', 'balance: the data’s pull = the decay’s pull'].join('\n') }] }, result: { k } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'step', once: true, pick: 'option',
      ask: 'After training, will the loss on fresh points be lower with weight decay than without?',
      options: ['lower with decay', 'higher with decay'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'end').meta; return m.vb < m.va ? 'lower with decay' : 'higher with decay'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'end').say,
    }],
    quiz: [
      { q: 'With SGD, an L2 penalty ½λ‖w‖² adds what to each weight’s gradient?', options: ['λ·w', 'λ', 'λ·w²', 'nothing'], answer: 0, why: 'The derivative of ½λw² is λw.' },
      { q: 'With η = 0.02 and λ = 2, decay alone keeps what share of a weight each step?', options: ['0.96', '0.98', '0.6', '2'], answer: 0, why: '1 − η·λ = 1 − 0.04.' },
      { q: 'Why does weight decay fight overfitting?', options: ['small weights make smooth functions that cannot bend round single points', 'it removes neurons', 'it adds training data', 'it lowers the training loss'], answer: 0, why: 'Sharp turns in the boundary need large weights.' },
      { q: 'Too much weight decay leads to…', options: ['underfitting: the network cannot even fit the training data well', 'more overfitting', 'exploding weights', 'no change at all'], answer: 0, why: 'The pull towards zero overwhelms the pull of the data.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
