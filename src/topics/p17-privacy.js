// Part 17, privacy: one step of DP-SGD drawn (each applicant's gradient clipped to length C, Gaussian noise of size
// σC added to the sum) and the approximate privacy–accuracy trade-off; then membership inference as a concept: an
// overfitted model is surer of its own training rows, and regularisation or private training shrinks the tell.
(function (S) {
  'use strict';
  const P = S.P17, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const SIG = [0, 0.5, 1, 2, 4, 8];

  S.topic({
    id: 'dp-sgd', family: 'p17-privacy', order: 10, title: 'Differential privacy in training: clip, then add noise',
    blurb: 'A trained model should not reveal whether any one person’s record was in its training data. DP-SGD gets there in two moves at every step: each person’s own gradient is clipped to length at most C, so no one can pull the model far, and Gaussian noise of standard deviation σC is added to the sum, so no one’s presence can be told apart. More noise means stronger privacy (a smaller ε) and, usually, a less accurate model. The ε shown is an approximate, teaching estimate, not a certified bound.',
    inputs: [
      { id: 'sigma', kind: 'slider', label: 'Noise multiplier σ', def: 1, min: 0, max: 8, step: 0.5, fmt: (v) => v.toFixed(1) },
      { id: 'clip', kind: 'slider', label: 'Clip length C', def: 1, min: 0.5, max: 3, step: 0.25, fmt: (v) => v.toFixed(2) },
      { id: 'seed', kind: 'number', label: 'Seed (batches and noise)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'σ 1, C 1', values: { sigma: 1, clip: 1, seed: 1 } },
      { label: 'No noise (no privacy)', values: { sigma: 0, clip: 1, seed: 1 } },
      { label: 'Heavy noise, σ 8', values: { sigma: 8, clip: 1, seed: 1 } },
      { label: 'Loose clipping, C 3', values: { sigma: 1, clip: 3, seed: 2 } },
    ],
    counters: [{ id: 'clipped', label: 'gradients clipped' }],
    run(inp) {
      const F = H().frames(['clipped']), q = F.q, L = P.loans(1), r = P.once('dp:' + JSON.stringify(inp), () => S.Rob.dpsgd({ sigma: inp.sigma, clip: inp.clip, seed: inp.seed, data: L }));
      const tr = Object.fromEntries(r.trace.map((t) => [t.kind, t])), norms = tr.grads.norms, B = norms.length, Cc = inp.clip;
      const gmax = Math.max(3.2, ...norms, Cc) * 1.05;
      const ax = C().axes({ prefix: 'dp:ax', x: 40, y: 70, w: 330, h: 190, xr: [0, B], yr: [0, gmax], xticks: 0, yticks: 1, xname: 'applicant in the batch', yname: 'gradient length', tickSize: 12 });
      const bars = (clip) => norms.map((n, k) => {
        const v = clip ? Math.min(n, Cc) : n, x = P.r2(ax.X(k + 0.15)), w = P.r2(ax.X(0.7) - ax.X(0)), top = ax.Y(v), out = [{ key: 'dp:b' + k, kind: 'cell', x, y: top, w, h: P.r2(ax.Y(0) - top), rx: 2, text: '', cls: 'bar ' + (n > Cc ? 'p17clipped' : 'p17ok'), audit: 'ignore' }];
        if (clip && n > Cc) out.push({ key: 'dp:o' + k, kind: 'cell', x, y: ax.Y(n), w, h: P.r2(ax.Y(Cc) - ax.Y(n)), rx: 2, text: '', cls: 'bar p17cut', audit: 'ignore' });
        return out;
      }).flat();
      const cl = [C().hline(ax, 'dp:c', Cc, 'p17clip'), H().text('dp:ct', ax.X(B) + 6, ax.Y(Cc) + 4, 'C', 'act', 13)];
      const keys = [...P.key('dp:k0', 40, 330, 'p17clipped', 'longer than C'), ...P.key('dp:k1', 190, 330, 'p17ok', 'within C')];
      const nClip = tr.clip.clipped;
      F.push([...ax.items, ...bars(false), ...cl, ...keys, H().head('dp:h', 40, 30, 'Step 1: each applicant’s own gradient')],
        `One training step of a logistic model on the loans data draws a batch of ${P.qpl(q, B, 'applicant')}. Each has their own gradient, of length ${q(Math.min(...norms), 2)} to ${q(Math.max(...norms), 2)}. Ordinary training would simply average them, so one unusual person could pull the model a long way.`, { phase: 'grads', B, min: Math.min(...norms), max: Math.max(...norms) });
      F.set('clipped', nClip);
      F.push([...ax.items, ...bars(true), ...cl, ...keys, H().head('dp:h', 40, 30, 'Clip every gradient to length C')],
        `${P.qpl(q, nClip, 'gradient')} ${nClip === 1 ? 'is' : 'are'} longer than C = ${q(Cc, 2)} and ${nClip === 1 ? 'is' : 'are'} scaled down to exactly C (the dashed tops are cut away). Now no single applicant can move the sum by more than C.`, { phase: 'clip', clipped: nClip, C: Cc });
      const sum = tr.noise.sum, noise = tr.noise.noise, tot = sum.map((v, j) => v + noise[j]), d = sum.length;
      const vmax = Math.max(1, ...sum.map(Math.abs), ...noise.map(Math.abs), ...tot.map(Math.abs)) * 1.1;
      const bx = C().axes({ prefix: 'dp:bx', x: 430, y: 70, w: 300, h: 190, xr: [0, d], yr: [-vmax, vmax], xticks: 0, yticks: 0, yname: 'per weight', tickSize: 12 });
      const comp = (vals, j, cls) => vals.map((v, k) => { const x = P.r2(bx.X(k + 0.1 + 0.27 * j)), top = Math.min(bx.Y(v), bx.Y(0)); return { key: 'dp:v' + j + '_' + k, kind: 'cell', x, y: top, w: P.r2(bx.X(0.25) - bx.X(0)), h: P.r2(Math.max(0.5, Math.abs(bx.Y(v) - bx.Y(0)))), rx: 1.5, text: '', cls: 'bar ' + cls, audit: 'ignore' }; });
      const wn = ['income', 'debt', 'years', 'missed', 'amount', 'bias'];
      const bitems = [...bx.items, C().hline(bx, 'dp:z', 0, ''), ...comp(sum, 0, 'p17grey'), ...comp(noise, 1, 'p17clipped'), ...comp(tot, 2, 'p17acc'), ...wn.map((w, k) => H().text('dp:wn' + k, bx.X(k + 0.5), bx.Y(-vmax) + 18, w, 'tick', 11.5, 'middle')),
        ...P.key('dp:l0', 430, 330, 'p17grey', 'sum of clipped'), ...P.key('dp:l1', 560, 330, 'p17clipped', 'noise'), ...P.key('dp:l2', 640, 330, 'p17acc', 'noisy sum')];
      F.push([...ax.items, ...bars(true), ...cl, ...bitems, H().head('dp:h', 40, 30, 'Add noise of standard deviation σC')],
        `The clipped gradients are added (a vector of length ${q(Math.hypot(...sum), 2)}) and Gaussian noise with standard deviation σC = ${q(inp.sigma * Cc, 2)} is added to every weight’s total; this noise has length ${q(Math.hypot(...noise), 2)}. The noise is what hides whether any one applicant was in the batch.`, { phase: 'noise', sumLen: Math.hypot(...sum), noiseLen: Math.hypot(...noise), sd: inp.sigma * Cc });
      F.push([...ax.items, ...bars(true), ...cl, ...bitems, H().head('dp:h', 40, 30, 'Train, many such steps')],
        `The noisy sum divided by the batch size is the step’s gradient (length ${q(Math.hypot(...tr.update.update), 3)}). After ${P.qpl(q, r.steps, 'step')} the model is ${q.pct(r.accuracy)} accurate on the test applicants. ` + (isFinite(r.eps) ? `A moments-style accountant puts the privacy loss at roughly ε ≈ ${q(r.eps, 2)} (δ = ${q.sg(1e-5, 1)}): an approximate, teaching estimate.` : 'With no noise there is no privacy guarantee at all: ε is unbounded.'), { phase: 'train', delta: 1e-5, upd: Math.hypot(...tr.update.update), steps: r.steps, acc: r.accuracy, eps: isFinite(r.eps) ? r.eps : null });
      const T = P.once('dpt:' + Cc, () => S.Rob.dpTradeoff({ clip: Cc, sigmas: SIG, seeds: [1, 2, 3] }));
      const cx = C().axes({ prefix: 'dp:cx', x: 60, y: 80, w: 400, h: 190, xr: [0, 8], yr: [0.4, 0.9], xticks: [0, 1, 2, 4, 8], yticks: 0.1, xname: 'noise multiplier σ', yname: 'test accuracy (mean of three runs)', tickSize: 12 });
      const pts = T.rows.map((row) => [row.sigma, row.accuracy]);
      const tab = S.Kit.dt.table({ prefix: 'dp:tb', x: 520, y: 80, cols: [{ name: 'σ', w: 50 }, { name: 'ε ≈', w: 70 }, { name: 'accuracy', w: 90 }], rows: T.rows.map((row) => [H().fx(row.sigma, 1), isFinite(row.eps) ? H().fx(row.eps, 2) : '∞', H().pct(row.accuracy)]), rowH: 26, size: 13, cls: (i) => (T.rows[i].sigma === inp.sigma ? 'lit' : '') });
      const first = T.rows[0], last = T.rows[T.rows.length - 1];
      F.push([...cx.items, C().line(cx, 'dp:tc', pts, 'p17acc'), ...pts.map((p, k) => C().dot(cx, 'dp:td' + k, p[0], p[1], 'p17acc', 4.5)), { key: 'dp:me', kind: 'dot', x: cx.X(inp.sigma), y: cx.Y(r.accuracy), r: 7, cls: 'ring p17me', audit: 'overlap-ok', layer: 'items', z: 3 }, ...tab.items, H().text('dp:tbt', 520, 62, 'ε is approximate (teaching estimate)', 'axname', 12.5), ...P.key('dp:km', 60, 330, 'p17me', 'this run (one seed)'), H().head('dp:h', 60, 30, 'The privacy–accuracy trade-off')],
        `Repeating the training at each noise level (mean of three runs): with no noise the model reaches ${q.pct(first.accuracy)} but promises nothing; at σ = ${q(last.sigma, 1)} the estimate is ε ≈ ${q(last.eps, 2)}, much stronger privacy, and accuracy falls to ${q.pct(last.accuracy)}. A smaller ε means stronger privacy; these ε values are rough teaching estimates, not certified bounds.`, { phase: 'trade', acc0: first.accuracy, accN: last.accuracy, sigN: last.sigma, epsN: last.eps });
      for (const f of F.list) f.narrowView = { x: 20, y: 0, w: 380, h: 350 };
      F.last().narrowView = { x: 20, y: 20, w: 470, h: 330 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'DP-SGD, one step', text: ['g_i = ∇ loss of applicant i', 'clip: ḡ_i = g_i · min(1, C / ‖g_i‖)', 'noise: s = Σ ḡ_i + N(0, σ²C² I)', 'step: w ← w − η · s / batch', 'ε: from σ, the sampling rate and the steps', '(approximate here; a smaller ε is more private)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'grads', pick: 'option', once: true,
      ask: 'How many of these gradients will be clipped?',
      options: (fr) => { const c = fr.find((f) => f.meta.phase === 'clip').meta.clipped; return [...new Set([Math.max(0, c - 4), c, Math.min(16, c + 3), 0])].sort((a, b) => a - b).map(String); },
      answer: (fr) => String(fr.find((f) => f.meta.phase === 'clip').meta.clipped),
      explain: (fr) => fr.find((f) => f.meta.phase === 'clip').say,
    }],
    quiz: [
      { q: 'Why clip each person’s gradient before adding noise?', options: ['it bounds how much any one person can change the update', 'it speeds up training', 'it removes the need for noise', 'it makes the loss convex'], answer: 0, why: 'Noise can only hide a contribution of known, bounded size.' },
      { q: 'Raising σ (more noise) usually…', options: ['strengthens privacy (smaller ε) and lowers accuracy', 'weakens privacy', 'raises accuracy', 'changes nothing'], answer: 0, why: 'That is the trade-off.' },
      { q: 'A model trained with σ = 0 offers…', options: ['no differential-privacy guarantee', 'perfect privacy', 'ε = 0', 'the strongest guarantee'], answer: 0, why: 'Without noise ε is unbounded.' },
      { q: 'The ε values on this page are…', options: ['approximate teaching estimates', 'certified bounds', 'exact for every attack', 'accuracies'], answer: 0, why: 'They come from a simplified accountant.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const MODELS = { deep: 'a fully grown tree', depth4: 'a tree of depth four', leaf20: 'a tree with at least twenty rows per leaf', logistic: 'logistic regression', dp: 'logistic regression trained with DP-SGD' };
  /** the chance a model gives the true label, for its training rows (members) and for unseen rows */
  P.memberRun = (seed) => P.once('mi:' + seed, () => {
    const L = P.loans(seed), conf = (pr, y) => pr.map((p, i) => (y[i] ? p : 1 - p));
    const out = {};
    for (const k of Object.keys(MODELS)) {
      let pTr, pTe;
      if (k === 'logistic' || k === 'dp') {
        const m = k === 'dp' ? S.Rob.dpsgd({ sigma: 1, data: L, seed: 1 }) : S.Rob.logreg(L.Xtr, L.ytr, {});
        const b = k === 'dp' ? m.b : m.b, w = m.w, pr = (x) => 1 / (1 + Math.exp(-(b + x.reduce((s, v, j) => s + v * w[j], 0))));
        pTr = L.Xtr.map(pr); pTe = L.Xte.map(pr);
      } else {
        const t = new S.ML.DecisionTreeClassifier(k === 'deep' ? {} : k === 'depth4' ? { maxDepth: 4 } : { minSamplesLeaf: 20 }).fit(L.Xtr, L.ytr);
        pTr = t.predictProba(L.Xtr).map((p) => p[1]); pTe = t.predictProba(L.Xte).map((p) => p[1]);
      }
      const acc = (p, y) => p.filter((v, i) => (v >= 0.5 ? 1 : 0) === y[i]).length / y.length;
      out[k] = { mem: conf(pTr, L.ytr), non: conf(pTe, L.yte), accTr: acc(pTr, L.ytr), accTe: acc(pTe, L.yte) };
      Object.assign(out[k], P.bestGuess(out[k].mem, out[k].non));
    }
    return out;
  });
  /** the best threshold guess "a member if the confidence is at least t": its TPR, FPR and advantage TPR − FPR */
  P.bestGuess = (mem, non) => {
    let best = { adv: 0, t: 1.01, tpr: 0, fpr: 0 };
    for (const t of [...new Set(mem.concat(non))].sort((a, b) => a - b)) {
      const tpr = mem.filter((c) => c >= t).length / mem.length, fpr = non.filter((c) => c >= t).length / non.length;
      if (tpr - fpr > best.adv + 1e-12) best = { adv: tpr - fpr, t, tpr, fpr };
    }
    return best;
  };
  S.topic({
    id: 'membership-inference', family: 'p17-privacy', order: 20, title: 'What a model leaks about its training rows',
    blurb: 'A model that fits its training rows too closely is noticeably surer of them than of rows it has never seen. That difference is a leak: from confidence alone, someone could guess whether a person’s record was used in training. Measuring the leak (the best guess’s true-positive rate minus its false-positive rate) and closing it with regularisation or private training is part of releasing a model responsibly.',
    inputs: [
      { id: 'model', kind: 'choice', label: 'Model', def: 'deep', options: Object.keys(MODELS).map((k) => ({ value: k, label: MODELS[k] })) },
      { id: 'seed', kind: 'number', label: 'Seed (the applicants)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A fully grown tree', values: { model: 'deep', seed: 1 } },
      { label: 'Larger leaves', values: { model: 'leaf20', seed: 1 } },
      { label: 'Logistic regression', values: { model: 'logistic', seed: 1 } },
      { label: 'Private training', values: { model: 'dp', seed: 1 } },
    ],
    counters: [{ id: 'models', label: 'models' }],
    run(inp) {
      const F = H().frames(['models']), q = F.q, all = P.memberRun(inp.seed), R = all[inp.model], mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      const ax = C().axes({ prefix: 'mi:ax', x: 40, y: 70, w: 340, h: 200, xr: [0, 1], yr: [0, 1], xticks: 0.2, yticks: 0.25, xname: 'confidence in the true outcome', yname: 'share of rows', tickSize: 12 });
      const hm = P.hist(ax, 'mi:m', R.mem, 0, 0.1, 10, 'p17mem', 1 / R.mem.length), hn = P.hist(ax, 'mi:n', R.non, 0, 0.1, 10, 'p17non', 1 / R.non.length);
      const keys = [...P.key('mi:k0', 40, 330, 'p17mem', 'training rows (members)'), ...P.key('mi:k1', 240, 330, 'p17non', 'unseen rows')];
      F.set('models', 1);
      F.push([...ax.items, ...hm.items, ...keys, H().head('mi:h', 40, 30, MODELS[inp.model])],
        `Train ${MODELS[inp.model]} on ${P.qpl(q, R.mem.length, 'applicant')} of the loans data. It is right on ${q.pct(R.accTr)} of them, but on ${q.pct(R.accTe)} of ${q.n(R.non.length)} applicants it never saw.` + (R.accTr - R.accTe > 0.1 ? ' A gap that size means it has memorised its training rows.' : ' The two are close: it has not memorised much.'), { phase: 'fit', accTr: R.accTr, accTe: R.accTe, nMem: R.mem.length, nNon: R.non.length });
      F.push([...ax.items, ...hm.items, ...hn.items, ...keys, H().head('mi:h', 40, 30, 'How sure is it?')],
        `Its confidence in the true outcome averages ${q(mean(R.mem), 3)} on its training rows and ${q(mean(R.non), 3)} on unseen rows. The filled bars (members) and the outlined bars (unseen) ${mean(R.mem) - mean(R.non) > 0.05 ? 'sit apart' : 'nearly coincide'}.`, { phase: 'hist', mMem: mean(R.mem), mNon: mean(R.non), counts: hm.counts, countsN: hn.counts });
      F.push([...ax.items, ...hm.items, ...hn.items, ...keys, C().vline(ax, 'mi:t', Math.min(1, R.t), 'p17thr'), H().head('mi:h', 40, 30, 'The best guess from confidence alone'), ...H().lines('mi:g', 430, 90, [['guess “member” if confidence ≥ ' + H().fx(R.t, 3), 'legend'], ['members caught (TPR): ' + H().fx(R.tpr, 3), 'mono'], ['false alarms (FPR):   ' + H().fx(R.fpr, 3), 'mono'], ['advantage TPR − FPR:  ' + H().fx(R.adv, 3), 'mono']], 13, 24)],
        `The best rule of the form “a member if the confidence is at least t” uses t = ${q(R.t, 3)}: it flags ${q.pct(R.tpr)} of members and ${q.pct(R.fpr)} of unseen rows, an advantage of ${q(R.adv, 3)} over guessing. Zero would mean confidence reveals nothing about membership.`, { phase: 'guess', t: R.t, tpr: R.tpr, fpr: R.fpr, adv: R.adv });
      const ks = Object.keys(MODELS), short = { deep: 'full tree', depth4: 'depth 4', leaf20: 'leaves ≥ 20', logistic: 'logistic', dp: 'DP logistic' };
      const hb = S.Kit.dt.hbars({ prefix: 'mi:b', x: 40, y: 80, w: 330, rowH: 30, vals: ks.map((k) => all[k].adv), names: ks.map((k) => short[k]), max: Math.max(0.05, ...ks.map((k) => all[k].adv)), fmt: (v) => H().fx(v, 3), nameW: 100, cls: (i) => (ks[i] === inp.model ? 'p17acc' : 'p17grey') });
      const hb2 = S.Kit.dt.hbars({ prefix: 'mi:c', x: 430, y: 80, w: 300, rowH: 30, vals: ks.map((k) => all[k].accTe), names: ks.map(() => ''), max: 1, fmt: (v) => H().pct(v), nameW: 10, cls: (i) => (ks[i] === inp.model ? 'p17acc' : 'p17grey') });
      const lo = ks.reduce((b, k) => (all[k].adv < all[b].adv ? k : b)), hi = ks.reduce((b, k) => (all[k].adv > all[b].adv ? k : b));
      F.set('models', ks.length);
      F.push([...hb, ...hb2, H().text('mi:bt', 40, 66, 'membership advantage (lower is better)', 'axname', 13), H().text('mi:ct', 440, 66, 'test accuracy', 'axname', 13), H().head('mi:h', 40, 30, 'Defences: regularise, or train privately')],
        `Across the five models the leak is largest for ${MODELS[hi]} (${q(all[hi].adv, 3)}) and smallest for ${MODELS[lo]} (${q(all[lo].adv, 3)}). Limiting memorisation (shallower trees, larger leaves, simpler models) and private training shrink the advantage, here at a test accuracy of ${q.pct(Math.min(...ks.map((k) => all[k].accTe)))} to ${q.pct(Math.max(...ks.map((k) => all[k].accTe)))}.`, { phase: 'compare', advs: ks.map((k) => all[k].adv), accs: ks.map((k) => all[k].accTe), hi: all[hi].adv, lo: all[lo].adv });
      for (const f of F.list) f.narrowView = { x: 20, y: 0, w: 400, h: 350 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Membership advantage', text: ['confidence c(x) = p(true outcome of x)', 'guess “member” if c(x) ≥ t', 'TPR = members flagged / members', 'FPR = unseen flagged / unseen', 'advantage = max over t of TPR − FPR', 'defences: regularise, DP training'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'fit', pick: 'option', once: true,
      ask: 'Will this model be clearly surer of its training rows than of unseen ones?',
      options: ['yes', 'no'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'hist').meta; return m.mMem - m.mNon > 0.05 ? 'yes' : 'no'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'hist').say,
    }],
    quiz: [
      { q: 'A guess flags 90% of members and 60% of unseen rows. Its advantage is…', options: ['0.30', '0.90', '1.50', '0.60'], answer: 0, why: 'TPR − FPR = 0.9 − 0.6.' },
      { q: 'Why does a fully grown tree leak membership?', options: ['it is nearly certain on rows it memorised', 'trees are encrypted', 'it never sees training rows', 'it has no leaves'], answer: 0, why: 'Its leaves hold single training rows, so it is sure of exactly those.' },
      { q: 'Which change most directly shrinks the leak?', options: ['regularising the model or training with DP-SGD', 'training longer', 'growing deeper trees', 'publishing the training set'], answer: 0, why: 'Both limit how much any one row shapes the model.' },
      { q: 'An advantage of zero means…', options: ['confidence says nothing about membership', 'the model is useless', 'every row is a member', 'the model is overfitted'], answer: 0, why: 'Members and unseen rows look the same to the guess.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
