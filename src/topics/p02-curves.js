// Part 2, threshold curves. The ROC curve with a live threshold: the first frame puts the slider's threshold on the
// curve, the score strip and the confusion matrix together; then the threshold sweeps down and the area builds up.
// AUC as a ranking: every positive against every negative in a grid, the share of pairs ordered correctly. The
// precision–recall curve with average precision drawn as its rectangles, beside the ROC area on rare positives.
(function (S) {
  'use strict';
  const H = S.P01, P = S.P02, C = () => S.Kit.curve, M = () => S.Metrics;
  const r2 = (v) => Math.round(v * 100) / 100;
  /** a filled area under a curve of points [[u, v]..], down to v = 0 */
  const area = (ax, key, pts, cls) => ({ key, kind: 'path', d: pts.length > 1 ? 'M' + ax.X(pts[0][0]) + ',' + ax.Y(0) + ' ' + pts.map((p) => 'L' + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' ') + ' L' + ax.X(pts[pts.length - 1][0]) + ',' + ax.Y(0) + ' Z' : 'M0,0', cls: 'area ' + (cls || ''), audit: 'ignore', layer: 'back', z: 2 });

  S.topic({
    id: 'eval-roc', family: 'p02-threshold', order: 20, title: 'The ROC curve and AUC',
    blurb: 'Sweep the threshold from the top score down to the bottom and plot, at every step, the true-positive rate (recall) against the false-positive rate (the share of negatives flagged). That path is the ROC curve: it starts at (0, 0) with nothing flagged and ends at (1, 1) with everything flagged. A model that guesses walks the diagonal; a good one climbs steeply first. The area under it, AUC, sums the curve up in one number that does not depend on any one threshold. The slider\'s threshold is the first frame, linked to its confusion matrix.',
    inputs: [
      { id: 'src', kind: 'choice', label: 'Task', def: 'loans', options: P.sourceOptions(['loans', 'cancer', 'moons', 'pumps', 'exam']) },
      { id: 't', kind: 'slider', label: 'Threshold', def: 0.4, min: 0, max: 1, step: 0.01, fmt: (v) => (+v).toFixed(2) },
    ],
    presets: [
      { label: 'Loans at 0.4', values: { src: 'loans', t: 0.4 } },
      { label: 'Loans at 0.15', values: { src: 'loans', t: 0.15 } },
      { label: 'Cancer: nearly perfect', values: { src: 'cancer', t: 0.5 } },
      { label: 'Exam: a weaker model', values: { src: 'exam', t: 0.5 } },
      { label: 'Rare pump faults', values: { src: 'pumps', t: 0.2 } },
    ],
    counters: [{ id: 'flag', label: 'flagged' }],
    run(inp) {
      const B = P.binary(inp.src), roc = M().roc(B.y, B.score), n = B.y.length, Pn = roc.P, Nn = roc.N;
      // the curve starts at the corner, where its dot would sit on both zeros: one 0, set off below and to the left
      const nz = (u) => (u === 0 ? '' : C().fmt(u));
      const ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 300, h: 300, xr: [0, 1], yr: [0, 1], xname: 'false-positive rate', yname: 'true-positive rate', tickSize: 12, fmtX: nz, fmtY: nz });
      ax.items.push({ key: 'ax:t:o', kind: 'text', x: 50 - 10, y: 50 + 300 + 19, text: '0', anchor: 'end', size: 12, cls: 'tick' });
      const sx = C().axes({ prefix: 'sx', x: 420, y: 290, w: 350, h: 90, xr: [0, 1], yr: [0, 1], yticks: 0, xname: 'P(' + B.posName + ')', grid: false, tickSize: 12 });
      const pts = roc.fpr.map((f, k) => [f, roc.tpr[k]]);
      const diag = { key: 'diag', kind: 'path', d: `M${ax.X(0)},${ax.Y(0)} L${ax.X(1)},${ax.Y(1)}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 };
      const F = new S.Frames(['flag']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, B.title + ': ' + Pn + ' ' + B.posName + ', ' + Nn + ' ' + B.negName + ' test examples');
      // the curve up to the point where the threshold has reached t (score ≥ t is flagged)
      const upto = (t) => { const k = roc.thresholds.reduce((b, th, i) => (th >= t ? i : b), 0); return k; };
      const state = (t, opts) => {
        const c = P.at(B, t), k = upto(t), part = pts.slice(0, k + 1);
        const fpr = M().div(c.fp, Nn), tpr = M().div(c.tp, Pn);
        return [title, ...ax.items, diag, area(ax, 'auc', opts.full ? pts : part, ''), C().line(ax, 'roc', opts.full ? pts : part, 'act'), C().dot(ax, 'pt', fpr, tpr, 'c1', 7),
          ...P.cm2({ prefix: 'cm', x: 540, y: 76, cell: 66, c, posName: B.posName, negName: B.negName }),
          ...sx.items, ...P.strip(sx, 's', B, { t, mark: true }).map((d) => Object.assign(d, { r: 2.4 })), { key: 'st', kind: 'path', d: `M${sx.X(Math.min(1, t))},${sx.o.y} V${sx.o.y + sx.o.h}`, cls: 'thr', audit: 'ignore', layer: 'lines' },
          H.text('rt', 420, 262, 'TPR ' + c.tp + '/' + Pn + ' = ' + H.f(tpr, 3) + '   FPR ' + c.fp + '/' + Nn + ' = ' + H.f(fpr, 3), opts.lit ? 'act' : 'strong', 13)].concat(opts.aucText ? [H.text('auct', ax.X(0.55), ax.Y(0.2), 'AUC = ' + H.f(roc.auc, 3), 'act', 16)] : []);
      };
      const c0 = P.at(B, inp.t);
      F.set('flag', c0.tp + c0.fp);
      F.add({ items: state(inp.t, { full: true, lit: true }), say: `Your threshold ${H.f(inp.t, 2)} flags ${c0.tp + c0.fp} examples: ${c0.tp} of the ${Pn} ${B.posName} (TPR ${H.f(M().div(c0.tp, Pn), 3)}) and ${c0.fp} of the ${Nn} ${B.negName} (FPR ${H.f(M().div(c0.fp, Nn), 3)}). That is the orange point on the curve.`, meta: { phase: 'yours', t: inp.t, tp: c0.tp, fp: c0.fp, tpr: M().div(c0.tp, Pn), fpr: M().div(c0.fp, Nn) }, view });
      F.set('flag', 0);
      F.add({ items: state(1.01, { full: false }), say: `Start the sweep above every score: nothing is flagged, so TPR = FPR = 0 and the curve starts at the corner (0, 0).`, meta: { phase: 'start' }, view });
      for (const t of [0.8, 0.6, 0.4, 0.2, 0.05]) {
        const c = P.at(B, t);
        F.set('flag', c.tp + c.fp);
        F.add({ items: state(t, { full: false }), say: `Threshold ${H.f(t, 2)}: ${c.tp + c.fp} flagged. TPR = ${c.tp} / ${Pn} = ${H.f(M().div(c.tp, Pn), 3)}, FPR = ${c.fp} / ${Nn} = ${H.f(M().div(c.fp, Nn), 3)}. Each ${B.posName} passed moves the curve up, each ${B.negName} moves it right.`, meta: { phase: 'sweep', t, tp: c.tp, fp: c.fp }, view });
      }
      F.set('flag', n);
      F.add({ items: state(-0.01, { full: true, aucText: true }), say: `Below every score everything is flagged: (1, 1). The shaded area under the curve is AUC = ${H.f(roc.auc, 3)} by the trapezoid rule; the diagonal of a guessing model has area 0.5.`, meta: { phase: 'auc', auc: roc.auc }, view });
      const ctx = { n, P: Pn, N: Nn, guess: 0.5 };
      F.list.forEach((f) => { f.meta.ctx = ctx; if (f.meta.tp !== undefined) { f.meta.flagged = f.meta.tp + f.meta.fp; f.meta.rates = [M().div(f.meta.tp, Pn), M().div(f.meta.fp, Nn)]; } });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'ROC', text: ['TPR = TP / P (recall)', 'FPR = FP / N', 'one point per threshold, from (0,0) to (1,1)', 'diagonal: guessing, AUC 0.5', 'AUC = P(score of a random positive', '      > score of a random negative)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will the area under this curve be above 0.8?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.auc > 0.8 ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Where does every ROC curve start and end?', options: ['(0, 0) and (1, 1)', '(0, 1) and (1, 0)', '(0.5, 0.5) and (1, 1)', 'anywhere'], answer: 0, why: 'Nothing flagged, then everything flagged.' },
      { q: 'A model that scores at random has AUC about…', options: ['0.5', '0', '1', '0.25'], answer: 0, why: 'Its curve follows the diagonal.' },
      { q: 'AUC = 0.9 means…', options: ['a random positive outscores a random negative 90% of the time', 'accuracy is 90%', '90% of positives are caught at 0.5', 'precision is 0.9'], answer: 0, why: 'AUC measures ranking, not any one threshold.' },
      { q: 'Multiplying every score by 2 changes the AUC…', options: ['not at all', 'by doubling it', 'by halving it', 'to 1'], answer: 0, why: 'The order of the scores is all that matters.' },
      { q: 'A model has AUC 0.2. What can you do?', options: ['flip its predictions to get AUC 0.8', 'nothing', 'it is perfect', 'raise the threshold'], answer: 0, why: 'It ranks consistently the wrong way round.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // AUC as a ranking
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'eval-auc-rank', family: 'p02-threshold', order: 30, title: 'AUC as a ranking',
    blurb: 'AUC has a second meaning that needs no curve. Pair every positive example with every negative one and ask: did the model give the positive the higher score? The share of pairs it ordered correctly, counting a tie as half, is exactly the area under the ROC curve. So AUC judges how well a model ranks, whatever threshold is used later.',
    inputs: [
      { id: 'src', kind: 'choice', label: 'Task', def: 'loans', options: P.sourceOptions(['loans', 'exam', 'cancer', 'moons']) },
      { id: 'k', kind: 'slider', label: 'Examples of each class', def: 10, min: 4, max: 14, step: 1 },
    ],
    presets: [
      { label: 'Loans, 10 + 10', values: { src: 'loans', k: 10 } },
      { label: 'Exam, 14 + 14', values: { src: 'exam', k: 14 } },
      { label: 'Cancer, 8 + 8', values: { src: 'cancer', k: 8 } },
    ],
    counters: [{ id: 'pairs', label: 'pairs compared' }],
    run(inp) {
      const B = P.binary(inp.src), k = inp.k;
      // a spread sample of each class: every m-th example in score order
      const pick = (cls) => { const idx = B.y.map((v, i) => (v === cls ? i : -1)).filter((i) => i >= 0).sort((a, b) => B.score[b] - B.score[a] || a - b); const step = idx.length / k; return Array.from({ length: k }, (_, j) => idx[Math.floor(j * step + step / 2)]); };
      const pos = pick(1), neg = pick(0), sp = pos.map((i) => B.score[i]), sn = neg.map((i) => B.score[i]);
      const cmp = sp.map((a) => sn.map((b) => (a > b ? 1 : a < b ? 0 : 0.5)));
      const wins = H.sum(cmp.flat()), total = k * k, aucPairs = wins / total;
      const roc = M().roc(pos.concat(neg).map((i) => B.y[i]), sp.concat(sn));
      const cs = Math.min(26, 300 / k), gx = 150, gy = 96;
      const F = new S.Frames(['pairs']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, B.title + ': ' + k + ' ' + B.posName + ' and ' + k + ' ' + B.negName + ' examples');
      const heads = [H.text('ch', gx + (k * (cs + 2)) / 2, gy - 44, B.negName + ' examples, scores high → low', 'tick', 12, 'middle'), H.text('rh', gx - 8, gy - 44, B.posName + ' ↓', 'tick', 12, 'end')]
        .concat(sn.map((s, j) => (k <= 10 || j % 2 === 0 ? H.text('cs' + j, gx + j * (cs + 2) + cs / 2, gy - 8, H.f(s, 2).replace(/^0\./, '.'), 'tick', 11, 'middle') : null)).filter(Boolean))
        .concat(sp.map((s, i) => H.text('rs' + i, gx - 8, gy + i * (cs + 2) + cs / 2 + 4, H.f(s, 2), 'tick', 12, 'end')));
      const cells = (rows) => {
        const it = [];
        for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
          const v = cmp[i][j], on = i < rows;
          it.push({ key: 'g' + i + '_' + j, kind: 'cell', x: r2(gx + j * (cs + 2)), y: r2(gy + i * (cs + 2)), w: r2(cs), h: r2(cs), rx: 2, text: '', cls: 'hm', audit: 'ignore', paint: { fill: !on ? 'var(--surface2)' : v === 1 ? 'color-mix(in srgb, var(--good) 70%, var(--node))' : v === 0 ? 'color-mix(in srgb, var(--bad) 70%, var(--node))' : 'var(--hot)' } });
        }
        return it;
      };
      const side = (lines) => lines.map((l, i) => H.text('sd' + i, 480, 110 + i * 26, l[0], l[1] || 'strong', 14));
      const legend = [{ key: 'lg1', kind: 'cell', x: 480, y: 300, w: 14, h: 14, rx: 2, text: '', cls: 'hm', paint: { fill: 'color-mix(in srgb, var(--good) 70%, var(--node))' } }, H.text('lg1t', 500, 312, B.posName + ' scored higher: right', 'legend', 13), { key: 'lg2', kind: 'cell', x: 480, y: 324, w: 14, h: 14, rx: 2, text: '', cls: 'hm', paint: { fill: 'color-mix(in srgb, var(--bad) 70%, var(--node))' } }, H.text('lg2t', 500, 336, B.negName + ' scored higher: wrong', 'legend', 13)];
      F.add({ items: [title, ...heads, ...cells(0), ...legend, ...side([[k + ' × ' + k + ' = ' + total + ' pairs', 'strong']])], say: `${k} ${B.posName} examples (rows) and ${k} ${B.negName} examples (columns), each sorted by score. Every cell is one pair: ${total} pairs in all.`, meta: { phase: 'grid', total }, view });
      const steps = [Math.ceil(k / 3), Math.ceil((2 * k) / 3), k];
      let done = 0;
      for (const rows of steps) {
        const w = H.sum(cmp.slice(0, rows).flat()); done = rows * k;
        F.set('pairs', done);
        F.add({ items: [title, ...heads, ...cells(rows), ...legend, ...side([['pairs so far: ' + done, 'strong'], ['ordered right: ' + H.g(w, 4), 'good']])], say: `The first ${rows} ${B.posName} rows: ${H.g(w, 4)} of their ${done} pairs have the ${B.posName} example scored higher.`, meta: { phase: 'fill', rows, wins: w, done }, view });
      }
      F.add({ items: [title, ...heads, ...cells(k), ...legend, ...side([['ordered right: ' + H.g(wins, 4) + ' / ' + total, 'good'], ['= ' + H.f(aucPairs, 4), 'act'], ['ROC area of these ' + 2 * k + ': ' + H.f(roc.auc, 4), 'strong']])], say: `${H.g(wins, 4)} of ${total} pairs are ordered correctly, a share of ${H.f(aucPairs, 4)}. The trapezoid area under the ROC curve of the same ${2 * k} examples is ${H.f(roc.auc, 4)}: the same number.`, meta: { phase: 'auc', wins, total, aucPairs, aucRoc: roc.auc }, view });
      F.list.forEach((f) => { f.meta.ctx = { k, n: 2 * k }; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'AUC as pairs', text: ['AUC = (1 / (P·N)) Σ over pairs (pos, neg) of', '  1 if score(pos) > score(neg)', '  ½ if equal, 0 otherwise', '= area under the ROC curve', 'only the order of the scores matters'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'grid', pick: 'option', ask: 'What share of the pairs will be ordered correctly?', options: ['above 0.8', 'between 0.6 and 0.8', 'below 0.6'], answer: (fr) => { const a = fr[fr.length - 1].meta.aucPairs; return a > 0.8 ? 'above 0.8' : a >= 0.6 ? 'between 0.6 and 0.8' : 'below 0.6'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: '3 positives and 2 negatives: 5 of the 6 pairs are ordered correctly. AUC is…', options: ['5/6', '3/5', '0.5', '1'], answer: 0, why: 'AUC is the share of correctly ordered pairs.' },
      { q: 'A pair whose two scores are equal counts as…', options: ['half a correct pair', 'a correct pair', 'a wrong pair', 'it is skipped'], answer: 0, why: 'A tie is a coin flip for the ranking.' },
      { q: 'Why does a monotone change of the scores (squaring them, say) leave AUC alone?', options: ['every pair keeps its order', 'AUC ignores scores', 'it does change it', 'squares are positive'], answer: 0, why: 'Only comparisons enter the count.' },
      { q: 'With 100 positives and 900 negatives, how many pairs are there?', options: ['90 000', '1000', '100', '810 000'], answer: 0, why: '100 × 900.' },
      { q: 'What does AUC not tell you?', options: ['how well calibrated the probabilities are', 'how well the model ranks', 'the chance a positive outscores a negative', 'the area under the ROC curve'], answer: 0, why: 'Any order-keeping change of the scores gives the same AUC, honest probabilities or not.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Precision–recall curve and average precision
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'eval-pr', family: 'p02-threshold', order: 40, title: 'Precision–recall curves and average precision',
    blurb: 'The precision–recall curve plots, for every threshold, the precision of the flagged examples against the recall. A model that guesses has precision equal to the share of positives, so the floor of this plot moves with the class balance, which is exactly why it is useful when positives are rare: a ROC curve can look excellent while most flags are still false alarms. Average precision sums the curve as rectangles: each step in recall times the precision reached there.',
    inputs: [{ id: 'src', kind: 'choice', label: 'Task', def: 'pumps', options: P.sourceOptions(['pumps', 'loans', 'cancer', 'exam']) }],
    presets: [
      { label: 'Rare pump faults', values: { src: 'pumps' } },
      { label: 'Loans: a quarter default', values: { src: 'loans' } },
      { label: 'Cancer', values: { src: 'cancer' } },
    ],
    counters: [{ id: 'th', label: 'thresholds' }],
    run(inp) {
      const B = P.binary(inp.src), pr = M().prCurve(B.y, B.score), roc = M().roc(B.y, B.score), n = B.y.length, Pn = roc.P, prev = Pn / n;
      const ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 320, h: 300, xr: [0, 1], yr: [0, 1.02], xname: 'recall', yname: 'precision', tickSize: 12 });
      const rx = C().axes({ prefix: 'rx', x: 520, y: 150, w: 200, h: 200, xr: [0, 1], yr: [0, 1], xticks: [0, 0.5, 1], yticks: [0, 0.5, 1], xname: 'FPR', yname: 'TPR', tickSize: 12 });
      // the curve from the highest threshold down: pr arrays run from the lowest threshold up, then end at recall 0
      const pts = pr.recall.map((r, k) => [r, pr.precision[k]]).reverse();
      const rects = (upto) => { const it = []; for (let k = 1; k <= upto && k < pts.length; k++) { const [r0] = pts[k - 1], [r1, p1] = pts[k]; if (r1 - r0 > 1e-12) it.push({ key: 'rc' + k, kind: 'cell', x: ax.X(r0), y: ax.Y(p1), w: r2(Math.max(0.5, ax.X(r1) - ax.X(r0))), h: r2(ax.Y(0) - ax.Y(p1)), rx: 0, text: '', cls: 'hbin soft', audit: 'ignore', layer: 'back' }); } return it; };
      const base = { key: 'base', kind: 'path', d: `M${ax.X(0)},${ax.Y(prev)} H${ax.X(1)}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 };
      const F = new S.Frames(['th']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, B.title + ': ' + Pn + ' of ' + n + ' test examples are ' + B.posName + ' (' + H.pct(prev) + ')');
      const side = (lines) => lines.map((l, i) => H.text('sd' + i, 420, 70 + i * 24, l[0], l[1] || 'strong', 14));
      const nT = pts.length - 1;
      const marks = [Math.max(1, Math.round(nT * 0.1)), Math.round(nT * 0.4), nT];
      F.add({ items: [title, ...ax.items, base, H.text('bt', ax.X(1), ax.Y(prev) - 6, 'guessing: ' + H.f(prev, 3), 'tick', 12, 'end'), ...side([['a guess flags at random:', 'tick'], ['precision = share of ' + B.posName + ' = ' + H.f(prev, 3), 'strong']])], say: `${Pn} of ${n} test examples are ${B.posName}, so flagging at random gives precision ${H.f(prev, 3)} at any recall: the dashed floor.`, meta: { phase: 'floor', prev, P: Pn, n }, view });
      for (const m of marks) {
        const part = pts.slice(0, m + 1), [rr, pp] = pts[m], ap = H.sum(part.slice(1).map((p, k) => (p[0] - part[k][0]) * p[1]));
        F.set('th', m);
        F.add({ items: [title, ...ax.items, base, ...rects(m), C().line(ax, 'pr', part, 'act'), C().dot(ax, 'pt', rr, pp, 'c1'), ...side([[m + ' of ' + nT + ' thresholds passed', 'tick'], ['recall ' + H.f(rr, 3) + ', precision ' + H.f(pp, 3), 'strong'], ['rectangles so far: ' + H.f(ap, 3), 'act']])], say: `Lowering the threshold past ${m} of the ${nT} distinct scores: recall ${H.f(rr, 3)} at precision ${H.f(pp, 3)}. Each step right adds a rectangle, its width the gain in recall and its height the precision there: ${H.f(ap, 3)} so far.`, meta: { phase: 'sweep', m, recall: rr, precision: pp, ap }, view });
      }
      F.add({ items: [title, ...ax.items, base, ...rects(nT), C().line(ax, 'pr', pts, 'act'), ...side([['average precision = ' + H.f(pr.ap, 3), 'act'], ['ROC AUC = ' + H.f(roc.auc, 3), 'strong'], ['the floor: ' + H.f(prev, 3), 'tick']]), ...rx.items, { key: 'rd', kind: 'path', d: `M${rx.X(0)},${rx.Y(0)} L${rx.X(1)},${rx.Y(1)}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 }, C().line(rx, 'roc', roc.fpr.map((f, k) => [f, roc.tpr[k]]), 'c0')], say: `Average precision, the sum of all the rectangles, is ${H.f(pr.ap, 3)}; the ROC AUC of the same scores is ${H.f(roc.auc, 3)} (the small curve).` + (prev < 0.15 ? ` With only ${H.pct(prev)} positives, the ROC curve hides how many false alarms come with each catch; the precision–recall curve shows them.` : ''), meta: { phase: 'ap', ap: pr.ap, auc: roc.auc, prev }, view });
      F.list.forEach((f) => { f.meta.ctx = { nT, P: Pn, n }; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'PR and AP', text: ['precision = TP / (TP + FP)', 'recall = TP / P', 'guessing: precision = P / n at every recall', 'AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ', 'rare positives: prefer PR over ROC'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'floor', pick: 'option', ask: 'Will average precision be lower than the ROC AUC for these scores?', options: ['lower', 'higher', 'about the same'], answer: (fr) => { const m = fr[fr.length - 1].meta; return m.ap < m.auc - 0.01 ? 'lower' : m.ap > m.auc + 0.01 ? 'higher' : 'about the same'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: '2% of examples are positive. A random scorer has average precision about…', options: ['0.02', '0.5', '0', '1'], answer: 0, why: 'Its precision is the positive share at every recall.' },
      { q: 'Why can ROC look good while precision is poor?', options: ['FPR divides by the many negatives, so many false alarms still give a small FPR', 'ROC ignores positives', 'precision is always low', 'they measure the same thing'], answer: 0, why: '100 false alarms among 100 000 negatives is FPR 0.001.' },
      { q: 'Average precision adds up…', options: ['each gain in recall times the precision at that point', 'the precisions at 0.5', 'TPR times FPR', 'the F1 scores'], answer: 0, why: 'AP = Σ (Rₙ − Rₙ₋₁) Pₙ.' },
      { q: 'At the lowest threshold every example is flagged. Precision there equals…', options: ['the share of positives', '1', '0', 'the recall'], answer: 0, why: 'All positives and all negatives are flagged.' },
      { q: 'Which curve\'s guessing baseline depends on the class balance?', options: ['precision–recall', 'ROC', 'both', 'neither'], answer: 0, why: 'ROC\'s guessing line is always the diagonal.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
