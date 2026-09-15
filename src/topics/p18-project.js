// Part 18, a checklist for a real project, drawn as a pipeline and run step by step on one table (small loans): the
// question, a data audit, the split, a baseline, a model, cross-validation, error analysis, fairness and robustness
// checks, an explanation, and monitoring after deployment. Every number on the way comes from running that step.
(function (S) {
  'use strict';
  const P = S.P18, H = () => S.Kit.ml, M = () => S.Metrics;
  const STEPS = [
    ['question', ['question']], ['audit', ['data', 'audit']], ['split', ['split']], ['baseline', ['baseline']], ['model', ['model']],
    ['validate', ['cross-', 'validation']], ['errors', ['error', 'analysis']], ['fair', ['fairness and', 'robustness']], ['explain', ['explanation']], ['monitor', ['deploy and', 'monitor']],
  ];
  const NAMES = ['income', 'debt ratio', 'years', 'missed', 'amount'];

  P.projectRun = (missing, seed) => P.once(['proj', missing, seed].join(':'), () => {
    const d = S.Data.load('loans', { n: 600, seed, missing }), n = d.y.length, X = d.X, y = d.y, grp = X.map((r) => r[5]);
    const cells = X.reduce((s, r) => s + r.slice(0, 5).filter((v) => !Number.isFinite(v)).length, 0), rowsMiss = X.filter((r) => r.slice(0, 5).some((v) => !Number.isFinite(v))).length;
    const perCol = [0, 1, 2, 3, 4].map((j) => X.filter((r) => !Number.isFinite(r[j])).length);
    const sp = S.Data.split(y, 0.25, 1), tr = sp.train, te = sp.test;
    const med = [0, 1, 2, 3, 4].map((j) => { const v = tr.map((i) => X[i][j]).filter(Number.isFinite).sort((a, b) => a - b), m = v.length; return m % 2 ? v[(m - 1) / 2] : (v[m / 2 - 1] + v[m / 2]) / 2; });
    const fill = (r) => r.slice(0, 5).map((v, j) => (Number.isFinite(v) ? v : med[j]));
    const Xtr = tr.map((i) => fill(X[i])), ytr = tr.map((i) => y[i]), Xte = te.map((i) => fill(X[i])), yte = te.map((i) => y[i]), gte = te.map((i) => grp[i]);
    const maj = ytr.filter((v) => v === 1).length * 2 > ytr.length ? 1 : 0, base = M().accuracy(yte, yte.map(() => maj)).value;
    const sc = P.scaler(Xtr), mdl = new S.ML.LogisticRegression({}).fit(sc.apply(Xtr), ytr), prob = (Z) => mdl.predictProba(sc.apply(Z)).map((r) => r[1]);
    const pte = prob(Xte), pred = pte.map((p) => (p >= 0.5 ? 1 : 0)), pr = M().precisionRecallF1(yte, pred, { posLabel: 1 });
    const acc = M().accuracy(yte, pred).value, auc = M().rocAuc(yte, pte).value, cm = M().confusion(yte, pred, { labels: [0, 1] }).counts;
    const cv = P.cvScores(P.MODELS[1], Xtr, ytr, 5);
    const fair = M().fairness(yte, pred, gte.map((g) => 'AB'[g]), { posLabel: 1 });
    const shifted = Xte.map((r) => [r[0] * 0.8].concat(r.slice(1))), accShift = M().accuracy(yte, prob(shifted).map((p) => (p >= 0.5 ? 1 : 0))).value;
    const imp = S.Explain.permutationImportance(prob, Xte, yte, { nRepeats: 5, seed: 1, score: (a, b) => M().rocAuc(a, b).value, names: NAMES });
    const fresh = S.Data.load('loans', { n: 300, seed: seed + 50 }).X.map((r) => r.slice(0, 5)), poorer = fresh.map((r) => [r[0] * 0.8].concat(r.slice(1)));
    const share = (Z) => prob(Z).filter((p) => p >= 0.5).length / Z.length;
    return { n, rate: y.filter((v) => v).length / n, cells, rowsMiss, perCol, nTr: tr.length, nTe: te.length, med, base, maj, acc, auc, recall: pr.recall, precision: pr.precision, cm, cv, cvMean: P.mean(cv), cvSd: P.sd(cv), fair, accShift, imp, shareTest: pred.filter((v) => v).length / pred.length, shareNew: share(fresh), sharePoor: share(poorer), testRate: yte.filter((v) => v).length / yte.length };
  });

  S.topic({
    id: 'project-checklist', family: 'p18-project', order: 10, title: 'A checklist for a real project',
    blurb: 'A model is one step of ten. Before it: a clear question, an audit of the data, an honest split and a baseline to beat. After it: validation, a look at the errors, checks for fairness and robustness, an explanation someone can question, and monitoring once it is in use, because the world it was trained on keeps moving. Here every step is run on one table, a lender’s small loans, with the numbers each step produces.',
    inputs: [
      { id: 'missing', kind: 'choice', label: 'Missing cells in the data', def: '0.03', options: [{ value: '0', label: 'none' }, { value: '0.03', label: 'about 3%' }, { value: '0.1', label: 'about 10%' }] },
      { id: 'seed', kind: 'number', label: 'Seed (the data)', def: 1, min: 1, max: 999 },
    ],
    presets: [{ label: 'Loans, 3% missing', values: { missing: '0.03', seed: 1 } }, { label: 'Clean data', values: { missing: '0', seed: 1 } }, { label: 'Many gaps, another sample', values: { missing: '0.1', seed: 4 } }],
    counters: [{ id: 'step', label: 'steps done' }],
    run(inp) {
      const R = P.projectRun(+inp.missing, inp.seed), F = H().frames(['step']), q = F.q;
      const N = (now) => {
        const net = S.Kit.flow.net('pc');
        STEPS.forEach(([id, lines], k) => { const row = k < 5 ? 0 : 1, col = row ? 9 - k : k; net.box(id, 90 + col * 140, 60 + row * 96, 116, 48, lines, 'p18step' + (k < now ? ' done' : k === now ? ' now' : '')); });
        STEPS.forEach(([id], k) => {
          if (k === 9) return;
          const nx = STEPS[k + 1][0];
          if (k === 4) net.wire('w' + k, [net.port(id, 's'), net.port(nx, 'n')], { head: true, cls: k < now ? 'p18on' : '' });
          else if (k < 4) net.wire('w' + k, [net.port(id, 'e'), net.port(nx, 'w')], { head: true, cls: k < now ? 'p18on' : '' });
          else net.wire('w' + k, [net.port(id, 'w'), net.port(nx, 'e')], { head: true, cls: k < now ? 'p18on' : '' });
        });
        return net.items;
      };
      const panel = (title, rows, extra) => [H().text('pc:pt', 32, 238, title, 'strong', 15), ...H().lines('pc:pl', 32, 268, rows, 13, 23), ...(extra || [])];
      const step = (k, title, rows, say, meta, extra) => { F.set('step', k + 1); F.push([...N(k), ...panel(title, rows, extra)], say, Object.assign({ phase: STEPS[k][0], k }, meta)); };
      step(0, '1. The question', [['predict whether an applicant will default', 'legend'], [R.n + ' past applications, ' + H().pct(R.rate) + ' defaulted', 'legend'], ['a missed default costs more than a false alarm', 'legend']],
        `Write the question down first: will this applicant default? We have ${P.qpl(q, R.n, 'past application')}, ${q.pct(R.rate)} of which defaulted, and a missed default costs the lender more than a refused good applicant. That decides the metric: recall of defaults matters, not accuracy alone.`, { n: R.n, rate: R.rate });
      step(1, '2. Audit the data', [['missing cells: ' + R.cells + ' (in ' + R.rowsMiss + ' rows)', 'legend'], ['per column: ' + NAMES.map((nm, j) => nm + ' ' + R.perCol[j]).join(', '), 'legend'], ['the group column (A, B) is sensitive: kept out of the model', 'legend']],
        R.cells ? `Look before modelling: ${P.qpl(q, R.cells, 'cell')} are missing, spread over ${P.qpl(q, R.rowsMiss, 'row')}; the worst column, ${NAMES[R.perCol.indexOf(Math.max(...R.perCol))]}, lacks ${q.n(Math.max(...R.perCol))}. The group column is a protected attribute: it stays out of the model but is kept to check fairness.` : `Look before modelling: no cell is missing here. The group column is a protected attribute: it stays out of the model but is kept to check fairness.`, { cells: R.cells, rowsMiss: R.rowsMiss, worst: Math.max(...R.perCol) });
      step(2, '3. Split, then clean', [[R.nTr + ' rows to train, ' + R.nTe + ' locked away to test (stratified)', 'legend'], ['gaps filled with the training medians:', 'legend'], [NAMES.map((nm, j) => nm + ' ' + H().sg(R.med[j], 3)).join(', '), 'math']],
        `Split first, stratified: ${q.n(R.nTr)} rows to train and ${q.n(R.nTe)} locked away. Only then fill the gaps, with medians from the training rows alone (income ${q.sg(R.med[0], 3)}): statistics taken from the test rows would leak them into training.`, { nTr: R.nTr, nTe: R.nTe, med: R.med });
      step(3, '4. A baseline to beat', [['always say “' + (R.maj ? 'defaults' : 'repays') + '”', 'legend'], ['test accuracy ' + H().fx(R.base, 3) + ', recall of defaults 0', 'legend']],
        `The simplest rule, always predicting the majority (${R.maj ? 'default' : 'repaid'}), scores ${q(R.base, 3)} accuracy on the test rows and finds no defaults. Every model from here on is judged against this.`, { base: R.base });
      step(4, '5. A model', [['logistic regression on the 5 scaled columns', 'legend'], ['accuracy ' + H().fx(R.acc, 3) + ', AUC ' + H().fx(R.auc, 3), 'legend'], ['recall of defaults ' + H().fx(R.recall, 2) + ', precision ' + H().fx(R.precision, 2), 'legend']],
        `Start with a simple, readable model: logistic regression on the five scaled columns. On the test rows its accuracy is ${q(R.acc, 3)} against the baseline’s ${q(R.base, 3)}; its AUC is ${q(R.auc, 3)}; it catches ${q.pct(R.recall)} of the defaults with precision ${q(R.precision, 2)}.`, { acc: R.acc, base: R.base, auc: R.auc, recall: R.recall, precision: R.precision });
      step(5, '6. Validate', [['5-fold cross-validation on the training rows', 'legend'], ['fold accuracies ' + R.cv.map((v) => H().fx(v, 2)).join(', '), 'math'], ['mean ' + H().fx(R.cvMean, 3) + ' ± ' + H().fx(R.cvSd, 3), 'legend']],
        `Is that test score luck? Five-fold cross-validation on the training rows gives ${R.cv.map((v) => q(v, 2)).join(', ')}: mean ${q(R.cvMean, 3)}, spread ${q(R.cvSd, 3)}. The test accuracy ${q(R.acc, 3)} sits ${Math.abs(R.acc - R.cvMean) <= 2 * R.cvSd ? 'within' : 'outside'} two spreads of it.`, { cv: R.cv, mean: R.cvMean, sd: R.cvSd, acc: R.acc });
      const cmItems = P.confusion('pc:cm', 520, 270, R.cm, ['repaid', 'default'], '');
      step(6, '7. Look at the errors', [['rows: true outcome; columns: prediction', 'legend'], [R.cm[1][0] + ' defaults missed, ' + R.cm[0][1] + ' false alarms', 'strong'], ['next: read the missed rows, look for a pattern', 'legend']],
        `The confusion matrix shows where it fails: ${P.qpl(q, R.cm[1][0], 'default')} missed and ${P.qpl(q, R.cm[0][1], 'false alarm')}. Missed defaults are the costly kind; reading those rows one by one is how features, thresholds or data problems get found.`, { fn: R.cm[1][0], fp: R.cm[0][1] }, cmItems);
      const gA = R.fair.groups.find((g) => g.group === 'A'), gB = R.fair.groups.find((g) => g.group === 'B');
      step(7, '8. Fairness and robustness', [['recall of defaults: group A ' + H().fx(gA.tpr, 2) + ', group B ' + H().fx(gB.tpr, 2), 'legend'], ['flagged as risky: A ' + H().pct(gA.selectionRate) + ', B ' + H().pct(gB.selectionRate), 'legend'], ['incomes 20% lower: accuracy ' + H().fx(R.accShift, 3), 'legend']],
        `By group: the model flags ${q.pct(gA.selectionRate)} of group A and ${q.pct(gB.selectionRate)} of group B as risky, and finds ${q.pct(gA.tpr)} and ${q.pct(gB.tpr)} of their defaults; the equalised-odds gap is ${q(R.fair.equalizedOdds, 2)}, although the group was never an input (income differs between them). And if incomes fell by a fifth, accuracy would go from ${q(R.acc, 3)} to ${q(R.accShift, 3)}.`, { selA: gA.selectionRate, selB: gB.selectionRate, tprA: gA.tpr, tprB: gB.tpr, eo: R.fair.equalizedOdds, acc: R.acc, accShift: R.accShift });
      const top = R.imp.order[0];
      const bars = S.Kit.dt.hbars({ prefix: 'pc:ib', x: 440, y: 254, w: 300, rowH: 24, vals: R.imp.order.map((j) => Math.max(0, R.imp.mean[j])), names: R.imp.order.map((j) => NAMES[j]), fmt: (v) => H().fx(v, 3), nameW: 90, cls: (i) => (i ? 'p18imp' : 'p18top') });
      step(8, '9. Explain', [['permutation importance on the test rows', 'legend'], ['(drop in AUC when a column is shuffled)', 'legend'], ['top: ' + NAMES[top], 'strong']],
        `Which columns does it lean on? Shuffling ${NAMES[top]} costs ${q(R.imp.mean[top], 3)} of AUC, the most of any column, then ${NAMES[R.imp.order[1]]} at ${q(R.imp.mean[R.imp.order[1]], 3)}. An explanation like this lets a loan officer question the model, and a refused applicant ask why.`, { top, topImp: R.imp.mean[top], second: R.imp.mean[R.imp.order[1]] }, bars);
      step(9, '10. Deploy and monitor', [['share predicted to default:', 'legend'], ['test rows ' + H().pct(R.shareTest) + ', a fresh batch ' + H().pct(R.shareNew), 'legend'], ['a batch with incomes 20% lower ' + H().pct(R.sharePoor), 'strong']],
        `Once in use, watch what comes in. The model flags ${q.pct(R.shareTest)} of the test rows; a fresh batch of applicants gets ${q.pct(R.shareNew)}, close enough. A batch where incomes are a fifth lower gets ${q.pct(R.sharePoor)}: a jump like that is the signal to check the inputs, gather new labels and retrain.`, { shareTest: R.shareTest, shareNew: R.shareNew, sharePoor: R.sharePoor });
      for (const f of F.list) f.narrowView = null;
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The checklist', text: ['1 the question and its cost of errors', '2 audit: gaps, odd values, sensitive columns', '3 split first, then clean with train statistics', '4 a baseline   5 a simple model', '6 cross-validate   7 read the errors', '8 fairness by group, robustness to shift', '9 explain   10 monitor inputs and outputs'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'baseline', pick: 'option', once: true,
      ask: 'Will the logistic model beat the baseline’s accuracy by more than five points?',
      options: ['yes', 'no'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'model').meta; return m.acc - m.base > 0.05 ? 'yes' : 'no'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'model').say,
    }],
    quiz: [
      { q: 'Why fill missing values only after the split?', options: ['statistics from the test rows would leak into training', 'it is faster', 'the test rows have no gaps', 'it makes no difference'], answer: 0, why: 'The test set must stay unseen in every way.' },
      { q: 'The model never sees the group column, yet flags one group more often. How?', options: ['other columns, such as income, differ between the groups', 'it guesses the group at random', 'it cannot happen', 'the group was in the labels'], answer: 0, why: 'Proxies carry the group information into the model.' },
      { q: 'What should monitoring after deployment watch?', options: ['the inputs and the share of each prediction, against what was seen in testing', 'only the training loss', 'nothing, the model is fixed', 'the number of features'], answer: 0, why: 'Shifts in the incoming data show up there first, before labels arrive.' },
      { q: 'What is the baseline for?', options: ['a floor every real model must clearly beat', 'the final model', 'a fairness check', 'tuning the learning rate'], answer: 0, why: 'A model no better than the majority rule adds nothing.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
