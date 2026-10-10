// Part 17, robustness, fairness and privacy: the exercises. Defensive throughout: they measure margins, read stress
// tests, spot drift, count fairness gaps, pick mitigations and defences, and work out the pieces of private training;
// none of them builds an attack. Every number is worked out here by the engines (S.Metrics.fairness, S.Rob.reweigh,
// S.Data for the drift tables) from the small cases written out below, or quoted from a topic. Two tables would cost
// too much to recompute on every page load (they train networks); their values were printed by a scratch run of
// S.Rob.robustCurve and S.Rob.dpTradeoff on the topics' own models and are written out with that note.
(function (S) {
  'use strict';
  const C = S.Content, M = S.Metrics, R = S.Rob, D = S.Data;
  if (!C || !M || !R || !D || C.exercises.has('rb-x-margin')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => String(r2(v)), n3 = (v) => String(r3(v));
  const mn = (v) => String(v).replace(/^-/, '−');
  const strCells = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));

  // =========================================================================================================
  // Chapter 1: brittle models, and hardening them
  // =========================================================================================================
  const w = [2, -1], b = -0.5, xp = [0.5, 0.3], sc = w[0] * xp[0] + w[1] * xp[1] + b, dist = Math.abs(sc) / Math.hypot(...w);
  C.exercise({ id: 'rb-x-margin', chapter: 'rb-brittle', title: 'How far from the boundary?', level: 2, kind: 'number',
    text: ['A linear classifier says class 1 when s = 2x₁ − x₂ − 0.5 is positive. A point at (0.5, 0.3) is classified as class 1. How far is it from the boundary s = 0, measured as a straight-line distance? This is the smallest change of the input that could change its answer. Three decimal places.'],
    answer: r3(dist), tolerance: 0.0015,
    whyNot: { [n3(sc)]: 'That is the score s itself. Distance divides it by the length of w = (2, −1).', [n3(Math.abs(sc) / 3)]: 'That divides by |2| + |−1|, the size of w for changes of equal size in every coordinate. A straight-line distance divides by √(2² + 1²).', '*': 'Distance from a line: |s| / ‖w‖.' },
    hints: ['For a linear model the distance from a point to the boundary is |s| / ‖w‖.', 's = 2 × 0.5 − 0.3 − 0.5 = ' + n3(sc) + '; ‖w‖ = √5 = ' + n3(Math.hypot(...w)) + '.', n3(sc) + ' / ' + n3(Math.hypot(...w)) + ' = ' + n3(dist) + ': any change shorter than that keeps the answer. A robust model keeps such margins large for typical points.'],
    solution: n3(dist), wrong: [n3(sc), n3(Math.abs(sc) / 3)] });

  const bopts = ['Some test points sit close to the boundary, so a small change of input moves them across it', 'The model has overfitted, so its test accuracy is wrong', 'Networks are random, so answers change at random', 'The test set was too small'];
  C.exercise({ id: 'rb-x-why-brittle', chapter: 'rb-brittle', title: 'Accurate, yet brittle', level: 1, kind: 'choice',
    text: ['A network gets 99.5% of its test points right, yet a change of 0.063 in each coordinate turns one right answer wrong, where the data spans 3.39. What explains it?'],
    options: bopts, answer: 0,
    whyNot: { 1: 'Its test accuracy is real; accuracy just does not measure how close each answer was to flipping.', 2: 'A trained network gives the same answer every time; the change of input is what moves it.', 3: 'A larger test set would show the same: some points lie near any boundary.' },
    hints: ['Accuracy counts right answers. Does it say how close they were?', 'Look at where the fragile point was: the test point closest to the boundary.', 'Points near the boundary need only a small change to cross it. Robustness is about margins, which accuracy does not see.'],
    solution: 0, wrong: [1, 3] });

  C.exercise({ id: 'rb-x-smooth-vote', chapter: 'rb-brittle', title: 'A smoothed answer', level: 1, kind: 'number',
    text: ['Randomised smoothing asks a model about 32 noisy copies of one input. 9 copies get class 0 and the rest class 1. What share of the votes does the smoothed answer get? Three decimal places.'],
    answer: r3(23 / 32), tolerance: 0.0015,
    whyNot: { [n3(9 / 32)]: 'That is the share of the losing class.', 23: 'That is the count; give the share of the 32 votes.', '*': 'Count the votes for the winning class and divide by 32.' },
    hints: ['The smoothed answer is the class with the most votes.', '32 − 9 = 23 copies vote class 1.', '23 / 32 = ' + n3(23 / 32) + ': class 1. A share well above one half means a small change of input is unlikely to flip the vote.'],
    solution: n3(23 / 32), wrong: [n3(9 / 32), '23'] });

  const dtab = { table: [['defence', 'clean accuracy', 'stress-tested at ε = 0.20'], ['none', '99.5%', '84.5%'], ['smoothing, σ = 0.30', '98.5%', '85%'], ['smoothing, σ = 0.60', '91%', '80%'], ['adversarial training at ε = 0.20', '97.5%', '87.5%']] };
  const dopts = ['Adversarial training: the best stress-tested accuracy for 2 points of clean accuracy', 'Smoothing with σ = 0.60: the most noise', 'No defence: the best clean accuracy', 'Smoothing with σ = 0.30: no loss on either'];
  C.exercise({ id: 'rb-x-defence-table', chapter: 'rb-brittle', title: 'Read a table of defences', level: 2, kind: 'choice',
    text: ['The moons network of [Hardening a model](#topic=robust-defend), with each defence, on the same 200 test points:', dtab, 'The model will face inputs that someone may nudge by up to 0.20. Which setting should be used?'],
    options: dopts, answer: 0,
    whyNot: { 1: 'Heavy noise costs clean accuracy (91%) and does worse under stress (80%).', 2: 'Under nudges of 0.20 it keeps only 84.5%.', 3: 'It loses a point of clean accuracy and gains only half a point under stress here.' },
    hints: ['Which column matters for this use?', 'Compare the stress-tested column, then what each costs on clean inputs.', 'Adversarial training: 87.5% under stress, the best, for 97.5% clean, 2 points below no defence. Too much smoothing noise hurts both columns.'],
    solution: 0, wrong: [1, 2] });

  // S.Rob.robustCurve(P17.moonsModel(1), eps, {n: 200}): the plain network of robust-defend, printed by a scratch run
  const RC = [[0, 0.995], [0.05, 0.995], [0.1, 0.95], [0.15, 0.89], [0.2, 0.845], [0.25, 0.725], [0.3, 0.585], [0.35, 0.455], [0.4, 0.345]];
  C.exercise({ id: 'rb-x-robust-curve', chapter: 'rb-brittle', title: 'Sketch a stress-test curve', level: 2, kind: 'predict-curve',
    text: ['The plain moons network of [Hardening a model](#topic=robust-defend) is stress-tested: every test point is nudged by ε in the direction that hurts it most, and accuracy is measured. At ε = 0 it is 0.995, and at ε = 0.2 it is 0.845. Sketch the accuracy from ε = 0 to 0.4.'],
    plot: { xr: [0, 0.4], yr: [0, 1], xname: 'ε of the nudge', yname: 'test accuracy', given: [RC[0], RC[4]], note: 'accuracy under the worst nudge of size ε' },
    answer: RC, tolerance: 0.08, checks: ['falling'],
    hints: ['Small nudges move only the points closest to the boundary.', 'The larger ε, the more points are within reach of the boundary: the fall speeds up.', 'The measured curve: ' + RC.map(([e, a]) => e + ': ' + a).join(', ') + '. Flat at first, then falling ever faster.'],
    solution: RC, wrong: [RC.map(([e]) => [e, 0.995]), RC.map(([e]) => [e, r3(0.995 - 0.25 * e)])] });

  const nopts = ['Adversarial training, with checks that inputs stay in the range seen in training', 'Monitoring the inputs for drift', 'Cleaning the training labels', 'Training on more ordinary messages'];
  C.exercise({ id: 'rb-x-defend-nudge', chapter: 'rb-brittle', title: 'Choose a defence: deliberate nudges', level: 2, kind: 'choice',
    text: ['A spam filter works well, but senders now alter their messages slightly, a few characters at a time, to slip through. The changes are small and the messages still read the same to people. Which defence fits?'],
    options: nopts, answer: 0,
    whyNot: { 1: 'Monitoring may notice that inputs changed, but it does not make the filter steadier against small changes.', 2: 'The labels are not the problem; the inputs are chosen to sit just past the boundary.', 3: 'More ordinary data does not teach the model about worst-case small changes.' },
    hints: ['Is the problem in the labels, in the drift of the world, or in small, deliberate changes of input?', 'Deliberate small changes are what stress tests and adversarial training address.', 'Adversarial training (train on nudged copies) plus input checks such as normalising text before it reaches the model.'],
    solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 2: when the world moves
  // =========================================================================================================
  const tr = D.load('loans', { n: 600, seed: 1 }), nw = D.load('loans', { n: 300, seed: 9 });
  const COLS = ['income', 'debt ratio', 'years employed', 'missed payments', 'amount'];
  const batch = nw.X.map((r) => [r[0] * 0.8, r[1], r[2], r[3], r[4] + 4]);
  const stat = (X, j) => { const c = X.map((r) => r[j]), m = c.reduce((a, v) => a + v, 0) / c.length; return [m, Math.sqrt(c.reduce((a, v) => a + (v - m) ** 2, 0) / c.length)]; };
  const stTr = COLS.map((_, j) => stat(tr.X, j)), stNw = COLS.map((_, j) => stat(batch, j)), stOk = COLS.map((_, j) => stat(nw.X, j));
  const moved = COLS.filter((_, j) => Math.abs(stNw[j][0] - stTr[j][0]) / stTr[j][1] > 0.3);
  C.exercise({ id: 'rb-x-shift-spot', chapter: 'rb-shift', title: 'Spot the shift in a summary', level: 2, kind: 'set',
    text: ['A loan model was trained on 600 applications. A new month brings 300. The mean and standard deviation of each input:', { table: [['input', 'training mean', 'training sd', 'new mean', 'new sd'], ...COLS.map((c, j) => [c, n2(stTr[j][0]), n2(stTr[j][1]), n2(stNw[j][0]), n2(stNw[j][1])])] }, 'Which inputs have shifted by more than ordinary sampling noise? Give the set of their names.'],
    answer: '{' + moved.join(', ') + '}', caseless: true, placeholder: 'input names, like {a, b}',
    hints: ['Compare each new mean with the training mean, in units of the training standard deviation.', 'A sample of 300 moves a mean by only a small fraction of a standard deviation by chance. Income moved by ' + n2((stTr[0][0] - stNw[0][0]) / stTr[0][1]) + ' sd and amount by ' + n2((stNw[4][0] - stTr[4][0]) / stTr[4][1]) + ' sd.', '{' + moved.join(', ') + '}. The others moved by at most ' + n2(Math.max(...[1, 2, 3].map((j) => Math.abs(stNw[j][0] - stTr[j][0]) / stTr[j][1]))) + ' sd: noise. For comparison, the same month without the shift had income mean ' + n2(stOk[0][0]) + '.'],
    solution: '{' + moved.join(', ') + '}', wrong: ['{income}', '{income, amount, debt ratio}'] });

  const kopts = ['Label shift: the classes look the same, but their shares changed', 'Covariate shift', 'Concept shift', 'No shift'];
  C.exercise({ id: 'rb-x-shift-kind', chapter: 'rb-shift', title: 'Which kind of shift?', level: 2, kind: 'choice',
    text: ['A fault detector was trained when 50% of the logged machines were faulty. This year: within the faulty machines the readings look as they did, and within the healthy ones too; but 80% of the logged machines are now faulty, because only machines already suspected are logged. Which kind of shift is this?'],
    options: kopts, answer: 0,
    whyNot: { 1: 'Each class looks the same; what moved is how common each class is. (The overall mix of inputs moves only as a result.)', 2: 'The rule linking readings to faults is unchanged.', 3: 'The class shares went from 50% to 80%.' },
    hints: ['What stayed the same, and what changed?', 'Same look within each class; different shares of the classes.', 'Label shift. Remedy: estimate the new shares and move the cut-off; the rest of the model can stay.'],
    solution: 0, wrong: [1, 2] });

  const ropts = ['Gather fresh labelled data and retrain: only labels reveal the new rule', 'Move the cut-off', 'Weight training rows that resemble the new inputs', 'Raise the drift threshold'];
  C.exercise({ id: 'rb-x-shift-remedy', chapter: 'rb-shift', title: 'A remedy for a new rule', level: 2, kind: 'choice',
    text: ['A model that flags late deliveries worked well until a carrier changed its routes. The orders look just as before (same sizes, same distances, same days), but which ones arrive late has changed. What should be done?'],
    options: ropts, answer: 0,
    whyNot: { 1: 'A new cut-off helps when only the class shares change; here the rule itself changed.', 2: 'The inputs did not move, so there is nothing to reweight towards.', 3: 'The input monitor would not see this at all: the inputs look the same.' },
    hints: ['Did the inputs change, the class shares, or the rule?', 'Same inputs, new outcomes for them: concept shift.', 'Only new labelled examples show the new rule: gather them and retrain, and watch outcomes as they arrive.'],
    solution: 0, wrong: [1, 2] });

  const PR = [0.25, 0.25, 0.25, 0.25], PB = [0.1, 0.2, 0.3, 0.4];
  const psi = (ref, now) => ref.reduce((s, r, k) => s + (now[k] - r) * Math.log(now[k] / r), 0), pv = psi(PR, PB);
  C.exercise({ id: 'rb-x-psi', chapter: 'rb-shift', title: 'A drift score by hand', level: 3, kind: 'number',
    text: ['An input’s training values were cut into four bins holding a quarter each. A new batch puts 0.1, 0.2, 0.3 and 0.4 of its rows in them. What is the population stability index, PSI = Σ (b − r) · ln(b / r)? Three decimal places. Is it over the usual warning level of 0.2?'],
    answer: r3(pv), tolerance: 0.002,
    whyNot: { 0: 'The shares changed, so the terms are not zero: each term is positive whenever b ≠ r.', [n3(PR.reduce((s, r, k) => s + Math.abs(PB[k] - r), 0))]: 'That adds the gaps |b − r|; each must be multiplied by ln(b / r).', '*': 'Work out (b − r) · ln(b / r) for each of the four bins and add them.' },
    hints: ['r = 0.25 for every bin. Every term is positive: (b − r) and ln(b / r) always share a sign.', 'Bin 1: (0.1 − 0.25) · ln(0.4) = ' + n3((0.1 - 0.25) * Math.log(0.4)) + '. Bin 4: (0.4 − 0.25) · ln(1.6) = ' + n3(0.15 * Math.log(1.6)) + '.', 'Terms ' + PB.map((v) => n3((v - 0.25) * Math.log(v / 0.25))).join(', ') + '; PSI = ' + n3(pv) + ', just over 0.2: a warning worth checking.'],
    solution: n3(pv), wrong: ['0', n3(PR.reduce((s, r, k) => s + Math.abs(PB[k] - r), 0))] });

  const mopts = ['The rule may have changed while the inputs stayed the same: check outcomes and gather fresh labels', 'Nothing is wrong: the monitor raised no alarm', 'The threshold is too high: lower it to 0.01', 'The inputs drifted, but too slowly to see'];
  C.exercise({ id: 'rb-x-monitor-read', chapter: 'rb-shift', title: 'No alarm, falling accuracy', level: 2, kind: 'choice',
    text: [{ topic: 'shift-monitor', frame: 9, input: { scen: 'concept', speed: 0.12, thr: 0.2, seed: 1 }, caption: 'Eight batches: drift scores and the accuracy found when the outcomes arrived.' }, 'The input monitor never raised an alarm, yet accuracy fell to 41% at batch 7. What does that suggest?'],
    options: mopts, answer: 0,
    whyNot: { 1: 'Accuracy fell from 98% to 41%: something is badly wrong, just not in the inputs.', 2: 'The scores stayed at most 0.153, much like batches with no drift at all: a lower threshold would cry wolf without finding this.', 3: 'The input scores stayed at the level of sampling noise throughout.' },
    hints: ['Which part of the data does the drift score look at?', 'Inputs unchanged, outcomes changed.', 'Concept shift: the rule moved, which only the outcomes show. Watch accuracy on delayed labels as well as the inputs.'],
    solution: 0, wrong: [1, 2] });

  const fopts = ['Monitoring the incoming inputs for drift, and the outcomes as they arrive', 'Adversarial training', 'Cleaning the training labels', 'Randomised smoothing'];
  C.exercise({ id: 'rb-x-defend-drift', chapter: 'rb-shift', title: 'Choose a defence: a changing season', level: 1, kind: 'choice',
    text: ['A heating-demand model was trained on two winters of data. It will now run all year, and the customers it serves change as new districts join. Nobody is trying to fool it. Which defence fits best?'],
    options: fopts, answer: 0,
    whyNot: { 1: 'Nobody is nudging inputs on purpose; the world itself is changing.', 2: 'The labels were right when they were gathered; the problem is that new data may look different.', 3: 'Smoothing steadies answers against small changes, not against a different season.' },
    hints: ['Is anyone choosing the inputs to fool the model?', 'The inputs will move into regions the model rarely saw: summer, new districts.', 'Monitoring: compare incoming inputs with the training inputs (a drift score per input) and track accuracy as the true demand arrives, then retrain on new data.'],
    solution: 0, wrong: [1, 2] });

  const psiSol = 'function psi(ref, now) {\n  let s = 0;\n  for (let k = 0; k < ref.length; k++) s += (now[k] - ref[k]) * Math.log(now[k] / ref[k]);\n  return s;\n}';
  C.exercise({ id: 'rb-x-code-psi', chapter: 'rb-shift', title: 'Write the drift score', level: 2, kind: 'code',
    text: ['Write `psi(ref, now)`: two lists of bin shares of the same length (each list adds up to 1, every share above 0). Return the population stability index Σₖ (now[k] − ref[k]) · ln(now[k] / ref[k]).'],
    fn: 'psi', params: ['ref', 'now'], start: 'function psi(ref, now) {\n  \n}', tolerance: 1e-9,
    ref: psi,
    cases: [[PR, PB], [[0.5, 0.5], [0.5, 0.5]], [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.08, 0.1, 0.1, 0.12, 0.12, 0.13, 0.12, 0.13]], [[0.3, 0.7], [0.6, 0.4]]],
    hints: ['One pass over the bins, adding one term each.', 'The term is (now − ref) times the natural logarithm of now / ref: Math.log.', psiSol],
    solution: psiSol,
    wrong: ['function psi(ref, now) {\n  let s = 0;\n  for (let k = 0; k < ref.length; k++) s += Math.abs(now[k] - ref[k]);\n  return s;\n}', 'function psi(ref, now) {\n  let s = 0;\n  for (let k = 0; k < ref.length; k++) s += (now[k] - ref[k]) * Math.log10(now[k] / ref[k]);\n  return s;\n}'] });

  // =========================================================================================================
  // Chapter 3: fairness between groups
  // =========================================================================================================
  // per group: TP, FN (repaid, refused), FP (defaulted, approved), TN; 1 = repays (approve)
  const GC = { A: [40, 10, 15, 35], B: [18, 12, 6, 24] };
  const yT = [], yP = [], gp = [];
  for (const [g, [tp, fn, fp, tn]] of Object.entries(GC)) {
    const add = (k, t, p) => { for (let i = 0; i < k; i++) { yT.push(t); yP.push(p); gp.push(g); } };
    add(tp, 1, 1); add(fn, 1, 0); add(fp, 0, 1); add(tn, 0, 0);
  }
  const FR = M.fairness(yT, yP, gp), gA = FR.groups.find((x) => x.group === 'A'), gB = FR.groups.find((x) => x.group === 'B');
  const gcTable = { table: [['truth', 'approved', 'refused'], ...Object.entries(GC).flatMap(([g, c]) => [['group ' + g + ', repaid', String(c[0]), String(c[1])], ['group ' + g + ', defaulted', String(c[2]), String(c[3])]])] };
  const gcText = 'A lender’s model on its test applicants, with “approve” as the positive prediction and “repaid” as the positive truth:';
  const selCells = { 'group A|approval rate': r3(gA.selectionRate), 'group B|approval rate': r3(gB.selectionRate) };
  C.exercise({ id: 'rb-x-sel-rates', chapter: 'rb-fair', title: 'Approval rates per group', level: 1, kind: 'table',
    text: [gcText, gcTable, 'What is each group’s approval (selection) rate? Three decimal places.'],
    rows: ['group A', 'group B'], cols: ['approval rate'], corner: '', cells: selCells, tolerance: 0.0015,
    hints: ['Approved = repaid and approved + defaulted and approved.', 'Group A: 40 + 15 = 55 approved of ' + gA.n + '. Group B: 18 + 6 = 24 of ' + gB.n + '.', 'A: ' + n3(gA.selectionRate) + '; B: ' + n3(gB.selectionRate) + '.'],
    solution: strCells({ 'group A|approval rate': n3(gA.selectionRate), 'group B|approval rate': n3(gB.selectionRate) }),
    wrong: [strCells({ 'group A|approval rate': n3(gA.tpr), 'group B|approval rate': n3(gB.tpr) })] });

  C.exercise({ id: 'rb-x-dpd', chapter: 'rb-fair', title: 'Demographic parity difference', level: 1, kind: 'number',
    text: [gcText, gcTable, 'What is the demographic parity difference? Three decimal places.'],
    answer: r3(FR.demographicParity), tolerance: 0.0015,
    whyNot: { [n3(Math.abs(gA.tpr - gB.tpr))]: 'That is the gap in true-positive rates, part of equalised odds. Parity compares the approval rates of everyone in each group.', [n3(gB.selectionRate / gA.selectionRate)]: 'That is the ratio of the rates; the difference subtracts them.', '*': 'Approval rate of A minus approval rate of B.' },
    hints: ['Demographic parity compares approval rates, whoever would really repay.', 'Rates: A ' + n3(gA.selectionRate) + ', B ' + n3(gB.selectionRate) + '.', n3(gA.selectionRate) + ' − ' + n3(gB.selectionRate) + ' = ' + n3(FR.demographicParity) + '.'],
    solution: n3(FR.demographicParity), wrong: [n3(Math.abs(gA.tpr - gB.tpr)), n3(gB.selectionRate / gA.selectionRate)] });

  const rfCells = { 'group A|TPR': r3(gA.tpr), 'group A|FPR': r3(gA.fpr), 'group B|TPR': r3(gB.tpr), 'group B|FPR': r3(gB.fpr) };
  C.exercise({ id: 'rb-x-tpr-fpr', chapter: 'rb-fair', title: 'Error rates per group', level: 2, kind: 'table',
    text: [gcText, gcTable, 'For each group, the true-positive rate (approved among those who repaid) and the false-positive rate (approved among those who defaulted). Three decimal places.'],
    rows: ['group A', 'group B'], cols: ['TPR', 'FPR'], corner: '', cells: rfCells, tolerance: 0.0015,
    hints: ['TPR = repaid and approved / all who repaid. FPR = defaulted and approved / all who defaulted.', 'Group A: 40 / 50 and 15 / 50. Group B: 18 / 30 and 6 / 30.', 'A: TPR ' + n3(gA.tpr) + ', FPR ' + n3(gA.fpr) + '; B: TPR ' + n3(gB.tpr) + ', FPR ' + n3(gB.fpr) + '.'],
    solution: strCells(Object.fromEntries(Object.entries(rfCells).map(([k, v]) => [k, n3(v)]))),
    wrong: [strCells({ 'group A|TPR': n3(40 / 55), 'group A|FPR': n3(15 / 55), 'group B|TPR': n3(18 / 24), 'group B|FPR': n3(6 / 24) })] });

  C.exercise({ id: 'rb-x-eod', chapter: 'rb-fair', title: 'Equalised odds difference', level: 2, kind: 'number',
    text: [gcText, gcTable, 'What is the equalised-odds difference, the larger of the TPR gap and the FPR gap? Three decimal places.'],
    answer: r3(FR.equalizedOdds), tolerance: 0.0015,
    whyNot: { [n3(FR.fprDifference)]: 'That is the FPR gap. Take the larger of the two gaps.', [n3(FR.demographicParity)]: 'That is the parity difference, from approval rates of everyone.', [n3((FR.tprDifference + FR.fprDifference) / 2)]: 'That averages the gaps; the measure takes the larger one.' },
    hints: ['TPR gap = |TPR of A − TPR of B|; FPR gap the same with FPR.', 'TPR gap ' + n3(FR.tprDifference) + '; FPR gap ' + n3(FR.fprDifference) + '.', 'The larger is ' + n3(FR.equalizedOdds) + ': repayers in group B are approved much less often than in group A.'],
    solution: n3(FR.equalizedOdds), wrong: [n3(FR.fprDifference), n3(FR.demographicParity)] });

  const dpdSol = 'function dpd(yPred, group) {\n  const n = {}, sel = {};\n  for (let i = 0; i < yPred.length; i++) {\n    const g = group[i];\n    n[g] = (n[g] || 0) + 1;\n    sel[g] = (sel[g] || 0) + (yPred[i] === 1 ? 1 : 0);\n  }\n  const rates = Object.keys(n).map((g) => sel[g] / n[g]);\n  return Math.max(...rates) - Math.min(...rates);\n}';
  C.exercise({ id: 'rb-x-code-dpd', chapter: 'rb-fair', title: 'Write the demographic parity difference', level: 2, kind: 'code',
    text: ['Write `dpd(yPred, group)`: yPred holds 0 or 1 for each person (1 is the favourable decision), group holds each person’s group (a string or a number; there may be more than two groups). Return the largest group selection rate minus the smallest.'],
    fn: 'dpd', params: ['yPred', 'group'], start: 'function dpd(yPred, group) {\n  \n}', tolerance: 1e-9,
    ref: (p, g) => M.fairness(p, p, g).demographicParity,
    cases: [[yP, gp], [[1, 0, 1, 1, 0, 0], ['a', 'a', 'a', 'b', 'b', 'b']], [[1, 1, 0, 0, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2, 0, 1, 2]], [[0, 0, 1, 1], ['x', 'y', 'x', 'y']], [[1, 0, 0, 0, 0], ['p', 'q', 'q', 'q', 'q']]],
    hints: ['Count, per group, how many people there are and how many got 1.', 'A plain object or a Map keyed by the group works for any number of groups. Then take the largest rate minus the smallest.', dpdSol],
    solution: dpdSol,
    wrong: ['function dpd(yPred, group) {\n  const g0 = group[0];\n  let a = 0, na = 0, b = 0, nb = 0;\n  for (let i = 0; i < yPred.length; i++) { if (group[i] === g0) { na++; a += yPred[i]; } else { nb++; b += yPred[i]; } }\n  return a / na - b / nb;\n}', 'function dpd(yPred, group) {\n  const n = {}, sel = {};\n  for (let i = 0; i < yPred.length; i++) { const g = group[i]; n[g] = (n[g] || 0) + 1; sel[g] = (sel[g] || 0) + yPred[i]; }\n  const counts = Object.keys(n).map((g) => sel[g]);\n  return (Math.max(...counts) - Math.min(...counts)) / yPred.length;\n}'] });

  const gopts = ['No: other features, such as income, differ between the groups and carry the group’s signal', 'Yes: a model cannot use what it never sees', 'Yes, as long as accuracy stays the same', 'No: removing a column always makes a model less fair'];
  C.exercise({ id: 'rb-x-drop-group', chapter: 'rb-fair', title: 'Is a blind model fair?', level: 1, kind: 'choice',
    text: ['A lender removes the group column from its data and retrains. Its gaps between the groups are now guaranteed to be zero. Is that right?'],
    options: gopts, answer: 0,
    whyNot: { 1: 'In the topic, the parity difference was 0.088 with or without the group column: income stood in for it.', 2: 'Accuracy says nothing about how errors are shared between groups.', 3: 'Removing it does not make things worse by itself; it just does not remove the gap, and it takes away the means to measure it.' },
    hints: ['What else in the data differs between the groups?', 'A feature that differs between groups is a proxy for the group.', 'No. Proxies carry the signal; keep the group aside to measure the gaps, and mitigate them directly.'],
    solution: 0, wrong: [1, 2] });

  const pyT = [], pgp = [];
  [['A', 70, 30], ['B', 50, 50]].forEach(([g, pos, neg]) => { for (let i = 0; i < pos; i++) { pyT.push(1); pgp.push(g); } for (let i = 0; i < neg; i++) { pyT.push(0); pgp.push(g); } });
  const PF = M.fairness(pyT, pyT, pgp);
  C.exercise({ id: 'rb-x-perfect-dpd', chapter: 'rb-fair', title: 'A perfect model’s parity gap', level: 2, kind: 'number',
    text: ['In group A, 70 of 100 applicants would repay; in group B, 50 of 100. A perfect model approves exactly those who would repay. What is its demographic parity difference? Two decimal places. (Its equalised-odds difference is 0: it makes no errors.)'],
    answer: r2(PF.demographicParity), tolerance: 0.005,
    whyNot: { 0: 'It makes no errors, so equalised odds holds; but it approves 70% of A and 50% of B.', '*': 'Its approval rates are the true repayment rates of each group.' },
    hints: ['A perfect model approves every repayer and nobody else.', 'Approval rates: 0.70 in A, 0.50 in B.', '0.70 − 0.50 = ' + n2(PF.demographicParity) + ', while its equalised-odds difference is ' + PF.equalizedOdds + '. When base rates differ, the two measures cannot both be zero for a useful model.'],
    solution: n2(PF.demographicParity), wrong: ['0'] });

  const wopts = ['Equal opportunity: the same true-positive rate among qualified applicants in each group', 'Demographic parity: the same invitation rate in each group', 'Accuracy in each group', 'The same false-positive rate in each group'];
  C.exercise({ id: 'rb-x-which-metric', chapter: 'rb-fair', title: 'Which gap matters here?', level: 2, kind: 'choice',
    text: ['A model picks which job applicants are invited to interview. The employer’s concern, agreed with the people affected: a qualified applicant should have the same chance of an invitation whatever their group. Which measure matches that concern?'],
    options: wopts, answer: 0,
    whyNot: { 1: 'Parity looks at everyone invited, qualified or not; the concern is about qualified applicants.', 2: 'Accuracy mixes both kinds of error and can be equal while qualified applicants of one group are passed over.', 3: 'False positives are unqualified applicants invited; the concern is about the qualified.' },
    hints: ['Whose chances does the concern talk about?', 'Qualified applicants are the positives; an invitation is a positive prediction.', 'Equal opportunity: the TPR in each group, one half of equalised odds.'],
    solution: 0, wrong: [1, 2] });

  const RW = R.reweigh([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'].map((g) => (g === 'A' ? 0 : 1)));
  const wOf = (g, l) => RW.table.find((t) => t.group === g && t.label === l).weight;
  const rwCells = { 'A|repaid': r3(wOf(0, 1)), 'A|defaulted': r3(wOf(0, 0)), 'B|repaid': r3(wOf(1, 1)), 'B|defaulted': r3(wOf(1, 0)) };
  C.exercise({ id: 'rb-x-reweigh', chapter: 'rb-fair', title: 'Reweighing by hand', level: 2, kind: 'table',
    text: ['16 training rows: in group A, 6 repaid and 2 defaulted; in group B, 3 repaid and 5 defaulted. Reweighing gives every row of a cell the weight P(group) · P(outcome) / P(group, outcome). Work out the four weights, to three decimal places.'],
    rows: ['A', 'B'], cols: ['repaid', 'defaulted'], corner: 'group', cells: rwCells, tolerance: 0.0015,
    hints: ['P(A) = P(B) = 8 / 16. P(repaid) = 9 / 16, P(defaulted) = 7 / 16. P(A, repaid) = 6 / 16.', 'A, repaid: (0.5 × 9/16) / (6/16) = 4.5 / 6.', 'A repaid ' + n3(wOf(0, 1)) + ', A defaulted ' + n3(wOf(0, 0)) + ', B repaid ' + n3(wOf(1, 1)) + ', B defaulted ' + n3(wOf(1, 0)) + '. The rare cells, A defaulted and B repaid, are weighted up.'],
    solution: strCells(Object.fromEntries(Object.entries(rwCells).map(([k, v]) => [k, n3(v)]))),
    wrong: [strCells({ 'A|repaid': n3(1 / wOf(0, 1)), 'A|defaulted': n3(1 / wOf(0, 0)), 'B|repaid': n3(1 / wOf(1, 1)), 'B|defaulted': n3(1 / wOf(1, 0)) })] });

  const preOpts = ['Reweighing the training rows by group and outcome', 'A cut-off per group at decision time', 'Dropping the group column and doing nothing else', 'Raising one cut-off for everyone'];
  C.exercise({ id: 'rb-x-mitigate-pre', chapter: 'rb-fair', title: 'Which mitigation fits: no group at decision time', level: 2, kind: 'choice',
    text: ['A lender may use applicants’ group to audit and train its model, but by law the decision itself must not look at an applicant’s group. Its model has a parity gap it wants to narrow. Which mitigation fits?'],
    options: preOpts, answer: 0,
    whyNot: { 1: 'A cut-off per group needs the group at decision time, which is not allowed here.', 2: 'Proxies keep the gap; dropping the column changes little.', 3: 'One cut-off for everyone moves both groups together.' },
    hints: ['Where may the group be used: in training, at decision time, or both?', 'Only in training.', 'Reweighing acts before training: the trained model then needs no group to decide. (In the topic it took the parity difference from 0.088 to 0.042.)'],
    solution: 0, wrong: [1, 2] });

  const postOpts = ['A cut-off per group on the vendor’s scores', 'Reweighing the training rows', 'Retraining the model with a fairness penalty', 'Collecting more data'];
  C.exercise({ id: 'rb-x-mitigate-post', chapter: 'rb-fair', title: 'Which mitigation fits: a model you cannot retrain', level: 2, kind: 'choice',
    text: ['A council buys a scoring model from a vendor and cannot retrain it or see its training data. It receives a score for each applicant, knows each applicant’s group, and may use it to make outcomes fairer. Which mitigation is open to it?'],
    options: postOpts, answer: 0,
    whyNot: { 1: 'Reweighing changes the training, which the council cannot touch.', 2: 'A penalty in the loss also needs retraining.', 3: 'More data cannot be used without retraining either.' },
    hints: ['Which stage can the council still act on?', 'Only the scores, after the model.', 'A cut-off per group, chosen on held-out data for the measure agreed (equal approval rates, or equal TPR). Report the other gaps too.'],
    solution: 0, wrong: [1, 2] });

  const BS = [0.91, 0.84, 0.77, 0.7, 0.63, 0.58, 0.52, 0.47, 0.41, 0.35], want = 0.6, kB = Math.round(want * BS.length), cut = BS[kB - 1];
  C.exercise({ id: 'rb-x-cutoff', chapter: 'rb-fair', title: 'A cut-off for one group', level: 3, kind: 'number',
    text: ['Group A’s approval rate is ' + want + '. The ten applicants of group B have chances of repaying ' + BS.join(', ') + '. Applicants at or above the cut-off are approved. What is the highest cut-off that gives group B the same approval rate?'],
    answer: cut, tolerance: 0.0005,
    whyNot: { [BS[kB]]: 'That approves ' + (kB + 1) + ' of 10.', [BS[kB - 2]]: 'That approves only ' + (kB - 1) + ' of 10.', 0.5: 'With a cut-off of 0.5, ' + BS.filter((v) => v >= 0.5).length + ' of 10 are approved.', '*': 'Sort the chances and approve the top ' + kB + '.' },
    hints: ['A rate of ' + want + ' among 10 people means ' + kB + ' approved.', 'The chances are sorted already, highest first: approve the first ' + kB + '.', 'The ' + kB + 'th highest chance is ' + cut + ': with the cut-off there, exactly ' + kB + ' are approved. Any cut-off above it but at most ' + BS[kB - 2] + ' approves fewer.'],
    solution: String(cut), wrong: [String(BS[kB]), '0.5'] });

  // =========================================================================================================
  // Chapter 4: privacy and the training data
  // =========================================================================================================
  C.exercise({ id: 'rb-x-mi-advantage', chapter: 'rb-private', title: 'Measure a membership leak', level: 1, kind: 'number',
    text: ['An auditor checks a model: a threshold on its confidence flags 92% of the rows it was trained on as members, and 64% of rows it never saw. What is the membership advantage, TPR − FPR? Two decimal places.'],
    answer: 0.28, tolerance: 0.005,
    whyNot: { 0.92: 'That is the true-positive rate alone; subtract the rate at which unseen rows are flagged.', 0.64: 'That is the false-positive rate.', 1.56: 'Subtract, do not add.' },
    hints: ['Advantage = the share of members flagged − the share of non-members flagged.', '0.92 − 0.64.', '0.28: confidence reveals a good deal about who was in the training set. Zero would mean nothing.'],
    solution: '0.28', wrong: ['0.92', '1.56'] });

  const miOpts = ['A shallower tree', 'Larger leaves (more rows per leaf)', 'Training with DP-SGD', 'Growing the tree until every leaf is pure', 'Training for many more epochs'];
  C.exercise({ id: 'rb-x-mi-defence', chapter: 'rb-private', title: 'Shrinking the leak', level: 2, kind: 'choice',
    text: ['A fully grown tree leaks membership: advantage 0.306. Which changes would make it leak less? Choose all that apply.'],
    options: miOpts, answer: [0, 1, 2],
    whyNot: { 3: 'A pure leaf per training row is exactly memorisation.', 4: 'Training longer lets a model fit its training rows more closely: more memorisation.' },
    hints: ['The leak comes from behaving differently on training rows.', 'Anything that limits memorisation narrows that difference; so does private training.', 'A shallower tree, larger leaves and DP-SGD. In the topic the leak ran from 0.306 for the full tree down to 0.029 for logistic regression.'],
    solution: [0, 1, 2], wrong: [[0, 1, 2, 3], [3, 4], [2]] });

  const G3 = [[0.6, -0.8], [1.8, 2.4], [-0.3, 0.4]], CL = 1, clipped = G3.map((g) => { const nr = Math.hypot(...g), f = Math.min(1, CL / nr); return g.map((v) => v * f); });
  const clCells = {}; clipped.forEach((g, i) => { clCells['person ' + (i + 1) + '|g₁'] = r3(g[0]); clCells['person ' + (i + 1) + '|g₂'] = r3(g[1]); });
  C.exercise({ id: 'rb-x-clip', chapter: 'rb-private', title: 'Clip three gradients', level: 2, kind: 'table',
    text: ['Three people’s gradients in one DP-SGD step: (' + G3.map((g) => g.map(mn).join(', ')).join('), (') + '). Clip each to length at most C = 1: a longer gradient is scaled down to length exactly 1, keeping its direction; a shorter one is left alone. Three decimal places.'],
    rows: ['person 1', 'person 2', 'person 3'], cols: ['g₁', 'g₂'], corner: '', cells: clCells, tolerance: 0.0015,
    hints: ['Lengths: ' + G3.map((g) => n3(Math.hypot(...g))).join(', ') + '.', 'Only person 2 is longer than 1: divide by ' + n3(Math.hypot(...G3[1])) + '.', clipped.map((g, i) => 'person ' + (i + 1) + ': (' + g.map((v) => mn(n3(v))).join(', ') + ')').join('; ') + '. Now nobody moves the sum by more than 1.'],
    solution: strCells(Object.fromEntries(Object.entries(clCells).map(([k, v]) => [k, n3(v)]))),
    wrong: [strCells(Object.fromEntries(G3.flatMap((g, i) => g.map((v, j) => ['person ' + (i + 1) + '|g' + '₁₂'[j], n3(v / Math.hypot(...g))]))))] });

  C.exercise({ id: 'rb-x-dp-noise', chapter: 'rb-private', title: 'How much noise?', level: 1, kind: 'number',
    text: ['DP-SGD clips each gradient to length C = 1.5 and uses noise multiplier σ = 2. What is the standard deviation of the Gaussian noise added to each weight’s summed gradient?'],
    answer: 3, tolerance: 0.001,
    whyNot: { 2: 'That is σ alone; the noise is scaled by the clipping length too, σC.', 1.5: 'That is C alone.', 0.75: 'Multiply σ by C, do not divide.' },
    hints: ['The noise must hide one person, who can move the sum by at most C.', 'Its standard deviation is σ times C.', '2 × 1.5 = 3.'],
    solution: '3', wrong: ['2', '1.5'] });

  const eOpts = ['ε = 0.5', 'ε = 2', 'ε = 8', 'No noise (ε unbounded)'];
  C.exercise({ id: 'rb-x-dp-eps', chapter: 'rb-private', title: 'Which promise is strongest?', level: 1, kind: 'choice',
    text: ['Four models of the same data were trained with differential privacy at different settings. Which gives the strongest privacy promise to the people in the data?'],
    options: eOpts, answer: 0,
    whyNot: { 1: 'A larger ε allows the model to change more with one person’s data: a weaker promise.', 2: 'The largest finite ε here: the weakest promise of the three.', 3: 'Without noise there is no promise at all.' },
    hints: ['ε bounds how much one person’s data can change what the model does.', 'Smaller bound, stronger promise.', 'ε = 0.5, the smallest.'],
    solution: 0, wrong: [2, 3] });

  // S.Rob.dpTradeoff({sigmas: [0, 1, 2, 4, 8]}): the dp-sgd topic's own trade-off, printed by a scratch run
  const DPT = [[0, '∞', 0.807], [1, 3.48, 0.804], [2, 1.68, 0.802], [4, 0.83, 0.763], [8, 0.41, 0.691]];
  const tOpts = ['σ = 4: ε ≈ 0.83, accuracy 0.763', 'σ = 8: ε ≈ 0.41, accuracy 0.691', 'σ = 2: ε ≈ 1.68, accuracy 0.802', 'σ = 0: accuracy 0.807'];
  C.exercise({ id: 'rb-x-dp-tradeoff', chapter: 'rb-private', title: 'Read the privacy–accuracy trade-off', level: 2, kind: 'choice',
    text: ['The loans model trained with DP-SGD at five noise levels (mean test accuracy of three runs; ε is the teaching estimate at δ = 10⁻⁵):', { table: [['σ', 'ε', 'accuracy'], ...DPT.map((r) => r.map(String))] }, 'The data owner sets a budget of ε ≤ 1. Which setting should be used?'],
    options: tOpts, answer: 0,
    whyNot: { 1: 'It meets the budget too, but gives up another 7 points of accuracy for privacy beyond what was asked.', 2: 'ε ≈ 1.68 breaks the budget.', 3: 'No noise means no promise at all.' },
    hints: ['First keep only the rows that meet ε ≤ 1.', 'σ = 4 and σ = 8 meet it.', 'σ = 4: the most accurate setting within the budget, 0.763, against 0.807 with no privacy. Report both numbers.'],
    solution: 0, wrong: [1, 2] });

  // a tiny labelled set: two classes, one wrong label inside the other class
  const KP = [[0.2, 0.3], [0.5, 0.2], [0.3, 0.7], [0.7, 0.6], [0.4, 0.45], [1.6, 1.5], [1.9, 1.7], [1.5, 1.9], [2.1, 1.3], [1.8, 2.1], [0.55, 0.5]];
  const KL = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
  const flagged = KP.map((p, i) => { const nn = KP.map((q, j) => [Math.hypot(p[0] - q[0], p[1] - q[1]), j]).filter(([, j]) => j !== i).sort((a, c) => a[0] - c[0]).slice(0, 3); return nn.filter(([, j]) => KL[j] !== KL[i]).length >= 2 ? i : -1; }).filter((i) => i >= 0);
  C.exercise({ id: 'rb-x-knn-clean', chapter: 'rb-private', title: 'Find the suspicious labels', level: 2, kind: 'point',
    text: ['Eleven training points with their labels (colours). Cleaning by vote: a point is set aside when at least 2 of its 3 nearest other points carry a different label. One point here would be set aside: click it.'],
    plot: { points: KP, classes: KL, xr: [0, 2.4], yr: [0, 2.4], xname: 'x₁', yname: 'x₂', equal: true }, mode: 'pick', answer: flagged[0],
    hints: ['Look for a point whose colour differs from everything around it.', 'Each such point’s 3 nearest neighbours all sit in the other cluster.', 'The point at (' + KP[flagged[0]].join(', ') + '): it carries the label of the far cluster but lies inside the near one, so its 3 nearest neighbours all outvote it.'],
    solution: { i: flagged[0] }, wrong: [{ i: 4 }, { i: 5 }] });
  if (flagged.length !== 1) C.errors.push('rb-x-knn-clean: the vote should set aside exactly one point, not ' + flagged.length);

  const lOpts = ['Cleaning the labels: a neighbour vote, double-labelling a sample, and tracking which source each label came from', 'Adversarial training', 'Monitoring the inputs for drift', 'Differential privacy'];
  C.exercise({ id: 'rb-x-defend-labels', chapter: 'rb-private', title: 'Choose a defence: doubtful labels', level: 1, kind: 'choice',
    text: ['A plant-disease model is trained on photos labelled by hundreds of volunteers. Spot checks show that a few volunteers labelled carelessly, and one seems to have labelled a whole batch wrongly on purpose. Which defence fits?'],
    options: lOpts, answer: 0,
    whyNot: { 1: 'The problem is in the labels, not in nudged inputs.', 2: 'Nothing about the incoming photos has changed.', 3: 'Differential privacy limits each person’s influence on the model, which helps a little, but it does not find wrong labels.' },
    hints: ['Where is the problem: the labels, the inputs, or the world?', 'Wrong labels, some scattered, some in one batch.', 'Data cleaning: a vote of neighbours catches scattered wrong labels; tracking the source catches a bad batch; a second labeller on a sample measures the damage.'],
    solution: 0, wrong: [1, 2] });

  const mCells = { 'a filter that people try to slip past with small changes|defence': 'A', 'a model whose customers change month by month|defence': 'B', 'labels gathered from many unchecked sources|defence': 'C' };
  C.exercise({ id: 'rb-x-defence-match', chapter: 'rb-private', title: 'Match each risk to a defence', level: 2, kind: 'table',
    text: ['Three risks, three defences: **A** adversarial training (with input checks), **B** monitoring for shift (inputs and outcomes), **C** data cleaning (neighbour votes, source tracking). Write the letter of the defence that fits each risk best.'],
    rows: Object.keys(mCells).map((k) => k.split('|')[0]), cols: ['defence'], corner: 'risk', cells: mCells, cellKind: 'text',
    hints: ['Deliberate small changes, a changing world, or bad labels?', 'Small deliberate changes: train for them. A changing world: watch for it. Bad labels: clean them.', 'A, B, C, in that order.'],
    solution: mCells, wrong: [Object.fromEntries(Object.keys(mCells).map((k, i) => [k, 'BCA'[i]])), Object.fromEntries(Object.keys(mCells).map((k) => [k, 'A']))] });
})(globalThis.Nooral = globalThis.Nooral || {});
