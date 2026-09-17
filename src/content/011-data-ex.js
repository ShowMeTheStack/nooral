// Part 1, data and features: the exercises. Every number in a text, an answer or a hint is worked out here by the
// engines (S.P01's statistics and preparation steps, which the part's topics use; S.ML; S.Data) from the small
// datasets written out below, or by the workflow runner for the workflow task, so nothing is typed from memory.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, H = S.P01, WF = S.WF;
  if (!C || !ML || !H || !WF || C.exercises.has('da-x-task')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => String(r2(v)), n3 = (v) => String(r3(v));
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  const col = (v) => v.map((x) => [x]);
  const gap = '—';

  // =========================================================================================================
  // Chapter 1: tables and what is in them
  // =========================================================================================================
  const topts = ['minutes a bus arrives late at the last stop', 'the species of a meadow flower', 'whether a small loan is repaid', 'the monthly rent of a flat', 'which digit, 0 to 9, a drawing shows'];
  C.exercise({ id: 'da-x-task', chapter: 'da-tables', title: 'Classes or amounts', level: 1, kind: 'choice',
    text: ['Each option is the label of a table. Which of them make a **regression** task? Choose all that do.'],
    options: topts, answer: [0, 3],
    hints: ['Ask of each label: is it an amount, or a name from a short list?', 'Minutes and money are amounts. A species, repaid or not, and a digit are classes, even when the classes are written as numbers.', 'Regression: the minutes late and the monthly rent. The other three are classification; a digit is a class, since 7 is not "more digit" than 3.'],
    solution: [0, 3], wrong: [[0], [0, 3, 4]] });

  const peng = S.Data.load('penguins', {}), pcats = peng.features.filter((f) => f.type === 'category').map((f) => f.name);
  C.exercise({ id: 'da-x-types', chapter: 'da-tables', title: 'Which columns are categories', level: 1, kind: 'set',
    text: ['The penguin table has the feature columns ' + peng.features.map((f) => '`' + f.name + '`').join(', ') + ', and the label `species`.', 'Which **feature** columns hold categories rather than amounts? Write them as a set, like {a, b}.'],
    answer: '{' + pcats.join(', ') + '}', caseless: true,
    hints: ['An amount can be added and compared: millimetres and grams. A category is a name from a short list.', 'Lengths, depths and the body mass are measurements' + (peng.features.some((f) => f.name === 'year') ? ', and count the year as an amount here: it is a number on a scale' : '') + '. Look for the columns whose values are names.', 'The categories are ' + pcats.join(' and ') + ': an island’s name and a sex. The label, species, is a category too, but it is not a feature.'],
    solution: '{' + pcats.join(', ') + '}', wrong: ['{island}', '{' + pcats.join(', ') + ', species}'] });

  const FV = [2, 0, 7, 3, 12, 4, 1, 5, 30], FS = H.sorted(FV), five = H.five(FV);
  C.exercise({ id: 'da-x-five', chapter: 'da-tables', title: 'Quartiles by hand', level: 2, kind: 'table',
    text: ['Nine bus trips were late by these minutes: ' + FV.join(', ') + '.', 'Find the quartiles the usual way: sort the values and take the value at position (n − 1)·q, counting from 0.'],
    rows: ['Q1', 'median', 'Q3', 'IQR'], cols: ['minutes late'], corner: '',
    cells: { 'Q1|minutes late': five.q1, 'median|minutes late': five.med, 'Q3|minutes late': five.q3, 'IQR|minutes late': five.iqr }, tolerance: 0.01,
    hints: ['Sorted: ' + FS.join(', ') + '. With n = 9, the positions are 8 × 0.25 = 2, 8 × 0.5 = 4 and 8 × 0.75 = 6.', 'Counting from 0, position 2 holds ' + FS[2] + ', position 4 holds ' + FS[4] + ' and position 6 holds ' + FS[6] + '.', 'Q1 = ' + five.q1 + ', median = ' + five.med + ', Q3 = ' + five.q3 + ', so IQR = ' + five.q3 + ' − ' + five.q1 + ' = ' + five.iqr + '. The mean, ' + n2(H.mean(FV)) + ', is pulled up by the 30.'],
    solution: { 'Q1|minutes late': String(five.q1), 'median|minutes late': String(five.med), 'Q3|minutes late': String(five.q3), 'IQR|minutes late': String(five.iqr) },
    wrong: [{ 'Q1|minutes late': String(five.q1), 'median|minutes late': n2(H.mean(FV)), 'Q3|minutes late': String(five.q3), 'IQR|minutes late': String(five.iqr) }, { 'Q1|minutes late': String(FS[1]), 'median|minutes late': String(five.med), 'Q3|minutes late': String(FS[7]), 'IQR|minutes late': String(FS[7] - FS[1]) }] });

  const ld = S.Data.load('loans', {}), yrs = H.col(ld, 2), ymean = H.mean(yrs), ymed = H.median(yrs);
  const sopts = ['Skewed right: a long tail of large values', 'Skewed left: a long tail of small values', 'Roughly symmetric', 'Two separate humps'];
  C.exercise({ id: 'da-x-skew', chapter: 'da-tables', title: 'Mean against median', level: 1, kind: 'choice',
    text: ['The column "years employed" of ' + yrs.length + ' loan applicants has median ' + n2(ymed) + ' years and mean ' + n2(ymean) + ' years. What shape is it most likely to have?'],
    options: sopts, answer: 0,
    whyNot: { 1: 'A tail of small values would pull the mean below the median.', 2: 'In a symmetric column the mean and the median nearly agree.', 3: 'Two humps can have any mean and median; nothing here points to them.' },
    hints: ['The mean is pulled towards a long tail; the median is not.', 'Here the mean is ' + n2(ymean - ymed) + ' years above the median.', 'Skewed right: most applicants have worked a few years and a few have worked for decades, which pulls the mean up to ' + n2(ymean) + ' while the median stays at ' + n2(ymed) + '.'],
    solution: 0, wrong: [1, 2] });

  const CX = [1, 2, 3, 4, 5], CY = [2, 4, 5, 4, 5], pr = H.pearson(CX, CY), sp = H.spearman(CX, CY);
  C.exercise({ id: 'da-x-corr', chapter: 'da-tables', title: 'Pearson’s r by hand', level: 2, kind: 'number',
    text: ['Five students: hours of study x = ' + CX.join(', ') + ' and practice papers done y = ' + CY.join(', ') + '.', 'Compute Pearson’s r. Three decimal places.'],
    answer: r3(pr.r), tolerance: 0.0015,
    whyNot: { [n3(sp)]: 'That is Spearman’s r, computed on ranks. Pearson’s r uses the values themselves.', [n3(pr.sxy / CX.length)]: 'That is the covariance. Divide the sum of products by √(Σ(x − x̄)² · Σ(y − ȳ)²) instead.' },
    hints: ['The means are x̄ = ' + pr.mx + ' and ȳ = ' + pr.my + '. Centre both columns.', 'The centred x are −2, −1, 0, 1, 2 and the centred y are ' + CY.map((v) => H.f(v - pr.my, 2)).join(', ') + '. Σ products = ' + n2(pr.sxy) + ', Σ(x − x̄)² = ' + n2(pr.sxx) + ', Σ(y − ȳ)² = ' + n2(pr.syy) + '.', 'r = ' + n2(pr.sxy) + ' / √(' + n2(pr.sxx) + ' × ' + n2(pr.syy) + ') = ' + n3(pr.r) + '.'],
    solution: n3(pr.r), wrong: [n3(sp), n3(pr.sxy / CX.length)] });

  const hv = S.Data.load('harvest', {}), rr = H.pearson(H.col(hv, 0), hv.y).r;
  const ctopts = ['Yield hardly depends on rain', 'Yield depends on rain, but not along a straight line', 'More rain lowers the yield a little, steadily', 'The correlation must have been computed wrongly'];
  C.exercise({ id: 'da-x-corr-trap', chapter: 'da-tables', title: 'A weak r, a strong pattern', level: 2, kind: 'choice',
    text: ['On the valley harvest table, Pearson’s r between rain and yield is ' + n3(rr) + '. A scatter plot shows yield rising with rain up to a point and falling beyond it. What is the right reading?'],
    options: ctopts, answer: 1,
    whyNot: { 0: 'The scatter shows a strong pattern; r only measures straight-line association.', 2: 'The scatter rises and then falls: a single steady slope misdescribes it.', 3: 'The number is right; it just answers a narrower question than "is there a relation?".' },
    hints: ['What shape of relation does Pearson’s r measure?', 'A rise followed by a fall gives positive products on one side and negative ones on the other, and they cancel.', 'Yield depends strongly on rain, but along a curve: r = ' + n3(rr) + ' only says that a straight line fits badly. Look at the scatter, and model the curve (a squared term, bins or a tree).'],
    solution: 1, wrong: [0, 2] });

  // =========================================================================================================
  // Chapter 2: gaps and wild values
  // =========================================================================================================
  const DR = [[54, 2, gap, 3.1], [71, 3, 4, 2.2], [gap, 1, 2, 0.8], [88, 3, 1, 4.0], [62, gap, 3, gap], [45, 2, 0, 1.5], [93, 4, 2, 3.3], [67, 3, gap, 2.7]];
  const kept = DR.filter((r) => r.every((v) => v !== gap)).length, gaps = DR.flat().filter((v) => v === gap).length;
  C.exercise({ id: 'da-x-drop-count', chapter: 'da-clean', title: 'What dropping costs', level: 1, kind: 'number',
    text: ['Eight flats; a dash is a missing cell.', { table: [['area (m²)', 'rooms', 'floor', 'distance (km)'], ...DR.map((r) => r.map(String))] }, 'If every row with a missing cell is dropped, how many rows are left?'],
    answer: kept, whyNot: { [String(DR.length - gaps)]: 'There are ' + gaps + ' gaps, but one row holds two of them.', '*': 'Count the rows that have no dash at all.' },
    hints: ['A row survives only if all four of its cells are present.', 'There are ' + gaps + ' gaps, in ' + (DR.length - kept) + ' different rows.', kept + ' rows survive, and ' + (DR.length - kept) + ' go, with ' + ((DR.length - kept) * 4 - gaps) + ' good cells thrown away with them.'],
    solution: String(kept), wrong: [String(DR.length - gaps), String(DR.length - 1)] });

  const ITR = [12, NaN, 15, 9, NaN, 14], ITE = [NaN, 20, 18], fillTr = H.mean(ITR.filter(H.isNum)), fillAll = H.mean(ITR.concat(ITE).filter(H.isNum));
  const shw = (a) => a.map((v) => (H.isNum(v) ? String(v) : gap)).join(', ');
  C.exercise({ id: 'da-x-impute', chapter: 'da-clean', title: 'The value that fills a gap', level: 1, kind: 'number',
    text: ['A column of pump pressures, with gaps. Training rows: ' + shw(ITR) + '. Test rows: ' + shw(ITE) + '.', 'Mean imputation is used, done properly. What value fills the gap in the **test** row?'],
    answer: fillTr, tolerance: 0.001,
    whyNot: { [n3(fillAll)]: 'That mean includes the test values: their numbers leak into the preparation. The fill value is learned from the training rows only.', 0: 'A gap is not a zero: zero is a real pressure, and far from the typical one.' },
    hints: ['The imputer is fitted once, on the training rows, then applied to every row.', 'The present training values are ' + ITR.filter(H.isNum).join(', ') + '.', 'Their mean is ' + ITR.filter(H.isNum).join(' + ') + ' over 4 = ' + n3(fillTr) + ', and that fills the gaps in the training and the test rows alike.'],
    solution: n3(fillTr), wrong: [n3(fillAll), '0'] });

  const SV = [4, 6, NaN, 8, NaN, 10, 12], svp = SV.filter(H.isNum), svm = H.mean(svp), svf = SV.map((v) => (H.isNum(v) ? v : svm));
  const sdB = H.sd(svp), sdA = H.sd(svf);
  const iopts = ['The mean stays at ' + n2(svm) + ' and the sd falls from ' + n2(sdB) + ' to ' + n2(sdA), 'The mean and the sd stay the same', 'The mean changes and the sd grows', 'The mean stays and the sd grows'];
  C.exercise({ id: 'da-x-impute-sd', chapter: 'da-clean', title: 'What filling does to a spread', level: 2, kind: 'choice',
    text: ['A column holds ' + shw(SV) + '. Both gaps are filled with the mean of the present values. Using the population standard deviation (divide by n), what happens to the column’s mean and sd?'],
    options: iopts, answer: 0,
    whyNot: { 1: 'Two more values now sit exactly at the mean: they add rows but no spread.', 2: 'Values added at the mean cannot move the mean.', 3: 'Values at the mean add nothing to the sum of squared distances, but they do add to n.' },
    hints: ['The present values are ' + svp.join(', ') + ', mean ' + n2(svm) + '.', 'Their squared distances from the mean add up to ' + n2(svp.reduce((s, v) => s + (v - svm) ** 2, 0)) + '. Filling adds two values at distance 0.', 'Before: √(' + n2(svp.reduce((s, v) => s + (v - svm) ** 2, 0)) + ' / 5) = ' + n2(sdB) + '. After: √(' + n2(svp.reduce((s, v) => s + (v - svm) ** 2, 0)) + ' / 7) = ' + n2(sdA) + '. The mean stays at ' + n2(svm) + ': the column looks more certain than it is.'],
    solution: 0, wrong: [1, 3] });

  const hg = { data: { set: 'harvest', n: 240, seed: 3, opts: { missing: 0.15 } }, hidden: { seed: 33, n: 500 }, metric: 'rmse', target: 0.64 };
  C.exercise(Object.assign({ id: 'da-x-wf-gaps', chapter: 'da-clean', title: 'A harvest table with holes', level: 3, kind: 'workflow',
    text: ['The co-operative’s records have gaps: about 15% of the cells in every column were never written down, in the training rows and in the 500 hidden plots alike. Build a workflow with a decision tree no deeper than 4 that predicts the yield with RMSE at most 0.64 tonnes per hectare on the hidden plots.'],
    task: 'regression',
    limits: { blocks: ['split', 'clean', 'impute', 'encode', 'tree', 'train', 'evaluate'], models: ['tree'], params: { tree: { maxDepth: [1, 4] } } },
    hints: ['Run the plainest workflow first: Train will say what it cannot handle.', 'Dropping every row with a gap throws away nearly half of them: with 4 feature columns and 15% gaps per cell, only about 0.85⁴ ≈ 52% of rows are complete. Fill the gaps instead.', 'Split → Impute (mean) → Encode → Decision tree with max depth 4 → Train → Evaluate: RMSE 0.615 on the hidden plots, against 0.682 when the rows with gaps are dropped.'],
    solution: pipe(hg.data, 'tree', { maxDepth: 4 }, [['impute', { strategy: 'mean' }], ['encode', {}]]),
    wrong: [pipe(hg.data, 'tree', { maxDepth: 4 }, [['clean', { mode: 'rows' }], ['encode', {}]]), pipe(hg.data, 'tree', { maxDepth: 2 }, [['impute', { strategy: 'mean' }], ['encode', {}]])] }, hg));

  const OV = [21, 24, 25, 22, 27, 23, 26, 95, 24], of = H.five(OV);
  const ocells = { 'Q1|stem height': of.q1, 'Q3|stem height': of.q3, 'IQR|stem height': of.iqr, 'lower fence|stem height': of.loF, 'upper fence|stem height': of.hiF };
  C.exercise({ id: 'da-x-fences', chapter: 'da-clean', title: 'The box-plot fences', level: 2, kind: 'table',
    text: ['Nine stem heights in cm: ' + OV.join(', ') + '.', 'Find Q1 and Q3 (sorted, positions (n − 1)·q counting from 0), the IQR, and the fences 1.5 × IQR beyond the box.'],
    rows: ['Q1', 'Q3', 'IQR', 'lower fence', 'upper fence'], cols: ['stem height'], corner: '', cells: ocells, tolerance: 0.01,
    hints: ['Sorted: ' + H.sorted(OV).join(', ') + '. With n = 9, Q1 is at position 2 and Q3 at position 6.', 'Q1 = ' + of.q1 + ' and Q3 = ' + of.q3 + ', so the IQR is ' + of.iqr + ' and 1.5 × IQR = ' + 1.5 * of.iqr + '.', 'Fences: ' + of.q1 + ' − ' + 1.5 * of.iqr + ' = ' + of.loF + ' and ' + of.q3 + ' + ' + 1.5 * of.iqr + ' = ' + of.hiF + '. Only the 95 lies outside.'],
    solution: Object.fromEntries(Object.entries(ocells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.fromEntries(Object.entries(ocells).map(([k, v]) => [k, String(k.startsWith('lower fence') ? of.q1 - of.iqr : k.startsWith('upper fence') ? of.q3 + of.iqr : v)]))] });

  const zm = H.mean(OV), zs = H.sd(OV), zhi = zm + 3 * zs;
  C.exercise({ id: 'da-x-zfence', chapter: 'da-clean', title: 'An outlier hiding behind its own spread', level: 3, kind: 'number',
    text: ['The same nine stem heights: ' + OV.join(', ') + '. The other common rule flags values more than 3 standard deviations from the mean.', 'Where is its upper fence, mean + 3 sd, using the population sd (divide by n)? One decimal place.'],
    answer: r2(zhi), tolerance: 0.06,
    whyNot: { [String(of.hiF)]: 'That is the box-plot fence, Q3 + 1.5 × IQR.', [n2(zm)]: 'That is the mean alone; add 3 standard deviations.' },
    hints: ['The mean is ' + n2(zm) + ': the 95 alone pulls it far above the other eight values.', 'The squared distances from the mean add up to ' + n2(OV.reduce((s, v) => s + (v - zm) ** 2, 0)) + '; divide by 9 and take the root: sd = ' + n2(zs) + '.', 'Upper fence = ' + n2(zm) + ' + 3 × ' + n2(zs) + ' = ' + H.f(zhi, 1) + '. The 95 lies inside it: the outlier stretched the sd so much that the rule cannot see it, while the box-plot fence, ' + of.hiF + ', catches it.'],
    solution: H.f(zhi, 1), wrong: [String(of.hiF), n2(zm)] });

  const aopts = ['Keep them as they are: they are the signal the model must learn', 'Clip them to the fences so they do not dominate', 'Drop the rows: outliers are errors', 'Replace them with the column mean'];
  C.exercise({ id: 'da-x-outlier-act', chapter: 'da-clean', title: 'Outliers that matter', level: 2, kind: 'choice',
    text: ['The box-plot rule flags 14 pump temperatures as outliers. The maintenance log shows that all 14 minutes were real faults. You are building a model to detect faults. What should you do with these values?'],
    options: aopts, answer: 0,
    whyNot: { 1: 'Clipping pulls the fault readings back towards normal ones: exactly the difference the model needs is erased.', 2: 'These are not errors: the log confirms them. Dropping them removes the faults from the training data.', 3: 'Filling them with the mean makes every fault look like a normal minute.' },
    hints: ['Is a flagged value a mistake, or a real event?', 'The log says the readings are true, and they mark the very thing the model must recognise.', 'Keep them: here the outliers are the signal. Clip or drop only values that are errors or that should not dominate a fit.'],
    solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 3: scales, codes and bins
  // =========================================================================================================
  const ZT = [4, 8, 6, 2, 10], zsc = H.scaler(col(ZT), 'standard'), zq = 9, zv = zsc.apply([[zq]])[0][0];
  C.exercise({ id: 'da-x-zscore', chapter: 'da-transform', title: 'One value, standardised', level: 1, kind: 'number',
    text: ['A standard scaler is fitted on the training values ' + ZT.join(', ') + ' (population sd: divide by n). A new row has the value ' + zq + '. What does it become? Three decimal places.'],
    answer: r3(zv), tolerance: 0.0015,
    whyNot: { [n3((zq - zsc.centre[0]) / H.sd(ZT, 1))]: 'That divides by the sample sd (n − 1). The scaler uses the population sd, dividing by n.', [n3(zq - zsc.centre[0])]: 'That only subtracts the mean; divide by the sd too.' },
    hints: ['z = (x − mean) / sd, with the mean and sd of the training values.', 'The mean is ' + zsc.centre[0] + ' and the sd is √((' + ZT.map((v) => (v - zsc.centre[0]) ** 2).join(' + ') + ') / 5) = ' + n3(zsc.scale[0]) + '.', '(' + zq + ' − ' + zsc.centre[0] + ') / ' + n3(zsc.scale[0]) + ' = ' + n3(zv) + '.'],
    solution: n3(zv), wrong: [n3((zq - zsc.centre[0]) / H.sd(ZT, 1)), n3(zq - zsc.centre[0])] });

  const MT = [20, 35, 60, 45, 30], msc = H.scaler(col(MT), 'minmax'), mq = 70, mv = msc.apply([[mq]])[0][0];
  C.exercise({ id: 'da-x-minmax-test', chapter: 'da-transform', title: 'Beyond the training range', level: 2, kind: 'number',
    text: ['A min-max scaler is fitted on the training values ' + MT.join(', ') + '. A test row has the value ' + mq + '. What does the scaler turn it into? Two decimal places.'],
    answer: r2(mv), tolerance: 0.006,
    whyNot: { 1: 'The scaler does not clip, and it is not refitted with the test row: a value above the training maximum lands above 1.', '*': 'Use the training minimum and range: (x − min) / (max − min).' },
    hints: ['z = (x − min) / (max − min), with the training min and max.', 'Training min = ' + msc.centre[0] + ', range = ' + msc.scale[0] + '.', '(' + mq + ' − ' + msc.centre[0] + ') / ' + msc.scale[0] + ' = ' + n2(mv) + '. New rows can fall outside [0, 1]: that is expected, not an error.'],
    solution: n2(mv), wrong: ['1', '0.5'] });

  const scopts = ['Standard scaling', 'Min-max scaling', 'Robust scaling (median and IQR)', 'No scaling'];
  C.exercise({ id: 'da-x-scaler-outliers', chapter: 'da-transform', title: 'The scaler for a column with outliers', level: 1, kind: 'choice',
    text: ['A column of incomes has a few enormous values, which are real. k-nearest neighbours will measure distances on it. Which scaler keeps the ordinary incomes well spread out?'],
    options: scopts, answer: 2,
    whyNot: { 0: 'The enormous values inflate the sd, so the ordinary incomes end up squeezed close together.', 1: 'The largest income sets the range: every ordinary income lands in a narrow band near 0.', 3: 'Unscaled incomes would decide every distance on their own.' },
    hints: ['Which of the three uses statistics that a few extreme values cannot move?', 'The median and the interquartile range depend only on the middle of the column.', 'Robust scaling: (x − median) / IQR. The ordinary incomes keep a spread of about one unit; the enormous ones simply land far out.'],
    solution: 2, wrong: [0, 1] });

  const mmRef = (train, values) => { const s = H.scaler(col(train), 'minmax'); return s.apply(col(values)).map((r) => r[0]); };
  const mmSol = 'function minMax(train, values) {\n  const lo = Math.min(...train), hi = Math.max(...train);\n  const range = hi - lo || 1;\n  return values.map((v) => (v - lo) / range);\n}';
  C.exercise({ id: 'da-x-code-minmax', chapter: 'da-transform', title: 'Write min-max scaling', level: 1, kind: 'code',
    text: ['Write `minMax(train, values)`: learn the minimum and the range (max − min) of the numbers in `train`, and return `values` scaled with them, (v − min) / range. If every training value is the same, use a range of 1, so nothing divides by zero.'],
    fn: 'minMax', params: ['train', 'values'], start: 'function minMax(train, values) {\n  \n}', tolerance: 1e-9,
    ref: mmRef, cases: [[[20, 35, 60, 45, 30], [20, 60, 40]], [[20, 35, 60, 45, 30], [70, 10]], [[-2, 3, 8], [-2, 0.5, 8]], [[5, 5, 5], [5, 7]], [[0.1, 0.4], [0.25, 1]]],
    hints: ['Math.min(...train) and Math.max(...train) give the two ends.', 'Map every value through (v − lo) / (hi − lo), with the range replaced by 1 when it is 0.', mmSol],
    solution: mmSol,
    wrong: ['function minMax(train, values) {\n  const lo = Math.min(...values), hi = Math.max(...values);\n  return values.map((v) => (v - lo) / (hi - lo || 1));\n}', 'function minMax(train, values) {\n  const lo = Math.min(...train), hi = Math.max(...train);\n  return values.map((v) => Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1))));\n}'] });

  const stRef = (train, test) => { const s = H.scaler(train, 'standard'); return [s.apply(train), s.apply(test)]; };
  const stSol = 'function standardise(train, test) {\n  const d = train[0].length, mean = [], sd = [];\n  for (let j = 0; j < d; j++) {\n    const c = train.map((r) => r[j]);\n    const m = c.reduce((a, b) => a + b, 0) / c.length;\n    const v = c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length;\n    mean.push(m); sd.push(Math.sqrt(v) || 1);\n  }\n  const apply = (X) => X.map((r) => r.map((x, j) => (x - mean[j]) / sd[j]));\n  return [apply(train), apply(test)];\n}';
  C.exercise({ id: 'da-x-code-standard', chapter: 'da-transform', title: 'Write a standard scaler', level: 3, kind: 'code',
    text: ['Write `standardise(train, test)`. Both are tables: arrays of rows, each row an array of numbers. Learn the mean and the population standard deviation (divide by n) of each column of `train`, then return `[train scaled, test scaled]`, each value replaced by (x − mean) / sd of its column. A column with sd 0 uses 1 instead.', 'Nothing about `test` may be used to learn the numbers.'],
    fn: 'standardise', params: ['train', 'test'], start: 'function standardise(train, test) {\n  \n}', tolerance: 1e-9,
    ref: stRef, cases: [[[[1, 10], [2, 20], [3, 30]], [[4, 40]]], [[[0, 5], [4, 5], [8, 5], [4, 5]], [[2, 5], [12, 6]]], [[[1.5, -2, 100], [2.5, 0, 300], [3.5, 2, 200]], [[2, 1, 250], [0, 0, 0]]]],
    hints: ['Work column by column: for column j collect train.map((r) => r[j]).', 'Mean = sum / n; sd = √(Σ (x − mean)² / n), replaced by 1 when it is 0. Then map every row of both tables.', stSol],
    solution: stSol,
    wrong: ['function standardise(train, test) {\n  const all = train.concat(test), d = train[0].length, mean = [], sd = [];\n  for (let j = 0; j < d; j++) {\n    const c = all.map((r) => r[j]);\n    const m = c.reduce((a, b) => a + b, 0) / c.length;\n    mean.push(m); sd.push(Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length) || 1);\n  }\n  const apply = (X) => X.map((r) => r.map((x, j) => (x - mean[j]) / sd[j]));\n  return [apply(train), apply(test)];\n}',
      'function standardise(train, test) {\n  const d = train[0].length, mean = [], sd = [];\n  for (let j = 0; j < d; j++) {\n    const c = train.map((r) => r[j]);\n    const m = c.reduce((a, b) => a + b, 0) / c.length;\n    mean.push(m); sd.push(Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / (c.length - 1)) || 1);\n  }\n  const apply = (X) => X.map((r) => r.map((x, j) => (x - mean[j]) / sd[j]));\n  return [apply(train), apply(test)];\n}'] });

  const eopts = ['One-hot: a 0/1 column per colour', 'Ordinal codes in alphabetical order', 'Ordinal codes in order of popularity', 'Leave the names as they are'];
  C.exercise({ id: 'da-x-encode-names', chapter: 'da-transform', title: 'Encoding a colour', level: 1, kind: 'choice',
    text: ['A table of used cars has a column "colour" with the values red, blue, silver, white and black. A linear model will predict the price. How should the colour be encoded?'],
    options: eopts, answer: 0,
    whyNot: { 1: 'Codes 0 to 4 make the model treat the colours as points on a line, black to white in equal steps, with one slope across them.', 2: 'Popularity is still not an amount along which price changes in equal steps.', 3: 'A linear model can only multiply numbers by weights.' },
    hints: ['Do the colours have a natural order with equal steps between them?', 'They are names. Each may have its own effect on the price.', 'One-hot: five 0/1 columns (or four, with an intercept), so each colour gets its own weight.'],
    solution: 0, wrong: [1, 3] });

  const oopts = ['Ordinal codes in size order: S 0, M 1, L 2, XL 3', 'One-hot, because sizes are categories', 'Ordinal codes in alphabetical order', 'Drop the column'];
  C.exercise({ id: 'da-x-encode-order', chapter: 'da-transform', title: 'Encoding a size', level: 2, kind: 'choice',
    text: ['A shop predicts how much fabric a T-shirt takes. The column "size" holds S, M, L or XL, and each size up takes about the same amount more. Which encoding fits a linear model best with the fewest columns?'],
    options: oopts, answer: 0,
    whyNot: { 1: 'One-hot works, with four columns, but it throws away the order the model could use with one weight.', 2: 'Alphabetical order is L, M, S, XL: the codes would no longer follow the sizes.', 3: 'Size is the most useful column here.' },
    hints: ['Here the categories do have a real order.', 'And the steps between neighbouring sizes are about equal, which is what one weight on a code assumes.', 'Ordinal codes in size order: one column, one weight, and each size up adds the same amount, as the shop says.'],
    solution: 0, wrong: [1, 2] });

  const HT = ['gas', 'oil', 'heat pump', 'gas', 'heat pump'], HC = HT.slice().sort().filter((v, i, a) => a.indexOf(v) === i);
  const OH = HT.map((v) => HC.map((c) => (c === v ? 1 : 0)));
  C.exercise({ id: 'da-x-onehot', chapter: 'da-transform', title: 'One-hot by hand', level: 2, kind: 'matrix',
    text: ['Five flats and their heating: ' + HT.join(', ') + '. Fill in their one-hot encoding, with the categories in alphabetical order as columns.'],
    rows: HT.map((v, i) => 'flat ' + (i + 1) + ': ' + v), cols: HC, corner: 'flat \\ column', answer: OH, tolerance: 0,
    hints: ['Each row gets a 1 in the column of its own category and 0 everywhere else.', 'So every row adds up to exactly 1.', 'Rows: ' + OH.map((r) => r.join(' ')).join('; ') + '.'],
    solution: OH.map((r) => r.map(String)), wrong: [HT.map((v, i) => HC.map((c, j) => String(j === i % 3 ? 1 : 0))), OH.map((r) => r.map((x) => String(1 - x)))] });

  const OC = [['day', 7], ['soil', 3], ['balcony', 2]], onum = 2, ocount = onum + OC.reduce((s, [, k]) => s + k - 1, 0), ofull = onum + OC.reduce((s, [, k]) => s + k, 0);
  C.exercise({ id: 'da-x-onehot-count', chapter: 'da-transform', title: 'How many columns after one-hot', level: 2, kind: 'number',
    text: ['A table has 2 numeric columns and 3 category columns: ' + OC.map(([n, k]) => n + ' (' + k + ' values)').join(', ') + '. Every category column is one-hot encoded and, since the model has an intercept, one column of each is dropped. How many columns does the model get?'],
    answer: ocount,
    whyNot: { [String(ofull)]: 'That keeps every one-hot column. With an intercept, one column per category is dropped.', 5: 'That is the table before encoding.', '*': 'Count k − 1 columns for a category with k values, plus the numeric columns.' },
    hints: ['A category with k values gives k one-hot columns; dropping one leaves k − 1.', 'Day: 6, soil: 2, balcony: 1.', onum + ' + 6 + 2 + 1 = ' + ocount + ' columns.'],
    solution: String(ocount), wrong: [String(ofull), '5'] });

  const BV = [1, 2, 2, 3, 4, 7, 8, 9, 13, 16], bh = H.histogram(BV, 3);
  const bcells = {}; bh.counts.forEach((c, i) => { bcells['bin ' + (i + 1) + '|from'] = bh.edges[i]; bcells['bin ' + (i + 1) + '|to'] = bh.edges[i + 1]; bcells['bin ' + (i + 1) + '|count'] = c; });
  C.exercise({ id: 'da-x-bins', chapter: 'da-transform', title: 'Three equal-width bins', level: 2, kind: 'table',
    text: ['Ten values: ' + BV.join(', ') + '. Cut their range into 3 bins of equal width and count the values in each. A value on an edge belongs to the bin on its right, except the maximum, which belongs to the last bin.'],
    rows: ['bin 1', 'bin 2', 'bin 3'], cols: ['from', 'to', 'count'], corner: '', cells: bcells, tolerance: 0.01,
    hints: ['The range runs from ' + BV[0] + ' to ' + BV[BV.length - 1] + ', so each bin is ' + n2((BV[BV.length - 1] - BV[0]) / 3) + ' wide.', 'Edges: ' + bh.edges.map(n2).join(', ') + '.', 'Counts: ' + bh.counts.join(', ') + '. Quantile bins would hold about 3 values each instead, with uneven widths.'],
    solution: Object.fromEntries(Object.entries(bcells).map(([k, v]) => [k, n2(v)])),
    wrong: [Object.fromEntries(Object.entries(bcells).map(([k, v]) => [k, k.endsWith('count') ? String(k.startsWith('bin 1') ? 3 : k.startsWith('bin 2') ? 4 : 3) : n2(v)]))] });

  // =========================================================================================================
  // Chapter 4: making, choosing and leaking features
  // =========================================================================================================
  const PR = [[2, 3], [-1, 4]], pf = ML.PolynomialFeatures({ degree: 2, includeBias: false }).fit(PR), PT = pf.transform(PR);
  const pnames = pf.names.map((n) => n.replace(/x(\d)/g, (m, d) => 'x' + '₁₂₃₄'[+d]).replace(/\^2/g, '²').replace(/ /g, '·'));
  C.exercise({ id: 'da-x-poly-row', chapter: 'da-features', title: 'Degree-2 features of a row', level: 2, kind: 'matrix',
    text: ['Polynomial features of degree 2 (without the constant 1) turn a row (x₁, x₂) into the columns ' + pnames.join(', ') + '. Fill them in for two rows.'],
    rows: PR.map((r) => '(' + r.map((v) => H.f(v)).join(', ') + ')'), cols: pnames, corner: 'row \\ column', answer: PT, tolerance: 0,
    hints: ['Each column is a product: x₁·x₂ multiplies the two values.', 'For (2, 3): 2, 3, 2² = 4, 2 × 3 = 6, 3² = 9.', 'Rows: ' + PT.map((r) => r.join(', ')).join('; ') + '. Note (−1)² = 1 and −1 × 4 = −4.'],
    solution: PT.map((r) => r.map(String)), wrong: [PT.map((r, i) => r.map((v, j) => String(j === 3 ? PR[i][0] + PR[i][1] : v))), PT.map((r, i) => r.map((v, j) => String(j === 2 ? 2 * PR[i][0] : v)))] });

  const pc4 = ML.PolynomialFeatures({ degree: 2, includeBias: false }).fit([[1, 2, 3, 4]]).combos.length;
  C.exercise({ id: 'da-x-poly-count', chapter: 'da-features', title: 'How fast columns multiply', level: 2, kind: 'number',
    text: ['A table has 4 numeric columns. Polynomial features of degree 2, without the constant 1, keep the 4 columns and add every square and every product of two different columns. How many columns result?'],
    answer: pc4, whyNot: { 10: 'Those are the new terms only: the 4 original columns stay too.', 20: 'x₁·x₂ and x₂·x₁ are the same column: count each pair once.', '*': 'Count the originals, the squares and the pairs.' },
    hints: ['Originals: 4. Squares: one per column.', 'Products of two different columns: one per pair, 4 × 3 / 2 of them.', '4 + 4 + 6 = ' + pc4 + '. With degree 3 the count grows to 34: features multiply fast.'],
    solution: String(pc4), wrong: ['10', '20'] });

  const ptr = S.Data.curve({ n: 20, shape: 'sine', noise: 0.25, seed: 1 }), pte = S.Data.curve({ n: 400, shape: 'sine', noise: 0.25, seed: 51 });
  const perr = [1, 2, 3, 4, 5, 6, 7, 8].map((d) => { const f = ML.PolynomialFeatures({ degree: d, includeBias: false }).fit(ptr.X), m = ML.LinearRegression().fit(f.transform(ptr.X), ptr.y); return [d, r3(ML.mse(pte.y, m.predict(f.transform(pte.X))))]; });
  const pbest = perr.reduce((a, b) => (b[1] < a[1] ? b : a));
  C.exercise({ id: 'da-x-poly-curve', chapter: 'da-features', title: 'Test error against the degree', level: 3, kind: 'predict-curve',
    text: ['Polynomial regression is fitted on 20 noisy points along a wave and scored on 400 fresh ones. The test error of a straight line (degree 1) is drawn. Sketch the test MSE as the degree grows to 8.'],
    plot: { xr: [1, 8], yr: [0, 0.35], xname: 'degree', yname: 'test MSE', given: perr.slice(0, 1), note: 'Drag from degree 1 to degree 8.' },
    answer: perr, tolerance: 0.035, checks: ['valley'],
    hints: ['A wave needs at least a cubic to bend twice; beyond that, the extra freedom follows noise.', 'The error drops sharply at degree 3, to ' + perr[2][1] + ', and is lowest at degree ' + pbest[0] + ' (' + pbest[1] + ').', 'Then it climbs again: ' + perr.map((p) => 'degree ' + p[0] + ': ' + p[1]).join(', ') + '.'],
    solution: perr, wrong: [perr.map(([d]) => [d, r3(perr[0][1] * (1 - (d - 1) / 8))]), perr.map(([d]) => [d, perr[0][1]])] });

  const xopts = ['Add the product x₁·x₂', 'Add the squares x₁² and x₂²', 'Scale both columns', 'Use more training rows'];
  C.exercise({ id: 'da-x-interaction', chapter: 'da-features', title: 'One column to fix XOR', level: 2, kind: 'choice',
    text: ['Points are blue where x₁ and x₂ have the same sign and orange where they differ. Logistic regression on (x₁, x₂) gets about 62% right. Which single change lets the same model separate the classes?'],
    options: xopts, answer: 0,
    whyNot: { 1: 'Squares are positive on both sides of zero: they cannot tell same-sign corners from opposite-sign ones.', 2: 'Scaling stretches the axes, but a straight boundary still cannot separate opposite corners.', 3: 'No amount of data lets a straight line separate this pattern.' },
    hints: ['Which quantity is positive exactly when x₁ and x₂ have the same sign?', 'Its sign is the class rule itself.', 'Add x₁·x₂: in the new columns a straight boundary (x₁·x₂ = 0) separates the classes, and the model gets about 97% right.'],
    solution: 0, wrong: [1, 2] });

  const fopts = ['It may drop both: each column alone is unrelated to the label', 'It keeps both: together they decide the label', 'It keeps one and drops the other', 'It cannot score them'];
  C.exercise({ id: 'da-x-select-filter', chapter: 'da-features', title: 'Two columns that only work together', level: 2, kind: 'choice',
    text: ['In a table the label is "same sign" or "different sign" of two columns a and b. A filter scores each column alone by its correlation with the label and keeps the best few. What happens to a and b?'],
    options: fopts, answer: 0,
    whyNot: { 1: 'A filter never looks at columns together: that is its blind spot.', 2: 'Both have about the same, near-zero, score alone.', 3: 'It scores them; the scores are just near zero.' },
    hints: ['Look at a alone: is the label more often "same sign" when a is large?', 'No: for each a, both labels are equally likely, so the correlation is about 0. The same holds for b.', 'Both score near zero alone and may be dropped, although together they decide the label. Forward selection or a tree, which judge columns together, can find them.'],
    solution: 0, wrong: [1, 2] });

  const lopts = ['days the account spent in collections', 'income at the time of application', 'number of missed payments before the application', 'amount asked for'];
  C.exercise({ id: 'da-x-leak-later', chapter: 'da-features', title: 'A column from the future', level: 1, kind: 'choice',
    text: ['A lender predicts, at the moment of application, whether a loan will default. Which column would leak the answer into training?'],
    options: lopts, answer: 0,
    whyNot: { 1: 'Income is known when the application is made.', 2: 'Payments missed before the application are known at the time.', 3: 'The amount asked for is part of the application.' },
    hints: ['Ask of each column: is it known at the moment the prediction is made?', 'One column is only recorded after a loan has gone bad.', 'Days in collections: it exists only for loans that already defaulted, so it nearly is the label. It will make test scores look excellent and be useless on new applications.'],
    solution: 0, wrong: [1, 2] });

  const spot = ['Missing values are filled with column means computed over all 2000 rows, before the split.', 'The 20 columns most correlated with the label are chosen over all 2000 rows.', 'The rows are split 80/20, keeping each patient’s visits together on one side.', 'The scaler is fitted on the training rows and applied to the test rows.', 'The regularisation strength is chosen by 5-fold cross-validation inside the training rows.', 'The test rows are scored once, at the very end.'];
  C.exercise({ id: 'da-x-leak-spot', chapter: 'da-features', title: 'Spot the leaks', level: 3, kind: 'choice',
    text: ['A clinic predicts which patients will need a second visit. Its pipeline, step by step, is below. Which steps leak information from the test rows into training? Choose every one that does.'],
    options: spot, answer: [0, 1],
    hints: ['Look for steps that learn something from data. From which rows do they learn it?', 'Filling with means and choosing columns both learn from data; here they see all 2000 rows, test rows included.', 'Steps 1 and 2 leak: the means and the chosen columns were shaped by the test rows. Grouping a patient’s visits on one side, fitting the scaler on training rows, tuning inside the training rows and scoring the test set once are all honest.'],
    solution: [0, 1], wrong: [[1], [0, 1, 2]] });

  const sopt2 = ['Expect much less than 97% on new rows: the columns were chosen on all the labels', 'Trust the 97%: cross-validation is always honest', 'Expect more than 97% once the model sees more data', 'The 97% is fine because the model is simple'];
  C.exercise({ id: 'da-x-leak-score', chapter: 'da-features', title: 'A score too good to be true', level: 2, kind: 'choice',
    text: ['A colleague has 120 rows and 5000 gene measurements. She keeps the 30 measurements most correlated with the label over all 120 rows, then reports a 5-fold cross-validated accuracy of 97%. What should you expect on new patients?'],
    options: sopt2, answer: 0,
    whyNot: { 1: 'Cross-validation is honest only if every step that learns, selection included, happens inside the training folds.', 2: 'More data does not repair a biased estimate.', 3: 'A simple model on leaked columns is still scored on leaked columns.' },
    hints: ['With 5000 columns and 120 rows, some columns line up with the labels by chance.', 'Choosing them on all rows lets every test fold help choose its own columns.', 'Much less than 97%: on pure noise this kind of pipeline scored 73% where the truth was 50%. Redo the selection inside each training fold for an honest estimate.'],
    solution: 0, wrong: [1, 3] });
})(globalThis.Nooral = globalThis.Nooral || {});
