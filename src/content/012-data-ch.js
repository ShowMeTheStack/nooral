// Part 1, data and features: the model challenge, the guided tour, the course, and the pages the diagram checker looks
// at. The challenge’s rows come from S.Data with a fixed seed and a share of blanked cells, and its hidden rows from
// another seed of the same generator with the same share of gaps, so a score means the same on every visit. The target
// and the marks were set from the engines’ own scores.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('da-ch-rents')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- a rent estimate from listings full of blanks -----------------------------------------------------------------
  const rents = { data: { set: 'rents', n: 200, seed: 5, opts: { missing: 0.25 } }, hidden: { seed: 55, n: 600 }, metric: 'r2', target: 0.76, marks: { silver: 0.77, gold: 0.775 } };
  C.challenge(Object.assign({ id: 'da-ch-rents', n: 11, title: 'Rents from half-filled listings', level: 2,
    story: 'A small letting agency puts every flat it handles on a web form: floor area, rooms, floor, distance to the centre, and whether there is a balcony. Every field is optional, and landlords skip what they cannot be bothered to look up, so about a quarter of the cells are blank, scattered at random.\n\nThe agency wants a fair first estimate of the monthly rent for each new listing, blanks and all. It scores an estimate by R² on 600 listings it has kept back, which have just as many blanks. Throwing away every listing with a blank would leave very few, and the new listings will have blanks too.',
    task: 'regression',
    limits: { blocks: ['split', 'clean', 'impute', 'encode', 'scale', 'linreg', 'ridge', 'knn', 'tree', 'train', 'evaluate'], models: ['linreg', 'ridge', 'knn', 'tree'], maxBlocks: 9 },
    hints: ['Run the plainest workflow first; Train will refuse the blanks. Then compare dropping rows (Clean) with filling them (Impute): how many training rows does each keep?', 'With five columns and a quarter of the cells blank, fewer than a quarter of the rows are complete. Fill the blanks from the training rows, and turn the balcony into numbers with Encode. Dropping the mostly-blank columns throws away everything.', 'Split → Impute (mean) → Encode → Linear regression → Train → Evaluate: R² 0.774 on the hidden listings. Encoding before imputing, so the balcony gaps are filled too, reaches 0.777, a gold mark.'],
    solution: pipe(rents.data, 'linreg', {}, [['encode', {}], ['impute', { strategy: 'mean' }]]),
    wrong: [pipe(rents.data, 'linreg', {}, [['clean', { mode: 'rows' }], ['encode', {}]]), pipe(rents.data, 'linreg', {}, [['clean', { mode: 'cols', maxMissing: 0.2 }], ['encode', {}]])] }, rents));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'da-tour', title: 'Tour: data and features', blurb: 'A table, its columns, its gaps and outliers, scaling and encoding, and a leak, in eight steps.',
    steps: [
      { route: 'part=data', target: '#main', text: 'Part 1 of the catalogue: data and features. Every topic here runs on real tables, our own stories and a few classic open sets.' },
      { route: 'topic=data-table', target: '#stageWrap', text: 'A dataset is a table: one row per example, one column per feature, and a label to predict.' },
      { target: '#btnPlay', text: 'Play steps through the topic; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=data-describe', target: '#inputs', text: 'Choose a column here: its box, mean, spread and histogram are worked out on the spot.' },
      { route: 'topic=data-missing', target: '#stageWrap', text: 'Gaps in a table, and four ways to fill them. Each changes the shape of the column in its own way.' },
      { route: 'topic=data-scaling', target: '#stageWrap', text: 'Why k-nearest neighbours needs scaling: one column in the thousands decides every distance alone.' },
      { route: 'topic=data-leakage', target: '#stageWrap', text: 'Leakage on pure noise: a leaky pipeline scores 73%, an honest one 54%. Every step that learns must learn from the training rows only.' },
      { route: 'learn', target: '[data-chapter="da-tables"]', text: 'The lessons and exercises of this part start here: quartiles and fences by hand, scalers to write, encodings to choose, leaks to spot, and a workflow that must cope with gaps.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'da-course', title: 'Data and features', blurb: 'All thirteen topics of part 1, from a table’s rows to a leak shown happening, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'da-table', note: 'Rows, features and labels.' },
      { topic: 'data-table', note: 'Any dataset as a table.' },
      { exercise: 'da-x-task', note: 'Classes or amounts?' },
      { topic: 'data-describe', note: 'A column’s centre, spread and shape.' },
      { exercise: 'da-x-five', note: 'Quartiles by hand.' },
      { topic: 'data-scatter-matrix', note: 'Every pair of columns at once.' },
      { topic: 'data-correlation', note: 'One number for a straight-line relation.' },
      { exercise: 'da-x-corr-trap', note: 'When r misleads.' },
      { lesson: 'da-missing', note: 'Drop or fill.' },
      { topic: 'data-missing', note: 'Four fillers side by side.' },
      { exercise: 'da-x-impute', note: 'A fill value, learned honestly.' },
      { topic: 'data-outliers', note: 'Two rules for wild values.' },
      { exercise: 'da-x-zfence', note: 'An outlier hiding behind its own spread.' },
      { topic: 'data-scaling', note: 'Three scalers and k-NN.' },
      { exercise: 'da-x-code-minmax', note: 'Write a scaler.' },
      { topic: 'data-encoding', note: 'Ordinal against one-hot.' },
      { exercise: 'da-x-onehot', note: 'One-hot by hand.' },
      { topic: 'data-binning', note: 'A staircase from bins.' },
      { topic: 'data-poly', note: 'Curves from powers.' },
      { topic: 'data-interactions', note: 'Products of columns.' },
      { exercise: 'da-x-poly-curve', note: 'Test error against the degree.' },
      { topic: 'data-select', note: 'Three ways to keep the useful columns.' },
      { lesson: 'da-leakage', note: 'How a score can lie.' },
      { topic: 'data-leakage', note: 'Leaks shown happening.' },
      { exercise: 'da-x-leak-spot', note: 'Spot the leaks in a pipeline.' },
      { exercise: 'da-x-wf-gaps', note: 'A workflow that copes with gaps.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson da-table')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'data').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['da-x-five', 'da-x-drop-count', 'da-x-fences', 'da-x-onehot', 'da-x-code-standard', 'da-x-poly-row', 'da-x-poly-curve', 'da-x-leak-spot', 'da-x-wf-gaps', 'da-x-bins'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course da-course', hash: 'course=da-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge da-ch-rents', hash: 'challenge/da-ch-rents', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
