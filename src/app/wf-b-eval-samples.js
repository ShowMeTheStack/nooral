// Sample workflows that end in the evaluation, comparison, inference and explanation blocks. Each is built when asked
// for, from the model blocks this build has (a sample whose model is missing falls back to one that is there).
(function (S) {
  'use strict';
  const WF = S.WF;
  const has = (...ids) => ids.find((id) => WF.def(id)) || null;
  const tidy = (d) => (WF.tidy ? WF.tidy(d) : d);
  const add = (id, title, blurb, order, spec) => WF.addSample({ id, title, blurb, cat: 'Evaluate and explain', order, build: () => tidy(WF.build(spec(), title)) });
  // data → split → model → train, the start of most samples: ids data1, split1, <model>1, train1
  const base = (set, opts, model, mp, test) => ({
    blocks: { data1: ['data', Object.assign({ set }, opts || {})], split1: ['split', { test: test || 0.3 }], m1: [model, mp || {}], train1: ['train', {}] },
    wires: [['data1.out', 'split1.in'], ['m1.model', 'train1.model'], ['split1.train', 'train1.data']],
  });

  add('evaluate-loans', 'Small loans: thresholds and calibration', 'Which applicants will default? A logistic regression judged on held-back applicants: its confusion matrix, the ROC and PR curves with a threshold to move, and whether its probabilities can be trusted.', 10, () => {
    const s = base('loans', {}, 'logistic', { C: 1 }, 0.3);
    Object.assign(s.blocks, { evaluate1: ['evaluate', {}], roc1: ['roc', { threshold: 0.5 }], calibration1: ['calibration', { bins: 8 }] });
    s.wires.push(['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data'], ['evaluate1.pred', 'roc1.pred'], ['evaluate1.pred', 'calibration1.pred']);
    return s;
  });
  add('compare-three', 'Three models on two moons', 'A straight boundary, a tree and the nearest neighbours learn the same moons: their scores, confusion matrices and boundaries side by side.', 20, () => {
    const t = has('tree') || 'logistic', k = has('knn') || 'logistic';
    return {
      blocks: { data1: ['data', { set: 'moons', n: 300, noise: 0.3 }], split1: ['split', { test: 0.3 }], m1: ['logistic', {}], m2: [t, t === 'tree' ? { maxDepth: 5 } : { C: 10 }], m3: [k, k === 'knn' ? { k: 9 } : { C: 0.1 }],
        train1: ['train', {}], train2: ['train', {}], train3: ['train', {}], compare1: ['compare', {}] },
      wires: [['data1.out', 'split1.in'], ['m1.model', 'train1.model'], ['m2.model', 'train2.model'], ['m3.model', 'train3.model'], ['split1.train', 'train1.data'], ['split1.train', 'train2.data'], ['split1.train', 'train3.data'],
        ['train1.model', 'compare1.a'], ['train2.model', 'compare1.b'], ['train3.model', 'compare1.c'], ['split1.test', 'compare1.data']],
    };
  });
  add('predict-exam', 'Ask a model: will this student pass?', 'A model of exam results learns from past students; type in hours of study, sleep and practice papers, and it answers with its probability and the past students most like yours.', 30, () => {
    const s = base('exam', {}, has('knn') || 'logistic', has('knn') ? { k: 7 } : {}, 0.25);
    Object.assign(s.blocks, { predict1: ['predict', {}] });
    s.wires.push(['train1.model', 'predict1.model'], ['split1.train', 'predict1.data']);
    return s;
  });
  add('predict-moons', 'Click to ask: two moons', 'A trained model on two moons: click anywhere on the plane and it says which moon the point belongs to, and how sure it is.', 31, () => {
    const s = base('moons', { n: 300, noise: 0.25 }, has('svm') || 'logistic', has('svm') ? { kernel: 'rbf', C: 3 } : { C: 10 }, 0.3);
    Object.assign(s.blocks, { predict1: ['predict', {}] });
    s.wires.push(['train1.model', 'predict1.model'], ['split1.train', 'predict1.data']);
    return s;
  });
  add('predict-digits', 'Draw a digit', 'A network trained on generated handwritten digits: draw one of your own and see the probability it gives each digit, then where in the image it looked.', 32, () => {
    const net = has('cnn', 'mlp', 'mlpnet') || 'logistic';
    const s = base('digits', { n: 600, size: 16 }, net, {}, 0.2);
    s.blocks.train1 = ['train', { epochs: 12, batch: 32, lr: 0.01 }];
    Object.assign(s.blocks, { evaluate1: ['evaluate', {}], predict1: ['predict', {}], saliency1: ['saliency', { count: 4 }] });
    s.wires.push(['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data'], ['train1.model', 'predict1.model'], ['split1.test', 'predict1.data'], ['train1.model', 'saliency1.model'], ['split1.test', 'saliency1.data']);
    return s;
  });
  add('explain-harvest', 'Why this harvest? Explaining a forest', 'A forest predicts the wheat a plot yields from rain, soil, fertiliser and sun. Which columns does it lean on, how does the yield follow the rain, and why did one plot get its number?', 40, () => {
    const f = has('forest', 'tree') || 'linreg';
    const s = base('harvest', {}, f, f === 'forest' ? { nEstimators: 30, maxDepth: 6 } : {}, 0.3);
    Object.assign(s.blocks, { importance1: ['importance', { repeats: 5 }], pdp1: ['pdp', { feature: 0 }], shap1: ['shap', { row: 0, rows: 20 }], lime1: ['lime', { row: 0 }] });
    for (const b of ['importance1', 'pdp1', 'shap1', 'lime1']) s.wires.push(['train1.model', b + '.model'], ['split1.test', b + '.data']);
    return s;
  });
  add('regression-residuals', 'Flat rents: the errors of a line', 'A straight line predicts rents from floor area, rooms, floor, distance and balcony. The residuals show where a line is not enough, and the learning curve whether more flats would help.', 50, () => {
    const s = base('rents', {}, has('linreg', 'ridge') || 'linreg', {}, 0.3);
    Object.assign(s.blocks, { evaluate1: ['evaluate', {}], residuals1: ['residuals', { feature: 0 }], learningcurve1: ['learningcurve', { steps: 7, repeats: 3 }] });
    s.wires.push(['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data'], ['evaluate1.pred', 'residuals1.pred'], ['m1.model', 'learningcurve1.model'], ['split1.train', 'learningcurve1.data']);
    return s;
  });
  add('learning-curve-tree', 'Too deep or just right? A learning curve', 'A decision tree on moons, trained on more and more rows and at every depth: the gap between the training and the validation score shows overfitting.', 51, () => {
    const t = has('tree') || 'logistic';
    return { blocks: { data1: ['data', { set: 'moons', n: 400, noise: 0.3 }], m1: [t, t === 'tree' ? { maxDepth: 12 } : {}], learningcurve1: ['learningcurve', { sweep: t === 'tree' ? 'maxDepth' : 'C' }] },
      wires: [['data1.out', 'learningcurve1.data'], ['m1.model', 'learningcurve1.model']] };
  });
})(globalThis.Nooral = globalThis.Nooral || {});
