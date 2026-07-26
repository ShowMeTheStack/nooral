// Sample workflows for the model families: a dataset, a split, a model, Train and Evaluate, each set up so the model's
// insides are worth looking at (a tree on the moons, an SVM on circles, a network learning spirals, a CNN on drawn
// digits, an LSTM reading messages, a transformer on reviews, regression on flat rents).
(function (S) {
  'use strict';
  const WF = S.WF;
  /** the usual five blocks: data → split → (model) → train → evaluate */
  const pipe = (data, model, train, split) => () => WF.build({
    blocks: { data1: ['data', data, 40, 40], split1: ['split', split || {}, 280, 40], [model[0] + '1']: [model[0], model[1] || {}, 280, 220], train1: ['train', train || {}, 520, 130], evaluate1: ['evaluate', {}, 760, 60] },
    wires: [['data1.out', 'split1.in'], [model[0] + '1.model', 'train1.model'], ['split1.train', 'train1.data'], ['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data']],
  });
  const add = (id, title, blurb, data, model, train, split) => WF.addSample({ id, title, blurb, cat: 'Models', build: pipe(data, model, train, split) });
  add('tree-moons', 'A decision tree on the moons', 'A tree of depth 4 cutting the plane into boxes, split by split.', { set: 'moons', n: 240, noise: 0.25 }, ['tree', { maxDepth: 4 }]);
  add('svm-circles', 'An SVM with an RBF kernel on circles', 'The widest margin round a ring: the support vectors are ringed.', { set: 'circles', n: 200, noise: 0.1 }, ['svm', { kernel: 'rbf', C: 1, gamma: 2 }]);
  add('forest-moons', 'A random forest on the moons', 'Three of its trees, each a little different, and the vote of all of them.', { set: 'moons', n: 240, noise: 0.3 }, ['forest', { nEstimators: 40 }]);
  add('forest-loans', 'A random forest on small loans', 'Who repays: a forest on a table of applicants.', { set: 'loans' }, ['forest', { nEstimators: 60, maxDepth: 8 }]);
  add('boost-moons', 'Gradient boosting on the moons', 'Stage by stage, each small tree fixes what the ones before got wrong.', { set: 'moons', n: 240, noise: 0.25 }, ['gboost', { nEstimators: 60, maxDepth: 2 }]);
  add('ada-xor', 'AdaBoost on XOR', 'Stumps that each draw one straight cut, reweighting the rows they miss.', { set: 'xor', n: 200 }, ['adaboost', { nEstimators: 40, maxDepth: 2 }]);
  add('knn-blobs', 'k nearest neighbours on blobs', 'A new point takes the vote of the k training points nearest it.', { set: 'blobs', n: 200 }, ['knn', { k: 7 }]);
  add('nb-moons', 'Gaussian naive Bayes on the moons', 'An axis-aligned ellipse per class, and the boundary where they meet.', { set: 'moons', n: 200, noise: 0.25 }, ['naivebayes', {}]);
  add('qda-blobs', 'Quadratic discriminant analysis on blobs', 'A tilted ellipse of its own for every class, and curved boundaries.', { set: 'blobs', n: 240 }, ['qda', {}]);
  add('perceptron-line', 'The perceptron on a noisy line', 'Every mistake nudges the line; watch it settle, or not.', { set: 'linear', n: 160 }, ['perceptron', { epochs: 5 }]);
  add('mlp-spirals', 'A network learning the spirals', 'Two hidden layers of eight neurons: every neuron carves its region of the plane.', { set: 'spirals', n: 300 }, ['mlp', { hidden: [8, 8], act: 'tanh' }], { epochs: 150, lr: 0.03, batch: 32 });
  add('cnn-digits', 'A convolutional network on drawn digits', 'Two convolutions and a classifier, trained here: its filters, feature maps and curves.', { set: 'digits', n: 300 }, ['cnn', { f1: 6, f2: 12, kernel: 5, hidden: 32 }], { epochs: 6, lr: 0.003, batch: 32 });
  add('lstm-spam', 'An LSTM reading messages', 'Word by word, its hidden state and gates decide: spam or not.', { set: 'spam' }, ['rnn', { cell: 'LSTM', hiddenSize: 12, embed: 12 }], { epochs: 20, lr: 0.01, batch: 16 });
  add('transformer-spam', 'A transformer reading messages', 'Two heads of attention between the words of each message.', { set: 'spam' }, ['transformer', { heads: 2, embed: 16 }], { epochs: 20, lr: 0.01, batch: 16 });
  add('regression-rents', 'Ridge regression on flat rents', 'A weighted sum of the features, and the path of every weight as the penalty changes.', { set: 'rents' }, ['ridge', { alpha: 10 }]);
  add('lasso-rents', 'The lasso on flat rents', 'As the penalty grows, weights reach zero one by one.', { set: 'rents' }, ['lasso', { alpha: 5 }]);
  add('svr-curve', 'Support vector regression on a curve', 'The ε-tube round the fit, and the points outside it that shape it.', { set: 'curve' }, ['svr', { C: 10, epsilon: 0.15 }]);
  add('boost-curve', 'Gradient boosting on a curve', 'Stage by stage the fit climbs onto the curve.', { set: 'curve' }, ['gboost', { nEstimators: 80, maxDepth: 2 }]);
})(globalThis.Nooral = globalThis.Nooral || {});
