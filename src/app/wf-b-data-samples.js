// Sample workflows of the data, prepare, split and unsupervised blocks: preparing a table without leaking the test
// rows into training, cross-validation, texts turned into word counts, and the clustering and map blocks side by side.
(function (S) {
  'use strict';
  const WF = S.WF;
  const tidy = (d) => (WF.tidy ? WF.tidy(d) : d);
  const add = (id, title, blurb, order, spec) => WF.addSample({ id, title, blurb, cat: 'Data, preparation and clusters', order, build: () => tidy(WF.build(spec, title)) });

  add('prepare-pumps', 'Preparing a table: gaps and scales', 'Pump readings with gaps. The split comes first; Impute and Scale learn from the training rows and do the same to the test rows, so nothing about the test rows leaks into training.', 10, {
    blocks: { data1: ['data', { set: 'pumps', n: 600 }], split1: ['split', { test: 0.3 }], impute1: ['impute', { strategy: 'median' }], scale1: ['scale', {}], logistic1: ['logistic', {}], train1: ['train', {}], evaluate1: ['evaluate', {}] },
    wires: [['data1.out', 'split1.in'], ['split1.train', 'impute1.in'], ['split1.test', 'impute1.test'], ['impute1.out', 'scale1.in'], ['impute1.test', 'scale1.test'], ['logistic1.model', 'train1.model'], ['scale1.out', 'train1.data'], ['train1.model', 'evaluate1.model'], ['scale1.test', 'evaluate1.data']],
  });
  add('leakage-loans', 'Leakage: scaling before the split', 'The same steps in the wrong order: the scaler and the feature selection see every row before the split, so the test rows have already shaped the model’s inputs. Open Scale to read the warning.', 20, {
    blocks: { data1: ['data', { set: 'loans' }], encode1: ['encode', {}], scale1: ['scale', {}], select1: ['select', { k: 4 }], split1: ['split', { test: 0.3 }], logistic1: ['logistic', {}], train1: ['train', {}], evaluate1: ['evaluate', {}] },
    wires: [['data1.out', 'encode1.in'], ['encode1.out', 'scale1.in'], ['scale1.out', 'select1.in'], ['select1.out', 'split1.in'], ['logistic1.model', 'train1.model'], ['split1.train', 'train1.data'], ['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data']],
  });
  add('clean-pumps', 'Cleaning: drop, clip and fill', 'Three ways to deal with a messy table, side by side: drop the rows with a gap, clip the wild values, or fill the gaps.', 30, {
    blocks: { data1: ['data', { set: 'pumps', n: 500 }], clean1: ['clean', { mode: 'rows' }], clean2: ['clean', { mode: 'clip' }], impute1: ['impute', { indicator: true }] },
    wires: [['data1.out', 'clean1.in'], ['data1.out', 'clean2.in'], ['data1.out', 'impute1.in']],
  });
  add('features-moons', 'New features: polynomial terms', 'Two moons cannot be split by a straight line, but in the space of their squares and cubes a straight boundary is a curve. Compare the model with and without the new features.', 40, {
    blocks: { data1: ['data', { set: 'moons', n: 300, noise: 0.2 }], split1: ['split', { test: 0.3 }], poly1: ['poly', { degree: 3 }], scale1: ['scale', {}], logistic1: ['logistic', { C: 10 }], train1: ['train', {}], evaluate1: ['evaluate', {}] },
    wires: [['data1.out', 'split1.in'], ['split1.train', 'poly1.in'], ['split1.test', 'poly1.test'], ['poly1.out', 'scale1.in'], ['poly1.test', 'scale1.test'], ['logistic1.model', 'train1.model'], ['scale1.out', 'train1.data'], ['train1.model', 'evaluate1.model'], ['scale1.test', 'evaluate1.data']],
  });
  add('cv-wildflowers', 'Cross-validation: five honest scores', 'Meadow wildflowers, five folds: each fold is predicted by a model that never saw it. The spread of the five scores shows how much one split could mislead.', 50, {
    blocks: { data1: ['data', { set: 'wildflowers' }], logistic1: ['logistic', {}], cv1: ['cv', { k: 5 }] },
    wires: [['data1.out', 'cv1.data'], ['logistic1.model', 'cv1.model']],
  });
  add('text-spam-bow', 'Texts as word counts: a spam filter', 'Short messages turned into a table of word counts (the vocabulary learned from the training messages only), then a logistic regression on the counts.', 60, {
    blocks: { data1: ['data', { set: 'spam' }], split1: ['split', { test: 0.3 }], vectorise1: ['vectorise', { weight: 'count', vocab: 150 }], logistic1: ['logistic', {}], train1: ['train', {}], evaluate1: ['evaluate', {}] },
    wires: [['data1.out', 'split1.in'], ['split1.train', 'vectorise1.in'], ['split1.test', 'vectorise1.test'], ['logistic1.model', 'train1.model'], ['vectorise1.out', 'train1.data'], ['train1.model', 'evaluate1.model'], ['vectorise1.test', 'evaluate1.data']],
  });
  add('clusters-kmeans', 'Clustering: k-means and a mixture', 'Blobs of points with no labels used: k-means from a poor start, round by round, and a Gaussian mixture whose clouds settle over the same rows.', 70, {
    blocks: { data1: ['data', { set: 'blobs', n: 300, seed: 4 }], kmeans1: ['kmeans', { k: 3, init: 'random', seed: 3 }], gmm1: ['gmm', { k: 3, seed: 2 }] },
    wires: [['data1.out', 'kmeans1.in'], ['data1.out', 'gmm1.in']],
  });
  add('clusters-moons', 'Clustering by shape: DBSCAN against k-means', 'Two moons: k-means cuts them with a straight line, DBSCAN follows each crescent by density, and the dendrogram shows how single linkage chains along them.', 80, {
    blocks: { data1: ['data', { set: 'moons', n: 240, noise: 0.07 }], kmeans1: ['kmeans', { k: 2 }], dbscan1: ['dbscan', { eps: 0.22, minPts: 5 }], hierarchical1: ['hierarchical', { linkage: 'single', k: 2 }] },
    wires: [['data1.out', 'kmeans1.in'], ['data1.out', 'dbscan1.in'], ['data1.out', 'hierarchical1.in']],
  });
  add('maps-wine', 'Maps of a wide table: PCA, t-SNE and UMAP', 'Thirteen measurements of wines from three cultivars, squeezed onto a page three ways. The maps never see the cultivar, yet group them.', 90, {
    blocks: { data1: ['data', { set: 'wine' }], scale1: ['scale', {}], pca1: ['pca', { n: 2 }], tsne1: ['tsne', { perplexity: 30, iters: 500 }], umap1: ['umap', {}] },
    wires: [['data1.out', 'scale1.in'], ['scale1.out', 'pca1.in'], ['scale1.out', 'tsne1.in'], ['scale1.out', 'umap1.in']],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
