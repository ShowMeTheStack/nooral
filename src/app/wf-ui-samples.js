// Sample workflows of the canvas: small, complete pipelines to start from. Other block files add their own with
// S.WF.addSample; every sample is laid out by the tidy-up, so none needs positions.
(function (S) {
  'use strict';
  const WF = S.WF;
  const tidy = (spec, name) => WF.tidy(WF.build(spec, name));
  WF.addSample({
    id: 'first-pipeline', title: 'A first pipeline', cat: 'Basics', order: 1,
    blurb: 'Two moons, split into training and test rows, a logistic regression trained on one and judged on the other.',
    build: () => tidy({ blocks: { data1: ['data', { set: 'moons', n: 300, noise: 0.25 }], split1: ['split', { test: 0.3 }], logistic1: ['logistic', {}], train1: ['train', {}], evaluate1: ['evaluate', {}] },
      wires: [['data1.out', 'split1.in'], ['logistic1.model', 'train1.model'], ['split1.train', 'train1.data'], ['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data']] }, 'A first pipeline'),
  });
  WF.addSample({
    id: 'train-vs-test', title: 'Training rows against test rows', cat: 'Basics', order: 2,
    blurb: 'The same trained model evaluated twice: on the rows it learned from and on rows it never saw. The gap between the two is how much it memorised.',
    build: () => tidy({ blocks: { data1: ['data', { set: 'moons', n: 200, noise: 0.35 }], split1: ['split', { test: 0.4 }], logistic1: ['logistic', { C: 100 }], train1: ['train', {}], evaluate1: ['evaluate', {}], evaluate2: ['evaluate', {}] },
      wires: [['data1.out', 'split1.in'], ['logistic1.model', 'train1.model'], ['split1.train', 'train1.data'], ['train1.model', 'evaluate1.model'], ['split1.train', 'evaluate1.data'], ['train1.model', 'evaluate2.model'], ['split1.test', 'evaluate2.data']] }, 'Training rows against test rows'),
  });
  WF.addSample({
    id: 'two-strengths', title: 'Two strengths of regularisation', cat: 'Basics', order: 3,
    blurb: 'One split, two logistic regressions: C = 100 holds the weights back hardly at all, C = 0.01 a great deal. Which judges the test rows better?',
    build: () => tidy({ blocks: { data1: ['data', { set: 'circles', n: 300 }], split1: ['split', { test: 0.3 }], logistic1: ['logistic', { C: 100 }], logistic2: ['logistic', { C: 0.01 }], train1: ['train', {}], train2: ['train', {}], evaluate1: ['evaluate', {}], evaluate2: ['evaluate', {}] },
      wires: [['data1.out', 'split1.in'], ['logistic1.model', 'train1.model'], ['split1.train', 'train1.data'], ['logistic2.model', 'train2.model'], ['split1.train', 'train2.data'], ['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data'], ['train2.model', 'evaluate2.model'], ['split1.test', 'evaluate2.data']] }, 'Two strengths of regularisation'),
  });
  WF.addSample({
    id: 'loans-table', title: 'A table with a story: small loans', cat: 'Basics', order: 4,
    blurb: 'Applicants for small loans, and whether they repaid: logistic regression on six columns, judged on held-back applicants.',
    build: () => tidy({ blocks: { data1: ['data', { set: 'loans' }], split1: ['split', { test: 0.25 }], logistic1: ['logistic', { C: 1 }], train1: ['train', {}], evaluate1: ['evaluate', {}] },
      wires: [['data1.out', 'split1.in'], ['logistic1.model', 'train1.model'], ['split1.train', 'train1.data'], ['train1.model', 'evaluate1.model'], ['split1.test', 'evaluate1.data']] }, 'Small loans'),
  });
})(globalThis.Nooral = globalThis.Nooral || {});
