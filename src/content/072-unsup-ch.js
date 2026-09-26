// Part 7, unsupervised learning: the model challenge, the guided tour, the course, and the pages the diagram checker
// looks at. The challenge's rows come from S.Data with a fixed seed, and its hidden rows from another seed of the same
// generator, so a score means the same on every visit. The target was set from the engines' own scores.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('un-ch-dials')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- 1. three dials for the night shift: at most three principal components -------------------------------------------
  const dials = { data: { set: 'pumps', n: 800, seed: 4 }, hidden: { seed: 44, n: 1000 }, metric: 'f1', target: 0.9, marks: { silver: 0.94, gold: 0.96 } };
  C.challenge(Object.assign({ id: 'un-ch-dials', n: 71, title: 'Three dials for the control room', level: 3,
    story: 'The water works is refitting its control room. Each pump sends four readings a minute: temperature, vibration, pressure and motor current. The new panel has room for only three dials per pump, so the engineers will show three combinations of the four readings, chosen by principal component analysis, and the fault alarm may look only at those three dials.\n\nSome readings were lost when a sensor dropped out. Faults are rare, so the plant scores the alarm by F1 on the faults, on a thousand minutes it has kept back.',
    task: 'classification',
    limits: { blocks: ['split', 'impute', 'scale', 'pca', 'knn', 'logistic', 'svm', 'tree', 'train', 'evaluate'], models: ['knn', 'logistic', 'svm', 'tree'], params: { pca: { n: [1, 3] } }, maxBlocks: 8 },
    hints: ['Fill the gaps first (Impute), then PCA, then a model; wire the test rows through every preparing block too.', 'PCA follows variance, and the readings are in very different units: scale them before PCA, or the dials will show mostly the largest reading. Open the PCA block to see what each component holds.', 'Split → Impute (mean) → Scale (standard) → PCA (3 components) → SVM (RBF, C = 10) → Train → Evaluate: F1 about 0.97 on the hidden minutes, a gold mark. With 2 components the best is near 0.78.'],
    solution: pipe(dials.data, 'svm', { kernel: 'rbf', C: 10 }, [['impute', { strategy: 'mean' }], ['scale', { how: 'standard' }], ['pca', { n: 3 }]]),
    wrong: [pipe(dials.data, 'svm', { kernel: 'rbf', C: 10 }, [['impute', { strategy: 'mean' }], ['scale', { how: 'standard' }], ['pca', { n: 2 }]]), pipe(dials.data, 'knn', { k: 5 }, [['impute', { strategy: 'mean' }], ['pca', { n: 2 }]])] }, dials));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'un-tour', title: 'Tour: unsupervised learning', blurb: 'k-means, a dendrogram, DBSCAN, mixtures, PCA and anomaly scores, in eight steps.',
    steps: [
      { route: 'part=unsupervised', target: '#main', text: 'Part 7 of the catalogue: unsupervised learning. No labels here: each topic finds structure in the points alone, with the engine’s own algorithms.' },
      { route: 'topic=km-steps', target: '#stageWrap', text: 'k-means alternates two steps: every point joins its nearest centre, every centre moves to the mean of its points.' },
      { target: '#btnPlay', text: 'Play steps through the iterations; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=hc-build', target: '#inputs', text: 'Agglomerative clustering builds a tree of merges. Change the linkage or slide the cut here: the clusters change without clustering again.' },
      { route: 'topic=dbscan-core', target: '#stageWrap', text: 'DBSCAN grows clusters through crowded neighbourhoods and leaves the strays as noise, with no k to choose.' },
      { route: 'topic=pca-project', target: '#stageWrap', text: 'PCA turns a direction round until the shadows of the points spread the most: the first principal component.' },
      { route: 'lesson=un-gmm', target: '#lessonView', text: 'Lessons run the topics at the frames that matter, ask for a prediction, and end with exercises: here, a mixture fitted by EM.' },
      { route: 'learn', target: '[data-chapter="un-kmeans"]', text: 'The lessons and exercises of this part start here: a k-means round by hand, a silhouette, the next merge, DBSCAN labels, one EM step, and power iteration in code.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'un-course', title: 'Unsupervised learning', blurb: 'All seventeen topics of part 7, from one k-means step to anomaly scores, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'un-km', note: 'Assign, move, repeat.' },
      { topic: 'km-steps', note: 'k-means step by step.' },
      { exercise: 'un-x-update', note: 'Move a centre yourself.' },
      { topic: 'km-plusplus', note: 'Starting the centres far apart.' },
      { topic: 'km-elbow', note: 'The elbow.' },
      { topic: 'km-silhouette', note: 'The silhouette.' },
      { exercise: 'un-x-silhouette', note: 'A silhouette by hand.' },
      { lesson: 'un-hier', note: 'A tree of merges.' },
      { topic: 'hc-build', note: 'The dendrogram you can cut.' },
      { topic: 'hc-linkage', note: 'Four linkages.' },
      { exercise: 'un-x-single-next', note: 'The next merge.' },
      { topic: 'dbscan-core', note: 'Core, border and noise.' },
      { topic: 'dbscan-params', note: 'Choosing ε.' },
      { exercise: 'un-x-dbscan-labels', note: 'Label the taxis.' },
      { topic: 'gmm-em', note: 'Mixtures by EM.' },
      { topic: 'gmm-bic', note: 'How many components.' },
      { exercise: 'un-x-em-resp', note: 'One responsibility.' },
      { lesson: 'un-pca', note: 'The direction of greatest spread.' },
      { topic: 'pca-project', note: 'PCA in two dimensions.' },
      { topic: 'pca-variance', note: 'Thirteen dimensions.' },
      { topic: 'pca-recon', note: 'Rebuilding images.' },
      { exercise: 'un-x-code-power', note: 'Power iteration in code.' },
      { topic: 'tsne-run', note: 't-SNE.' },
      { topic: 'umap-run', note: 'A UMAP-style map.' },
      { topic: 'iforest-cuts', note: 'Random cuts isolate the odd.' },
      { topic: 'iforest-scores', note: 'Scores over the plane.' },
      { exercise: 'un-x-iforest-score', note: 'An anomaly score.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson un-km')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'unsupervised').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['un-x-assign', 'un-x-update', 'un-x-elbow', 'un-x-code-kmeans', 'un-x-single-next', 'un-x-dbscan-labels', 'un-x-em-resp', 'un-x-pca-k', 'un-x-cumvar-curve', 'un-x-wf-pca'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course un-course', hash: 'course=un-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge un-ch-dials', hash: 'challenge/un-ch-dials', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
