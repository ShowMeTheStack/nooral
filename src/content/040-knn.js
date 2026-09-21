// Part 4, nearest neighbours and probabilistic models: the chapters and the lessons. The exercises are in
// 041-knn-ex.js; the model challenges, the tour, the course and the pages the diagram checker looks at are in
// 042-knn-ch.js. Each lesson opens with a question, runs the topic that shows the idea at a frame chosen for the point
// being made, asks for a prediction, sums up, and leads to its exercises. Every value quoted here was worked out by
// the engines (S.ML) on the topics' own inputs.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.lessons.has('nn-voronoi')) return;
  const X = (id) => ({ exercise: id });

  // a small workflow: Dataset → Split → (preparation) → model → Train → Evaluate
  const flow = (set, model, mp, prep) => {
    const blocks = { data: ['data', set], split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  C.chapter({ id: 'nn-knn', part: 'neighbours', n: 1, title: 'k nearest neighbours', blurb: 'A model that only remembers: Voronoi cells, the vote of k neighbours, the distance that decides who is near, and votes weighted by closeness.' });
  C.chapter({ id: 'nn-search', part: 'neighbours', n: 2, title: 'Fast search and many dimensions', blurb: 'A k-d tree that skips whole boxes of points, and why nearness itself fades as the number of features grows.' });
  C.chapter({ id: 'nn-bayes', part: 'neighbours', n: 3, title: 'Naive Bayes', blurb: 'Bayes’ rule with features treated as independent: bell curves for measurements, word counts for messages, and smoothing for the word never seen.' });
  C.chapter({ id: 'nn-da', part: 'neighbours', n: 4, title: 'Discriminant analysis', blurb: 'A full Gaussian per class: one shared shape gives a straight boundary (LDA), a shape per class a curved one (QDA).' });

  // =========================================================================================================
  // Chapter 1: k nearest neighbours
  // =========================================================================================================
  C.lesson({
    id: 'nn-voronoi', chapter: 'nn-knn', title: 'A model that only remembers', minutes: 6,
    summary: 'One-nearest-neighbour stores the training points and gives a new point the class of the closest one. The plane splits into Voronoi cells, one per point, and the decision boundary is made of the cell walls between classes.',
    blocks: [
      'Most models learn something compact: a line, a tree, a few weights. Is there a model that learns nothing at all, keeps every example it was shown, and still predicts well?',
      { def: 'k-nearest neighbours', text: 'to classify a new point, find the k training points closest to it and take a vote of their classes. With k = 1 the new point simply copies its nearest neighbour.' },
      { topic: 'knn-voronoi', frame: 0, caption: '16 points in 3 classes. Training is nothing more than storing them: all the work waits until a question is asked.' },
      { topic: 'knn-voronoi', frame: 1, caption: 'Each point owns its Voronoi cell, the places nearer to it than to any other point. The 16 cells meet along 35 walls, each on the perpendicular bisector of two points.' },
      { predict: { q: 'Only some of the 35 walls separate two different classes. What happens at the other walls when 1-NN predicts?', options: ['Nothing: both sides predict the same class, so the wall is not part of the boundary', 'The prediction flips there', 'Points on those walls are left unclassified'], answer: 0, explain: 'A wall between two cells of the same class changes the nearest point but not the answer. Only 11 of the 35 walls lie between classes, and those 11 make the decision boundary.' } },
      { topic: 'knn-voronoi', frame: 2, caption: 'The 11 walls between classes form the boundary, a chain of straight pieces. Every training point sits in its own cell, so it predicts itself: training accuracy is always 100%.' },
      { topic: 'knn-voronoi', frame: 4, caption: 'A query at (5.5, 4.5) lies in the cell of its nearest point, 0.851 away and orange, so 1-NN says orange. Finding the cell and finding the nearest neighbour are the same question.' },
      { tip: 'A training accuracy of 100% says nothing here: 1-NN would score 100% on its training points even if the labels were random. Judge it only on points it has not stored.' },
      { h: 'Recap' },
      { list: ['k-NN keeps the training points and does its work at prediction time.', '1-NN cuts the plane into Voronoi cells; the walls between classes are the boundary.', 'Its training accuracy is always 100%, so only held-out points can judge it.'] },
      X('nn-x-nearest-pick'), X('nn-x-dist-matrix'), X('nn-x-loo-matrix'),
    ],
  });

  C.lesson({
    id: 'nn-k', chapter: 'nn-knn', title: 'k against smoothness', minutes: 8,
    summary: 'Asking more neighbours smooths the boundary: k = 1 wraps every stray point in an island, a large k washes out real detail. The training error tends to rise with k; the test error falls and then rises, and its lowest point picks k.',
    blocks: [
      'One nearest neighbour trusts a single point. If that point is a mislabelled oddity, every place in its cell inherits the mistake. How many neighbours should vote?',
      { topic: 'knn-k', frame: 0, caption: 'k = 1 on 80 noisy moon points: training error 0%, test error 4% on 80 fresh points. Look for the small islands of one colour inside the other.' },
      { predict: { q: 'Raise k from 1 to 61, with 80 training points. What happens to the training error?', options: ['It stays at 0%', 'It rises', 'It falls'], answer: 1, explain: 'With k = 1 each point votes for itself. With more voters a point can be outvoted by its neighbours, so the training error climbs: 1% at k = 3, 21% at k = 61.' } },
      { topic: 'knn-k', frame: 1, caption: 'k = 3: the islands melt away. Training error 1%, test error 1%: the lowest test error of the sweep.' },
      { topic: 'knn-k', frame: 5, caption: 'k = 61, three quarters of all the points: the map is so smooth it cuts across both moons. Training error 21%, test error 24%: the model now underfits.' },
      { table: [['k', 'Training error', 'Test error'], ['1', '0%', '4%'], ['3', '1%', '1%'], ['7', '3%', '4%'], ['15', '5%', '9%'], ['31', '10%', '13%'], ['61', '21%', '24%']], caption: 'The sweep above. Small k overfits (training error far below test error), large k underfits (both high).' },
      'k plays the part that depth plays for a tree or the degree for a polynomial: it sets how flexible the model is. It is chosen the same way, by the error on points the model has not stored, best with cross-validation.',
      { workflow: flow({ set: 'moons', n: 200, noise: 0.3, seed: 2 }, 'knn', { k: 5 }), live: true, height: 460, caption: 'k-NN on noisy moons, running here. Double-click the k-NN block to change k, and open Evaluate to see which points change side.' },
      { tip: 'With two classes, an odd k avoids tied votes.' },
      { h: 'Recap' },
      { list: ['Small k: a jagged boundary with islands, low training error, high variance.', 'Large k: a smooth boundary that misses real detail, high bias.', 'Choose k by held-out error; the training error tends to rise with k, so it cannot choose.'] },
      X('nn-x-vote'), X('nn-x-k-overfit'), X('nn-x-k-curve'),
    ],
  });

  C.lesson({
    id: 'nn-metrics', chapter: 'nn-knn', title: 'Who counts as near', minutes: 8,
    summary: 'Nearness depends on the distance. Euclidean, Manhattan and Chebyshev distance draw a circle, a diamond and a square around a query, and can pick different neighbours. Units matter even more: a feature measured in thousands drowns one measured in fractions.',
    blocks: [
      'A lender compares two applicants: incomes 42 and 45 thousand, debt ratios 0.20 and 0.65. Are they close? On income they differ by 3, on the debt ratio by 0.45. The second gap matters far more to the lender, but a plain distance sees mostly the first.',
      { h: 'Three distances' },
      { math: 'Euclidean  d₂(a, b) = √(Σ (aᵢ − bᵢ)²)\nManhattan  d₁(a, b) = Σ |aᵢ − bᵢ|\nChebyshev  d∞(a, b) = maxᵢ |aᵢ − bᵢ|' },
      { topic: 'knn-metrics', frame: 0, caption: 'A query at (3.5, 6.25) among 40 points. On the right, the points at distance 1 under each metric: a circle, a diamond and a square.' },
      { predict: { q: 'Two points, one 2 to the right of the query and one 1.2 right and 1.2 up. Which is nearer under Manhattan distance?', options: ['The first, at 2 against 2.4', 'The second', 'They tie'], answer: 0, explain: 'Manhattan adds the gaps: 2 against 1.2 + 1.2 = 2.4. Under Euclidean distance the second is nearer, √(1.2² + 1.2²) = 1.7 against 2: diagonal moves are cheap on a circle and dear on a diamond.' } },
      { topic: 'knn-metrics', frame: 2, caption: 'Manhattan: the 6 nearest all lie within 2.12, inside the diamond. It shares 5 of its 6 neighbours with the Euclidean set; the ringed point is the one that changed.' },
      { topic: 'knn-metrics', frame: 4, caption: 'Every metric still votes green here. Near a boundary the swapped neighbour can flip the vote.' },
      { h: 'Units first' },
      'Before any metric, the features must be on comparable scales. Standardising each feature, (x − mean) / standard deviation, makes a step of one unit mean "one typical spread" on every axis. On a test quarter of 500 small loans, k-NN with k = 5 catches the defaults with F1 0.24 on raw features and 0.5 once they are standardised (accuracy 0.744 against 0.824).',
      { workflow: flow({ set: 'loans', n: 500, seed: 7 }, 'knn', { k: 5 }, [['scale', { how: 'standard' }]]), live: true, height: 460, caption: 'Scale → k-NN on the loans table, running here. Delete the Scale block and wire the split straight to Train to watch F1 fall.' },
      { h: 'Recap' },
      { list: ['The metric decides the neighbours: circle, diamond or square.', 'Manhattan adds the gaps, Chebyshev keeps only the largest.', 'Standardise the features first, or the one with the largest units decides alone.'] },
      X('nn-x-metric-set'), X('nn-x-scale-share'), X('nn-x-code-dist'), X('nn-x-wf-loans'),
    ],
  });

  C.lesson({
    id: 'nn-weights', chapter: 'nn-knn', title: 'Weighting the votes', minutes: 6,
    summary: 'With uniform weights each of the k neighbours has one vote. With distance weights a neighbour counts 1/d, so the nearest ones speak loudest. The vote then becomes a probability, and a training point, at distance 0 from itself, outvotes everyone.',
    blocks: [
      'Seven neighbours vote: three blue ones far away and four orange ones close by, or the other way round. Should a neighbour at the edge of the circle count as much as one right beside the query?',
      { topic: 'knn-weights', frame: 0, caption: 'The 7 nearest neighbours of the query (5, 4.25) lie between 1.14 and 1.7 away.' },
      { topic: 'knn-weights', frame: 1, caption: 'Uniform weights: 3 votes for blue, 4 for orange, so orange with probability 4/7 = 0.571.' },
      { math: 'weight of a neighbour at distance d:  w = 1 / d\nP(class) = (sum of the weights of its neighbours) / (sum of all the weights)' },
      { predict: { q: 'Switch to distance weights. The nearest neighbour is 1.14 away. What is its weight?', options: ['1.14', '0.875', '1'], answer: 1, explain: '1 / 1.14 = 0.875. The farthest, 1.7 away, weighs 0.587.' } },
      { topic: 'knn-weights', frame: 2, caption: 'Distance weights: 2.14 for blue against 2.65 for orange, so orange with probability 0.553, as before. Here the weighting softens the vote without changing it.' },
      { topic: 'knn-weights', frame: 3, caption: 'On its own training points, every point is at distance 0 from itself and takes the whole vote: training accuracy 100% against 83% with uniform weights.' },
      { warn: 'With distance weights the training accuracy is always 100%, whatever k. It cannot show overfitting; only held-out points can.' },
      { h: 'Recap' },
      { list: ['Uniform: each of the k neighbours has one vote.', 'Distance: weight 1/d, so near neighbours count more and the vote becomes a smooth probability.', 'A point at distance 0 takes the whole vote, which makes training accuracy meaningless.'] },
      X('nn-x-weighted'), X('nn-x-weighted-flip'), X('nn-x-train-distance'), X('nn-x-code-knn'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: fast search and many dimensions
  // =========================================================================================================
  C.lesson({
    id: 'nn-kd', chapter: 'nn-search', title: 'A k-d tree skips whole boxes', minutes: 7,
    summary: 'Brute force measures the distance to every stored point. A k-d tree cuts the points into nested boxes at medians; the search goes down the near side first, then skips every box farther away than the best neighbour found so far.',
    blocks: [
      'A library of a million points, and a query every millisecond. Measuring a million distances per query is too slow. Can we rule out most points without measuring them?',
      { topic: 'kd-search', frame: 0, caption: '30 points cut into 14 leaves of at most 3 points. Each dashed line is a median split of the wider side of its box.' },
      { topic: 'kd-search', frame: 1, caption: 'At the root the query has x0 = 6.3 against the split 3.69: it belongs on the right, so the search goes there first.' },
      { topic: 'kd-search', frame: 6, caption: 'After two leaves, the best neighbour so far is 1.183 away. The circle of that radius is the search ball: only boxes that reach inside it can hold anything nearer.' },
      { predict: { q: 'The next box on the way back up is 1.64 away from the query. The best neighbour so far is 1.183 away. Must the search open it?', options: ['Yes, it may hold a nearer point', 'No: every point in it is at least 1.64 away', 'Only if it holds more than 3 points'], answer: 1, explain: 'The distance to a box is a lower bound for every point inside it. 1.64 > 1.183, so the box is skipped with its 3 points unmeasured.' } },
      { topic: 'kd-search', frame: 7, caption: 'The box at depth 3 is 1.64 away: skipped. Then two larger boxes, 2.62 and 2.93 away, go the same way, with 8 and 15 points.' },
      { topic: 'kd-search', frame: 10, caption: 'Done: the nearest point at 1.18, found after only 4 of the 30 distances, with 3 boxes skipped whole. Brute force gives the same answer.' },
      { tip: 'The answer is exact: the tree only saves work. Its advantage shrinks as the number of features grows, because the search ball then reaches into almost every box.' },
      { h: 'Recap' },
      { list: ['A k-d tree splits at medians, alternating to the wider side of each box.', 'The search visits the near side first and keeps the best distance found.', 'Any box farther than that distance is skipped with all its points.'] },
      X('nn-x-kd-split'), X('nn-x-kd-skip'),
    ],
  });

  C.lesson({
    id: 'nn-curse', chapter: 'nn-search', title: 'The curse of dimensionality', minutes: 6,
    summary: 'In many dimensions random points are all about equally far from a query: the nearest is hardly nearer than the farthest. Nearest-neighbour methods then lose their footing, unless the data really lie near a low-dimensional surface.',
    blocks: [
      'k-NN rests on one idea: the nearest points are similar. That needs some points to be clearly nearer than others. With 300 random points in a cube, how much nearer is the nearest than the farthest, as the cube gains dimensions?',
      { topic: 'curse-dim', frame: 0, caption: 'One dimension: the nearest of 300 points is 0.0000218 from the query, the farthest 0.83. Distances spread over the whole range.' },
      { predict: { q: 'In 1000 dimensions, the nearest distance divided by the farthest is about…', options: ['0.0001, as in one dimension', '0.3', '0.9'], answer: 2, explain: 'It is 0.886: the nearest of 300 points is 12 away and the farthest 13.5. Almost every point is about as far as every other.' } },
      { topic: 'curse-dim', frame: 3, caption: '10 dimensions: the nearest is 0.585 away and the farthest 1.67, a ratio of 0.35. The histogram is bunching up.' },
      { topic: 'curse-dim', frame: 6, caption: '1000 dimensions: nearest 12, farthest 13.5, ratio 0.886. "Nearest" no longer means much.' },
      'Real data are seldom uniform in all their dimensions: images and measurements usually lie near a surface of far fewer dimensions, and there neighbours still mean something. The defences are fewer, better features (selection, PCA), more data, and a model that weighs features instead of treating them all alike.',
      { h: 'Recap' },
      { list: ['As dimensions grow, the nearest and farthest distances bunch together.', 'The ratio nearest / farthest climbs towards 1: 0.0000263 in one dimension, 0.886 in a thousand.', 'Remedies: drop useless features, reduce dimensions, get more data.'] },
      X('nn-x-curse-ratio'), X('nn-x-curse-curve'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: naive Bayes
  // =========================================================================================================
  C.lesson({
    id: 'nn-gnb', chapter: 'nn-bayes', title: 'Gaussian naive Bayes', minutes: 8,
    summary: 'Naive Bayes models how each class generates its data, then turns that round with Bayes’ rule. It assumes the features are independent within a class, so each class needs only a mean and a variance per feature: a bell curve along each axis.',
    blocks: [
      'k-NN asks "which stored points are near?". Another way is to ask "what does a typical ashbell look like, and how well does this flower fit that picture?" and pick the class it fits best.',
      { math: 'P(class | x) ∝ P(class) · P(x | class)\nnaive:  P(x | class) = P(x₁ | class) · P(x₂ | class) · …' },
      { topic: 'nb-gauss', frame: 1, caption: 'Class ashbell: prior 0.333, means 2.02 and 0.62, variances 0.157 and 0.0136. The bells along the edges are its two normal curves.' },
      { topic: 'nb-gauss', frame: 4, caption: 'Each place takes the class with the largest prior times bell heights. The boundaries curve because the classes have different variances. Right on 91% of the training points.' },
      { predict: { q: 'A probe at (4.4, 1.26). Its log scores are ashbell −33.00, copperleaf −1.78 and moonwort −3.98. Which class wins, and roughly how surely?', options: ['Moonwort, narrowly', 'Copperleaf, by a wide margin', 'Copperleaf, with probability about 0.9'], answer: 2, explain: 'Copperleaf has the largest score. Its lead over moonwort is 2.2 in logs, a factor e^2.2 ≈ 9, so copperleaf gets about 0.9 and moonwort most of the rest.' } },
      { topic: 'nb-gauss', frame: 5, caption: 'The probe: add the log prior and the log heights of the two bells, then exponentiate and normalise. Copperleaf with 0.9.' },
      { h: 'When independence fails' },
      { topic: 'nb-gauss', input: { set: 'tilted', n: 120, seed: 5, qx: 0.48, qy: 0.52 }, frame: 3, caption: 'Tilted clouds: within each class the two features rise together, but naive Bayes can only draw ellipses along the axes. Still 88% right on the training points: the assumption is wrong, yet the boundary is often good enough.' },
      { workflow: 'nb-moons', title: 'Gaussian naive Bayes on the moons', text: 'An axis-aligned ellipse per class, and the boundary where they meet.' },
      { h: 'Recap' },
      { list: ['Bayes’ rule: prior times likelihood, then normalise.', 'Naive: features independent within a class, so one bell per feature per class.', 'Fast, needs little data, gives probabilities; misses correlations inside a class.'] },
      X('nn-x-gnb-loglik'), X('nn-x-gnb-posterior'), X('nn-x-gnb-tilted'),
    ],
  });

  C.lesson({
    id: 'nn-spam', chapter: 'nn-bayes', title: 'A spam filter, word by word', minutes: 8,
    summary: 'Multinomial naive Bayes counts words. Each word of a message adds the log of P(word | spam) / P(word | not spam) to the log-odds of spam, starting from the prior log-odds; the sigmoid of the total is the probability of spam.',
    blocks: [
      '"Are you free for lunch? Reply today." A friend, or an advertiser? A filter has read 90 messages, 40 of them spam. Which words in this one should worry it?',
      { topic: 'nb-spam', frame: 0, caption: 'Before any word: the prior log-odds are ln(40 / 50) = −0.22, a slight lean towards "not spam".' },
      { math: 'P(word | spam) = (count in spam + α) / (words in spam + α · vocabulary)\nlog-odds += ln P(word | spam) − ln P(word | not spam)' },
      { topic: 'nb-spam', frame: 3, caption: '"free" appeared 12 times in spam and never elsewhere: it adds 2.55, and the log-odds jump to 1.32.' },
      { predict: { q: '"lunch" appeared 0 times in spam and twice in the other messages. Which way does it push?', options: ['Towards spam', 'Away from spam', 'Neither: it was never seen in spam, so it is ignored'], answer: 1, explain: 'It adds −1.12: a word common in ordinary messages and absent from spam is evidence against spam. Smoothing gives it a small probability in spam rather than none.' } },
      { topic: 'nb-spam', frame: 8, caption: 'After 7 words the log-odds are 3.40, so P(spam) = 1 / (1 + e^−3.40) = 0.968: spam. Shuffle the words and nothing changes: only counts matter.' },
      { workflow: 'text-spam-bow', title: 'Words to counts to a model', text: 'Messages turned into word counts on the canvas, then a logistic regression on the counts. Swap in naive Bayes (multinomial) and compare.' },
      { tip: 'Spam filters face senders who adapt, padding messages with ordinary words. The defence is to retrain often on fresh mail and to combine word evidence with other signals, not to trust any single word.' },
      { h: 'Recap' },
      { list: ['Start from the prior log-odds, ln(spam messages / other messages).', 'Each word adds ln P(word | spam) − ln P(word | not spam).', 'P(spam) is the sigmoid of the total; word order is ignored.'] },
      X('nn-x-nb-word-prob'), X('nn-x-nb-posterior'), X('nn-x-nb-logodds'), X('nn-x-nb-order'), X('nn-x-code-nb'),
    ],
  });

  C.lesson({
    id: 'nn-smoothing', chapter: 'nn-bayes', title: 'Smoothing: the word never seen', minutes: 6,
    summary: 'Without smoothing, a word that never appeared in spam has probability 0 there, and one such word vetoes spam whatever the rest says. Adding α to every count fixes it; too large an α flattens all the evidence.',
    blocks: [
      '"Your lunch for today." Three words lean towards spam, but "lunch" never once appeared in a spam message. Should one word ever overrule all the others?',
      { topic: 'nb-smoothing', frame: 1, caption: 'α = 0 (computed with 10⁻¹⁰): ln P(lunch | spam) = −29.23, and the message is spam with probability below 0.001. One unseen word vetoes spam.' },
      { predict: { q: 'Raise α from 0 to 1. What happens to P(spam) for this message?', options: ['It stays near 0', 'It rises a lot', 'It rises to exactly 0.5'], answer: 1, explain: 'With α = 1, P(lunch | spam) = 1 / (496 + 435) instead of almost 0, so the other words get their say: P(spam) = 0.817.' } },
      { topic: 'nb-smoothing', frame: 5, caption: 'α = 1, add-one or Laplace smoothing: P(spam) = 0.817.' },
      { topic: 'nb-smoothing', frame: 7, caption: 'α = 5: every word probability drifts towards 1 / 435 and the evidence weakens: P(spam) falls back to 0.779.' },
      { table: [['α', '0', '0.01', '0.1', '0.5', '1', '2', '5'], ['P(spam)', '< 0.001', '0.109', '0.521', '0.776', '0.817', '0.822', '0.779']], caption: 'The same message under each α. Like k for k-NN, α is chosen on held-out messages.' },
      { h: 'Recap' },
      { list: ['A zero count makes a probability 0 and its logarithm −∞: one word vetoes a class.', 'Add α to every count (α = 1 is Laplace smoothing).', 'Too much α pulls every word towards the same probability and weakens the evidence.'] },
      X('nn-x-nb-zero'), X('nn-x-nb-alpha-curve'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: discriminant analysis
  // =========================================================================================================
  C.lesson({
    id: 'nn-lda-qda', chapter: 'nn-da', title: 'LDA and QDA: a Gaussian per class', minutes: 9,
    summary: 'Discriminant analysis fits a full Gaussian to each class, tilt included. LDA makes every class share one covariance, which makes the boundary straight; QDA gives each class its own, which bends it. QDA needs more data to estimate its shapes.',
    blocks: [
      'Gaussian naive Bayes draws only ellipses along the axes. What if each class is allowed a tilted ellipse, and must every class have the same one?',
      { topic: 'lda-qda', frame: 1, caption: 'Both models start alike: a mean and a prior per class, here (−1.39, −0.269) and (1.2, 0.313), with priors 0.5 and 0.5.' },
      { topic: 'lda-qda', frame: 2, caption: 'LDA pools every class’s spread into one covariance and draws the same ellipse at every mean.' },
      { topic: 'lda-qda', frame: 4, caption: 'QDA gives each class its own: class 1 is wide along x₁ (variance 2.03) and narrow along x₂ (0.215), unlike class 0.' },
      { predict: { q: 'When every class shares one covariance, the quadratic terms of the two log-likelihoods are equal. What shape is the boundary?', options: ['A straight line', 'A circle', 'A parabola'], answer: 0, explain: 'The x·Σ⁻¹·x terms cancel in the difference, leaving a linear function: LDA’s log-odds here are 2.33·x₁ + 0.693·x₂ + 0.205, and its boundary is the line where that is 0.' } },
      { topic: 'lda-qda', frame: 6, caption: 'On 400 fresh points: LDA 89%, QDA 91%. QDA wins because the classes really do differ in shape. It estimates 10 numbers for its means and covariances, LDA 7.' },
      { topic: 'lda-qda', input: { data: 'diff', n: 14, seed: 5 }, frame: 5, caption: 'With 14 points QDA must guess each class’s shape from 7: 100% on its training points, against 93% for LDA. More numbers to estimate means more room to overfit.' },
      { table: [['Model', 'Per class', 'Boundary', 'Needs'], ['Gaussian naive Bayes', 'axis-aligned ellipse', 'curved', 'little data'], ['LDA', 'one shared tilted ellipse', 'straight', 'moderate data'], ['QDA', 'its own tilted ellipse', 'curved', 'the most data']], caption: 'Three generative classifiers, from fewest numbers to most.' },
      { workflow: flow({ set: 'blobs', n: 200, noise: 0.5, seed: 4 }, 'qda', {}), live: true, height: 460, caption: 'QDA on three blobs, running here. Swap the model block for LDA and compare the boundaries in Evaluate.' },
      { h: 'Recap' },
      { list: ['LDA: one shared covariance, a linear boundary, fewer parameters.', 'QDA: a covariance per class, a quadratic boundary, more data needed.', 'Naive Bayes is the special case with diagonal covariances.'] },
      X('nn-x-lda-qda'), X('nn-x-qda-count'), X('nn-x-lda-score'), X('nn-x-wf-flowers'),
      { challenge: 'nn-ch-pumps' }, { challenge: 'nn-ch-exam' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
