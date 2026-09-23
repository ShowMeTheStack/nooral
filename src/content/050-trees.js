// Part 5, trees and ensembles: the chapters and the lessons. The exercises are in 051-trees-ex.js; the model
// challenges, the tour, the course and the pages the diagram checker looks at are in 052-trees-ch.js. Each lesson
// opens with a question, runs the topic that shows the idea at a frame chosen for the point being made, asks for a
// prediction, sums up, and leads to its exercises. Every value quoted here was printed by the topics themselves
// (S.ML's trees, forests and boosters on the topics' own inputs).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('tr-impurity')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'tr-grow', part: 'trees', n: 1, title: 'Growing a tree', blurb: 'Gini and entropy measure how mixed a node is; every threshold is weighed and the best one kept, split after split, for classes and for numbers.' });
  C.chapter({ id: 'tr-size', part: 'trees', n: 2, title: 'How big a tree should be', blurb: 'Depth and the smallest leaf trade a close fit for a steady one; cost-complexity pruning cuts the weakest links of a grown tree.' });
  C.chapter({ id: 'tr-forest', part: 'trees', n: 3, title: 'Bagging and random forests', blurb: 'Many trees on bootstrap samples, a few features per split, a free test set from the rows each tree never saw, and two ways to ask which features mattered.' });
  C.chapter({ id: 'tr-boost', part: 'trees', n: 4, title: 'Boosting', blurb: 'AdaBoost reweights its mistakes, gradient boosting fits what is left over at a gentle rate, and five tree models meet on the same points.' });

  // =========================================================================================================
  // Chapter 1: growing a tree
  // =========================================================================================================
  C.lesson({
    id: 'tr-impurity', chapter: 'tr-grow', title: 'How mixed is a node?', minutes: 7,
    summary: 'A tree judges a split by how pure it leaves the two sides. Gini impurity is the chance that two points drawn from a node disagree; entropy counts the bits needed to name a point’s class. Both are 0 for a pure node and largest for an even mix.',
    blocks: [
      'A doctor’s rule of thumb, a bank’s loan checklist and a field guide to wild flowers all have the same shape: ask one yes-or-no question, then another depending on the answer, until the answer is clear. A decision tree learns such a checklist from data. Before it can choose a question, it needs a number for “clear”.',
      { def: 'Gini impurity', text: 'of a node whose classes have shares p₁, …, p_K: G = 1 − Σ p_k². It is the chance that two points drawn at random from the node (with replacement) belong to different classes.' },
      { topic: 'cart-grow', frame: 0, caption: 'Sixty points of two moons, 30 of each class, all in the root. With shares ½ and ½, G = 1 − ¼ − ¼ = 0.5, the worst a two-class node can be.' },
      { predict: { q: 'A node holds 1 point of class 0 and 21 of class 1. Roughly what is its Gini impurity?', options: ['About 0.09', 'About 0.5', 'About 0.95', 'Exactly 0'], answer: 0, explain: '1 − (1/22)² − (21/22)² = 42/484 ≈ 0.0868. One stray point keeps it off zero, but only just.' } },
      { topic: 'cart-grow', frame: 2, caption: 'The first split, x2 ≤ 0.0185, sends 1 | 21 left (Gini 0.0868) and 29 | 9 right (0.361). The split is scored by the average of the two, weighted by size: (22 · 0.0868 + 38 · 0.361) / 60 = 0.261, down from 0.5.' },
      { h: 'Entropy' },
      { def: 'Entropy', text: 'H = −Σ p_k log₂ p_k, in bits: 0 for a pure node, 1 for an even two-class mix.' },
      { topic: 'cart-grow', frame: 2, input: { criterion: 'entropy' }, caption: 'The same split under entropy: the left side has 0.267 bits, the right 0.79, and the weighted average falls from 1 to 0.598. Entropy picked the same cut as Gini here, as it usually does.' },
      { table: [['Node', 'Gini', 'Entropy (bits)'], ['30 | 30', '0.5', '1'], ['1 | 21', '0.0868', '0.267'], ['29 | 9', '0.361', '0.79'], ['pure', '0', '0']], caption: 'Both measures rank nodes alike. Entropy punishes a small minority a little more, and Gini is a little cheaper to compute: no logarithms.' },
      { tip: 'Accuracy would be a poor judge of splits. A split can leave the majority class on both sides, so the accuracy does not move, while one side has become much purer. Gini and entropy see that progress; accuracy does not.' },
      { h: 'Recap' },
      { list: ['Gini impurity is 1 − Σ p_k²: the chance that two random points of the node disagree.', 'Entropy is −Σ p_k log₂ p_k bits. Both are 0 when pure and largest for an even mix (0.5 and 1 for two classes).', 'A split is scored by the size-weighted average impurity of its two sides; the lower, the better.'] },
      X('tr-x-gini-node'), X('tr-x-entropy-node'), X('tr-x-gini-entropy'), X('tr-x-split-table'), X('tr-x-code-gini'),
    ],
  });

  C.lesson({
    id: 'tr-greedy', chapter: 'tr-grow', title: 'Every threshold weighed, split by split', minutes: 9,
    summary: 'At each node CART tries every feature and every threshold halfway between neighbouring values, keeps the cut with the lowest weighted impurity, and repeats on each side. It never looks ahead, so a problem whose first good cut only pays off later can defeat it.',
    blocks: [
      'With 60 points and two features there are 118 places a first cut could go. Which one does the tree take, and does taking the best cut now give the best tree in the end?',
      { topic: 'cart-grow', frame: 1, caption: 'Node 0 tries 118 thresholds. The best on x2 leaves a weighted Gini of 0.261; the best on x1 only reaches 0.324. The candidate curves along the axes show the score of every threshold.' },
      { steps: ['Sort the node’s points by one feature.', 'Try a threshold halfway between each pair of neighbouring values; points with x ≤ t go left.', 'Score each threshold by the weighted Gini of the two sides.', 'Do the same for every feature, keep the lowest, and split.', 'Repeat on each child until it is pure, too small, or at the depth limit.'] },
      { topic: 'cart-grow', frame: 7, caption: 'Node 4 (29 | 9) on the right gets its own search: 74 thresholds, best x1 ≤ 1.67 with weighted Gini 0.185. The children of a node never look at points outside it.' },
      { topic: 'cart-grow', frame: 14, caption: 'Finished: 4 splits, 5 leaves, depth 3, and 56 of 60 training points (93%) on their own class’s side. Every cut runs along an axis, so the boundary is a staircase.' },
      { predict: { q: 'In the XOR pattern, opposite corners share a class. What does the best single first cut do?', options: ['Separates the classes almost perfectly', 'Barely lowers the impurity: every vertical or horizontal cut leaves both classes on each side', 'Splits off one whole class', 'It cannot be computed'], answer: 1, explain: 'Any axis cut through the middle leaves one corner of each class on either side. The tree takes the best of a bad lot and only a later cut finishes the job.' } },
      { topic: 'cart-grow', frame: 2, input: { set: 'xor', n: 48, depth: 3, criterion: 'gini', seed: 2 }, caption: 'On XOR the best first cut, x2 ≤ 0.867, only shaves off 3 points: the weighted Gini falls from 0.5 to 0.467. Greedy search has no way to know that a cut through the middle would pay off one level down.' },
      { h: 'Try it on the canvas' },
      { workflow: 'tree-moons', title: 'A tree on the moons', text: 'A decision tree trained on the two moons. Open the Train block to watch the tree grow and the boundary fill in, and double-click the model to change its depth.' },
      { h: 'Recap' },
      { list: ['Candidate thresholds sit halfway between neighbouring values; x ≤ t goes left.', 'Each node keeps the feature and threshold with the lowest weighted impurity of its children.', 'The search is greedy: it never looks ahead, which is why patterns like XOR give it trouble.', 'Axis-parallel cuts make a staircase boundary.'] },
      X('tr-x-best-threshold'), X('tr-x-candidates'), X('tr-x-stump-matrix'), X('tr-x-xor-choice'), X('tr-x-code-split'),
    ],
  });

  C.lesson({
    id: 'tr-regtree', chapter: 'tr-grow', title: 'Regression trees: a mean in every leaf', minutes: 7,
    summary: 'For a number instead of a class, a leaf predicts the mean of its points, and a split is chosen to make the sum of squared errors around the two means as small as possible. The prediction is a staircase of flat steps.',
    blocks: [
      'A tree can predict a number too: the yield of a field, the delay of a bus. What should a leaf say, and how is a cut judged when there are no classes to be pure?',
      { topic: 'reg-tree', frame: 0, caption: 'Forty points on a noisy sine. As one leaf the tree predicts their mean, −0.0307, everywhere; the squared error around it averages 0.441 per point.' },
      { topic: 'reg-tree', frame: 2, caption: 'The first cut, x ≤ −0.0985, puts 21 points on the left (mean −0.571) and 19 on the right (mean 0.566). The sum of squared errors falls from 17.7 to 4.75: the impurity here is the variance, and a split is scored by how much it removes.' },
      { predict: { q: 'The finished tree is asked about an x far to the right of every training point. What does it predict?', options: ['A value that keeps following the sine', 'The mean of the right-most leaf, a flat value', 'Zero', 'It refuses to predict'], answer: 1, explain: 'Every x beyond the last threshold lands in the same leaf, whose prediction is the mean of its training points. A tree cannot extrapolate a trend.' } },
      { topic: 'reg-tree', frame: 23, caption: 'Depth 3 gives 8 flat steps. The mean squared error on the training points is 0.0151 (R² = 0.966), down from 0.441 for the single mean.' },
      { math: 'leaf value = mean of its y\nsplit score = SSE(left) + SSE(right),  SSE = Σ (y − mean)²' },
      { h: 'Recap' },
      { list: ['A regression leaf predicts the mean of its training targets.', 'Splits minimise the summed squared error of the two sides (the variance times the count).', 'The prediction is piecewise constant and flat outside the training range.'] },
      X('tr-x-sse-split'), X('tr-x-reg-predict'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: how big a tree should be
  // =========================================================================================================
  C.lesson({
    id: 'tr-depth', chapter: 'tr-size', title: 'Depth and the smallest leaf', minutes: 7,
    summary: 'Left alone, a tree grows until every leaf is pure and fits every training point, noise included. Limiting the depth or demanding a minimum number of points per leaf stops it earlier; the right limit is the one that scores best on fresh data.',
    blocks: [
      'A tree that keeps splitting will eventually give each noisy point its own leaf and score 100% on its training data. Is that the tree you want?',
      { topic: 'tree-size', frame: 0, caption: 'Depth 1, a single cut: 83.8% of the 80 training points and 73.0% of 300 fresh points. Too simple: it underfits.' },
      { topic: 'tree-size', frame: 4, caption: 'Depth 5: 8 leaves, 97.5% on training and 82.3% on fresh points. Look at the small boxes the tree has drawn around single stray points.' },
      { predict: { q: 'Going from depth 5 to depth 12, the training accuracy rises to 100%. What happens to the accuracy on fresh points?', options: ['It rises to about 100% too', 'It stays about the same, at about 82%', 'It drops to about 50%'], answer: 1, explain: 'The extra leaves only wrap noise points, which tells nothing about new ones. Here the test score stays at 82%; on other data it falls.' } },
      { topic: 'tree-size', frame: 7, caption: 'Depth 12 stops at 12 leaves and depth 7: every leaf is pure and the training score is 100%. The best test score, 82.3%, came at depth 5, but every depth from 2 on is within one point of it: about 82%. Depth 2, the simplest of them, is the one to keep.' },
      { h: 'The smallest leaf' },
      { topic: 'tree-size', frame: 4, input: { knob: 'leaf' }, caption: 'Instead of depth, demand at least 6 points in every leaf: 6 leaves, 96.3% on training and 83.0% on fresh points. A minimum leaf size stops splits where they would chase single points, wherever they are in the tree.' },
      { topic: 'tree-size', frame: 7, input: { knob: 'leaf' }, caption: 'With leaves of 1 the tree has 12 leaves again and 100% on training, but 82.0% on fresh points: the best, 83.0%, came at a minimum leaf of 6.' },
      { warn: 'Choose the depth or the leaf size by a validation set or cross-validation, never by the training score: the training score always prefers the biggest tree.' },
      { h: 'Recap' },
      { list: ['An unlimited tree fits its training data perfectly, noise and all.', 'Max depth caps how many questions are asked; min leaf caps how few points a question may decide for.', 'Pick either by the score on data the tree did not see.'] },
      X('tr-x-curve-depth'), X('tr-x-overfit-choice'), X('tr-x-minleaf'),
    ],
  });

  C.lesson({
    id: 'tr-prune', chapter: 'tr-size', title: 'Cost-complexity pruning', minutes: 8,
    summary: 'Grow the full tree, then cut it back. Each internal node t has a price g(t): the impurity its subtree saves per extra leaf. The node with the smallest g is the weakest link and is collapsed first; raising α walks through a whole sequence of smaller trees.',
    blocks: [
      'A depth limit stops a split before seeing what follows it. A cut that looks useless may enable a good one below it, as on XOR. Could we grow the whole tree first and then remove the parts that do not earn their keep?',
      { def: 'Cost complexity', text: 'R_α(T) = R(T) + α·|leaves of T|, where R(T) is the total weighted impurity of the leaves. α is the price of one leaf.' },
      { topic: 'ccp-prune', frame: 0, caption: 'The full tree on 40 noisy moon points: 9 pure leaves, total impurity 0, 81% on 300 test points. Node 7’s subtree has 3 leaves with impurity 0; as one leaf it would have 0.0455. So g = (0.0455 − 0) / (3 − 1) = 0.0227.' },
      { math: 'g(t) = (R(t) − R(T_t)) / (|leaves of T_t| − 1)\nR(t): the node as a single leaf     T_t: its subtree' },
      { predict: { q: 'The weakest link has g = 0.0227. What happens to it once α passes 0.0227?', options: ['Nothing yet', 'Its subtree is collapsed into one leaf', 'The whole tree is cut to the root'], answer: 1, explain: 'At α above g(t), keeping the subtree costs more in leaves than it saves in impurity, so it is replaced by a leaf. Other subtrees, with larger g, stay.' } },
      { topic: 'ccp-prune', frame: 2, caption: 'At α = 0.0234 the tree is down to 5 leaves (impurity 0.0923, 81% on test). Next weakest: node 3, g = (0.13 − 0.0455) / 2 = 0.042.' },
      { topic: 'ccp-prune', frame: 3, caption: 'At α = 0.042: 3 leaves, impurity 0.176, and 85% on the test points, the best of the whole path.' },
      { topic: 'ccp-prune', frame: 5, caption: 'At α = 0.255 only the root is left: 50%. Every α gives one tree on the path; cross-validation picks the α.' },
      { h: 'Recap' },
      { list: ['Pruning grows the full tree and then collapses subtrees, weakest link first.', 'g(t) = (R(t) − R(T_t)) / (leaves − 1): the impurity saved per extra leaf.', 'Each α > 0 gives one tree of the pruning path; choose α on held-out data.'] },
      X('tr-x-ccp-g'), X('tr-x-ccp-which'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: bagging and random forests
  // =========================================================================================================
  C.lesson({
    id: 'tr-bagging', chapter: 'tr-forest', title: 'Bagging: many trees, one vote', minutes: 8,
    summary: 'A deep tree changes a lot when its training rows change: high variance. Bagging trains many deep trees, each on a bootstrap sample of the rows drawn with replacement, and lets them vote. The vote keeps the trees’ flexibility and averages away much of their wobble.',
    blocks: [
      'Train the same deep tree on two samples of the same data and the boundaries differ in many places. If each tree is right in a different place and wrong in a different place, what happens when they vote?',
      { def: 'Bootstrap sample', text: 'n rows drawn from the n training rows with replacement. Some rows appear several times and, on average, a share (1 − 1/n)ⁿ ≈ 1/e ≈ 0.368 never appears.' },
      { topic: 'bagging', frame: 1, caption: 'Tree 1 drew 36 different rows out of 60 (17 of them more than once, drawn larger); 24 rows were never drawn. Alone it scores 83% on 300 test points.' },
      { predict: { q: 'Nine trees score between 81% and 93% alone, 86% on average. Where will their majority vote land?', options: ['Below the average tree, about 80%', 'Near the average, about 86%', 'Above the average, near the best single tree'], answer: 2, explain: 'Their mistakes are partly independent, so the majority is wrong less often than a typical member: the vote scores 89%.' } },
      { topic: 'bagging', frame: 10, caption: 'Nine trees score 86% on average and 93% at best; their vote scores 89%, with a smoother boundary. The pale bands are where the trees disagree.' },
      { tip: 'Bagging helps models with high variance, like deep trees. A model that barely changes with its sample, such as a stump or a linear model, gains little from it: its trees would all vote alike.' },
      { h: 'Recap' },
      { list: ['Each tree trains on its own bootstrap sample of the rows.', 'About 37% of rows are left out of each sample.', 'The majority vote (or the mean, for numbers) cancels much of the trees’ variance.'] },
      X('tr-x-oob-share'), X('tr-x-bootstrap-set'),
    ],
  });

  C.lesson({
    id: 'tr-forest', chapter: 'tr-forest', title: 'Random forests and the out-of-bag score', minutes: 9,
    summary: 'If one feature is strong, bagged trees all split on it first and look alike, so their votes are correlated. A random forest offers each split only a random few features, which makes the trees differ more. The rows a tree never saw give a free test: the out-of-bag score.',
    blocks: [
      'Bagging works best when the trees make different mistakes. But if every bootstrap sample favours the same strong feature, every tree starts with the same question. How can we make them disagree more, without making each one much worse?',
      { topic: 'rf-features', frame: 1, caption: 'Small loans, 6 features. The root of the first tree may only try income and debt: it splits on debt ≤ 0.394 (weighted Gini 0.36). The other 4 features were not offered here.' },
      { topic: 'rf-features', frame: 8, caption: 'Over 30 trees, with 2 features per split, 6 different features start a tree; with all 6 offered (plain bagging), only 4 do. On 300 test rows the forest scores 75% and the bagged trees 72%.' },
      { predict: { q: 'Offered only 2 of 6 features at each split, each single tree is…', options: ['better than a bagged tree', 'a little worse, but the forest as a whole is better', 'much worse, and so is the forest'], answer: 1, explain: 'A tree sometimes has to make do with a weaker feature, so it gets slightly worse; but the trees are less alike, so the vote improves more than the members lose.' } },
      { h: 'A test set for free' },
      { topic: 'rf-oob', frame: 1, caption: 'Tree 1 left 16 of 40 rows out of its sample, so it can vote on those as if they were new. Its votes are right for 13 of them (81%).' },
      { topic: 'rf-oob', frame: 10, caption: 'After 10 trees every row has had a vote from trees that never saw it. The out-of-bag score is 85%; 300 truly fresh points give 90%. No rows had to be set aside.' },
      { h: 'On the canvas' },
      { workflow: 'forest-loans', title: 'A forest on the small loans', text: 'A random forest on the loans table, with the evaluation and its confusion matrix; open the model block to see the trees.' },
      { h: 'Recap' },
      { list: ['A random forest is bagging plus a random subset of features (often √d) at every split.', 'Less alike trees give a better vote, even though each tree is a little weaker.', 'Each row is scored only by the trees that left it out: the out-of-bag score, a built-in estimate of test accuracy.'] },
      X('tr-x-maxfeatures'), X('tr-x-oob-vote'), X('tr-x-wf-moons'),
    ],
  });

  C.lesson({
    id: 'tr-importance', chapter: 'tr-forest', title: 'Which features mattered?', minutes: 7,
    summary: 'Impurity importance adds up how much each feature’s splits lowered the impurity during training; it is quick, but it credits noise columns that happened to make splits. Permutation importance shuffles one column on fresh rows and measures how much the score drops; it asks what the model really relies on.',
    blocks: [
      'A forest of 25 trees predicts who passes an exam from study hours, sleep and practice papers. To test the methods, two columns were added that should not matter: pure noise, and a near copy of study. Which features does each method say mattered?',
      { topic: 'rf-importance', frame: 0, caption: 'Impurity importance: study did the most to lower the impurity, 0.286 of the total. Even the noise column gets 0.134: deep trees find something to split on in any column with many distinct values.' },
      { topic: 'rf-importance', frame: 1, caption: 'Permutation importance starts from the score on test rows the forest never saw: 74% of 300. Then each column is shuffled 5 times, breaking its link to the target.' },
      { predict: { q: 'Shuffling the noise column on fresh rows will cost about how much accuracy?', options: ['About 13 points, in line with its impurity share', 'About nothing: the forest’s accuracy does not depend on it', 'All of it'], answer: 1, explain: 'Shuffling a column the model does not really need leaves its predictions on fresh rows about as good. Here it even rose a little: the drop was −4.1 points, chance at work.' } },
      { topic: 'rf-importance', frame: 6, caption: 'Shuffling noise costs −4.1 points on average: no loss at all. Sleep costs 7.5 points and study 7.1.' },
      { topic: 'rf-importance', frame: 7, caption: 'The near copy of study shares the credit. Shuffling the copy alone costs little (0.33 points), because study still carries the information; study itself still costs 7.1. Correlated features hide each other from permutation importance.' },
      { h: 'Recap' },
      { list: ['Impurity importance is computed on the training data and favours columns with many split points, noise included.', 'Permutation importance is the drop in a score on held-out rows when one column is shuffled.', 'Correlated features split their importance; read both measures with that in mind.'] },
      X('tr-x-importance'), X('tr-x-perm-drop'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: boosting
  // =========================================================================================================
  C.lesson({
    id: 'tr-adaboost', chapter: 'tr-boost', title: 'AdaBoost: weights grow on mistakes', minutes: 9,
    summary: 'Boosting builds its learners one after another, each aimed at what the earlier ones got wrong. AdaBoost gives every point a weight, trains a stump on the weighted points, gives the stump a say α from its weighted error ε, and multiplies the weights of its mistakes by e^α before the next round.',
    blocks: [
      'Bagging trains its trees side by side, each blind to the others. What if each new learner could see where the ones before it went wrong, and concentrate there?',
      { topic: 'adaboost', frame: 0, caption: 'Forty points, each with weight 1/40 = 0.025. A dot’s size shows its weight.' },
      { math: 'ε = (weight of the misclassified points) / (total weight)\nα = ln((1 − ε) / ε)\nweight of each mistake × e^α, then all weights rescaled to sum to 1' },
      { topic: 'adaboost', frame: 1, caption: 'Round 1: the stump misses 6 points carrying 0.15 of the weight. So α = ln(0.85 / 0.15) = 1.73, and the 6 weights are multiplied by e^α = 5.67: the next stump will care about them much more.' },
      { predict: { q: 'A round with ε = 0.5, no better than a coin, would get what say α?', options: ['α = 0: its vote counts for nothing', 'α = 1', 'α = ln 2', 'A negative α'], answer: 0, explain: 'ln((1 − 0.5) / 0.5) = ln 1 = 0. A better learner (smaller ε) gets a larger α; this engine stops boosting if a round is no better than chance.' } },
      { topic: 'adaboost', frame: 5, caption: 'Round 5 misses 25 points, but they carry only 0.306 of the weight, so α = 0.82. Its learner was aimed at the heavy points, and the vote of 5 rounds now gets 98% right.' },
      { topic: 'adaboost', frame: 9, caption: 'After 8 rounds the vote is right on every training point. The heaviest point carries 0.0814 of the weight, 3.26 times its start: boosting fought over the points near the boundary.' },
      { workflow: 'ada-xor', title: 'AdaBoost on XOR', text: 'Stumps cannot draw XOR on their own; a weighted vote of them can. Open the model block to step through the rounds.' },
      { h: 'Recap' },
      { list: ['Each round trains a weak learner on weighted points.', 'Its say is α = ln((1 − ε)/ε): large when the weighted error ε is small.', 'Misclassified points are multiplied by e^α and all weights rescaled, so the next learner aims at them.', 'The prediction is the α-weighted vote of all rounds.'] },
      X('tr-x-ada-alpha'), X('tr-x-ada-weights'), X('tr-x-code-ada'),
    ],
  });

  C.lesson({
    id: 'tr-gboost', chapter: 'tr-boost', title: 'Gradient boosting fits the residuals', minutes: 9,
    summary: 'Gradient boosting starts from a constant, the mean, and then adds small trees one at a time, each fitted to the residuals of the model so far. Each tree is added at a learning rate ν < 1, so the model creeps towards the data; smaller steps need more stages but usually generalise better.',
    blocks: [
      'A first model gets the rough shape of a curve and misses here and there. The misses, y − ŷ, are themselves a pattern. What if the next model learned the misses, and we added it on?',
      { topic: 'gb-residuals', frame: 0, caption: 'Stage 0 predicts the mean, 0.0917, everywhere. The residuals below the plot are what it gets wrong; their mean square is 0.554.' },
      { topic: 'gb-residuals', frame: 1, caption: 'Stage 1: a depth-2 tree is fitted to the residuals; its 4 leaves predict from −0.783 to 0.782. Added at rate 0.5 (dashed), it moves the model to the solid line, and the training squared error falls to 0.186.' },
      { math: 'F₀(x) = mean of y\nr = y − Fₘ₋₁(x)            (the residuals)\nFₘ(x) = Fₘ₋₁(x) + ν · treeₘ(x)   (treeₘ fitted to r)' },
      { predict: { q: 'Stage after stage, what happens to the size of the residuals the next tree must fit?', options: ['They grow', 'They shrink, and so do the leaf values of each new tree', 'They stay the same'], answer: 1, explain: 'Each stage removes part of what was left; by stage 6 the new tree’s leaves only span −0.109 to 0.042.' } },
      { topic: 'gb-residuals', frame: 6, caption: 'Stage 6: the training squared error is 0.0151 and the latest tree is tiny. For squared error the residual is exactly minus the gradient of the loss, which gives the method its name.' },
      { h: 'The learning rate' },
      { topic: 'gb-shrinkage', frame: 0, caption: 'Three runs on the same noisy sine with depth-3 trees. After 1 stage, rate ν = 1 has training error 0.0899, ν = 0.3 has 0.307 and ν = 0.1 has 0.449: small steps start slowly.' },
      { topic: 'gb-shrinkage', frame: 9, caption: 'After 60 stages ν = 1 fits the training points exactly (error 0) and has a test error of 0.285; ν = 0.1 keeps a training error of 0.0212 and the lowest test error, 0.195. Small steps, many stages, and stop when the validation error turns up.' },
      { workflow: 'boost-curve', title: 'Gradient boosting on a curve', text: 'Gradient boosting regression on a one-dimensional curve: open the model block to watch the stages add up.' },
      { h: 'Recap' },
      { list: ['Start from the mean; each stage fits a small tree to the current residuals.', 'Add each tree times ν: Fₘ = Fₘ₋₁ + ν·treeₘ.', 'Smaller ν needs more stages and overfits later; choose the number of stages on validation data.'] },
      X('tr-x-gb-stage'), X('tr-x-shrink-choice'), X('tr-x-wf-curve'),
    ],
  });

  C.lesson({
    id: 'tr-compare', chapter: 'tr-boost', title: 'Five tree models on the same points', minutes: 7,
    summary: 'One deep tree, bagging, a random forest, AdaBoost and gradient boosting, trained on the same points and scored on the same fresh ones. Here all five fit their training points; they differ on fresh ones, where the single tree does worst and the ensembles’ steadier boundaries do better.',
    blocks: [
      'Every model in this part was built from the same piece: a tree. Put them side by side on one problem. Which is best, and is the answer the same on every dataset?',
      { topic: 'ensemble-compare', frame: 1, caption: 'One deep tree on 60 noisy moon points: 100% on training and 86% on 300 fresh points, with a jagged boundary.' },
      { topic: 'ensemble-compare', frame: 3, caption: 'A random forest of 25 trees: also 100% on training, but 89% on fresh points and a smoother boundary.' },
      { topic: 'ensemble-compare', frame: 6, caption: 'Best on fresh points here: bagging with 89%, against 86% for the single tree. The gap between training and test accuracy is the variance the ensembles remove.' },
      { predict: { q: 'On a cleaner problem, the ring-and-disc circles, will the ensembles beat the single tree by as much?', options: ['By much more', 'By less: the single tree is already good when the data are clean', 'The single tree will win by far'], answer: 1, explain: 'With little noise there is less variance to remove. On the circles the single tree gets 92% and the best ensemble 93%.' } },
      { topic: 'ensemble-compare', frame: 6, input: { set: 'circles', n: 100, noise: 0.12, trees: 25, seed: 2 }, caption: 'Circles with little noise: the single tree gets 92%, AdaBoost with stumps 93%. The ensembles still help, by one point instead of three.' },
      { table: [['Model', 'Built', 'Reduces', 'Watch out for'], ['One tree', 'greedy splits', '—', 'high variance; easy to read'], ['Bagging', 'trees in parallel on bootstrap samples', 'variance', 'trees too alike'], ['Random forest', 'bagging + a few features per split', 'variance, more', 'importance on noise columns'], ['AdaBoost', 'stumps in sequence, reweighting mistakes', 'bias', 'label noise gets heavy weights'], ['Gradient boosting', 'small trees on residuals at rate ν', 'bias, then variance', 'too many stages at a large ν']], caption: 'What each tree model is for.' },
      { workflow: 'compare-three', title: 'Compare models on the canvas', text: 'Three models trained on one split and compared side by side; swap one for a forest or gradient boosting.' },
      { h: 'Recap' },
      { list: ['Averaging (bagging, forests) lowers variance; boosting lowers bias step by step.', 'A single tree is easiest to explain; ensembles are usually more accurate on fresh data.', 'Compare on held-out data, never on the training points.'] },
      X('tr-x-which-model'), X('tr-x-wf-circles'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
