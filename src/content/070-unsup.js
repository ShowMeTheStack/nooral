// Part 7, unsupervised learning: the chapters and the lessons. The exercises are in 071-unsup-ex.js; the model
// challenge, the tour, the course and the pages the diagram checker looks at are in 072-unsup-ch.js. Each lesson opens
// with a question, runs the topic that shows the idea at a frame chosen for the point being made, asks for a
// prediction, sums up, and leads to its exercises. Every value quoted here was printed by the topics themselves
// (S.ML's k-means, linkage, DBSCAN, mixtures, PCA, t-SNE, UMAP-style maps and isolation forests on the topics' inputs).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('un-km')) return;
  const X = (id) => ({ exercise: id });
  /** a small workflow for the canvas card: blocks {id: [type, params, x, y]}, wires ['a.out', 'b.in'] */
  const wf = (blocks, wires) => (S.WF ? S.WF.build({ blocks, wires }) : 'first-pipeline');

  C.chapter({ id: 'un-kmeans', part: 'unsupervised', n: 1, title: 'k-means', blurb: 'Assign every point to its nearest centre, move every centre to the mean of its points, repeat; start the centres far apart; and choose k by the elbow and the silhouette.' });
  C.chapter({ id: 'un-shapes', part: 'unsupervised', n: 2, title: 'Clusters of other shapes', blurb: 'A tree of merges you can cut at any height, clusters grown through crowded neighbourhoods with noise left over, and soft clusters of any ellipse by EM.' });
  C.chapter({ id: 'un-reduce', part: 'unsupervised', n: 3, title: 'Fewer dimensions', blurb: 'PCA finds the directions of greatest spread, tells how much each one holds, and rebuilds the data from a few; t-SNE and UMAP-style maps keep neighbours together.' });
  C.chapter({ id: 'un-anomaly', part: 'unsupervised', n: 4, title: 'Odd ones out', blurb: 'An isolation forest cuts at random and measures how quickly each point stands alone: the odd ones need few cuts.' });

  // =========================================================================================================
  // Chapter 1: k-means
  // =========================================================================================================
  C.lesson({
    id: 'un-km', chapter: 'un-kmeans', title: 'k-means step by step', minutes: 8,
    summary: 'k-means places k centres, then repeats two steps: every point joins its nearest centre, and every centre moves to the mean of its points. Each step can only lower the inertia, the summed squared distance to the centres, so it stops when no point changes.',
    blocks: [
      'A delivery firm wants three depots for its customers in a city, each depot close to the customers it serves. No one has labelled the customers; there are only their positions. Where should the depots go?',
      { def: 'Inertia', text: 'The sum, over every point, of its squared distance to the centre it is assigned to. k-means tries to make it small.' },
      { topic: 'km-steps', frame: 0, caption: '3 centres start on the points nearest the bottom-left corner: a poor start, on purpose, so the steps show.' },
      { topic: 'km-steps', frame: 1, caption: 'Assign: every point joins its nearest centre; the shading shows each centre’s region. The clusters have 18, 65 and 7 points and the inertia is 1289.08.' },
      { predict: { q: 'Next, each centre moves to the mean of its points. What can happen to the inertia?', options: ['It can only fall or stay the same', 'It can rise if a centre moves the wrong way', 'It always halves'], answer: 0, explain: 'For a fixed group of points, their mean is the place with the smallest summed squared distance. So moving there cannot make the inertia larger.' } },
      { topic: 'km-steps', frame: 2, caption: 'Move: each centre goes to the mean of its points. The orange centre moves furthest, by 3.402, and the inertia falls to 425.26.' },
      { topic: 'km-steps', frame: 3, caption: 'Assign again: with the centres in new places, 21 points change cluster and the inertia falls to 373.90.' },
      { topic: 'km-steps', frame: 11, caption: 'After 6 assignments not one point changes its centre: k-means has converged, with inertia 91.28.' },
      { steps: ['Pick k starting centres.', 'Assign: each point joins the centre at the smallest distance.', 'Update: each centre moves to the mean of the points assigned to it.', 'Repeat until no point changes its centre.'] },
      { warn: 'k-means always converges, but only to a local minimum of the inertia: a different start can give a different, worse answer. The next lesson starts the centres more wisely.' },
      { h: 'Try it on the canvas' },
      { workflow: wf({ data1: ['data', { set: 'blobs', n: 300 }, 40, 40], kmeans1: ['kmeans', { k: 3 }, 300, 40] }, [['data1.out', 'kmeans1.in']]), title: 'k-means on blobs', text: 'A k-means block on three round blobs. Open it to step through the assignments and the moves, and double-click it to change k.' },
      { h: 'Recap' },
      { list: ['Assign each point to its nearest centre; move each centre to the mean of its points; repeat.', 'Neither step can raise the inertia, so k-means stops, usually after a few rounds.', 'It finds a local minimum, which depends on the start.'] },
      X('un-x-assign'), X('un-x-update'), X('un-x-inertia'), X('un-x-code-kmeans'),
    ],
  });

  C.lesson({
    id: 'un-kpp', chapter: 'un-kmeans', title: 'Better starts: k-means++', minutes: 6,
    summary: 'k-means++ picks the first centre at random and every next one with probability in proportion to D², its squared distance to the nearest centre chosen so far. Far points are likely picks, so the centres start spread out, and k-means then ends lower and more reliably.',
    blocks: [
      'If two starting centres land in the same group, k-means may never pull one of them out: another group then gets no centre of its own. How can the start avoid that without knowing the groups?',
      { topic: 'km-plusplus', frame: 0, caption: 'The first centre is point 56, at random. Every point is drawn with a size that grows with D², its squared distance to that centre: the total D² is 1487.3.' },
      { def: 'k-means++ seeding', text: 'Choose the next centre at random, with point x picked with probability D(x)² / Σ D². Points far from every centre so far are the likely picks.' },
      { predict: { q: 'A point sits right on top of a centre already chosen. What is its chance of being picked next?', options: ['Zero: its D² is 0', 'The same as any other point', 'The highest of all'], answer: 0, explain: 'Its D² is 0, so its share of the total is 0: k-means++ never puts two centres on the same spot.' } },
      { topic: 'km-plusplus', frame: 1, caption: 'Centre 2 is point 54. The engine drew 3 candidates in proportion to D² and kept the one that lowered the total most: the total falls to 699.4.' },
      { topic: 'km-plusplus', frame: 4, caption: 'Over ten seeds, k-means started this way ends with mean inertia 78.7 and reaches the best value found, 78.6, in 6 of the ten runs. From 4 random points the mean is 93.3, and the best is reached in 1 run of ten.' },
      { tip: 'Libraries also run k-means several times from different starts and keep the lowest inertia. k-means++ makes each run better; several runs make a bad one less likely.' },
      { h: 'Recap' },
      { list: ['k-means++ picks each new centre with probability in proportion to D².', 'The centres start spread out, one in each group far more often.', 'It is the default start of k-means in most libraries.'] },
      X('un-x-kpp-prob'),
    ],
  });

  C.lesson({
    id: 'un-choosek', chapter: 'un-kmeans', title: 'How many clusters?', minutes: 8,
    summary: 'The inertia always falls as k grows, so its smallest value is no guide. Look for the elbow, where adding a centre stops paying much; or compute the silhouette, which compares each point’s distance to its own cluster with the distance to the nearest other one.',
    blocks: [
      'With more centres every customer is closer to a depot, and with one depot per customer the inertia is zero. So how many clusters does the data really have?',
      { topic: 'km-elbow', frame: 3, caption: 'Four groups of points. Going from 3 centres to 4 cuts the inertia from 232.7 to 78.6, by 66%: the fourth centre found a group of its own.' },
      { predict: { q: 'And from 4 centres to 5?', options: ['Another big drop, about 66%', 'A small drop', 'The inertia rises'], answer: 1, explain: 'A fifth centre has no new group to find: it splits one that was already well served. The drop is 16%.' } },
      { topic: 'km-elbow', frame: 8, caption: 'The curve bends at k = 4, the point furthest below the straight line from k = 1 to k = 8: the elbow.' },
      { topic: 'km-elbow', frame: 8, input: { set: 'moons', kmax: 8 }, caption: 'Two moons have no real elbow: going to 3 centres cuts 35% and the next one 34%. The moons are not round groups, so no k suits k-means well.' },
      { h: 'The silhouette' },
      { def: 'Silhouette of a point', text: 'a = its mean distance to the other points of its own cluster; b = its mean distance to the points of the nearest other cluster. s = (b − a) / max(a, b), between −1 and 1: near 1 fits well, near 0 sits on a border, negative is probably in the wrong cluster.' },
      { topic: 'km-silhouette', frame: 1, caption: 'Point 22, in the blue cluster: a = 1.404 and b = 4.011, so s = (4.011 − 1.404) / 4.011 = 0.650. It fits well.' },
      { topic: 'km-silhouette', frame: 3, caption: 'Every point’s silhouette as a bar. The mean is 0.655 and no point is negative.' },
      { topic: 'km-silhouette', frame: 3, input: { set: 'three', k: 5 }, caption: 'Too many clusters: with k = 5 on the same three groups the mean falls to 0.444 and 2 points turn negative.' },
      { topic: 'km-silhouette', frame: 4, caption: 'The mean silhouette for k = 2 to 6 is highest, 0.655, at k = 3: the k to choose.' },
      { h: 'Recap' },
      { list: ['The inertia always falls with k; look for the elbow where the fall slows.', 'The silhouette s = (b − a) / max(a, b) compares a point’s own cluster with the nearest other one.', 'Choose the k with the highest mean silhouette, and look at the clusters: the numbers only advise.'] },
      X('un-x-elbow'), X('un-x-silhouette'), X('un-x-sil-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: clusters of other shapes
  // =========================================================================================================
  C.lesson({
    id: 'un-hier', chapter: 'un-shapes', title: 'A tree of merges', minutes: 9,
    summary: 'Agglomerative clustering starts with every point alone and merges the two closest clusters, again and again, until one is left. The dendrogram records every merge at its height; cutting it at any height gives a clustering. The linkage says what “closest” means between clusters.',
    blocks: [
      'A biologist wants to group bird songs by similarity, but does not know how many kinds there are, and would like to see the groups within groups too. Can a clustering give every level at once?',
      { topic: 'hc-build', frame: 0, caption: '18 points, each a cluster of its own. With average linkage, the distance between two clusters is the mean distance over every pair of their points.' },
      { topic: 'hc-build', frame: 1, caption: 'Merge 1: the two closest points join at distance 0.131. The dendrogram on the right draws the join at that height.' },
      { topic: 'hc-build', frame: 16, caption: 'Merge 16 joins two clusters of 6 at distance 4.385: a tall join means two well-separated groups.' },
      { topic: 'hc-build', frame: 18, caption: 'Cut the tree at height 2.285: it crosses 4 branches, so the points fall into 4 clusters. Move the cut to get more or fewer, without clustering again.' },
      { table: [['Linkage', 'Distance between clusters A and B'], ['single', 'the closest pair, one point from each'], ['complete', 'the farthest pair'], ['average', 'the mean over all pairs'], ['Ward', 'how much merging them adds to the spread within clusters']], caption: 'Four meanings of “closest clusters”.' },
      { predict: { q: 'Two round groups are joined by a thin bridge of points. Which linkage is likely to follow the bridge and call it all one cluster?', options: ['Single', 'Complete', 'Ward'], answer: 0, explain: 'Single linkage needs only one close pair to merge two clusters, so it walks along the bridge point by point: it chains.' } },
      { topic: 'hc-linkage', frame: 1, caption: 'The bridge, cut for 2 clusters: single linkage gives sizes 19 and 11. It has chained along the bridge into one of the groups.' },
      { topic: 'hc-linkage', frame: 4, caption: 'Ward gives 15 and 15: it prefers clusters of similar spread, much as k-means does.' },
      { topic: 'hc-linkage', frame: 1, input: { set: 'moons', n: 40, k: 2 }, caption: 'Where chaining helps: on two moons single linkage gives 20 and 20, one moon each, while Ward gives 31 and 9.' },
      { h: 'Recap' },
      { list: ['Start with every point alone; merge the two closest clusters; repeat.', 'The dendrogram holds every merge; a cut at any height gives a clustering.', 'Single linkage chains along thin paths; complete keeps clusters compact; Ward makes even, round clusters.'] },
      X('un-x-single-next'), X('un-x-complete-next'), X('un-x-link-dist'),
    ],
  });

  C.lesson({
    id: 'un-dbscan', chapter: 'un-shapes', title: 'DBSCAN: crowds and noise', minutes: 8,
    summary: 'DBSCAN calls a point core if at least minPts points lie within ε of it, itself included. A cluster grows from a core point through every core point it can reach; points near a core point but not crowded themselves are border points; the rest are noise. It finds clusters of any shape and needs no k.',
    blocks: [
      'GPS fixes of a city’s taxis waiting for fares bunch up at ranks of every shape: along a street, round a square. A few fixes are strays from passing cabs. Can a clustering find the ranks without being told how many, and leave the strays out?',
      { topic: 'dbscan-core', frame: 0, caption: 'Point 8 with its circle of radius ε = 0.6. To be a core point it needs at least minPts = 5 points in the circle, itself included.' },
      { topic: 'dbscan-core', frame: 1, caption: 'Point 8 has only 2 points within ε: not core. Counting every point the same way, 78 of the 110 are core points (filled).' },
      { topic: 'dbscan-core', frame: 2, caption: 'Cluster 1 starts at core point 0: it takes every point within ε, then the points within ε of each core point it reached, and so on: 31 points.' },
      { predict: { q: 'A point lies within ε of a core point, but has only 3 neighbours itself (minPts = 5). What is it?', options: ['A core point', 'A border point: in the cluster, but it cannot extend it', 'Noise'], answer: 1, explain: 'It joins the cluster of the core point that reaches it, but a cluster never grows through it.' } },
      { topic: 'dbscan-core', frame: 5, caption: '3 clusters, with 78 core points, 14 border points (hollow) and 18 noise points (grey), which belong to no cluster.' },
      { topic: 'dbscan-core', frame: 4, input: { set: 'moons', eps: 0.5, minPts: 5 }, caption: 'On two moons DBSCAN finds both crescents, 2 clusters, with 1 noise point. k-means would cut each moon in half.' },
      { h: 'Choosing ε' },
      { topic: 'dbscan-params', frame: 0, caption: 'Each point’s distance to its 5th nearest point, sorted. The curve is flat for points in crowds and climbs for isolated ones; it bends at about 0.759, a natural ε for minPts = 5.' },
      { topic: 'dbscan-params', frame: 1, caption: 'ε = 0.379, too strict: 4 clusters and 56 noise points, whole stretches of clusters called noise.' },
      { topic: 'dbscan-params', frame: 5, caption: 'ε = 1.517, too loose: everything runs into 1 cluster.' },
      { h: 'Recap' },
      { list: ['Core: at least minPts points within ε (itself included). Border: within ε of a core point but not core. Noise: neither.', 'Clusters grow from core points through core points, so they can have any shape.', 'Choose ε near the bend of the sorted k-distance curve; DBSCAN struggles when the crowds have very different densities.'] },
      X('un-x-dbscan-labels'), X('un-x-shape-choice'), X('un-x-eps-effect'),
    ],
  });

  C.lesson({
    id: 'un-gmm', chapter: 'un-shapes', title: 'Gaussian mixtures by EM', minutes: 9,
    summary: 'A Gaussian mixture models the data as k bell-shaped clouds, each with a mean, a covariance and a weight. EM alternates: the E-step gives every point a responsibility from each component, the M-step refits each component to the points weighted by those responsibilities. Clusters are soft, and can be any ellipse.',
    blocks: [
      'k-means draws its borders halfway between centres, which suits round clusters of equal size. What if the clusters are long, tilted ellipses, or a point really is halfway between two of them?',
      { topic: 'gmm-em', frame: 0, caption: '3 components start on randomly picked points, each a wide round cloud with equal weight. The solid ellipse is one standard deviation, the dashed one two.' },
      { def: 'Responsibility', text: 'r_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j): the share of point i that component k claims. Each point’s responsibilities add up to 1.' },
      { topic: 'gmm-em', frame: 1, caption: 'Iteration 1. E-step: every point gets its responsibilities. M-step: each component takes the responsibility-weighted mean and covariance of the points. The green mean moves furthest, 0.641.' },
      { predict: { q: 'Can the log-likelihood fall from one EM iteration to the next?', options: ['No: each E and M step can only raise it or keep it', 'Yes, often', 'Only on the first iteration'], answer: 0, explain: 'Like k-means for the inertia, EM never makes its objective worse; it stops when the gain falls under a tolerance.' } },
      { topic: 'gmm-em', frame: 9, caption: 'Iteration 15: the log-likelihood rose by less than the tolerance, so EM stops at −2.3280. The ellipses have stretched along the groups.' },
      { topic: 'gmm-em', frame: 10, caption: 'The answer is soft: the ringed point 13 is 67% orange and 33% blue, where two ellipses overlap.' },
      { h: 'How many components?' },
      { def: 'BIC', text: '−2 · log-likelihood + (number of parameters) · ln n. The fit is rewarded and every parameter is charged; the lowest BIC wins.' },
      { topic: 'gmm-bic', frame: 2, caption: 'k = 3: log-likelihood −209.5 with 17 parameters, so BIC = 419.0 + 17 × ln 90 = 495.5, lower than with 2 components.' },
      { topic: 'gmm-bic', frame: 3, caption: 'k = 4: the log-likelihood improves only to −208.2 while the parameters grow to 23: BIC 519.9. The extra component does not pay for itself.' },
      { h: 'Recap' },
      { list: ['A mixture has a mean, a covariance and a weight per component.', 'E-step: responsibilities. M-step: weighted means, covariances and weights. The log-likelihood never falls.', 'Clusters are soft and elliptical; choose k by the lowest BIC.'] },
      X('un-x-em-resp'), X('un-x-em-mean'), X('un-x-bic'), X('un-x-gmm-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: fewer dimensions
  // =========================================================================================================
  C.lesson({
    id: 'un-pca', chapter: 'un-reduce', title: 'PCA: the direction of greatest spread', minutes: 8,
    summary: 'PCA centres the data and finds the direction along which it spreads the most, the first principal component, then the next at right angles, and so on. They are the eigenvectors of the covariance matrix, and the eigenvalues are the variances along them.',
    blocks: [
      'A pair of measurements that rise and fall together, like a child’s height and arm span, carry almost one number’s worth of information. How can we find that one number?',
      { topic: 'pca-project', frame: 0, caption: '60 points with their mean (2.11, 0.93). The covariance matrix has variances 1.989 and 1.330 and covariance 1.313.' },
      { topic: 'pca-project', frame: 1, caption: 'A direction at 90° from the best one: the points’ shadows on it have variance only 0.306.' },
      { predict: { q: 'Turning the direction towards the best one, what happens to the variance of the shadows?', options: ['It grows', 'It shrinks', 'It stays the same: rotation does not change variance'], answer: 0, explain: 'At 55° away it is 1.196, at 25° it is 2.529, and along the best direction 3.013.' } },
      { topic: 'pca-project', frame: 4, caption: 'PC1 = (0.788, 0.615), the covariance matrix’s top eigenvector: variance λ₁ = 3.013, more than any direction tried. PC2 is at right angles with λ₂ = 0.306. Together they hold all the variance: 91% and 9%.' },
      { math: 'Σ v = λ v     (Σ the covariance matrix, v a component, λ the variance along it)\nexplained variance ratio of PC k = λ_k / (λ₁ + λ₂ + …)' },
      { topic: 'pca-project', frame: 5, caption: 'Keep only PC1: each point slides onto the line and is described by one number. The squared error is 18.053 = 59 × 0.306: what PCA drops is exactly the variance along the components it leaves out.' },
      { warn: 'PCA follows variance, so a feature measured in large units dominates. Scale the features first unless their units are comparable.' },
      { h: 'Recap' },
      { list: ['Centre the data; PC1 is the direction of greatest variance, each next PC the greatest at right angles to those before.', 'The components are the covariance matrix’s eigenvectors; the eigenvalues are the variances along them.', 'Keeping k components loses exactly the variance of the rest.'] },
      X('un-x-pca-ratio'), X('un-x-pca-score'), X('un-x-code-power'),
    ],
  });

  C.lesson({
    id: 'un-pcavar', chapter: 'un-reduce', title: 'How many components to keep', minutes: 7,
    summary: 'In many dimensions, the explained variance ratio of each component says how much of the spread it holds; the running total says how many to keep. A few components often hold most of it, and the data can be rebuilt from them with little loss.',
    blocks: [
      'A wine laboratory measures 13 chemical properties of every bottle. Are there really 13 independent things going on?',
      { topic: 'pca-variance', frame: 0, caption: '178 wines with 13 features. PC1 carries 36% of the variance, PC2 19% and PC3 11%; the running total passes 90% at 8 components.' },
      { predict: { q: 'The last 5 of the 13 components together hold how much of the spread?', options: ['Under a tenth', 'About a third', 'About half'], answer: 0, explain: 'The first 8 pass 90%, so the other 5 hold under 10%.' } },
      { topic: 'pca-variance', frame: 1, caption: 'PC1 is a weighted sum of the 13 features; its largest weights are on flavanoids (0.42) and total phenols (0.39).' },
      { topic: 'pca-variance', frame: 3, caption: 'Every wine placed by its first two scores: 55% of the variance in one flat picture. The colours are the known kinds, which PCA never used, yet they fall into separate regions.' },
      { h: 'Rebuilding images' },
      { topic: 'pca-recon', frame: 1, caption: '150 digit images of 100 pixels each, rebuilt from 1 component: 16% of the pixel variance kept, and only a blur.' },
      { topic: 'pca-recon', frame: 5, caption: 'With 20 components 89% of the variance is kept and the digits are easy to read: 20 numbers per image instead of 100.' },
      { h: 'Try it on the canvas' },
      { workflow: wf({ data1: ['data', { set: 'wildflowers', n: 300 }, 40, 40], scale1: ['scale', { how: 'standard' }, 300, 40], pca1: ['pca', { n: 2 }, 560, 40] }, [['data1.out', 'scale1.in'], ['scale1.out', 'pca1.in']]), title: 'PCA of the wildflowers', text: 'Scale, then PCA to two components. Open the PCA block for the explained variance and the flowers placed by their two scores.' },
      { h: 'Recap' },
      { list: ['The explained variance ratio of each component is its eigenvalue over the total.', 'Keep enough components for the running total you need, often 90% or 95%.', 'The data rebuilt from k components keeps exactly that share of the variance.'] },
      X('un-x-pca-k'), X('un-x-cumvar-curve'), X('un-x-wf-pca'),
    ],
  });

  C.lesson({
    id: 'un-maps', chapter: 'un-reduce', title: 'Maps that keep neighbours together', minutes: 8,
    summary: 't-SNE and UMAP-style maps place points in 2-D so that points close in the data stay close on the map. They keep neighbourhoods, not distances: the size of a cluster on the map and the gaps between clusters mean little.',
    blocks: [
      'PCA is a projection: it keeps the large-scale spread and can squash separate groups on top of each other. If what matters is which points are near which, can a map keep that instead?',
      { topic: 'tsne-run', frame: 0, caption: '48 points in 8 dimensions start at random, almost on top of one another. Each point’s neighbourhood is set by perplexity 12: about that many effective neighbours.' },
      { topic: 'tsne-run', frame: 6, caption: 'For the first 250 iterations the attractions are multiplied by 12, so groups pull into tight clumps.' },
      { topic: 'tsne-run', frame: 10, caption: 'Iteration 299: points close in 8 dimensions sit together. Trustworthiness, 1 when each point’s 5 nearest map neighbours are its nearest in the data, is 0.984.' },
      { predict: { q: 'On the finished map, one cluster looks twice as wide as another and far from it. What can you conclude?', options: ['Little: t-SNE keeps neighbours, not sizes or gaps', 'The wide cluster is twice as spread in the data', 'The two clusters are the most different groups'], answer: 0, explain: 't-SNE adapts its scale to each neighbourhood, so sizes and distances between clusters are not to be read off the map.' } },
      { topic: 'tsne-run', frame: 10, input: { set: 'blobs', perp: 2, seed: 1 }, caption: 'Perplexity 2: tiny neighbourhoods. The trustworthiness drops to 0.849 and the groups break into fragments.' },
      { h: 'A UMAP-style map' },
      { topic: 'umap-run', frame: 0, caption: 'The fuzzy graph: each of the 48 points links to its 10 nearest neighbours; merging the two directions leaves 286 edges. The layout starts from a scaled PCA projection.' },
      { topic: 'umap-run', frame: 7, caption: 'Epoch 200: edges pull their ends together and random pairs push apart, until the layout settles. Trustworthiness 0.965.' },
      { h: 'Recap' },
      { list: ['t-SNE matches neighbour probabilities in the data and on the map; perplexity sets the neighbourhood size.', 'A UMAP-style map lays out a k-nearest-neighbour graph by pulling along edges and pushing random pairs.', 'Both keep neighbourhoods; cluster sizes and gaps on the map should not be read as distances.'] },
      X('un-x-map-choice'), X('un-x-perplexity'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: odd ones out
  // =========================================================================================================
  C.lesson({
    id: 'un-iforest', chapter: 'un-anomaly', title: 'Isolation forests: odd points are easy to cut off', minutes: 8,
    summary: 'An isolation forest cuts the data at random: a random feature, a random value in its range, again and again. A point off on its own is isolated after a few cuts; one in a crowd needs many. The average number of cuts, over many trees, becomes a score: near 1 is an anomaly.',
    blocks: [
      'Fraud is rare and takes forms no one has seen yet, so there are no labels to learn from. Is there a way to find what is odd without defining what odd looks like?',
      { topic: 'iforest-cuts', frame: 0, caption: '60 points. Follow two through one random tree: point 55, off on its own, and point 1, in the middle of the crowd.' },
      { topic: 'iforest-cuts', frame: 1, caption: 'Cut 1 for point 55: x₂ at 2.44. Its side keeps 1 point of 60: it is isolated after one cut.' },
      { predict: { q: 'How many cuts will point 1, in the crowd, need on average?', options: ['About 1, like point 55', 'Many more: around 10', 'It can never be isolated'], answer: 1, explain: 'Each random cut shaves off only a few of its neighbours. Over 100 trees it needs 10.33 on average.' } },
      { topic: 'iforest-cuts', frame: 7, caption: 'Point 1 after 6 cuts still shares its cell with 32 points; the depth limit adds 6.08 for the cuts they would still need: path length 12.08.' },
      { def: 'Anomaly score', text: 's(x) = 2^(−E[h(x)] / c(n)), where E[h(x)] is the mean path length over the trees and c(n) the average path length of a search in a random tree of n points. Near 1: an anomaly; about 0.5 or below: ordinary.' },
      { topic: 'iforest-cuts', frame: 8, caption: 'Over 100 trees point 55 needs 4.20 cuts and point 1 needs 10.33. With c = 7.34 for 60 points the scores are 0.673 and 0.377.' },
      { topic: 'iforest-scores', frame: 1, caption: 'Pump readings, scored over the plane. With contamination 5% the threshold is 0.580: 15 readings above it are flagged.' },
      { topic: 'iforest-scores', frame: 2, caption: 'Of the 15 flagged, 13 are real faults: precision 87%, and 68% of the 19 faults caught. Pressure-drop faults barely change these two readings, so no method looking at them alone could catch those.' },
      { tip: 'An anomaly score says “unusual”, not “bad”. Use it to direct attention, and check what it flags; and give it the features in which the faults you care about would show.' },
      { h: 'Recap' },
      { list: ['Random cuts isolate odd points quickly and crowded ones slowly.', 'The score 2^(−E[h] / c(n)) is near 1 for anomalies, about 0.5 or below for ordinary points.', 'Contamination sets the share of points flagged; the one-class SVM of part 6 does the same job another way.'] },
      X('un-x-iforest-score'), X('un-x-iforest-choice'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
