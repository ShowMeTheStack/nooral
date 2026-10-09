// Part 16, graph neural networks: the chapters and the lessons. The exercises are in 161-gnn-ex.js; the tour, the
// course and the pages the diagram checker looks at are in 162-gnn-ch.js. Each lesson opens with a question, runs the
// topic that shows the idea at a frame chosen for the point being made, asks for a prediction, sums up, and leads to
// its exercises. Every value quoted here was printed by the topics themselves (S.GNN on the topics' own graphs).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('gn-graphs')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'gn-graphs', part: 'graphs', n: 1, title: 'Graphs as tables of links', blurb: 'Nodes, links and the adjacency matrix; degrees, mirror symmetry and two-step walks; and the self loops and scaling a GCN adds before it mixes anything.' });
  C.chapter({ id: 'gn-passing', part: 'graphs', n: 2, title: 'Message passing and the GCN layer', blurb: 'A node listens to its neighbours: messages weighed and added, three ways to weigh them, and the whole layer written as two matrix products and a ReLU.' });
  C.chapter({ id: 'gn-nodes', part: 'graphs', n: 3, title: 'Learning node labels', blurb: 'Labelling a whole graph from a few labelled nodes, why the links help when neighbours share a class, and why too many layers make every node alike.' });
  C.chapter({ id: 'gn-whole', part: 'graphs', n: 4, title: 'Whole graphs', blurb: 'One vector per graph from any number of nodes: mean, sum and max readouts, and why the answer must not depend on how the nodes are numbered.' });

  // =========================================================================================================
  // Chapter 1: graphs as tables of links
  // =========================================================================================================
  C.lesson({
    id: 'gn-graphs', chapter: 'gn-graphs', title: 'A graph and its adjacency matrix', minutes: 6,
    summary: 'A graph is a set of nodes and the links between them. Written as a square table, the adjacency matrix A, row i has a 1 in column j when i and j are linked. The ones in a row count the node’s degree; for two-way links the table is its own mirror image.',
    blocks: [
      'Seven people, and who is friends with whom. A table of rows and columns has no place for “who knows whom”, yet that is often the most telling thing about a person. How do we hand a friendship network to a model that only multiplies matrices?',
      { topic: 'graph-adjacency', frame: 0, caption: 'The friends graph: 7 nodes, so A is 7 × 7, and 8 links. Every cell starts blank.' },
      { def: 'Adjacency matrix', text: 'A square table with one row and one column per node: A[i][j] = 1 when nodes i and j are linked, else 0.' },
      { topic: 'graph-adjacency', frame: 3, caption: 'Row 2 fills in: node 2 is linked to nodes 0, 1 and 3, so its row has three ones, and its degree is 3.' },
      { predict: { q: 'The 8 links are all two-way. How many ones will the finished matrix hold?', options: ['8', '16', '49'], answer: 1, explain: 'Each link is written twice, at (i, j) and at (j, i): 16 ones, which is also the sum of the degrees.' } },
      { topic: 'graph-adjacency', frame: 8, caption: 'The finished matrix equals its own transpose. The 8 links give 16 ones, and the degrees add up to 16 as well.' },
      { topic: 'graph-adjacency', frame: 11, input: { graph: 'citations', loops: 'no' }, caption: 'Citations run one way, so this A is not symmetric: 14 ones, one per citation. A GCN lets information flow both ways, so it adds the outlined cells: 28 ones, degrees from 1 to 5.' },
      { tip: 'Each node also carries its own features, a row of a second table H: a person’s age and interests, a paper’s words, an atom’s element. A graph model uses both tables.' },
      { h: 'Recap' },
      { list: ['A[i][j] = 1 when i and j are linked; the ones in row i are node i’s degree.', 'Two-way links make A symmetric, and the ones add up to twice the number of links.', 'One-way links (citations) are usually taken both ways, A + Aᵀ, before a GCN uses them.'] },
      X('gn-x-adj-fill'), X('gn-x-degrees'), X('gn-x-links'), X('gn-x-directed'),
    ],
  });

  C.lesson({
    id: 'gn-walks', chapter: 'gn-graphs', title: 'Two steps away: the square of A', minutes: 6,
    summary: 'Multiplying A by itself counts walks: entry (i, j) of A² is the number of two-step walks from i to j, the neighbours i and j share. Powers of A are how far information can travel in so many rounds of message passing.',
    blocks: [
      'Two people are not friends but share three friends. Are they in the same circle? Counting shared neighbours by eye is slow on a large graph. Is there a matrix that does it?',
      { math: '(A²)[i][j] = Σₖ A[i][k] · A[k][j]\n         = the number of nodes k linked to both i and j' },
      { topic: 'graph-adjacency', frame: 9, caption: 'In the friends graph, nodes 0 and 1 share 1 neighbour, node 2: A² has 1 there, the most of any pair.' },
      { predict: { q: 'In the citations graph, taken both ways, papers 0 and 5 share three neighbours. What does A² hold at (0, 5)?', options: ['1', '3', '0, because they are not linked'], answer: 1, explain: 'Each shared neighbour k gives one walk 0 → k → 5, so the entry is 3.' } },
      { topic: 'graph-adjacency', frame: 12, input: { graph: 'citations', loops: 'no' }, caption: 'Papers 0 and 5 share 3 neighbours, 1, 6 and 7, so A² has 3 there. Two rounds of message passing reach exactly these two-step neighbours.' },
      { p: 'The same holds further out: (Aᵏ)[i][j] counts walks of k steps. A node that is k links away from another can first be heard after k rounds of message passing, which chapter 2 builds.' },
      { h: 'Recap' },
      { list: ['(A²)[i][j] counts the neighbours i and j share: walks of two steps.', 'Aᵏ counts walks of k steps.', 'k rounds of message passing let a node hear nodes up to k links away.'] },
      X('gn-x-a2-entry'), X('gn-x-a2-matrix'), X('gn-x-hops'),
    ],
  });

  C.lesson({
    id: 'gn-norm', chapter: 'gn-graphs', title: 'Self loops and scaling: Â', minutes: 7,
    summary: 'Before mixing, a GCN adds a self loop to every node, A + I, so a node keeps its own features, and divides entry (i, j) by √(d̂ᵢ d̂ⱼ), where d̂ counts the loop. The result, Â, keeps busy nodes from swamping their neighbours.',
    blocks: [
      'If a node simply added up its neighbours’ numbers, a hub with a hundred links would get a total a hundred times larger than a node with one, and its own features would be lost entirely. What should the weights be?',
      { topic: 'graph-adjacency', frame: 15, input: { graph: 'communities', loops: 'yes' }, caption: 'Three clubs with self loops: the identity puts a one on the diagonal, every degree grows by one (they now run from 2 to 5), and the matrix holds 42 ones.' },
      { math: 'Â = D̂^−½ (A + I) D̂^−½\nÂ[i][j] = 1 / √(d̂ᵢ · d̂ⱼ)  when i = j or i and j are linked, else 0\nd̂ᵢ = degree of i + 1   (the loop counts)' },
      { topic: 'gcn-layer', frame: 0, caption: 'The friends graph with self loops: d̂ runs from 3 to 4, and node 2 has d̂ = 4.' },
      { predict: { q: 'Node 0 has d̂ = 3 and node 2 has d̂ = 4. What weight does Â give the link between them?', options: ['1 / 7', '1 / √12 ≈ 0.289', '1 / 4 = 0.25'], answer: 1, explain: '1 / √(3 × 4) = 1 / √12 ≈ 0.289: both ends are taken into account, symmetrically.' } },
      { topic: 'gcn-layer', frame: 1, caption: 'The link from node 0 (d̂ = 3) to node 2 (d̂ = 4) gets 0.289. Row sums are no longer whole numbers; here they run from 0.91 to 1.12, so no node’s total runs away.' },
      { topic: 'gcn-layer', frame: 1, input: { graph: 'star', norm: 'none', node: 0, seed: 1 }, caption: 'With no scaling, Â is just A + I: the hub, node 0, adds up 8 rows at full weight while the quietest node adds only 2. Row sums run from 2 to 8.' },
      { topic: 'gcn-layer', frame: 1, input: { graph: 'clusters', norm: 'mean', node: 0, seed: 3 }, caption: 'The other common choice divides each row by its own d̂, so it sums to one: node 0 gives each of its 5 entries 0.200, a plain average.' },
      { h: 'Recap' },
      { list: ['A + I lets each node keep its own features.', 'Â[i][j] = 1 / √(d̂ᵢ d̂ⱼ): busy senders and busy receivers are both turned down.', 'Without scaling, totals grow with the degree; with the mean, each row sums to one.'] },
      X('gn-x-loop-deg'), X('gn-x-ahat-entry'), X('gn-x-ahat'), X('gn-x-ahat-rowsum'), X('gn-x-mean-adj'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: message passing and the GCN layer
  // =========================================================================================================
  C.lesson({
    id: 'gn-message', chapter: 'gn-passing', title: 'Message passing: a node listens', minutes: 8,
    summary: 'In one round of message passing every node gathers its neighbours’ numbers, each times a weight, and adds them into its new numbers. All nodes do it at once, from the old numbers. After k rounds a node has heard from every node up to k links away.',
    blocks: [
      'You want to guess a person’s interests, but know little about them. You do know their friends. A sensible guess mixes what you know of them with what you know of their friends. How would that look as arithmetic?',
      { topic: 'message-passing', frame: 0, caption: 'Every node of the friends graph carries two numbers, h = (h₁, h₂). Follow node 2.' },
      { topic: 'message-passing', frame: 1, caption: 'Node 2 has 3 neighbours, 0, 1 and 3, and with the GCN’s self loop it hears 4 senders. A message from u is scaled by 1 / √(d̂ᵤ · d̂ᵥ).' },
      { topic: 'message-passing', frame: 2, caption: 'From node 0: weight 1 / √(3 × 4) = 0.289, so the message is 0.289 × [0.9, 0.1] = [0.260, 0.029].' },
      { predict: { q: 'The four messages are [0.260, 0.029], [0.231, 0.087], [0.150, 0.050] and [0.115, 0.144]. What is node 2’s new first number?', options: ['0.756', '0.189', '0.600'], answer: 0, explain: '0.260 + 0.231 + 0.150 + 0.115 = 0.756: the messages are added.' } },
      { topic: 'message-passing', frame: 6, caption: 'Adding the 4 messages gives m = [0.756, 0.310]; node 2’s numbers were [0.6, 0.2]. Its own numbers are kept, mixed with its neighbours’.' },
      { topic: 'message-passing', frame: 7, caption: 'Every node does the same at the same moment, each from the old numbers. The largest result is at node 4, [0.28, 0.81].' },
      { topic: 'message-passing', frame: 8, caption: 'A second round: node 2’s value [0.721, 0.311] now depends on 5 nodes, itself, 3 neighbours and node 4, two links away.' },
      { warn: 'All nodes update together, from the numbers of the round before. Updating one node and then letting its neighbours use the new value is a different, order-dependent computation.' },
      { h: 'Recap' },
      { list: ['A round: every node adds up weighted copies of its neighbours’ numbers (and, for a GCN, its own).', 'All nodes update at once, from the previous round’s numbers.', 'k rounds reach k links out.'] },
      X('gn-x-gcn-agg'), X('gn-x-mp-round'), X('gn-x-receptive'), X('gn-x-code-gcn'),
    ],
  });

  C.lesson({
    id: 'gn-rules', chapter: 'gn-passing', title: 'Three ways to weigh the messages', minutes: 7,
    summary: 'The GCN weighs a message by 1 / √(d̂ᵤ d̂ᵥ) and includes the node itself; the mean rule (as in GraphSAGE) averages the neighbours and sets the result beside the node’s own numbers; the sum rule adds everything at full weight, so a hub’s total grows with its degree.',
    blocks: [
      'A star: one hub linked to seven leaves. Should the hub’s new numbers be seven times larger than a leaf’s? It depends on the question. Counting how many friends someone has needs the sum; describing what their friends are like needs an average.',
      { topic: 'message-passing', frame: 9, input: { graph: 'star', node: 0, rule: 'sum', rounds: '1' }, caption: 'The sum rule at the hub: adding the 7 messages gives m = [7.000, 2.750]; the hub’s own numbers were [1.0, 1.8]. Nothing scales the total down.' },
      { predict: { q: 'Under the GCN rule, what will the hub’s weight on its own numbers be? It has 7 neighbours.', options: ['1', '1 / 8 = 0.125', '1 / 7'], answer: 1, explain: 'With the loop, d̂ = 8, and the self weight is 1 / √(8 × 8) = 0.125.' } },
      { topic: 'message-passing', frame: 2, input: { graph: 'star', node: 0, rule: 'gcn', rounds: '1' }, caption: 'The GCN rule at the hub: from itself, weight 1 / √(8 × 8) = 0.125.' },
      { topic: 'message-passing', frame: 10, input: { graph: 'star', node: 0, rule: 'gcn', rounds: '1' }, caption: 'Adding the 8 messages gives m = [1.691, 0.814], not far from the scale of the inputs.' },
      { topic: 'message-passing', frame: 5, input: { graph: 'friends', node: 2, rule: 'sage', rounds: '2' }, caption: 'The mean rule at node 2 of the friends graph: m = [0.700, 0.300], the plain average of its 3 neighbours. A GraphSAGE layer sets this beside the node’s own h, so it can tell itself apart from its neighbours.' },
      { table: [['Rule', 'Weight of a message from u to v', 'The node itself', 'Grows with degree?'], ['GCN', '1 / √(d̂ᵤ d̂ᵥ)', 'included, by the self loop', 'barely'], ['mean (GraphSAGE)', '1 / (degree of v)', 'kept apart, set beside the average', 'no'], ['sum', '1', 'not included', 'yes']], caption: 'The sum keeps counts; the mean and the GCN keep averages.' },
      { h: 'Recap' },
      { list: ['GCN: symmetric scaling and a self loop.', 'Mean: an average of the neighbours, kept beside the node’s own numbers.', 'Sum: full weight, so the total counts the neighbours too.'] },
      X('gn-x-sage-round'), X('gn-x-sum-vs-mean'),
    ],
  });

  C.lesson({
    id: 'gn-layer', chapter: 'gn-passing', title: 'A GCN layer as matrices', minutes: 8,
    summary: 'A GCN layer is H′ = ReLU(Â H W): every node’s features go through the same weights W (as in a dense layer), then Â mixes each node with its neighbours, then the ReLU bends. The weights are shared by all nodes, so the layer’s size does not depend on the graph’s.',
    blocks: [
      'Message passing mixes numbers, but nothing in it learns. Where do the learned weights go, and how many are there for a graph of a million nodes?',
      { math: 'Z  = H W        (every node’s row times the same W)\nM  = Â Z        (each node mixes the rows of itself and its neighbours)\nH′ = ReLU(M)' },
      { topic: 'gcn-layer', frame: 2, caption: 'Z = HW: for node 2, 0.60 × [0.31, −1.22] + 0.20 × [0.07, 1.18] = [0.200, −0.495]. W is shared by all nodes, so the layer has only 4 weights whatever the size of the graph.' },
      { topic: 'gcn-layer', frame: 3, caption: 'Row 2 of Â picks out 4 rows of Z, itself and its neighbours, with their weights, giving [0.256, −0.556].' },
      { predict: { q: 'Node 2 has [0.256, −0.556] after mixing. What leaves the layer after the ReLU?', options: ['[0.256, −0.556]', '[0.256, 0]', '[0, 0]'], answer: 1, explain: 'The ReLU sets every negative number to zero and keeps the rest.' } },
      { topic: 'gcn-layer', frame: 4, caption: 'The ReLU sets 3 of the 14 numbers to zero. Node 2 leaves the layer with [0.256, 0.000]. A second layer repeats all of this, reaching two links out.' },
      { tip: 'Â(HW) and (ÂH)W are the same product, by associativity. Libraries choose the order that is cheaper: transform first when W shrinks the features, mix first when it grows them.' },
      { topic: 'gcn-layer', frame: 3, input: { graph: 'star', norm: 'none', node: 0, seed: 1 }, caption: 'Without the scaling, the same layer gives the hub [2.793, −4.442]; the biggest output in the graph has size 5.25, at a busy node. Stacked layers would make it worse.' },
      { h: 'Recap' },
      { list: ['H′ = ReLU(Â H W): transform, mix, bend.', 'W is shared by every node: a layer from f features to g has f·g weights and g biases, whatever the graph.', 'Scaling by Â keeps the outputs of busy nodes in check.'] },
      X('gn-x-layer'), X('gn-x-params'), X('gn-x-code-layer'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: learning node labels
  // =========================================================================================================
  C.lesson({
    id: 'gn-classify', chapter: 'gn-nodes', title: 'Labelling a graph from a few labelled nodes', minutes: 7,
    summary: 'In node classification the whole graph is used at once, but only a few nodes carry labels; the loss is taken on those alone. Because every layer mixes each node with its neighbours, what the model learns from the labelled nodes spreads to the rest.',
    blocks: [
      'Thirty members of three clubs, and the club is known for only nine of them. Their features are noisy, but members mostly link to members of their own club. Can we label the other twenty-one?',
      { topic: 'node-classify', frame: 0, caption: '30 nodes in 3 classes; only 9 nodes (3 per class, double rings) have labels to train on. The other 21 are the test.' },
      { p: 'Each epoch runs the whole graph through two GCN layers and a softmax, but takes the cross-entropy loss on the 9 labelled nodes only. This is called semi-supervised: the unlabelled nodes still shape every prediction through the links.' },
      { topic: 'node-classify', frame: 2, caption: 'Epoch 5: the loss on the labelled nodes is 1.142; they are 67% right, and the test nodes 62%.' },
      { predict: { q: 'By epoch 20 all 9 labelled nodes are right. Will the test nodes be right too?', options: ['Yes, all of them', 'Most, not all', 'No better than chance'], answer: 1, explain: 'At epoch 20 the test nodes are 81% right: 4 test nodes are still wrong.' } },
      { topic: 'node-classify', frame: 3, caption: 'Epoch 20: the labelled nodes are 100% right and the test nodes 81%: 4 test nodes still wrong (dark outline).' },
      { topic: 'node-classify', frame: 6, caption: 'Epoch 200: still 81% on the test nodes. The training labels were learned long ago; the loss on them alone cannot say more.' },
      { topic: 'node-classify', frame: 3, input: { data: 'communities', kind: 'gcn', perClass: 1, seed: 1 }, caption: 'With only 1 label per class (3 in all), epoch 20 reaches 81% on the 27 test nodes, but the run then wanders: 78% at epoch 200.' },
      { h: 'Recap' },
      { list: ['Node classification: one graph, a few labelled nodes, a loss on those alone.', 'The links carry what is learned from labelled nodes to unlabelled ones.', 'Fitting the labelled nodes perfectly says little: watch the held-out nodes.'] },
      X('gn-x-semi-count'), X('gn-x-test-wrong'),
    ],
  });

  C.lesson({
    id: 'gn-links-help', chapter: 'gn-nodes', title: 'When do the links help?', minutes: 6,
    summary: 'A GCN averages each node with its neighbours. That helps when neighbours tend to share a class (homophily): the noise in each node’s own features averages out. A plain MLP, which ignores the links, sees only the noise.',
    blocks: [
      'Is a GCN always better than a model that ignores the graph? Mixing in the neighbours helps only if the neighbours have something useful to say.',
      { topic: 'node-classify', frame: 7, caption: 'On the same 9 labelled nodes, a plain MLP reaches 62% on the test nodes, against 81% for the GCN.' },
      { topic: 'node-classify', frame: 6, input: { data: 'communities', kind: 'mlp', perClass: 3, seed: 1 }, caption: 'The MLP itself: 100% on its labelled nodes, 62% on the test nodes, 8 still wrong. It learns each node from its own noisy features alone.' },
      { predict: { q: 'GraphSAGE’s mean rule also uses the links. On these three clubs, will it do better than the MLP?', options: ['Yes', 'No: only the GCN’s scaling helps', 'Only with more labels'], answer: 0, explain: 'It reaches 86% at epoch 200, against 62% for the MLP: what matters is that it averages over neighbours that share a class.' } },
      { topic: 'node-classify', frame: 6, input: { data: 'communities', kind: 'sage', perClass: 3, seed: 1 }, caption: 'GraphSAGE: 86% on the test nodes at epoch 200, 3 still wrong.' },
      { p: 'The same holds on the citations set of [Node classification](#topic=node-classify) (set its graph to the papers in three fields): with 3 labels per class the GCN gets all 27 test nodes right, against 96% for the MLP. Papers mostly cite papers on their own subject.' },
      { warn: 'When neighbours tend to differ (in a network of buyers and sellers, say), plain averaging blurs exactly what tells nodes apart. Then keep the node’s own features apart, as GraphSAGE does, or reconsider the model.' },
      { h: 'Recap' },
      { list: ['Links help when neighbours tend to share a class.', 'Averaging over such neighbours cancels the noise in each node’s own features.', 'Compare against a model that ignores the links: it is the honest baseline.'] },
      X('gn-x-mlp-vs-gcn'), X('gn-x-homophily'),
    ],
  });

  C.lesson({
    id: 'gn-smooth', chapter: 'gn-nodes', title: 'Over-smoothing: too many layers', minutes: 7,
    summary: 'Each GCN layer averages a node with its neighbours. Repeated many times, the averaging spreads over the whole graph and every node ends up with nearly the same features. The classes melt together, and accuracy falls. A few layers usually do best.',
    blocks: [
      'One layer reaches one link out, two layers two. Why not use twenty layers and let every node hear the whole graph?',
      { topic: 'over-smoothing', frame: 0, caption: 'Cosine similarity between every pair of nodes’ features, grouped by class. Before any propagation the mean similarity is 0.07, with faint blocks on the diagonal.' },
      { topic: 'over-smoothing', frame: 1, caption: 'After 1 propagation by Â the mean similarity is 0.31, and a 1-layer GCN reaches 81% test accuracy: the blocks sharpen.' },
      { predict: { q: 'What happens to the mean similarity after 16 propagations?', options: ['It stays near 0.3', 'It approaches 1: every node points the same way', 'It falls back to 0'], answer: 1, explain: 'Repeated averaging over a connected graph drives every row towards the same direction: 0.99 after 16 propagations.' } },
      { topic: 'over-smoothing', frame: 4, caption: 'After 8 propagations the mean similarity is 0.88 and an 8-layer GCN reaches 62%: the blocks start to merge.' },
      { topic: 'over-smoothing', frame: 5, caption: 'After 16 propagations the similarity is 0.99 and a 16-layer GCN reaches 24%: the classes have melted together.' },
      { topic: 'over-smoothing', frame: 6, input: { data: 'citations' }, caption: 'The citations set tells the same story: best at 1 layer, 96%; 44% at 16 layers.' },
      { tip: 'Remedies keep each node’s own signal alive: few layers, skip (residual) connections that add a node’s earlier features back, and normalisations that push nodes apart again.' },
      { h: 'Recap' },
      { list: ['Each layer averages over neighbours; many layers average over everything.', 'Similarity between nodes climbs towards 1, and accuracy falls.', 'Two or three layers are the usual choice.'] },
      X('gn-x-smooth-curve'), X('gn-x-depth-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: whole graphs
  // =========================================================================================================
  C.lesson({
    id: 'gn-readout', chapter: 'gn-whole', title: 'Readout: one vector per graph', minutes: 7,
    summary: 'To classify whole graphs, the node vectors that the GCN layers produce are pooled into one vector of fixed size, column by column: by their mean, their sum or their maximum. A dense layer then classifies that vector, whatever the number of nodes.',
    blocks: [
      'Is this molecule toxic? Is this road network a ring or a hub? Now the label belongs to a whole graph, and graphs come with 8 nodes or 16. A dense layer needs an input of one fixed size. How do we get one?',
      { topic: 'graph-readout', frame: 0, caption: '60 small graphs of three kinds, rings, stars and two clusters joined by a bridge, with 8 to 16 nodes each: 40 to train on, 20 to test.' },
      { topic: 'graph-readout', frame: 1, caption: 'One test graph, a ring with 10 nodes: two GCN layers turn each node’s two numbers into 16, one row per node.' },
      { predict: { q: 'The mean readout pools 10 rows of 16 numbers. What size is the result?', options: ['10 numbers', '16 numbers', '160 numbers'], answer: 1, explain: 'It averages column by column: one row of 16, whatever the number of nodes.' } },
      { topic: 'graph-readout', frame: 2, caption: 'The mean readout gives one row of 16; a dense layer and a softmax give ring 0.52, star 0.44, two clusters 0.04. The model says ring, which is right.' },
      { topic: 'graph-readout', frame: 4, caption: 'After 20 epochs the training graphs are 93% right and the test graphs 80%. The pooled vectors of the three kinds sit in their own regions.' },
      { topic: 'graph-readout', frame: 4, input: { readout: 'sum', seed: 1 }, caption: 'With the sum readout: 98% of the training graphs and 95% of the test graphs. A sum also grows with the size of the graph, which is extra information here.' },
      { table: [['Readout', 'What it keeps', 'Good when'], ['mean', 'what a typical node looks like', 'size does not matter (the share of some kind of node)'], ['sum', 'how much of everything there is', 'counts matter (how many atoms of a kind)'], ['max', 'the strongest signal in each column', 'one special node decides (a hub, a reactive group)']], caption: 'All three are blind to the order of the nodes.' },
      { h: 'Recap' },
      { list: ['A readout pools any number of node rows into one row of fixed size.', 'Mean, sum and max each keep something different.', 'A dense layer and a softmax classify the pooled row.'] },
      X('gn-x-readout-compute'), X('gn-x-readout-choice'), X('gn-x-readout-max'), X('gn-x-code-readout'),
    ],
  });

  C.lesson({
    id: 'gn-invariance', chapter: 'gn-whole', title: 'Numbering the nodes must not matter', minutes: 6,
    summary: 'The same graph can be written with its nodes in any order. A GCN layer just reorders its output rows to match, and a mean, sum or max over the rows ignores the order, so the prediction for the graph does not depend on how the nodes were numbered.',
    blocks: [
      'Two chemists draw the same molecule and number its atoms differently. Their adjacency matrices differ, row for row. A model that gave the two drawings different answers would be useless. What makes a graph model immune to the numbering?',
      { topic: 'graph-readout', frame: 3, caption: 'Renumber the nodes (last becomes first) and run the network again: the rows come out in a new order, but the mean does not care. The pooled rows are identical.' },
      { math: 'renumbering by a permutation P:   A → P A Pᵀ,   H → P H\na GCN layer:   ReLU(P Â Pᵀ P H W) = P · ReLU(Â H W)     (Pᵀ P = I)\na readout:     mean(P H′) = mean(H′)' },
      { predict: { q: 'Which pooling would make the answer depend on the numbering?', options: ['The sum of the rows', 'The first row alone', 'The largest value in each column'], answer: 1, explain: 'The first row is whichever node happens to be numbered 0: renumber, and it changes.' } },
      { topic: 'graph-readout', frame: 3, input: { readout: 'max', seed: 2 }, caption: 'The max readout is just as blind to order: the pooled rows are identical.' },
      { topic: 'graph-readout', frame: 4, input: { readout: 'max', seed: 2 }, caption: 'Max with seed 2: 88% of the training graphs and 95% of the test graphs. A maximum keeps the strongest signal in each column, such as the busiest hub of a star.' },
      { h: 'Recap' },
      { list: ['Renumbering the nodes permutes the rows of H and of A.', 'GCN layers carry the permutation through; mean, sum and max remove it.', 'Flattening the rows, or taking one of them, would not.'] },
      X('gn-x-perm-choice'), X('gn-x-perm-rows'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
