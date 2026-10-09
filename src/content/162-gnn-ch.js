// Part 16, graph neural networks: the guided tour, the course, and the pages the diagram checker looks at. The
// workflow canvas has no graph blocks, so this part sets no model challenge; its exercises do the work by hand and
// in code.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.courses.has('gn-course')) return;

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'gn-tour', title: 'Tour: graph neural networks', blurb: 'Adjacency, message passing, a GCN layer, node labels, over-smoothing and readout, in eight steps.',
    steps: [
      { route: 'part=graphs', target: '#main', text: 'Part 16 of the catalogue: graph neural networks. Every topic runs the engine’s own GCN layers on small graphs of our own, with every weight shown.' },
      { route: 'topic=graph-adjacency', target: '#stageWrap', text: 'A graph written as its adjacency matrix, row by row, then squared to count two-step walks.' },
      { target: '#btnPlay', text: 'Play fills the matrix one node at a time; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=message-passing', target: '#inputs', text: 'Choose a node and a rule here, GCN, mean or sum, and watch every message it receives, with its weight.' },
      { route: 'topic=gcn-layer', target: '#stageWrap', text: 'The same layer as matrices: Z = HW, then ÂZ, then the ReLU.' },
      { route: 'topic=over-smoothing', target: '#stageWrap', text: 'Why deeper is not better: after many layers every node looks alike, and accuracy falls.' },
      { route: 'lesson=gn-readout', target: '#lessonView', text: 'Lessons run the topics at the frames that matter and ask for a prediction on the way: here, pooling a whole graph into one vector.' },
      { route: 'learn', target: '[data-chapter="gn-graphs"]', text: 'The lessons and exercises of this part start here: build Â by hand, pass messages, run a GCN layer, write it in code, and choose a readout.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'gn-course', title: 'Graph neural networks', blurb: 'All six topics of part 16, from an adjacency matrix to a graph-level readout, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'gn-graphs', note: 'A graph as a table of links.' },
      { topic: 'graph-adjacency', note: 'Rows, degrees, A² and self loops.' },
      { exercise: 'gn-x-adj-fill', note: 'Write one yourself.' },
      { lesson: 'gn-norm', note: 'Self loops and the scaling of Â.' },
      { exercise: 'gn-x-ahat', note: 'Build Â by hand.' },
      { topic: 'message-passing', note: 'A node listens to its neighbours.' },
      { exercise: 'gn-x-gcn-agg', note: 'One aggregation by hand.' },
      { exercise: 'gn-x-mp-round', note: 'A whole round.' },
      { topic: 'gcn-layer', note: 'Transform, mix, bend.' },
      { exercise: 'gn-x-code-gcn', note: 'The aggregation in code.' },
      { lesson: 'gn-classify', note: 'A few labels, a whole graph.' },
      { topic: 'node-classify', note: 'GCN, GraphSAGE and an MLP side by side.' },
      { exercise: 'gn-x-mlp-vs-gcn', note: 'When the links help.' },
      { topic: 'over-smoothing', note: 'Too many layers.' },
      { exercise: 'gn-x-depth-choice', note: 'Choose the depth.' },
      { lesson: 'gn-readout', note: 'One vector per graph.' },
      { topic: 'graph-readout', note: 'Mean, sum and max, trained.' },
      { exercise: 'gn-x-readout-choice', note: 'Choose the readout.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson gn-graphs')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'graphs').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['gn-x-adj-fill', 'gn-x-ahat', 'gn-x-gcn-agg', 'gn-x-mp-round', 'gn-x-layer', 'gn-x-code-gcn', 'gn-x-smooth-curve', 'gn-x-depth-choice', 'gn-x-readout-compute', 'gn-x-perm-rows'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course gn-course', hash: 'course=gn-course', ready: "document.querySelector('#courseView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
