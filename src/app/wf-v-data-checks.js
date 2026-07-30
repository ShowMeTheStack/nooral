// What the diagram checker looks at for the data, prepare, split and unsupervised viewers: each viewer open on the
// canvas, in a sample workflow that exercises it.
(function (S) {
  'use strict';
  if (!S.Checks || S.Checks.some((c) => c.group === 'canvas-data')) return;
  const ready = "!!document.querySelector('.wf-viewer .wfd svg, .wf-viewer .wfd table')";
  const open = (sample, id) => "await Nooral.app.page.loadSample(" + JSON.stringify(sample) + "); await Nooral.app.page.whenIdle(); Nooral.app.page.open(" + JSON.stringify(id) + "); await new Promise((r) => setTimeout(r, 300));";
  for (const [sample, id, name] of [
    ['prepare-pumps', 'data1', 'a built-in table'], ['prepare-pumps', 'split1', 'the split'], ['prepare-pumps', 'impute1', 'impute'], ['prepare-pumps', 'scale1', 'scale, fitted on the training rows'],
    ['leakage-loans', 'encode1', 'encode'], ['leakage-loans', 'scale1', 'scale before the split'], ['leakage-loans', 'select1', 'select'],
    ['clean-pumps', 'clean1', 'clean: drop rows'], ['clean-pumps', 'clean2', 'clean: clip'], ['features-moons', 'data1', 'a 2-D set'], ['features-moons', 'poly1', 'polynomial features'],
    ['cv-wildflowers', 'cv1', 'cross-validation'], ['text-spam-bow', 'data1', 'a text set'], ['text-spam-bow', 'vectorise1', 'vectorise'],
    ['clusters-kmeans', 'kmeans1', 'k-means'], ['clusters-kmeans', 'gmm1', 'Gaussian mixture'], ['clusters-moons', 'dbscan1', 'DBSCAN'], ['clusters-moons', 'hierarchical1', 'hierarchical'],
    ['maps-wine', 'pca1', 'PCA'], ['maps-wine', 'tsne1', 't-SNE'], ['maps-wine', 'umap1', 'UMAP'],
  ]) S.Checks.push({ group: 'canvas-data', name: name + ' (' + sample + ')', hash: 'canvas', ready, setup: open(sample, id), sel: '#main' });
})(globalThis.Nooral = globalThis.Nooral || {});
