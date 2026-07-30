// What the diagram checker looks at for the evaluation, comparison, inference and explanation viewers: each sample
// opened on the canvas with one block's viewer showing (and a clicked confusion cell, a moved threshold, a drawn digit).
(function (S) {
  'use strict';
  if (!S.Checks || S.Checks.some((c) => c.group === 'canvas-eval')) return;
  const ready = "!!document.querySelector('.wf-viewer .wf-ev') && Nooral.app.page && Nooral.app.page.whenIdle";
  const open = (sample, block, then) => "await Nooral.app.page.loadSample(" + JSON.stringify(sample) + "); await Nooral.app.page.whenIdle(); Nooral.app.page.open(" + JSON.stringify(block) + "); await new Promise((r) => setTimeout(r, 250));" + (then || '');
  const click = (key) => " const e = document.querySelector('[data-key=\"" + key + "\"]'); if (e) e.dispatchEvent(new MouseEvent('click', { bubbles: true })); await new Promise((r) => setTimeout(r, 200));";
  const add = (name, sample, block, then) => S.Checks.push({ group: 'canvas-eval', name, hash: 'canvas', ready, setup: open(sample, block, then), sel: '#main' });
  add('evaluate: confusion matrix', 'evaluate-loans', 'evaluate1');
  add('evaluate: a cell clicked open', 'evaluate-loans', 'evaluate1', click('cm:c0_0'));
  add('evaluate: shares of each true class', 'evaluate-loans', 'evaluate1', " const b = [...document.querySelectorAll('.wf-seg button')].find((x) => /true class/.test(x.textContent)); if (b) b.click(); await new Promise((r) => setTimeout(r, 150));");
  add('roc and pr', 'evaluate-loans', 'roc1');
  add('roc: threshold moved', 'evaluate-loans', 'roc1', " const s = document.getElementById('wfThreshold'); s.value = '0.25'; s.dispatchEvent(new Event('input', { bubbles: true })); await new Promise((r) => setTimeout(r, 150));");
  add('calibration', 'evaluate-loans', 'calibration1');
  add('compare three models', 'compare-three', 'compare1');
  add('predict: a table', 'predict-exam', 'predict1', ' await new Promise((r) => setTimeout(r, 300));');
  add('predict: the plane', 'predict-moons', 'predict1', ' await new Promise((r) => setTimeout(r, 300));');
  add('evaluate: numbers', 'regression-residuals', 'evaluate1');
  add('residuals', 'regression-residuals', 'residuals1');
  add('learning curve', 'regression-residuals', 'learningcurve1');
  add('learning and validation curves', 'learning-curve-tree', 'learningcurve1');
  add('permutation importance', 'explain-harvest', 'importance1');
  add('partial dependence', 'explain-harvest', 'pdp1');
  add('shap', 'explain-harvest', 'shap1');
  add('lime', 'explain-harvest', 'lime1');
  add('evaluate: images in a cell', 'predict-digits', 'evaluate1');
  add('predict: a drawn digit', 'predict-digits', 'predict1', " const t = Nooral.app.page.remote; void t; const px = new Array(784).fill(0); for (let j = 6; j < 22; j++) { px[j * 28 + 14] = 1; px[j * 28 + 15] = 0.8; } if (Nooral.WF.VK.padOf) await Nooral.WF.VK.padOf(px); await new Promise((r) => setTimeout(r, 300));");
  add('saliency and grad-cam', 'predict-digits', 'saliency1');
})(globalThis.Nooral = globalThis.Nooral || {});
