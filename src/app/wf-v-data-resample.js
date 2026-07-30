// The viewer of the Resample block: the rows of each class before and after, and what happened to the test rows.
(function (S) {
  'use strict';
  const V = S.WFV;
  if (!V) return;
  V.register('resample', V.viewer((view, ctx) => {
    const el = ctx.el, b = view.before || [], a = view.after || [], sum = (x) => x.reduce((p, q) => p + q, 0);
    el.appendChild(V.facts([['rows in', sum(b)], ['rows out', sum(a)], ['how', view.how === 'over' ? 'oversampling' : 'undersampling'], view.test ? ['test rows, unchanged', view.test] : null]));
    el.appendChild(V.table(['class', 'rows before', 'rows after'], (view.classes || []).map((c, k) => [c, b[k], a[k]]), { cell: (i, j) => (j === 2 && a[i] !== b[i] ? 'hi' : '') }));
    el.appendChild(V.note(view.how === 'over'
      ? 'Rows of the smaller classes were copied at random until each class had its share: the model now pays the rare class more attention, but it learns from copies, so it can fit their noise.'
      : 'Rows of the larger classes were left out at random: the classes are even now, but the model learns from fewer rows.'));
    el.appendChild(V.note(view.test ? 'The test rows went through untouched: a score on them still says how the model does on rows as they really come.' : 'Wire the test rows through the second input to keep them with the training rows; they are never resampled.', view.test ? 'good' : ''));
  }));
})(globalThis.Nooral = globalThis.Nooral || {});
