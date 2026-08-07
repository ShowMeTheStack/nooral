// The workflow kind: build a pipeline that reaches a target score on a hidden test set within limits. The answer is a
// workflow document; it is run through the engines (at once in node, in the workflow's worker in the page) and the
// trained model is scored on rows it never saw (src/app/challenges-core.js does the scoring for both).
(function (S) {
  'use strict';
  const K = (S.Check = S.Check || {});
  K.kinds = K.kinds || {};
  const say = (x, r) => {
    const C = S.Challenges;
    if (r.limits && r.limits.length) return { ok: false, message: 'The workflow breaks the limits: ' + r.limits[0] + '.', details: r.limits.slice(1) };
    if (r.score === null || r.score === undefined) return { ok: false, message: 'The workflow could not be scored: ' + (r.problems[0] || 'it has no trained model') + '.', details: r.problems.slice(1) };
    const line = 'It scores ' + C.metricLabel(x) + ' ' + C.fmt(r.score) + ' on ' + r.n + ' hidden rows';
    if (r.ok) return { ok: true, message: 'Right. ' + line + ', and the target is ' + C.goal(x) + '.' + (x.why ? ' ' + x.why : ''), score: r.score };
    return { ok: false, message: line + ', short of the target ' + C.goal(x) + '.' + (x.whyNot ? ' ' + x.whyNot : ''), score: r.score };
  };
  const blank = (a) => !a || !Array.isArray(a.blocks) || !a.blocks.length;
  K.kinds.workflow = {
    check(x, a) {
      if (blank(a)) return { ok: false, message: 'Build a workflow first.' };
      return say(x, S.Challenges.evaluate(a, x));
    },
    async checkAsync(x, a) {
      if (blank(a)) return { ok: false, message: 'Build a workflow first.' };
      return say(x, await S.Challenges.evaluateAsync(a, x));
    },
    lint(x) {
      const p = [];
      if (!x.data || !x.data.set) p.push('data: {set, opts, seed, n} is needed');
      if (!x.metric || !(S.WF && S.WF.METRICS[x.metric])) p.push('metric must be one of ' + (S.WF ? Object.keys(S.WF.METRICS).join(', ') : ''));
      if (!Number.isFinite(x.target)) p.push('target must be a number');
      if (!x.hidden || x.hidden.seed === undefined) p.push('hidden: {seed, n} is needed');
      else if (x.data && x.hidden.seed === (x.data.seed || 1)) p.push('the hidden seed must differ from the public one');
      return p;
    },
  };
})(globalThis.Nooral = globalThis.Nooral || {});
