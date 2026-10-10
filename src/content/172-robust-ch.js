// Part 17, robustness, fairness and privacy: a model challenge with a fairness constraint, the guided tour, the
// course, and the pages the diagram checker looks at. The challenge is scored by F1 on hidden applicants and must
// also keep the demographic parity difference between the two groups small; its target and marks were set from the
// engines' own scores of the candidate workflows.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || C.courses.has('rb-course')) return;

  // ---- fair alarms on small loans: F1 on defaults, with the parity gap at most 0.06 ---------------------------------
  if (WF && !C.challenges.has('rb-ch-fair-loans')) {
    const pipe = (model, mp, cols) => {
      const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
      const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
      if (cols) { blocks.sel = ['select', { mode: 'columns', columns: cols }]; wires.push(['split.train', 'sel.in'], ['split.test', 'sel.test'], ['sel.out', 'tr.data'], ['sel.test', 'ev.data']); }
      else wires.push(['split.train', 'tr.data'], ['split.test', 'ev.data']);
      return WF.build({ blocks, wires });
    };
    const data = { set: 'loans', n: 600, seed: 17 }, noGroup = ['income', 'debt_ratio', 'years_employed', 'missed_payments', 'amount'];
    C.challenge({
      id: 'rb-ch-fair-loans', n: 171, title: 'Fair alarms on small loans', level: 3, data, hidden: { seed: 171, n: 2000 },
      metric: 'f1', metricName: 'F1 on defaults', target: 0.55, marks: { silver: 0.565, gold: 0.575 },
      constraints: [{ kind: 'parity', group: 'group', max: 0.06 }],
      story: 'A lender of small loans wants a warning on applications likely to default, so a person looks at them twice. It will judge the warning by F1 on the defaults among 2,000 applicants it has kept back.\n\nThere is a second condition. Applicants belong to group A or group B, and the warning must not fall much more often on one group: the share of applicants flagged in each group may differ by at most 0.06 (the demographic parity difference). A warning that breaks this is not used, however good its F1. Group B has lower incomes on average, so a model can treat the groups differently without ever seeing the group.',
      task: 'classification',
      limits: { blocks: ['split', 'select', 'scale', 'resample', 'logistic', 'naivebayes', 'tree', 'train', 'evaluate'], models: ['logistic', 'naivebayes', 'tree'], params: { tree: { maxDepth: [1, 6] } }, maxBlocks: 8 },
      hints: [
        'Two numbers count: F1 on the defaults, and the gap between the groups’ flag rates. Defaults are about a quarter of the rows, so the default threshold of 0.5 flags too few: lower it on the model block.',
        'Lowering the threshold catches more defaults but widens the gap when the model leans on the group column. Logistic regression on every column at threshold 0.35 reaches F1 0.547 with a gap of 0.10. Take the group column out with Select features, and try a model that gives honest probabilities for a low threshold.',
        'Split → Select features (keep income, debt_ratio, years_employed, missed_payments and amount) → Naive Bayes with threshold 0.3 → Train → Evaluate: F1 0.577 with a parity gap of 0.055, a gold mark. With the group column left in, the same model scores 0.581 but its gap is 0.126, and the warning is refused.',
      ],
      solution: pipe('naivebayes', { threshold: 0.3 }, noGroup),
      wrong: [pipe('naivebayes', { threshold: 0.3 }), pipe('logistic', { threshold: 0.35 }), pipe('naivebayes', { threshold: 0.35 }, noGroup), pipe('logistic', {})],
    });
  }

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'rb-tour', title: 'Tour: robustness, fairness and privacy', blurb: 'Brittle answers and their defences, drift, fairness gaps and private training, in eight steps.',
    steps: [
      { route: 'part=responsible', target: '#main', text: 'Part 17 of the catalogue: robustness, fairness and privacy. Each topic shows a weakness on a toy model of our own, then the defence.' },
      { route: 'topic=robust-defend', target: '#stageWrap', text: 'Stress-testing a model, then hardening it: adversarial training and smoothing, with what each costs in clean accuracy.' },
      { target: '#btnPlay', text: 'Play steps through the defences; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=shift-monitor', target: '#inputs', text: 'Choose what drifts here, the inputs, the class shares or the rule, and watch which ones the input monitor can see.' },
      { route: 'topic=fair-metrics', target: '#stageWrap', text: 'A confusion matrix per group: approval rates for demographic parity, error rates for equalised odds.' },
      { route: 'topic=dp-sgd', target: '#stageWrap', text: 'Private training: each person’s gradient clipped, noise added, and the privacy–accuracy trade-off.' },
      { route: 'lesson=rb-mitigate', target: '#lessonView', text: 'Lessons run the topics at the frames that matter: here, two mitigations and the gaps each one opens or closes.' },
      { route: 'learn', target: '[data-chapter="rb-brittle"]', text: 'The lessons and exercises of this part start here: margins, drift scores, fairness gaps from confusion matrices, clipping by hand, and choosing the right defence.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'rb-course', title: 'Robustness, fairness and privacy', blurb: 'All nine topics of part 17, each weakness with its defence, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'rb-brittle', note: 'Accurate is not the same as robust.' },
      { topic: 'robust-brittle', note: 'A margin you can measure.' },
      { exercise: 'rb-x-margin', note: 'Distance to the boundary.' },
      { topic: 'robust-defend', note: 'Adversarial training and smoothing.' },
      { exercise: 'rb-x-defence-table', note: 'Read the stress test.' },
      { lesson: 'rb-shift-kinds', note: 'The world moves.' },
      { topic: 'shift-kinds', note: 'Covariate, label and concept shift.' },
      { exercise: 'rb-x-shift-spot', note: 'Spot it in a summary.' },
      { topic: 'shift-monitor', note: 'A drift score per batch.' },
      { exercise: 'rb-x-psi', note: 'PSI by hand.' },
      { lesson: 'rb-fair-metrics', note: 'Counting per group.' },
      { topic: 'fair-metrics', note: 'Parity and equalised odds.' },
      { exercise: 'rb-x-eod', note: 'From four counts per group.' },
      { topic: 'fair-mitigate', note: 'Reweighing and per-group cut-offs.' },
      { exercise: 'rb-x-mitigate-pre', note: 'Which mitigation fits.' },
      { topic: 'membership-inference', note: 'What a model gives away.' },
      { topic: 'dp-sgd', note: 'Clip, then add noise.' },
      { exercise: 'rb-x-dp-tradeoff', note: 'A budget and a curve.' },
      { topic: 'data-poisoning', note: 'Cleaning out bad labels.' },
      { exercise: 'rb-x-defence-match', note: 'The right defence for each risk.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson rb-brittle')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'responsible').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['rb-x-robust-curve', 'rb-x-defence-table', 'rb-x-shift-spot', 'rb-x-monitor-read', 'rb-x-tpr-fpr', 'rb-x-code-dpd', 'rb-x-reweigh', 'rb-x-clip', 'rb-x-dp-tradeoff', 'rb-x-knn-clean', 'rb-x-defence-match'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'challenges', name: 'challenge rb-ch-fair-loans', hash: 'challenge/rb-ch-fair-loans', ready: "document.querySelector('#challengeView')", sel: '#main' });
    S.Checks.push({ group: 'challenges', name: 'a constraint broken', hash: 'challenge/rb-ch-fair-loans', fresh: true, setup: "const x = Nooral.Content.challenges.get('rb-ch-fair-loans'); Nooral.app.page.load(x.wrong[0]); Nooral.app.page.submit();", ready: "document.querySelector('#chResult.ok, #chResult.bad') && document.querySelector('.ch-constraint')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course rb-course', hash: 'course=rb-course', ready: "document.querySelector('#courseView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
