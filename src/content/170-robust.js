// Part 17, robustness, fairness and privacy: the chapters and the lessons. The exercises are in 171-robust-ex.js; the
// tour, the course and the pages the diagram checker looks at are in 172-robust-ch.js. The part is defensive: each
// lesson names a weakness as a concept, shows it on a toy model of our own, and spends its weight on the defence.
// Every value quoted here was printed by the topics themselves (S.Rob and S.Metrics on the topics' own data).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('rb-brittle')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'rb-brittle', part: 'responsible', n: 1, title: 'Brittle models, and hardening them', blurb: 'Why a small change to an input can flip a confident answer, how far a point sits from the boundary, and two defences: training on nudged examples and voting over noisy copies.' });
  C.chapter({ id: 'rb-shift', part: 'responsible', n: 2, title: 'When the world moves', blurb: 'Covariate, label and concept shift, what each does to a model, and how to watch a deployed model for drift before it fails quietly.' });
  C.chapter({ id: 'rb-fair', part: 'responsible', n: 3, title: 'Fairness between groups', blurb: 'Approval rates and error rates per group, demographic parity and equalised odds, why dropping the group column is not enough, and two mitigations with their costs.' });
  C.chapter({ id: 'rb-private', part: 'responsible', n: 4, title: 'Privacy and the training data', blurb: 'What a model can give away about the people it was trained on, differentially private training and its price in accuracy, and cleaning out poisoned labels.' });

  // =========================================================================================================
  // Chapter 1: brittle models, and hardening them
  // =========================================================================================================
  C.lesson({
    id: 'rb-brittle', chapter: 'rb-brittle', title: 'Why a small nudge can flip the answer', minutes: 7,
    summary: 'A model that is right on every test point can still be fragile: a point near its boundary needs only a small change to land on the other side. The model’s gradient with respect to its input shows which directions matter most, and the smallest change that flips an answer measures how robust that answer is.',
    blocks: [
      'A model is 100% accurate on its test set. Is it safe to rely on? Accuracy says how often it is right on typical inputs; it says nothing about how close each answer was to going the other way.',
      { topic: 'robust-brittle', frame: 0, caption: 'A small network gets 199 of 200 test points of two moons right, 99.5%. The test point closest to the boundary, at (1.10, −0.01), is called class 0 with probability 0.816, which is right.' },
      { topic: 'robust-brittle', frame: 1, caption: 'The gradient of the loss with respect to the input, not the weights, is (2.020, −2.236): the direction in which the input would have to change for the model to be most wrong.' },
      { p: 'The same gradient that training uses to fix the weights can be taken with respect to the input. It tells us how sensitive the answer is to each feature, and in which direction the answer is weakest. Defenders use it to test their own models before anyone else does.' },
      { topic: 'robust-brittle', frame: 3, caption: 'Moved by 0.15 in each coordinate, small beside the spread of the data, the point gets class 1 with probability 0.889: a right answer turned wrong.' },
      { predict: { q: 'For a typical point deep inside its class, will the smallest change that flips the answer be larger or smaller than for the point near the boundary?', options: ['Larger', 'Smaller', 'The same'], answer: 0, explain: 'The point near the boundary flips at ε = 0.063; a typical point needs ε = 0.320.' } },
      { topic: 'robust-brittle', frame: 4, caption: 'Along the worst direction, the smallest step that flips the point near the boundary is ε = 0.063, while the data spans 3.39 along x₁.' },
      { topic: 'robust-brittle', frame: 4, input: { which: 'typical', eps: 0.4, seed: 1 }, caption: 'A typical point needs ε = 0.320. Robustness is a property of each answer: its distance from the boundary.' },
      { warn: 'This weakness matters wherever inputs can be chosen by someone with a motive: spam, fraud checks, content filters. The answer is not secrecy about the model but models with room around their answers, and checks on the inputs.' },
      { h: 'Recap' },
      { list: ['High accuracy does not mean the answers are stable.', 'The input gradient shows which small changes matter most.', 'The smallest change that flips an answer, its margin, measures how robust that answer is.'] },
      X('rb-x-margin'), X('rb-x-why-brittle'),
    ],
  });

  C.lesson({
    id: 'rb-defend', chapter: 'rb-brittle', title: 'Hardening a model: adversarial training and smoothing', minutes: 8,
    summary: 'Adversarial training adds worst-case nudged copies of the training points to every batch, so the boundary learns to keep room around the data. Randomised smoothing asks the model about many noisy copies of an input and takes the vote. Both trade a little clean accuracy for steadier answers; neither is a guarantee.',
    blocks: [
      'If small nudges are the weakness, can we teach the model to ignore them, or make each answer depend less on any exact input?',
      { topic: 'robust-defend', frame: 0, caption: 'A network trained the plain way: 99.5% of the 200 test points right, with a boundary (black) close to many of them.' },
      { topic: 'robust-defend', frame: 1, caption: 'Stress test: every test point is moved by ε = 0.20 in the direction that hurts this model most. Accuracy falls to 84.5%. Testing your own model this way is the first defence.' },
      { topic: 'robust-defend', frame: 2, caption: 'Adversarial training replaces half of every batch by its own nudged copies at ε = 0.20. The boundary (green) keeps more room around the points; clean accuracy is 97.5%, against 99.5% before.' },
      { predict: { q: 'Stress-tested at the same ε, will the adversarially trained model keep more of its accuracy than the plain one?', options: ['Yes', 'No, it was only trained differently', 'Only on clean points'], answer: 0, explain: 'It keeps 87.5% under the stress test, against 84.5% for the plain network.' } },
      { topic: 'robust-defend', frame: 3, caption: 'Under the stress test the adversarially trained model keeps 87.5%, against 84.5%. Its curve stays higher as ε grows.' },
      { topic: 'robust-defend', frame: 4, caption: 'Smoothing: ask the plain model about 32 noisy copies of a nudged point, with noise σ = 0.30. 24 vote for class 1 and 8 against, so the smoothed answer is class 1, which is right again.' },
      { topic: 'robust-defend', frame: 5, caption: 'All three: no defence 99.5% clean and 84.5% nudged; smoothing 98.5% and 85%; adversarial training 97.5% and 87.5%.' },
      { topic: 'robust-defend', frame: 5, input: { eps: 0.2, sigma: 0.6, seed: 1 }, caption: 'Too much noise costs more than it buys: with σ = 0.60, smoothing gives 91% clean and 80% nudged.' },
      { tip: 'Cheaper checks help too: reject inputs outside the range seen in training, round inputs to a coarser precision so tiny changes vanish, and watch for unusual inputs, as chapter 2 does for drift.' },
      { h: 'Recap' },
      { list: ['Stress-test your own model with worst-case nudges before deploying it.', 'Adversarial training: train on nudged copies, so the boundary keeps room around the data.', 'Smoothing: vote over noisy copies. Each defence costs some clean accuracy; none is a guarantee.'] },
      X('rb-x-smooth-vote'), X('rb-x-defence-table'), X('rb-x-robust-curve'), X('rb-x-defend-nudge'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: when the world moves
  // =========================================================================================================
  C.lesson({
    id: 'rb-shift-kinds', chapter: 'rb-shift', title: 'Three kinds of distribution shift', minutes: 8,
    summary: 'A model learns the world its training data came from. Covariate shift moves the inputs into regions it rarely saw; label shift changes how common each class is; concept shift changes the rule linking inputs to labels. Each hurts in its own way and has its own remedy.',
    blocks: [
      'A model trained last year is used this year. It has not changed; the world may have. In what ways, and which of them should worry us most?',
      { topic: 'shift-kinds', frame: 0, caption: 'Training inputs sit around x₁ = −1.28, where the true rule (dashed wave) is almost flat. A logistic regression gets 98% of its 400 training points right.' },
      { topic: 'shift-kinds', frame: 1, caption: 'Covariate shift: the test inputs have moved, x₁ now averaging 0.24, onto the crest of the wave the model never saw. The rule has not changed.' },
      { topic: 'shift-kinds', frame: 2, caption: 'The unchanged model gets 78% of the 400 shifted test points right, against 96% with no shift.' },
      { topic: 'shift-kinds', frame: 2, input: { kind: 'label', amount: 1, seed: 1 }, caption: 'Label shift: class one is now 95% of the test set, against 50% in training, though each class looks the same. The model gets 92%, against 95%: its cut-off was tuned to the old shares.' },
      { predict: { q: 'Concept shift: the inputs look exactly as before, but the rule has moved. Can a monitor that watches only the inputs notice?', options: ['Yes, the inputs will look different', 'No: only the true outcomes can show it', 'Only if the shift is large'], answer: 1, explain: 'The inputs keep their distribution (x₁ averages −1.32 against −1.28); only the labels have changed.' } },
      { topic: 'shift-kinds', frame: 2, input: { kind: 'concept', amount: 0.5, seed: 1 }, caption: 'Concept shift: points that look exactly like training points now carry other labels. The model gets 71%, against 96%.' },
      { topic: 'shift-kinds', frame: 3, caption: 'Across all amounts, the worst drop is 0.28 for covariate shift, 0.04 for label shift and 0.51 for concept shift.' },
      { table: [['Shift', 'What changes', 'Visible in the inputs?', 'Remedy'], ['covariate', 'where the inputs fall', 'yes', 'labelled data where the inputs now fall, or weight training rows like the new ones'], ['label', 'how common each class is', 'partly (the mix of predictions moves)', 'estimate the new shares and move the cut-off'], ['concept', 'the rule itself', 'no', 'fresh labels and retraining']], caption: 'Each kind has its own defence.' },
      { h: 'Recap' },
      { list: ['Covariate shift: new inputs, same rule.', 'Label shift: same classes, new shares.', 'Concept shift: same inputs, new rule; only outcomes reveal it.'] },
      X('rb-x-shift-spot'), X('rb-x-shift-kind'), X('rb-x-shift-remedy'),
    ],
  });

  C.lesson({
    id: 'rb-monitor', chapter: 'rb-shift', title: 'Watching a deployed model for drift', minutes: 8,
    summary: 'Compare each new batch of inputs with the training inputs, bin by bin: the population stability index (PSI) sums (b − r)·ln(b / r) over the bins. Over about 0.2 is a warning. Input drift is a signal to check, not proof of harm; and a change of rule leaves the inputs alone, so outcomes must be watched too.',
    blocks: [
      'The true outcomes of a loan model arrive months later. Is there an earlier sign that something has changed?',
      { topic: 'shift-monitor', frame: 0, caption: 'The model is trained on 400 rows and gets 98% of them right. Each input’s training values are cut into ten bins of equal share.' },
      { math: 'r_k = training share in bin k (one tenth each)\nb_k = this batch’s share in bin k\nPSI = Σ_k (b_k − r_k) · ln(b_k / r_k)        under 0.1 calm, over 0.2 drifted' },
      { topic: 'shift-monitor', frame: 1, caption: 'Batch 1: the drift score is 0.331 on x₁ and 0.071 on x₂; 0.331 is over the threshold 0.20, so an alarm is raised. When its outcomes arrive, the model is 93% right.' },
      { predict: { q: 'The alarm sounded at batch 1, with accuracy still 93%. Was it a false alarm?', options: ['Yes, accuracy was fine', 'No: it was an early warning, and accuracy fell later', 'There is no way to tell'], answer: 1, explain: 'The inputs kept drifting; by batch 7 accuracy was 59%, against 98% in training.' } },
      { topic: 'shift-monitor', frame: 9, caption: 'The input monitor raised 8 alarms, the first at batch 1. Accuracy was lowest at batch 7, 59%.' },
      { topic: 'shift-monitor', frame: 9, input: { scen: 'concept', speed: 0.12, thr: 0.2, seed: 1 }, caption: 'A silent failure: the rule drifts while the inputs stay put. The monitor never raised an alarm, yet accuracy fell to 41% at batch 7. Watch accuracy on delayed labels, not just the inputs.' },
      { topic: 'shift-monitor', frame: 9, input: { scen: 'none', speed: 0.12, thr: 0.2, seed: 3 }, caption: 'When nothing drifts, the scores stay small, at most 0.170: sampling noise. A threshold set too low would cry wolf.' },
      { h: 'Recap' },
      { list: ['PSI compares a batch’s bin shares with the training shares.', 'An alarm is a reason to check the data and the outcomes, not proof of harm.', 'Concept shift does not show in the inputs: watch outcomes as they arrive.'] },
      X('rb-x-psi'), X('rb-x-monitor-read'), X('rb-x-defend-drift'), X('rb-x-code-psi'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: fairness between groups
  // =========================================================================================================
  C.lesson({
    id: 'rb-fair-metrics', chapter: 'rb-fair', title: 'Measuring fairness between groups', minutes: 8,
    summary: 'Split the test set by group and count per group: the approval (selection) rate, the true-positive rate among those who repaid and the false-positive rate among those who did not. Demographic parity compares approval rates; equalised odds compares both error rates.',
    blocks: [
      'A lender’s model is 79% accurate. Does it treat applicants of group A and group B alike? One accuracy number cannot say: we must count each group separately.',
      { topic: 'fair-metrics', frame: 0, caption: 'The test set holds 108 applicants of group A and 72 of group B. In truth 80% of group A repaid and 69% of group B.' },
      { topic: 'fair-metrics', frame: 1, caption: 'Group A: of 86 who repaid, 82 were approved (TPR 0.95); of 22 who defaulted, 13 were approved (FPR 0.59). 95 of 108 approved: a rate of 0.88.' },
      { topic: 'fair-metrics', frame: 2, caption: 'Group B: TPR 0.86, FPR 0.64, and 57 of 72 approved: a rate of 0.79.' },
      { def: 'Demographic parity difference', text: 'The gap between the groups’ approval rates. Zero when every group is approved equally often.' },
      { predict: { q: 'Approval rates are 0.88 and 0.79. What is the demographic parity difference, to two places?', options: ['0.09', '0.79', '1.67'], answer: 0, explain: 'The topic computes it from the exact rates: 0.088.' } },
      { topic: 'fair-metrics', frame: 3, caption: 'Demographic parity difference: 0.088. It ignores who would really repay.' },
      { def: 'Equalised odds difference', text: 'The larger of the TPR gap and the FPR gap: repayers should be approved equally often in each group, and so should defaulters.' },
      { topic: 'fair-metrics', frame: 4, caption: 'The TPR gap is 0.093 and the FPR gap 0.045, so the equalised-odds difference is 0.093.' },
      { h: 'Recap' },
      { list: ['Count a confusion matrix per group.', 'Demographic parity: equal approval rates.', 'Equalised odds: equal TPR and equal FPR; its difference is the larger gap.'] },
      X('rb-x-sel-rates'), X('rb-x-dpd'), X('rb-x-tpr-fpr'), X('rb-x-eod'), X('rb-x-code-dpd'),
    ],
  });

  C.lesson({
    id: 'rb-fair-limits', chapter: 'rb-fair', title: 'Blind is not fair, and not every measure can hold', minutes: 6,
    summary: 'Removing the group column does not remove the group’s signal when other features, such as income, differ between groups. And when the true rates differ, demographic parity and equalised odds cannot both be met by any useful model: choosing which matters is a decision, not a calculation.',
    blocks: [
      'The simplest proposal: never show the model the group. Then it cannot discriminate. Can it?',
      { topic: 'fair-metrics', frame: 5, caption: 'With the group column removed and the model retrained, the parity difference is 0.088 and the equalised-odds difference 0.093: the same as with it. Income differs between the groups, so it carries the group’s signal anyway.' },
      { p: 'A feature that differs between groups acts as a proxy for the group. Dropping the group column also throws away the means to measure the gap; keep it aside for checking, as the project checklist in part 18 does.' },
      { predict: { q: 'In truth 80% of group A and 69% of group B repaid. A perfect model approves exactly the repayers. What is its demographic parity difference?', options: ['0', 'About 0.10', 'It depends on the threshold'], answer: 1, explain: 'It approves 80% of A and 69% of B: a parity gap equal to the gap in true rates, while its equalised-odds difference is 0.' } },
      { p: 'So the perfect model satisfies equalised odds and violates demographic parity. When base rates differ, closing one gap means opening another, or giving up accuracy. Which gap matters depends on the harm: refusing good applicants, or lending to people who cannot repay. That decision belongs with the people affected.' },
      { h: 'Recap' },
      { list: ['Proxies carry the group’s signal when the group column is dropped.', 'Keep the group aside to measure gaps.', 'With different base rates the measures conflict; choose openly.'] },
      X('rb-x-drop-group'), X('rb-x-perfect-dpd'), X('rb-x-which-metric'),
    ],
  });

  C.lesson({
    id: 'rb-mitigate', chapter: 'rb-fair', title: 'Narrowing the gaps: reweighing and per-group cut-offs', minutes: 8,
    summary: 'Before training, reweighing gives each (group, outcome) cell the weight P(group)·P(outcome) / P(group, outcome), so outcome and group are independent in the weighted data. After training, a cut-off per group can match approval rates or true-positive rates. Each narrows the gap it aims at, sometimes widening another.',
    blocks: [
      'Gaps measured: what can be done? A fix can act on the data before training, on the training itself, or on the scores afterwards.',
      { topic: 'fair-mitigate', frame: 0, caption: 'Without mitigation: approval rates 0.88 and 0.79, a parity difference of 0.088 and an equalised-odds difference of 0.093, at 79% accuracy.' },
      { math: 'weight(group g, outcome y) = P(g) · P(y) / P(g, y)' },
      { topic: 'fair-mitigate', frame: 1, caption: 'Reweighing: cells rarer than independence would predict are weighted up, group A defaulted to 1.144 and group B repaid to 1.066.' },
      { topic: 'fair-mitigate', frame: 2, caption: 'Trained on the weighted rows: parity difference 0.042 (was 0.088), equalised-odds difference 0.091, accuracy 80%.' },
      { predict: { q: 'Cut-offs per group chosen to equalise approval rates: will the equalised-odds difference shrink too?', options: ['Yes, both gaps close', 'Not necessarily: it may grow', 'It stays exactly the same'], answer: 1, explain: 'Parity falls to 0.032, but the equalised-odds difference rises to 0.182.' } },
      { topic: 'fair-mitigate', frame: 3, caption: 'Per-group cut-offs for equal approval rates (0.557 for A, 0.381 for B): parity difference 0.032, but equalised-odds difference 0.182.' },
      { topic: 'fair-mitigate', frame: 3, input: { target: 'opportunity', seed: 1 }, caption: 'Cut-offs chosen instead so that repayers are approved equally often: parity difference 0.037, equalised-odds difference 0.045, accuracy 80%.' },
      { table: [['When', 'Method', 'Needs'], ['before training', 'reweighing, resampling', 'the group of each training row'], ['during training', 'a fairness penalty or constraint in the loss', 'a model that can take it'], ['after training', 'a cut-off per group', 'the group at decision time, which the law may not allow']], caption: 'Where each mitigation acts.' },
      { h: 'Recap' },
      { list: ['Reweighing makes outcome independent of group in the training data.', 'Per-group cut-offs can target parity or equal opportunity.', 'Say which gap you closed, and report the others.'] },
      X('rb-x-reweigh'), X('rb-x-mitigate-pre'), X('rb-x-mitigate-post'), X('rb-x-cutoff'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: privacy and the training data
  // =========================================================================================================
  C.lesson({
    id: 'rb-membership', chapter: 'rb-private', title: 'What a model gives away about its training rows', minutes: 7,
    summary: 'A model that memorises is more confident on rows it trained on than on new ones. That difference can reveal whether a person’s record was in the training set. The membership advantage, the best true-positive rate minus false-positive rate of such a test, measures the leak; limiting memorisation shrinks it.',
    blocks: [
      'A clinic trains a model on its patients and shares the model. Could the model itself reveal who was a patient? If it behaves differently on the rows it saw, yes. Here we measure that difference as an auditor would, to decide how much a model may be trusted with.',
      { topic: 'membership-inference', frame: 0, caption: 'A fully grown tree on 420 applicants: right on 100% of them, but on 69% of 180 it never saw. A gap that size means memorisation.' },
      { topic: 'membership-inference', frame: 1, caption: 'Its confidence in the true outcome averages 1.000 on training rows and 0.694 on unseen rows.' },
      { topic: 'membership-inference', frame: 2, caption: 'The best threshold flags 100% of members and 69% of unseen rows: an advantage of 0.306 over guessing.' },
      { predict: { q: 'Logistic regression is 81% right on its training rows and 79% on unseen ones. Will its advantage be larger or smaller than the tree’s?', options: ['Much smaller', 'About the same', 'Larger'], answer: 0, explain: 'Its advantage is 0.029: confidence on members and non-members nearly coincide.' } },
      { topic: 'membership-inference', frame: 2, input: { model: 'logistic', seed: 1 }, caption: 'Logistic regression: the best threshold flags 81% of members and 78% of unseen rows, an advantage of 0.029.' },
      { topic: 'membership-inference', frame: 3, caption: 'Across five models the leak is largest for the fully grown tree (0.306) and smallest for logistic regression (0.029).' },
      { h: 'Recap' },
      { list: ['Memorisation shows as a gap between training and test behaviour.', 'Membership advantage = TPR − FPR of the best membership test: 0 means nothing leaks.', 'Defences: less memorisation (simpler models, larger leaves, regularisation) and private training.'] },
      X('rb-x-mi-advantage'), X('rb-x-mi-defence'),
    ],
  });

  C.lesson({
    id: 'rb-dp', chapter: 'rb-private', title: 'Differential privacy in training: clip, then add noise', minutes: 8,
    summary: 'DP-SGD bounds each person’s influence on a training step by clipping every example’s gradient to length C, then adds Gaussian noise of standard deviation σC to the sum. The noise hides whether any one person was in the batch; an accountant turns σ, the sampling rate and the number of steps into a privacy loss ε.',
    blocks: [
      'To promise that the model would be almost the same with or without any one person’s record, each step must not depend much on any one record. How can a gradient step be made so?',
      { topic: 'dp-sgd', frame: 0, caption: 'One step of a logistic model on the loans data: 16 applicants, each with their own gradient, of length 0.86 to 2.04.' },
      { topic: 'dp-sgd', frame: 1, caption: '14 gradients are longer than C = 1.00 and are scaled down to exactly C. Now no single applicant can move the sum by more than C.' },
      { predict: { q: 'With C = 1 and σ = 1, what is the standard deviation of the noise added to each weight’s total?', options: ['1', '16', '1/16'], answer: 0, explain: 'σC = 1 × 1 = 1. Dividing by the batch afterwards scales noise and signal alike.' } },
      { topic: 'dp-sgd', frame: 2, caption: 'The clipped gradients add up to length 6.33; noise with standard deviation σC = 1.00 is added, of length 2.41.' },
      { topic: 'dp-sgd', frame: 3, caption: 'After 79 steps the model is 82% accurate on the test applicants, with a privacy loss of roughly ε ≈ 3.48 (δ = 1×10⁻⁵), an approximate teaching estimate.' },
      { warn: 'The ε values here come from a simplified accountant meant for teaching. Real deployments use a vetted library’s accountant, and state ε, δ and what counts as one person’s data.' },
      { h: 'Recap' },
      { list: ['Clip every example’s gradient to length C: nobody moves a step by more than C.', 'Add Gaussian noise with standard deviation σC to the sum.', 'A smaller ε means stronger privacy.'] },
      X('rb-x-clip'), X('rb-x-dp-noise'), X('rb-x-dp-eps'),
    ],
  });

  C.lesson({
    id: 'rb-dp-tradeoff', chapter: 'rb-private', title: 'The privacy–accuracy trade-off', minutes: 6,
    summary: 'More noise gives a smaller ε, stronger privacy, and costs accuracy. Reading the trade-off curve, choose the most accurate setting that meets the privacy budget, and check it against a model with no privacy at all.',
    blocks: [
      'How much accuracy does privacy cost? It depends on the noise, the data and the model; the honest answer is a curve.',
      { topic: 'dp-sgd', frame: 4, caption: 'Repeating the training at each noise level: with no noise the model reaches 81% but promises nothing; at σ = 8.0, ε ≈ 0.41 and accuracy falls to 69%.' },
      { predict: { q: 'A budget of ε ≤ 1 is set. Which noise level should be chosen?', options: ['The smallest σ that meets ε ≤ 1', 'The largest σ available', 'No noise, then report ε = 1'], answer: 0, explain: 'Every σ that meets the budget protects enough; the smallest of them costs the least accuracy.' } },
      { topic: 'dp-sgd', frame: 3, input: { sigma: 8, clip: 1, seed: 1 }, caption: 'One run at σ = 8: the noise has length 19.26 against a clipped sum of 6.33; this run ends at 62% accuracy, with ε ≈ 0.41.' },
      { tip: 'More data helps privacy: the same noise is spread over a larger sum, and each person is a smaller part of every step.' },
      { h: 'Recap' },
      { list: ['Noise buys privacy (smaller ε) with accuracy.', 'Pick the most accurate setting within the budget.', 'Report ε, δ and the accuracy without privacy for comparison.'] },
      X('rb-x-dp-tradeoff'),
    ],
  });

  C.lesson({
    id: 'rb-poison', chapter: 'rb-private', title: 'Poisoned labels, and cleaning them out', minutes: 7,
    summary: 'If some training labels are wrong, by accident or on purpose, the model learns the wrong boundary. A simple cleaning step asks each point’s nearest neighbours to vote and sets aside points whose label most of them disagree with. It helps when bad labels are scattered, and fails when they cluster, so knowing where data came from matters as much.',
    blocks: [
      'Training data is often gathered from many hands. What if some labels are wrong, and how would we notice?',
      { topic: 'data-poisoning', frame: 0, caption: '180 training points on either side of a straight rule: trained on true labels, 97% of the 120 test points right.' },
      { topic: 'data-poisoning', frame: 1, caption: '18 labels (10%) in one region are wrong. Retrained, the boundary tilts towards them and test accuracy falls to 84%.' },
      { topic: 'data-poisoning', frame: 2, caption: 'Cleaning: each point asks its 7 nearest neighbours for a vote. 22 points disagree with most of them and are set aside: 13 of the 18 wrong labels and 9 honest points near the boundary.' },
      { topic: 'data-poisoning', frame: 3, caption: 'Retrained without them: 91%, against 84% poisoned and 97% clean.' },
      { predict: { q: 'Now 25% of the labels are wrong, all in one region. Will the neighbour vote still catch most of them?', options: ['Yes', 'No: they outvote the honest points around them', 'It catches all of them'], answer: 1, explain: 'It sets aside only 3 of the 45 wrong labels; accuracy ends at 59%.' } },
      { topic: 'data-poisoning', frame: 3, input: { share: 0.25, k: 7, seed: 1 }, caption: 'With 45 wrong labels crowded together their neighbours vote with them: the cleaned model gets 59%, worse than the 68% poisoned. No single check is enough.' },
      { h: 'Recap' },
      { list: ['Wrong labels pull the boundary towards them.', 'A neighbour vote sets aside labels that disagree with their surroundings.', 'Clustered bad labels defeat it: track where data comes from, and compare against a trusted set.'] },
      X('rb-x-knn-clean'), X('rb-x-defend-labels'), X('rb-x-defence-match'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
