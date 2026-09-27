// Part 8, optimisation: the chapters and the lessons. The exercises are in 081-opt-ex.js; the model challenge, the
// tour, the course and the pages the diagram checker looks at are in 082-opt-ch.js. Each lesson opens with a question,
// runs the topic that shows the idea at a frame chosen for the point being made, asks for a prediction, sums up, and
// leads to its exercises. Every value quoted here was printed by the topics themselves (the engine's optimisers,
// S.T.optim, run on the topics' loss surfaces).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('op-surface')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'op-descent', part: 'optimisation', n: 1, title: 'Going downhill', blurb: 'A loss as a landscape over the parameters, the gradient as the steepest way up, one step at a time downhill, how large a step can be, and the noisy steps of stochastic descent.' });
  C.chapter({ id: 'op-faster', part: 'optimisation', n: 2, title: 'Faster ways down', blurb: 'Momentum remembers the way it was going; AdaGrad, RMSProp and Adam give every coordinate a step size of its own; six optimisers raced on the same surfaces.' });
  C.chapter({ id: 'op-sched', part: 'optimisation', n: 3, title: 'Schedules', blurb: 'Lowering the learning rate as training goes, by steps, exponentially or along a cosine, and starting it small with a warm-up.' });
  C.chapter({ id: 'op-land', part: 'optimisation', n: 4, title: 'The shape of the landscape', blurb: 'Convex losses where every way down leads to the bottom; bumpy ones with local minima and their basins; plateaus where the gradient all but vanishes.' });

  // =========================================================================================================
  // Chapter 1: going downhill
  // =========================================================================================================
  C.lesson({
    id: 'op-surface', chapter: 'op-descent', title: 'A loss is a landscape', minutes: 6,
    summary: 'A model’s loss is a function of its parameters. With two parameters it is a surface over a plane: a height for every setting. The gradient points the steepest way up, at right angles to the contour lines, and training walks the other way.',
    blocks: [
      'Training a model means choosing its parameters to make the loss small. With two parameters we can draw every choice at once. What does the problem look like?',
      { topic: 'opt-surface', frame: 0, caption: 'A long ravine: the loss L(u, v) drawn as a height over the plane of the two parameters u and v, from 0 to 20 over this box.' },
      { topic: 'opt-surface', frame: 4, caption: 'The same surface seen from above, as a contour map: each line joins points of equal loss, darker where the loss is lower.' },
      { def: 'Gradient', text: '∇L = (∂L/∂u, ∂L/∂v): the direction in which the loss rises fastest, and how fast. −∇L is the steepest way down.' },
      { topic: 'opt-surface', frame: 5, caption: 'Each green arrow is −∇L. Longer arrows mark steeper ground; the steepest here is 9.63 long.' },
      { predict: { q: 'How does the downhill arrow at a point meet the contour line through that point?', options: ['At a right angle', 'Along the line', 'At 45°'], answer: 0, explain: 'Along the contour line the loss does not change, so all of the slope lies across it.' } },
      { topic: 'opt-surface', frame: 6, caption: 'At (−4, 1.6) the loss is 6.72 and ∇L = (−0.8, 6.4): almost all of the slope is in v. The red arrow, −∇L, crosses the level line at a right angle.' },
      { topic: 'opt-surface', frame: 7, caption: 'The lowest point is (0, 0). The Hessian’s eigenvalues there are 4 and 0.2: steep across, flat along, 20 times apart. That shape makes descent zigzag.' },
      { h: 'Recap' },
      { list: ['The loss is a surface over the parameters; contour lines join equal losses.', 'The gradient points the steepest way up, at right angles to the contour lines.', 'Very different curvatures in different directions make a ravine.'] },
      X('op-x-grad'), X('op-x-hessian'),
    ],
  });

  C.lesson({
    id: 'op-gd', chapter: 'op-descent', title: 'Gradient descent, one step at a time', minutes: 7,
    summary: 'Gradient descent repeats one rule: θ ← θ − η∇L(θ). The steps are long where the ground is steep and shrink as the gradient fades near the bottom. In a ravine the steepest direction points across it, so the walk zigzags.',
    blocks: [
      'Standing on a hillside in fog, you can feel only the slope under your feet. What is a sensible rule for getting down?',
      { math: 'θ ← θ − η ∇L(θ)       (η: the learning rate)' },
      { topic: 'opt-gd', frame: 0, caption: 'A tilted valley. Start at (−2.5, 0.6), where the loss is 2.766, with η = 0.3.' },
      { topic: 'opt-gd', frame: 1, caption: 'Step 1: the gradient is (−2.52, −1.28). Times η = 0.3 the step is (0.756, 0.384), and the loss falls from 2.766 to 1.033.' },
      { predict: { q: 'Nearer the bottom, what happens to the length of the steps, with η unchanged?', options: ['They shrink', 'They stay the same', 'They grow'], answer: 0, explain: 'A step is η times the gradient, and the gradient fades towards the bottom.' } },
      { topic: 'opt-gd', frame: 13, caption: 'After 25 steps the walk is at (−0.063, 0.063), loss 0.00161; the last step was only 0.0122 long, because the gradient there is 0.0408.' },
      { topic: 'opt-gd', frame: 4, input: { surf: 'ravine', lr: 0.45, steps: 40 }, caption: 'A ravine with η = 0.45: the gradient points mostly across the ravine, so the walk bounces from wall to wall, v = 1.6, −1.28, 1.02, −0.819, while it creeps along the floor.' },
      { topic: 'opt-gd', frame: 15, input: { surf: 'ravine', lr: 0.45, steps: 40 }, caption: 'It gets there, after 40 steps: loss 0.000846. Most of its effort went into the zigzag.' },
      { h: 'Try it on the canvas' },
      { workflow: 'mlp-spirals', title: 'A network learning the spirals', text: 'The Train block uses the same rule on the network’s weights, one batch at a time. Double-click Train to change the optimiser and the learning rate, and watch the loss curve.' },
      { h: 'Recap' },
      { list: ['Gradient descent steps by −η∇L.', 'Steps are long on steep ground and shrink near the bottom.', 'In a ravine the steepest direction points across it, so the path zigzags.'] },
      X('op-x-gd-step'), X('op-x-gd-loss'), X('op-x-gd-curve'), X('op-x-code-gd'),
    ],
  });

  C.lesson({
    id: 'op-lr', chapter: 'op-descent', title: 'The learning rate: too small, just right, too large', minutes: 7,
    summary: 'Each step multiplies the error along a direction of curvature λ by 1 − ηλ. If any factor is larger than 1 in size the walk explodes, so η must stay below 2/λmax. The slowest factor sets the pace, so a ravine, with very different curvatures, is slow for any η.',
    blocks: [
      'A larger step gets there sooner, surely? Up to a point. Where is that point, and what happens beyond it?',
      { topic: 'opt-lr', frame: 0, caption: 'A bowl with curvatures λmax = 2 and λmin = 0.4. Gradient descent is stable only below 2/λmax = 1. Three rates, 0.075, 0.833 and 1.06, and a fourth, 0.6, start from the same point.' },
      { math: 'along a direction of curvature λ:  error ← (1 − ηλ) · error\nstable when |1 − ηλ| < 1 for every λ, that is  0 < η < 2 / λmax' },
      { topic: 'opt-lr', frame: 1, caption: 'The factors along the steep direction: 0.85 for the small rate, −0.667 for the balanced one, −1.12 for the large one. A negative factor jumps over the floor to the other wall.' },
      { predict: { q: 'With the factor −1.12, what happens over 30 steps?', options: ['The loss shrinks, slowly', 'The loss grows: every overshoot is bigger than the last', 'It stays the same'], answer: 1, explain: '|−1.12| > 1, so the error along that direction grows by 12% a step, flipping sides each time.' } },
      { topic: 'opt-lr', frame: 9, caption: 'After 30 steps: 0.155 for the small rate, still creeping; 7.52 × 10⁻¹¹ for the balanced one; 1620 for the large one. The rate 0.6, inside the limit, ends at 6.78 × 10⁻⁸.' },
      { topic: 'opt-lr', frame: 0, input: { surf: 'ravine', lr: 0.3, steps: 60 }, caption: 'The ravine: λmax = 4 and λmin = 0.2, so η must stay below 0.5; then the flat direction shrinks by only 1 − 0.5 · 0.2 = 0.9 a step at best. The ratio λmax/λmin, here 20, measures how badly conditioned a problem is.' },
      { tip: 'Scaling the features makes the loss of a linear model rounder, so a larger η is stable and the slow direction is not so slow. That is one reason to scale before gradient descent.' },
      { h: 'Recap' },
      { list: ['Each step multiplies the error along curvature λ by 1 − ηλ.', 'Stable only for η < 2/λmax; beyond it the walk explodes.', 'The pace is set by the slowest direction; a large λmax/λmin is slow for any η.'] },
      X('op-x-lr-limit'), X('op-x-lr-choice'),
    ],
  });

  C.lesson({
    id: 'op-sgd', chapter: 'op-descent', title: 'Stochastic and mini-batch descent', minutes: 7,
    summary: 'The full gradient sums over every example. A mini-batch estimates it from a few, at a fraction of the cost, pointing roughly but not exactly downhill. For the same work SGD takes many more steps; with a fixed rate it never quite settles, but jitters round the bottom.',
    blocks: [
      'The loss of a model is an average over the training examples, and so is its gradient. With a million examples, one exact step costs a million gradients. Is the exact gradient worth that?',
      { topic: 'opt-sgd', frame: 0, caption: '40 points round a line. The loss of the line y = a·x + b is half the mean squared error, a bowl whose bottom is at a = 1.32, b = −0.63, loss 0.121. Three runs start together with η = 0.25.' },
      { topic: 'opt-sgd', frame: 1, caption: 'Step 1: a mini-batch of 5 points gives the gradient (−1.93, 1.06); all 40 give (−1.7, 1.32). They point within 9 degrees of each other: roughly downhill.' },
      { predict: { q: 'Which run reaches a low loss for the least computing, counted in single-example gradients?', options: ['Full batch: its steps are exact', 'Mini-batches or single points: their steps are cheap', 'All the same'], answer: 1, explain: 'A full step costs 40 gradients; for that price one-point SGD takes 40 noisy steps, which together go much further.' } },
      { topic: 'opt-sgd', frame: 9, caption: 'Step 20: losses 0.1206 (full batch), 0.1217 (mini-batch) and 0.1236 (one point).' },
      { topic: 'opt-sgd', frame: 12, caption: 'After 40 steps: 0.1206, 0.1217 and 0.1478, against the best possible 0.1206. With η fixed the noisy runs never settle: near the bottom each batch still pulls its own way.' },
      { table: [['Batch', 'Cost of a step', 'Direction', 'Near the bottom'], ['all examples', 'n gradients', 'exact', 'settles'], ['a mini-batch of B', 'B gradients', 'a noisy estimate', 'jitters, less for larger B'], ['one example', '1 gradient', 'very noisy', 'jitters most']], caption: 'Mini-batches of tens to hundreds are the usual compromise, and suit parallel hardware.' },
      { h: 'Recap' },
      { list: ['A mini-batch gradient is a cheap, noisy estimate of the full gradient.', 'For the same work, stochastic descent takes many more steps and usually gets further.', 'With a fixed rate it jitters round the bottom; a falling rate, chapter 3, lets it settle.'] },
      X('op-x-batch-grad'), X('op-x-sgd-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: faster ways down
  // =========================================================================================================
  C.lesson({
    id: 'op-momentum', chapter: 'op-faster', title: 'Momentum and Nesterov', minutes: 8,
    summary: 'Momentum keeps a velocity: v ← βv + ∇L, θ ← θ − ηv. Where the gradients agree from step to step the velocity builds up; where they flip, as across a ravine, they cancel. Nesterov looks ahead, taking the gradient where the velocity is about to carry it.',
    blocks: [
      'In the ravine, plain descent spends its steps bouncing across. The bounces flip sign each time; the drift along the floor always points the same way. Could the walk remember its direction?',
      { math: 'v ← β v + ∇L(θ)\nθ ← θ − η v           (β between 0 and 1; the first v is the first gradient)' },
      { topic: 'opt-momentum', frame: 0, caption: 'Three runs from the same start in the ravine with η = 0.1: plain descent, momentum and Nesterov, both with β = 0.8.' },
      { topic: 'opt-momentum', frame: 2, caption: 'Step 2: momentum’s velocity is (−1.42, 8.96). The blue arrow is where it would coast on its own, −ηβv.' },
      { predict: { q: 'After 40 steps, which is lower, plain descent or momentum?', options: ['Plain descent', 'Momentum, by orders of magnitude', 'About the same'], answer: 1, explain: 'Plain descent ends at 0.318, momentum at 5.13 × 10⁻⁴: the flips across the ravine cancel in v and the drift along it adds up.' } },
      { topic: 'opt-momentum', frame: 12, caption: 'After 40 steps: plain 0.318, momentum 5.13 × 10⁻⁴, Nesterov 1.66 × 10⁻⁴.' },
      { topic: 'opt-momentum', frame: 3, caption: 'The cost: at step 3 momentum’s loss went up, to 2.21 from 1.44, as the built-up velocity carried it past the floor. Memory pays where gradients agree and costs a few overshoots where they swing.' },
      { topic: 'opt-momentum', frame: 14, input: { surf: 'ravine', lr: 0.1, beta: 0.95, steps: 60 }, caption: 'With β = 0.95 the memory is longer: after 60 steps momentum is at 0.0203 and Nesterov at 0.00468, both with more overshooting on the way.' },
      { h: 'Recap' },
      { list: ['Momentum keeps a velocity v ← βv + ∇L and steps by −ηv.', 'Gradients that agree build up speed; gradients that flip cancel.', 'Nesterov takes the gradient where the velocity is heading, and overshoots less.'] },
      X('op-x-momentum-table'), X('op-x-ravine-choice'),
    ],
  });

  C.lesson({
    id: 'op-adaptive', chapter: 'op-faster', title: 'A step size for each coordinate', minutes: 9,
    summary: 'AdaGrad, RMSProp and Adam divide each coordinate’s gradient by the root of a running sum or average of its squares, so steep and flat coordinates move at similar speeds. Adam also keeps a momentum-like average of the gradients and corrects both averages for starting at zero.',
    blocks: [
      'In the ravine one coordinate’s gradient is 8 times the other’s. A rate safe for the steep one starves the flat one. What if each coordinate had a rate of its own, set by the size of its own gradients?',
      { topic: 'opt-adaptive', frame: 0, caption: 'Adam with η = 0.3, beside AdaGrad, RMSProp and plain gradient descent, in the ravine. The bars show, for each coordinate, the gradient, the root of the running g² it is divided by, and the step.' },
      { math: 'AdaGrad\n  h ← h + g²\n  θ ← θ − η g / (√h + ε)\nRMSProp\n  s ← ρs + (1 − ρ) g²\n  θ ← θ − η g / (√s + ε)\nAdam\n  m ← β₁m + (1 − β₁) g\n  v ← β₂v + (1 − β₂) g²\n  m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)\n  θ ← θ − η m̂ / (√v̂ + ε)' },
      { topic: 'opt-adaptive', frame: 1, caption: 'Step 1: the gradient is (−0.8, 6.4), yet Adam’s step is (0.3, −0.3): each coordinate, divided by its own size, moves about η. RMSProp’s first step is 10 times as long: its average of g² starts at zero and nothing corrects for that.' },
      { predict: { q: 'What is Adam’s very first step, for any gradient g with no zero coordinate?', options: ['About η in each coordinate, against the sign of g', 'η times g', 'Zero: the averages start at zero'], answer: 0, explain: 'After one step m̂ = g and v̂ = g², so the step is η g / |g| = η times the sign of g.' } },
      { topic: 'opt-adaptive', frame: 12, caption: 'After 40 steps: gradient descent 0.0113, AdaGrad 0.17, RMSProp 2.58 × 10⁻²⁴, Adam 0.042. Adam’s first moment m = (0.0586, −0.095) is its momentum.' },
      { topic: 'opt-adaptive', frame: 14, input: { method: 'adagrad', surf: 'ravine', lr: 0.3, steps: 60 }, caption: 'AdaGrad’s sums only grow (here to 9.49 and 162), so its steps keep shrinking even where the ground is flat: after 60 steps it is at 0.0755. RMSProp and Adam forget old gradients and stay lively.' },
      { tip: 'Adam with η around 0.001 to 0.01 is the usual first choice for training networks: it needs little tuning. Plain SGD with momentum and a schedule can match or beat it, with more care.' },
      { h: 'Recap' },
      { list: ['Dividing by the root of the squared gradients gives each coordinate its own step size.', 'AdaGrad sums forever, so it slows down; RMSProp and Adam use moving averages.', 'Adam adds momentum and a bias correction: its first step is about η in each coordinate.'] },
      X('op-x-adam-step'), X('op-x-adam-two'), X('op-x-rmsprop'), X('op-x-adagrad'), X('op-x-code-adam'),
    ],
  });

  C.lesson({
    id: 'op-race', chapter: 'op-faster', title: 'Six optimisers, one surface', minutes: 7,
    summary: 'Raced on the same surfaces, each with its best rate, no optimiser wins everywhere. The adaptive ones shine in a ravine; on a saddle and a plateau the ranking changes again. Choose by trying, on the problem at hand.',
    blocks: [
      'Every optimiser has a story for why it is faster. Put them side by side on the same surface, each with its own best learning rate: who wins?',
      { topic: 'opt-race', frame: 0, caption: 'All six start at (−4, 1.6) in the ravine, loss 6.72. Their rates are each the best of a grid over 100 steps: GD 0.3, momentum 0.05, Nesterov 0.3, AdaGrad 0.8, RMSProp 0.5, Adam 0.5.' },
      { topic: 'opt-race', frame: 3, caption: 'Step 10: RMSProp leads with loss 1.08 × 10⁻⁸, then AdaGrad at 0.0957; momentum trails at 0.617.' },
      { predict: { q: 'Is the same optimiser first on a saddle?', options: ['Yes: the best optimiser is best everywhere', 'Not necessarily: the ranking depends on the surface'], answer: 1, explain: 'On the saddle, gradient descent takes the lead by step 15 and keeps the lowest loss on the way.' } },
      { topic: 'opt-race', frame: 11, caption: 'After 100 steps, RMSProp and AdaGrad have reached the bottom; the others are still above it, momentum by 5.3 × 10⁻⁵.' },
      { topic: 'opt-race', frame: 10, input: { surf: 'saddle', steps: 80 }, caption: 'On the saddle, 4 have reached the bottom after 80 steps, ranked gradient descent, Nesterov, AdaGrad and RMSProp; momentum and Adam are still above it.' },
      { tip: 'The race is on a two-parameter surface with every rate tuned. Real networks have millions of parameters and noisy gradients, where Adam’s low need for tuning often matters most.' },
      { h: 'Recap' },
      { list: ['Tune each optimiser’s rate before comparing them.', 'No optimiser is best on every surface.', 'Adaptive methods handle very different curvatures; momentum handles consistent drift.'] },
      X('op-x-plateau-choice'), X('op-x-wf-momentum'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: schedules
  // =========================================================================================================
  C.lesson({
    id: 'op-schedules', chapter: 'op-sched', title: 'Learning-rate schedules', minutes: 7,
    summary: 'With noisy gradients, a rate large enough for fast early progress keeps the parameters jittering round the minimum for ever. A schedule lowers the rate as training goes: in steps, exponentially, or along a half cosine, so the jitter dies down into a settle.',
    blocks: [
      'Stochastic descent with a fixed rate jitters round the bottom. A smaller rate jitters less, but crawls at the start. Can we have the large rate early and the small one late?',
      { topic: 'opt-schedules', frame: 0, caption: 'Four runs of one-point stochastic descent, all starting at η = 0.3, fed the same points in the same order. Only the schedule differs (top left).' },
      { table: [['Schedule', 'Rate at step t (from η₀)'], ['constant', 'η₀'], ['step decay', 'η₀ × 0.3 every 20 steps'], ['exponential', 'η₀ × γᵗ, here falling to 2% at the end'], ['cosine', 'η₀ × ½(1 + cos(πt / N))']], caption: 'The four schedules of a run of N = 60 steps.' },
      { topic: 'opt-schedules', frame: 4, caption: 'Step 30: the rates used are 0.3, 0.09, 0.0453 and 0.158; the losses are all close to 0.16.' },
      { predict: { q: 'At the end of the run, which path stays closest to the best fit?', options: ['The constant rate', 'A falling one', 'All the same'], answer: 1, explain: 'Over the last fifth of the run the constant rate strays up to 0.586 from the best fit; the cosine run only 0.207.' } },
      { topic: 'opt-schedules', frame: 7, caption: 'After 60 steps the constant rate still strays up to 0.586 from the best fit, the cosine run 0.207: a falling rate turns the jitter into a settle.' },
      { h: 'Recap' },
      { list: ['Large rates make fast early progress; small rates settle.', 'Step, exponential and cosine schedules lower the rate as training goes.', 'The cosine schedule falls slowly at first, fastest in the middle, and gently to its end.'] },
      X('op-x-step-sched'), X('op-x-cosine'), X('op-x-sched-curve'),
    ],
  });

  C.lesson({
    id: 'op-warmup', chapter: 'op-sched', title: 'Warm-up: starting small', minutes: 6,
    summary: 'In its first steps an optimiser knows little: Adam’s averages hold one or two gradients, and its first step is about η in every coordinate whatever the slope. A warm-up raises the rate from near zero over the first steps, so early steps cannot throw the parameters somewhere bad.',
    blocks: [
      'Adam’s first step is about η in each coordinate, even on a steep wall where the right step is tiny. With a large η, what can that first step do?',
      { topic: 'opt-warmup', frame: 0, caption: 'Adam with η = 0.5 in the banana valley, twice from the same start: once at full rate from the first step, once warming up in a straight line over the first 15 steps.' },
      { math: 'warm-up over W steps:  η_t = η · (t + 1) / W  for t < W, then the schedule that follows' },
      { topic: 'opt-warmup', frame: 1, caption: 'Step 1: without warm-up Adam’s step is 0.707 long, though it has seen only one gradient. With warm-up the rate is 0.0333 and the step 0.0471.' },
      { predict: { q: 'Which run reaches the higher loss on its way?', options: ['The cold start, at full rate', 'The warm-up run', 'Neither: both only go down'], answer: 0, explain: 'The cold start climbs to a loss of 89; the warm one never goes above 7.64.' } },
      { topic: 'opt-warmup', frame: 9, caption: 'After 60 steps: 1.79 without warm-up, 0.0152 with it. The cold start climbed to 89 on its way; the warm one never went above 7.64.' },
      { tip: 'Warm-up matters most for large models trained with Adam and large rates; transformers are usually trained with a warm-up followed by a decay.' },
      { h: 'Recap' },
      { list: ['Early steps are taken with little knowledge of the terrain.', 'A warm-up raises the rate gradually over the first W steps.', 'It prevents early blow-ups and lets a larger peak rate be used.'] },
      X('op-x-warmup'), X('op-x-warmup-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: the shape of the landscape
  // =========================================================================================================
  C.lesson({
    id: 'op-convex', chapter: 'op-land', title: 'Convex or not: the chord test', minutes: 7,
    summary: 'A loss is convex when every chord between two points of its graph lies on or above the graph. Then every way down leads to the one bottom, and gradient descent cannot be trapped. Linear and logistic regression and the SVM have convex losses; neural networks do not.',
    blocks: [
      'Gradient descent only ever looks at the slope under its feet. When can we be sure that wherever it stops is the best place there is?',
      { def: 'Convex', text: 'L is convex when L(λa + (1 − λ)b) ≤ λL(a) + (1 − λ)L(b) for all a, b and 0 ≤ λ ≤ 1: the chord from (a, L(a)) to (b, L(b)) is never below the graph. For a smooth L of one variable, the same as L″ ≥ 0 everywhere.' },
      { topic: 'opt-convex', frame: 6, input: { fn: 'square' }, caption: 'A parabola, L(x) = ½(x − 0.5)²: every chord stays above it, and gradient descent from x = −2.8 and from 2.8 both end at x = 0.5, loss 0.' },
      { topic: 'opt-convex', frame: 2, caption: 'A double well, L(x) = x⁴/4 − x² + 0.3x: the chord from x = −2 to 0.4 dips below the graph, by 0.129 at x = 0.03. One such chord is enough: not convex.' },
      { predict: { q: 'Gradient descent starts from x = −2.8 and from x = 2.8 on the double well. Where do the runs end?', options: ['Both at the lowest point', 'In different dips, one of them not the lowest', 'Neither stops'], answer: 1, explain: 'They stop at x = −1.48 (loss −1.43) and x = 1.33 (loss −0.588): the slope is zero at both, but only one is the lowest point.' } },
      { topic: 'opt-convex', frame: 5, caption: 'The test by curvature: L″ dips below zero, to −2 at x = 0. Where the curvature is negative the graph bends down like a hilltop.' },
      { topic: 'opt-convex', frame: 6, caption: 'The run from the right stops at x = 1.33, loss −0.588: a local minimum. It looks like the global one from inside it.' },
      { h: 'Recap' },
      { list: ['Convex: every chord on or above the graph; for one variable, L″ ≥ 0.', 'On a convex loss every local minimum is a global one.', 'Linear and logistic regression and the SVM are convex; networks are not.'] },
      X('op-x-chord'), X('op-x-convex-set'),
    ],
  });

  C.lesson({
    id: 'op-local', chapter: 'op-land', title: 'Local minima, basins and plateaus', minutes: 8,
    summary: 'On a bumpy loss, where gradient descent ends depends on where it starts: every pit has a basin of starts that drain into it. On a plateau the gradient almost vanishes and plain descent crawls, while momentum and Adam, which do not depend on the size of the gradient alone, cross it.',
    blocks: [
      'A network’s loss is far from convex. Does gradient descent end in the same place whatever the start?',
      { topic: 'opt-local', frame: 0, caption: 'A bumpy loss with 4 pits, the deepest at (−1.24, −1.15) with loss −1.42. Gradient descent with η = 0.3 sets off from each of 16 starts on a grid.' },
      { topic: 'opt-local', frame: 4, caption: 'After 120 steps: 6 runs in the deepest pit, 4 in the pit at −0.783, 3 at −0.524 and 3 at −0.28.' },
      { predict: { q: 'What decides which pit a run ends in?', options: ['Only where it starts: the basin it begins in', 'The size of the pit’s loss', 'Chance at every step'], answer: 0, explain: 'Plain gradient descent is deterministic: the start fixes the path, and the path drains into the pit whose basin holds the start.' } },
      { topic: 'opt-local', frame: 5, caption: 'The basins: every place coloured like a pit drains into it. The deepest pit’s basin covers 37% of the map, the largest share.' },
      { h: 'Plateaus' },
      { topic: 'opt-plateau', frame: 0, caption: 'Start at (3.6, 0.9) on a plateau round a pit: the loss is 0.9993, almost its ceiling of 1, and the gradient is only 0.00275 long.' },
      { topic: 'opt-plateau', frame: 8, caption: 'Below loss 0.5: plain descent at step 114, momentum at step 29, Adam at step 24. On a plateau the size of the gradient, not the distance to the pit, sets the pace of plain descent.' },
      { tip: 'Several starts, momentum, and the noise of mini-batches all help a run out of a poor basin or across a plateau. In large networks, poor local minima are rarer than plateaus and saddles.' },
      { h: 'Recap' },
      { list: ['On a bumpy loss the start decides the pit: each pit has its basin.', 'Several starts, keeping the best, raise the chance of a deep pit.', 'On a plateau plain descent crawls; momentum and Adam cross it far sooner.'] },
      X('op-x-plateau-grad'), X('op-x-saddle'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
