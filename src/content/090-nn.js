// Part 9, neural networks: the chapters and the lessons. The exercises are in 091-nn-ex.js; the model challenge, the
// tour, the course and the pages the diagram checker looks at are in 092-nn-ch.js. Each lesson opens with a question,
// runs the topic that shows the idea at a frame chosen for the point being made, asks for a prediction, sums up, and
// leads to its exercises. Every value quoted here was printed by the topics themselves (the engine's layers,
// activations, autodiff and optimisers, S.T and S.DL).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('ne-neuron')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'ne-units', part: 'networks', n: 1, title: 'Neurons and the forward pass', blurb: 'One neuron as a weighted sum and a squash, the activations and their slopes, and a small network computing a value on every wire.' });
  C.chapter({ id: 'ne-learn', part: 'networks', n: 2, title: 'How a network learns', blurb: 'Backpropagation on the computational graph with every gradient shown, the loss surface of a tiny network with its twins and saddle, and why the first weights matter.' });
  C.chapter({ id: 'ne-reg', part: 'networks', n: 3, title: 'Fitting without memorising', blurb: 'A large network memorising thirty points, then the cures: dropout, weight decay and early stopping; batch and layer normalisation; and the playground, where every neuron’s region is drawn.' });

  // =========================================================================================================
  // Chapter 1: neurons and the forward pass
  // =========================================================================================================
  C.lesson({
    id: 'ne-neuron', chapter: 'ne-units', title: 'One neuron: a weighted sum and a squash', minutes: 6,
    summary: 'A neuron multiplies each input by a weight, adds a bias, and passes the sum z through an activation. Its slope at z decides how strongly training can move each weight. Over the plane, one neuron splits the inputs with a straight line.',
    blocks: [
      'Every network in this part, however large, is built from one small piece repeated. What does that piece compute, and what can it decide on its own?',
      { topic: 'p09-neuron', frame: 0, caption: 'Two inputs, x₁ = 0.5 and x₂ = −1, two weights, 1.2 and −0.8, and a bias of 0.3. The curve on the right is the activation, here the sigmoid.' },
      { math: 'z = w₁x₁ + w₂x₂ + b\na = f(z)' },
      { topic: 'p09-neuron', frame: 3, caption: 'The weighted sum: 0.60 + 0.80 + 0.3 = 1.70. Both products push z up, because the second weight and the second input are both negative.' },
      { predict: { q: 'The sigmoid squashes any number into (0, 1). Roughly what is sigmoid(1.70)?', options: ['About 0.85', 'About 1.70', 'Exactly 0.5'], answer: 0, explain: 'sigmoid(1.70) = 1 / (1 + e^−1.70) = 0.846: past the middle of the curve, not yet flat.' } },
      { topic: 'p09-neuron', frame: 4, caption: 'The output is 0.846, and the slope of the curve there is 0.131. That slope is what training will use.' },
      { topic: 'p09-neuron', frame: 5, caption: 'How the output moves with each weight: ∂a/∂w₁ = f′(z) · x₁ = 0.065 and ∂a/∂w₂ = −0.131. A weight on a larger input gets a larger push.' },
      { topic: 'p09-neuron', frame: 4, input: { x1: -1, x2: 0.5, w1: 1.5, w2: -1, b: 0, act: 'relu' }, caption: 'A ReLU neuron with z = −2: its output is 0 and so is its slope. For this input no gradient can flow back through it.' },
      { topic: 'p09-neuron', frame: 6, input: { x1: 1, x2: 1, w1: 2, w2: 2, b: -3, act: 'sigmoid' }, caption: 'Weights 2 and 2 with bias −3 over the whole plane: the line z = 0 lies 1.06 from the origin, and only points where both inputs are large get past it, an AND of the two.' },
      { h: 'Recap' },
      { list: ['A neuron computes z = Σ wᵢxᵢ + b and outputs a = f(z).', 'The slope f′(z) scales every weight’s gradient; a flat or switched-off neuron learns nothing from that input.', 'One neuron splits the plane with a straight line, |b| / ‖w‖ from the origin.'] },
      X('ne-x-neuron-out'), X('ne-x-neuron-relu'), X('ne-x-neuron-line'),
    ],
  });

  C.lesson({
    id: 'ne-acts', chapter: 'ne-units', title: 'Activations and their slopes', minutes: 7,
    summary: 'Sigmoid and tanh squash their input and go flat at both ends; ReLU passes positives unchanged and blocks negatives; GELU is a smooth ReLU. Their slopes multiply every gradient on its way back, so a slope under one in every layer shrinks the gradient towards nothing.',
    blocks: [
      'Without an activation, a stack of layers is one big weighted sum: a straight line, however deep. The activation bends it. But which bend, and does the choice matter?',
      { topic: 'p09-activations', frame: 1, caption: 'tanh at z₀ = 1.5: the output is 0.905 and the tangent’s slope 0.181.' },
      { topic: 'p09-activations', frame: 3, caption: 'Shaded where the slope is below 0.05: 64% of the range from −6 to 6. There the neuron is saturated and passes back almost no gradient.' },
      { table: [['Activation', 'Formula', 'Slope'], ['sigmoid', '1 / (1 + e^−z)', 'σ(z)(1 − σ(z)), at most 0.25'], ['tanh', '(e^z − e^−z) / (e^z + e^−z)', '1 − tanh²(z), at most 1'], ['ReLU', 'max(0, z)', '1 for z > 0, 0 for z < 0'], ['GELU', '≈ ½ z (1 + tanh(√(2/π)(z + 0.044715 z³)))', 'smooth, near 1 for large z']], caption: 'The four activations of the topics.' },
      { topic: 'p09-activations', frame: 4, caption: 'All four at z₀ = 1.5: sigmoid 0.818 (slope 0.149), tanh 0.905 (0.181), ReLU 1.5 (1), GELU 1.4 (1.128).' },
      { predict: { q: 'A gradient goes back through 10 sigmoid layers whose neurons all sit at z = 0, where the slope is largest. How much of it is left?', options: ['About a millionth', 'About a quarter', 'All of it'], answer: 0, explain: 'Each layer multiplies it by at most 0.25, and 0.25¹⁰ = 9.5 × 10⁻⁷.' } },
      { topic: 'p09-activations', frame: 5, input: { act: 'sigmoid', z0: 0, depth: 10 }, caption: 'Even at the sigmoid’s steepest point, 10 layers leave 9.5 × 10⁻⁷ of the gradient: the first layers would barely learn.' },
      { topic: 'p09-activations', frame: 5, input: { act: 'relu', z0: 2, depth: 20 }, caption: 'ReLU on its positive side has slope exactly 1: after 20 layers the whole gradient is still there. Its price is the flat negative side, where a neuron can go dead.' },
      { h: 'Recap' },
      { list: ['Sigmoid and tanh saturate at both ends; their slopes vanish there.', 'Backpropagation multiplies the slopes of every layer: many slopes under one make the gradient vanish.', 'ReLU and GELU keep a slope near 1 for positive inputs, which is why deep networks use them.'] },
      X('ne-x-sig-slope'), X('ne-x-slopes-table'), X('ne-x-vanish'), X('ne-x-act-choice'), X('ne-x-code-act'),
    ],
  });

  C.lesson({
    id: 'ne-forward', chapter: 'ne-units', title: 'The forward pass: a value on every wire', minutes: 7,
    summary: 'A layer is many neurons reading the same inputs. Each hidden neuron computes its weighted sum and activation; each output neuron sums the hidden values into a score; softmax turns the scores into probabilities. Repeated at every point of the plane, the same arithmetic draws the boundary.',
    blocks: [
      'A trained network is handed one point. What happens between the point going in and a class coming out, wire by wire?',
      { topic: 'p09-forward', frame: 0, caption: 'The point (0.2, 0.9) enters a network of 3 tanh neurons trained on the two moons. Blue wires carry positive weights, vermilion negative; thicker is larger.' },
      { topic: 'p09-forward', frame: 1, caption: 'Hidden neuron h₁: −4.01 × 0.2 + (−3.01) × 0.9 + 2.55 = −0.96, and tanh(−0.96) = −0.74.' },
      { math: 'hidden:  zⱼ = Σᵢ wᵢⱼ xᵢ + bⱼ,   hⱼ = f(zⱼ)\noutput:  sₖ = Σⱼ vⱼₖ hⱼ + cₖ\nsoftmax: pₖ = e^sₖ / Σ e^s' },
      { topic: 'p09-forward', frame: 4, caption: 'Output 0 adds up the three hidden values through its own wires, 2.25, −2.10 and 2.38, plus its bias −0.44: a score of 2.08. Outputs have no activation.' },
      { predict: { q: 'The scores are 2.08 for class 0 and −2.71 for class 1. What decides the probabilities?', options: ['Only the gap between the scores', 'Only the larger score', 'The sum of the scores'], answer: 0, explain: 'With two classes softmax gives p(class 1) = σ(s₁ − s₀), so adding the same number to both scores changes nothing.' } },
      { topic: 'p09-forward', frame: 6, caption: 'The gap is −4.79, so p(class 1) = σ(−4.79) = 0.008: class 0, with 0.992.' },
      { topic: 'p09-forward', frame: 7, caption: 'The same arithmetic at every point of the plane: the shading is p(class 1), the boundary where the scores are equal. These weights get all 120 training points right.' },
      { topic: 'p09-forward', frame: 7, input: { weights: 'random' }, caption: 'The same network with the weights it started with: the boundary is wherever chance put it, and 19% of the training points are right. Training is what moved every wire.' },
      { h: 'Try it on the canvas' },
      { workflow: 'mlp-spirals', title: 'A network learning the spirals', text: 'Two hidden layers of eight neurons on the canvas. Train it, then open the MLP block to see every weight drawn as a wire and the probe point’s value on every neuron.' },
      { h: 'Recap' },
      { list: ['Each hidden neuron computes a weighted sum of all the inputs and an activation.', 'Output neurons sum the hidden values into scores; softmax turns scores into probabilities.', 'A layer of n inputs and m neurons has n·m weights and m biases.'] },
      X('ne-x-forward'), X('ne-x-softmax'), X('ne-x-count-weights'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: how a network learns
  // =========================================================================================================
  C.lesson({
    id: 'ne-backprop', chapter: 'ne-learn', title: 'Backpropagation on the computational graph', minutes: 9,
    summary: 'Write the network as a graph of small operations. The forward pass computes every node’s value; the backward pass starts with ∂L/∂L = 1 and hands each node’s gradient to its inputs, times the local derivative of its own operation. One backward pass gives the gradient of every weight, and a finite-difference check confirms it.',
    blocks: [
      'Training needs ∂L/∂w for every weight w. Nudging each weight in turn and re-running the network would cost one pass per weight. Can one pass give them all?',
      { topic: 'p09-backprop', frame: 0, caption: 'One tanh neuron and a linear output, as a graph of products, sums, a tanh and a square. Inputs 0.5 and −1, target 1; the five weights start at −0.2, 0.3, 1.2, 0.9 and −1.4.' },
      { topic: 'p09-backprop', frame: 9, caption: 'The forward pass, node by node, ends with y = −0.802 and the loss L = 3.249.' },
      { topic: 'p09-backprop', frame: 11, caption: 'Backward from the top: L = d² has local derivative 2d, so ∂L/∂d = −3.605.' },
      { table: [['Operation', 'Gradient sent to each input'], ['a + b', 'the gradient, unchanged, to both'], ['a · b', 'to a: b times the gradient; to b: a times the gradient'], ['f(a)', 'f′(a) times the gradient'], ['a²', '2a times the gradient']], caption: 'The local rules. Every node needs only its own inputs and the gradient that arrives from above.' },
      { predict: { q: 'y = m₃ + c, and ∂L/∂y = −3.605. What is ∂L/∂c?', options: ['−3.605: an addition passes the gradient on unchanged', '0: c is a constant', '−3.605 times c'], answer: 0, explain: '∂y/∂c = 1, so c receives the gradient that y holds.' } },
      { topic: 'p09-backprop', frame: 14, caption: 'At the product m₃ = v · h each input receives the other times the gradient: ∂L/∂v = 0.664 × −3.605 = −2.394, and ∂L/∂h = 0.900 × −3.605 = −3.244.' },
      { topic: 'p09-backprop', frame: 15, caption: 'Through the tanh: its local slope is 1 − h² = 0.559, so ∂L/∂u = −1.814.' },
      { topic: 'p09-backprop', frame: 20, caption: 'The check: nudging each weight by ±10⁻⁶ and measuring the loss gives the same five gradients; the worst relative difference is 1.5 × 10⁻¹⁰.' },
      { topic: 'p09-backprop', frame: 21, caption: 'One gradient step with η = 0.1: the bias c moves most, from −1.4 to −1.04, and the loss falls from 3.2485 to 1.1813.' },
      { topic: 'p09-backprop', frame: 15, input: { x1: 1, x2: 1, t: 1, act: 'relu', seed: 2, lr: 0.1 }, caption: 'A ReLU that is off for this input, u = −0.9: its local slope is zero, so ∂L/∂u = 0 and w₁, w₂ and b get no gradient at all, though the loss is 1.69. Only c will move.' },
      { h: 'Recap' },
      { list: ['Forward: every node’s value. Backward: every node’s gradient, from the loss down.', 'Each node passes back the gradient it holds times its local derivative.', 'One backward pass costs about as much as one forward pass, and gives every gradient; finite differences check them.'] },
      X('ne-x-bp-out'), X('ne-x-bp-graph'), X('ne-x-bp-relu'), X('ne-x-update'), X('ne-x-update-loss'), X('ne-x-code-dense'),
    ],
  });

  C.lesson({
    id: 'ne-surface', chapter: 'ne-learn', title: 'The loss surface of a tiny network', minutes: 7,
    summary: 'A network y = v · tanh(w · x) has only two weights, so its loss can be drawn whole. It has two equally good minima, mirror twins, because flipping the sign of both weights changes nothing; between them, at the origin, a flat saddle. Where gradient descent ends depends on where it starts.',
    blocks: [
      'For a linear model the loss is a single bowl. What does the loss of even the smallest network look like?',
      { topic: 'p09-loss-surface', frame: 0, caption: 'Every point of the left plane is a pair of weights (w, v), shaded by the loss on 13 points. The lowest point of the grid: w = −1.6, v = −1.17, loss 0.0004.' },
      { topic: 'p09-loss-surface', frame: 1, caption: 'Its mirror, w = 1.6, v = 1.17, has exactly the same loss: tanh is odd, so v · tanh(w x) = (−v) · tanh(−w x).' },
      { predict: { q: 'At the origin, w = v = 0, the gradient is exactly zero. If gradient descent starts there, what happens?', options: ['It never moves', 'It slides to the nearer twin', 'It jumps to the lowest point'], answer: 0, explain: 'Each step is −η times the gradient, and the gradient is zero: the walk stays put on the saddle.' } },
      { topic: 'p09-loss-surface', frame: 2, caption: 'The origin: loss 0.9958 and gradient (0, 0). Not a minimum but a saddle: downhill towards either twin.' },
      { topic: 'p09-loss-surface', frame: 9, caption: 'From w = 0.2, v = 0.1 gradient descent with rate 0.2 settles in the twin with positive weights, loss 0.0005.' },
      { topic: 'p09-loss-surface', frame: 9, input: { start: 'left' }, caption: 'Started from w = −2, v = −0.5 instead, it settles in the other twin, loss 0.0014. The start decides the twin.' },
      { topic: 'p09-loss-surface', frame: 9, input: { data: 'line', start: 'plateau' }, caption: 'Started with a large w on points along a line, the tanh neuron saturates: after 120 steps the loss is still 0.0582 and the gradient only 0.03. A plateau.' },
      { tip: 'A hidden layer of n neurons has many twins: swap any two neurons, or flip the signs of one tanh neuron’s weights in and out, and the function is the same. There is never one best set of weights.' },
      { h: 'Recap' },
      { list: ['Even a two-weight network has a loss with several minima.', 'Symmetries make equally good twins; between them sit saddles where the gradient is zero.', 'Saturated neurons make plateaus where descent crawls.'] },
      X('ne-x-twins'), X('ne-x-saddle'),
    ],
  });

  C.lesson({
    id: 'ne-init', chapter: 'ne-learn', title: 'Initialisation: why the first weights matter', minutes: 7,
    summary: 'Each layer multiplies the spread of its input by roughly √n times the spread of its weights (and by the activation’s gain). Too small and the signal fades layer by layer; too large and tanh saturates while the gradient grows on the way back. Xavier (variance 2 / (fan-in + fan-out)) suits tanh; He (variance 2 / fan-in) suits ReLU.',
    blocks: [
      'Before training, the weights are random numbers. Does it matter how large they are, as long as training fixes them?',
      { topic: 'p09-init', frame: 0, caption: 'Six dense layers of 64 tanh neurons and 128 random inputs of spread 1.01. The weights are drawn with standard deviation 0.049.' },
      { topic: 'p09-init', frame: 7, caption: 'Layer by layer the spread falls from 0.35 to 0.004: each layer passes on less than it receives.' },
      { topic: 'p09-init', frame: 8, caption: 'Backward it is no better: the gradient reaching the first layer has spread 1.1 × 10⁻⁴. The first layers would hardly learn.' },
      { predict: { q: 'Now the weights are drawn with standard deviation about 1, with tanh. What happens to the activations?', options: ['They pile up at ±1: the neurons saturate', 'They keep a healthy spread', 'They fade to zero'], answer: 0, explain: 'A sum of 64 inputs with weights of size 1 is large, so tanh is pushed into its flat ends: 82% of the values are beyond ±0.95.' } },
      { topic: 'p09-init', frame: 1, input: { scheme: 'large' }, caption: 'Too large: 82% of layer 1’s values are beyond ±0.95, where tanh is flat; and the gradient grows on the way back, to 19 at the first layer.' },
      { topic: 'p09-init', frame: 9, caption: 'The four ways with tanh (last layer’s spread, first layer’s gradient): std 0.05 gives 0.004 and 1.1 × 10⁻⁴; std 1 gives 0.95 and 19; Xavier 0.32 and 0.22; He 0.58 and 0.46.' },
      { math: 'Xavier:  Var(w) = 2 / (fan-in + fan-out)      for tanh and sigmoid\nHe:      Var(w) = 2 / fan-in                   for ReLU (it zeroes half its inputs)' },
      { topic: 'p09-init', frame: 9, input: { scheme: 'he', act: 'relu' }, caption: 'With ReLU, He keeps the last layer’s spread at 1.1 and the first layer’s gradient at 4.8; Xavier lets them sink to 0.14 and 0.11.' },
      { h: 'Recap' },
      { list: ['Too small: the signal and the gradient fade. Too large: saturation, and gradients that explode.', 'Xavier balances the variance forwards and backwards; use it for tanh and sigmoid.', 'He doubles the variance to make up for ReLU’s zeros; use it for ReLU.'] },
      X('ne-x-init-choice'), X('ne-x-zero-init'), X('ne-x-xavier-std'), X('ne-x-init-curve'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: fitting without memorising
  // =========================================================================================================
  C.lesson({
    id: 'ne-overfit', chapter: 'ne-reg', title: 'Overfitting: memorising the training set', minutes: 6,
    summary: 'A network with far more weights than training points can fit every point, noise and all. Its training loss falls towards zero while the loss on fresh points, after an early low, climbs. More data narrows the gap; so do the cures of the next lessons.',
    blocks: [
      'A network gets every training point right. Is it a good model?',
      { topic: 'p09-overfit', frame: 0, caption: '1218 weights, 30 training points (large dots), and 400 fresh points from the same moons held back to judge it.' },
      { topic: 'p09-overfit', frame: 1, caption: 'Step 20: the boundary has the shape of the moons; 84% of the fresh points are right.' },
      { predict: { q: 'Training goes on to step 500. What happens to the loss on the fresh points?', options: ['It climbs, while the training loss keeps falling', 'It keeps falling with the training loss', 'It stays where it was at step 20'], answer: 0, explain: 'The network bends its boundary round single noisy points: the fresh loss rises from 0.351 (step 10) to 1.409.' } },
      { topic: 'p09-overfit', frame: 3, caption: 'Step 120: every training point right, training loss 0.015, but the fresh loss is 0.846 and rising. The boundary curls round single points: it is learning the noise.' },
      { topic: 'p09-overfit', frame: 6, caption: 'After 500 steps: fresh loss 1.409 against 1.9 × 10⁻⁴ in training. The best moment for fresh points was step 10.' },
      { topic: 'p09-overfit', frame: 6, input: { size: 'big', n: '300' }, caption: 'The same network with 300 training points: the fresh loss ends at 0.551, its lowest 0.235. Ten times the data, a far smaller gap.' },
      { h: 'Recap' },
      { list: ['Training loss measures fit; the loss on fresh points measures what we want.', 'A large network on little data memorises: its fresh loss rises while its training loss falls.', 'More data helps; dropout, weight decay and early stopping help when there is no more.'] },
      X('ne-x-overfit-curve'), X('ne-x-gap-choice'),
    ],
  });

  C.lesson({
    id: 'ne-dropout', chapter: 'ne-reg', title: 'Dropout: training with neurons switched off', minutes: 6,
    summary: 'At every training step dropout switches off each hidden neuron with probability p and scales the survivors by 1 / (1 − p), so a layer’s output is unchanged on average. No neuron can rely on a particular partner. At test time nothing is switched off and nothing is scaled.',
    blocks: [
      'The overfitting network leans on particular neurons to carve round particular points. What if any neuron might be missing at any step?',
      { topic: 'p09-dropout', frame: 0, caption: 'Dropout with p = 0.5: each hidden neuron is off with probability 0.5, and every survivor’s output is multiplied by 2.' },
      { topic: 'p09-dropout', frame: 1, caption: 'Step 1: 7 neurons of the first hidden layer and 5 of the second are off. Only the survivors’ weights are updated.' },
      { topic: 'p09-dropout', frame: 3, caption: 'Step 3: another mask, another thinned network. Training is in effect an average over many of them.' },
      { predict: { q: 'At test time, what does dropout do?', options: ['Nothing: every neuron is on and nothing is scaled', 'It switches off half the neurons, as in training', 'It multiplies every output by 1 − p'], answer: 0, explain: 'Inverted dropout already scaled the survivors during training, so the full network’s outputs have the right size as they are.' } },
      { topic: 'p09-dropout', frame: 4, caption: 'At test time every neuron is on and nothing is scaled.' },
      { topic: 'p09-dropout', frame: 5, caption: 'The overfitting experiment again for 500 steps: without dropout the fresh loss ends at 1.409 with 83% right; with it, 0.880 and 89%.' },
      { h: 'Recap' },
      { list: ['Training: each neuron off with probability p, survivors scaled by 1 / (1 − p).', 'Test: no mask, no scaling.', 'It stops neurons from relying on each other, and smooths the boundary.'] },
      X('ne-x-dropout-test'), X('ne-x-dropout-scale'),
    ],
  });

  C.lesson({
    id: 'ne-decay-early', chapter: 'ne-reg', title: 'Weight decay and early stopping', minutes: 8,
    summary: 'Weight decay shrinks every weight by the factor 1 − ηλ at every step, so only weights the data keeps pulling stay large; small weights make a smooth function. Early stopping watches the loss on held-out data, keeps the best weights so far, and stops when it has not improved for a set number of checks.',
    blocks: [
      'The memorising network grows large weights to make sharp bends. Could we simply keep the weights small, or stop before the bends form?',
      { topic: 'p09-weight-decay', frame: 0, caption: 'Decay λ = 2 with η = 0.02: every weight is multiplied by 0.96 before its gradient step. Over a hundred steps with no push from the data, that alone shrinks a weight to 0.017 of its size.' },
      { math: 'decoupled weight decay (AdamW):  w ← w − η·λ·w − η · (the optimiser’s step)' },
      { topic: 'p09-weight-decay', frame: 1, caption: 'One real step on the first layer’s largest weight, −2.2542: the decay takes −0.0902 and the Adam step −0.0200, so it ends at −2.1441.' },
      { topic: 'p09-weight-decay', frame: 3, caption: 'After 500 steps, total weight size 22.18 without decay and 7.81 with it; fresh loss 1.409 against 0.315.' },
      { h: 'Early stopping' },
      { topic: 'p09-early-stop', frame: 0, caption: 'Every 10 steps, measure the loss on 400 held-out points. Keep a copy of the weights at every new lowest; stop after 5 checks in a row without one.' },
      { predict: { q: 'The held-out loss reaches its lowest, 0.351, at check 1 and never beats it. With patience 5, at which check does training stop?', options: ['Check 6', 'Check 5', 'Check 1'], answer: 0, explain: 'Checks 2 to 6 are five in a row without a new lowest, so it stops at check 6, step 60.' } },
      { topic: 'p09-early-stop', frame: 4, caption: 'Check 6, step 60: five checks without beating 0.351. Stop.' },
      { topic: 'p09-early-stop', frame: 5, caption: 'Stopped at step 60 instead of 500, saving 88% of the work, with the weights of step 10: held-out loss 0.351 against 1.409 at the end of the full run.' },
      { h: 'Recap' },
      { list: ['Weight decay multiplies each weight by 1 − ηλ every step; the data must keep a weight large.', 'Early stopping keeps the weights with the lowest held-out loss and stops after a patience of checks.', 'Both trade a little training fit for a smoother function that does better on fresh data.'] },
      X('ne-x-decay'), X('ne-x-early'), X('ne-x-wf-overfit'),
    ],
  });

  C.lesson({
    id: 'ne-norm', chapter: 'ne-reg', title: 'Batch normalisation and layer normalisation', minutes: 7,
    summary: 'Batch norm standardises each feature over the examples of a batch, then scales by γ and shifts by β; it depends on the other examples, so it keeps running statistics for test time and struggles with tiny batches. Layer norm standardises each example over its own features, and works the same with a batch of one.',
    blocks: [
      'One feature arrives in the tens, another below one. And as the earlier layers learn, the scale of what reaches a layer keeps changing. Could each layer simply re-scale its inputs?',
      { topic: 'p09-norm', frame: 1, caption: 'A batch of 4 examples and 3 features. Batch norm works column by column: f₁ has mean 57.5 and spread 8.789; f₂ has 1.25 and 0.622.' },
      { topic: 'p09-norm', frame: 2, caption: 'Each value has its column’s mean taken away and is divided by its spread: the first example’s f₁ becomes (52 − 57.5) / 8.789 = −0.626.' },
      { topic: 'p09-norm', frame: 4, caption: 'Layer norm works row by row: example x₁ is standardised over its own three features and becomes [1.41, −0.63, −0.78].' },
      { predict: { q: 'Only the last example of the batch is replaced. Which output of the first example changes?', options: ['Its batch norm output', 'Its layer norm output', 'Both', 'Neither'], answer: 0, explain: 'Batch norm’s column statistics include the replaced example; layer norm looks only at the first example’s own features.' } },
      { topic: 'p09-norm', frame: 5, caption: 'Its batch norm output moves from [−0.63, −0.72, −1.24] to [−0.35, −0.66, −1.37]; its layer norm output stays [1.41, −0.63, −0.78].' },
      { topic: 'p09-norm', frame: 6, caption: '10 tanh layers with weights drawn too small: without normalisation the spread falls from 0.3 to 2.2 × 10⁻⁵; with batch norm before each tanh it ends at 0.64.' },
      { table: [['', 'Batch norm', 'Layer norm'], ['statistics over', 'the batch, per feature', 'the features, per example'], ['batch of one', 'meaningless', 'works'], ['at test time', 'running averages from training', 'the same as in training'], ['usual home', 'convolutional networks', 'transformers, recurrent networks']] },
      { h: 'Recap' },
      { list: ['Both standardise, then scale by a learned γ and shift by a learned β.', 'Batch norm needs reasonably large batches and running statistics at test time.', 'Layer norm is independent of the batch, so it suits small batches and sequences.'] },
      X('ne-x-norm-choice'), X('ne-x-bn'),
    ],
  });

  C.lesson({
    id: 'ne-play', chapter: 'ne-reg', title: 'The playground: every neuron’s region', minutes: 7,
    summary: 'Each first-layer neuron is a soft step across a line in the plane; a second layer mixes those steps into curved shapes; the output adds them up. Too few neurons cannot draw the boundary a dataset needs; a second layer gets far more shape from the same number of neurons.',
    blocks: [
      'How many neurons does a ring need? A spiral? Draw every neuron over the plane, and the answer can be seen.',
      { topic: 'p09-playground', frame: 0, caption: 'One hidden layer of 4 tanh neurons on a ring round a disc. Each square is one hidden neuron over the plane: blue positive, vermilion negative.' },
      { topic: 'p09-playground', frame: 3, caption: 'Step 133: loss 0.049 and every point right. Four soft steps across four lines, added up, close round the disc.' },
      { predict: { q: 'Now only 2 hidden neurons. Can one layer of 2 tanh neurons close a ring?', options: ['No: two lines cannot enclose the disc', 'Yes, with enough steps', 'Yes, if the learning rate is larger'], answer: 0, explain: 'With one hidden layer the output adds up regions bounded by straight lines; two lines leave the disc open on two sides.' } },
      { topic: 'p09-playground', frame: 6, input: { neurons: 2 }, caption: 'Two neurons: after 400 steps only 81% are right, loss 0.421.' },
      { topic: 'p09-playground', frame: 6, input: { data: 'spirals', layers: '2', neurons: 6, act: 'relu', lr: 0.03, steps: 800, seed: 1 }, caption: 'Spirals with two layers of 6 ReLU neurons, 74 weights: after 800 steps every point is right. The second layer’s squares are shapes made of the first layer’s shapes.' },
      { playground: { data: 'spirals', layers: [6, 6], act: 'tanh', lr: 0.03 }, title: 'Spirals in the playground', text: 'Start with two layers of six neurons. Take neurons away one at a time: how few can still wind round both arms?' },
      { h: 'Recap' },
      { list: ['A first-layer neuron is a soft step across a line; later layers combine steps into shapes.', 'One hidden layer adds up straight-edged regions; a ring needs at least three.', 'Depth buys shape: two layers of a few neurons can follow a spiral.'] },
      { challenge: 'ne-ch-spiral' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
