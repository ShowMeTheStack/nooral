// Part 10, convolutional networks: the chapters and the lessons. The exercises are in 101-cnn-ex.js; the model
// challenge, the tour, the course and the pages the diagram checker looks at are in 102-cnn-ch.js. Each lesson opens
// with a question, runs the topic that shows the idea at a frame chosen for the point being made, asks for a
// prediction, sums up, and leads to its exercises. Every value quoted here was printed by the topics themselves (the
// engine's convolution, pooling and autodiff, S.T, and the small networks trained on generated digits and shapes).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('cn-slide')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'cn-conv', part: 'cnn', n: 1, title: 'The convolution', blurb: 'A small kernel sliding over a picture with its dot products; padding, stride and dilation and the sizes they give; pooling; and filters that read several channels at once.' });
  C.chapter({ id: 'cn-inside', part: 'cnn', n: 2, title: 'Inside a trained network', blurb: 'An image flowing through a trained network map by map, the filters it learned drawn as pictures, and how far each unit can see.' });
  C.chapter({ id: 'cn-train', part: 'cnn', n: 3, title: 'Training, and asking why', blurb: 'Training on generated digits and shapes with the confusion matrix filling in, augmentation for small sets, residual shortcuts for deep stacks, and saliency maps and Grad-CAM to ask where the evidence lies.' });

  // =========================================================================================================
  // Chapter 1: the convolution
  // =========================================================================================================
  C.lesson({
    id: 'cn-slide', chapter: 'cn-conv', title: 'Convolution: a kernel sliding over a picture', minutes: 7,
    summary: 'A convolution slides a small grid of weights, the kernel, over the picture. At each place it multiplies the kernel by the pixels under it and adds the products (plus a bias). The same few weights are used at every place, so a pattern is found wherever it is, at a fraction of a dense layer’s cost.',
    blocks: [
      'A dense layer looking at a 16 × 16 image needs a weight for every pixel, for every neuron, and a stroke learned in one corner means nothing in another. Could one small detector be moved over the whole picture instead?',
      { topic: 'conv-slide', frame: 0, caption: 'A 6 × 6 picture of a corner (1 is ink) and a 3 × 3 kernel that answers to vertical edges: −1 on the left column, +1 on the right. It fits 4 places across and 4 down: a 4 × 4 output.' },
      { math: 'y[i, j] = b + Σᵤ Σᵥ x[i + u, j + v] · w[u, v]       u, v = 0, 1, 2' },
      { topic: 'conv-slide', frame: 2, caption: 'Row 1, column 2: three products are not zero, 1 × −1 + 1 × 1 + 1 × −1, so the output is −1.' },
      { predict: { q: 'Where will the output be most negative?', options: ['Where ink lies under the kernel’s −1 column and paper under its +1 column', 'Where the picture is blank', 'Where ink fills the whole window'], answer: 0, explain: 'Ink on the left and paper on the right is an edge from ink to paper; the corner’s upright stroke gives −3 at row 3, column 2.' } },
      { topic: 'conv-slide', frame: 10, caption: 'Row 3, column 2: the upright stroke lies under the kernel’s left column: 1 × −1 three times, −3.' },
      { topic: 'conv-slide', frame: 17, caption: 'The whole map: largest 1, smallest −3, and 6 of the 16 values negative. The same 9 weights were used at all 16 places.' },
      { topic: 'conv-slide', frame: 26, input: { picture: 'plus', kernel: 'blur' }, caption: 'Another kernel, another job: every weight 1/9 averages each 3 × 3 window, blurring a plus sign. The largest value is 0.556, five ink pixels out of nine.' },
      { h: 'Recap' },
      { list: ['A convolution slides one kernel over the picture and takes a dot product at each place.', 'The output is large where the picture looks like the kernel.', 'Sharing the kernel’s weights across places finds a pattern anywhere, with few weights.'] },
      X('cn-x-conv-cell'), X('cn-x-conv-map'), X('cn-x-conv-relu'), X('cn-x-code-conv'),
    ],
  });

  C.lesson({
    id: 'cn-geometry', chapter: 'cn-conv', title: 'Padding, stride and dilation', minutes: 7,
    summary: 'Without padding, each convolution shrinks the map by k − 1. Padding with p rings of zeros keeps the size when p = (k − 1)/2. A stride s skips places and divides the size by about s. Dilation d spreads the kernel’s taps d apart, so it spans d(k − 1) + 1 pixels with the same weights. One formula gives the output size.',
    blocks: [
      'A 3 × 3 kernel on a 6 × 6 picture fits only 4 times each way. After a few layers nothing would be left. How do networks control the size of their maps?',
      { math: 'out = ⌊(n + 2p − d(k − 1) − 1) / s⌋ + 1' },
      { topic: 'conv-geometry', frame: 0, caption: 'A 6 × 6 picture with one ring of zeros, a 3 × 3 kernel, stride 1: the padded picture is 8 wide, so the output is 6 × 6, the same size: “same” padding.' },
      { topic: 'conv-geometry', frame: 1, caption: 'The first place: 5 of the 9 taps fall on padding zeros. Corners are read less often than the middle.' },
      { topic: 'conv-geometry', frame: 8, caption: 'Every place visited: the busiest pixel is read 9 times, the top-left corner 4.' },
      { predict: { q: 'An 8 × 8 picture, one ring of padding, a 3 × 3 kernel and stride 2. How large is the output?', options: ['4 × 4', '8 × 8', '3 × 3'], answer: 0, explain: '(8 + 2 − 3) / 2 = 3.5, rounded down 3, plus one: 4.' } },
      { topic: 'conv-geometry', frame: 0, input: { n: 8, k: 3, p: 1, s: 2, d: 1 }, caption: 'Stride 2: the window jumps 2 pixels, and the output is 4 × 4, half the size each way.' },
      { topic: 'conv-geometry', frame: 0, input: { n: 8, k: 3, p: 0, s: 1, d: 2 }, caption: 'Dilation 2: the 9 taps are 2 apart, so the kernel spans 5 pixels with 9 weights. With no padding an 8 × 8 picture gives a 4 × 4 output.' },
      { h: 'Recap' },
      { list: ['out = ⌊(n + 2p − d(k − 1) − 1) / s⌋ + 1 each way.', '“Same” padding, p = (k − 1)/2 at stride 1, keeps the size.', 'Stride shrinks the map; dilation widens what a kernel sees without more weights.'] },
      X('cn-x-size-valid'), X('cn-x-size-table'), X('cn-x-same-pad'), X('cn-x-span'),
    ],
  });

  C.lesson({
    id: 'cn-pool', chapter: 'cn-conv', title: 'Pooling: shrinking a map', minutes: 6,
    summary: 'Pooling slides a window over a map and keeps one number per window: the largest (max pooling) or the mean (average pooling). It has no weights. A 2 × 2 window moving 2 at a time halves each side and keeps a strong response even if it moves a little within its window.',
    blocks: [
      'After a convolution and ReLU, a map says where each pattern was found. Do we need to know where to the exact pixel?',
      { topic: 'conv-pool', frame: 0, caption: 'A 6 × 6 map after ReLU. A 2 × 2 window moving 2 at a time fits 3 times each way: 36 numbers become 9.' },
      { topic: 'conv-pool', frame: 1, caption: 'The first window holds 0, 1, 3 and 0: only the 3 goes on.' },
      { predict: { q: 'The whole map shifts one pixel to the right. Do the pooled values change?', options: ['Some stay exactly the same', 'All of them change', 'None of them change'], answer: 0, explain: 'A largest value that stays inside its window is passed on unchanged; one that crosses into the next window moves the result.' } },
      { topic: 'conv-pool', frame: 10, caption: 'After the shift, 4 of the 9 pooled values are exactly as before: a little tolerance to position.' },
      { topic: 'conv-pool', frame: 0, input: { map: 'digit' }, caption: 'A real map: channel 5 of the trained network’s first layer for a 7. Pooling turns its middle 64 numbers into 16.' },
      { topic: 'conv-pool', frame: 0, input: { kind: 'max', size: 3, stride: 1 }, caption: 'A 3 × 3 window moving 1 at a time overlaps itself: the 6 × 6 map gives 4 × 4.' },
      { h: 'Recap' },
      { list: ['Max pooling keeps the largest value of each window; average pooling the mean.', 'Pooling has no weights; its output size follows the same formula as a convolution’s.', 'It shrinks the maps and buys a little tolerance to small shifts.'] },
      X('cn-x-pool-map'), X('cn-x-pool-overlap'), X('cn-x-pool-grad'), X('cn-x-code-pool'),
    ],
  });

  C.lesson({
    id: 'cn-channels', chapter: 'cn-conv', title: 'Many channels: one filter, many kernels', minutes: 7,
    summary: 'A colour picture is three grids, one per channel. A filter has one kernel per input channel; it convolves each channel with its own kernel and adds the results (and one bias) into a single map. A layer of O filters over C channels with k × k kernels has O · (C · k · k + 1) weights and biases, and makes O maps for the next layer.',
    blocks: [
      'A grey picture is one grid of numbers. A colour picture is three, and after the first layer there may be dozens of maps. What does a filter do with more than one grid?',
      { topic: 'conv-channels', frame: 0, caption: 'Red on the left, blue on the right: three 5 × 5 grids. The filter has a 3 × 3 kernel for each: 27 weights and one bias, making one 3 × 3 map.' },
      { topic: 'conv-channels', frame: 1, caption: 'At the first place the red window times the red kernel adds up to 3.' },
      { topic: 'conv-channels', frame: 2, caption: 'The green kernel is all zeros: this filter ignores green.' },
      { predict: { q: 'The red part gives 3, the green 0 and the blue 3. What is the output at this place (the bias is 0)?', options: ['6: the channels’ sums are added', '3: the largest of them', 'Three numbers, one per channel'], answer: 0, explain: 'A filter adds its channels together into one number per place: 3 + 0 + 3 = 6.' } },
      { topic: 'conv-channels', frame: 4, caption: '3 + 0 + 3 = 6: one number of the output map.' },
      { topic: 'conv-channels', frame: 8, input: { picture: 'dot', filter: 'yellow' }, caption: 'A filter for yellow on blue: red and green kernels add, the blue kernel subtracts, and a bias of −2. At the dot’s centre red 5, green 5, blue −4: together 4.' },
      { topic: 'conv-channels', frame: 13, caption: 'A layer of 16 such filters would have 16 · (27 + 1) = 448 weights and biases, and make 16 maps: the next layer’s filters each read 16 channels.' },
      { h: 'Recap' },
      { list: ['A filter has one kernel per input channel and adds their results into one map.', 'A layer of O filters makes O maps: the channels of the next layer.', 'Weights and biases: O · (C · k · k + 1).'] },
      X('cn-x-channels'), X('cn-x-params-conv'), X('cn-x-params-dense'), X('cn-x-share-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: inside a trained network
  // =========================================================================================================
  C.lesson({
    id: 'cn-flow', chapter: 'cn-inside', title: 'Feature maps: an image flowing through', minutes: 8,
    summary: 'A small trained network reads a 16 × 16 digit: six 5 × 5 filters, ReLU, max pooling, twelve more filters, ReLU and pooling again, then a dense layer of 32 and the scores. Layer by layer the maps get smaller and more numerous, and each value depends on more of the picture.',
    blocks: [
      'We know each piece. What does the whole stack do to one picture, map by map?',
      { topic: 'cnn-flow', frame: 0, caption: 'A 7 the network never trained on: 16 × 16 grey levels, 256 numbers.' },
      { topic: 'cnn-flow', frame: 1, caption: 'Six 5 × 5 filters with same padding give six 16 × 16 maps, values from −1.05 to 4.39.' },
      { topic: 'cnn-flow', frame: 2, caption: 'ReLU zeroes 528 of the 1536 values: what is left says where each filter’s pattern is present.' },
      { topic: 'cnn-flow', frame: 4, caption: 'Twelve filters over the six pooled maps, no padding: twelve 4 × 4 maps. Each value depends on a 14 × 14 patch, most of the picture.' },
      { predict: { q: 'After the second pooling there are twelve 2 × 2 maps. How many numbers reach the dense layer?', options: ['48', '12', '256'], answer: 0, explain: 'Twelve maps of four values each, laid out in one list: 48.' } },
      { topic: 'cnn-flow', frame: 6, caption: 'Twelve 2 × 2 maps, 48 numbers, laid out in one list.' },
      { topic: 'cnn-flow', frame: 8, caption: 'The scores and softmax: 7 at 100%. Right.' },
      { topic: 'cnn-flow', frame: 8, input: { cls: 5, k: 8 }, caption: 'A 5 it gets wrong: 8 at 88%, then 5 at 7%.' },
      { workflow: 'cnn-digits', title: 'The same kind of network on the canvas', text: 'Train it on 300 drawn digits, then open the CNN block: the filters, a probe image’s feature maps at every layer, and the curves.' },
      { h: 'Recap' },
      { list: ['Convolution, ReLU, pooling, repeated: maps get smaller and more numerous.', 'Deeper values depend on larger patches of the input.', 'The last maps are flattened into a list for the dense layers and the scores.'] },
      X('cn-x-flow-shapes'), X('cn-x-params-table'),
    ],
  });

  C.lesson({
    id: 'cn-filters', chapter: 'cn-inside', title: 'Learned filters, drawn as pictures', minutes: 6,
    summary: 'A first-layer filter is a tiny picture: blue weights add the pixel, vermilion ones take it away. Before training the filters are speckle; as training goes on they turn into smooth patches, stroke detectors and edge detectors, whose maps light up along the strokes. Deeper filters combine maps, not pixels, and are harder to read.',
    blocks: [
      'Nobody told the network to look for strokes or edges. What do its first filters end up looking for?',
      { topic: 'cnn-filters', frame: 0, caption: 'Step 0: six random filters, speckled, and their speckled maps on a 4.' },
      { topic: 'cnn-filters', frame: 3, caption: 'Step 600: smooth patches. Four filters are mostly blue, adding up ink; two set blue against vermilion, the mark of an edge detector.' },
      { predict: { q: 'A 5 × 5 filter has −1 in its left two columns and +1 in its right two. What does it answer to?', options: ['An upright edge, paper on the left and ink on the right', 'A level stroke', 'A blank area'], answer: 0, explain: 'It takes away the left and adds the right: largest where the right is inked and the left is not.' } },
      { topic: 'cnn-filters', frame: 6, caption: 'Filter 3 changes mostly from top to bottom, so it answers to level strokes and edges; on this 4 it is strongest at row 10, column 9.' },
      { topic: 'cnn-filters', frame: 10, caption: 'The second layer: 12 filters, each with a 5 × 5 kernel for every one of the 6 first-layer maps, 1800 weights. They combine maps, not pixels.' },
      { h: 'Recap' },
      { list: ['First-layer filters can be drawn as pictures: blue adds, vermilion takes away.', 'Training turns random speckle into stroke and edge detectors.', 'Deeper filters combine maps and are read through what they respond to.'] },
      X('cn-x-filter-read'),
    ],
  });

  C.lesson({
    id: 'cn-field', chapter: 'cn-inside', title: 'Receptive fields: how far a unit can see', minutes: 7,
    summary: 'The receptive field of a unit is the patch of the input it depends on. Each k × k layer adds (k − 1) · d · jump to it, where the jump is the product of the strides so far; pooling and strides multiply the jump, so later layers grow the field faster. The engine confirms each field: the unit’s gradient is non-zero across exactly that patch.',
    blocks: [
      'A 3 × 3 kernel sees 9 pixels. Yet a network recognises a whole digit. How much of the picture does a unit deep in the network see?',
      { math: 'r₀ = 1, j₀ = 1\nrₗ = rₗ₋₁ + (kₗ − 1) · dₗ · jₗ₋₁\njₗ = jₗ₋₁ · sₗ' },
      { topic: 'cnn-field', frame: 4, input: { arch: 'three' }, caption: 'Three 3 × 3 convolutions: 3, then 5, then 7. Each adds 2. The middle pixel’s gradient is 49 times the corner’s: the edges of a field count least.' },
      { topic: 'cnn-field', frame: 2, caption: 'With pooling between: after a 3 × 3 convolution (field 3), pooling by two adds one step and doubles the jump, field 4, jump 2.' },
      { predict: { q: 'The next 3 × 3 convolution comes after the jump has doubled to 2. How much does it add to the field of 4?', options: ['4', '2', '8'], answer: 0, explain: '(k − 1) · d · jump = 2 · 1 · 2 = 4, so the field becomes 8.' } },
      { topic: 'cnn-field', frame: 3, caption: 'Field 8: the engine finds the gradient non-zero across exactly 8 × 8 pixels.' },
      { topic: 'cnn-field', frame: 5, caption: 'After the last 3 × 3 convolution: 18 × 18, with only three small kernels.' },
      { topic: 'cnn-field', frame: 3, input: { arch: 'dilated' }, caption: 'Dilations 1, 2 and 4 grow the field to 15 without any pooling or more weights.' },
      { h: 'Recap' },
      { list: ['A layer adds (k − 1) · d · jump to the field; strides and pooling multiply the jump.', 'Pooling or striding early makes later layers see much more.', 'The middle of a field counts most: the effective field is smaller than its edges.'] },
      X('cn-x-field'), X('cn-x-field-table'), X('cn-x-field-dilated'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: training, and asking why
  // =========================================================================================================
  C.lesson({
    id: 'cn-training', chapter: 'cn-train', title: 'Training on digits, with the confusion matrix', minutes: 7,
    summary: 'A small CNN learns generated digits in a few hundred steps. The confusion matrix on validation images shows it: predictions piled in a few columns at first, then the diagonal filling in. The cells that stay off the diagonal name the confusions left, and each holds the images to look at.',
    blocks: [
      'Accuracy says how many images are right. Which ones are wrong, and wrong as what?',
      { topic: 'cnn-train', frame: 0, caption: 'Before training: loss 2.78 and 18 of 300 validation images right, 6%, about what guessing among 10 classes gives.' },
      { topic: 'cnn-train', frame: 2, caption: 'Step 100, 3200 images seen: 281 of the 300 right, 94%. The diagonal fills in.' },
      { predict: { q: 'Training accuracy ends near 100%. If validation accuracy also ends near 100%, what does that say?', options: ['It learned the classes, not the particular images', 'It memorised the training set', 'The validation set is too easy to mean anything'], answer: 0, explain: 'A memorising network would do much worse on images it has not seen; close numbers mean what it learned carries over.' } },
      { topic: 'cnn-train', frame: 4, caption: 'Step 600: 299 of 300 right; the one mistake is a 5 taken for an 8. Training 100%, validation 99.7%.' },
      { topic: 'cnn-train', frame: 4, input: { set: 'shapes' }, caption: 'Shapes: 96% on both. The largest cell off the diagonal says 4 squares were taken for triangles.' },
      { h: 'Recap' },
      { list: ['Row: the true class; column: the prediction. The diagonal is right.', 'The largest cell off the diagonal is the confusion to look at first.', 'Training and validation accuracy close together: learned, not memorised.'] },
      X('cn-x-cm-recall'), X('cn-x-cm-worst'),
    ],
  });

  C.lesson({
    id: 'cn-augment', chapter: 'cn-train', title: 'Data augmentation: more images from the same ones', minutes: 6,
    summary: 'With only 40 training images a CNN memorises them. Augmentation trains on a fresh random copy of each image every time: shifted, turned and resized a little. The network can no longer memorise exact pixels, and it learns what stays the same; but changes that alter the class, like turning a 6 into a 9, must be left out.',
    blocks: [
      'A digit moved one pixel to the left is the same digit. Can we tell the network that, without collecting more images?',
      { topic: 'cnn-augment', frame: 1, caption: 'Eight copies of a 2, each shifted up to 2 pixels, turned up to 17° and resized up to 10%. Each is still clearly a 2.' },
      { topic: 'cnn-augment', frame: 2, caption: 'Step 100, 40 training images: the plain run already knows all of them (74% of new images right); the augmented run is still learning (88% and 68%).' },
      { predict: { q: 'After 600 steps, which run does better on new images?', options: ['The augmented run', 'The plain run', 'They end the same'], answer: 0, explain: 'The plain run ends at 78% on new images, the augmented run at 97%.' } },
      { topic: 'cnn-augment', frame: 6, caption: 'Both get every training image right; on new images the plain run is right 78% of the time, the augmented one 97%.' },
      { topic: 'cnn-augment', frame: 1, input: { cls: 6, shift: 4, rotate: 45, scale: 30, seed: 3 }, caption: 'Too much: shifts of 4 pixels and turns of 45° push copies off the picture or towards other digits. A 6 turned far enough is a 9.' },
      { h: 'Recap' },
      { list: ['Augmentation trains on random variations of each image.', 'It says what should not matter, and fights memorising on small sets.', 'Only use changes that keep the class: no 180° turns for digits.'] },
      X('cn-x-augment-choice'), X('cn-x-wf-few'),
    ],
  });

  C.lesson({
    id: 'cn-residual', chapter: 'cn-train', title: 'Residual connections: a shortcut for the gradient', minutes: 6,
    summary: 'In a deep plain stack the gradient is a product of one factor per layer, so it fades or explodes. A residual block outputs h + f(h): its derivative is 1 + f′, so the identity carries the gradient straight through every block whatever f does.',
    blocks: [
      'Initialisation and normalisation help, but past a few dozen layers plain networks still train badly. Is there a way for the gradient to skip the layers?',
      { topic: 'cnn-residual', frame: 0, caption: '20 layers, each a 3 × 3 convolution over 4 channels, with small random weights. The plain and the residual stack use exactly the same numbers.' },
      { topic: 'cnn-residual', frame: 2, caption: 'Plain, backward: the gradient at the input is 3.74 × 10⁻⁷, against 1.04 at the last layer. It has vanished.' },
      { math: 'plain:     hₗ = f(hₗ₋₁)            ∂hₗ/∂hₗ₋₁ = f′\nresidual:  hₗ = hₗ₋₁ + f(hₗ₋₁)    ∂hₗ/∂hₗ₋₁ = 1 + f′' },
      { predict: { q: 'With the same weights, roughly how large is the residual stack’s gradient at the input?', options: ['About the size of the gradient at the top', 'Far smaller still', 'Exactly zero'], answer: 0, explain: 'Each factor is 1 + f′ with small f′: it goes from 1.04 at the top to 1.28 at the input.' } },
      { topic: 'cnn-residual', frame: 4, caption: 'Residual, backward: 1.04 at the last layer, 1.28 at the input.' },
      { topic: 'cnn-residual', frame: 5, input: { depth: 30, init: 'large', seed: 1 }, caption: 'With large weights the plain stack’s gradient explodes to 2720 at the input; the residual stack’s ends at 6.25: 0.0023 times the plain one.' },
      { h: 'Recap' },
      { list: ['A plain stack multiplies one factor per layer: the gradient fades or explodes.', 'A residual block adds its input back: its factor is 1 + f′.', 'Shortcuts let networks of a hundred layers and more train.'] },
      X('cn-x-residual'),
    ],
  });

  C.lesson({
    id: 'cn-explain', chapter: 'cn-train', title: 'Saliency maps and Grad-CAM', minutes: 8,
    summary: 'A saliency map backpropagates a class score to the input: its size at each pixel says how much that pixel moves the score. Grad-CAM weights the last convolution’s maps by the average gradient of the score over each, adds them and keeps the positive part: a coarse map of where the evidence lies. Blanking the pixels a map points at, and watching the score fall, tests it.',
    blocks: [
      'The network says 3. Did it look at the strokes of the 3, or at something else in the picture? Can we ask it where its evidence lies?',
      { topic: 'cnn-saliency', frame: 1, caption: 'Backpropagating the score of class 3 to the input: one number per pixel. The largest, 2.94 per unit of ink, is at row 7, column 8.' },
      { topic: 'cnn-saliency', frame: 2, caption: 'The sizes, ignoring signs, laid over the image; the 12 ink pixels with the largest positive gradient are outlined.' },
      { predict: { q: 'The 12 outlined pixels are blanked. Compared with blanking 12 random ink pixels, what happens to the score of 3?', options: ['It falls much further', 'About the same', 'It rises'], answer: 0, explain: 'From 3.49 it falls to −4.12, against about −0.09 for random pixels: the map pointed at what matters.' } },
      { topic: 'cnn-saliency', frame: 3, caption: 'Blanking the marked pixels: 3.49 to −4.12. Blanking 12 random ink pixels: about −0.09.' },
      { h: 'Grad-CAM' },
      { math: 'αₖ = mean over map k of ∂s_c / ∂Aᵏ\nCAM = max(0, Σₖ αₖ Aᵏ), scaled to 0…1' },
      { topic: 'cnn-gradcam', frame: 1, caption: 'An 8: the average gradient over each of the 12 last maps gives its weight. Map 12 counts most; 8 of the 12 weights are negative.' },
      { topic: 'cnn-gradcam', frame: 3, caption: 'The weighted sum, cut at zero and stretched over the image: the evidence for 8 lies lower right. Each cell covers about 4 × 4 pixels: it says where, not which pixel.' },
      { topic: 'cnn-gradcam', frame: 3, input: { cls: 7, k: 1 }, caption: 'A 7: only 11% of the map’s weight falls on ink. For this network, where the strokes are not is as telling as where they are.' },
      { tip: 'These maps describe what a model leans on; they are a check, not a proof. A map that points at the background or a watermark is a warning that the model learned a shortcut, which is worth fixing in the data.' },
      { h: 'Recap' },
      { list: ['Saliency: |∂score/∂pixel|, one backward pass, pixel-sharp but noisy.', 'Grad-CAM: the last maps weighted by their mean gradients, positive part: coarse but readable.', 'Test a map by blanking what it points at and watching the score.'] },
      X('cn-x-saliency-choice'), X('cn-x-gradcam'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
