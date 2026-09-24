// Part 6, support vector machines: the chapters and the lessons. The exercises are in 061-svm-ex.js; the model
// challenges, the tour, the course and the pages the diagram checker looks at are in 062-svm-ch.js. Each lesson opens
// with a question, runs the topic that shows the idea at a frame chosen for the point being made, asks for a
// prediction, sums up, and leads to its exercises. Every value quoted here was printed by the topics themselves (the
// SMO solver of S.ML on the topics' own inputs).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('sv-street')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'sv-margin', part: 'svm', n: 1, title: 'The widest street', blurb: 'A line with the most room on both sides, the few points that hold it in place, the price C of letting points into the street, and the hinge loss that the price is paid in.' });
  C.chapter({ id: 'sv-kernels', part: 'svm', n: 2, title: 'Kernels', blurb: 'Lift the points into more dimensions and a straight cut becomes a curve; a kernel computes the lifted dot product without ever lifting. Polynomial and RBF kernels, and γ against overfitting.' });
  C.chapter({ id: 'sv-solver', part: 'svm', n: 3, title: 'Inside the solver', blurb: 'The weights as a sum over support vectors, the multipliers α that the dual problem asks for, and SMO, which finds them two at a time.' });
  C.chapter({ id: 'sv-more', part: 'svm', n: 4, title: 'More classes, numbers and one class', blurb: 'One-vs-rest and one-vs-one for many classes, support vector regression with its ε-tube, and the one-class SVM that draws a border round normal data.' });

  // =========================================================================================================
  // Chapter 1: the widest street
  // =========================================================================================================
  C.lesson({
    id: 'sv-street', chapter: 'sv-margin', title: 'Which line would you trust?', minutes: 8,
    summary: 'When two groups can be split by a straight line, many lines do it. The support vector machine picks the one with the widest empty street around it. Only the points on the edges of the street, the support vectors, decide where it goes.',
    blocks: [
      'A courier firm sorts parcels into two vans by their weight and size. Last week’s parcels split cleanly: there is a gap between the two piles. Any line through the gap sorts last week perfectly. Which one should sort next week’s parcels?',
      { topic: 'svm-max-margin', frame: 0, caption: '24 points in two classes. A straight line can split them, and so can many others: tilt it a little or slide it and it still works.' },
      { topic: 'svm-max-margin', frame: 3, caption: 'Line C, one of the candidates: the widest empty street along its direction is 1.557 wide. Lines A and B leave only 0.964 and 0.981: a new point near either of them could easily land on the wrong side.' },
      { predict: { q: 'The SVM tries every direction and every offset. What does it maximise?', options: ['The number of points it gets right', 'The width of the empty street around the line', 'The distance from the line to the mean of each class', 'The number of support vectors'], answer: 1, explain: 'Every candidate already gets every point right. The SVM keeps the line with the most room on both sides, so that small changes in new points do not flip their class.' } },
      { topic: 'svm-max-margin', frame: 4, caption: 'The widest street: 1.807, wider than any line tried. Its edges touch 3 points, 1 blue and 2 orange. These are the support vectors, ringed.' },
      { def: 'Margin', text: 'The line is f(x) = w·x + b = 0 and the edges of the street are f = +1 and f = −1. The width of the street is 2 / ‖w‖, so a shorter w means a wider street. The SVM finds the shortest w that keeps every point at |f| ≥ 1 on its own side.' },
      { topic: 'svm-max-margin', frame: 5, caption: 'Here w = (1.04, 0.37) and b = −0.13, so ‖w‖ = 1.107 and the width is 2 / 1.107 = 1.807.' },
      { math: 'minimise  ½‖w‖²\nsubject to  yᵢ (w·xᵢ + b) ≥ 1  for every point i   (yᵢ = +1 or −1)' },
      { h: 'Only the support vectors matter' },
      { topic: 'svm-max-margin', frame: 6, caption: 'Delete the faded point, deep inside its own side: fit again and nothing moves. A point off the street has no say.' },
      { topic: 'svm-max-margin', frame: 7, caption: 'Delete a support vector instead: the street was resting on it, so the line turns and the street widens to 1.884.' },
      { tip: 'Because the answer depends only on the points at the edge, the SVM is calm about points far from the boundary. It is not calm about a mislabelled point near it, which is why the next lesson softens the margin.' },
      { h: 'Recap' },
      { list: ['Of all the lines that split the classes, the SVM keeps the one with the widest empty street.', 'The street’s edges are f = ±1, and its width is 2 / ‖w‖.', 'The support vectors are the points on the edges; removing any other point changes nothing.'] },
      X('sv-x-width'), X('sv-x-sv-set'), X('sv-x-remove'),
    ],
  });

  C.lesson({
    id: 'sv-soft', chapter: 'sv-margin', title: 'The soft margin and the price C', minutes: 8,
    summary: 'Real classes overlap, and then no street is empty. The soft margin lets points into the street or across it, each paying slack ξ, and C sets the price of a unit of slack: a small C buys a wide street with many points inside, a large C a narrow one with few.',
    blocks: [
      'A lender’s past applicants do not split cleanly into those who repaid and those who did not: some careful-looking ones defaulted. No empty street exists. Should the line still try to keep every point out of the street?',
      { topic: 'svm-soft-c', frame: 0, caption: '40 points whose classes overlap: no straight line puts every blue point on one side and every orange on the other. The hard margin of the last lesson has no answer here.' },
      { def: 'Slack', text: 'ξᵢ = max(0, 1 − yᵢ f(xᵢ)): how far point i falls short of its own edge of the street. 0 outside the street, between 0 and 1 inside it, more than 1 on the wrong side.' },
      { math: 'minimise  ½‖w‖² + C · Σ ξᵢ' },
      { topic: 'svm-soft-c', frame: 2, caption: 'C = 0.1: slack is cheap. The street is 1.994 wide, with 17 points inside it and 4 on the wrong side, and 24 of the 40 points are support vectors. (At C = 0.01 it is 5.541 wide and every point is one.)' },
      { predict: { q: 'C goes from 0.01 up to 100. What happens to the street?', options: ['It widens', 'It narrows, and fewer points sit inside it', 'It stays the same: C only changes the speed of training', 'It vanishes: no line is found'], answer: 1, explain: 'A larger C makes each unit of slack dearer, so the line accepts a shorter margin to keep points out of the street.' } },
      { topic: 'svm-soft-c', frame: 3, caption: 'C = 1: the street is 0.864 wide, 11 points inside it and 3 on the wrong side; their slacks, the dashed lines, add up to 8.65. Training accuracy 93%.' },
      { topic: 'svm-soft-c', frame: 5, caption: 'C = 100: the street has narrowed to 0.298 round the overlap, and only 8 support vectors remain. Training accuracy 95%: a little better on these points, but the line now bends to a handful of them.' },
      { table: [['C', 'Width', 'Σξ', 'Support vectors', 'Training accuracy'], ['0.01', '5.541', '26.97', '40', '90%'], ['0.1', '1.994', '13.24', '24', '90%'], ['1', '0.864', '8.65', '14', '93%'], ['10', '0.405', '6.44', '10', '93%'], ['100', '0.298', '5.91', '8', '95%']], caption: 'The same 40 points at five prices.' },
      { warn: 'A large C chases the training points and a small C ignores them. Choose C on a validation set or by cross-validation, never by the training accuracy, which tends to favour a large C.' },
      { h: 'Recap' },
      { list: ['Slack ξ measures how far a point falls short of its edge of the street.', 'The soft margin minimises ½‖w‖² + C·Σξ: C is the price of slack.', 'Small C: wide street, many support vectors, a smooth line. Large C: narrow street, few support vectors, a line that bends to single points.'] },
      X('sv-x-c-choice'), X('sv-x-c-curve'),
    ],
  });

  C.lesson({
    id: 'sv-hinge', chapter: 'sv-margin', title: 'The hinge loss', minutes: 7,
    summary: 'The slack is a loss: max(0, 1 − y·f(x)), the hinge. It is flat for points safely on their side, so they do not pull on the line, and grows in a straight line for the rest. The soft-margin SVM is regularised hinge-loss minimisation.',
    blocks: [
      'Logistic regression is pulled, a little, by every point, even by those it already gets right with confidence. Does the SVM care about such points at all?',
      { topic: 'svm-hinge', frame: 0, caption: 'The SVM with C = 1 on 40 points. On the right is the hinge loss against m = y·f(x): m ≥ 1 is outside the street on the point’s own side, 0 < m < 1 inside the street, m < 0 on the wrong side.' },
      { def: 'Hinge loss', text: 'ℓ(m) = max(0, 1 − m), with m = y·f(x) and y = ±1. Zero for m ≥ 1, a straight slope below.' },
      { topic: 'svm-hinge', frame: 1, caption: '26 points have m ≥ 1 and lose nothing, however far out they lie: the hinge is flat there.' },
      { predict: { q: 'A point is on the wrong side with m = −0.97. What is its hinge loss?', options: ['0', '0.97', '1.97', '0.03'], answer: 2, explain: '1 − (−0.97) = 1.97. Every point on the wrong side loses more than 1.' } },
      { topic: 'svm-hinge', frame: 3, caption: '3 points are on the wrong side and each loses more than 1; the worst, at m = −0.97, loses 1.97. The loss grows in a straight line, so one far outlier cannot drag the line as hard as a squared loss would.' },
      { topic: 'svm-hinge', frame: 4, caption: 'Σ hinge = 8.647 and ½‖w‖² = 2.677, so ½‖w‖² + C·Σ hinge = 11.323, the same value the dual problem reaches: the two views of the SVM agree at the optimum.' },
      { tip: 'The hinge lies above the 0–1 loss, which counts mistakes, everywhere. It is a convex stand-in for counting mistakes, and convex means the solver cannot get stuck in a poor local minimum.' },
      { h: 'Recap' },
      { list: ['The hinge loss is max(0, 1 − y·f(x)): the slack of the soft margin.', 'Points safely outside the street lose nothing and do not pull on the line.', 'The SVM minimises ½‖w‖² + C·Σ hinge: a loss plus a penalty on the size of w.'] },
      X('sv-x-hinge-table'), X('sv-x-hinge-sum'), X('sv-x-code-hinge'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: kernels
  // =========================================================================================================
  C.lesson({
    id: 'sv-lift', chapter: 'sv-kernels', title: 'A straight cut in a higher space', minutes: 8,
    summary: 'A disc inside a ring cannot be cut by a line. Add a third coordinate, the squared distance from the centre, and the disc sits low while the ring rises: now a flat plane separates them, and where it cuts the bowl is a circle when seen from above. A kernel computes the dot products of the lifted points without lifting them.',
    blocks: [
      'A pond sits in the middle of a meadow. Sensors in the water and sensors on the grass report only their map position. No straight fence separates the two. Can a linear method still learn the shore?',
      { topic: 'svm-kernel-lift', frame: 0, caption: '60 points: an orange disc inside a blue ring. The best straight line gets only 53% of them right.' },
      { topic: 'svm-kernel-lift', frame: 2, caption: 'Lift every point to the height z = x₁² + x₂². The disc stays low, up to z = 0.39, and the ring rises from z = 0.80: a gap opens between them.' },
      { predict: { q: 'What shape does a flat plane z = c cut from the bowl z = x₁² + x₂², seen from above?', options: ['A straight line', 'A circle', 'A parabola', 'Two crossing lines'], answer: 1, explain: 'x₁² + x₂² = c is a circle of radius √c round the centre.' } },
      { topic: 'svm-kernel-lift', frame: 3, caption: 'A linear SVM on the three coordinates finds a flat plane between the layers, with 4 support vectors, and gets 100% of the lifted points right.' },
      { topic: 'svm-kernel-lift', frame: 4, caption: 'Back on the map, the plane meets the bowl in a circle centred at (0.04, 0.00) with radius 0.768. A straight cut in 3-D is a curved cut in 2-D.' },
      { h: 'The kernel trick' },
      'The SVM never needs the lifted points themselves, only their dot products. A kernel K(x, x′) gives that dot product straight from the original coordinates.',
      { topic: 'svm-kernel-lift', frame: 5, caption: 'For x = (−0.74, −0.60) and x′ = (0.16, −0.45), lifting both and multiplying gives 0.3575, and K(x, x′) = x·x′ + ‖x‖²‖x′‖² gives 0.3575 straight away.' },
      { h: 'Recap' },
      { list: ['A map φ lifts the points into more dimensions, where a flat cut may separate them.', 'A flat cut in the lifted space is a curved boundary in the original one.', 'A kernel K(x, x′) = φ(x)·φ(x′) gives the lifted dot product without building φ: the kernel trick.'] },
      X('sv-x-lift-radius'), X('sv-x-ring-choice'),
    ],
  });

  C.lesson({
    id: 'sv-poly', chapter: 'sv-kernels', title: 'The polynomial kernel', minutes: 7,
    summary: 'The polynomial kernel (x·x′ + 1)^d stands for every product of the inputs up to degree d. Degree 2 can draw ellipses and hyperbolas; higher degrees bend more, with fewer support vectors, and values that grow fast away from the data.',
    blocks: [
      'Lifting by hand needs the right extra feature, the squared radius for a ring. What if we do not know which feature to add? Add all of them up to some degree, and let the SVM weigh them.',
      { def: 'Polynomial kernel', text: 'K(x, x′) = (γ x·x′ + r)^d. With γ = 1 and r = 1 it equals the dot product of all monomials of the inputs up to degree d, each with a fixed weight.' },
      { topic: 'svm-poly', frame: 1, caption: 'XOR corners, degree 1: a straight line, 48 support vectors, training accuracy 65%.' },
      { predict: { q: 'Two inputs, degree 2: how many features does the kernel stand for (counting the constant)?', options: ['3', '4', '6', '9'], answer: 2, explain: '1, x₁, x₂, x₁², x₂² and x₁x₂: six.' } },
      { topic: 'svm-poly', frame: 2, caption: 'Degree 2: 6 features, 24 support vectors, training accuracy 96%. The product x₁x₂ is exactly what XOR needs: its sign is the class.' },
      { topic: 'svm-poly', frame: 5, caption: 'Degree 5: 21 features and only 11 support vectors, still 96%. Higher degrees bend more, and their values grow fast away from the data, so scaling the inputs matters.' },
      { topic: 'svm-poly', frame: 6, caption: 'The shortcut, checked: for (−0.84, −0.94) and (−0.69, 0.67) the kernel (x·x′ + 1)² gives 0.9142, and the six features multiplied pairwise give 0.9142 too.' },
      { topic: 'svm-poly', frame: 2, input: { set: 'circles', C: 1, seed: 1 }, caption: 'The ring round a disc at degree 2: 23 support vectors and 100% of the points right. Degree 1 had managed 53%.' },
      { h: 'Recap' },
      { list: ['(x·x′ + 1)^d stands for every monomial of the inputs up to degree d.', 'Two inputs: 3, 6, 10, 15, 21 features for degrees 1 to 5.', 'Scale the inputs first: polynomial values explode away from the data.'] },
      X('sv-x-poly-value'), X('sv-x-poly-count'),
    ],
  });

  C.lesson({
    id: 'sv-rbf', chapter: 'sv-kernels', title: 'The RBF kernel and γ', minutes: 9,
    summary: 'The RBF kernel exp(−γ‖x − x′‖²) puts a bump on every support vector; γ sets how narrow the bumps are. Small γ gives a nearly straight boundary, large γ wraps every training point. Together with C it is chosen by validation.',
    blocks: [
      'The RBF kernel is the default of most SVM libraries. It has one knob of its own, γ. What does turning it do?',
      { def: 'RBF kernel', text: 'K(x, x′) = exp(−γ ‖x − x′‖²): 1 when the points coincide, falling towards 0 as they move apart. Its reach is about 1/√γ.' },
      { math: 'f(x) = Σᵢ αᵢ yᵢ exp(−γ ‖xᵢ − x‖²) + b      (a sum of bumps, one per support vector)' },
      { topic: 'svm-rbf', frame: 1, caption: 'γ = 0.1: each bump reaches about 3.16. 36 support vectors; training accuracy 90%, held-back accuracy 77%. The boundary is nearly straight.' },
      { topic: 'svm-rbf', frame: 3, caption: 'γ = 1: reach 1.00. Training accuracy 91% and held-back accuracy 83%.' },
      { predict: { q: 'γ goes up to 100. What happens to the held-back accuracy?', options: ['It climbs to 100% with the training accuracy', 'It falls, while the training accuracy reaches 100%', 'It stays exactly at 83%'], answer: 1, explain: 'Tiny bumps wrap the training points one by one: the boundary learns those points and nothing about the shape behind them.' } },
      { topic: 'svm-rbf', frame: 7, caption: 'γ = 100: reach 0.10. All 70 training points are support vectors; training accuracy 100%, held-back accuracy 70%.' },
      { topic: 'svm-rbf', frame: 8, caption: 'On the held-back points γ = 0.3 to 10 tie for best, at 83%, and γ = 100 falls to 70%. That is overfitting at the large end and underfitting at the small one, the same trade-off as a tree’s depth.' },
      { topic: 'svm-rbf', frame: 8, input: { set: 'moons', C: 100, seed: 2 }, caption: 'With C = 100 on another draw, the best γ is 0.1 (100% held back) and γ = 100 falls to 63%. A large C and a large γ both let the boundary bend; tune them together.' },
      { h: 'Try it on the canvas' },
      { workflow: 'svm-circles', title: 'An SVM with an RBF kernel on circles', text: 'Open the Train block to see the boundary and the ringed support vectors; double-click the SVM to change γ and C and watch them change.' },
      { h: 'Recap' },
      { list: ['K(x, x′) = exp(−γ‖x − x′‖²): a bump on each support vector, reaching about 1/√γ.', 'Small γ underfits (nearly straight), large γ overfits (every point a support vector).', 'Search C and γ together, on a grid of powers of ten, scored by validation.'] },
      X('sv-x-rbf-value'), X('sv-x-kernel-table'), X('sv-x-gamma-curve'), X('sv-x-gamma-choice'), X('sv-x-code-rbf'), X('sv-x-wf-ring'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: inside the solver
  // =========================================================================================================
  C.lesson({
    id: 'sv-dual', chapter: 'sv-solver', title: 'The weights as a sum over support vectors', minutes: 8,
    summary: 'The solver does not search for w directly. It finds one multiplier α per point, between 0 and C, and builds w = Σ αᵢ yᵢ xᵢ. Points with α = 0 drop out; the rest are the support vectors. The α of the two classes balance: Σ αᵢ yᵢ = 0.',
    blocks: [
      'An SVM with a kernel has no w you can write down: it lives in the lifted space. So what does the solver actually store?',
      { def: 'The dual problem', text: 'maximise Σ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ) subject to 0 ≤ αᵢ ≤ C and Σ αᵢ yᵢ = 0. Then w = Σ αᵢ yᵢ φ(xᵢ), and f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b.' },
      { topic: 'svm-dual', frame: 0, caption: 'The linear SVM with C = 1 on 16 points has w = (1.18, 1.45). Only 7 points have α > 0: the support vectors. On the right, w is built from them alone, starting from zero.' },
      { topic: 'svm-dual', frame: 1, caption: 'Point 0, orange (y = +1), has α = 1, at the cap C: its term αyx = (0.27, 0.34) points towards it.' },
      { predict: { q: 'Point 12 has α = 0.007. How much does it change w?', options: ['A lot: every support vector counts equally', 'Very little: its term is α times its coordinates', 'Nothing: α below C means it is ignored'], answer: 1, explain: 'Its term αyx is (0.01, 0.00): it touches the margin but barely leans on it.' } },
      { topic: 'svm-dual', frame: 7, caption: 'The last term, from point 12 with α = 0.007, adds (0.01, 0.00): the sum is now (1.18, 1.45).' },
      { topic: 'svm-dual', frame: 8, caption: 'The arrows end exactly at w. The orange multipliers add up to 2.7413 and the blue ones to 2.7413: the constraint Σ αᵢyᵢ = 0. The dual objective is 3.746.' },
      { table: [['α', 'Where the point is'], ['0', 'outside the street, on its own side: no say'], ['between 0 and C', 'exactly on an edge of the street'], ['C', 'inside the street, or on the wrong side']], caption: 'What a multiplier says about its point.' },
      { h: 'Recap' },
      { list: ['The solver finds one α per point, with 0 ≤ α ≤ C.', 'w = Σ αᵢyᵢxᵢ, and with a kernel f(x) = Σ αᵢyᵢK(xᵢ, x) + b: only points with α > 0 count.', 'The multipliers balance across the classes: Σ αᵢyᵢ = 0.'] },
      X('sv-x-w-from-alpha'), X('sv-x-alpha-balance'), X('sv-x-alpha-cap'), X('sv-x-decision'),
    ],
  });

  C.lesson({
    id: 'sv-smo', chapter: 'sv-solver', title: 'SMO: two multipliers at a time', minutes: 7,
    summary: 'Sequential minimal optimisation solves the dual by moving two multipliers at a time: the pair that breaks the optimality conditions most. Two is the fewest that can move while Σ αᵢyᵢ stays 0, and the best move for a pair has a closed form, clipped to the box [0, C].',
    blocks: [
      'The dual problem has one unknown per point. For thousands of points a general quadratic solver is slow. Is there a way to make progress with almost no algebra at each step?',
      { topic: 'svm-smo', frame: 0, caption: 'All 12 multipliers start at α = 0: every constraint holds, but nothing is solved (dual objective 0).' },
      { predict: { q: 'Why does SMO move two multipliers at once, never one?', options: ['Two is faster to compute than one', 'Moving a single α would break Σ αᵢyᵢ = 0; a pair can move together and keep it', 'The kernel needs two points', 'It is an arbitrary convention'], answer: 1, explain: 'Change one α alone and the balance Σ αᵢyᵢ = 0 breaks. Change two along the line that keeps their weighted sum fixed and it holds.' } },
      { topic: 'svm-smo', frame: 1, caption: 'Step 1: blue point 9 and orange point 0 break the conditions most, by a gap of 2. Both α move from 0 to 0.8436; the dual objective rises to 0.8436.' },
      { topic: 'svm-smo', frame: 2, caption: 'Step 2: the pair 9 and 3. The best move would take α₉ past C = 1, so it is clipped at the edge of the box: α₉ = 1 and α₃ = 0.1564.' },
      { topic: 'svm-smo', frame: 8, caption: 'After 7 steps no pair breaks the conditions by more than the tolerance 0.001. 4 multipliers are above zero, the support vectors; every other α is still exactly 0.' },
      { tip: 'Each step needs only one row of kernel values per chosen point, so SMO never has to hold the whole n × n kernel matrix. That is what made SVMs practical on large data.' },
      { h: 'Recap' },
      { list: ['SMO moves the pair that breaks the optimality conditions most.', 'A pair is the smallest move that keeps Σ αᵢyᵢ = 0.', 'Each move is solved in closed form and clipped to 0 ≤ α ≤ C; it stops when every gap is under the tolerance.'] },
      X('sv-x-smo-why'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: more classes, numbers and one class
  // =========================================================================================================
  C.lesson({
    id: 'sv-multi', chapter: 'sv-more', title: 'Many classes from two-class machines', minutes: 7,
    summary: 'An SVM splits two classes. For K classes, one-vs-rest trains K machines, each one class against all the others, and takes the largest output; one-vs-one trains a machine for every pair, K(K − 1)/2 of them, and lets them vote.',
    blocks: [
      'A nursery sorts seedlings into three kinds by two leaf measurements. One SVM can only answer yes or no. How do several of them name one of three?',
      { topic: 'svm-multiclass', frame: 0, caption: '45 points in three classes. First one-vs-rest, then one-vs-one.' },
      { topic: 'svm-multiclass', frame: 1, caption: 'One-vs-rest, machine 1: blue against both others together. It puts all 15 blue points on its positive side.' },
      { topic: 'svm-multiclass', frame: 4, caption: 'Each place goes to the class whose machine gives the largest value. At the square in the middle the three machines say −0.78, −0.78 and −1.11: every machine says “not mine”, and the largest of three negatives still decides.' },
      { predict: { q: 'With 5 classes, how many machines does one-vs-one train?', options: ['5', '10', '20', '25'], answer: 1, explain: 'One per pair: 5 · 4 / 2 = 10.' } },
      { topic: 'svm-multiclass', frame: 5, caption: 'One-vs-one, machine 1: blue against orange only, trained on their 30 points. The green points play no part in it.' },
      { topic: 'svm-multiclass', frame: 8, caption: 'Each pair votes. At the square the votes are 1, 1 and 1: a tie, broken here in favour of blue. Training accuracy 98%, and the two schemes agree on every training point.' },
      { table: [['', 'One-vs-rest', 'One-vs-one'], ['Machines for K classes', 'K', 'K(K − 1)/2'], ['Rows per machine', 'all n', 'only the two classes'], ['Decision', 'largest output', 'most votes']], caption: 'One-vs-one trains more machines, but each on less data; it is the default of many SVM libraries.' },
      { h: 'Recap' },
      { list: ['One-vs-rest: K machines, the largest output wins.', 'One-vs-one: K(K − 1)/2 machines, one per pair, the most votes win.', 'Both reduce a K-class problem to two-class SVMs.'] },
      X('sv-x-ovo-count'), X('sv-x-ovo-vote'),
    ],
  });

  C.lesson({
    id: 'sv-svr', chapter: 'sv-more', title: 'Support vector regression and its ε-tube', minutes: 7,
    summary: 'SVR fits a curve with a tube of half-width ε around it. Points inside the tube cost nothing; points outside pay their distance beyond it. Only those on or outside the tube are support vectors, so a wider tube gives fewer of them and a flatter fit.',
    blocks: [
      'An SVM for numbers: is there a margin, and what plays the part of the support vectors?',
      { def: 'ε-insensitive loss', text: 'ℓ(y, f(x)) = max(0, |y − f(x)| − ε): zero inside a tube of half-width ε round the fit, a straight slope outside. A hinge on each side.' },
      { topic: 'svm-svr', frame: 0, caption: '40 noisy points along a curve. SVR with the RBF kernel, γ = 0.5 and C = 10, fits them with tubes of different half-widths.' },
      { topic: 'svm-svr', frame: 1, caption: 'ε = 0.05: almost every point sticks out of the thin tube, so 28 of the 40 are support vectors.' },
      { predict: { q: 'ε grows to 0.6. What happens to the number of support vectors?', options: ['It grows', 'It shrinks', 'It stays at 28'], answer: 1, explain: 'A fat tube swallows most points; only those on or outside it remain support vectors.' } },
      { topic: 'svm-svr', frame: 5, caption: 'ε = 0.6: only 4 support vectors, and the error against the true curve has grown to 0.14. The few points left decide the curve on their own.' },
      { topic: 'svm-svr', frame: 2, caption: 'ε = 0.1: 18 support vectors and the smallest error against the true curve, 0.0025. A tube about as wide as the noise works best.' },
      { workflow: 'svr-curve', title: 'Support vector regression on a curve', text: 'Open the Train block to see the tube and the support vectors outside it; change ε and C to see the tube widen and the curve flatten.' },
      { h: 'Recap' },
      { list: ['SVR fits f with a tube of half-width ε; points inside it cost nothing.', 'Points on or outside the tube are the support vectors.', 'Wider tube: fewer support vectors, a smoother fit, eventually too smooth.'] },
      X('sv-x-eps-table'), X('sv-x-wf-flowers'),
    ],
  });

  C.lesson({
    id: 'sv-oneclass', chapter: 'sv-more', title: 'The one-class SVM: a border round normal data', minutes: 6,
    summary: 'With no labels at all, the one-class SVM draws a border round the bulk of the data and flags what falls outside it. ν bounds the share of points left outside from above and the share of support vectors from below.',
    blocks: [
      'A plant has thousands of minutes of pump readings and almost none are labelled as faults. Can an SVM learn what normal looks like, and flag the rest?',
      { topic: 'svm-one-class', frame: 0, caption: '100 points with no labels; 10 of them are strays scattered at random. The one-class SVM with the RBF kernel, γ = 0.5, draws a border for several values of ν.' },
      { def: 'ν (nu)', text: 'A number between 0 and 1. At most a share ν of the points fall outside the border, and at least a share ν are support vectors.' },
      { topic: 'svm-one-class', frame: 3, caption: 'ν = 0.1: 1 point is outside the border and 17 are support vectors. 1 of the 10 strays is flagged.' },
      { predict: { q: 'ν goes up to 0.4. Roughly how many of the 100 points can end up outside the border?', options: ['At most about 10', 'At most about 40', 'Exactly 40', 'All of them'], answer: 1, explain: 'ν is an upper bound on the outside share: up to 40%. Here 37 points end up outside.' } },
      { topic: 'svm-one-class', frame: 5, caption: 'ν = 0.4: 37 points outside and 43 support vectors, and all 10 strays are flagged, along with many ordinary points.' },
      { topic: 'svm-one-class', frame: 3, input: { set: 'pumps', gamma: 0.5, seed: 1 }, caption: 'On pump readings, ν = 0.1 flags 16 minutes, 9 of them logged as faults. Set ν near the share of faults you expect.' },
      { tip: 'The isolation forest of part 7 does the same job by another route: it counts how few random cuts it takes to isolate a point. Both flag the rare and the odd; neither knows which odd points matter.' },
      { h: 'Recap' },
      { list: ['The one-class SVM needs no labels: it draws a border round the bulk of the data.', 'ν bounds the outside share from above and the support-vector share from below.', 'Choose ν near the expected share of anomalies; γ sets how tightly the border hugs the data.'] },
      X('sv-x-nu'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
