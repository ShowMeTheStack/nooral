// Part 3, linear models: the chapters and the lessons. The exercises are in 031-linear-ex.js; the model challenges, the
// tour, the course and the pages the diagram checker looks at are in 032-linear-ch.js. Each lesson opens with a
// question, runs the topic that shows the idea at a frame chosen for the point being made, asks for a prediction, sums
// up, and leads to its exercises. Every value quoted here was worked out by the engines (S.ML) on the topics' own
// inputs.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('li-line')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'li-ls', part: 'linear', n: 1, title: 'Least squares and gradient descent', blurb: 'The best line as a sum of squares, the same line as a projection, the closed form against gradient descent on one loss bowl, and polynomials that fit too well.' });
  C.chapter({ id: 'li-reg', part: 'linear', n: 2, title: 'Ridge, lasso and elastic net', blurb: 'A penalty on the size of the weights: paths as λ slides, why the lasso lands on exact zeros, and a mix of the two for features that come in pairs.' });
  C.chapter({ id: 'li-class', part: 'linear', n: 3, title: 'Linear classifiers', blurb: 'The sigmoid and log loss, a boundary forming epoch by epoch, softmax for many classes, and the perceptron that updates only on its mistakes.' });

  // =========================================================================================================
  // Chapter 1: least squares and gradient descent
  // =========================================================================================================
  C.lesson({
    id: 'li-line', chapter: 'li-ls', title: 'The least-squares line', minutes: 7,
    summary: 'Of all straight lines through a cloud of points, least squares picks the one whose squared vertical misses add up to the least. Setting the derivatives to zero gives two equations, the normal equations, and the line always passes through the mean point.',
    blocks: [
      'Ten points lie roughly along a rising line. You could draw a line through them by eye, and so could a friend, and the two lines would differ. Which one is best? To answer that we need a number that says how badly a line misses, and then the line that makes the number smallest.',
      { def: 'Residual', text: 'the vertical miss yᵢ − ŷᵢ of a point from the line ŷ = w·x + b. Least squares makes the sum of the squared residuals, Σ (yᵢ − w·xᵢ − b)², as small as possible.' },
      { topic: 'ls-line', frame: 1, caption: 'Start with no slope: the flat line at the mean ȳ = 4.86. Each residual is drawn as a square; their total area, 15.7, is the total sum of squares. Any line worth having must do better than this.' },
      { topic: 'ls-line', frame: 3, caption: 'The derivatives with respect to b and w give the normal equations. The matrix holds n = 10, Σx = 46.9 and Σx² = 291.4; the right side holds Σy = 48.64 and Σxy = 255.1. Two equations, two unknowns.' },
      { math: 'n·b    + Σx·w  = Σy\nΣx·b  + Σx²·w = Σxy' },
      { topic: 'ls-line', frame: 4, caption: 'The solution w = 0.378, b = 3.09. The squares shrink from 15.7 to 5.46, so R² = 1 − 5.46 / 15.7 = 0.652: the line explains 65% of the squared spread of y.' },
      { predict: { q: 'Turn the best line a little about the mean point, steeper or flatter. What happens to the total area of the squares?', options: ['It grows either way', 'It shrinks if the line gets steeper', 'It stays the same for small turns', 'It depends on the noise'], answer: 0, explain: 'At the minimum every small change makes the sum larger. The topic turns the line by ±0.15 in slope and both give 7.07, above the 5.46 of the best line.' } },
      { topic: 'ls-line', frame: 5, caption: 'Both turned lines give 7.07. At the best line the residuals sum to zero and are uncorrelated with x: that is exactly what the two normal equations say.' },
      { h: 'One point far away' },
      'Squaring punishes big misses much more than small ones. A single point far from the rest therefore pulls hard on the line.',
      { topic: 'ls-line', input: { outlier: 'yes', noise: 0.5 }, frame: 4, caption: 'The same kind of data with one outlier added: the slope drops to 0.296 and R² falls to 0.336. One point out of eleven has bent the line away from the other ten.' },
      { tip: 'R² compares the line with the flat line at the mean: 1 means every point is on the line, 0 means the line is no better than the mean. On new data it can even be negative.' },
      { h: 'Recap' },
      { list: ['Least squares minimises Σ (yᵢ − w·xᵢ − b)², the total area of the residual squares.', 'The normal equations come from setting both derivatives to zero; the best line passes through (x̄, ȳ).', 'At the optimum the residuals sum to zero and are uncorrelated with x.', 'Squares make outliers pull hard: one far point can tilt the whole line.'] },
      X('li-x-ls-normal'), X('li-x-ls-r2'), X('li-x-ls-residuals'),
    ],
  });

  C.lesson({
    id: 'li-projection', chapter: 'li-ls', title: 'Least squares as a projection', minutes: 7,
    summary: 'Put the n targets in one vector of n numbers. Every line’s predictions lie in the plane spanned by the column of ones and the column of inputs; least squares picks the point of that plane nearest to y, and the residual meets the plane at a right angle.',
    blocks: [
      'The normal equations look like algebra to be ground through. There is a picture behind them that explains at once why the residuals sum to zero, and it needs only three points. Turn the data on its side: instead of three points in the plane, think of one arrow in three dimensions.',
      { topic: 'ls-projection', frame: 0, caption: 'The targets y = (1, 3, 2) are one arrow: each axis is one observation. Drag to turn the view.' },
      { topic: 'ls-projection', frame: 2, caption: 'The columns 1 = (1, 1, 1) and x = (0, 1, 2) span a plane. Every prediction b·1 + w·x lies in it. The flat guess (2, 2, 2) is 1.41 from y; least squares looks for the nearest point of the plane.' },
      { predict: { q: 'The nearest point of a plane to an arrow’s tip is where the gap meets the plane at…', options: ['a right angle', 'an angle of 45°', 'the origin', 'the flat guess (2, 2, 2)'], answer: 0, explain: 'Drop a perpendicular. Any other point of the plane is farther, by Pythagoras.' } },
      { topic: 'ls-projection', frame: 3, caption: 'The solution b = 1.5, w = 0.5 gives ŷ = (1.5, 2, 2.5). The residual r = (−0.5, 1, −0.5) drops straight down to the plane; its length 1.22 is shorter than the 1.41 of the flat guess.' },
      { topic: 'ls-projection', frame: 4, caption: 'Right angles to both columns: r · 1 = 0, so the residuals sum to zero, and r · x = 0. These two dot products are the normal equations. And ‖r‖² = 1.5 is the sum of squared residuals.' },
      { math: 'Xᵀ (y − X β) = 0\n⇔  Xᵀ X β = Xᵀ y' },
      'With more inputs the plane becomes a larger flat space spanned by more columns, but the picture is the same: least squares projects y onto the space of all possible predictions.',
      { topic: 'ls-projection', input: { y: ['1', '2', '3'] }, frame: 3, caption: 'When y already lies in the plane (here y = (1, 2, 3) = 1 + x), the projection is y itself and the residual has length 0: the line passes through every point.' },
      { h: 'Recap' },
      { list: ['The n targets form one vector; every possible prediction lies in the span of the columns of X.', 'Least squares picks the nearest point of that span: the orthogonal projection of y.', 'The residual is perpendicular to every column; those dot products are the normal equations.', '‖r‖² is the sum of squared residuals.'] },
      X('li-x-ls-project'), X('li-x-ls-normal'),
    ],
  });

  C.lesson({
    id: 'li-descent', chapter: 'li-ls', title: 'The closed form against gradient descent', minutes: 9,
    summary: 'The loss of a line is a bowl over the plane of (w, b). The closed form jumps to its floor; gradient descent walks downhill. How fast it walks, or whether it blows up, depends on the step η and on the shape of the bowl, which depends on the scale of x.',
    blocks: [
      'For a line, the normal equations give the answer in one go. For logistic regression and neural networks there is no such formula, and the method that trains them is gradient descent. A line is the best place to watch it, because we know the answer it should reach.',
      { math: 'L(w, b) = (1/2n) Σ (yᵢ − w·xᵢ − b)²\nw ← w − η ∂L/∂w\nb ← b − η ∂L/∂b' },
      { topic: 'gd-closed', frame: 0, caption: 'The raw x runs from 0 to 10. The closed form lands at w* = 0.582, b* = 2.2 (green). The bowl is a long narrow valley: it curves 269 times more steeply one way than the other.' },
      { topic: 'gd-closed', frame: 1, caption: 'Gradient descent starts at (0, 0), where the loss is 16.5 and the gradient is (−34.6, −5.51). The first step follows minus the gradient, almost entirely along w.' },
      { topic: 'gd-closed', frame: 16, caption: 'After 60 epochs at η = 0.0251 the loss is 0.567, still above the floor 0.363: along the flat floor of the valley the gradient is tiny, so the walk crawls.' },
      { predict: { q: 'The steepest direction of this bowl allows η up to 2 / 38.8 = 0.0515. What happens with η = 0.062?', options: ['It gets there faster', 'Every step overshoots further: it diverges', 'It stops at once', 'It finds a different minimum'], answer: 1, explain: 'Past 2 divided by the largest curvature, each step lands farther up the opposite wall than it started. The loss of a line has a single minimum, so there is nothing else to find.' } },
      { topic: 'gd-closed', input: { scale: 'raw', lr: Math.log10(0.062), epochs: 30 }, frame: 14, caption: 'With η = 0.062 the steps bounce between the walls of the valley and grow: gradient descent diverges, while the closed form had the answer from the start.' },
      { h: 'Scale the input' },
      'Standardise x (subtract its mean, divide by its spread) and the bowl turns round: both directions curve alike, and one step size suits both.',
      { topic: 'gd-closed', input: { scale: 'std', lr: -1, epochs: 60 }, frame: 16, caption: 'Standardised x, η = 0.1: after 60 epochs gradient descent is within 0.0102 of the closed form. The limit is now η < 2, not 0.0515.' },
      { tip: 'Why not always use the closed form? It needs to solve a d × d system, which grows quickly with the number of features, and it exists only for squared error. Gradient descent needs one gradient per step and works for any differentiable loss.' },
      { h: 'Recap' },
      { list: ['Gradient descent steps against the gradient: w ← w − η ∂L/∂w.', 'Too small an η crawls; above 2 / (largest curvature) it diverges.', 'Unscaled inputs make the bowl a narrow valley, so a safe step is tiny; standardising makes it round.', 'For least squares the closed form gives the same answer in one solve.'] },
      X('li-x-gd-step'), X('li-x-gd-code'), X('li-x-gd-curve'), X('li-x-gd-lrmax'),
    ],
  });

  C.lesson({
    id: 'li-poly', chapter: 'li-ls', title: 'Polynomials and overfitting', minutes: 8,
    summary: 'Least squares on x, x², x³, … fits curves with the same machinery. Each extra degree lowers the training error, but past some degree the curve bends to pass through the noise and the error on fresh points explodes. Choose the degree on data the fit did not see.',
    blocks: [
      'A straight line cannot follow a wave. Add columns x², x³ and so on to the design matrix and the same least squares fits a polynomial: the model is still linear in its weights, only the features have changed. How many powers should we add?',
      { topic: 'poly-overfit', frame: 1, caption: 'Twelve noisy points on a sine wave. Degree 1, a straight line: training error 0.0795, test error 0.28 on 60 fresh points. It underfits.' },
      { topic: 'poly-overfit', frame: 3, caption: 'Degree 3: training error 0.021, test error 0.0782. Both fell: the bends help.' },
      { predict: { q: 'Going from degree 3 to degree 7, what happens to the two errors?', options: ['Both keep falling', 'Training error falls, test error rises', 'Both rise', 'Training error rises, test error falls'], answer: 1, explain: 'More columns can only lower the training error, because the old fit is still available. But the extra freedom goes into chasing the noise of twelve points.' } },
      { topic: 'poly-overfit', frame: 7, caption: 'Degree 7: training error 0.00844, test error 9.71. The largest weight is 58.8: huge weights of opposite signs nearly cancel at the training points and swing wildly between them.' },
      { topic: 'poly-overfit', frame: 12, caption: 'Degree 11 passes through every one of the 12 points (training error 0) and has a test error of 38643. The best test error, 0.0782, was at degree 3.' },
      { h: 'More data' },
      { topic: 'poly-overfit', input: { n: 40, maxDeg: 14 }, frame: 15, caption: 'With 40 points instead of 12, degree 14 has test error 0.0763, close to the best 0.0655 at degree 3: more data tames a flexible model.' },
      'The same lesson in a workflow: one model scored on its training rows and on held-out rows. Open the blocks and compare the two numbers.',
      { workflow: 'train-vs-test', title: 'Training rows against test rows', text: 'A model scored on the rows it learned from and on rows it never saw.' },
      { warn: 'Never choose the degree, or any other setting, by the training error: it always prefers the most flexible model. Use a validation set or cross-validation.' },
      { h: 'Recap' },
      { list: ['Polynomial regression is least squares on the columns x, x², …, x^d.', 'The training error falls with every degree; the test error falls, then rises.', 'Overfitted polynomials have huge weights of opposite signs.', 'More data, or a penalty on the weights (next chapter), lets a flexible model generalise.'] },
      X('li-x-poly-pick'), X('li-x-poly-curve'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: ridge, lasso and elastic net
  // =========================================================================================================
  C.lesson({
    id: 'li-paths', chapter: 'li-reg', title: 'Ridge and lasso paths', minutes: 8,
    summary: 'Add a penalty on the size of the weights to the squared error: λ‖w‖² for ridge, λ‖w‖₁ for the lasso. As λ shrinks from very large to zero the weights grow from zero to the least-squares solution. Ridge shrinks them all smoothly; the lasso brings them in one at a time.',
    blocks: [
      'The overfitted polynomial had weights in the thousands. What if we charged the fit for large weights? The fit then has to balance two things, missing the points and having big weights, and a single number λ sets the exchange rate.',
      { math: 'ridge:  ‖y − Xw‖² + λ ‖w‖²\nlasso:  (1/2n) ‖y − Xw‖² + λ Σ |wⱼ|' },
      { topic: 'reg-paths', frame: 0, caption: '60 rows and 8 standardised features, of which only a few matter. With no penalty, least squares gives the bars at the right and a training error of 2.94.' },
      { topic: 'reg-paths', frame: 1, caption: 'λ = 2.43 is just strong enough to hold every lasso weight at exactly 0: the model predicts the mean for everyone, training MSE 14.7.' },
      { topic: 'reg-paths', frame: 3, caption: 'λ = 1.08: three features have left zero, x1, x2 and x3. The others are still exactly 0.' },
      { predict: { q: 'Ridge on the same data at a large λ: how many of the 8 weights are exactly zero?', options: ['All 8', 'The same 5 as the lasso', 'None', 'It depends on the seed'], answer: 2, explain: 'Ridge shrinks every weight toward 0 but never onto it: at λ = 10000 the largest weight is 0.0145, and none is exactly 0.' } },
      { topic: 'reg-paths', input: { method: 'ridge' }, frame: 19, caption: 'The ridge path: every weight shrinks smoothly as λ grows, and none reaches zero at any finite λ. Ridge never removes a feature.' },
      { topic: 'reg-paths', frame: 19, caption: 'The lasso path read right to left: x1 came in first, the feature that explains most. A λ in between keeps only the features that earn their place.' },
      { tip: 'Scale the features before a penalty. The penalty treats every weight alike, and a feature measured in metres needs a weight a thousand times larger than the same feature in millimetres.' },
      'Try it on flat rents: the workflow below trains a lasso. Open it, double-click the model block to change α, and watch which weights reach zero.',
      { workflow: 'lasso-rents', title: 'The lasso on flat rents', text: 'As the penalty grows, the weights of the five features reach zero one by one.' },
      { h: 'Recap' },
      { list: ['Ridge adds λ‖w‖², the lasso λ‖w‖₁; λ = 0 is plain least squares.', 'As λ grows, all weights shrink; the training error rises.', 'The lasso sets weights exactly to zero, so it also selects features; ridge never does.', 'Standardise the features first, so the penalty treats them alike.'] },
      X('li-x-reg-which'), X('li-x-reg-path'), X('li-x-reg-ridge1d'), X('li-x-reg-lambda'), X('li-x-reg-wf'),
    ],
  });

  C.lesson({
    id: 'li-corners', chapter: 'li-reg', title: 'Why the lasso gives zeros', minutes: 7,
    summary: 'Penalising the weights is the same as limiting them to a budget: a disc for ridge, a diamond for the lasso. The best weights inside the budget are where the error’s elliptical rings first touch the shape. A diamond has corners on the axes, and rings often touch there: one weight is then exactly zero.',
    blocks: [
      'Why should one penalty give exact zeros and the other never? Look at the same problem another way: instead of paying for large weights, allow the weights only a fixed budget, and ask where the best weights inside the budget lie.',
      { topic: 'reg-geometry', frame: 0, caption: 'The least-squares weights ŵ = (1.88, 0.607), with the error rising in elliptical rings around them, tilted because the two features are correlated.' },
      { topic: 'reg-geometry', frame: 3, caption: 'At 85% of ŵ’s size, ridge touches the disc at (1.55, 0.633) and the lasso touches the diamond at (1.69, 0.42), on an edge: both weights still non-zero.' },
      { predict: { q: 'Shrink the budget further. Where will the rings first touch the diamond?', options: ['Always on an edge', 'At a corner on the w₁ axis', 'At the origin', 'At a corner on the w₂ axis'], answer: 1, explain: 'The rings are centred far out along w₁, and the corner on that axis sticks out toward them. Touching at a corner means w₂ = 0 exactly.' } },
      { topic: 'reg-geometry', frame: 6, caption: 'At 42% of the budget the lasso touches the corner (1.04, 0): w₂ is exactly 0. Ridge, at (0.714, 0.42), still has both.' },
      { topic: 'reg-geometry', frame: 10, caption: 'Ridge shrinks both weights all the way down and never zeroes either. The disc has no corners for the rings to catch on.' },
      { h: 'The same in one formula' },
      'Coordinate descent, which the engine uses for the lasso, updates one weight at a time. With the others fixed, the best value of wⱼ is a soft threshold: compute ρⱼ, how much feature j agrees with what is left of y, then move it toward zero by nλ, and stop at zero if it would cross.',
      { math: 'ρⱼ = Σᵢ xᵢⱼ (yᵢ − Σₖ≠ⱼ xᵢₖ wₖ)\nwⱼ = S(ρⱼ, nλ) / Σᵢ xᵢⱼ²\nS(ρ, t) = sign(ρ) · max(|ρ| − t, 0)' },
      'The flat stretch of the soft threshold, |ρ| ≤ nλ, is the corner of the diamond seen from one axis: a whole range of ρ gives exactly 0.',
      { h: 'Recap' },
      { list: ['A penalty λ is the same as a budget on the weights: a disc for ridge, a diamond for the lasso.', 'The solution is where the error’s rings first touch the shape.', 'The diamond’s corners sit on the axes, so the lasso often lands with some weights exactly 0.', 'Coordinate descent applies the soft threshold one weight at a time.'] },
      X('li-x-reg-soft'), X('li-x-reg-softcode'), X('li-x-reg-cdcode'),
    ],
  });

  C.lesson({
    id: 'li-enet', chapter: 'li-reg', title: 'Elastic net for twin features', minutes: 6,
    summary: 'When two features are near copies, the lasso picks one and drops the other, somewhat at random. The elastic net mixes the two penalties: it keeps the lasso’s zeros for useless features but moves twins together.',
    blocks: [
      'A house’s floor area in square metres and the same area estimated from the number of rooms carry almost the same information. Which one should a sparse model keep? The lasso has no good answer: either does almost as well, and it takes whichever wins by a hair.',
      { math: 'elastic net:  (1/2n) ‖y − Xw‖²\n             + λ r ‖w‖₁\n             + (λ (1 − r) / 2) ‖w‖²' },
      { topic: 'elastic-net', frame: 0, caption: 'Six features: two near twins that both matter, four of noise. The elastic-net ball for r = 0.5 (shaded) lies between the lasso diamond and the ridge disc: corners on the axes, bulging between them.' },
      { topic: 'elastic-net', frame: 2, caption: 'Early on the path, the lasso gives the twins 0.959 and 0; the elastic net gives 0.232 and 0.123. The lasso has picked one twin; the mix uses both.' },
      { predict: { q: 'Further along the path, what do the four noise features get?', options: ['Large weights from both', 'Weights near zero from both', 'Zero from the lasso, large from the elastic net', 'The same weights as the twins'], answer: 1, explain: 'The L1 part still pushes useless weights to zero; the elastic net keeps the lasso’s sparsity where it is deserved.' } },
      { topic: 'elastic-net', frame: 17, caption: 'Over the whole path the lasso let one twin in 3 steps before the other, while the elastic net moved them together. The noise features stay near zero in both.' },
      { tip: 'r = 1 is the lasso and r → 0 is ridge. A value in between is a safe default when features come in correlated groups.' },
      { h: 'Recap' },
      { list: ['Among near-identical features, the lasso keeps one and drops the rest.', 'The elastic net adds a ridge part, which spreads the weight over the twins.', 'It still sets useless weights to exactly 0.'] },
      X('li-x-reg-twins'), X('li-x-reg-which'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: linear classifiers
  // =========================================================================================================
  C.lesson({
    id: 'li-sigmoid', chapter: 'li-class', title: 'The sigmoid and log loss', minutes: 8,
    summary: 'Logistic regression turns a linear score z = w·x + b into a probability with the sigmoid σ(z) = 1 / (1 + e^−z), and is trained by making the log loss, −ln of the probability given to the true class, small on average. Its gradient is simply (p − y)·x.',
    blocks: [
      'Forty students, their hours of study, and whether they passed. A straight line through 0s and 1s predicts values below 0 and above 1, which are no use as probabilities. We want a curve that stays between 0 and 1 and rises with the hours.',
      { math: 'z = w·x + b\np = σ(z) = 1 / (1 + e^−z)\nloss if y = 1:  −ln p\nloss if y = 0:  −ln (1 − p)' },
      { topic: 'logit-sigmoid', frame: 0, caption: 'Passes at the top, fails at the bottom, by hours of study. Training starts with w = 0 and b = 0.' },
      { topic: 'logit-sigmoid', frame: 1, caption: 'With w = 0 every student gets p = σ(0) = 0.5 and pays −ln 0.5 = 0.693. The gradient, ∂L/∂w = −0.196 in standardised units, says: make w larger.' },
      { predict: { q: 'A student who passed is given P(pass) = 0.1. What does the log loss charge for that point?', options: ['0.1', '0.9', '−ln 0.1 = 2.3', '−ln 0.9 = 0.105'], answer: 2, explain: 'The loss is −ln of the probability given to what actually happened: a confident mistake costs a lot, and the cost has no upper limit as p goes to 0.' } },
      { topic: 'logit-sigmoid', frame: 2, caption: 'Epoch 1: z = 0.0185·x − 0.204 in hours, mean log loss 0.672. The ringed point, a pass after only 2.1 hours, is the costliest.' },
      { topic: 'logit-sigmoid', frame: 13, caption: 'After 120 epochs: z = 0.168·x − 1.87, mean log loss 0.599. Points far on the wrong side still cost the most; moving the curve to help them would hurt more of the others.' },
      { h: 'The gradient' },
      'Differentiate the mean log loss and almost everything cancels. Each point contributes its error p − y, weighted by its input:',
      { math: '∂L/∂w = (1/n) Σ (pᵢ − yᵢ) xᵢ\n∂L/∂b = (1/n) Σ (pᵢ − yᵢ)' },
      'It has exactly the form of the least-squares gradient, with the probability in place of the prediction.',
      { topic: 'logit-sigmoid', input: { data: 'apart', n: 30, lr: 1, epochs: 600, seed: 2 }, frame: 17, caption: 'When the classes do not overlap at all, a steeper sigmoid always lowers the loss: w is 2.07 after 600 epochs and would keep growing. A penalty on w (the C of logistic regression) gives it a finite answer.' },
      { h: 'Recap' },
      { list: ['p = σ(w·x + b) is a probability between 0 and 1.', 'Log loss charges −ln of the probability given to the true class.', 'The gradient is the mean of (p − y)·x: the same shape as for least squares.', 'On separable data the weights grow without bound unless penalised.'] },
      X('li-x-cl-sigmoid'), X('li-x-cl-logloss'), X('li-x-cl-grad'), X('li-x-cl-code'), X('li-x-cl-half'),
    ],
  });

  C.lesson({
    id: 'li-boundary', chapter: 'li-class', title: 'A decision boundary forming', minutes: 7,
    summary: 'With two inputs, logistic regression’s boundary p = 0.5 is the straight line w₁x₁ + w₂x₂ + b = 0. Training turns and moves it until the log loss stops falling. On curved or crossed data no straight line is good enough, and new features are needed.',
    blocks: [
      'With one input, the decision was a threshold on the hours. With two inputs, the places where z = 0 form a line, and that line is the model’s whole decision. Where does it go during training, and what can it not do?',
      { topic: 'logit-boundary', frame: 1, caption: 'Two clouds. At w = 0 every point gets p = 0.5. The first gradient is (−0.412, −0.321) for the two weights: the red arrow is the first step, η = 0.5 times minus it.' },
      { topic: 'logit-boundary', frame: 6, caption: 'Epoch 11: mean log loss 0.302 and 6 of 80 points on the wrong side. The weights grow, so the band of uncertain p around the line narrows.' },
      { topic: 'logit-boundary', frame: 13, caption: 'After 150 epochs: log loss 0.189, 93% of points on their own side. The remaining mistakes sit where the clouds overlap.' },
      { predict: { q: 'XOR: opposite corners share a class. What accuracy does logistic regression reach?', options: ['About 100%', 'About 75%', 'About 50%, like guessing', 'It depends on the learning rate'], answer: 2, explain: 'The four corners balance each other, so the log loss is lowest with the weights near 0 and every p near 0.5. The topic’s run ends with log loss 0.693 and 51% on their own side. Even the best possible line, one that cuts off a single corner, gets only about 75%.' } },
      { topic: 'logit-boundary', input: { data: 'xor', n: 80, lr: 1, epochs: 100, seed: 3 }, frame: 12, caption: 'XOR after 100 epochs: log loss 0.693, the same as at the start, and 51% right: no position of the line does much better than guessing.' },
      { topic: 'logit-boundary', input: { data: 'moons', n: 120, lr: 1, epochs: 200, seed: 5 }, frame: 14, caption: 'Two moons: 86% on their own side. The line cuts through both hooks; curved features such as x₁², x₁x₂ and x₂² would let the boundary bend.' },
      'Logistic regression in a full workflow: data, a split, the model, training and evaluation. Open the Train block to see the boundary.',
      { workflow: 'first-pipeline', live: true, height: 460, caption: 'Logistic regression on the two moons, running in the page.' },
      { h: 'Recap' },
      { list: ['The boundary p = 0.5 is the line (or plane) w·x + b = 0.', 'Larger weights make the probability change faster across it.', 'A straight boundary cannot separate XOR or follow the moons; add curved features or use another model.'] },
      X('li-x-cl-wf'), X('li-x-cl-choose'),
    ],
  });

  C.lesson({
    id: 'li-softmax', chapter: 'li-class', title: 'Softmax regression: many classes', minutes: 6,
    summary: 'For K classes, keep K linear scores and turn them into probabilities with the softmax: exponentiate each and divide by the sum. Train with cross-entropy, −ln of the true class’s probability. Only the differences between scores matter.',
    blocks: [
      'A meadow has three kinds of wildflower. The sigmoid gives one probability; we need three that add up to 1. Give each class its own score and let them compete.',
      { math: 'zₖ = wₖ·x + bₖ\npₖ = e^zₖ / Σⱼ e^zⱼ' },
      { topic: 'softmax', frame: 0, caption: 'All weights start at 0, so every score is 0 and each class gets 1/3. The cross-entropy is ln 3 = 1.1.' },
      { topic: 'softmax', frame: 1, caption: 'Epoch 1: cross-entropy 0.898, and 89% of points already in their own class’s region.' },
      { predict: { q: 'Add 5 to all three scores at some point. What happens to the three probabilities?', options: ['All rise', 'Nothing changes', 'The largest rises', 'They no longer add up to 1'], answer: 1, explain: 'e^(z+5) = e⁵·e^z: the factor e⁵ appears above and below the fraction and cancels. Only differences between scores matter.' } },
      { topic: 'softmax', frame: 11, caption: 'After 120 epochs: cross-entropy 0.255. Every region is convex and every border is a straight line where two scores are equal.' },
      { topic: 'softmax', input: { data: 'flowers', n: 150, lr: 0.5, epochs: 200, seed: 1 }, frame: 12, caption: 'The wildflowers by their petals: cross-entropy 0.182 after 200 epochs. The shading fades where the winning probability is small, near the borders.' },
      { tip: 'With two classes, softmax reduces to the sigmoid of the difference of the two scores: logistic regression is softmax regression with K = 2.' },
      { h: 'Recap' },
      { list: ['K classes, K scores, softmax into K probabilities.', 'Cross-entropy is −ln of the true class’s probability.', 'Adding a constant to every score changes nothing.', 'The regions are convex, with straight borders.'] },
      X('li-x-cl-softmax'), X('li-x-cl-shift'),
    ],
  });

  C.lesson({
    id: 'li-perceptron', chapter: 'li-class', title: 'The perceptron', minutes: 6,
    summary: 'The perceptron predicts the sign of w·x + b and learns only from mistakes: when a point with label t ∈ {+1, −1} is on the wrong side, it adds η·t·x to w and η·t to b. On separable data it stops after finitely many updates; on overlapping data it never settles.',
    blocks: [
      'The oldest learning rule of all is also the simplest. No probabilities, no loss curve: look at one point; if it is on the right side do nothing, otherwise nudge the line toward it.',
      { math: 'if t·(w·x + b) ≤ 0:\n   w ← w + η t x\n   b ← b + η t' },
      { topic: 'perceptron', frame: 1, caption: 'Point 0 at (0.83, −0.81) has label −1 and scores 0, on the line: a mistake. w becomes (−0.83, 0.81) and b becomes −1.' },
      { topic: 'perceptron', frame: 2, caption: 'Point 1 at (0.61, 1.08) has label +1 but scores −0.631: a mistake. Adding it gives w = (−0.22, 1.89) and b = 0.' },
      { predict: { q: 'The next pass through all 24 points finds no mistakes. What does the perceptron do?', options: ['Keeps nudging toward the widest gap', 'Stops: nothing triggers an update', 'Starts again with w = 0', 'Halves η'], answer: 1, explain: 'Updates happen only on mistakes. With none, the weights stay where they are, wherever that is.' } },
      { topic: 'perceptron', frame: 5, caption: 'Done after 2 updates. The line separates the classes, but it is one of many; nothing pushed it to the middle of the gap.' },
      { topic: 'perceptron', input: { data: 'overlap', n: 24, eta: 1, epochs: 6, seed: 2 }, frame: 24, caption: 'Overlapping classes: 17 updates and 8 points still wrong. Some point is always on the wrong side, so the line keeps swinging.' },
      { workflow: 'perceptron-line', title: 'The perceptron on a noisy line', text: 'Every mistake nudges the line; open the Train block and watch it settle, or not.' },
      { h: 'Recap' },
      { list: ['The perceptron predicts sign(w·x + b) and updates only on mistakes.', 'On separable data it stops, at some separating line, not the widest one.', 'On overlapping data it never settles; logistic regression, with its smooth loss, does.'] },
      X('li-x-cl-perc'), X('li-x-cl-overlap'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
