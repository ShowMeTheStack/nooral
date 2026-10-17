# Nooral

Nooral is machine learning you can see inside. Drag a dataset, a split, a model and an
evaluation onto a canvas, wire them up, and open any block: the boundary forms as the model
trains, the tree grows split by split, the weights of a network thicken and change colour,
and a click on a cell of the confusion matrix lists the very rows the model got wrong. Under
every picture a sentence says what that step did, with the actual values.

Students meet machine learning as a library call that prints a score. They can fit a model
and quote "accuracy 0.93", but what the model learned stays out of sight: which boundary
produced that number, which examples it got wrong and why, what a split or a weight or an
attention head is doing, how any of it changed as training went on. On a whiteboard I can
work one step of gradient descent or one split of a tree by hand; a real model is a few
lines of code and a number nobody can question. Nooral opens every model up. Every model in
it is written for the tool, from least squares to a small GPT, so any intermediate value can
be drawn: the candidate thresholds a tree weighed, the multipliers SMO moved, the Q, K and V
of an attention layer, the gradient on every wire of a network. And the engines are checked
against scikit-learn and finite differences, so what you see is what the algorithm does.

The course is defensive where it touches misuse. Adversarial examples, poisoned labels and
what a model leaks about its training rows are shown on toy classifiers of our own, each
together with its defence (adversarial training and smoothing, cleaning the labels,
differential privacy), and never as a recipe against a real system.

Live at https://nooral.jyotiprakash.org

![The home page: a workflow of Dataset, Split, Train · k-NN and Evaluate, the k-NN boundary with k = 9 over two moons, and its confusion matrix with 30 and 32 right and 94% of test rows right](docs/images/home.png)

## What it is

Nooral is one HTML file. There is nothing to install and nothing is sent anywhere; your
workflows, answers and progress are kept in the browser. Under the page are engines written
for it: a tensor library with reverse-mode autodiff over a recorded graph, with layers
(dense, convolution, pooling, normalisation, embedding, RNN, LSTM, GRU, multi-head attention,
the transformer block) and the optimisers from SGD to AdamW; the classical algorithms (linear
and logistic regression, ridge and lasso, k-NN with a k-d tree, naive Bayes, LDA and QDA,
CART, random forests, AdaBoost, gradient boosting, SVMs by SMO and SVR, k-means++,
hierarchical clustering, DBSCAN, Gaussian mixtures by EM, PCA, t-SNE, a UMAP-style map and
the isolation forest); a BPE tokenizer and a GPT with a KV cache; an autoencoder, a VAE, a
GAN and diffusion; bandits, a grid world, cart-pole and the methods that solve them; GCN
layers; and every metric and explanation the course uses, down to Kernel SHAP. Training runs
in a Web Worker or in short slices, so the page does not freeze, and the same seed always
gives the same model. Every operation a student can watch returns a trace of its steps, and
every picture is drawn from that trace.

The data is small to medium, so a model trains in seconds. There are 26 datasets: 8 on the
plane (moons, circles, blobs, spirals, XOR, a checkerboard, a noisy line and a curve); 7
tables with stories of our own (a valley harvest, city bus delays, meadow wildflowers, small
loans, pump sensors, flat rents and exam results); drawn digits, shapes and icons, generated
from seeds; 5 sets of original short texts; and four classic open tables (Iris, Wine, Breast
Cancer Wisconsin and Palmer Penguins), credited in `NOTICE`. You can also bring a CSV file or
a folder of images, which stay in your browser.

There are 204 topics in 79 families, in eighteen parts, which follow the usual order of a
machine-learning course:

1. **Data and features** (13 topics in 6 families): tables, rows, features and labels, one
   column described, the scatter matrix and correlation, missing values and imputation,
   outliers, scaling and why k-NN needs it, one-hot and ordinal encoding, binning,
   polynomial and interaction features, feature selection, and leakage, shown happening.
2. **Splitting and evaluation** (16 in 6): training, validation and test sets, k-fold
   cross-validation, the confusion matrix with many classes and normalised, accuracy and
   balanced accuracy, macro, micro and weighted averages, MCC, precision, recall and F1, ROC
   and AUC (and AUC as a ranking), precision–recall curves, calibration and reliability
   diagrams, log loss, regression metrics and residuals, bias, variance and learning
   curves, validation curves, and grid and random search.
3. **Linear models** (12 in 7): the least-squares line and least squares as a projection,
   the closed form against gradient descent, polynomial regression and overfitting, ridge
   and lasso paths and why lasso gives zeros, the elastic net, logistic regression and a
   boundary forming, softmax regression, the perceptron, and LDA and QDA.
4. **Nearest neighbours and probabilistic models** (9 in 3): one neighbour and its Voronoi
   cells, k against smoothness, distance metrics and weighted votes, a k-d tree search, the
   curse of dimensionality, Gaussian naive Bayes, a spam filter word by word, and smoothing.
5. **Trees and ensembles** (12 in 5): a tree growing split by split with every candidate
   weighed, regression trees, depth and the smallest leaf, cost-complexity pruning, bagging,
   random forests and the out-of-bag score, impurity against permutation importance,
   AdaBoost, gradient boosting and shrinkage, and five tree models on the same points.
6. **Support vector machines** (11 in 4): the widest street and its support vectors, the
   soft margin and C, the hinge loss, the kernel trick as a lift into 3-D, polynomial and
   RBF kernels, the weights as a sum over support vectors, SMO two multipliers at a time,
   one-vs-rest and one-vs-one, SVR and its ε-tube, and the one-class SVM.
7. **Unsupervised learning** (17 in 7): k-means step by step, k-means++, the elbow and the
   silhouette, agglomerative clustering with four linkages and a dendrogram you can cut,
   DBSCAN and its ε and minPts, Gaussian mixtures by EM and BIC, PCA, explained variance and
   images rebuilt from a few components, t-SNE, a UMAP-style map, and the isolation forest.
8. **Optimisation** (12 in 4): loss surfaces from the side and from above, gradient descent
   and the learning rate, stochastic and mini-batch descent, momentum and Nesterov, AdaGrad,
   RMSProp and Adam, six optimisers raced on the same surface, schedules and warm-up,
   convexity, local minima and plateaus.
9. **Neural networks** (12 in 5): one neuron, activations and their slopes, the forward
   pass, backpropagation on the computational graph, the loss surface of a tiny network,
   initialisation, overfitting, dropout, weight decay, early stopping, batch and layer
   normalisation, and the playground.
10. **Convolutional networks** (12 in 4): the sliding kernel, padding, stride and dilation,
    pooling, channels, an image flowing through a trained CNN, learned filters, receptive
    fields, a CNN trained on generated digits and shapes, augmentation, residual
    connections, saliency maps and Grad-CAM.
11. **Sequences and embeddings** (9 in 3): one-hot vectors against embeddings, word2vec on
    our own text and the map of its vectors with analogies, an RNN unrolled through time,
    vanishing and exploding gradients, LSTM and GRU gates, predicting the next character,
    and sequence to sequence.
12. **Transformers** (12 in 3): tokens and embeddings, positional encodings, scaled
    dot-product attention matrix by matrix, why the scores are divided by √dₖ, causal and
    padding masks, multi-head attention, the feed-forward layer, residuals and layer
    normalisation, the full block traced, encoder against decoder, a tiny transformer
    learning to reverse, and the key–value cache.
13. **Language models** (12 in 4): byte-pair encoding, cutting a sentence into tokens,
    next-token prediction, cross-entropy and perplexity, training a tiny GPT, scaling, greedy
    choice and temperature, top-k and top-p, the KV cache, pretraining, fine-tuning,
    instruction and preference tuning, in-context learning on a toy task, and the limits:
    invented facts and copied bias.
14. **Generative models** (12 in 4): the autoencoder's bottleneck, walking the VAE's latent
    map, the reparameterisation trick, the KL term, a GAN duel on a ring, mode collapse, a
    GAN on small digits, diffusion forward and back on the moons and on 16×16 images, and
    four generative families side by side.
15. **Reinforcement learning** (10 in 4): a multi-armed bandit and the regret of greedy,
    ε-greedy, UCB and Thompson sampling, a Markov decision process, value and policy
    iteration, Monte Carlo control, Q-learning against SARSA on the cliff, exploration,
    REINFORCE balancing a pole, and a small deep Q-network with replay.
16. **Graph neural networks** (6 in 3): graphs and the adjacency matrix, message passing, a
    GCN layer as matrices, node classification from a few labels, over-smoothing, and
    graph-level readout.
17. **Robustness, fairness and privacy** (9 in 4), the defensive part: why models are
    brittle and how adversarial training and smoothing harden them, three kinds of
    distribution shift and watching for drift, fairness between groups and narrowing the
    gaps, differential privacy in training (clip, then add noise), what a model leaks about
    its training rows, and poisoned labels and cleaning them out.
18. **Practice and interpretability** (8 in 3): imbalanced data, choosing a model by
    cross-validation, no free lunch, permutation importance, partial dependence and ICE
    curves, SHAP-style contributions, LIME-style local surrogates, and a checklist for a
    real project.

Besides the topics there are 197 lessons, 563 exercises, 17 model challenges, 18 guided
tours and 18 courses.

## The canvas

The **canvas** is where a workflow is built, the way it is in Orange, with no code. There are
53 blocks in ten groups: data (a built-in dataset, a CSV file, a folder of images, the data
of a challenge), preparation (clean, impute, encode, scale, select features, polynomial
features, PCA, vectorise text, resample), a train/test split and cross-validation, 19 models
from linear regression to a CNN, an LSTM and a transformer classifier, Train, five kinds of
evaluation (the confusion matrix and its scores, ROC and PR curves, calibration, residuals,
learning curves), Compare, Predict, five explanations (permutation importance, partial
dependence, SHAP values, LIME, saliency and Grad-CAM), and six unsupervised blocks (k-means,
DBSCAN, hierarchical clustering, a Gaussian mixture, t-SNE and UMAP). Drag a block from the
list, draw a wire from port to port (ports are typed, and coloured by what they carry), and
double-click a block for its parameters: k, C, γ, the depth, the learning rate, the layers,
the heads. Everything downstream runs again as soon as something changes. A tidy layout keeps
wires from crossing. There are 39 sample workflows to start from, and a workflow can be saved
as a file or shared as a link.

![The canvas with a comparison: a dataset of two moons split into training and test rows, a logistic regression, a decision tree and k nearest neighbours each trained, and a Compare block taking the three trained models and the test data](docs/images/workflow.png)

Open any block and it shows what is inside. An Evaluate block shows the confusion matrix;
click a cell and the rows behind it are listed, here the 26 applicants who defaulted but were
predicted to repay, with every score worked out from the four counts.

![The canvas with the Evaluate block open on small loans: a logistic regression right on 146 of 180 test rows, the confusion matrix 128, 8, 26, 18 with the cell of 26 defaulted-but-predicted-repaid picked, the scores worked out (accuracy 146/180 = 0.811, F1 of defaulted 0.514, MCC 0.428), precision and recall per class, and the 26 rows listed below](docs/images/canvas.png)

A Train block can be played, paused, stepped an epoch or a batch at a time and reset, with
the training and validation curves drawn as it goes; a model block shows its own insides: the
tree with its splits, the SVM with its margin and support vectors, the network with its
weights, the boundary over the plane.

![The Train block of a network learning the spirals: done after 150 epochs, training accuracy 94.2% with 114 parameters, controls to train again, step an epoch or a batch, and the loss and accuracy of training and validation rows against the epoch](docs/images/canvasmlp.png)

## Watching a topic

A topic has inputs, a few presets, a stage, and controls to play, pause, step either way,
scrub and change the speed. Each step is narrated with the values it uses. Beside the stage
the definition panel shows the rule or formula with the line in use lit, and a list of every
step to jump to; counters keep a tally where it matters (epochs, splits, sweeps, gradients
computed). Before key moments the player can stop and ask what comes next, and each topic
ends with a quiz. A topic with its input can be shared as a link.

Logistic regression on two moons: the probability over the plane, the boundary turning
epoch by epoch, the loss falling and the weights travelling, and the points still on the
wrong side ringed.

![A decision boundary forming: logistic regression on two moons at epoch 28, the probability of class 1 shaded over the plane with the straight boundary, the 18 misclassified points ringed, the mean log loss falling to 0.306, and the path of the two weights](docs/images/boundary.png)

A tree grows split by split. For every candidate threshold on each feature the weighted Gini
is drawn, and the lowest is the cut taken; the plane is cut into boxes as the tree below it
grows.

![A tree growing on two moons: the plane cut into four boxes, the weighted Gini of every cut on x1 and on x2 with the lowest marked, and the tree with root x2 ≤ 0.0185 and the node x1 ≤ 1.67 just split, the impurity falling from 0.361 to 0.185](docs/images/tree.png)

The kernel trick is a lift into 3-D: a ring that no line can separate becomes a bowl that a
plane cuts, and the plane comes back down as a circle.

![The kernel trick: points on a ring lifted onto the bowl z = x₁² + x₂², a plane cutting the bowl with its four support vectors ringed, and back in 2-D the boundary as a circle of radius 0.768 with every point right](docs/images/svm.png)

k-means alternates assigning and moving, and the inertia can only fall.

![k-means after its third iteration: three centres with the dashed paths they moved along, the plane coloured by the nearest centre, the inertia falling to 315.85, and the three centres with their point counts](docs/images/kmeans.png)

The optimisers race on the same surface, each with its best learning rate.

![Six optimisers raced on a ravine, L = 0.1u² + 2v²: the paths of gradient descent, momentum, Nesterov, AdaGrad, RMSProp and Adam over the contours, and the loss above the lowest point against the step on a log scale, RMSProp leading at step 40](docs/images/race.png)

Backpropagation runs on the computational graph, every value and every gradient shown, and
each gradient is the local derivative times the one that arrived.

![Backpropagation on a tanh neuron: the graph from x₁, w₁, x₂, w₂ and b through the sums, h = tanh(u), the output y and the squared error L = 3.249, with ∂ filled in from the loss back to u = −1.814, worked as 1 − h² = 0.559 times −3.244](docs/images/backprop.png)

A trained CNN reads a drawn 7: the feature maps after each convolution and pool, the dense
layer, and the chances of each class.

![An image flowing through a trained CNN: the 16 × 16 input, six maps after the first ReLU and their pools, twelve after the second, the dense layer, and the softmax giving 7 at 100%](docs/images/cnn.png)

![Grad-CAM for an 8: the image, the twelve maps of the last convolution with their weights, the weighted sum cut at 0, and the map stretched over the image, marking the lower right as the evidence](docs/images/gradcam.png)

Scaled dot-product attention is shown matrix by matrix: Q, K and V, the scores, the
softmax, and each output as a blend of the values.

![Scaled dot-product attention on "rain fell on roofs": Q, K and V as heat maps, the weights A, A times V, and the output of "fell" worked out as 0.22 + 0.02 + 0.29 − 0.03 = 0.50, with arcs from "fell" to the words it attends to](docs/images/attention.png)

![Multi-head attention in a tiny transformer that reverses digits: the four heads of two layers as heat maps over 2 7 4 0 5 = 5 0 4 7, layer 0 head 1 pointing at the right digit 5 times out of 5](docs/images/heads.png)

![Walking the VAE's latent map: the test digits' codes for 0, 1, 3 and 7 on the plane, a walk from the 0s across the map, the grid of what the decoder draws at each point, and the five decoded images along the walk](docs/images/vae.png)

![Value iteration on a grid world: the value of each cell after six sweeps with the best move as an arrow, two pits at −1 and the goal at +1, and the largest change per sweep falling below 10⁻⁶](docs/images/rl.png)

![Message passing with the GCN rule at node 2: the graph, the table of features, and the four messages from nodes 0, 1, 3 and itself, each weighted by 1/√(d̂ᵤ d̂ᵥ)](docs/images/gnn.png)

![SHAP-style contributions for one applicant: from the average output 0.189, years employed, missed payments and income pushing the chance of default down and debt ratio and amount pushing it up, to this row's output 0.063](docs/images/shap.png)

## The playground

The **playground** is a small network on 2-D data. Pick one of seven datasets, switch inputs
on (x₁, x₂, their squares, their product, their sines), add layers and neurons, choose the
activation, the learning rate, the optimiser, the regularisation and the batch size, and
press play. Every neuron is drawn as a small map of what it says over the plane; every weight
is a line whose colour is its sign and whose thickness is its size; the output is drawn over
the plane with the training and test points and the boundary at p = 0.5; and the training
and test loss fall below it. Click a neuron to see its own map large.

![The playground on the spirals after 300 epochs: four hidden layers of 8, 8, 6 and 4 ReLU neurons over seven inputs, every neuron a small map, the weights as blue and orange lines, the output with a spiral boundary and 87% of test points right, and the training and test loss](docs/images/playground.png)

## The text lab

The **text lab** shows how a language model reads and writes, on about 175,000 characters
of stories written for Nooral. It has three tabs.

**Byte-pair encoding** learns merges one at a time: the pairs are counted, the most frequent
becomes the next token, the text gets shorter in tokens, and a sentence is cut with the
merges so far. It can learn from your own text too.

![Byte-pair encoding after 60 merges: the most frequent pairs counted (g + h 422 next), the length of the text in tokens falling against the number of merges, the merges so far, and a sentence of 83 characters cut into 55 tokens](docs/images/bpe.png)

**Train a tiny GPT** trains a decoder-only transformer here in the browser, character by
character. By default (Auto) its size depends on the browser: where WebGPU works it trains on
the graphics card, four layers of width 192 with six heads and a context of 128 characters,
1,816,128 parameters, in about a minute for its 1,200 planned steps; elsewhere it trains on the
CPU, three layers of width 96 with four heads and a context of 64, 347,712 parameters, over
2,000 steps. The line above the loss says which size and which backend it took, and either
size can be chosen by hand, as can a small, quick one. It trains in a Web Worker, so the page
keeps answering; play, pause or step it, and watch the training and validation loss fall
(perplexity is e to the loss), a sample written every 150 steps go from noise to words, and
the attention of every head on a prompt.

![The tiny GPT after 900 steps on the graphics card: Auto chose the GPU size, 1,816,128 parameters and 61 characters, the training loss at 1.297 and the validation loss at 1.374 (perplexity 4.0) falling from above 4, and the samples written every 150 steps, from "The tare wery sland from the hat" at step 300 to "The was had to card. \"I will a left.\"" at step 900](docs/images/gpt.png)

**Sampling** writes with the model one character at a time. The bars show its ten most likely
next characters before and after the rule: greedy choice, a temperature, top-k and top-p,
and what survives is scaled back up to 1. Before the transformer is trained it samples from
character pairs counted in the stories, so there is always something to sample from.

![Sampling after "The night market": the ten most likely next characters, each with the model's probability and the one after a temperature of 0.8 and top-k = 5 (h 34.0% → 46.3%, the space 27.2% → 34.9%, and five cut), and the rule's settings](docs/images/sampling.png)

## Challenges, lessons, exercises, tours and courses

The **challenges** are 17 datasets with stories of their own (a lender pricing the odds, a
pump that hums wrong, a robot mower that must never drive into a flower bed), each with a
target score and limits on what you may use: which blocks and models, the range of a
parameter, how many epochs, how many blocks. Build a workflow, train it and submit it: it is scored on a hidden test set it has
never seen. Attempts and best scores stay in the browser.

![The challenges: cards for rents from half-filled listings, the odds a lender can price, the flower bed in the lawn, a forecast for the valley farms, the pump that hums wrong, who will pass the exam and more, each with its difficulty, its target (R² ≥ 0.76, log loss ≤ 0.48, accuracy ≥ 0.91, RMSE ≤ 0.355, F1 ≥ 0.9 …) and whether it has been tried](docs/images/challenge.png)

The **lessons** are short readings, a few to each chapter, with live topic players inside
them, predictions to answer, and links to the exercises that follow. The **exercises** are
checked by the page. The 563 exercises are of ten kinds: 177 ask for a number, 156 are
multiple choice, 83 fill in a table, 53 ask for code, 29 ask you to sketch a curve, 24 fill
in a matrix (a confusion matrix, a one-hot encoding, a row of polynomial features), 17 ask
for a workflow that reaches a score on rows it never sees, 15 ask for a set, 8 ask you to
click a point on a plot, and one asks for a string. Code runs in a sealed Web Worker, with no
page, no network and a time limit. Every exercise has three hints, the last a worked answer.

A **tour** walks through a part of the page a step at a time, lighting what to look at; there
is one for each of the eighteen parts. A **course** is an ordered path
through the topics, lessons and exercises of a part, each step with a word on what to look
for. Progress through all of these is kept in the browser.

![Learn: lessons read and exercises solved, the guided tours, and part 1 with its chapters, each a list of lessons with their reading times and exercises with their difficulty](docs/images/learn.png)

## Phone, projector and themes

Every page works on a phone. The topic stage comes first with the narration under it, the
inputs follow, and the definition and steps open in a drawer from the bottom. Projector mode
(Shift+P) makes the type larger and the lines stronger for a classroom. There is a light and
a dark theme, or the one the system uses.

![Multi-head attention in the dark theme: the four heads of a tiny transformer reversing digits](docs/images/dark.png)

<img src="docs/images/phone.png" width="300" alt="The playground on a phone: the controls, and a network of two tanh layers on the circles after 80 epochs, every neuron a small map">

## Keyboard

| Key | What it does |
| --- | --- |
| Space | play or pause a topic |
| → ←, Home, End | next step, step back, first or last step |
| R | play again from the start |
| F, Shift+F | fit the picture, the stage on the whole screen |
| + − | zoom in, out |
| B | a bigger stage |
| D | the definition or the steps (on a phone, the drawer) |
| P, Q | predictions on or off, the quiz |
| Enter, O | on the canvas: a block's parameters, its viewer |
| Delete | on the canvas: delete what is selected |
| Ctrl+Z, Ctrl+Y | on the canvas: undo, redo |
| L | on the canvas: tidy the layout |
| T, Shift+P | light or dark, projector mode |
| Esc | close a dialog, the viewer or the menu |
| ? | the list of keys |

## Running it locally

Nooral is the single file `index.html`. Open it from any web server:

```
python3 -m http.server 8000
```

and go to http://localhost:8000. To change it, edit the files under `src` and run

```
node build.mjs
```

which joins the sources listed in `src/manifest.json` into `index.html`. While working,
`node tools/devpage.mjs 8080` serves http://127.0.0.1:8080/dev.html, the same page with every
source file loaded on its own, so a change needs no build.

## Correctness

A model that draws the wrong boundary teaches the wrong thing, so the engines are checked
against references.

- **Against scikit-learn.** Scripts in `tests/ref` run scikit-learn, NumPy and SciPy on fixed
  seeds and datasets and store what they give as JSON; the tests compare the engines with it.
  Checked this way: linear and logistic regression, ridge, lasso and the elastic net, class
  weights, k-NN, naive Bayes, LDA and QDA, trees, random forests, AdaBoost and gradient
  boosting, SVMs (decision values, support vectors and Platt probabilities) and SVR, k-means,
  hierarchical clustering, DBSCAN, Gaussian mixtures, PCA, t-SNE, the isolation forest,
  feature selection, imputation and scaling, every classification, regression and clustering
  metric, ROC and PR curves, calibration, partial dependence and the fairness metrics.
  Python is used only by these scripts; nothing in the page depends on it.
- **Against finite differences.** The gradient of every differentiable operation is compared
  with central differences in float64, on random shapes and seeds, and small versions of the
  deep models are checked the same way end to end.
- **Learning.** Every demo model must reach a known score on its demo set within a time
  budget: networks on XOR and the moons, a CNN on digits and shapes, an LSTM, a transformer
  that reverses sequences, the GPT on the stories, the generative models, the RL agents and
  the GCN.
- **Determinism.** The same seed gives the same model, and models survive a save and a load.
- **WebGPU.** The tensor engine also has a WebGPU backend (WGSL shaders for matrix products,
  convolution, softmax and attention, and a path that trains the GPT on the graphics card,
  which the text lab uses wherever WebGPU works). In headless Chrome every kernel is
  compared with the CPU; where a browser has no WebGPU the check is reported as skipped, not
  passed.

The narration is checked as well. For every topic, run with its default input, every preset
and random inputs, each value a sentence quotes is compared with a working-out done again
apart from the engines: closed forms, normal equations, finite differences, or scikit-learn
where the data match.

## Testing

`npm test`, which is `node tests/run.mjs`, runs every test in `tests`. Each can also run on
its own: the engine tests `node tests/tensor.mjs`, and likewise `grad`, `tensor-learn`,
`tensor-gpu`, `classic`, `metrics`, `data`, `dl`, `gen`, `rl`, `gnn`, `rob`, `playground`,
`textlab`, `challenges` and the workflow tests `wf`, `wf-data`, `wf-eval`, `wf-layout` and
`wf-models`; and the narration checks `node tests/claims-p01.mjs` to `claims-p18.mjs`, one
for each part.

`node tools/check-topics.mjs` runs every topic with its default input and every preset, and
checks that every frame draws, that the steps are narrated, that the predictions can be
answered, and that the topic has a quiz. `node tools/check-content.mjs` checks every
exercise (its solution passes its own checker and every one of its wrong answers fails; code
runs under the sandbox's time limit; a workflow exercise's reference workflow reaches its
score on the hidden set and every wrong one does not), every lesson, every challenge, and
every tour and course step.

`node tools/check-diagrams.mjs` draws the sample scenes and every page and topic in both
themes and at desktop and phone sizes, and reports shapes that overlap, wires that cross or
run through blocks, labels that collide or are cut off, text smaller than 11 pixels, and
pages wider than the screen. `node tests/e2e/run.mjs` runs the end to end tests in headless
Chrome: the shell, the canvas and its data, evaluation and model viewers, the playground,
the text lab, the challenges, lessons and exercises. `node tools/readme-shots.mjs` makes the
pictures in this file.

## Project layout

```
src/core      the engines: tensors and autodiff, the CPU, WebAssembly and WebGPU kernels, the classical
              algorithms, deep learning, the GPT, generative models, RL, GNNs, metrics and explanations,
              robustness, fairness and privacy, the datasets
src/stage     the stage that draws and animates a frame, and its drawing kits (scatter and boundary plots,
              heat maps, network and tree diagrams, contours, curves, a dendrogram, token strips, grids ...)
src/topics    the topics, a file per family group: inputs, presets, frames, narration, predictions and quiz
src/content   the lessons, exercises, challenges, tours and courses, a file per part
src/app       the page: the canvas and its blocks and viewers, the topic player, the playground, the text
              lab, the challenges, lessons, checkers, styles
src/demo      the sample scenes (index.html#demo)
tests         the engine tests, the scikit-learn references in tests/ref, the narration checks, and the end
              to end tests in tests/e2e
tools         the Chrome driver, the topic, content and diagram checkers, screenshots
```

## License

Nooral is free software under the GNU General Public License, version 3 or later. See
`LICENSE` for the full text and `COPYRIGHT` for the notice.

Under section 7 of the license there are additional terms, written out in full in `NOTICE`.
Any copy, modification or derived work, including a hosted version, must keep the attribution
"Based on Nooral by Jyotiprakash Mishra (https://nooral.jyotiprakash.org)" in its source and
show it in its about screen or other legal notices. This applies to the program, the topics
with their narration and quizzes, the lessons, exercises and challenges, and screenshots or
recordings of the animations used as teaching material. A modified version must be marked as
changed and must not be presented as the original Nooral. The name Nooral is not licensed for
derived works.

The algorithms and the mathematics of machine learning belong to everyone. The datasets,
texts, stories, challenges, examples, pictures and explanations in Nooral were made for it,
apart from four classic open datasets (Iris, Wine, Breast Cancer Wisconsin and Palmer
Penguins), which are credited, with their licences and the changes made, in `NOTICE`.

Copyright (C) 2026 Jyotiprakash Mishra
