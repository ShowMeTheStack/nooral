// EXERCISE KINDS: the reference for content writers. An exercise is
//   S.Content.exercise({ id, chapter, title, level: 1..3, kind, text: ['paragraph' | lesson block, ...],
//     hints: [hint 1, hint 2, 'the worked answer'], solution: <an answer that passes>,
//     wrong: [<answers that fail>, ...], why: 'said after a right answer', whyNot: 'said after a wrong one',
//     sample: true (placeholder content, warned about), ...the fields of its kind })
// tools/check-content.mjs runs the solution and every wrong answer through the checker: the solution must pass and
// every wrong answer must fail. Write a wrong answer for each mistake you expect, and read its message: it is what
// the student sees (node tools/check-content.mjs <id prefix> --verbose). Numbers in text and answers come from the
// engines (src/core), worked out in a scratch script with tests/load.mjs, never typed from memory.
//
// Every checker returns { ok, message, details? }; message is one plain sentence with the actual values.
//
// ---- the basic kinds (src/app/check-basic.js, check-table.js, check-code.js) ----------------------------------------
//   choice      options: ['..'], answer: index | [indexes] (several right: the reader picks all); whyNot: {index: 'why
//               that option is wrong'}. Options and text may use `code`, **bold** and [links](#topic=id).
//                                                                                      Answer: index or [indexes]
//   string      answer: 'text' | [...] or accept(text) -> true | false | 'why not' | {ok, why}; set: true compares as
//               sets; caseless: true; spaces: true keeps single spaces; placeholder.                Answer: text
//   number      answer: a number; tolerance (absolute, e.g. 0.005 for an answer to two decimals); higherLower: true
//               says whether the answer is more or less; whyNot: 'sentence' or {value: 'why that slip', '*': 'any
//               other value'} (keys are the numbers as typed, like '0.75'). Fractions like 3/4 are not read here: ask
//               for a decimal.                                                                    Answer: number
//   set         answer: '{a, b}' | '(1, 2), (2, 3)' | '{{a, b}, {c}}' (or several), or accept(value, canonical);
//               caseless. For a set of row numbers, feature names, cluster members.                 Answer: text
//   table       rows: ['r0', ..], cols: ['c0', ..], cells: {'row|col': expected}, cellKind: 'number' | 'text' | 'bit' |
//               'set' (numbers when every expected value is a number), tolerance (number cells), corner (the
//               top-left label). Only the cells listed are asked for. A table of a training step, a k-means
//               assignment, a confusion matrix's per-class counts. With no wrong answers listed, the solution
//               with one cell changed is tried.                                        Answer: {'row|col': text}
//   code        fn: 'sigmoid', params: ['z'] (named in the message when the function is missing), start: 'function
//               sigmoid(z) {\n  \n}' (the editor's first text), tests: [{args: [..], want}] or ref: (...args) => want
//               with cases: [[args], ...], tolerance (numbers, and arrays of numbers at any depth, compared to within
//               it: use it for anything computed in floating point), timeLimit (ms, default 2000), prelude: 'helper
//               source' shown read-only above the editor and run before the student's code. The code runs in a
//               sandboxed worker (no page, no network), case by case; the first failing case is the message.
//                                                                                       Answer: JavaScript source
//
// ---- the machine-learning kinds (src/app/check-ml.js, check-workflow.js; answer areas in exercises-ml.js) ----------
//   matrix      Fill in a matrix: a confusion matrix from a list of predictions, attention weights from scores.
//               rows: ['cat', 'dog'] (row names: the true classes, or the queries), cols: [..] (the predicted classes,
//               or the keys), answer: [[..], ..] (numbers), tolerance (0 for counts; 0.01 for weights), corner: 'true
//               \ predicted' (the top-left label), given: [[i, j], ..] (cells shown filled in), rowsSumTo: 1 (attention:
//               a live sum beside each row, and a wrong answer whose rows do not add up is told so), diag: true (a
//               confusion matrix: the diagonal of correct answers is tinted), total: n (counts: a wrong answer whose cells do
//               not add up to n is told so), note: 'the line under the grid' (by default it says what goes in the cells:
//               counts when every answer is a whole number, weights with rowsSumTo, else numbers), rowLabel, colLabel
//               (what rows and columns are, in the counts note). The grid is drawn as the matrix, wrong cells marked. With no wrong
//               answers listed, the solution with one cell changed is tried.
//                                                     Answer: [[text or number, ..], ..] (given cells ignored)
//   point       Click on a plot. plot: {points: [[x, y], ..], classes: [k, ..] (colours the points), labels: [..]
//               (names of the points, for pick), xr: [a, b], yr: [a, b], xname, yname, centroids: [[x, y], ..]
//               (drawn as crosses), lines: [{x} | {y} | {pts: [[x, y], ..]}] (guides drawn faint)}. A plot of one
//               feature (points as plain numbers, no yr, or strip: true) is drawn as a strip with no y axis, the points
//               spread up and down only where they would overlap; it takes mode 'x' or 'pick'. mode: 'xy' (click a
//               place: the next centroid; answer [x, y], tolerance a distance) | 'x' (a vertical line: a split
//               threshold; answer x) | 'y' (a horizontal line; answer y) | 'pick' (click one of the points; answer: its
//               index, or [indexes]; whyNot: {index: 'why not that one'}), tolerance (default 0.1).
//                                                    Answer: {x, y} | {x} | {y} | {i}
//   predict-curve  Sketch where a curve goes: a learning curve as data grows, a loss as training runs on, accuracy as
//               k grows. plot: {xr, yr, xname, yname, given: [[x, y], ..] (the part already known, drawn faint), note:
//               'what the sketch is of'}, answer: [[x, y], ..] (the true curve, x increasing, from the engines),
//               tolerance (the mean gap allowed, in y units; default a tenth of the y range), checks: ['rising' |
//               'falling' | 'peak' | 'valley'] (shape rules that must hold too). Each given point is also marked with a dot, so a
//               single known point shows. The sketch must cover 80% of the
//               answer's x range; the message says where it is too high or too low.
//                                                    Answer: [[x, y], ..] (points along the sketch)
//   workflow    Build a pipeline that reaches a target score on a hidden test set. data: {set, opts, seed, n} (the
//               public rows, from S.Data: the workflow gets them from a Challenge data block), task, metric: 'accuracy'
//               | 'f1' | 'precision' | 'recall' | 'balanced' | 'auc' | 'logloss' | 'r2' | 'mae' | 'rmse' | 'mse'
//               (S.WF.METRICS; F1, precision and recall are binary for two classes, class 1 positive, and macro for
//               more), target, limits: {blocks: [allowed block types], models: [allowed model types], params: {type:
//               {param: [min, max]}} (null for no bound), epochs, maxBlocks}, hidden: {seed, n} (the hidden rows:
//               the same set and options, another seed), starter: a workflow doc to begin from (S.WF.build({blocks,
//               wires}); S.WF.challengeBlock(data) gives the data block). The scored model is the Train block that feeds
//               an Evaluate block (else the last Train block); its preparation is repeated on the hidden rows.
//               solution and every wrong answer are workflow docs; the checker runs each headless.
//                                                    Answer: a workflow document {blocks, wires}
//
// Hints: three, the last a worked answer. Lesson blocks: src/app/lessons.js. Model challenges (the same fields as
// workflow, plus id, n, title, level, story, marks: {silver, gold}, metricName (optional: the metric's name as shown;
// by default WF.METRICS[metric].label) and three hints): src/app/challenges-core.js.
(function (S) {
  'use strict';
  const K = (S.Check = S.Check || {});
  /** the kinds this file documents, for tools that list them */
  K.kindNames = ['choice', 'string', 'number', 'set', 'table', 'code', 'matrix', 'point', 'predict-curve', 'workflow'];
})(globalThis.Nooral = globalThis.Nooral || {});
