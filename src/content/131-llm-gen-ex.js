// Part 13, language models: the exercises of chapters 3 and 4 (generating text; from a tiny model to a large one).
// Probabilities come from the softmax of S.ML and the text lab's sampling rule S.TL.cut; the preset sizes from
// S.DL.gptPresets; figures quoted from the topics (the limits run, the in-context model) were printed by them.
(function (S) {
  'use strict';
  const C = S.Content, TL = S.TL;
  if (!C || !TL || !TL.cut || C.exercises.has('ll-x-temp-table')) return;
  const ML = S.ML, ln = Math.log, ex = Math.exp;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const f2 = (v) => r2(v).toFixed(2), f3 = (v) => r3(v).toFixed(3);
  const str = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));

  // =========================================================================================================
  // Chapter 3: generating text
  // =========================================================================================================
  const Z = [2, 1, 0], TS = [0.5, 1, 2], TOK = ['“a”', '“b”', '“c”'], tc = {};
  TS.forEach((t) => { const p = ML.softmax(Z.map((z) => z / t)); TOK.forEach((k, i) => { tc[k + '|T = ' + t] = r2(p[i]); }); });
  const tWrong = Object.assign({}, str(tc)); TOK.forEach((k, i) => { tWrong[k + '|T = 0.5'] = String(r2(ML.softmax(Z.map((z) => z * 0.5))[i])); });
  C.exercise({ id: 'll-x-temp-table', chapter: 'll-generate', title: 'One distribution at three temperatures', level: 2, kind: 'table',
    text: ['A model gives the scores a: 2, b: 1, c: 0. Divide the scores by the temperature T, then take the softmax. Fill in the probabilities at T = 0.5, 1 and 2, to two places.'],
    rows: TOK, cols: TS.map((t) => 'T = ' + t), corner: 'token \\ T', cells: tc, tolerance: 0.011,
    hints: ['At T = 1 the scores go in as they are: e², e¹, e⁰ over their sum.', 'At T = 0.5 the scores become 4, 2, 0; at T = 2 they become 1, 0.5, 0.', TS.map((t) => 'T = ' + t + ': ' + TOK.map((k) => tc[k + '|T = ' + t]).join(', ')).join('; ') + '. Lower T sharpens, higher T flattens.'],
    solution: str(tc), wrong: [tWrong] });

  const gOpts = ['It always gives the same text for the same prompt, and tends to repeat the commonest phrases', 'It always gives the most varied text', 'It needs a random seed', 'It never repeats itself'];
  C.exercise({ id: 'll-x-greedy', chapter: 'll-generate', title: 'What greedy decoding does', level: 1, kind: 'choice',
    text: ['In [Greedy choice and temperature](#topic=sampling-temperature) greedy decoding writes “he would wall she wall said that he woul”. Which describes greedy decoding?'],
    options: gOpts, answer: 0,
    whyNot: { 1: 'It takes the single likeliest token every time: the least varied choice there is.', 2: 'Nothing is drawn at random, so there is no seed to set.', 3: 'With nothing random, once it returns to a context it has seen, it repeats what it did there.' },
    hints: ['Is anything drawn at random?', 'Greedy takes the top token at every step: the same prompt, the same steps.', 'Deterministic and safe, and prone to loops through the commonest words, as “said that he woul…” shows.'],
    solution: 0, wrong: [1, 2] });

  const tSol = 'function withTemperature(scores, T) {\n  const z = scores.map((s) => s / T), m = Math.max(...z);\n  const e = z.map((v) => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);\n  return e.map((v) => v / sum);\n}';
  C.exercise({ id: 'll-x-code-temp', chapter: 'll-generate', title: 'Write softmax with a temperature', level: 2, kind: 'code',
    text: ['Write `withTemperature(scores, T)`: divide each score by T, then return the softmax, a list of probabilities adding up to 1. Subtract the largest value before taking exponentials, so large scores do not overflow.'],
    fn: 'withTemperature', params: ['scores', 'T'], start: 'function withTemperature(scores, T) {\n  \n}', tolerance: 1e-9,
    ref: (s, t) => ML.softmax(s.map((v) => v / t)),
    cases: [[Z, 1], [Z, 0.5], [Z, 2], [[0, 0, 0, 0], 0.7], [[1000, 999], 1], [[1.2, 0.4, -0.3, -1], 0.1], [[5, -5], 10]],
    hints: ['First z_i = scores[i] / T.', 'Then e^(z_i − max z), divided by their sum.', tSol],
    solution: tSol,
    wrong: ['function withTemperature(scores, T) {\n  const e = scores.map((s) => Math.exp(s * T)), sum = e.reduce((a, b) => a + b, 0);\n  return e.map((v) => v / sum);\n}', 'function withTemperature(scores, T) {\n  const e = scores.map((s) => Math.exp(s / T)), sum = e.reduce((a, b) => a + b, 0);\n  return e.map((v) => v / sum);\n}'] });

  const PR = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04], K3 = TL.cut(PR, { topK: 3 }), PP = 0.85, NP = TL.cut(PR, { topP: PP }).kept.filter(Boolean).length;
  C.exercise({ id: 'll-x-topk', chapter: 'll-generate', title: 'Renormalise after top-k', level: 2, kind: 'number',
    text: ['Six tokens have probabilities 0.40, 0.25, 0.15, 0.10, 0.06 and 0.04. Top-k sampling with k = 3 keeps the three most likely and renormalises. What probability does the most likely token have now? Two decimal places.'],
    answer: r2(K3.probs[0]), tolerance: 0.005,
    whyNot: { '0.4': 'The three kept tokens must add up to 1 again: divide by what they held.', '0.33': 'Renormalising keeps the proportions; it does not make the kept tokens equal.', '0.8': 'That is what the three held together.', '*': 'Divide 0.40 by 0.40 + 0.25 + 0.15.' },
    hints: ['The three kept tokens hold 0.40 + 0.25 + 0.15.', 'That is 0.80; each kept probability is divided by it.', '0.40 / 0.80 = ' + f2(K3.probs[0]) + '.'],
    solution: f2(K3.probs[0]), wrong: ['0.4', '0.33', '0.8'] });

  C.exercise({ id: 'll-x-topp', chapter: 'll-generate', title: 'How large is the nucleus?', level: 2, kind: 'number',
    text: ['The same six probabilities, 0.40, 0.25, 0.15, 0.10, 0.06 and 0.04. Top-p sampling with p = ' + PP + ' keeps the fewest most likely tokens whose probabilities add up to at least ' + PP + '. How many tokens does it keep?'],
    answer: NP, tolerance: 0,
    whyNot: { 3: 'The top three hold 0.80, short of ' + PP + '.', 5: 'After four tokens the total is already 0.90.', 6: 'Top-p stops as soon as the total reaches p.', '*': 'Add the probabilities from the top until the total reaches ' + PP + '.' },
    hints: ['Add from the top: 0.40, then 0.65, …', '0.40 + 0.25 + 0.15 = 0.80, still below ' + PP + '.', 'With the fourth token the total is 0.90 ≥ ' + PP + ': ' + NP + ' tokens.'],
    solution: String(NP), wrong: ['3', '5'] });

  const cutRef = (p, k, q) => TL.cut(p, { topK: k, topP: q }).probs;
  const cutSol = 'function topKTopP(probs, k, p) {\n  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b);\n  const keep = new Set();\n  let total = 0;\n  for (const i of order.slice(0, k)) {\n    if (total >= p - 1e-12) break;\n    keep.add(i);\n    total += probs[i];\n  }\n  const sum = [...keep].reduce((s, i) => s + probs[i], 0);\n  return probs.map((v, i) => (keep.has(i) ? v / sum : 0));\n}';
  C.exercise({ id: 'll-x-code-cut', chapter: 'll-generate', title: 'Write top-k and top-p together', level: 3, kind: 'code',
    text: ['Write `topKTopP(probs, k, p)`: a next-token distribution, k ≥ 1 and 0 < p ≤ 1. Sort the tokens from most to least likely (on a tie the lower index first) and keep the first k. Renormalise those k so they add up to 1, then walk down them adding their renormalised probabilities, and keep the fewest from the top whose total reaches at least p. Return a list of the same length as `probs`: the kept tokens renormalised to add up to 1, every other token 0.'],
    fn: 'topKTopP', params: ['probs', 'k', 'p'], start: 'function topKTopP(probs, k, p) {\n  \n}', tolerance: 1e-9,
    ref: cutRef,
    cases: [[PR, 3, 1], [PR, 6, 0.85], [PR, 3, 0.7], [PR, 1, 1], [[0.1, 0.2, 0.3, 0.4], 4, 0.5], [[0.25, 0.25, 0.25, 0.25], 2, 1], [[0.5, 0.3, 0.2], 2, 0.9], [[0.05, 0.6, 0.05, 0.3], 3, 0.95]],
    hints: ['Sort indices by probability, highest first; the first k survive top-k.', 'Renormalise those k, then walk down them adding the renormalised values; stop once the sum reaches p.', 'function topKTopP(probs, k, p) {\n  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b).slice(0, k);\n  const tk = order.reduce((s, i) => s + probs[i], 0), keep = new Set();\n  let acc = 0;\n  for (const i of order) { keep.add(i); acc += probs[i] / tk; if (acc >= p - 1e-12) break; }\n  const sum = [...keep].reduce((s, i) => s + probs[i], 0);\n  return probs.map((v, i) => (keep.has(i) ? v / sum : 0));\n}'],
    solution: 'function topKTopP(probs, k, p) {\n  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b).slice(0, k);\n  const tk = order.reduce((s, i) => s + probs[i], 0), keep = new Set();\n  let acc = 0;\n  for (const i of order) { keep.add(i); acc += probs[i] / tk; if (acc >= p - 1e-12) break; }\n  const sum = [...keep].reduce((s, i) => s + probs[i], 0);\n  return probs.map((v, i) => (keep.has(i) ? v / sum : 0));\n}',
    wrong: [cutSol, 'function topKTopP(probs, k, p) {\n  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b).slice(0, k);\n  const keep = new Set(order);\n  const sum = order.reduce((s, i) => s + probs[i], 0);\n  return probs.map((v, i) => (keep.has(i) ? v / sum : 0));\n}'] });

  const KP = 20, KN = 10, NOC = Array.from({ length: KN }, (_, j) => KP + j).reduce((a, b) => a + b, 0), WC = KP + KN - 1;
  C.exercise({ id: 'll-x-kv-count', chapter: 'll-generate', title: 'Positions without a cache', level: 2, kind: 'number',
    text: ['A prompt of ' + KP + ' tokens, then ' + KN + ' new tokens generated one at a time. Without a KV cache, every step runs all the positions so far through the blocks (the first step runs the ' + KP + ' prompt positions, the next ' + (KP + 1) + ', …). How many positions are run in all? (With a cache it is ' + KP + ' for the first step and 1 for each later one: ' + WC + '.)'],
    answer: NOC, tolerance: 0,
    whyNot: { [KP + KN]: 'That is the length of the finished text; every step reruns everything before it.', [WC]: 'That is the count with a cache.', [KN * (KP + KN)]: 'The steps grow from ' + KP + ' to ' + (KP + KN - 1) + ' positions; not every step runs ' + (KP + KN) + '.', '*': 'Add ' + KP + ' + ' + (KP + 1) + ' + … + ' + (KP + KN - 1) + '.' },
    hints: ['Step 1 runs ' + KP + ' positions, step 2 runs ' + (KP + 1) + ', step ' + KN + ' runs ' + (KP + KN - 1) + '.', 'That is ' + KN + ' numbers, from ' + KP + ' to ' + (KP + KN - 1) + ': their mean is ' + (KP + (KN - 1) / 2) + '.', KN + ' × ' + (KP + (KN - 1) / 2) + ' = ' + NOC + ' positions, against ' + WC + ' with a cache.'],
    solution: String(NOC), wrong: [String(KP + KN), String(WC)] });

  const GP = S.DL.gptPresets.webgpu, ML_ = GP.model.layers * 2 * GP.context * GP.model.d;
  C.exercise({ id: 'll-x-kv-memory', chapter: 'll-generate', title: 'The memory a full cache holds', level: 2, kind: 'number',
    text: ['The larger GPT the text lab trains on a graphics card has ' + GP.model.layers + ' blocks, width ' + GP.model.d + ' and a context of ' + GP.context + ' tokens. Each block keeps a key and a value, each ' + GP.model.d + ' numbers, for every position. How many numbers does the cache hold when the context is full?'],
    answer: ML_, tolerance: 0,
    whyNot: { [ML_ / 2]: 'Each position keeps both a key and a value.', [ML_ / GP.model.layers]: 'Every block keeps its own keys and values.', '*': 'blocks × 2 × positions × width.' },
    hints: ['What is kept: for each block, for each position, a key and a value.', 'Each key and each value has ' + GP.model.d + ' numbers.', GP.model.layers + ' × 2 × ' + GP.context + ' × ' + GP.model.d + ' = ' + ML_.toLocaleString('en') + ' numbers, for one sequence.'],
    solution: String(ML_), wrong: [String(ML_ / 2), String(ML_ / GP.model.layers)] });

  // =========================================================================================================
  // Chapter 4: from a tiny model to a large one
  // =========================================================================================================
  const SG = { pretraining: 1, 'instruction tuning': 2, 'preference tuning': 3 }, SR = ['instruction tuning', 'preference tuning', 'pretraining'];
  const sc = Object.fromEntries(SR.map((r) => [r + '|stage', SG[r]]));
  C.exercise({ id: 'll-x-stages', chapter: 'll-beyond', title: 'Put the stages in order', level: 1, kind: 'table',
    text: ['Number the three training stages of an assistant model in the order they are applied, 1 first.'],
    rows: SR, cols: ['stage'], corner: '', cells: sc,
    hints: ['Which stage gives the model its knowledge of language at all?', 'Pretraining comes first; preference tuning polishes behaviour at the end.', 'pretraining 1, instruction tuning 2, preference tuning 3. Instruction tuning is a fine-tuning: on many tasks, written as requests.'],
    solution: str(sc), wrong: [Object.assign(str(sc), { 'pretraining|stage': '4', 'preference tuning|stage': '1' })] });

  const BETA = 0.5, DA = 2, DB = -3, MG = BETA * (DA - DB), PL = ln(1 + ex(-MG));
  C.exercise({ id: 'll-x-pref-loss', chapter: 'll-beyond', title: 'The preference loss after some tuning', level: 3, kind: 'number',
    text: ['Preference tuning with β = ' + BETA + '. Compared with the frozen original, the tuned model now gives the chosen answer A a log-probability ' + DA + ' higher, and the rejected answer B one 3 lower. The loss is −ln σ(β · ((log p(A) − log p(B)) − (the same under the original))). What is it? Three decimal places.'],
    answer: r3(PL), tolerance: 0.002,
    whyNot: { '0.693': 'That is the loss at the start, when the margin is 0.', [f3(ln(1 + ex(-BETA * (DA + DB))))]: 'B fell by 3, so the difference grew by 2 − (−3) = 5, not 2 + (−3).', [f3(ln(1 + ex(-(DA - DB))))]: 'Multiply the margin by β = ' + BETA + ' first.', '*': 'margin = β × (2 − (−3)); loss = ln(1 + e^(−margin)).' },
    hints: ['The change in log p(A) − log p(B) compared with the original is 2 − (−3).', 'margin = ' + BETA + ' × 5 = ' + MG + '; σ(' + MG + ') = ' + f3(1 / (1 + ex(-MG))) + '.', '−ln ' + f3(1 / (1 + ex(-MG))) + ' = ln(1 + e^−' + MG + ') = ' + f3(PL) + '.'],
    solution: f3(PL), wrong: ['0.693', f3(ln(1 + ex(-BETA * (DA + DB))))] });

  const iOpts = ['Guess: no digit stands out, and it is right about one time in ten', 'Answer correctly almost always, as for a repeated key', 'Answer with the most common digit of its training data, always right', 'Copy the digit of the previous pair'];
  C.exercise({ id: 'll-x-icl', chapter: 'll-beyond', title: 'A key with nothing to copy', level: 1, kind: 'choice',
    text: ['The model of [In-context learning on a toy task](#topic=lm-in-context) learned to answer a repeated key by finding its earlier pair. In “b9 b?” it answers 9. What does it do in “a1 b2 c3 d?”, where “d” has not appeared?'],
    options: iOpts, answer: 0,
    whyNot: { 1: 'It answers repeated keys right 99% of the time; for new keys there is nothing to copy.', 2: 'Every training sequence gave the letters new random digits, so no digit is “the” answer for d.', 3: 'Its attention looks for the same key, not for the nearest pair.' },
    hints: ['What did it learn: facts about letters, or a way to look things up?', 'It learned to look back for the same key and copy its digit.', 'With no earlier “d” it spreads its probability: its top digit gets 0.12, and over many such questions it is right 10% of the time, as chance would be.'],
    solution: 0, wrong: [1, 2] });

  const AQ = [[1, 0], [0, 1]], AK = [[2, 0], [0, 2], [1, 1]], DK = 2;
  const AW = AQ.map((q) => ML.softmax(AK.map((k) => (q[0] * k[0] + q[1] * k[1]) / Math.sqrt(DK))).map(r2));
  C.exercise({ id: 'll-x-attn', chapter: 'll-beyond', title: 'Where a query looks', level: 2, kind: 'matrix',
    text: ['A tiny attention head with d = 2. Two queries, q₁ = (1, 0) asking for key “a” and q₂ = (0, 1) asking for key “c”, meet three keys: k₁ = (2, 0) for an earlier “a”, k₂ = (0, 2) for an earlier “c”, and k₃ = (1, 1) for something in between. Each weight is the softmax, along the row, of q · k / √2. Two places.'],
    rows: ['q₁ (“a?”)', 'q₂ (“c?”)'], cols: ['k₁ (“a”)', 'k₂ (“c”)', 'k₃'], corner: 'query ↓  key →', answer: AW, tolerance: 0.011, rowsSumTo: 1,
    hints: ['The scores: q₁ · k = (2, 0, 1) and q₂ · k = (0, 2, 1).', 'Divide by √2 = 1.414: (1.414, 0, 0.707) for q₁; then the softmax of each row.', 'q₁: ' + AW[0].join(', ') + '; q₂: ' + AW[1].join(', ') + '. Each query looks most at its own key, so the value stored there is what gets copied.'],
    solution: AW.map((r) => r.map(String)), wrong: [AQ.map((q) => ML.softmax(AK.map((k) => q[0] * k[0] + q[1] * k[1])).map((v) => String(r2(v)))), [AW[1].map(String), AW[0].map(String)]] });

  const lOpts = ['It writes the likeliest continuation with no step that checks what it knows, so invented text reads as fluently as true text', 'It looks the king up and finds nothing, then guesses', 'It refuses, because the word is unknown', 'It copies a sentence about a king from its text'];
  C.exercise({ id: 'll-x-limits', chapter: 'll-beyond', title: 'Why the model invents', level: 1, kind: 'choice',
    text: ['The harbour text never mentions a king. Asked to continue “the king said that ”, the tiny GPT writes fluent-looking words anyway. Why?'],
    options: lOpts, answer: 0,
    whyNot: { 1: 'A plain language model has nothing to look up in: only its weights.', 2: 'Every letter of “king” is in its vocabulary; it has no notion of a word being unknown, and nothing makes it stop.', 3: 'There is no sentence about a king in the text to copy.' },
    hints: ['What is the model trained to do, every single step?', 'It is trained to give likely next characters, not to check facts.', 'So it continues as it always does: likely letters, likely words. Its own text even scores 0.45 per character, not far from the 0.66 of a sentence it read, and nothing in the output marks the difference.'],
    solution: 0, wrong: [1, 2] });

  const HE = 0.415, SHE = 0.118, SHARE = SHE / (HE + SHE);
  C.exercise({ id: 'll-x-bias-share', chapter: 'll-beyond', title: 'How much of the skew survives', level: 2, kind: 'number',
    text: ['After “the weaver said that ” the harbour text has “she” 6 times and “he” once: “she” takes 6 / 7 = 0.86 of the two. The tiny GPT gives “he” 0.415 and “she” 0.118. What share of the two does the model give “she”? Two decimal places.'],
    answer: r2(SHARE), tolerance: 0.005,
    whyNot: { '0.86': 'That is the share in the text, after this prompt; take the model’s two probabilities.', '0.12': 'That is p(“she”) itself; divide by p(“he”) + p(“she”).', '0.28': 'Divide by the sum of both, not by p(“he”).', '*': '0.118 / (0.415 + 0.118).' },
    hints: ['A share of the two: p(she) / (p(he) + p(she)).', '0.415 + 0.118 = 0.533.', '0.118 / 0.533 = ' + f2(SHARE) + ', against 0.86 in the text: the model follows the text’s overall habit, where “he” is commoner, and the minority case is erased.'],
    solution: f2(SHARE), wrong: ['0.86', '0.12'] });

  const dOpts = ['Test sets that probe made-up facts and skews, run before and after every change', 'Give the model sources to quote and show them, so claims can be checked', 'Balance the training and tuning data, and train it to say when it does not know', 'Raise the temperature, so answers vary more', 'Make the answers longer and more confident'];
  C.exercise({ id: 'll-x-defence', chapter: 'll-beyond', title: 'Which steps defend against these limits?', level: 1, kind: 'choice',
    text: ['A team deploys a language model and worries about invented facts and copied bias. Which of these steps help? Choose all that apply.'],
    options: dOpts, answer: [0, 1, 2],
    whyNot: { 3: 'A higher temperature makes the text vary more; it does not make it truer or fairer.', 4: 'Confidence in the wording is exactly what hides an invented fact.' },
    hints: ['Which steps change what the model knows or let someone check it?', 'Measuring, grounding in sources, and better data all help; none is finished once.', 'The first three. Temperature and confident wording change the style, not the truth.'],
    solution: [0, 1, 2], wrong: [[0, 1, 2, 3], [3, 4], [0]] });

  // the right option is written first above; each choice is turned round so the answers do not always sit first
  function rot(id, k) {
    const x = C.exercises.get(id), n = x.options.length, to = (i) => (i + k) % n, map = (a) => (Array.isArray(a) ? a.map(to).sort((p, q) => p - q) : to(a));
    const opts = new Array(n); x.options.forEach((o, i) => { opts[to(i)] = o; }); x.options = opts;
    x.answer = map(x.answer); x.solution = map(x.solution); x.wrong = (x.wrong || []).map(map);
    if (x.whyNot && typeof x.whyNot === 'object') x.whyNot = Object.fromEntries(Object.entries(x.whyNot).map(([i, w]) => [String(to(+i)), w]));
  }
  rot('ll-x-greedy', 3); rot('ll-x-icl', 2); rot('ll-x-limits', 1); rot('ll-x-defence', 2);
})(globalThis.Nooral = globalThis.Nooral || {});
