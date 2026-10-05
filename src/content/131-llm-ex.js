// Part 13, language models: the exercises. Hand computations are worked out here with the engines the topics use
// (S.DL.bpe for merges and cutting, the softmax of S.ML, the sampling rule S.TL.cut of the text lab, the stored runs of
// S.P13D, the n-gram counts of S.P13C in 130-llm.js); figures quoted from the topics were printed by the topics
// themselves. Two exercises at the end of each of the first two chapters are small model challenges: a tokenizer and a
// counting model that must reach a target on the held-out tenth of the harbour text within a budget.
(function (S) {
  'use strict';
  const C = S.Content, B = S.DL && S.DL.bpe, P = S.P13, Q = S.P13C;
  if (!C || !B || !P || !Q || C.exercises.has('ll-x-bpe-first')) return;
  const ML = S.ML, ln = Math.log, ex = Math.exp;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const f2 = (v) => r2(v).toFixed(2), f3 = (v) => r3(v).toFixed(3);
  const sh = (t) => String(t).replace(/ /g, '·');
  const merges = (text, n) => B.train(text, n).trace.filter((s) => s.kind === 'merge');

  // =========================================================================================================
  // Chapter 1: tokens
  // =========================================================================================================
  const T1 = 'tea for ten, then tea for three.', M1 = merges(T1, 1)[0];
  const p1 = (p) => '“' + sh(p[0]) + '” + “' + sh(p[1]) + '”';
  const o1 = [p1(M1.pair), '“t” + “e”', '“e” + “a”', '“t” + “h”'];
  C.exercise({ id: 'll-x-bpe-first', chapter: 'll-tokens', title: 'The first merge', level: 1, kind: 'choice',
    text: ['Byte-pair encoding is trained on the text `' + T1 + '`. Pairs are counted inside each chunk: a word with the space in front of it (· marks a space), or a run of punctuation. Which pair becomes the first merge?'],
    options: o1, answer: 0,
    whyNot: { 1: '“te” stands in “tea”, “ten” and “tea”: 3 times. Another pair stands 4 times.', 2: '“ea” stands only in the two “tea”s.', 3: '“th” stands in “then” and “three”: 2 times.' },
    hints: ['Write each chunk out: “tea”, “·for”, “·ten”, “,”, “·then”, …', 'A space followed by “t” starts “·ten”, “·then”, “·tea” and “·three”.', p1(M1.pair) + ' stands ' + M1.count + ' times, more than any other pair, so it is merged first; the text falls from 32 tokens to ' + M1.tokens + '.'],
    solution: 0, wrong: [1, 3] });

  C.exercise({ id: 'll-x-bpe-tokens', chapter: 'll-tokens', title: 'How much shorter after one merge?', level: 1, kind: 'number',
    text: ['The harbour text of [Byte-pair encoding: learning merges from a text](#topic=bpe-build) has 7008 characters, so it starts as 7008 tokens. The first merge, “·” + “t”, stands in 390 places. How many tokens is the text after this merge?'],
    answer: 6618, tolerance: 0,
    whyNot: { 7008: 'Every place the pair stood now holds one token instead of two.', 6228: 'Each place saves one token, not two: the pair becomes one token.', 7398: 'Merging makes the text shorter, not longer.', '*': 'Each of the 390 places saves exactly one token.' },
    hints: ['Two tokens become one wherever the pair stands.', 'So each of the 390 places saves one token.', '7008 − 390 = 6618 tokens.'],
    solution: '6618', wrong: ['7008', '6228'] });

  const T3 = 'the pot, the pan, the pen and ten hens.', M3 = merges(T3, 3), start3 = Array.from(T3).length;
  const c3 = {}; M3.forEach((m, i) => { c3['merge ' + (i + 1) + '|times merged'] = m.count; c3['merge ' + (i + 1) + '|tokens after'] = m.tokens; });
  C.exercise({ id: 'll-x-bpe-table', chapter: 'll-tokens', title: 'Three merges by hand', level: 2, kind: 'table',
    text: ['Run byte-pair encoding by hand on `' + T3 + '`, ' + start3 + ' characters, so ' + start3 + ' tokens. Chunks are a word with the space in front of it, or a run of punctuation; pairs never cross a chunk. When two pairs are equally common, the one that sorts first wins (a space sorts before every letter). For each of the first three merges, write how many places the pair stood in and how many tokens the text has after it.'],
    rows: ['merge 1', 'merge 2', 'merge 3'], cols: ['times merged', 'tokens after'], corner: '', cells: c3,
    hints: ['Count “h” + “e”: it stands in every “the” and in “hens”.', 'Merge 1 is ' + p1(M3[0].pair) + ' (' + M3[0].count + ' times). Then ' + p1(M3[1].pair) + ' and ' + p1(M3[2].pair) + ' tie at ' + M3[1].count + '; “·p” sorts first.', M3.map((m, i) => 'Merge ' + (i + 1) + ': ' + p1(m.pair) + ', ' + m.count + ' places, ' + m.tokens + ' tokens left').join('; ') + '.'],
    solution: Object.fromEntries(Object.entries(c3).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(c3).map(([k, v]) => [k, String(v)])), { 'merge 1|tokens after': String(start3 - 2 * M3[0].count) })] });

  const tOpts = ['The vocabulary grows by one token and the texts get shorter, but the model needs a row of weights for every token', 'The vocabulary shrinks and the texts get longer', 'Nothing changes after the first few merges', 'The texts get shorter and nothing is paid for it'];
  C.exercise({ id: 'll-x-bpe-trade', chapter: 'll-tokens', title: 'What more merges buy and cost', level: 1, kind: 'choice',
    text: ['A tokenizer is trained with 1,000 merges instead of 100. What changes?'],
    options: tOpts, answer: 0,
    whyNot: { 1: 'Each merge adds a token to the vocabulary and can only join pieces, never split them.', 2: 'Each merge removes the pair everywhere it stands: the text keeps getting shorter, more slowly.', 3: 'Every token needs its own embedding row and output score, so a larger vocabulary means a larger model.' },
    hints: ['What does one merge add, and what does it remove?', 'Each merge: one more token in the vocabulary, fewer tokens in the text.', 'Shorter texts mean fewer steps per text, but every token needs its own row in the embedding and its own output score.'],
    solution: 0, wrong: [1, 3] });

  const mergeSol = 'function mergePair(tokens, a, b) {\n  const out = [];\n  for (let i = 0; i < tokens.length; i++) {\n    if (i + 1 < tokens.length && tokens[i] === a && tokens[i + 1] === b) { out.push(a + b); i++; }\n    else out.push(tokens[i]);\n  }\n  return out;\n}';
  const mergeRef = (t, a, b) => { const o = []; for (let i = 0; i < t.length; i++) { if (i + 1 < t.length && t[i] === a && t[i + 1] === b) { o.push(a + b); i++; } else o.push(t[i]); } return o; };
  C.exercise({ id: 'll-x-code-merge', chapter: 'll-tokens', title: 'Write one merge', level: 2, kind: 'code',
    text: ['Write `mergePair(tokens, a, b)`: a list of tokens (strings) and a pair to merge. Return a new list in which every place where `a` is directly followed by `b` holds the single token `a + b`. Go from left to right, and do not reuse a token already merged: merging “a” + “a” in [“a”, “a”, “a”] gives [“aa”, “a”].'],
    fn: 'mergePair', params: ['tokens', 'a', 'b'], start: 'function mergePair(tokens, a, b) {\n  \n}',
    ref: mergeRef,
    cases: [[[' ', 't', 'h', 'e'], ' ', 't'], [['t', 'h', 'e', ' ', 't', 'h', 'e', 'n'], 'h', 'e'], [['a', 'a', 'a'], 'a', 'a'], [['a', 'a', 'a', 'a'], 'a', 'a'], [['x', 'y'], 'y', 'x'], [[' t', 'he', ' t', 'en'], ' t', 'he'], [[], 'a', 'b']],
    hints: ['Walk through the list with an index, building a new list.', 'When tokens[i] is a and tokens[i + 1] is b, push a + b and skip one extra place.', mergeSol],
    solution: mergeSol,
    wrong: ['function mergePair(tokens, a, b) {\n  return tokens.join(\'\').split(a + b).join(a + b).split(\'\');\n}', 'function mergePair(tokens, a, b) {\n  const out = [];\n  for (let i = 0; i < tokens.length; i++) {\n    if (tokens[i] === a && tokens[i + 1] === b) out.push(a + b);\n    else out.push(tokens[i]);\n  }\n  return out;\n}'] });

  const RUN = P.bpeRun('harbour', 60), W = 'children', WP = RUN.tokenizer.split(W), shown30 = RUN.merges.slice(0, 30);
  const normPieces = (s) => String(s).split('|').map((x) => x.trim().replace(/^[·_]/, ' ')).filter((x) => x !== '');
  C.exercise({ id: 'll-x-cut-word', chapter: 'll-tokens', title: 'Cut a word with learned merges', level: 2, kind: 'string',
    text: ['These are the first 30 merges learned on the harbour text, in order (· is a space):', { math: shown30.map((m, i) => (i + 1) + '. ' + sh(m[0]) + ' + ' + sh(m[1])).reduce((rows, s, i) => { if (i % 3 === 0) rows.push(''); rows[rows.length - 1] += s.padEnd(22); return rows; }, []).map((r) => r.trimEnd()).join('\n') }, 'Cut the word `' + W + '` (at the start of a text, so with no space before it) by applying these merges in order. Write the pieces separated by |, like `t | he`.'],
    placeholder: 'c | h | …',
    accept: (s) => { const g = normPieces(s); if (g.join('') !== W) return 'The pieces must spell “' + W + '” exactly, in order.'; return g.join('|') === WP.join('|') ? true : g.length === W.length ? 'That is the word as single characters: some merges in the list fit it.' : 'Not quite: replay the merges in their order, each one everywhere it fits.'; },
    hints: ['Start from single letters: c | h | i | l | d | r | e | n.', 'Go down the list: only merges whose two pieces stand side by side in the word apply. “r” + “e” is one; look for another.', 'Merge 17, “r” + “e”, and merge 28, “l” + “d”, apply: ' + WP.join(' | ') + '.'],
    solution: WP.join(' | '), wrong: [Array.from(W).join(' | '), 'chi | ld | ren'] });

  const uOpts = ['It becomes the unknown token: the tokenizer has no piece for it', 'It is turned into the same letter in lower case', 'It is dropped from the text', 'It gets a new token on the spot'];
  C.exercise({ id: 'll-x-unknown', chapter: 'll-tokens', title: 'A character never seen', level: 1, kind: 'choice',
    text: ['The tokenizer of [Cutting a sentence into tokens](#topic=bpe-tokenise) was trained on the harbour text, which is all lower case. What happens to the “K” in “The Keeper climbs.”?'],
    options: uOpts, answer: 0,
    whyNot: { 1: 'The tokenizer knows nothing of upper and lower case; it only knows the pieces it learned.', 2: 'Nothing is dropped: every character gets an id, if only the unknown one.', 3: 'The vocabulary is fixed when training ends.' },
    hints: ['Where do the tokens come from?', 'The vocabulary is the 26 characters of the harbour text plus the merges.', '“K” is not among them, so it becomes the unknown token, id 0. Byte-level tokenizers avoid this: every text is a sequence of the 256 bytes.'],
    solution: 0, wrong: [1, 2] });

  // the tokenizer challenge: at most 40 merges, learned on the first nine tenths, cutting the last tenth short
  const HT = S.P13T.harbour, CUT = Q.cut(), HTR = HT.slice(0, CUT), HVA = HT.slice(CUT), TRCH = B.pretokenize(HTR);
  const BASE = [...new Set(HTR)].sort(), BUDGET = 40, TARGET_TOK = 425;
  const tokCount = (ms) => B.fromMerges(BASE, ms).encode(HVA).length;
  const bpeSol = 'function learnMerges(chunks, n) {\n  let words = chunks.map((c) => Array.from(c));\n  const merges = [];\n  for (let m = 0; m < n; m++) {\n    const count = new Map();\n    for (const w of words) for (let i = 0; i + 1 < w.length; i++) { const k = w[i] + \'\\u0000\' + w[i + 1]; count.set(k, (count.get(k) || 0) + 1); }\n    let best = null, bn = 0;\n    for (const [k, c] of count) if (c > bn || (c === bn && k < best)) { best = k; bn = c; }\n    if (!best) break;\n    const [a, b] = best.split(\'\\u0000\');\n    merges.push([a, b]);\n    words = words.map((w) => { const o = []; for (let i = 0; i < w.length; i++) { if (i + 1 < w.length && w[i] === a && w[i + 1] === b) { o.push(a + b); i++; } else o.push(w[i]); } return o; });\n  }\n  return merges;\n}';
  const bpeCheck = (got) => {
    if (!Array.isArray(got) || !got.every((m) => Array.isArray(m) && m.length === 2 && m.every((x) => typeof x === 'string' && x.length))) return 'it should be a list of pairs of strings, like [[" ", "t"], ["h", "e"]]';
    if (got.length > BUDGET) return 'that is ' + got.length + ' merges; the budget is ' + BUDGET;
    const n = tokCount(got);
    return n <= TARGET_TOK ? true : 'cut with these merges, the held-out text takes ' + n + ' tokens; the target is at most ' + TARGET_TOK;
  };
  C.exercise({ id: 'll-x-ch-bpe', chapter: 'll-tokens', title: 'Challenge: a tokenizer within 40 merges', level: 3, kind: 'code',
    text: ['The harbour town wants its text stored in as few tokens as possible, with a vocabulary that stays small. Write `learnMerges(chunks, n)`: `chunks` is the first nine tenths of the harbour text already cut into chunks (each word with the space in front of it, or a run of punctuation), and `n` is the budget of merges. Return the merges, a list of pairs like `[" ", "t"]`, in the order they should be applied.',
      'The check cuts the held-out last tenth of the text, ' + HVA.length + ' characters it never gives you, with your merges replayed in order, starting from the ' + BASE.length + ' characters of the text. **Target: at most ' + TARGET_TOK + ' tokens with at most ' + BUDGET + ' merges.** Any merges that reach it pass.'],
    fn: 'learnMerges', params: ['chunks', 'n'], start: 'function learnMerges(chunks, n) {\n  const merges = [];\n  \n  return merges;\n}', timeLimit: 4000,
    tests: [{ args: [TRCH, BUDGET], check: bpeCheck }],
    hints: ['Byte-pair encoding: count every pair of neighbours inside the chunks, merge the commonest everywhere, repeat.', 'Keep each chunk as an array of tokens; after choosing a pair, rebuild every chunk with the pair merged (as in [Write one merge](#exercise=ll-x-code-merge)). Use all ' + BUDGET + ' merges.', 'Plain byte-pair encoding with all ' + BUDGET + ' merges cuts the held-out text into ' + tokCount(merges(HTR, BUDGET).map((m) => m.pair)) + ' tokens: ' + bpeSol],
    solution: bpeSol,
    wrong: [bpeSol.replace('for (let m = 0; m < n; m++)', 'for (let m = 0; m < n / 2; m++)'), bpeSol.replace('for (let m = 0; m < n; m++)', 'for (let m = 0; m < n + 10; m++)'), 'function learnMerges(chunks, n) {\n  return [];\n}'] });

  // =========================================================================================================
  // Chapter 2: predicting the next token
  // =========================================================================================================
  const AT = Q.after('t'), ATH = AT.rows[0][1], VQ = Q.vocab().length, BIG = (ATH + 1) / (AT.total + VQ);
  C.exercise({ id: 'll-x-bigram', chapter: 'll-predict', title: 'A bigram probability with smoothing', level: 2, kind: 'number',
    text: ['In the harbour text, “t” is followed by another character ' + AT.total + ' times, and by “h” ' + ATH + ' times. The text uses ' + VQ + ' different characters. With add-one smoothing (1 added to the count of every possible next character), what is P(h | t)? Three decimal places.'],
    answer: r3(BIG), tolerance: 0.002,
    whyNot: { [f3(ATH / AT.total)]: 'That is (almost) the share without smoothing: add 1 to the count of “th”, and 1 for each of the ' + VQ + ' possible characters to the total.', '*': 'Add 1 to the count of “th”, and 1 for each of the ' + VQ + ' possible characters to the total.' },
    hints: ['Without smoothing it would be ' + ATH + ' / ' + AT.total + '.', 'Add-one adds 1 to every one of the ' + VQ + ' possible next characters, so the total after “t” grows by ' + VQ + '.', '(' + ATH + ' + 1) / (' + AT.total + ' + ' + VQ + ') = ' + (ATH + 1) + ' / ' + (AT.total + VQ) + ' = ' + f3(BIG) + '.'],
    solution: f3(BIG), wrong: [f3(ATH / AT.total), '0.5'] });

  const bgRef = (text, a, b, k) => { const V = new Set(text).size; let n = 0, m = 0; for (let i = 0; i + 1 < text.length; i++) if (text[i] === a) { m++; if (text[i + 1] === b) n++; } return (n + k) / (m + k * V); };
  const bgSol = 'function bigramProb(text, a, b, k) {\n  const V = new Set(text).size;\n  let pair = 0, from = 0;\n  for (let i = 0; i + 1 < text.length; i++) {\n    if (text[i] === a) { from++; if (text[i + 1] === b) pair++; }\n  }\n  return (pair + k) / (from + k * V);\n}';
  C.exercise({ id: 'll-x-code-bigram', chapter: 'll-predict', title: 'Write a smoothed bigram model', level: 2, kind: 'code',
    text: ['Write `bigramProb(text, a, b, k)`: the probability that character `b` comes next after character `a`, counted in `text` with add-k smoothing. Count the places where `a` is followed by any character (an `a` at the very end has nothing after it), and among them those where it is followed by `b`. V is the number of different characters in the text. Return (count of a b + k) / (count of a followed by anything + k · V).'],
    fn: 'bigramProb', params: ['text', 'a', 'b', 'k'], start: 'function bigramProb(text, a, b, k) {\n  \n}', tolerance: 1e-9,
    ref: bgRef,
    cases: [['the tea', 't', 'h', 0], ['the tea', 't', 'e', 1], ['abab', 'b', 'a', 0], ['abab', 'b', 'a', 1], ['aaaa', 'a', 'a', 0.5], ['the quay', 'q', 'u', 0.1], [Q.text(), 't', 'h', 1], [Q.text(), 'q', 'z', 0.1]],
    hints: ['Loop over the positions i where text[i] has a next character, i + 1 < text.length.', 'Count from (text[i] is a) and pair (also text[i + 1] is b); V = new Set(text).size.', bgSol],
    solution: bgSol,
    wrong: ['function bigramProb(text, a, b, k) {\n  let pair = 0, from = 0;\n  for (let i = 0; i + 1 < text.length; i++) if (text[i] === a) { from++; if (text[i + 1] === b) pair++; }\n  return (pair + k) / (from + k);\n}', 'function bigramProb(text, a, b, k) {\n  const V = new Set(text).size;\n  let pair = 0, from = 0;\n  for (let i = 0; i < text.length; i++) if (text[i] === a) { from++; if (text[i + 1] === b) pair++; }\n  return (pair + k) / (from + k * V);\n}'] });

  const NG4 = Q.ngram(4, Q.K), NG6 = Q.ngram(6, Q.K), NG3 = Q.ngram(3, Q.K);
  const nOpts = ['Most of the held-out 5-character contexts never appear in the training text, so their counts say nothing', 'Longer contexts make the model forget the common letters', 'Smoothing stops working for long contexts', 'The 6-gram model has too few table entries'];
  C.exercise({ id: 'll-x-ngram-best', chapter: 'll-predict', title: 'Why a longer context does worse', level: 2, kind: 'choice',
    text: ['Counting models on the harbour text (the lesson [Counting: the n-gram model](#lesson=ll-ngram)): with 2 characters of context the held-out loss is ' + f3(NG3.ce) + ' nats a character, with 5 it is ' + f3(NG6.ce) + '. Why does the longer context do worse?'],
    options: nOpts, answer: 0,
    whyNot: { 1: 'A longer context does not remove anything a shorter one knew; the counts just become too sparse.', 2: 'Smoothing still keeps every probability above zero; it cannot invent what was never counted.', 3: 'It has far more: ' + NG6.table.toLocaleString('en') + ', almost all of them empty.' },
    hints: ['Look at the column of contexts never seen in training.', Math.round(100 * NG4.share) + '% of the held-out 3-character contexts were never seen; for 5 characters it is ' + Math.round(100 * NG6.share) + '%.', 'An unseen context leaves only the smoothing: a near-uniform guess, ln 26 ≈ 3.26 nats. The more such contexts, the worse the model, however good its counts elsewhere.'],
    solution: 0, wrong: [1, 3] });

  const SC = [1.2, 0.4, -0.3, -1.0], SCP = ML.softmax(SC);
  C.exercise({ id: 'll-x-softmax', chapter: 'll-predict', title: 'From scores to a probability', level: 1, kind: 'number',
    text: ['A model that knows only four characters gives the scores h: 1.2, s: 0.4, t: −0.3 and w: −1.0. What probability does the softmax give “h”? Three decimal places.'],
    answer: r3(SCP[0]), tolerance: 0.002,
    whyNot: { [f3(1.2 / (1.2 + 0.4))]: 'Scores are not probabilities: take e to each score first.', '0.25': 'That would be four equal scores.', '*': 'p(h) = e^1.2 / (e^1.2 + e^0.4 + e^−0.3 + e^−1.0).' },
    hints: ['The softmax: p_i = e^(z_i) / Σ e^(z_j).', 'e^1.2 = ' + f3(ex(1.2)) + ', e^0.4 = ' + f3(ex(0.4)) + ', e^−0.3 = ' + f3(ex(-0.3)) + ', e^−1.0 = ' + f3(ex(-1)) + '.', f3(ex(1.2)) + ' / ' + f3(SC.reduce((s, z) => s + ex(z), 0)) + ' = ' + f3(SCP[0]) + '.'],
    solution: f3(SCP[0]), wrong: ['0.25', f3(1.2 / 1.6)] });

  C.exercise({ id: 'll-x-cost', chapter: 'll-predict', title: 'The cost of one guess', level: 1, kind: 'number',
    text: ['At one position the true next character is “s”, and the model gave it p = 0.25. What does this position add to the cross-entropy, in nats? Three decimal places.'],
    answer: r3(-ln(0.25)), tolerance: 0.002,
    whyNot: { '0.25': 'That is the probability; the cost is −ln p.', '0.75': 'The cost is −ln p, not 1 − p.', '-1.386': 'The cost is minus the logarithm, so it is positive.', '0.602': 'Use the natural logarithm: nats, not base-10 digits.' },
    hints: ['Each position costs −ln p of the true next token.', '−ln 0.25 = ln 4.', 'ln 4 = ' + f3(ln(4)) + ' nats.'],
    solution: f3(ln(4)), wrong: ['0.25', '0.75', '0.602'] });

  C.exercise({ id: 'll-x-perplexity', chapter: 'll-predict', title: 'Perplexity of a shuffled sentence', level: 1, kind: 'number',
    text: ['In [Cross-entropy and perplexity](#topic=lm-cross-entropy), the shuffled sentence “ropes the coils sailor the.” costs the trained model 36.00 nats over its 26 scored positions. What is the perplexity? Two decimal places.'],
    answer: r2(ex(36 / 26)), tolerance: 0.02,
    whyNot: { [f3(36 / 26)]: 'That is the cross-entropy; perplexity is e to that power.', '36': 'Divide by the 26 positions first.', '*': 'Perplexity = e^(36.00 / 26).' },
    hints: ['Cross-entropy is the mean cost.', '36.00 / 26 = ' + f3(36 / 26) + ' nats a character.', 'e^' + f3(36 / 26) + ' = ' + f2(ex(36 / 26)) + ': as unsure as choosing among about 4 characters.'],
    solution: f2(ex(36 / 26)), wrong: [f3(36 / 26), '36'] });

  const CP = [0.52, 0.95, 0.95, 0.2], CC = CP.map((p) => -ln(p)), CM = CC.reduce((a, b) => a + b, 0) / CC.length;
  const ceRows = ['“h” after “t”', '“e” after “th”', '· after “the”', '“s” after “the·”'], ceCells = {};
  ceRows.forEach((r, i) => { ceCells[r + '|p'] = String(CP[i]); ceCells[r + '|cost'] = r3(CC[i]); });
  ceCells['cross-entropy|cost'] = r3(CM); ceCells['perplexity|cost'] = r3(ex(CM));
  const ceSol = Object.fromEntries(Object.entries(ceCells).filter(([k]) => k.endsWith('|cost')).map(([k, v]) => [k, String(v)]));
  C.exercise({ id: 'll-x-ce-table', chapter: 'll-predict', title: 'Cross-entropy of four guesses', level: 2, kind: 'table',
    text: ['The first four scored positions of “the sailor coils the ropes.”, with the probability the trained model gave the true next character (rounded to two places). Fill in the cost of each, then the cross-entropy of the four and its perplexity. Within 0.01.', { table: [['position', 'p of the true next']].concat(ceRows.map((r, i) => [r, String(CP[i])])) }],
    rows: ceRows.concat(['cross-entropy', 'perplexity']), cols: ['cost'], corner: '', cells: Object.fromEntries(Object.entries(ceCells).filter(([k]) => k.endsWith('|cost'))), tolerance: 0.01,
    hints: ['Each cost is −ln p.', 'Costs: ' + CC.map(f3).join(', ') + '.', 'Their mean is ' + f3(CM) + ' nats, and e^' + f3(CM) + ' = ' + f3(ex(CM)) + '.'],
    solution: ceSol, wrong: [Object.assign({}, ceSol, { 'perplexity|cost': f3(CM) })] });

  const ceRef = (probs, targets) => probs.reduce((s, row, i) => s - ln(row[targets[i]]), 0) / probs.length;
  const ceCode = 'function crossEntropy(probs, targets) {\n  let s = 0;\n  for (let i = 0; i < probs.length; i++) s -= Math.log(probs[i][targets[i]]);\n  return s / probs.length;\n}';
  C.exercise({ id: 'll-x-code-ce', chapter: 'll-predict', title: 'Write cross-entropy', level: 2, kind: 'code',
    text: ['Write `crossEntropy(probs, targets)`: `probs[i]` is the model’s probability for every token at position i, and `targets[i]` is the index of the token that really came next. Return the mean of −ln probs[i][targets[i]] over the positions, in nats.'],
    fn: 'crossEntropy', params: ['probs', 'targets'], start: 'function crossEntropy(probs, targets) {\n  \n}', tolerance: 1e-9,
    ref: ceRef,
    cases: [[[[0.25, 0.75]], [0]], [[[0.5, 0.5], [0.9, 0.1]], [1, 0]], [[[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [1 / 3, 1 / 3, 1 / 3]], [2, 2, 0]], [[[1, 0]], [0]], [CP.map((p) => [p, 1 - p]), [0, 0, 0, 0]]],
    hints: ['Pick, at each position, the probability of the token that really came.', 'Sum −Math.log of those, then divide by the number of positions.', ceCode],
    solution: ceCode,
    wrong: ['function crossEntropy(probs, targets) {\n  let s = 0;\n  for (let i = 0; i < probs.length; i++) s -= Math.log(probs[i][targets[i]]);\n  return s;\n}', 'function crossEntropy(probs, targets) {\n  let s = 0;\n  for (let i = 0; i < probs.length; i++) s -= Math.log10(probs[i][targets[i]]);\n  return s / probs.length;\n}'] });

  const GV = S.P13D.gpt.val.map(([s, v]) => [s, +v.toFixed(3)]), GE = GV[GV.length - 1][1], GTA = S.P13D.gpt.hist.slice(-25).reduce((a, b) => a + b, 0) / 25;
  C.exercise({ id: 'll-x-train-curve', chapter: 'll-predict', title: 'Sketch the held-out loss of the tiny GPT', level: 2, kind: 'predict-curve',
    text: ['The tiny GPT of [Training a tiny GPT](#topic=lm-gpt-training) starts at a held-out loss of ' + GV[0][1] + ' nats a character and trains for 1600 steps, ending at ' + GE + '. Sketch the held-out loss over the steps.'],
    plot: { xr: [0, 1600], yr: [1.2, 3.6], xname: 'training step', yname: 'held-out loss (nats)', given: [GV[0]], note: 'loss on held-out text, every 50 steps' },
    answer: GV, tolerance: 0.15, checks: ['falling'],
    hints: ['Where is the loss after 100 steps: nearer the start or nearer the end?', 'The easy patterns (common letters, short words) are learned first: the loss falls fast early, then slowly.', 'The run’s values: ' + GV.filter(([s]) => [0, 50, 100, 200, 400, 800, 1600].includes(s)).map(([s, v]) => 'step ' + s + ': ' + v).join(', ') + '.'],
    solution: GV, wrong: [GV.map(([s]) => [s, r3(GV[0][1] + ((GE - GV[0][1]) * s) / 1600)]), GV.map(([s]) => [s, GV[0][1]])] });

  const gOpts = ['Part of what it learned is particular to the training text: it has begun to memorise', 'The held-out text is in another language', 'The held-out loss is measured wrongly', 'Training loss is always higher than held-out loss'];
  C.exercise({ id: 'll-x-gap', chapter: 'll-predict', title: 'The gap between the two losses', level: 2, kind: 'choice',
    text: ['At the end of training the tiny GPT’s training batches average ' + f2(GTA) + ' nats a character, while held-out text costs it ' + GE + '. What does the gap mean?'],
    options: gOpts, answer: 0,
    whyNot: { 1: 'Both parts come from the same harbour text.', 2: 'It is the same cross-entropy on different characters.', 3: 'Usually the other way round: a model does better on what it trained on.' },
    hints: ['Which text did the weights see?', 'The training loss is measured on text the model was fitted to; the held-out loss on text it never saw.', 'The difference, ' + f2(GE - GTA) + ' nats, is what it memorised rather than learned. More text, dropout or stopping earlier narrow it.'],
    solution: 0, wrong: [1, 3] });

  const SR = S.P13D.scale.runs.map((r) => [r.params, r.val[r.val.length - 1][1]]);
  const lx = SR.map(([n]) => ln(n)), ly = SR.map(([, v]) => ln(v)), mx = lx.reduce((a, b) => a + b, 0) / lx.length, my = ly.reduce((a, b) => a + b, 0) / ly.length;
  const BSL = lx.reduce((s, x, i) => s + (x - mx) * (ly[i] - my), 0) / lx.reduce((s, x) => s + (x - mx) * (x - mx), 0), b3 = r3(BSL);
  const L48 = r3(SR[SR.length - 1][1]), N48 = SR[SR.length - 1][0], PRED = L48 * Math.pow(4, b3);
  C.exercise({ id: 'll-x-scaling', chapter: 'll-predict', title: 'Read off a scaling law', level: 3, kind: 'number',
    text: ['In [Scaling: bigger models, lower loss](#topic=lm-scaling) the held-out loss follows L ≈ A · N^b with b = ' + b3 + '. The widest model has N = ' + N48.toLocaleString('en') + ' parameters and L = ' + L48 + '. If the law held, what loss would a model with 4 times as many parameters reach? Three decimal places.'],
    answer: r3(PRED), tolerance: 0.004,
    whyNot: { [f3(L48 * Math.pow(2, b3))]: 'That is one doubling; 4 times the parameters is two doublings.', [f3(L48 * (1 + 4 * b3))]: 'The law multiplies by 4^b; it does not subtract.', [f3(L48)]: 'With b < 0, more parameters lower the loss.', '*': 'L_new = L · 4^b.' },
    hints: ['Under L = A · N^b, multiplying N by c multiplies L by c^b.', '4^' + b3 + ' = ' + f3(Math.pow(4, b3)) + '.', L48 + ' × ' + f3(Math.pow(4, b3)) + ' = ' + f3(PRED) + '. Only if the text and the training grew too: with this little text the law would soon break.'],
    solution: f3(PRED), wrong: [f3(L48 * Math.pow(2, b3)), f3(L48)] });

  const SK = SR.map(([n, v]) => [r2(n / 1000), r3(v)]);
  C.exercise({ id: 'll-x-scale-curve', chapter: 'll-predict', title: 'Sketch loss against parameters', level: 3, kind: 'predict-curve',
    text: ['Five widths of the same GPT, from ' + SR[0][0].toLocaleString('en') + ' to ' + N48.toLocaleString('en') + ' parameters, each trained 400 steps on the harbour text. The smallest ends at a held-out loss of ' + SK[0][1] + '. Sketch the final held-out loss against the parameters, in thousands.'],
    plot: { xr: [0, 62], yr: [1.6, 2.5], xname: 'parameters (thousands)', yname: 'held-out loss (nats)', given: [SK[0]], note: 'final held-out loss of each width' },
    answer: SK, tolerance: 0.06, checks: ['falling'],
    hints: ['More parameters, same training: does the loss rise or fall?', 'It falls, quickly at first and then more slowly: a power law with a small negative exponent.', 'The runs: ' + SK.map(([n, v]) => n + ' thousand: ' + v).join(', ') + '.'],
    solution: SK, wrong: [SK.map(([n]) => [n, SK[0][1]]), SK.map(([n]) => [n, r3(SK[0][1] + 0.004 * (n - SK[0][0]))])] });

  // the counting challenge: a model that sees only the two characters before each held-out character
  const CHV = Q.vocab(), K2 = 3, TARGET_CE = 1.8;
  const pos = []; for (let i = CUT + 1; i < HT.length; i++) pos.push(i);
  const g = S.T.rng(13); for (let i = pos.length - 1; i > 0; i--) { const j = g.int(i + 1); const t = pos[i]; pos[i] = pos[j]; pos[j] = t; }
  const CTXS = pos.map((i) => HT.slice(i - K2 + 1, i)), TGT = pos.map((i) => CHV.indexOf(HT[i]));
  const ngCheck = (got) => {
    const V = CHV.length, R = CTXS.length;
    if (!Array.isArray(got) || got.length !== R * V) return 'it should be one flat list of ' + R + ' × ' + V + ' = ' + R * V + ' probabilities' + (Array.isArray(got) && got.length === R ? ', not a list of rows' : '');
    let s = 0;
    for (let r = 0; r < R; r++) {
      const row = got.slice(r * V, r * V + V);
      if (!row.every((v) => typeof v === 'number' && isFinite(v) && v >= 0)) return 'row ' + r + ' (context “' + CTXS[r] + '”) holds something that is not a probability: ' + row.find((v) => !(typeof v === 'number' && isFinite(v) && v >= 0));
      const t = row.reduce((a, b) => a + b, 0);
      if (Math.abs(t - 1) > 1e-6) return 'row ' + r + ' (context “' + CTXS[r] + '”) adds up to ' + (+t.toPrecision(6)) + ', not 1';
      if (!(row[TGT[r]] > 0)) return 'row ' + r + ' (context “' + CTXS[r] + '”) gives the character that really came next probability 0: an infinite cost';
      s -= ln(row[TGT[r]]);
    }
    const ce = s / R;
    return ce <= TARGET_CE ? true : 'its held-out loss is ' + f3(ce) + ' nats a character; the target is at most ' + TARGET_CE;
  };
  const ngSol = 'function predict(train, vocab, contexts) {\n  const text = train.join(\'\'), V = vocab.length, k = 0.1;\n  const pair = new Map(), ctx = new Map();\n  for (let i = 2; i < text.length; i++) {\n    const c = text.slice(i - 2, i);\n    pair.set(c + text[i], (pair.get(c + text[i]) || 0) + 1);\n    ctx.set(c, (ctx.get(c) || 0) + 1);\n  }\n  return contexts.flatMap((c) => vocab.map((ch) => ((pair.get(c + ch) || 0) + k) / ((ctx.get(c) || 0) + k * V)));\n}';
  C.exercise({ id: 'll-x-ch-ngram', chapter: 'll-predict', title: 'Challenge: a counting model under 1.8 nats', level: 3, kind: 'code',
    text: ['Before training a network, the harbour town wants to know how far plain counting goes. Write `predict(train, vocab, contexts)`. `train` is the first nine tenths of the harbour text as a list of single characters; `vocab` is its ' + CHV.length + ' characters in order; `contexts` lists, in shuffled order, the **two characters before** each of the ' + CTXS.length + ' held-out characters. Return the probabilities as one flat list, row after row: for each context in turn, a probability for every character of `vocab`, in its order, adding up to 1. So the list has ' + CTXS.length + ' × ' + CHV.length + ' = ' + (CTXS.length * CHV.length) + ' numbers, and row r is items r · ' + CHV.length + ' to r · ' + CHV.length + ' + ' + (CHV.length - 1) + '.',
      'The check scores your rows against the characters that really came next, which you never see. **Target: a held-out cross-entropy of at most ' + TARGET_CE + ' nats a character**, with no more context than the two characters given.'],
    fn: 'predict', params: ['train', 'vocab', 'contexts'], start: 'function predict(train, vocab, contexts) {\n  const text = train.join(\'\');\n  \n}', timeLimit: 4000,
    tests: [{ args: [Array.from(HTR), CHV, CTXS], check: ngCheck }],
    hints: ['Count, in the training text, how often each two-character context is followed by each character.', 'A pair never seen gets probability 0 and an infinite cost: add a small k to every count. Try a few values of k; add-one (k = 1) smooths too much here.', 'A trigram model with add-' + Q.K + ' smoothing scores ' + f3(NG3.ce) + ': ' + ngSol],
    solution: ngSol,
    wrong: [ngSol.replace('const text = train.join(\'\'), V = vocab.length, k = 0.1;', 'const text = train.join(\'\'), V = vocab.length, k = 1;'), 'function predict(train, vocab, contexts) {\n  return contexts.flatMap(() => vocab.map(() => 1 / vocab.length));\n}', ngSol.replace('k = 0.1;', 'k = 0;')] });

  // the right option is written first above; each choice is turned round so the answers do not always sit first
  function rot(id, k) {
    const x = C.exercises.get(id), n = x.options.length, to = (i) => (i + k) % n, map = (a) => (Array.isArray(a) ? a.map(to).sort((p, q) => p - q) : to(a));
    const opts = new Array(n); x.options.forEach((o, i) => { opts[to(i)] = o; }); x.options = opts;
    x.answer = map(x.answer); x.solution = map(x.solution); x.wrong = (x.wrong || []).map(map);
    if (x.whyNot && typeof x.whyNot === 'object') x.whyNot = Object.fromEntries(Object.entries(x.whyNot).map(([i, w]) => [String(to(+i)), w]));
  }
  rot('ll-x-bpe-first', 2); rot('ll-x-bpe-trade', 1); rot('ll-x-unknown', 3); rot('ll-x-ngram-best', 2); rot('ll-x-gap', 1);
})(globalThis.Nooral = globalThis.Nooral || {});
