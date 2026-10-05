// Part 13, language models: the chapters and the lessons. The exercises are in 131-llm-ex.js; the tour, the course and
// the pages the diagram checker looks at are in 132-llm-ch.js. Each lesson opens with a question, runs the topic that
// shows the idea at a frame chosen for the point being made, asks for a prediction, sums up, and leads to its
// exercises. Every value quoted was printed by the topics themselves (the stored runs of S.P13D and the live models
// rebuilt from them), or is counted here from the part's own text (the n-gram tables, S.P13C below).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('ll-bpe')) return;
  const X = (id) => ({ exercise: id });

  // ---- n-gram models counted on the harbour text, shared with the exercises -----------------------------------------
  // The same split as the tiny GPT's run: the first nine tenths of the text train, the last tenth is held out. Each
  // held-out character (positions cut + 1 to the end, as the GPT is scored below) is predicted from the n − 1
  // characters before it, with add-k smoothing over the 26 characters of the text.
  const P13C = (S.P13C = S.P13C || {});
  P13C.text = () => S.P13T.harbour;
  P13C.cut = () => Math.floor(P13C.text().length * 0.9);
  P13C.vocab = () => [...new Set(P13C.text())].sort();
  /** counts of (context, next) and of contexts in a training text, for contexts of n − 1 characters */
  P13C.counts = (train, n) => {
    const pair = new Map(), ctx = new Map();
    for (let i = n - 1; i < train.length; i++) { const c = train.slice(i - n + 1, i), key = c + '\u0000' + train[i]; pair.set(key, (pair.get(key) || 0) + 1); ctx.set(c, (ctx.get(c) || 0) + 1); }
    return { pair, ctx };
  };
  /** the held-out cross-entropy (nats a character) of an n-gram model with add-k smoothing; and how many held-out
   *  contexts the training text never showed */
  P13C.ngram = (n, k) => {
    const H = P13C.text(), cut = P13C.cut(), V = P13C.vocab().length, M = P13C.counts(H.slice(0, cut), n);
    let s = 0, m = 0, unseen = 0;
    for (let i = cut + 1; i < H.length; i++) {
      const c = H.slice(i - n + 1, i), a = M.pair.get(c + '\u0000' + H[i]) || 0, b = M.ctx.get(c) || 0;
      s -= Math.log((a + k) / (b + k * V)); m++; if (!b) unseen++;
    }
    return { n, k, ce: s / m, m, unseen, share: unseen / m, contexts: M.ctx.size, table: Math.pow(V, n) };
  };
  /** what follows a context in the whole text, commonest first: [[char, count]], total */
  P13C.after = (ctx) => {
    const H = P13C.text(), c = new Map();
    let tot = 0;
    for (let i = ctx.length; i < H.length; i++) if (H.slice(i - ctx.length, i) === ctx) { c.set(H[i], (c.get(H[i]) || 0) + 1); tot++; }
    return { rows: [...c.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)), total: tot };
  };
  // the trained character GPT on the same 700 held-out characters, each read with up to 32 characters before it
  // (S.P13.gpt() run window by window; it takes seconds, so the value is kept here)
  P13C.GPT_HELD = 1.622;
  P13C.K = 0.1;

  const f3 = (v) => v.toFixed(3), f2 = (v) => v.toFixed(2), pct = (v) => Math.round(100 * v) + '%';
  const sh = (c) => (c === ' ' ? '·' : c);
  const NG = [1, 2, 3, 4, 5, 6].map((n) => P13C.ngram(n, P13C.K));
  const best = NG.reduce((a, b) => (b.ce < a.ce ? b : a));
  const T = P13C.after('t'), TH = T.rows[0];

  C.chapter({ id: 'll-tokens', part: 'llm', n: 1, title: 'Tokens', blurb: 'Text cut into pieces a model can count: byte-pair encoding learning its merges from a text, then cutting a new sentence with them.' });
  C.chapter({ id: 'll-predict', part: 'llm', n: 2, title: 'Predicting the next token', blurb: 'Counting models and neural ones, cross-entropy and perplexity, a tiny GPT trained on our own text, and what more parameters buy.' });
  C.chapter({ id: 'll-generate', part: 'llm', n: 3, title: 'Generating text', blurb: 'From probabilities to words: greedy choice, temperature, top-k and top-p, and the key–value cache that keeps generation fast.' });
  C.chapter({ id: 'll-beyond', part: 'llm', n: 4, title: 'From a tiny model to a large one', blurb: 'The training stages that turn a next-token predictor into an assistant, learning from examples in the prompt, and the limits that remain.' });

  // =========================================================================================================
  // Chapter 1: tokens
  // =========================================================================================================
  C.lesson({
    id: 'll-bpe', chapter: 'll-tokens', title: 'Learning merges from a text', minutes: 7,
    summary: 'A language model reads tokens: pieces of text from a fixed vocabulary. Byte-pair encoding builds that vocabulary from a text by counting. It starts from single characters, finds the commonest pair of neighbours, merges it into one new token everywhere, and counts again. Each merge makes the vocabulary one larger and the text shorter.',
    blocks: [
      'A model needs a fixed list of symbols to give probabilities to. Single letters make the list short but the texts long. Whole words make texts short, but the list would never end, and a new word would have no symbol at all. Is there something in between that a computer can find on its own?',
      { topic: 'bpe-build', frame: 0, caption: 'The harbour text, written for Nooral: 7008 characters, so 7008 tokens to begin with, from 26 different characters. The commonest pair of neighbours is a space followed by “t”, 390 times.' },
      { def: 'Byte-pair encoding (BPE)', text: 'start with one token per character; count every pair of neighbouring tokens (inside a word, with its leading space); merge the commonest pair into a new token everywhere it stands; repeat for as many merges as wanted.' },
      { topic: 'bpe-build', frame: 1, caption: 'Merge 1: “·” + “t” becomes “·t” in all 390 places, so the text is 390 tokens shorter: 6618. Counting again, “h” + “e” leads with 372.' },
      { predict: { q: 'After merges 1 (“·t”) and 2 (“he”), which pair do you expect to be the commonest?', options: ['“·” + “s”', '“t” + “h”', '“·t” + “he”', '“e” + “·”'], answer: 2, explain: 'The two new tokens stand side by side in every “·the”: 289 times. Merge 3 makes “·the” one token.' } },
      { topic: 'bpe-build', frame: 3, caption: 'Merge 3 builds on the first two: “·t” + “he” → “·the”, 289 times. Merges grow longer pieces out of shorter ones.' },
      { topic: 'bpe-build', frame: 13, caption: 'After 12 merges: a vocabulary of 38 tokens, and the text needs 5093 of them, 1.38 characters per token.' },
      { topic: 'bpe-build', input: { text: 'garden', merges: 3 }, frame: 4, caption: 'The same rule on a shorter text of our own, 458 characters: the order differs (“he” first, then “·t”), but after 3 merges “·the” is again a single token. What gets merged depends only on what the text repeats.' },
      { tip: 'Ties between equally common pairs are broken by taking the pair that sorts first, so the same text always gives the same merges.' },
      { textlab: { tab: 'bpe' }, title: 'Learn merges on your own text', text: 'The text lab’s BPE tab runs up to hundreds of merges and shows the pairs counted at every step.' },
      { h: 'Recap' },
      { list: ['Tokens sit between characters and words; their vocabulary is learned from text.', 'BPE: count neighbouring pairs, merge the commonest, repeat.', 'Each merge adds one token to the vocabulary and shortens the text by the pair’s count.'] },
      X('ll-x-bpe-first'), X('ll-x-bpe-tokens'), X('ll-x-bpe-table'), X('ll-x-bpe-trade'), X('ll-x-code-merge'),
    ],
  });

  C.lesson({
    id: 'll-tokenise', chapter: 'll-tokens', title: 'Cutting a new sentence into tokens', minutes: 6,
    summary: 'A trained tokenizer cuts a new sentence by replaying its merges in the order they were learned: first into chunks (a word with the space before it), then each chunk from single characters up. Common words end as one or two tokens; rare words fall into several pieces; a character never seen has no token at all.',
    blocks: [
      'The tokenizer has learned its merges. Now a sentence arrives that it has never seen. How is it cut, and why do some words cost more tokens than others?',
      { topic: 'bpe-tokenise', frame: 0, caption: '“the sailor sells a blue sail at the market.”: 43 characters, cut first into 10 chunks, each word with the space in front of it.' },
      { predict: { q: 'The tokenizer has learned 60 merges on the harbour text. How many tokens will this 43-character sentence become?', options: ['10, one per word', '22', '36', '43'], answer: 1, explain: 'The merges bring it down to 22 tokens: common words such as “·the” and “·sailor” are single tokens, but “·sells”, “·blue” and “·market” still take three or more.' } },
      { topic: 'bpe-tokenise', frame: 2, caption: 'Merges 3 and 4 apply: “·the” and “·s” appear and the sentence is 36 tokens. Merges are replayed in their learned order, not longest first.' },
      { topic: 'bpe-tokenise', frame: 9, caption: 'The end: 22 tokens from a vocabulary of 87, 1.95 characters per token. Each token is now a number, its id, and that is all the model receives.' },
      { topic: 'bpe-tokenise', input: { sentence: 'a jazz quartet played in the lighthouse.', merges: 100 }, frame: 14, caption: 'New words cost more. “j” and “z” never appear in the harbour text, so they become the unknown token, and “·jazz”, “·quartet” and “·played” fall into three or more pieces: 27 tokens, 1.48 characters each.' },
      { warn: 'A character the tokenizer never saw cannot be written at all here: capitals in “The Keeper climbs.” become the unknown token. Real tokenizers start from the 256 possible bytes instead of characters, so every text can be written, at worst one byte per token.' },
      { p: 'This is why the number of tokens, not of words, sets the cost of a text for a model, and why texts in languages, spellings or codes that were rare in the tokenizer’s training text cost more.' },
      { h: 'Recap' },
      { list: ['A new sentence: chunks, then the merges replayed in learned order, then ids.', 'Frequent words become one token; rare words cost several.', 'Byte-level tokenizers can write any text; a character-level one cannot write what it never saw.'] },
      X('ll-x-cut-word'), X('ll-x-unknown'), X('ll-x-ch-bpe'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: predicting the next token
  // =========================================================================================================
  const ngRows = NG.map((r) => [String(r.n), r.n === 1 ? 'none' : (r.n - 1) + ' character' + (r.n > 2 ? 's' : ''), r.table.toLocaleString('en'), pct(r.share), f3(r.ce)]);
  C.lesson({
    id: 'll-ngram', chapter: 'll-predict', title: 'Counting: the n-gram model', minutes: 8,
    summary: 'The oldest language model counts. To predict the next character after “t”, count what followed “t” in the training text and divide. Looking at more characters before helps, until most of the contexts met in new text were never seen in training. A neural model shares what it learns between similar contexts, so it can use a long context without having seen it before.',
    blocks: [
      'In the harbour text, what usually comes after the letter “t”? You do not need a network to answer that: you can count.',
      { table: [['next character', 'times after “t”', 'share']].concat(T.rows.slice(0, 6).map(([c, n]) => [sh(c), String(n), f3(n / T.total)])), caption: '“t” stands ' + T.total + ' times in the harbour text with a character after it. The share of each is the counting model’s probability: P(' + TH[0] + ' | t) = ' + TH[1] + ' / ' + T.total + ' = ' + f3(TH[1] / T.total) + '.' },
      { def: 'n-gram model', text: 'P(next | the n − 1 characters before) = count(context followed by next) / count(context). A bigram model (n = 2) looks one character back, a trigram model two.' },
      { topic: 'lm-cross-entropy', frame: 1, caption: 'The trained GPT, after “t”, gives “h” p = 0.52: nearly the share counted in the table. With one character of context there is little more to know than the counts.' },
      { p: 'Counts have a hole: a pair never seen gets probability 0, and −ln 0 is infinite. Add-k smoothing adds a small k to every count, here k = ' + P13C.K + ', so nothing is impossible. Then score each model on the last tenth of the text, which it did not count.' },
      { predict: { q: 'Which context length gives the lowest held-out loss on the harbour text?', options: ['1 character (bigram)', '2 characters (trigram)', '5 characters', 'the longer the better'], answer: 1, explain: 'With 2 characters of context the loss is ' + f3(NG[2].ce) + ' nats a character; with 3 it is already worse, ' + f3(NG[3].ce) + ', and with 5 it is ' + f3(NG[5].ce) + '.' } },
      { table: [['n', 'context', 'table entries', 'held-out contexts never seen', 'held-out loss (nats)']].concat(ngRows), caption: 'n-gram models with add-' + P13C.K + ' smoothing, counted on the first 90% of the harbour text and scored on the ' + NG[0].m + ' characters after it. The best is n = ' + best.n + ' at ' + f3(best.ce) + '; for longer contexts more and more of them were never seen.' },
      { p: 'The table also grows: a trigram table has ' + NG[2].table.toLocaleString('en') + ' entries and a 6-gram one ' + NG[5].table.toLocaleString('en') + ', though the text has only ' + NG[5].contexts.toLocaleString('en') + ' different 5-character contexts. Most entries are empty, and what was counted after “sails” teaches nothing about “sail.” or “saile”, however alike they look.' },
      { p: 'A neural model reads every character as a vector, so contexts that look alike give similar predictions. The tiny GPT of this part, 27,328 parameters with a context of 32 characters, scores ' + P13C.GPT_HELD + ' nats a character on the same held-out characters, below the best counting model’s ' + f3(best.ce) + '. On a few thousand characters the gap is small; on large text the long context is what makes the difference.' },
      { h: 'Recap' },
      { list: ['An n-gram model predicts from counts of what followed the last n − 1 characters.', 'Smoothing keeps unseen pairs from getting probability 0.', 'Longer contexts help until most of them were never seen; a neural model generalises between similar contexts.'] },
      X('ll-x-bigram'), X('ll-x-code-bigram'), X('ll-x-ngram-best'), X('ll-x-ch-ngram'),
    ],
  });

  C.lesson({
    id: 'll-next', chapter: 'll-predict', title: 'A network that predicts the next character', minutes: 6,
    summary: 'A GPT reads the text so far and puts out one score for every token in its vocabulary; a softmax turns the scores into probabilities. To write, it picks a token, appends it, and reads the longer text again. Its confidence changes from place to place: inside a word the next letter is often clear, at the start of a word many are possible.',
    blocks: [
      'Give the model “the weaver said that ”. What does it actually put out, and how does that become text?',
      { topic: 'lm-next-token', frame: 0, caption: 'The model reads the 21 characters of the prompt. Out of the last position come 26 scores, one per character it knows, turned into probabilities that add up to one.' },
      { topic: 'lm-next-token', frame: 1, caption: 'After “…said that ” the most likely next character is “h”, p = 0.48; then “s” (0.23) and “t” (0.10). The model takes the first and reads again.' },
      { predict: { q: 'The model has written “the weaver said that he ”. How sure will it be of the next character?', options: ['Very sure, over 0.9', 'Unsure: below 0.3', 'Exactly 0.5'], answer: 1, explain: 'The start of a word: “w” leads with only 0.29, “s” has 0.28. After “he” it was 0.96 sure of the space.' } },
      { topic: 'lm-next-token', frame: 4, caption: 'At the start of the next word the model is unsure: “w” 0.29, “s” 0.28, “b” 0.08.' },
      { topic: 'lm-next-token', frame: 7, caption: 'The model added “he·wou”. It was surest of the space after “he” (0.96) and least sure of the “w” (0.29, ringed).' },
      { topic: 'lm-next-token', input: { prompt: 'the lighth', steps: 5 }, frame: 2, caption: 'A tiny model also errs with confidence. After “the lighth” it adds “t” (p = 0.74), not the “o” of “lighthouse”, then wants to end the word with a full stop, p = 0.61.' },
      { tip: 'Everything a language model does, answering, translating, summarising, comes from this one step repeated: a probability for every possible next token.' },
      { h: 'Recap' },
      { list: ['Scores for every token, a softmax, probabilities that sum to one.', 'Generation appends one token and reads again.', 'Confidence varies: high inside words, lower at their start.'] },
      X('ll-x-softmax'),
    ],
  });

  C.lesson({
    id: 'll-ce', chapter: 'll-predict', title: 'Scoring guesses: cross-entropy and perplexity', minutes: 7,
    summary: 'A language model is scored by the probability it gave to what really came next. Each position costs −ln p; the mean cost is the cross-entropy, in nats per token, and e to that power is the perplexity: the number of equally likely choices the model is as unsure as. Guessing uniformly among V tokens costs ln V, perplexity V.',
    blocks: [
      'Two models read the same sentence. How can one number say which predicted it better?',
      { topic: 'lm-cross-entropy', frame: 2, caption: 'After “th” the model gave the true next character, “e”, p = 0.95, so it pays −ln 0.95 = 0.06. A confident right guess costs little.' },
      { topic: 'lm-cross-entropy', frame: 4, caption: 'After “the·” it gave “s” only p = 0.20: the cost is −ln 0.20 = 1.61. Many words could start here.' },
      { math: 'cost at a position = −ln p(the true next token)\ncross-entropy = mean cost over the positions   (nats per token)\nperplexity = e^cross-entropy\nuniform over V tokens: cross-entropy ln V, perplexity V' },
      { predict: { q: 'The 26 costs of “the sailor coils the ropes.” add up to 29.06. What is the perplexity?', options: ['about 1.1', 'about 3.1', 'about 29'], answer: 1, explain: '29.06 / 26 = 1.118 nats a character, and e^1.118 = 3.06: as unsure as choosing among about 3 characters.' } },
      { topic: 'lm-cross-entropy', frame: 10, caption: 'The mean cost is 1.118 nats a character, perplexity 3.06. Guessing uniformly among 26 characters would cost ln 26 = 3.258, perplexity 26.' },
      { topic: 'lm-cross-entropy', input: { text: 'the sailor coils the ropes.', model: 'untrained' }, frame: 10, caption: 'The same sentence through the untrained model: 3.348 nats, perplexity 28.44, a little worse than uniform guessing.' },
      { topic: 'lm-cross-entropy', input: { text: 'ropes the coils sailor the.', model: 'trained' }, frame: 10, caption: 'The same words shuffled: 1.384 nats, perplexity 3.99. The words are familiar, so most letters are still easy; only the order is wrong.' },
      { p: 'Cross-entropy is the loss a language model is trained on, and perplexity the way it is usually reported. Comparisons only hold between models with the same tokens: a cost per character and a cost per word-sized token are different numbers.' },
      { h: 'Recap' },
      { list: ['Cost −ln p of the true next token; cross-entropy is the mean.', 'Perplexity = e^cross-entropy, the effective number of choices.', 'Uniform guessing over V tokens: ln V nats, perplexity V.'] },
      X('ll-x-cost'), X('ll-x-perplexity'), X('ll-x-ce-table'), X('ll-x-code-ce'),
    ],
  });

  C.lesson({
    id: 'll-train', chapter: 'll-predict', title: 'Training a tiny GPT', minutes: 8,
    summary: 'A GPT is trained by showing it windows of text and lowering the cross-entropy of every next character in them. The loss on held-out text falls fast at first, as it learns which characters are common, then slowly, as it learns words and their order. The gap between training and held-out loss is what it memorised rather than learned.',
    blocks: [
      'A network of 27,328 random numbers, a few thousand characters of text, and one rule: make the next character more likely. What does it learn first?',
      { topic: 'lm-gpt-training', frame: 0, caption: 'Step 0: random weights. The held-out loss is 3.401, near ln 26 = 3.258, and the sample after “the sailor ” is noise.' },
      { topic: 'lm-gpt-training', frame: 2, caption: 'Step 100: held-out loss 2.228, perplexity 9.28. Common short words such as “the” and “he” appear.' },
      { predict: { q: 'By step 1600 the training batches average a loss of 1.35. Where will the held-out loss be?', options: ['Also about 1.35', 'Higher, about 1.6', 'Lower, about 1.1'], answer: 1, explain: 'It ends at 1.635. The model is scored on text it never trained on, and part of what it learned was particular to its training text.' } },
      { topic: 'lm-gpt-training', frame: 5, caption: 'Step 800: held-out loss 1.684, perplexity 5.39. Many words are spelt right and sentences end with full stops.' },
      { topic: 'lm-gpt-training', frame: 7, caption: 'The end, 1600 steps: held-out 1.635 against 1.35 on the training batches. Sampled at temperature 0.8: “the sailor fin from the bats the clim. the fisher over…”.' },
      { p: 'Every step takes a batch of 16 windows of 32 characters, computes the cross-entropy at every position at once (the causal mask lets each position see only the characters before it), and takes one optimiser step. The loss curve falls fast and then flattens: the easy patterns are learned first.' },
      { textlab: { tab: 'gpt' }, title: 'Train a larger GPT here', text: 'The text lab trains a GPT on the long corpus of stories and dialogues written for Nooral, on the graphics card where the browser allows it: watch both losses and the samples improve.' },
      { tip: 'The text lab’s default model has width 96, 3 layers and a context of 64 characters; a run of a few minutes reaches a held-out loss near 1.4 to 1.5 nats a character, where most sampled words are real words.' },
      { h: 'Recap' },
      { list: ['Training lowers the mean cross-entropy over every position of every window.', 'Held-out loss falls fast, then slowly; samples go from noise to words.', 'The train–held-out gap measures memorising.'] },
      X('ll-x-train-curve'), X('ll-x-gap'),
    ],
  });

  C.lesson({
    id: 'll-scale', chapter: 'll-predict', title: 'Bigger models, lower loss', minutes: 6,
    summary: 'Train the same design at several widths for the same number of steps, and the held-out loss falls as the parameters grow, close to a straight line on log–log axes. Such scaling laws let a team predict a large model’s loss from small runs, but they hold only while the data and the compute grow with the model.',
    blocks: [
      'If a model with twice the parameters were trained the same way, how much better would it be? A small experiment answers it for our text.',
      { topic: 'lm-scaling', frame: 0, caption: 'Width 8: 2224 parameters, 400 steps, a held-out loss of 2.315.' },
      { predict: { q: 'From width 8 (2224 parameters) to width 48 (59,424), about 27 times the parameters. What happens to the held-out loss?', options: ['It halves', 'It falls by about a fifth, to about 1.87', 'It barely moves', 'It rises: too many parameters'], answer: 1, explain: 'It falls from 2.315 to 1.865, steadily, a little with every widening.' } },
      { topic: 'lm-scaling', frame: 4, caption: 'Width 48: 59,424 parameters, held-out loss 1.865, below every smaller model.' },
      { topic: 'lm-scaling', frame: 5, caption: 'On log–log axes the five points lie near a line, ln L = a + b ln N with slope b = −0.066: each doubling of the parameters multiplies the loss by 2^b = 0.956.' },
      { math: 'L(N) ≈ A · N^b,   b < 0\nln L = ln A + b · ln N        a straight line on log–log axes\ndoubling N multiplies L by 2^b' },
      { warn: 'A fitted line is not a promise. With only a few thousand characters a bigger model soon memorises the text and the held-out loss stops falling. Real scaling runs grow the data and the compute together with the model.' },
      { h: 'Recap' },
      { list: ['Same design, same training, more parameters: lower loss.', 'Close to a power law: a straight line on log–log axes.', 'It holds only while data and compute grow too.'] },
      X('ll-x-scaling'), X('ll-x-scale-curve'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: generating text
  // =========================================================================================================
  C.lesson({
    id: 'll-temp', chapter: 'll-generate', title: 'Greedy choice and temperature', minutes: 7,
    summary: 'Greedy decoding always takes the most likely token: the same prompt always gives the same text, often dull and repetitive. Sampling draws by the probabilities instead. A temperature T divides the scores before the softmax: T below 1 sharpens the distribution towards greedy, T above 1 flattens it towards a uniform draw.',
    blocks: [
      'The model has given probabilities. Now one character must be chosen. Take the likeliest every time, or roll a die weighted by the probabilities?',
      { topic: 'sampling-temperature', frame: 0, caption: 'After “the sailor said that ” the model gives “h” 0.45 and “s” 0.27, at temperature 1.' },
      { math: 'p_i = exp(z_i / T) / Σ_j exp(z_j / T)\nT → 0: all the probability on the largest score (greedy)\nT = 1: the model’s own probabilities\nT → ∞: uniform' },
      { predict: { q: 'Divide the scores by T = 0.5 before the softmax. What happens to p(“h”) = 0.45?', options: ['It falls', 'It stays 0.45', 'It rises'], answer: 2, explain: 'Dividing by 0.5 doubles the gaps between scores, so the favourite gains: 0.45 becomes 0.71.' } },
      { topic: 'sampling-temperature', frame: 2, caption: 'The same scores at three temperatures. T = 0.5: “h” rises to 0.71 and the rest shrink. T = 1.6: “h” falls to 0.27, and the eighth character rises from 0.02 to 0.03.' },
      { topic: 'sampling-temperature', frame: 3, caption: 'Greedy decoding: “he would wall she wall said that he woul”, the same 40 characters every time, looping through the commonest words.' },
      { topic: 'sampling-temperature', frame: 5, caption: 'T = 0.5 with seed 4 writes “she sailor said that he waver said that ”; T = 1.6 with the same seed writes “d hat hn idenk. theago boer…”, where only 2 of the 9 words are words of the text.' },
      { tip: 'Low temperatures suit tasks with one right answer; higher ones give variety. Most systems sit a little below 1.' },
      { h: 'Recap' },
      { list: ['Greedy: always the most likely token, deterministic and repetitive.', 'Temperature divides the scores: below 1 sharper, above 1 flatter.', 'Sampling with a seed is repeatable; without one, every run differs.'] },
      X('ll-x-temp-table'), X('ll-x-greedy'), X('ll-x-code-temp'),
    ],
  });

  C.lesson({
    id: 'll-topk', chapter: 'll-generate', title: 'Top-k and top-p: cutting the tail', minutes: 7,
    summary: 'Even at a sensible temperature, the many unlikely tokens together hold real probability, and one bad draw can derail a text. Top-k keeps only the k most likely tokens; top-p keeps the fewest whose probabilities add to at least p. What is kept is renormalised. Top-p adapts: a few tokens when the model is sure, many when it is not.',
    blocks: [
      'Each unlikely character is rarely drawn. But there are many of them. How often does one of them get through?',
      { topic: 'sampling-topk-topp', frame: 0, caption: 'After “the boats ”, “t” leads with 0.29, and the 21 least likely characters share 0.35: a third of all draws would be one of them.' },
      { topic: 'sampling-topk-topp', frame: 1, caption: 'Top-k with k = 5 keeps 5 characters holding 0.65 of the probability. Each is divided by 0.65: “t” goes from 0.29 to 0.45.' },
      { predict: { q: 'Top-p with p = 0.80 on the same distribution: will it keep more or fewer than 5 characters?', options: ['Fewer', 'More', 'Exactly 5'], answer: 1, explain: 'The top 5 hold only 0.65, short of 0.80, so the nucleus needs 9 characters (0.81).' } },
      { topic: 'sampling-topk-topp', frame: 2, caption: 'Top-p with p = 0.80 needs 9 characters (0.81 together). The model is unsure here, so the nucleus is larger than k.' },
      { topic: 'sampling-topk-topp', input: { prompt: 'the lighthou', k: 5, p: 0.9 }, frame: 2, caption: 'After “the lighthou” the model is surer: “t” has 0.42 and top-p = 0.90 needs only 7 characters.' },
      { topic: 'sampling-topk-topp', frame: 3, caption: 'Both rules together, top-k first, then top-p inside it: 5 characters remain, and the draw gives “a”, with renormalised probability 0.22.' },
      { textlab: { tab: 'sample', sample: { prompt: 'the boats ', topK: 5, topP: 0.8, temperature: 1 } }, title: 'Sample with your own rule', text: 'The text lab’s sampling tab draws one character at a time with the probability bars before and after the rule.' },
      { h: 'Recap' },
      { list: ['Top-k: keep the k most likely tokens.', 'Top-p: keep the fewest whose total reaches p; it adapts to the model’s certainty.', 'Renormalise what is kept, then draw.'] },
      X('ll-x-topk'), X('ll-x-topp'), X('ll-x-code-cut'),
    ],
  });

  C.lesson({
    id: 'll-kv', chapter: 'll-generate', title: 'The KV cache', minutes: 6,
    summary: 'Each new token attends to the keys and values of every token before it. Those do not change once computed, so a model can keep them, a key–value cache, and run only the newest token through its blocks at each step. Only the newest token runs through the blocks; its attention still reads every cached key and value, but nothing is recomputed. The price is memory for every kept key and value.',
    blocks: [
      'To write the 100th token, the model attends to the 99 before it. Must it recompute all 99 every time?',
      { topic: 'kv-cache-generation', frame: 1, caption: 'Step 1, the prefill: the 11 prompt characters of “the keeper ” run through the blocks together, giving “a”. With a cache, each position’s keys and values are kept.' },
      { topic: 'kv-cache-generation', frame: 2, caption: 'Step 2: without a cache all 12 positions run again; with it only the newest one runs, and reads the other 11 positions’ keys and values from memory. So far 23 positions against 12.' },
      { predict: { q: 'Writing 8 characters after an 11-character prompt, how many positions go through the blocks in all, without a cache?', options: ['19', '88', '116'], answer: 2, explain: '11 + 12 + 13 + … + 18 = 116. With the cache it is 11 for the prefill and then 1 per new character: 18.' } },
      { topic: 'kv-cache-generation', frame: 9, caption: 'Both runs write the same text, “and·the·”: 116 positions without the cache, 18 with it, 6.4 times less work, for 2304 kept numbers.' },
      { topic: 'kv-cache-generation', input: { prompt: 'a ', n: 16 }, frame: 13, caption: 'A short prompt and a long reply: 152 positions without the cache against 17 with it, 8.9 times less. Without the cache the work grows with the square of the length, with it only linearly.' },
      { math: 'without a cache:  P + (P + 1) + … + (P + n − 1) positions\nwith a cache:     P + (n − 1)\nmemory:  layers × 2 (keys, values) × positions × width' },
      { h: 'Recap' },
      { list: ['Keys and values of past tokens never change: keep them.', 'Each step then runs one position instead of the whole text.', 'The cost is memory, growing with layers, length and width.'] },
      X('ll-x-kv-count'), X('ll-x-kv-memory'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: from a tiny model to a large one
  // =========================================================================================================
  C.lesson({
    id: 'll-stages', chapter: 'll-beyond', title: 'From predictor to assistant', minutes: 8,
    summary: 'A large assistant model is shaped in stages. Pretraining on a vast amount of text teaches next-token prediction, and with it grammar, facts and style. Instruction tuning, a fine-tuning on many tasks phrased as requests, teaches it to answer. Preference tuning then shows pairs of answers and raises the one people judged better.',
    blocks: [
      'A model trained only to continue text, asked a question, may answer with more questions. What turns it into something that answers?',
      { topic: 'lm-training-stages', frame: 0, caption: 'Three stages, left to right. The same network passes through all of them: each starts from the weights the last one left.' },
      { topic: 'lm-training-stages', frame: 1, caption: 'Pretraining is what our tiny GPT did, on 7008 characters, ending at a held-out loss of 1.635. A large model reads a large share of the world’s public text.' },
      { topic: 'lm-training-stages', frame: 3, caption: 'Instruction tuning is fine-tuning on thousands of different tasks, each a request with a good reply written by people. The model learns the habit of answering what was asked.' },
      { predict: { q: 'Preference tuning starts with the tuned copy equal to the original. What is the preference loss at that moment?', options: ['0', 'ln 2 = 0.693', '1'], answer: 1, explain: 'The margin over the original is 0, so the loss is −ln σ(0) = ln 2 = 0.693.' } },
      { topic: 'lm-training-stages', frame: 4, caption: 'Preference tuning on the tiny GPT itself: a prompt, a chosen answer A and a rejected answer B. Before tuning the loss is ln 2 = 0.693.' },
      { topic: 'lm-training-stages', frame: 5, caption: 'After 10 steps with β = 0.5: log p(A) went from −33.25 to −30.06 and log p(B) from −34.12 to −40.32; the loss fell to 0.009.' },
      { math: 'margin = β · ((log p(A) − log p(B)) − (the same under the frozen original))\nloss = −ln σ(margin)' },
      { topic: 'lm-training-stages', frame: 6, caption: 'Each stage changes the same weights with different data and a different goal. Most of what the model knows comes from pretraining; the later stages mostly change how it behaves.' },
      { tip: 'Learning from human preferences can also run through a separate reward model and reinforcement learning; the direct loss shown here reaches a similar goal with ordinary gradient descent.' },
      { h: 'Recap' },
      { list: ['Pretraining: next token on vast text.', 'Instruction tuning: fine-tuning on examples of the wanted behaviour, many tasks as requests.', 'Preference tuning: raise the better of two answers, staying near the original.'] },
      X('ll-x-stages'), X('ll-x-pref-loss'),
    ],
  });

  C.lesson({
    id: 'll-icl', chapter: 'll-beyond', title: 'Learning from the prompt', minutes: 7,
    summary: 'A model can use examples given in its prompt without any change to its weights: in-context learning. A tiny GPT trained on sequences of key–value pairs, whose meanings change in every sequence, can only answer a repeated key by finding its earlier pair and copying the value. It learns that procedure, not any fact, and attention is where it happens.',
    blocks: [
      'Show a model “c7 a2 e5” and then ask “a?”. Nothing in its training told it what “a” means today. Can it answer?',
      { topic: 'lm-in-context', frame: 0, caption: 'Each pair is a key, a letter a to f, then a digit. In every training sequence the letters got new random digits: the only way to answer is to look back.' },
      { topic: 'lm-in-context', frame: 1, caption: 'Key “a” appeared earlier with 2. The model gives 2 probability 1.00, and its last block attends most (1.00) to position 4, the digit of that earlier pair.' },
      { predict: { q: 'Then comes “f?”, a key not seen before in this sequence. What will the model do?', options: ['Answer with certainty', 'Spread its probability over the digits: a guess', 'Refuse to answer'], answer: 1, explain: 'There is nothing to copy. Its top digit, 9, gets only 0.11, about one in ten.' } },
      { topic: 'lm-in-context', frame: 3, caption: 'Key “f” has not appeared: the top digit gets 0.11. It is guessing, and the probabilities show it.' },
      { topic: 'lm-in-context', frame: 5, caption: 'Over 2400 questions in 300 held-out sequences: a key seen before is answered right 99% of the time, a new key 10%, the chance of guessing one digit in ten.' },
      { p: 'Large models do the same on a much richer scale. A few examples in a prompt (“few-shot” prompting) show the format and the mapping, and the model continues the pattern. Nothing is stored: the next prompt starts from the same weights.' },
      { h: 'Recap' },
      { list: ['In-context learning: using examples in the prompt, with no weight change.', 'Attention finds the earlier example; the model copies what it needs.', 'With no example to find, it can only guess.'] },
      X('ll-x-icl'), X('ll-x-attn'),
    ],
  });

  C.lesson({
    id: 'll-limits', chapter: 'll-beyond', title: 'Limits: invented facts and copied bias', minutes: 7,
    summary: 'A language model writes the likely continuation, whether or not it knows anything about the subject; fluent text gives no warning that it is made up. And it returns the skews of its training text, often stronger than they were. The defences are measurement, grounding answers in sources that can be checked, better data and tuning, and checking again.',
    blocks: [
      'Ask the tiny GPT about a king. The harbour text never mentions one. What does it say?',
      { topic: 'lm-limits', frame: 1, caption: 'It writes “she grow ater. the wind the qu…” at temperature 0.8, averaging p = 0.45 per character: nothing stops it, and there is no step where it checks what it knows.' },
      { topic: 'lm-limits', frame: 2, caption: 'A sentence it really read scores 0.66 per character; its invented text 0.45. The difference is real but hidden: the reader sees only fluent text.' },
      { predict: { q: 'In the training text, “the weaver said that ” is followed by “she” 6 times and “he” once. Which will the model prefer after that prompt?', options: ['“she”, as its text says', 'Both equally', '“he”'], answer: 2, explain: 'It gives “he” 0.415 and “she” 0.118. “he” is commoner in the text as a whole, and a small model leans on the commonest pattern: the minority case is erased.' } },
      { topic: 'lm-limits', frame: 4, caption: 'The model gives “he” 0.415 and “she” 0.118 after “the weaver said that ”: it follows the text’s overall habit and erases the 6 sentences that said otherwise.' },
      { topic: 'lm-limits', input: { prompt: 'the dragon ', role: 'sailor', seed: 5 }, frame: 4, caption: 'After “the sailor said that ”, where the text has “he” 7 times and “she” once, it gives “he” 0.380 and “she” 0.139: the skew copied, “he” about 2.7 times as likely.' },
      { h: 'Defences' },
      { list: ['Measure: test sets that probe made-up facts and skews, before and after every change.', 'Ground answers: give the model sources to quote, and show them, so claims can be checked.', 'Fix the data and tune on balanced examples; let the model say when it does not know.', 'Check again: none of these is ever finished.'] },
      { h: 'Recap' },
      { list: ['Fluency is not knowledge: a model invents with the same confidence.', 'Models copy, and can amplify, the skews of their data.', 'Defend by measuring, grounding, better data and repeated checks.'] },
      X('ll-x-limits'), X('ll-x-bias-share'), X('ll-x-defence'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
