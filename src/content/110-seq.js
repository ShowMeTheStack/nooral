// Part 11, sequences and embeddings: the chapters and the lessons. The exercises are in 111-seq-ex.js; the model
// challenge, the guided tour, the course and the pages the diagram checker looks at are in 112-seq-ch.js. Each lesson
// opens with a question, runs the topic that shows the idea at a frame chosen for the point being made, asks for a
// prediction, sums up and leads to its exercises. Every value quoted here was printed by the topics themselves:
// word2vec trained here on the part's own corpus, and the small recurrent networks shipped in p11-0-data.js.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || !S.WF || C.lessons.has('sq-onehot')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'sq-embed', part: 'sequences', n: 1, title: 'Words as vectors', blurb: 'One-hot vectors, where every word is as far from every other; embedding tables, where a lookup is a product; word2vec learning the table from the company words keep; and a map of the result, with neighbours and analogies.' });
  C.chapter({ id: 'sq-rnn', part: 'sequences', n: 2, title: 'Recurrent networks', blurb: 'One cell applied at every step with the same weights, unrolled into a deep network, and the gradient that fades or explodes on its way back through time.' });
  C.chapter({ id: 'sq-gates', part: 'sequences', n: 3, title: 'Gated cells: LSTM and GRU', blurb: 'Gates between 0 and 1 that decide what to forget, what to write and what to show; a memory updated by adding, so the gradient can travel far back.' });
  C.chapter({ id: 'sq-seq', part: 'sequences', n: 4, title: 'Sequence models at work', blurb: 'A character model predicting the next letter, its loss and perplexity, sampling at a temperature, an encoder–decoder squeezing its input into one vector, and a recurrent classifier on the canvas where word order decides.' });

  // =========================================================================================================
  // Chapter 1: words as vectors
  // =========================================================================================================
  C.lesson({
    id: 'sq-onehot', chapter: 'sq-embed', title: 'One-hot vectors and embedding tables', minutes: 7,
    summary: 'A network reads numbers, not words. One-hot coding gives each word its own position in a vector as long as the vocabulary, so every pair of words is exactly as far apart. An embedding table gives each word a short row of learned numbers instead; looking a word up is the same as multiplying its one-hot vector by the table, and words used alike get rows that point the same way.',
    blocks: [
      'To a network, “king” is just a string of letters. How do we turn a word into numbers, and can the numbers say that “king” is closer to “queen” than to “bread”?',
      { topic: 'one-hot-embeddings', frame: 1, caption: 'The vocabulary holds 88 words. “king” becomes 88 numbers: all zero except a 1 at its id, 52.' },
      { topic: 'one-hot-embeddings', frame: 2, caption: 'Any two different one-hot vectors share no position: their dot product is 0 and their distance is √2 ≈ 1.41, whichever two words they are.' },
      { predict: { q: 'The vocabulary grows from 88 to 10,000 words. What happens to the distance between the one-hot vectors of “king” and “queen”?', options: ['It stays √2', 'It grows with the vocabulary', 'It shrinks towards 0'], answer: 0, explain: 'Each vector still has a single 1, in a different place: the difference has one +1 and one −1, so its length is √2 whatever the size.' } },
      { topic: 'one-hot-embeddings', frame: 3, caption: 'An embedding table: one row of 16 learned numbers per word. The rows of “king” and “queen” already look alike.' },
      { math: 'onehot(king) · E = row 52 of E       (a 1 × 88 vector times an 88 × 16 table)' },
      { topic: 'one-hot-embeddings', frame: 4, caption: 'The product adds up 88 rows, each times 0 except row 52 times 1: it is that row. An embedding layer does this product as a lookup.' },
      { topic: 'one-hot-embeddings', frame: 5, caption: 'Cosines between rows now mean something: “bread” and “rice” 0.99, “king” and “red” −0.41.' },
      { topic: 'one-hot-embeddings', frame: 5, input: { words: ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl'] }, caption: 'Among people only, every cosine is high, from 0.70 (“king” and “girl”) to 0.98 (“boy” and “girl”): they all fill the same slots in the sentences.' },
      { topic: 'one-hot-embeddings', frame: 6, caption: '88 numbers per word as one-hot, 16 as an embedding; the table holds 1408 learned numbers.' },
      { h: 'Recap' },
      { list: ['One-hot: one position per word; every pair of words is √2 apart, so it says only that words differ.', 'An embedding table holds a short learned row per word; looking up is multiplying by a one-hot vector.', 'Cosine similarity compares rows by direction: near 1 for words used alike.'] },
      X('sq-x-onehot-dist'), X('sq-x-table-size'), X('sq-x-lookup'), X('sq-x-cosine'),
    ],
  });

  C.lesson({
    id: 'sq-w2v', chapter: 'sq-embed', title: 'word2vec: learning from neighbouring words', minutes: 8,
    summary: 'word2vec learns the table from plain text. Each word has a centre vector v and a context vector u. For every word and each neighbour within the window, training pushes σ(u·v) towards 1, and pushes it towards 0 for a few words drawn at random (negative sampling). Words that turn up in the same places get pulled in the same directions, so their vectors end up alike.',
    blocks: [
      'Nobody tells the computer that a goat and a horse are both animals. Can it work that out from sentences alone?',
      { topic: 'word2vec-train', frame: 0, caption: 'The corpus: 5238 words of our own sentences, 88 distinct words kept. Each word is paired with the words up to 2 places either side.' },
      { math: 'loss of a pair (c, o) = −ln σ(u_o · v_c) − Σₖ ln σ(−u_k · v_c)       k: the negatives drawn at random' },
      { topic: 'word2vec-train', frame: 1, caption: 'The pair (cat, fed) and 5 negatives. The context vectors start at zero, so every score is σ(0) = 0.50 and the loss is 6 · ln 2 = 4.159.' },
      { predict: { q: 'After training, what should σ(u·v) be for the real pair, and for the negatives?', options: ['Higher for the real pair, near 0 for the negatives', 'Exactly 1 for every pair', 'Near 0 for all of them'], answer: 0, explain: 'The loss rewards a high score for a pair seen in the text and a low one for random words. After 8 epochs: 0.41 for (cat, fed), at most 0.16 for the negatives.' } },
      { topic: 'word2vec-train', frame: 3, caption: 'One epoch: the mean loss per pair falls to 2.755; words of the same kind already have a mean cosine of 0.65, words of different kinds −0.10.' },
      { topic: 'word2vec-train', frame: 10, caption: 'Eight epochs: loss 1.724, and the kinds sit in clumps of their own colour.' },
      { topic: 'word2vec-train', frame: 11, caption: 'The pair (cat, fed) scores 0.41, the negatives between 0.00 and 0.16: its loss fell from 4.159 to 1.219.' },
      { topic: 'word2vec-train', frame: 12, caption: 'The nearest words to “cat”: rabbit, goat, horse, dog and sheep. It never saw a definition, only the places these words appear.' },
      { topic: 'word2vec-train', frame: 5, input: { epochs: 1, probe: 'three' }, caption: 'After one epoch only, the nearest words to “three” are soup, rice, two, five and green: half learned. Numbers and food share the sentence “we bought … loaves of …”.' },
      { tip: 'Embeddings learn whatever patterns the text holds, including unwanted ones. Trained on real text, they can tie jobs or qualities to groups of people; that is worth measuring before they are used to make decisions.' },
      { h: 'Recap' },
      { list: ['Each word has a centre vector and a context vector; a pair’s score is σ(u·v).', 'Real neighbours are pushed towards 1, a few random words (negatives) towards 0.', 'Words found in the same places end up with similar vectors.'] },
      X('sq-x-pairs'), X('sq-x-sgns-loss'), X('sq-x-w2v-curve'),
    ],
  });

  C.lesson({
    id: 'sq-analogy', chapter: 'sq-embed', title: 'The map of words: neighbours and analogies', minutes: 7,
    summary: 'Sixteen numbers per word cannot be drawn, but the two directions along which the words spread most can. Neighbours are found in the full vectors by cosine. A difference of two vectors can carry a relation: if woman − man points the same way as queen − king, then king + woman − man lands near queen. On a small corpus some relations come out cleanly and others blur.',
    blocks: [
      'If the vector of a word stands for how it is used, does the step from “man” to “woman” mean anything on its own?',
      { topic: 'word2vec-map', frame: 0, caption: 'The 16 numbers of each vector flattened onto the two directions of largest spread. They keep 55% of the spread: a fair picture, not the whole truth.' },
      { topic: 'word2vec-map', frame: 1, caption: 'Neighbours are measured in the full 16 numbers: “horse” is nearest rabbit (0.93), sheep (0.92), cat, goat and dog.' },
      { math: 'answer = the word whose vector is nearest king + woman − man (leaving out king, woman and man)' },
      { topic: 'word2vec-map', frame: 3, caption: 'The step woman − man added to “king”: the ring. In this slice man, woman, king and the ring make a parallelogram.' },
      { predict: { q: 'Which word should be nearest to the ring?', options: ['queen', 'woman', 'prince'], answer: 0, explain: 'queen at 0.94, then princess and prince at 0.78; woman is one of the three words given and is left out.' } },
      { topic: 'word2vec-map', frame: 4, caption: 'queen (0.94) comes first: the step from “man” to “woman” carried the relation.' },
      { topic: 'word2vec-map', frame: 4, input: { analogy: 'age' }, caption: 'Not every relation comes out: king + young − old gives girl, boy and princess, and “prince” is not among the top three.' },
      { h: 'Recap' },
      { list: ['The 2-D map is a projection; distances are judged in the full vectors.', 'Nearest neighbours by cosine are words used in the same places.', 'Vector differences can carry relations; with little text, some relations blur.'] },
      X('sq-x-analogy'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: recurrent networks
  // =========================================================================================================
  C.lesson({
    id: 'sq-unroll', chapter: 'sq-rnn', title: 'A recurrent network, unrolled through time', minutes: 8,
    summary: 'A recurrent network reads a sequence one step at a time and carries a hidden state h from step to step: h_t = tanh(x_t W_x + h_{t−1} W_h + b). The same weights are used at every step. Unrolled, a sequence of 8 steps is a network 8 layers deep whose layers all share their weights; an output layer reads h at each step.',
    blocks: [
      'A dense network takes a fixed number of inputs. A sentence or a stream of bits can be any length. How can one network read them all?',
      { math: 'h_t = tanh(x_t · W_x + h_{t−1} · W_h + b)\nŷ_t = softmax(h_t · W_y + b_y)' },
      { topic: 'rnn-unrolled', frame: 0, caption: 'A cell with 4 hidden numbers that counts the ones in a bit stream: W_x, W_h, a bias and an output layer, 69 parameters in all.' },
      { topic: 'rnn-unrolled', frame: 1, caption: 'Step 1 reads x = 1. The state starts at zero, so only the input and the bias count; tanh squashes [−0.30, 1.12, 0.69, 3.39] to h = [−0.30, 0.81, 0.60, 1.00].' },
      { topic: 'rnn-unrolled', frame: 2, caption: 'Step 2 reads 0. The old state through W_h gives [1.15, −1.44, 0.70, 1.32]: it is the state, not the input, that remembers the count of 1.' },
      { predict: { q: 'How many different input weight matrices does the network use over the 8 steps?', options: ['One, used at every step', 'Eight, one per step', 'It depends on the bits read'], answer: 0, explain: 'The cell is the same at every step. That is what lets one network read a sequence of any length.' } },
      { topic: 'rnn-unrolled', frame: 9, caption: 'All 8 counts are right. Unrolled, the network is 8 layers deep, every layer the same cell.' },
      { topic: 'rnn-unrolled', frame: 9, input: { bits: '11111111' }, caption: 'Eight ones: it gets 6 of the 8 counts right. By step 7 every number of h is pinned near ±1 by tanh, and the state has no room left to tell 6 from 7.' },
      { h: 'Recap' },
      { list: ['The hidden state carries what has been read so far.', 'The same W_x, W_h and b are used at every step: sequences of any length.', 'Unrolled, a sequence of n steps is an n-layer network with shared weights.'] },
      X('sq-x-rnn-step'), X('sq-x-rnn-params'), X('sq-x-share-choice'), X('sq-x-code-rnn'),
    ],
  });

  C.lesson({
    id: 'sq-bptt', chapter: 'sq-rnn', title: 'Gradients through time: fading and exploding', minutes: 8,
    summary: 'Training an unrolled network sends the gradient back through every step. Each step multiplies it by W_hᵀ and by the slope of tanh, so after k steps it has been multiplied k times: it shrinks like sᵏ when the factor s is below one, and grows like sᵏ when it is above. Exploding gradients are tamed by clipping; vanishing ones need a different cell.',
    blocks: [
      'The loss is read at the last step. How much does it say about what the network should have done 30 steps earlier?',
      { topic: 'rnn-gradients', frame: 0, caption: 'A cell of 8 numbers runs for 30 steps. Its W_h is a rotation scaled by 0.9: it stretches every direction by exactly 0.9.' },
      { math: '∂L/∂h_{t−1} = (∂L/∂h_t ⊙ (1 − h_t²)) · W_hᵀ        k steps back: about sᵏ times the size' },
      { topic: 'rnn-gradients', frame: 1, caption: 'The loss is the sum of the last state, so the gradient there is all ones: size 2.83.' },
      { predict: { q: 'Each step back multiplies the size by about 0.78. Roughly how large is the gradient 29 steps back?', options: ['Far below a hundredth of 2.83', 'About 2.83 × 0.78', 'About half of 2.83'], answer: 0, explain: '0.776 multiplied by itself 29 times is about 0.0006, so 2.83 becomes 0.0018.' } },
      { topic: 'rnn-gradients', frame: 5, caption: '29 steps back the gradient has size 0.0018. A mistake at the end cannot teach the first steps anything.' },
      { topic: 'rnn-gradients', frame: 5, input: { scale: 1.2, act: 'linear', seed: 2 }, caption: 'A linear cell stretched by 1.2: every step multiplies by exactly 1.200, and 29 steps back the size is 560. It has exploded.' },
      { topic: 'rnn-gradients', frame: 6, input: { scale: 1.2, act: 'linear', seed: 2 }, caption: 'Stretches 0.8, 1.0 and 1.2 on a log scale: three straight lines, 0.0044, 2.8 and 560 after 29 steps.' },
      { topic: 'rnn-gradients', frame: 7, caption: 'The cures: clip the gradient’s size when it explodes; for vanishing, a cell with a memory updated by adding (next lesson).' },
      { h: 'Recap' },
      { list: ['Backpropagation through time multiplies one factor per step.', 'A factor below one makes the gradient vanish, above one explode, both like sᵏ.', 'Clipping caps an exploding gradient; vanishing needs gated cells.'] },
      X('sq-x-grad-factor'), X('sq-x-grad-curve'), X('sq-x-tanh-factor'), X('sq-x-clip'), X('sq-x-cure-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: gated cells
  // =========================================================================================================
  C.lesson({
    id: 'sq-lstm', chapter: 'sq-gates', title: 'LSTM: gates that decide what to keep', minutes: 8,
    summary: 'An LSTM keeps a memory c beside its output h. Three gates, each a sigmoid between 0 (shut) and 1 (open), one value per memory slot, decide: the forget gate f how much of the old memory to keep, the input gate i how much of a new candidate g to write, the output gate o how much of the memory to show. The memory is updated by adding, c_t = f ⊙ c_{t−1} + i ⊙ g, so the gradient along it is multiplied only by f.',
    blocks: [
      'A plain cell rewrites its whole state at every step, and its gradient fades. What if the cell could choose, slot by slot, to keep its memory untouched?',
      { math: 'f, i, o = σ(x W + h_{t−1} U + b)   (one set of weights per gate)\ng = tanh(x W_g + h_{t−1} U_g + b_g)\nc_t = f ⊙ c_{t−1} + i ⊙ g\nh_t = o ⊙ tanh(c_t)' },
      { topic: 'lstm-gates', frame: 0, caption: 'An LSTM with 4 memory slots counting the ones of a bit stream; trained for 800 steps, it counts every step of new streams right.' },
      { topic: 'lstm-gates', frame: 1, caption: 'Step 1 reads 1. In slot 2 the input gate is open (0.98) and the candidate is 0.98: the memory goes from 0.00 to 0.97.' },
      { predict: { q: 'Step 2 reads 0. What should slot 2’s forget gate do?', options: ['Stay near 1, so the count is kept', 'Drop near 0, so the memory is cleared', 'Stay near 0.5'], answer: 0, explain: 'The count must survive the zero: the forget gate is 0.98 and the memory stays near 1.' } },
      { topic: 'lstm-gates', frame: 4, caption: 'Step 4, another 1: forget gate 0.99 keeps 1.82, the input gate adds about 1: the memory becomes 2.80.' },
      { topic: 'lstm-gates', frame: 9, caption: 'Slot 2 is the counter: it rises by 0.95 on average when a 1 arrives and moves by 0.45 on average, up or down, on a 0 (from −0.48 to 0.74).' },
      { topic: 'lstm-gates', frame: 9, input: { bits: '10000001' }, caption: 'Six zeros in a row: the memory rises by 0.97 on average when a 1 arrives and changes by 0.68 on a 0, and all 8 counts are still right.' },
      { tip: 'Along the memory, one step back multiplies the gradient by f only, not by a weight matrix and a tanh slope. With f near 1 it can travel back many steps.' },
      { h: 'Recap' },
      { list: ['Forget gate: how much old memory to keep. Input gate: how much new candidate to add. Output gate: how much to show.', 'The memory is updated by adding, c = f ⊙ c + i ⊙ g.', 'With the forget gate near 1, memory and gradient both survive many steps.'] },
      X('sq-x-lstm-cell'), X('sq-x-lstm-keep'), X('sq-x-lstm-path'),
    ],
  });

  C.lesson({
    id: 'sq-gru', chapter: 'sq-gates', title: 'GRU: one gate to keep or rewrite', minutes: 7,
    summary: 'A GRU has no separate memory: its state h is the memory. An update gate z mixes the old state with a candidate n, h_t = z ⊙ h_{t−1} + (1 − z) ⊙ n, so z near 1 holds the state and z near 0 rewrites it. A reset gate r decides how much of the old state the candidate may read. Two gates and three blocks of weights, against an LSTM’s three gates and four blocks.',
    blocks: [
      'The LSTM has a memory and three gates. Is all of it needed, or can a simpler cell hold a value just as well?',
      { math: 'r = σ(x W_r + h U_r + b_r)        z = σ(x W_z + h U_z + b_z)\nn = tanh(x W_n + r ⊙ (h U_n) + b_n)\nh_t = z ⊙ h_{t−1} + (1 − z) ⊙ n' },
      { topic: 'gru-gates', frame: 0, caption: 'A GRU with 4 numbers of state that says which letter came last in a stream of letters and blanks.' },
      { topic: 'gru-gates', frame: 1, caption: 'Step 1 reads A. For number 2 the update gate is 0.01: it keeps almost none of the old 0.00 and takes the candidate 1.00, giving 0.99.' },
      { topic: 'gru-gates', frame: 2, caption: 'A blank: the update gate is 0.99, so the state is held at 0.99.' },
      { predict: { q: 'Step 4 reads the letter B. What will number 2’s update gate do?', options: ['Drop near 0 to write the new letter', 'Stay near 1 to hold A', 'Go to 0.5'], answer: 0, explain: 'A new letter must replace the old one: the gate is 0.00 and the state goes from 0.99 to −0.99, the candidate.' } },
      { topic: 'gru-gates', frame: 4, caption: 'B arrives: the gate drops to 0.00 and the state is rewritten to −0.99.' },
      { topic: 'gru-gates', frame: 11, caption: 'Number 2’s update gate averages 0.01 when a letter arrives and 0.99 over blanks: it opens to write and closes to hold.' },
      { topic: 'gru-gates', frame: 11, input: { stream: 'B--------C' }, caption: 'Eight blanks in a row: the gate averages 1.00 over them and the answer is right at every step.' },
      { h: 'Recap' },
      { list: ['The GRU’s update gate z keeps the old state (z near 1) or takes the candidate (z near 0).', 'The reset gate decides how much of the old state the candidate reads.', 'Fewer gates and weights than an LSTM; often just as good.'] },
      X('sq-x-gru-update'), X('sq-x-gru-vs-lstm'), X('sq-x-code-gru'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: sequence models at work
  // =========================================================================================================
  C.lesson({
    id: 'sq-nextchar', chapter: 'sq-seq', title: 'Predicting the next character', minutes: 7,
    summary: 'A character model reads text one character at a time and, at every step, gives a probability to each of the 27 characters that might come next. Its loss at a step is −ln of the probability it gave the character that came; the mean loss over a text, raised as e^loss, is the perplexity: how many equally likely characters it is choosing among, on average.',
    blocks: [
      'After “th”, almost anyone would bet on “e”. Can a small recurrent network learn such bets from text alone?',
      { topic: 'next-char', frame: 0, caption: 'An LSTM over 27 characters, trained for 1500 steps on 15076 characters of our sentences: it guesses 87% of held-out characters right.' },
      { topic: 'next-char', frame: 2, caption: 'After “th”: “e” at 0.99. The loss is −ln 0.99 = 0.01: almost nothing to learn here.' },
      { topic: 'next-char', frame: 4, caption: 'After “the·”: its best guess is “m” at 0.15, and the real “o” got only 0.04: a loss of 3.12. A new word is hard to guess.' },
      { predict: { q: 'After “the·o”, how sure will it be of the next letter?', options: ['Very sure: “l”, for “old”', 'Unsure: many words start with o', 'It cannot tell'], answer: 0, explain: 'In this corpus a word after “the” that starts with o is nearly always “old”: “l” gets 0.98.' } },
      { topic: 'next-char', frame: 5, caption: '“l” at 0.98. Once a word has begun, the rest is easy.' },
      { math: 'loss = mean of −ln p(the real next character)        perplexity = e^loss' },
      { topic: 'next-char', frame: 18, caption: 'Over 16 characters the mean loss is 0.44: perplexity e^0.44 = 1.55, against 27 for blind guessing.' },
      { h: 'Recap' },
      { list: ['A next-character model gives a probability to every character at every step.', 'Its loss is −ln of the probability of what came; low where text is predictable.', 'Perplexity e^loss: the number of equally likely choices it is torn between.'] },
      X('sq-x-char-loss'), X('sq-x-perplexity'),
    ],
  });

  C.lesson({
    id: 'sq-sampling', chapter: 'sq-seq', title: 'Writing text: greedy, sampling and temperature', minutes: 6,
    summary: 'To write, the model feeds each character it chooses back in as the next input. Greedy writing always takes the likeliest character, and soon repeats itself. Sampling draws each character by its probability. Temperature divides the scores before the softmax: below 1 it sharpens the choice towards greedy, above 1 it flattens it towards random.',
    blocks: [
      'The model gives probabilities, not a text. How does it write, and how adventurous should it be?',
      { topic: 'next-char', frame: 19, caption: 'Greedy: always the likeliest character. It continues “the old king left” with “ the cottage and he wore”.' },
      { math: 'p_k = exp(z_k / T) / Σⱼ exp(z_j / T)        T: the temperature' },
      { predict: { q: 'Sampling at temperature 1.6 instead. What will the text look like?', options: ['More varied, with misspelt words', 'Exactly the greedy text', 'Only spaces'], answer: 0, explain: 'A high temperature flattens the probabilities, so unlikely characters get drawn: “·old·kinted·her·doa·chee”.' } },
      { topic: 'next-char', frame: 13, input: { prompt: 'she is the', temp: 1.6, seed: 2 }, caption: 'Temperature 1.6: the characters drawn had probability 0.64 on average, and the spelling falls apart.' },
      { topic: 'next-char', frame: 20, input: { temp: 0.2 }, caption: 'Temperature 0.2: the drawn characters had probability 1.00 on average; the text is the greedy one.' },
      { h: 'Recap' },
      { list: ['Writing feeds each chosen character back in.', 'Greedy takes the likeliest; sampling draws by probability.', 'Temperature below 1 sharpens, above 1 flattens.'] },
      X('sq-x-temp'), X('sq-x-temp-choice'), X('sq-x-code-temp'),
    ],
  });

  C.lesson({
    id: 'sq-seq2seq', chapter: 'sq-seq', title: 'Sequence to sequence: encode, then decode', minutes: 7,
    summary: 'When the output is a sequence of its own length, an encoder reads the whole input and hands its last state, the context, to a decoder that writes the output one token at a time, each fed back in. Everything about the input must pass through that one vector. Past the lengths it trained on it fails, which is what attention later fixes.',
    blocks: [
      '“three one four one” in, “3141” out: input and output have their own lengths and their own vocabularies. How can a recurrent network map one to the other?',
      { topic: 'seq2seq', frame: 0, caption: 'Number words in, digits out. The encoder and the decoder are each a GRU with 32 numbers of state, trained together for 1500 steps on inputs of one to 5 words.' },
      { topic: 'seq2seq', frame: 5, caption: 'The encoder’s last state is the context: 32 numbers that must hold all 4 words and their order.' },
      { topic: 'seq2seq', frame: 6, caption: 'The decoder starts from the context with the start mark and writes “3” (0.99).' },
      { topic: 'seq2seq', frame: 10, caption: 'Each digit is fed back in; after “1” it writes the end mark (0.97).' },
      { predict: { q: 'It was trained on inputs of one to 5 words. How will it do on inputs of 8 words?', options: ['Almost never exactly right', 'As well as at 4 words', 'A little worse than at 5 words'], answer: 0, explain: 'One vector of 32 numbers, never trained to hold 8 words: 0% exactly right.' } },
      { topic: 'seq2seq', frame: 11, caption: '97% exactly right at 4 words, 0% at 8.' },
      { topic: 'seq2seq', frame: 11, input: { digits: '5830271' }, caption: 'Seven words: “5” and “8” come out right, then it loses track and writes “1” where “3” should be.' },
      { h: 'Recap' },
      { list: ['An encoder reads the input; its last state, the context, starts the decoder.', 'The decoder writes one token at a time, each fed back in, until the end mark.', 'One fixed vector is a bottleneck for long inputs; attention lets the decoder look back.'] },
      X('sq-x-s2s-curve'), X('sq-x-bottleneck-choice'),
    ],
  });

  C.lesson({
    id: 'sq-order', chapter: 'sq-seq', title: 'When word order decides: a recurrent model on the canvas', minutes: 7,
    summary: '“Good at first, but it broke in a week” uses friendly words and is a bad review. A bag of words counts the words and loses their order; a recurrent network reads them in order and can follow the turn. On the canvas’s reviews with a turn, word counts with logistic regression reach an accuracy of 0.71 on 180 held-out reviews, and a simple RNN 0.84, as long as it reads to the end.',
    blocks: [
      'Two reviews hold exactly the same words in a different order, and say opposite things. Which models can tell them apart?',
      { list: ['“the soup was cold, but the staff were kind” and “the staff were kind, but the soup was cold” leave a reader with different verdicts.', 'Word counts give both the same row: any model that reads only counts must give them the same answer.', 'A recurrent network reads “but” and then what follows it, so the end of the review can weigh most.'] },
      { topic: 'gru-gates', frame: 10, input: { stream: 'B--------C' }, caption: 'What a gated cell does at a turn: it holds B over eight blanks (update gate near 1), then rewrites its state the moment C arrives (gate 0.01). A review’s “but” calls for the same move.' },
      { predict: { q: 'A recurrent block reads only the first 8 words of each review. Will it do better than word counts?', options: ['No: the turn usually comes later and it never sees it', 'Yes: any recurrent network beats word counts', 'They will be exactly equal'], answer: 0, explain: 'With 8 words the simple RNN reaches only 0.62 on held-out reviews; reading 24 words it reaches 0.84.' } },
      { workflow: S.WF.build({ blocks: { data: ['data', { set: 'opinions' }], split: ['split', { test: 0.25 }], rnn: ['rnn', { cell: 'RNN' }], train: ['train', { epochs: 10 }], ev: ['evaluate', {}] }, wires: [['data.out', 'split.in'], ['rnn.model', 'train.model'], ['split.train', 'train.data'], ['train.model', 'ev.model'], ['split.test', 'ev.data']] }, 'Reviews with a turn'), title: 'A simple RNN reading reviews with a turn', text: 'Open it on the canvas, train, and open the RNN block: one review’s hidden state drawn word by word. Then set the longest text read to 8 and train again.' },
      { tip: 'Order is not everything: word counts are quick, need little data and are easy to explain. Reach for a sequence model when the meaning depends on order: negations, turns, what comes last.' },
      { h: 'Recap' },
      { list: ['A bag of words throws order away; the same words in another order give the same row.', 'A recurrent network reads in order and can weigh what comes after a turn.', 'It must read far enough: a text cut short loses the end that decides.'] },
      X('sq-x-model-choice'), X('sq-x-wf-maxlen'), { challenge: 'sq-ch-turns' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
