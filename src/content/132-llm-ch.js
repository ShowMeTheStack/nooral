// Part 13, language models: the guided tour, the course, and the pages the diagram checker looks at. The workflow
// canvas has no block that trains a language model and no metric for next-token loss on hidden text, so this part's
// model challenges are the two code exercises at the end of its first two chapters (131-llm-ex.js): a tokenizer that
// must cut the held-out tenth of the harbour text into at most 425 tokens with 40 merges, and a counting model that
// must reach a held-out loss of 1.8 nats a character from two characters of context.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.courses.has('ll-course')) return;

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'll-tour', title: 'Tour: language models', blurb: 'Tokens, next-token prediction, a tiny GPT, sampling rules and the text lab, in eight steps.',
    steps: [
      { route: 'part=llm', target: '#main', text: 'Part 13 of the catalogue: language models. Every topic runs a tiny GPT trained here on a text written for Nooral, rebuilt from its stored weights, so every probability shown is the model’s own.' },
      { route: 'topic=bpe-build', target: '#stageWrap', text: 'Byte-pair encoding learns its tokens by counting: the commonest pair of neighbours is merged, merge after merge.' },
      { target: '#btnPlay', text: 'Play runs the merges one by one; the arrow keys step forwards or back.' },
      { route: 'topic=lm-gpt-training', target: '#stageWrap', text: 'The tiny GPT in training: the loss on held-out text falls, and the samples go from noise to words.' },
      { route: 'topic=sampling-topk-topp', target: '#inputs', text: 'The sampling rules have their inputs here: change k, p or the temperature and watch which characters survive.' },
      { route: 'textlab/sample', target: '#tlView', text: 'The text lab trains a larger GPT in this browser and writes with it one character at a time, with the probability bars before and after the rule.' },
      { route: 'lesson=ll-ngram', target: '#lessonView', text: 'Lessons run the topics at the frames that matter and ask for a prediction on the way: here, counting models against a neural one.' },
      { route: 'learn', target: '[data-chapter="ll-tokens"]', text: 'The lessons and exercises of this part start here, with two small challenges: a tokenizer within 40 merges and a counting model under 1.8 nats.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'll-course', title: 'Language models', blurb: 'All thirteen topics of part 13, from byte-pair encoding to the limits of a model that writes, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'll-bpe', note: 'Tokens learned by counting.' },
      { topic: 'bpe-build', note: 'Merge by merge.' },
      { exercise: 'll-x-bpe-table', note: 'Three merges by hand.' },
      { topic: 'bpe-tokenise', note: 'A new sentence cut.' },
      { exercise: 'll-x-cut-word', note: 'Cut a word yourself.' },
      { lesson: 'll-ngram', note: 'Counting models.' },
      { exercise: 'll-x-bigram', note: 'A bigram probability.' },
      { topic: 'lm-next-token', note: 'A network predicts.' },
      { topic: 'lm-cross-entropy', note: 'Scoring the guesses.' },
      { exercise: 'll-x-ce-table', note: 'Cross-entropy by hand.' },
      { topic: 'lm-gpt-training', note: 'The tiny GPT learns.' },
      { exercise: 'll-x-train-curve', note: 'Sketch its loss.' },
      { topic: 'lm-scaling', note: 'Wider is better.' },
      { lesson: 'll-temp', note: 'Choosing the next token.' },
      { topic: 'sampling-temperature', note: 'Greedy and temperature.' },
      { topic: 'sampling-topk-topp', note: 'Cutting the tail.' },
      { exercise: 'll-x-topp', note: 'The size of a nucleus.' },
      { topic: 'kv-cache-generation', note: 'Writing without recomputing.' },
      { exercise: 'll-x-kv-count', note: 'Count the work.' },
      { lesson: 'll-stages', note: 'From predictor to assistant.' },
      { topic: 'lm-training-stages', note: 'Three stages, one network.' },
      { topic: 'lm-in-context', note: 'Learning from the prompt.' },
      { exercise: 'll-x-attn', note: 'Where a query looks.' },
      { topic: 'lm-limits', note: 'Invented facts and copied bias.' },
      { exercise: 'll-x-defence', note: 'What defends against them.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson ll-bpe')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'llm').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['ll-x-bpe-table', 'll-x-cut-word', 'll-x-ch-bpe', 'll-x-ce-table', 'll-x-train-curve', 'll-x-scale-curve', 'll-x-ch-ngram', 'll-x-temp-table', 'll-x-code-cut', 'll-x-stages', 'll-x-attn', 'll-x-defence'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course ll-course', hash: 'course=ll-course', ready: "document.querySelector('#courseView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
