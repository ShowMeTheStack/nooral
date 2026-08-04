// Parts, families and topics. A part is one of the eighteen parts of the course; a family is a group of topics on
// one idea (k-means, the decision tree, attention); a topic is one picture with its own inputs and presets, a scene
// for every step with narration that names the actual values, predictions to make on the way and a quiz at the end.
(function (S) {
  'use strict';

  /** The eighteen parts, in teaching order (the home page lists the same) */
  S.parts = [
    { id: 'data', n: 1, title: 'Data and features', blurb: 'Tables, rows and features; statistics and distributions; missing values and outliers; scaling, encoding and binning; polynomial features; and leakage, shown happening.' },
    { id: 'evaluation', n: 2, title: 'Splitting and evaluation', blurb: 'Train, validation and test splits, cross-validation drawn fold by fold, the confusion matrix you can click, ROC and PR curves with a live threshold, calibration, regression errors, learning curves and hyperparameter search.' },
    { id: 'linear', n: 3, title: 'Linear models', blurb: 'Least squares as a projection, the closed form against gradient descent, polynomial regression and overfitting, ridge and lasso paths, logistic and softmax regression, and the perceptron.' },
    { id: 'neighbours', n: 4, title: 'Nearest neighbours and probabilistic models', blurb: 'k-NN with its Voronoi regions and the curse of dimensionality, naive Bayes (Gaussian ellipses and word counts), and linear and quadratic discriminant analysis.' },
    { id: 'trees', n: 5, title: 'Trees and ensembles', blurb: 'A tree growing split by split with every candidate weighed, pruning, bagging and random forests, AdaBoost reweighting its mistakes, and gradient boosting fitting residuals.' },
    { id: 'svm', n: 6, title: 'Support vector machines', blurb: 'The widest margin and its support vectors, the soft margin and the hinge loss, kernels as a lift into 3-D, SMO pair by pair, multi-class schemes and the ε-tube of SVR.' },
    { id: 'unsupervised', n: 7, title: 'Unsupervised learning', blurb: 'k-means step by step, hierarchical clustering with a dendrogram you can cut, DBSCAN, Gaussian mixtures by EM, PCA, t-SNE and UMAP-style maps, and anomaly detection.' },
    { id: 'optimisation', n: 8, title: 'Optimisation', blurb: 'Loss surfaces, gradient descent and its stochastic cousins, momentum, AdaGrad, RMSProp and Adam raced on saddles and ravines, schedules and warm-up.' },
    { id: 'networks', n: 9, title: 'Neural networks', blurb: 'Neurons and activations, an MLP drawn live, forward values, backpropagation with every gradient shown, initialisation, regularisation and normalisation, and the playground.' },
    { id: 'cnn', n: 10, title: 'Convolutional networks', blurb: 'The sliding kernel and its dot products, padding, stride and pooling, channels and feature maps, learned filters, receptive fields, a small network trained here, and saliency.' },
    { id: 'sequences', n: 11, title: 'Sequences and embeddings', blurb: 'One-hot vectors and trained word embeddings, RNNs unrolled through time, vanishing gradients, LSTM and GRU gates step by step, and sequence-to-sequence.' },
    { id: 'transformers', n: 12, title: 'Transformers', blurb: 'Tokens and positions, scaled dot-product attention with every matrix shown, masks, multi-head attention, the full block, and a tiny transformer learning a toy task.' },
    { id: 'llm', n: 13, title: 'Language models', blurb: 'Byte-pair encoding merge by merge, next-token prediction and perplexity, a tiny GPT trained here, sampling with temperature, top-k and top-p, the KV cache, and what larger models add.' },
    { id: 'generative', n: 14, title: 'Generative models', blurb: 'Autoencoders and their bottleneck, VAEs with a latent map you can walk, a GAN duel and mode collapse, and diffusion step by step.' },
    { id: 'rl', n: 15, title: 'Reinforcement learning', blurb: 'Bandits and regret, Markov decision processes, value and policy iteration on a grid, Q-learning and SARSA, policy gradients on a pole, and deep Q-learning.' },
    { id: 'graphs', n: 16, title: 'Graph neural networks', blurb: 'Graphs and adjacency, message passing step by step, node classification, over-smoothing and graph-level readout.' },
    { id: 'responsible', n: 17, title: 'Robustness, fairness and privacy', blurb: 'Why models are brittle and how to harden them, distribution shift, fairness metrics and mitigation, and differential privacy: each weakness with its defence.' },
    { id: 'practice', n: 18, title: 'Practice and interpretability', blurb: 'Imbalanced data, model selection, permutation importance, partial dependence, SHAP-style and LIME-style explanations, and a checklist for a real project.' },
  ];
  S.partOf = (id) => S.parts.find((p) => p.id === id) || null;
  S.families = [];
  S.topics = [];

  const partRank = (id) => { const i = S.parts.findIndex((p) => p.id === id); return i < 0 ? 99 : i; };
  /** family: {id, part, title, blurb, order} */
  S.family = function (f) {
    if (S.families.some((x) => x.id === f.id)) return S.families.find((x) => x.id === f.id);
    S.families.push(f);
    S.families.sort((a, b) => partRank(a.part) - partRank(b.part) || (a.order || 0) - (b.order || 0));
    return f;
  };
  /**
   * topic: {id, family, title, blurb, order,
   *   inputs: [{id, kind: 'string'|'choice'|'number'|'slider'|'list'|'text'|'program', label, def,
   *             options: [{value, label}] (choice), min, max (number, slider; length for string and list),
   *             step, log (slider: the slider moves the exponent, base 10), fmt(v) (slider: the value shown),
   *             integer: false (number: decimals allowed),
   *             alphabet (string: allowed symbols), check(v, input) -> error sentence or null}],
   *   presets: [{label, values}],
   *   run(input) -> [frame] or {frames, def, counters},
   *   counters: [{id, label}], predict: [...], quiz: [{q, options, answer, why}], camera: 'fit-all'|'follow'}
   * A frame is {items: [stage items], say: 'narration', hl: [ids lit in the definition panel],
   *   picks: {item key: info} (or item.info): clicking that item shows the info under the narration, e.g. a
   *   confusion cell's examples; info = {title, text, lines, cols, rows: [[..]], images: [{px (0..15), w, h, label}]};
   *   give such items the class 'clickable',
   *   counters: {id: n}, predict: {ask, options, answer, explain, pick: 'option'|'item'}, mark, slow, view}.
   */
  S.topic = function (t) {
    if (S.topics.some((x) => x.id === t.id)) throw new Error('two topics with id ' + t.id);
    S.topics.push(t);
    const order = (x) => (x.order === undefined ? 500 : x.order);
    const n = S.topics.map((x, k) => [x, k]);
    const fr = (x) => { const f = S.families.find((y) => y.id === x.family); return f ? partRank(f.part) * 1000 + (f.order || 0) : 99999; };
    n.sort((a, b) => fr(a[0]) - fr(b[0]) || order(a[0]) - order(b[0]) || a[1] - b[1]);
    S.topics.length = 0;
    for (const [x] of n) S.topics.push(x);
    return t;
  };
  S.byId = (id) => S.topics.find((t) => t.id === id) || null;
  // A link to another topic inside narration or a definition: [[topic-id]] (shows its title) or [[topic-id|words]].
  S.TOPIC_LINK = /\[\[([a-z0-9][a-z0-9-]*)(?:\|([^\]]+))?\]\]/g;
  /** split text at its topic links: strings and {id, text} pieces; the text is the words given, else the title */
  S.topicLinks = function (text) {
    const out = [], re = new RegExp(S.TOPIC_LINK.source, 'g');
    let last = 0, m;
    text = String(text === undefined || text === null ? '' : text);
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const t = S.byId(m[1]);
      out.push({ id: m[1], text: m[2] || (t ? '“' + t.title + '”' : m[1]), missing: !t });
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  };
  /** the same text with each topic link as plain words */
  S.plainLinks = (text) => S.topicLinks(text).map((x) => (typeof x === 'string' ? x : x.text)).join('');
  S.familyOf = (t) => S.families.find((f) => f.id === t.family) || null;
  S.topicPart = (t) => { const f = S.familyOf(t); return f ? S.partOf(f.part) : null; };

  /** The value of each input: the one given, or its default */
  S.inputOf = function (t, given) {
    const o = {};
    for (const i of t.inputs || []) o[i.id] = given && given[i.id] !== undefined ? given[i.id] : (typeof i.def === 'function' ? i.def() : i.def);
    return o;
  };
  /** Read a field's text into a value of the input's kind */
  S.parseInput = function (inp, text) {
    switch (inp.kind) {
      case 'number': return text === '' ? NaN : Number(text);
      case 'slider': return Number(text);
      case 'list': return String(text).split(/[\s,]+/).filter(Boolean).map((x) => (x === 'ε' || x === 'eps' ? '' : x));
      case 'string': return String(text).replace(/\s+/g, '') === 'ε' ? '' : String(text).replace(/\s+/g, inp.spaces ? ' ' : '');
      default: return String(text);
    }
  };
  S.showInput = function (inp, v) {
    if (inp.kind === 'list') return (v || []).map((x) => (x === '' ? 'ε' : x)).join(', ');
    return v === undefined || v === null ? '' : String(v);
  };
  /** Check an input against the topic's limits: an error message, or null */
  S.checkInput = function (t, input) {
    for (const i of t.inputs || []) {
      const v = input[i.id];
      const L = i.label || i.id;
      if (i.kind === 'number') {
        if (!Number.isFinite(v) || (i.integer !== false && !Number.isInteger(v))) return L + ': a whole number';
        if (i.min !== undefined && v < i.min) return L + ': at least ' + i.min;
        if (i.max !== undefined && v > i.max) return L + ': at most ' + i.max;
      }
      if (i.kind === 'string') {
        if (typeof v !== 'string') return L + ': a string';
        if (i.alphabet) { const bad = [...v].find((c) => !i.alphabet.includes(c)); if (bad !== undefined) return L + ': "' + bad + '" is not in the alphabet {' + [...i.alphabet].join(', ') + '}'; }
        if (i.max !== undefined && [...v].length > i.max) return L + ': at most ' + i.max + ' symbols, so the picture stays readable';
        if (i.min !== undefined && [...v].length < i.min) return L + ': at least ' + i.min + ' symbols';
      }
      if (i.kind === 'list') {
        if (!Array.isArray(v)) return L + ': a list';
        if (i.max !== undefined && v.length > i.max) return L + ': at most ' + i.max + ' items';
        if (i.min !== undefined && v.length < i.min) return L + ': at least ' + i.min + ' item' + (i.min === 1 ? '' : 's');
      }
      if (i.kind === 'choice' && i.options && !i.options.some((o) => o.value === v)) return L + ': choose one of the options';
      if (i.kind === 'slider') {
        if (!Number.isFinite(v)) return L + ': a number';
        if (i.min !== undefined && v < i.min) return L + ': at least ' + i.min;
        if (i.max !== undefined && v > i.max) return L + ': at most ' + i.max;
      }
      if (i.check) { let e; try { e = i.check(v, input); } catch (x) { e = x.message; } if (e) return e; }
    }
    if (t.check) return t.check(input);
    return null;
  };

  /** A helper for building a topic's frames: keeps the running counters and stamps them on each frame */
  class Frames {
    constructor(counters) { this.list = []; this.counts = {}; for (const c of counters || []) this.counts[typeof c === 'string' ? c : c.id] = 0; }
    count(id, n) { this.counts[id] = (this.counts[id] || 0) + (n === undefined ? 1 : n); return this; }
    set(id, n) { this.counts[id] = n; return this; }
    add(frame) { frame.counters = Object.assign({}, this.counts, frame.counters || {}); this.list.push(frame); return frame; }
    get length() { return this.list.length; }
    last() { return this.list[this.list.length - 1]; }
  }
  S.Frames = Frames;

  /** Run a topic with an input: frames (each drawable on the stage), the definition panel and the counters */
  S.run = function (t, input) {
    let out;
    try { out = t.run(input); } catch (e) { return { ok: false, error: e.message || String(e), frames: [{ items: [], say: 'This could not be run: ' + (e.message || e) }], stack: e.stack }; }
    if (Array.isArray(out)) out = { frames: out };
    out = out || { frames: [] };
    const frames = (out.frames || []).filter(Boolean);
    for (const f of frames) { if (!f.items) f.items = []; if (f.items.some((x) => !x)) f.items = f.items.filter(Boolean); }
    // one camera for the whole run, so nothing jumps: the box around every frame (a frame may set its own)
    if (t.camera !== 'follow') {
      let box = null;
      for (const f of frames) { const b = S.stageBounds(f.items, t.cameraOpts); if (b) box = box ? S.geom.union([box, b]) : b; }
      if (box) for (const f of frames) if (!f.view) f.view = box;
    }
    // predictions declared on the topic: find the frames they ask at
    // a prediction that cannot be worked out is reported (a topic checker fails on it), not dropped quietly
    const predictErrors = [];
    for (const p of t.predict || []) {
      for (let i = 0; i < frames.length - 1; i++) {
        if (frames[i].predict) continue;
        let hit = false;
        try { hit = p.at(frames[i], i, frames, input); } catch (e) { hit = false; predictErrors.push('prediction ' + (t.predict.indexOf(p) + 1) + ': at() fails at frame ' + i + ': ' + (e.message || e)); }
        if (!hit) continue;
        let q = null;
        try {
          q = { ask: typeof p.ask === 'function' ? p.ask(frames, i, input) : p.ask, answer: typeof p.answer === 'function' ? p.answer(frames, i, input) : p.answer, options: p.options ? (typeof p.options === 'function' ? p.options(frames, i, input) : p.options) : null, explain: p.explain ? (typeof p.explain === 'function' ? p.explain(frames, i, input) : p.explain) : null, pick: p.pick || (p.options ? 'option' : 'item') };
        } catch (e) { q = null; predictErrors.push('prediction ' + (t.predict.indexOf(p) + 1) + ' at frame ' + i + ': ' + (e.message || e)); }
        if (!q || q.answer === null || q.answer === undefined) continue;
        frames[i].predict = q;
        if (p.once) break;
      }
    }
    return { ok: true, frames, def: out.def || (t.def ? t.def(input) : null), counters: out.counters || t.counters || [], result: out.result, predictErrors };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
