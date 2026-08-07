// Model challenges and workflow tasks: a dataset, limits, and a target score on a hidden test set. S.Challenges holds
// what the pages and the checkers share: the limits checked against a workflow, the scoring (at once in node, through
// the workflow's worker in the page), medals, the reader's attempts kept in this browser, and the content check.
//   S.Challenges.limitsProblems(doc, x) -> [sentences]        S.Challenges.limitSentences(x) -> [sentences]
//   S.Challenges.evaluate(doc, x) -> {ok, score, reached, problems, medal}   (sync: node, the content checker)
//   S.Challenges.evaluateAsync(doc, x) -> Promise of the same (the page: S.WF.Remote, scored in the worker)
//   S.Challenges.solved(id), best(id), attempts(id), record(id, result)
//   S.Challenges.lint(x, {verbose}) -> {problems, notes, answers}   (tools/check-content.mjs)
//   S.Challenges.hiddenCount(x) -> the number of held-out rows the score is taken on
// x is a challenge (S.Content.challenge) or a workflow exercise: {data, task, metric, target, marks, limits, hidden}.
// limits: {blocks, models, params: {type: {param: [lo, hi]}} (a 'layers' parameter: every layer), epochs, maxBlocks,
//   neurons: n (at most n hidden neurons in all, over every MLP block), weights: n (at most n weights and biases in
//   each network block, counted by building it with S.DL), shape: {inputs} | {size}, and classes (what it reads)}
(function (S) {
  'use strict';
  const store = () => (S.UI && S.UI.storage) || { get: (k, d) => d, set() {} };
  const WF = () => S.WF;
  const fmt = (v) => (v === null || v === undefined || !isFinite(v) ? '—' : String(+(+v).toFixed(3)));
  const metricOf = (x) => (WF().METRICS && WF().METRICS[x.metric]) || { label: x.metric, higher: true };
  /** a parameter's value as numbers: a 'layers' value [6, 6] (or "6, 6") gives each layer */
  const numbers = (v) => (Array.isArray(v) ? v : typeof v === 'string' ? v.split(/[\s,]+/).filter(Boolean) : [v]).map(Number);
  const hiddenOf = (b) => numbers(WF().defaults(b.type, b.params).hidden).filter((v) => v > 0);
  /** the number of weights and biases of a network block, built as its trainer builds it */
  const weightsOf = (b, shape) => {
    const p = WF().defaults(b.type, b.params);
    if (b.type === 'mlp') return S.DL.build('mlp', { inputs: shape.inputs, hidden: hiddenOf(b), outputs: shape.classes, seed: 1 }).numParams();
    if (b.type === 'cnn') return S.DL.build('cnn', { size: shape.size, channels: 1, classes: shape.classes, filters: [+p.f1, +p.f2], kernel: +p.kernel, hidden: [+p.hidden], seed: 1 }).numParams();
    return 0;
  };
  const counts = new Map();

  const C = (S.Challenges = {
    fmt,
    metricLabel: (x) => x.metricName || metricOf(x).label,
    higher: (x) => metricOf(x).higher !== false,
    /** is this score at least the target (or at most, for an error)? */
    reaches(x, score, target) { if (score === null || score === undefined || !isFinite(score)) return false; const t = target === undefined ? x.target : target; return C.higher(x) ? score >= t - 1e-12 : score <= t + 1e-12; },
    /** 'gold' | 'silver' | 'bronze' (the target) | null */
    medal(x, score) {
      if (!C.reaches(x, score)) return null;
      const m = x.marks || {};
      if (m.gold !== undefined && C.reaches(x, score, m.gold)) return 'gold';
      if (m.silver !== undefined && C.reaches(x, score, m.silver)) return 'silver';
      return 'bronze';
    },
    /** the goal in words: "F1 ≥ 0.85 on 400 hidden rows" */
    goal(x) { return C.metricLabel(x) + ' ' + (C.higher(x) ? '≥ ' : '≤ ') + x.target + (x.constraints && x.constraints.length ? ', with ' + x.constraints.map((c) => WF().constraintText(c)).join(' and ') : ''); },

    /** the limits as sentences a reader can follow */
    limitSentences(x) {
      const L = x.limits || {}, out = [], name = (t) => { const d = WF().def(t); return d ? d.label : t; };
      if (L.blocks) out.push('Blocks you may use: ' + L.blocks.filter((t) => t !== 'challenge-data').map(name).join(', ') + '.');
      if (L.models) out.push('Models you may train: ' + L.models.map(name).join(', ') + '.');
      for (const [t, ps] of Object.entries(L.params || {})) for (const [p, [lo, hi]] of Object.entries(ps)) {
        const d = WF().def(t), pd = d && d.params.find((q) => q.id === p);
        out.push(name(t) + ': ' + (pd ? pd.label.replace(/\s*\(.*\)$/, '').replace(/^./, (c) => c.toLowerCase()) : p) + (pd && pd.kind === 'layers' ? ', each layer' : '') + (lo !== null && lo !== undefined && hi !== null && hi !== undefined ? ' between ' + lo + ' and ' + hi : lo !== null && lo !== undefined ? ' at least ' + lo : ' at most ' + hi) + '.');
      }
      if (L.epochs) out.push('At most ' + L.epochs + ' epochs of training.');
      if (L.maxBlocks) out.push('At most ' + L.maxBlocks + ' blocks on the canvas.');
      if (L.neurons) out.push('At most ' + L.neurons + ' hidden neurons in all, over every layer.');
      if (L.weights) out.push('At most ' + L.weights.toLocaleString('en') + ' weights and biases in the network.');
      const n = C.hiddenCount(x);
      out.push('The score is taken on ' + (n ? n.toLocaleString('en') : 'fresh') + ' hidden rows that the workflow never sees.');
      return out;
    },
    /** the held-out rows the score uses: hidden.n, or (n: 0, all that is held out) as many as the hidden set has */
    hiddenCount(x) {
      const hd = x.hidden || {}, d = x.data || {};
      if (hd.n) return hd.n;
      const key = JSON.stringify([d, hd]);
      if (!counts.has(key)) { let n = null; try { n = WF().hiddenSet(d, hd).y.length; } catch (e) { /* the set is not loadable here */ } counts.set(key, n); }
      return counts.get(key);
    },
    sizeLimits: { hiddenOf, weightsOf },
    /** what in a workflow breaks the limits: [sentences] (empty when it keeps to them) */
    limitsProblems(doc, x) {
      const L = x.limits || {}, out = [], W = WF();
      const name = (b) => W.label(b);
      const data = doc.blocks.filter((b) => b.type === 'challenge-data');
      if (!data.length) out.push('the workflow has no Challenge data block: the task’s rows come from it');
      for (const b of data) {
        const p = b.params || {}, d = x.data || {};
        if (p.set !== d.set || (+p.seed || 1) !== (d.seed || 1) || (+p.n || 0) !== (d.n || 0) || JSON.stringify(p.opts || null) !== JSON.stringify(d.opts || null)) out.push('the Challenge data block was changed: it must give the task’s own rows');
      }
      for (const b of doc.blocks) {
        if (b.type === 'challenge-data') continue;
        if (L.blocks && !L.blocks.includes(b.type)) out.push(name(b) + ' is not one of the blocks this task allows');
        const def = W.def(b.type);
        if (L.models && def && def.cat === 'Models' && !L.models.includes(b.type)) out.push(name(b) + ' is not one of the models this task allows');
        for (const [p, [lo, hi]] of Object.entries((L.params || {})[b.type] || {})) {
          // a 'layers' parameter (the hidden layers of an MLP) is checked layer by layer
          const vs = numbers(W.defaults(b.type, b.params)[p]), pd = def && def.params.find((q) => q.id === p), many = vs.length > 1;
          const pn = pd ? pd.label.replace(/\s*\(.*\)$/, '').toLowerCase() : p;
          vs.forEach((v, i) => {
            const what = name(b) + ': ' + (many ? 'layer ' + (i + 1) + ' of the ' : 'the ') + pn;
            if (!Number.isFinite(v)) out.push(what + ' cannot be read as a number');
            else if (lo !== null && lo !== undefined && v < lo) out.push(what + ' is ' + v + ', below the limit of ' + lo);
            else if (hi !== null && hi !== undefined && v > hi) out.push(what + ' is ' + v + ', above the limit of ' + hi);
          });
        }
        if (L.epochs && b.type === 'train') {
          const src = (W.feeds(doc, b.id).model || [])[0], m = src && W.byId(doc, src), fam = m && W.families && W.families[m.type];
          const e = +W.defaults('train', b.params).epochs;
          if (fam && fam.iterative && e > L.epochs) out.push(name(b) + ' trains for ' + e + ' epochs; the limit is ' + L.epochs);
        }
      }
      if (L.maxBlocks && doc.blocks.length > L.maxBlocks) out.push('the workflow has ' + doc.blocks.length + ' blocks; the limit is ' + L.maxBlocks);
      if (L.neurons) {
        const n = doc.blocks.filter((b) => b.type === 'mlp').reduce((s, b) => s + hiddenOf(b).reduce((a, v) => a + v, 0), 0);
        if (n > L.neurons) out.push('the network has ' + n + ' hidden neurons; the limit is ' + L.neurons);
      }
      if (L.weights && L.shape) for (const b of doc.blocks.filter((q) => q.type === 'mlp' || q.type === 'cnn')) {
        const n = weightsOf(b, Object.assign({ classes: 2 }, L.shape));
        if (n > L.weights) out.push(name(b) + ' has ' + n.toLocaleString('en') + ' weights and biases; the limit is ' + L.weights.toLocaleString('en'));
      }
      return out;
    },
    /** the result of a scored run, as the pages and checkers use it */
    result(x, sc, limits) {
      const problems = limits.concat(sc.problems || []);
      // a constraint not met (or not measured) means the target is not reached, whatever the score
      const kept = !(sc.constraints || []).some((c) => !c.ok);
      const reached = !limits.length && kept && C.reaches(x, sc.score);
      return { ok: reached, reached, score: sc.score, n: sc.n, model: sc.model, problems, limits, constraints: sc.constraints || null, medal: limits.length || !kept ? null : C.medal(x, sc.score) };
    },
    /** score a workflow at once (node; small workflows on the page) */
    evaluate(doc, x) {
      const W = WF();
      doc = W.fromJSON(doc);
      const limits = C.limitsProblems(doc, x);
      const s = W.runSync(doc);
      return C.result(x, W.scoreHidden(s, { data: x.data, hidden: x.hidden || { seed: 7 }, metric: x.metric, constraints: x.constraints }), limits);
    },
    /** score a workflow in the workflow's worker (the page), so training does not freeze it */
    async evaluateAsync(doc, x, onEvent) {
      const W = WF();
      doc = W.fromJSON(doc);
      const limits = C.limitsProblems(doc, x);
      if (!W.Remote) return C.evaluate(doc, x);
      const r = new W.Remote({ onEvent });
      try {
        r.load(doc);
        await r.idle();
        return C.result(x, await r.fn('scoreHidden', { data: x.data, hidden: x.hidden || { seed: 7 }, metric: x.metric, constraints: x.constraints }), limits);
      } finally { r.dispose(); }
    },

    // ---- the reader's attempts, kept in this browser --------------------------------------------------------------
    all() { try { return Object.assign({}, store().get('challenges', {})); } catch (e) { return {}; } },
    attempts(id) { const a = C.all()[id]; return (a && a.attempts) || []; },
    best(id) {
      const x = S.Content && S.Content.challenges.get(id), list = C.attempts(id).filter((a) => !a.limits && a.score !== null && isFinite(a.score));
      if (!list.length) return null;
      return list.reduce((b, a) => (!b || (x && !C.higher(x) ? a.score < b.score : a.score > b.score) ? a : b), null);
    },
    solved(id) { return C.attempts(id).some((a) => a.reached); },
    /** keep an attempt: its rank among the reader's own attempts (1 is best) */
    record(id, res) {
      const all = C.all(), x = S.Content && S.Content.challenges.get(id);
      const a = all[id] = all[id] || { attempts: [] };
      const at = { t: Date.now(), score: res.score, reached: !!res.reached, medal: res.medal || null, limits: res.limits && res.limits.length ? res.limits.length : 0 };
      a.attempts.push(at);
      if (a.attempts.length > 50) a.attempts = a.attempts.slice(-50);
      try { store().set('challenges', all); } catch (e) { /* storage may be off: the result still shows */ }
      const ok = a.attempts.filter((q) => !q.limits && q.score !== null && isFinite(q.score));
      const better = (p, q) => (x && !C.higher(x) ? p.score < q.score : p.score > q.score);
      return { rank: at.limits || at.score === null ? null : 1 + ok.filter((q) => q !== at && better(q, at)).length, of: ok.length };
    },
    saveDraft(id, doc) { try { const all = C.all(); (all[id] = all[id] || { attempts: [] }).draft = doc; store().set('challenges', all); } catch (e) { /* off */ } },
    draft(id) { const a = C.all()[id]; return a && a.draft ? a.draft : null; },
    /** a fresh starting workflow: the task's starter, or the challenge data alone */
    starter(x) {
      const W = WF();
      if (x.starter) return W.fromJSON(typeof x.starter === 'function' ? x.starter() : x.starter);
      const doc = W.newDoc(x.title);
      const [type, params] = W.challengeBlock(x.data);
      doc.blocks.push({ id: 'data', type, params: W.defaults(type, params), x: 80, y: 160 });
      return doc;
    },

    // ---- the content check ----------------------------------------------------------------------------------------
    lint(x, opts) {
      const cons = (r) => (r.constraints || []).map((c) => '; ' + c.label.replace(/ ≤.*$/, '') + ' ' + fmt(c.value) + (c.ok ? '' : ' (broken)')).join('');
      const problems = [], notes = [];
      let answers = 0;
      if (!x.story) problems.push('no story');
      if (!WF().METRICS[x.metric]) problems.push('unknown metric ' + x.metric + ' (the metrics: ' + Object.keys(WF().METRICS).join(', ') + ')');
      if (!(x.data && x.data.set) || !(S.Data && S.Data.doc(x.data.set))) problems.push('no dataset ' + (x.data && x.data.set));
      if (!x.hidden || x.hidden.seed === undefined) problems.push('no hidden: {seed, n}');
      else if (x.data && x.hidden.seed === (x.data.seed || 1)) problems.push('the hidden seed is the public one: the hidden rows would be the training rows');
      if (problems.length) return { problems, notes, answers };
      for (const c of x.constraints || []) {
        if (!WF().CONSTRAINTS[c.kind]) problems.push('unknown constraint ' + c.kind + ' (the constraints: ' + Object.keys(WF().CONSTRAINTS).join(', ') + ')');
        const feats = (S.Data.load(x.data.set, Object.assign({}, x.data.opts || {}, { seed: 1 })).features || []).map((f) => f.name);
        if (!feats.includes(c.group)) problems.push('the constraint names a column ' + c.group + ' that the set does not have');
        if (!(c.max >= 0)) problems.push('the constraint has no max');
      }
      for (const t of Object.keys((x.limits && x.limits.params) || {})) if (!WF().def(t)) problems.push('the limits name a block type ' + t + ' that does not exist');
      for (const t of (x.limits && x.limits.blocks) || []) if (!WF().def(t)) problems.push('the limits allow a block type ' + t + ' that does not exist');
      if (x.limits && x.limits.weights && !x.limits.shape) problems.push('limits.weights needs limits.shape ({inputs} or {size}, and classes) to count a network’s weights');
      if (x.starter) { try { const d = C.starter(x); for (const p of WF().problems(d).filter((q) => !q.waiting)) problems.push('the starter: ' + p.msg); const lp = C.limitsProblems(d, x); if (lp.length) problems.push('the starter breaks the limits: ' + lp.join('; ')); } catch (e) { problems.push('the starter cannot be read: ' + e.message); } }
      if (!x.solution) problems.push('no reference solution');
      else {
        answers++;
        const r = C.evaluate(typeof x.solution === 'function' ? x.solution() : x.solution, x);
        if (!r.ok) problems.push('the reference solution does not reach ' + C.goal(x) + ': ' + (r.problems.length ? r.problems.join('; ') : 'it scores ' + fmt(r.score)));
        notes.push('solution: ' + metricOf(x).label + ' ' + fmt(r.score) + (r.medal ? ' (' + r.medal + ')' : '') + cons(r));
      }
      const wrong = x.wrong || [];
      if (wrong.length < 2) problems.push('at least two wrong workflows are needed (it has ' + wrong.length + ')');
      wrong.forEach((w, i) => {
        answers++;
        const r = C.evaluate(typeof w === 'function' ? w() : w, x);
        if (r.ok) problems.push('wrong workflow ' + (i + 1) + ' reaches the target: ' + fmt(r.score));
        notes.push('wrong ' + (i + 1) + ': ' + (r.problems.length ? r.problems.join('; ') : metricOf(x).label + ' ' + fmt(r.score)) + cons(r));
      });
      return { problems, notes, answers };
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
