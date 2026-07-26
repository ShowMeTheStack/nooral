// The workflow engine (S.WF): the block registry, the document of a workflow (blocks and the wires between their
// typed ports), and a session that runs it: blocks in dependency order, each again only when its parameters or
// something upstream changed, training blocks driven batch by batch so they can be played, paused and stepped.
// Nothing here touches the page: the same code runs in the page, in the workflow's Web Worker and in node (the
// content checker runs reference workflows with it).
//
// A block type:  S.WF.block(type, {cat, label, blurb, icon, inputs, outputs, params, run | train, view, summary,
//                                  methods, order})
//   cat       one of S.WF.CATS                       inputs / outputs  [{id, label, type, optional}] or (params) => [...]
//   type      of a port: 'table' | 'images' | 'text' | 'model' | 'predictions', or a list of them (an input that takes
//             any of several); 'data' is short for ['table', 'images', 'text']; an output of type 'same:in' carries
//             whatever kind its input 'in' gets (a split of images is images)
//   params    [{id, label, kind: 'int' | 'number' | 'choice' | 'bool' | 'text' | 'seed' | 'layers' | 'columns' | 'file',
//               def, min, max, step, log, options: [[value, label]] | (params, ins) => [...], help, show(params) -> bool}]
//   run(ctx)  -> {out: {port: value}, ...anything view() wants}: a block that computes at once
//   train(ctx) -> a trainer {epochs, batches, step() -> {epoch, batch, loss, ...}, done(), current() -> result,
//             snapshot() -> view}: a block that learns batch by batch (the session drives it; current() is the result
//             so far, as run would return it)
//   view(result, ctx) -> plain JSON for the block's viewer (computed where the block ran: it crosses from the worker)
//   summary(result) -> a few words shown on the block ("200 rows · 2 classes", "accuracy 0.93")
//   methods   {name(result, args, ctx) -> JSON}: what a viewer may ask later (predict a row the student typed)
// ctx = {id, params, in: {port: value}, seed, rng, progress(fraction, words), doc, session}
//
// A document: {v: 1, name, seed, blocks: [{id, type, params, x, y}], wires: [{id, from: [block, port], to: [block, port]}]}
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  WF.CATS = ['Data', 'Prepare', 'Split', 'Models', 'Train', 'Evaluate', 'Compare', 'Predict', 'Explain', 'Unsupervised'];
  WF.TYPES = {
    table: { label: 'table', blurb: 'rows of features with a target' },
    images: { label: 'images', blurb: 'small images with their labels' },
    text: { label: 'text', blurb: 'short texts with their labels' },
    model: { label: 'model', blurb: 'a model, trained or not' },
    predictions: { label: 'predictions', blurb: 'what a model said for each row, beside the truth' },
  };
  const REG = new Map();
  WF.block = function (type, def) {
    REG.set(type, Object.assign({ type, inputs: [], outputs: [], params: [], order: 50, icon: 'block' }, def));
    return REG.get(type);
  };
  WF.def = (type) => REG.get(type) || null;
  WF.list = (cat) => [...REG.values()].filter((d) => !cat || d.cat === cat).sort((a, b) => WF.CATS.indexOf(a.cat) - WF.CATS.indexOf(b.cat) || a.order - b.order);
  const accepts = (t) => (t === 'data' ? ['table', 'images', 'text'] : [].concat(t));
  WF.accepts = accepts;

  /** a block's parameters with every default filled in */
  WF.defaults = function (type, given) {
    const d = WF.def(type), o = {};
    for (const p of (d && d.params) || []) o[p.id] = given && given[p.id] !== undefined ? given[p.id] : typeof p.def === 'function' ? p.def() : p.def;
    if (given) for (const k of Object.keys(given)) if (!(k in o)) o[k] = given[k];
    return o;
  };
  /** the ports of a block as placed: {inputs, outputs}, each port {id, label, type (a list for inputs), optional} */
  WF.ports = function (b) {
    const d = WF.def(b.type);
    if (!d) return { inputs: [], outputs: [] };
    const p = WF.defaults(b.type, b.params);
    const ins = typeof d.inputs === 'function' ? d.inputs(p) : d.inputs, outs = typeof d.outputs === 'function' ? d.outputs(p) : d.outputs;
    return { inputs: ins.map((x) => Object.assign({}, x, { type: accepts(x.type) })), outputs: outs.map((x) => Object.assign({}, x)) };
  };

  // ---- documents --------------------------------------------------------------------------------------------------
  WF.newDoc = (name) => ({ v: 1, name: name || 'Untitled workflow', seed: 1, blocks: [], wires: [] });
  WF.clone = (doc) => JSON.parse(JSON.stringify(doc));
  WF.byId = (doc, id) => doc.blocks.find((b) => b.id === id) || null;
  /** a fresh block id for a type: logreg1, logreg2, ... */
  WF.newId = function (doc, type) {
    const base = String(type).replace(/[^a-z0-9]/gi, '').toLowerCase() || 'b';
    for (let i = 1; ; i++) if (!doc.blocks.some((b) => b.id === base + i)) return base + i;
  };
  WF.add = function (doc, type, params, x, y) {
    if (!WF.def(type)) throw new Error('there is no block called ' + type);
    const b = { id: WF.newId(doc, type), type, params: WF.defaults(type, params), x: x || 0, y: y || 0 };
    doc.blocks.push(b);
    return b;
  };
  WF.remove = function (doc, id) {
    doc.blocks = doc.blocks.filter((b) => b.id !== id);
    doc.wires = doc.wires.filter((w) => w.from[0] !== id && w.to[0] !== id);
  };
  /** may this output feed that input? true, or a sentence saying why not */
  WF.canWire = function (doc, from, to) {
    const a = WF.byId(doc, from[0]), b = WF.byId(doc, to[0]);
    if (!a || !b) return 'one end of the wire is not on the canvas';
    if (a.id === b.id) return 'a block cannot feed itself';
    const op = WF.ports(a).outputs.find((p) => p.id === from[1]), ip = WF.ports(b).inputs.find((p) => p.id === to[1]);
    if (!op || !ip) return 'that port does not exist';
    const t = WF.outType(doc, a.id, op.id);
    if (t === 'data' ? !ip.type.some((x) => accepts('data').includes(x)) : !ip.type.includes(t)) return 'the ' + (ip.label || ip.id) + ' input of ' + WF.label(b) + ' takes ' + ip.type.map((x) => WF.TYPES[x].label).join(' or ') + ', not ' + (t === 'data' ? 'data' : WF.TYPES[t].label);
    if (WF.reaches(doc, b.id, a.id)) return 'that wire would make a loop';
    return true;
  };
  /** connect, replacing whatever fed that input before; returns the wire or throws with the reason */
  WF.connect = function (doc, from, to) {
    const ok = WF.canWire(doc, from, to);
    if (ok !== true) throw new Error(ok);
    doc.wires = doc.wires.filter((w) => !(w.to[0] === to[0] && w.to[1] === to[1]));
    const w = { id: 'w' + from[0] + '.' + from[1] + '-' + to[0] + '.' + to[1], from: from.slice(), to: to.slice() };
    doc.wires.push(w);
    return w;
  };
  /** the kind an output carries: its own type, or for 'same:in' what feeds that input ('data' while nothing does) */
  WF.outType = function (doc, id, port, depth) {
    const b = WF.byId(doc, id), op = b && WF.ports(b).outputs.find((p) => p.id === port);
    if (!op) return null;
    if (!/^same:/.test(op.type)) return op.type;
    const src = WF.feeds(doc, id)[op.type.slice(5)];
    return src && (depth || 0) < 50 ? WF.outType(doc, src[0], src[1], (depth || 0) + 1) || 'data' : 'data';
  };
  WF.label = (b) => (b.label || (WF.def(b.type) ? WF.def(b.type).label : b.type));
  /** does a path of wires lead from block a to block b? */
  WF.reaches = function (doc, a, b) {
    const seen = new Set([a]), todo = [a];
    while (todo.length) { const x = todo.pop(); if (x === b) return true; for (const w of doc.wires) if (w.from[0] === x && !seen.has(w.to[0])) { seen.add(w.to[0]); todo.push(w.to[0]); } }
    return false;
  };
  /** the blocks in an order where every block comes after the ones that feed it */
  WF.order = function (doc) {
    const indeg = new Map(doc.blocks.map((b) => [b.id, 0]));
    for (const w of doc.wires) if (indeg.has(w.to[0]) && indeg.has(w.from[0])) indeg.set(w.to[0], indeg.get(w.to[0]) + 1);
    const out = [], ready = doc.blocks.filter((b) => !indeg.get(b.id)).map((b) => b.id);
    while (ready.length) {
      const id = ready.shift();
      out.push(id);
      for (const w of doc.wires) if (w.from[0] === id && indeg.has(w.to[0])) { indeg.set(w.to[0], indeg.get(w.to[0]) - 1); if (!indeg.get(w.to[0])) ready.push(w.to[0]); }
    }
    if (out.length !== doc.blocks.length) throw new Error('the wires make a loop');
    return out;
  };
  /** which output feeds each input of a block: {port: [block, port]} */
  WF.feeds = (doc, id) => Object.fromEntries(doc.wires.filter((w) => w.to[0] === id).map((w) => [w.to[1], w.from]));
  /** every block downstream of the given ones (themselves included) */
  WF.downstream = function (doc, ids) {
    const seen = new Set(ids), todo = [...ids];
    while (todo.length) { const x = todo.pop(); for (const w of doc.wires) if (w.from[0] === x && !seen.has(w.to[0])) { seen.add(w.to[0]); todo.push(w.to[0]); } }
    return seen;
  };
  /** what is wrong with a document: [{block, msg}] (blocks of unknown types, bad wires, unfilled inputs) */
  WF.problems = function (doc) {
    const out = [];
    for (const b of doc.blocks) if (!WF.def(b.type)) out.push({ block: b.id, msg: 'there is no block called ' + b.type });
    for (const w of doc.wires) { const ok = WF.canWire(Object.assign({}, doc, { wires: doc.wires.filter((x) => x !== w) }), w.from, w.to); if (ok !== true) out.push({ block: w.to[0], msg: ok }); }
    for (const b of doc.blocks) {
      if (!WF.def(b.type)) continue;
      const f = WF.feeds(doc, b.id);
      for (const p of WF.ports(b).inputs) if (!p.optional && !f[p.id]) out.push({ block: b.id, msg: WF.label(b) + ' needs something wired into its ' + (p.label || p.id) + ' input', waiting: true });
    }
    return out;
  };
  /** read a document from JSON (a saved file, a share link), checking its shape */
  WF.fromJSON = function (obj) {
    if (typeof obj === 'string') obj = JSON.parse(obj);
    if (!obj || !Array.isArray(obj.blocks) || !Array.isArray(obj.wires)) throw new Error('This is not a Nooral workflow.');
    const doc = Object.assign(WF.newDoc(), obj, { v: 1 });
    doc.blocks = obj.blocks.filter((b) => b && b.id && b.type).map((b) => ({ id: String(b.id), type: String(b.type), params: WF.defaults(b.type, b.params || {}), x: +b.x || 0, y: +b.y || 0, label: b.label }));
    doc.wires = obj.wires.filter((w) => w && w.from && w.to && WF.byId(doc, w.from[0]) && WF.byId(doc, w.to[0])).map((w) => Object.assign({ id: w.id || 'w' + w.from.join('.') + '-' + w.to.join('.'), from: [String(w.from[0]), String(w.from[1])], to: [String(w.to[0]), String(w.to[1])] }, Array.isArray(w.via) ? { via: w.via.map((p) => [+p[0], +p[1]]) } : {}));
    return doc;
  };

  // ---- shared registries --------------------------------------------------------------------------------------------
  /** sample workflows: WF.addSample({id, title, blurb, cat, build() -> doc}); WF.sample(id) -> a fresh copy */
  WF.samples = WF.samples || new Map();
  WF.addSample = (x) => { WF.samples.set(x.id, x); return x; };
  WF.sample = (id) => { const x = WF.samples.get(id); if (!x) return null; const d = x.build(); d.name = d.name && d.name !== 'Untitled workflow' ? d.name : x.title; return d; };
  /**
   * a block from a doc: build a small workflow in one go. spec: {blocks: {id: [type, params, x, y]}, wires: [['a.out',
   * 'b.in'], ...]} -> doc (ids as given, so content can name them)
   */
  WF.build = function (spec, name) {
    const doc = WF.newDoc(name);
    if (spec.seed !== undefined) doc.seed = spec.seed;
    for (const [id, [type, params, x, y]] of Object.entries(spec.blocks)) doc.blocks.push({ id, type, params: WF.defaults(type, params || {}), x: x || 0, y: y || 0 });
    for (const [a, b] of spec.wires || []) { const [fa, fp] = a.split('.'), [ta, tp] = b.split('.'); WF.connect(doc, [fa, fp], [ta, tp]); }
    return doc;
  };
  /** page-side viewers of blocks: WF.viewers[type] (or ['cat:' + category]) = (host, view, api) -> {update(view), dispose()} */
  WF.viewers = WF.viewers || {};
  /** extra work the workflow's worker can do for the page: WF.workerFns[name](session, args) -> JSON */
  WF.workerFns = WF.workerFns || {};

  // ---- hashing: a block runs again only when this changes ----------------------------------------------------------
  const fnv = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(36); };
  WF.hash = fnv;
  WF.hashes = function (doc) {
    const out = {};
    for (const id of WF.order(doc)) {
      const b = WF.byId(doc, id), f = WF.feeds(doc, id);
      out[id] = fnv(b.type + '|' + JSON.stringify(b.params) + '|' + doc.seed + '|' + Object.keys(f).sort().map((k) => k + '<' + out[f[k][0]] + '.' + f[k][1]).join(','));
    }
    return out;
  };
  /** the seed a block uses: its own seed parameter when it has one, else one made from the workflow's seed and its id */
  WF.seedOf = (doc, b) => (b.params && Number.isFinite(+b.params.seed) && b.params.seed !== '' && b.params.seed !== null ? +b.params.seed : (parseInt(fnv(String(doc.seed) + ':' + b.id), 36) % 100000) + 1);

  // ---- a session: runs a document and keeps what each block made --------------------------------------------------
  /**
   * new S.WF.Session({onEvent(e)}): e is {type: 'state', id, state: 'waiting' | 'queued' | 'running' | 'training' |
   * 'paused' | 'done' | 'error', progress, words, summary, error, ms} | {type: 'view', id, view} | {type: 'curve', id,
   * point: {epoch, batch, loss, ...}} | {type: 'idle'}.
   */
  class Session {
    constructor(opts) {
      this.opts = opts || {};
      this.doc = WF.newDoc();
      this.res = {}; // id -> {hash, result, view, state, error, summary}
      this.trainers = {}; // id -> {trainer, mode: 'play' | 'pause' | 'epoch' | 'batch', hash}
      this.running = false;
      this.again = false;
    }
    emit(e) { if (this.opts.onEvent) try { this.opts.onEvent(e); } catch (x) { /* a listener's slip is not the run's */ } }
    setState(id, state, extra) { const r = (this.res[id] = this.res[id] || {}); r.state = state; this.emit(Object.assign({ type: 'state', id, state }, extra || {})); }
    /** take a new version of the document: blocks whose hash changed (and what they feed) will run again */
    load(doc) {
      this.doc = WF.clone(doc);
      const hs = WF.hashes(this.doc);
      for (const id of Object.keys(this.res)) if (!hs[id]) { delete this.res[id]; delete this.trainers[id]; }
      for (const [id, h] of Object.entries(hs)) if (this.res[id] && this.res[id].hash !== h) { this.res[id].stale = true; if (this.trainers[id] && this.trainers[id].hash !== h) delete this.trainers[id]; }
      this.hs = hs;
      return this;
    }
    /** does this block need to run? */
    needs(id) { const r = this.res[id]; return !r || r.stale || r.hash !== this.hs[id] || r.state === 'waiting'; }
    /** the blocks that need to run, in order */
    plan() { return WF.order(this.doc).filter((id) => this.needs(id)); }
    /** the inputs of a block from what upstream made, or null with the reason it must wait */
    gather(b) {
      const f = WF.feeds(this.doc, b.id), ins = {};
      for (const p of WF.ports(b).inputs) {
        const src = f[p.id];
        if (!src) { if (p.optional) continue; return { wait: 'needs its ' + (p.label || p.id) + ' input wired' }; }
        const r = this.res[src[0]];
        if (!r || r.state === 'error') return { wait: 'waits for ' + WF.label(WF.byId(this.doc, src[0])) };
        if (!r.result || !r.result.out || r.result.out[src[1]] === undefined) return { wait: 'waits for ' + WF.label(WF.byId(this.doc, src[0])) };
        ins[p.id] = r.result.out[src[1]];
      }
      return { ins };
    }
    ctx(b, ins) {
      const seed = WF.seedOf(this.doc, b), sess = this;
      return { id: b.id, hash: this.hs[b.id], block: b, params: WF.defaults(b.type, b.params), in: ins, seed, rng: S.ML && S.ML.rng ? S.ML.rng(seed) : null, doc: this.doc, session: this,
        progress(frac, words) { sess.emit({ type: 'state', id: b.id, state: 'running', progress: frac, words }); } };
    }
    finish(b, ctx, result, t0) {
      const d = WF.def(b.type);
      let view = null, summary = '';
      try { view = d.view ? d.view(result, ctx) : null; } catch (e) { view = { error: 'This view could not be drawn: ' + (e.message || e) }; }
      try { summary = d.summary ? d.summary(result, view) : ''; } catch (e) { summary = ''; }
      Object.assign(this.res[b.id] = this.res[b.id] || {}, { hash: ctx.hash, result, view, summary, error: null, stale: false, ctx });
      // a model that learned more feeds everything after it again, though nothing there changed
      if (d.train) for (const x of WF.downstream(this.doc, [b.id])) if (x !== b.id && this.res[x]) this.res[x].stale = true;
      this.setState(b.id, 'done', { summary, ms: Date.now() - t0 });
      this.emit({ type: 'view', id: b.id, view });
    }
    fail(b, e) {
      Object.assign(this.res[b.id] = this.res[b.id] || {}, { hash: this.hs[b.id], result: null, view: null, error: String((e && e.message) || e), stale: false });
      this.setState(b.id, 'error', { error: this.res[b.id].error });
      if (this.opts.debug && e && e.stack) console.error(e.stack);
    }
    wait(b, why) {
      Object.assign(this.res[b.id] = this.res[b.id] || {}, { hash: this.hs[b.id], result: null, view: null, error: null, stale: false });
      this.setState(b.id, 'waiting', { words: why });
    }
    /** run everything that needs it, at once (node, the content checker, scoring): trainers run to their end */
    runSync() {
      for (const id of WF.order(this.doc)) {
        if (!this.needs(id)) continue;
        const b = WF.byId(this.doc, id), d = WF.def(b.type), g = this.gather(b), t0 = Date.now();
        if (!d) { this.fail(b, new Error('there is no block called ' + b.type)); continue; }
        if (g.wait) { this.wait(b, g.wait); continue; }
        const ctx = this.ctx(b, g.ins);
        try {
          if (d.train) { const tr = d.train(ctx); let guard = 0; while (!tr.done() && guard++ < 1e7) tr.step(); this.finish(b, ctx, tr.current(), t0); }
          else this.finish(b, ctx, d.run(ctx), t0);
        } catch (e) { this.fail(b, e); }
      }
      return this;
    }
    /**
     * run what needs it, a slice at a time so the thread stays free for messages (pause, step); a training block
     * is driven by its mode: 'play' runs to the end, 'pause' waits, 'epoch' and 'batch' run one and then pause
     */
    async run() {
      if (this.running) { this.again = true; return; }
      this.running = true;
      try {
        do {
          this.again = false;
          for (const id of WF.order(this.doc)) {
            if (this.again) break;
            if (!this.needs(id)) continue;
            const b = WF.byId(this.doc, id), d = WF.def(b.type), g = this.gather(b), t0 = Date.now();
            if (!d) { this.fail(b, new Error('there is no block called ' + b.type)); continue; }
            if (g.wait) { this.wait(b, g.wait); continue; }
            const ctx = this.ctx(b, g.ins);
            if (!d.train) { this.setState(id, 'running', { progress: 0 }); await tick(); }
            try {
              if (d.train) { if (!(await this.drive(b, d, ctx, t0))) break; }
              else this.finish(b, ctx, await d.run(ctx), t0);
            } catch (e) { this.fail(b, e); }
          }
        } while (this.again);
      } finally { this.running = false; this.emit({ type: 'idle' }); }
    }
    /** drive one trainer; false when the run must start over (the document or a control changed) */
    async drive(b, d, ctx, t0) {
      let T = this.trainers[b.id];
      if (!T || T.hash !== this.hs[b.id]) {
        const start = (this.startMode && this.startMode[b.id]) || (this.opts.autoTrain === false ? 'pause' : 'play');
        if (this.startMode) delete this.startMode[b.id];
        T = this.trainers[b.id] = { trainer: d.train(ctx), mode: start, hash: this.hs[b.id] };
      }
      const tr = T.trainer;
      let last = 0, lastView = Date.now();
      for (;;) {
        if (this.again || this.trainers[b.id] !== T) return false;
        if (tr.done()) break;
        if (T.mode === 'pause') {
          // the model so far goes downstream, so a paused run can still be evaluated
          this.finish(b, ctx, tr.current(), t0);
          this.setState(b.id, 'paused', { progress: tr.progress ? tr.progress() : 0, summary: this.res[b.id].summary, words: 'paused at epoch ' + tr.epoch });
          return true;
        }
        const before = tr.epoch;
        const r = tr.step() || {};
        if (r.loss !== undefined || r.metrics) this.emit({ type: 'curve', id: b.id, point: Object.assign({ epoch: tr.epoch, batch: tr.batch }, r) });
        if (T.mode === 'batch' || (T.mode === 'epoch' && tr.epoch !== before)) T.mode = 'pause';
        if (Date.now() - last > 40) {
          last = Date.now();
          this.emit({ type: 'state', id: b.id, state: 'training', progress: tr.progress ? tr.progress() : 0, words: 'epoch ' + tr.epoch + (tr.epochs ? ' of ' + tr.epochs : '') });
          if (Date.now() - lastView > 400 && tr.snapshot) { lastView = Date.now(); try { this.emit({ type: 'view', id: b.id, view: tr.snapshot(), live: true }); } catch (e) { /* a snapshot is a courtesy */ } }
          await tick();
        }
      }
      this.finish(b, ctx, tr.current(), t0);
      return true;
    }
    /** a training block's controls: 'play' | 'pause' | 'epoch' | 'batch' | 'reset' (back to the untrained model) */
    control(id, action) {
      if (!WF.byId(this.doc, id)) return Promise.resolve();
      if (action === 'reset') { delete this.trainers[id]; (this.startMode = this.startMode || {})[id] = 'pause'; }
      else if (this.trainers[id]) this.trainers[id].mode = action;
      else (this.startMode = this.startMode || {})[id] = action;
      if (action !== 'pause' && this.res[id]) this.res[id].stale = true;
      if (this.running) { this.again = true; return Promise.resolve(); }
      return this.run();
    }
    /** a viewer's question for a block that has run: its methods[name](result, args, ctx) */
    call(id, name, args) {
      const b = WF.byId(this.doc, id), d = b && WF.def(b.type), r = this.res[id];
      if (!d || !d.methods || !d.methods[name]) throw new Error('the block cannot answer ' + name);
      if (!r || !r.result) throw new Error(WF.label(b) + ' has not run yet');
      return d.methods[name](r.result, args || {}, r.ctx);
    }
    /** what each block shows: {id: {state, summary, error, view}} */
    snapshot() { return Object.fromEntries(Object.entries(this.res).map(([id, r]) => [id, { state: r.state, summary: r.summary, error: r.error, view: r.view }])); }
    output(id, port) { const r = this.res[id]; return r && r.result && r.result.out ? r.result.out[port] : undefined; }
  }
  const tick = () => new Promise((r) => (typeof setImmediate === 'function' ? setImmediate(r) : setTimeout(r, 0)));
  WF.Session = Session;
  /** run a document to the end, at once: the session (node, scoring, tests) */
  WF.runSync = (doc, opts) => new Session(opts).load(doc).runSync();
})(globalThis.Nooral = globalThis.Nooral || {});
