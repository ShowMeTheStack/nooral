// The workflow's worker: a S.WF.Session running in a Web Worker made from the page's own engine code (every core file
// and the workflow engine and blocks, app/wf-a-*.js and app/wf-b-*.js), so training never freezes the page. Where no
// worker can start, the same session runs on the page, a slice at a time.
//   const r = new S.WF.Remote({onEvent(e)}); r.load(doc); r.control(id, action); await r.call(id, method, args);
//   await r.fn(name, args); await r.idle(); r.views[id], r.states[id]; r.dispose()
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const ENGINE = /(^|\/)src\/(core\/[^/]+|app\/wf-[ab]-[^/]+)\.js$/;
  const SECTION = /^(core\/[^ ]+|app\/wf-[ab]-[^ ]+)\.js$/;

  /** what the worker runs: it serves one session and answers the page */
  const SERVE = function () {
    const WF = self.Nooral.WF;
    let seq = 0;
    const clean = (m) => { try { return JSON.parse(JSON.stringify(m)); } catch (e) { return { type: m.type, id: m.id, error: 'could not send: ' + e.message }; } };
    const post = (m) => { try { self.postMessage(m); } catch (e) { self.postMessage(clean(m)); } };
    const session = new WF.Session({ onEvent: (e) => post(e.type === 'idle' ? Object.assign({ seq }, e) : e) });
    self.onmessage = async (ev) => {
      const m = ev.data || {};
      try {
        if (m.cmd === 'load') { seq = m.seq; session.load(m.doc); session.run(); }
        else if (m.cmd === 'control') { seq = m.seq; session.control(m.id, m.action); }
        else if (m.cmd === 'call') post({ type: 'reply', rid: m.rid, ok: true, value: session.call(m.id, m.method, m.args) });
        else if (m.cmd === 'fn') {
          const f = WF.workerFns[m.name];
          if (!f) throw new Error('the worker has no function ' + m.name);
          post({ type: 'reply', rid: m.rid, ok: true, value: await f(session, m.args) });
        }
      } catch (e) { if (m.rid) post({ type: 'reply', rid: m.rid, ok: false, error: String((e && e.message) || e) }); else post({ type: 'error', error: String((e && e.message) || e) }); }
    };
    post({ type: 'ready' });
  };

  /** the worker's source, from the dev page's script tags or the built page's sections; null when neither is there */
  WF.workerSource = function () {
    if (typeof document === 'undefined') return null;
    const tags = [...document.querySelectorAll('script[src]')].filter((s) => ENGINE.test(s.getAttribute('src')));
    const tail = '\n;(' + SERVE.toString() + ')();\n';
    if (tags.length) return 'self.Nooral = {};\nimportScripts(' + tags.map((t) => JSON.stringify(new URL(t.getAttribute('src'), location.href).href)).join(', ') + ');' + tail;
    const text = [...document.scripts].map((s) => s.textContent || '').find((t) => t.includes('// ==== app/wf-a-core.js ===='));
    if (!text) return null;
    const marks = [...text.matchAll(/(?:^|\n)\/\/ ==== ([^\n]+) ====\n/g)];
    const parts = [];
    marks.forEach((m, i) => { if (SECTION.test(m[1])) parts.push(text.slice(m.index, i + 1 < marks.length ? marks[i + 1].index : text.length)); });
    return 'self.Nooral = {};\n' + parts.join('\n') + tail;
  };
  let url = null;
  const workerUrl = () => {
    if (url) return url;
    if (typeof Worker === 'undefined' || typeof Blob === 'undefined') return null;
    const src = WF.workerSource();
    if (!src) return null;
    url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    return url;
  };

  class Remote {
    constructor(opts) {
      this.opts = opts || {};
      this.views = {};
      this.states = {};
      this.seq = 0;
      this.done = 0;
      this.waiters = [];
      this.replies = new Map();
      this.rid = 0;
      this.lastDoc = null;
      this.start(this.opts.page);
    }
    start(onPage) {
      const u = onPage ? null : workerUrl();
      if (u) {
        try {
          this.worker = new Worker(u);
          this.worker.onmessage = (e) => this.receive(e.data);
          this.worker.onerror = (e) => { if (e.preventDefault) e.preventDefault(); this.fallBack('the worker failed: ' + (e.message || 'unknown')); };
          this.mode = 'worker';
          return;
        } catch (e) { this.worker = null; }
      }
      this.mode = 'page';
      let seqOf = 0;
      this.session = new WF.Session({ onEvent: (e) => this.receive(e.type === 'idle' ? Object.assign({ seq: seqOf }, e) : e) });
      this.local = (m) => {
        try {
          if (m.cmd === 'load') { seqOf = m.seq; this.session.load(m.doc); this.session.run(); }
          else if (m.cmd === 'control') { seqOf = m.seq; this.session.control(m.id, m.action); }
          else if (m.cmd === 'call') this.receive({ type: 'reply', rid: m.rid, ok: true, value: this.session.call(m.id, m.method, m.args) });
          else if (m.cmd === 'fn') {
            const f = WF.workerFns[m.name];
            if (!f) throw new Error('there is no function ' + m.name);
            Promise.resolve(f(this.session, m.args)).then((value) => this.receive({ type: 'reply', rid: m.rid, ok: true, value }), (err) => this.receive({ type: 'reply', rid: m.rid, ok: false, error: String(err.message || err) }));
          }
        } catch (err) { if (m.rid) this.receive({ type: 'reply', rid: m.rid, ok: false, error: String(err.message || err) }); }
      };
    }
    /** the worker broke (a script that does not load): carry on, on the page */
    fallBack(why) {
      if (this.mode === 'page' || this.disposed) return;
      try { this.worker.terminate(); } catch (e) { /* gone */ }
      this.worker = null;
      if (typeof console !== 'undefined') console.warn('Workflow worker: ' + why + '; running on the page instead.');
      this.start(true);
      if (this.lastDoc) this.send({ cmd: 'load', doc: this.lastDoc, seq: this.seq });
      for (const [rid, r] of this.replies) { r.reject(new Error('the worker stopped')); this.replies.delete(rid); }
    }
    send(m) {
      if (this.disposed) return;
      if (this.worker) this.worker.postMessage(m);
      else setTimeout(() => this.local(m), 0);
    }
    receive(e) {
      if (!e || this.disposed) return;
      if (e.type === 'reply') { const r = this.replies.get(e.rid); if (r) { this.replies.delete(e.rid); if (e.ok) r.resolve(e.value); else r.reject(new Error(e.error)); } return; }
      if (e.type === 'ready') return;
      if (e.type === 'state') this.states[e.id] = Object.assign({}, this.states[e.id] && e.state === this.states[e.id].state ? this.states[e.id] : {}, e);
      if (e.type === 'view' && !e.live) this.views[e.id] = e.view;
      if (e.type === 'idle') {
        this.done = Math.max(this.done, e.seq || 0);
        const ready = this.waiters.filter((w) => w.seq <= this.done);
        this.waiters = this.waiters.filter((w) => w.seq > this.done);
        for (const w of ready) w.resolve();
      }
      if (this.opts.onEvent) try { this.opts.onEvent(e); } catch (x) { if (typeof console !== 'undefined') console.error(x); }
    }
    /** a new version of the workflow: what changed runs again */
    load(doc) {
      this.lastDoc = JSON.parse(JSON.stringify(doc));
      for (const id of Object.keys(this.states)) if (!doc.blocks.some((b) => b.id === id)) { delete this.states[id]; delete this.views[id]; }
      this.send({ cmd: 'load', doc: this.lastDoc, seq: ++this.seq });
    }
    control(id, action) { this.send({ cmd: 'control', id, action, seq: ++this.seq }); }
    ask(m) { return new Promise((resolve, reject) => { const rid = ++this.rid; this.replies.set(rid, { resolve, reject }); this.send(Object.assign({ rid }, m)); }); }
    call(id, method, args) { return this.ask({ cmd: 'call', id, method, args }); }
    fn(name, args) { return this.ask({ cmd: 'fn', name, args }); }
    /** resolves when everything sent so far has run */
    idle() { if (this.done >= this.seq) return Promise.resolve(); return new Promise((resolve) => this.waiters.push({ seq: this.seq, resolve })); }
    dispose() {
      this.disposed = true;
      if (this.worker) try { this.worker.terminate(); } catch (e) { /* gone */ }
      if (this.session) this.session.again = true;
      for (const w of this.waiters) w.resolve();
      this.waiters = [];
    }
  }
  WF.Remote = Remote;
})(globalThis.Nooral = globalThis.Nooral || {});
