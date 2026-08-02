// The text lab's link to its transformer: a Web Worker made from the core engines and textlab-a-model.js (so
// training never stalls the page), or, where no worker can be made, the same engine on the page in short slices.
// One link lives as long as the page is loaded, so the model trained in the GPT tab is still there for sampling.
//   TL.link() -> {mode: 'worker' | 'page', send(msg), ask(what, args) -> Promise, on(fn) -> off, info, step, ...}
(function (S) {
  'use strict';
  const TL = (S.TL = S.TL || {});
  /** the code of the lab's worker: the core files, then this lab's engine file, then its serve loop */
  TL.workerSource = function () {
    if (typeof document === 'undefined' || !S.T || !S.T.worker || !S.T.worker.source) return null;
    const base = S.T.worker.source();
    if (!base) return null;
    let mine = null;
    const tag = [...document.querySelectorAll('script[src]')].find((s) => /(^|\/)src\/app\/textlab-a-model\.js$/.test(s.getAttribute('src')));
    if (tag) mine = 'importScripts(' + JSON.stringify(new URL(tag.getAttribute('src'), location.href).href) + ');';
    else {
      const mark = '// ==== app/textlab-a-model.js ====';
      const text = [...document.scripts].map((s) => s.textContent || '').find((t) => t.includes(mark));
      if (!text) return null;
      const a = text.indexOf(mark), b = text.indexOf('\n// ==== ', a + mark.length);
      mine = text.slice(a, b < 0 ? text.length : b);
    }
    // the serve call at the very end (the same words appear earlier, inside the core's own code)
    const at = base.lastIndexOf('Nooral.T.worker.serve(self);');
    return at < 0 ? null : base.slice(0, at) + mine + '\nNooral.TL.serve(self);\n';
  };
  let link = null;
  TL.link = function () {
    if (link) return link;
    const fns = new Set(), waiting = new Map();
    let n = 0, w = null;
    const L = { mode: 'page', info: null, step: 0, running: false, ready: false, history: [], val: [], samples: [], error: null };
    const receive = (m) => {
      if (m.type === 'answer') { const p = waiting.get(m.id); if (p) { waiting.delete(m.id); p.ok(m.value); } return; }
      if (m.type === 'error' && m.id !== undefined && waiting.has(m.id)) { const p = waiting.get(m.id); waiting.delete(m.id); p.no(new Error(m.message)); return; }
      if (m.type === 'ready') { L.ready = true; L.info = m.info; L.cfg = Object.assign({}, L.cfg, m.info.cfg); L.step = 0; L.history = []; L.val = [{ step: 0, loss: m.val }]; L.samples = []; L.error = null; }
      if (m.type === 'progress') { L.step = m.step; L.history.push({ step: m.step, loss: m.loss }); if (m.val !== undefined) L.val.push({ step: m.step, loss: m.val }); }
      if (m.type === 'sample') L.samples.unshift({ step: m.step, text: m.text, prompt: m.prompt });
      if (m.type === 'state') { L.running = m.running; if (m.step !== undefined) L.step = m.step; }
      if (m.type === 'error') { L.running = false; L.error = m.message; }
      for (const f of fns) try { f(m); } catch (e) { /* a view's slip */ }
    };
    try {
      const src = TL.workerSource();
      if (src && typeof Worker !== 'undefined') {
        w = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
        w.onmessage = (e) => receive(e.data);
        w.onerror = (e) => { e.preventDefault(); receive({ type: 'error', message: 'the worker failed: ' + (e.message || 'unknown') }); };
        L.mode = 'worker';
      }
    } catch (e) { w = null; }
    if (!w) {
      // the same engine on the page: its messages come back on the next turn of the event loop
      const fake = { postMessage: (m) => setTimeout(() => receive(m), 0), onmessage: null };
      TL.serve(fake);
      w = { postMessage: (m) => fake.onmessage({ data: m }), terminate() {} };
    }
    L.send = (m) => w.postMessage(m);
    L.ask = (what, args) => new Promise((ok, no) => { const id = ++n; waiting.set(id, { ok, no }); w.postMessage({ cmd: 'ask', id, what, args }); });
    L.on = (f) => { fns.add(f); return () => fns.delete(f); };
    // cfg {size: 'auto' | 'webgpu' | 'cpu' | 'small'}: the worker decides 'auto' and reports the size it took in info
    L.init = (cfg, text) => { L.ready = false; L.running = false; L.cfg = Object.assign({}, TL.DEFAULTS, cfg); L.send({ cmd: 'init', cfg: L.cfg, text, every: 150, prompt: 'The ' }); };
    link = L;
    return L;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
