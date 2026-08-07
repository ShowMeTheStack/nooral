// Running a student's JavaScript safely: each run is a fresh Web Worker made from a blob, with the network, storage
// and further workers taken away before the student's code is read, a time limit enforced from outside (the worker
// is terminated), and every message it sends signed with a secret the student's code cannot reach. The page never
// trusts the worker's verdict: the worker only returns what the function returned, and the page compares.
//   S.Sandbox.run({src, fn, tests: [{args, want}], timeLimit, onLog}) -> Promise<report>
//   S.Sandbox.runSync(...) -> report, where the host provides S.Sandbox.syncRunner (node's vm, for the checker)
//   tests: [{args, want}] or [{args, check(got, args) -> true | 'why not'}] (several right answers), or {args, free: true}
//   report = {ok, compileError, cases: [{i, args, want, got, error, ms, pass, why, diff}], firstFail, timedOut, logs}
//   S.Sandbox.show(v) -> a value as JavaScript would write it: 5n, [1, 2], Uint8Array(4) [de ad be ef]
//   S.Sandbox.showAs(task, v) -> the same, in hex when the task has radix: 16 (width: digits)
//   S.Sandbox.same(want, got) -> true or a sentence saying how they differ
(function (S) {
  'use strict';
  const SB = (S.Sandbox = S.Sandbox || {});
  const LIMIT = 2000;
  // bytes from another realm (a worker's message, a vm context) fail instanceof, so ask for the tag instead
  const isBytes = (v) => ArrayBuffer.isView(v) && Object.prototype.toString.call(v) === '[object Uint8Array]';
  SB.isBytes = isBytes;

  SB.LONG = 80;
  SB.show = function show(v, depth) {
    depth = depth || 0;
    if (typeof v === 'bigint') return v + 'n';
    // a long string (a paragraph passed to a function) shows its start and its length, not every character
    if (typeof v === 'string') return v.length > SB.LONG ? JSON.stringify(v.slice(0, SB.LONG - 20)).slice(0, -1) + '…" (' + v.length + ' characters)' : JSON.stringify(v);
    if (v === undefined) return 'undefined';
    if (v === null) return 'null';
    if (typeof v === 'number') return Object.is(v, -0) ? '-0' : String(v);
    if (isBytes(v)) return 'bytes [' + Array.from(v.slice(0, 24), (b) => b.toString(16).padStart(2, '0')).join(' ') + (v.length > 24 ? ' …' : '') + ']';
    if (Array.isArray(v)) return depth > 2 ? '[…]' : '[' + v.slice(0, 16).map((x) => show(x, depth + 1)).join(', ') + (v.length > 16 ? ', … (' + v.length + ' items)' : '') + ']';
    if (typeof v === 'object') { try { return depth > 2 ? '{…}' : '{' + Object.entries(v).slice(0, 8).map(([k, x]) => k + ': ' + show(x, depth + 1)).join(', ') + '}'; } catch (e) { return String(v); } }
    if (typeof v === 'function') return 'a function';
    return String(v);
  };
  /** a value as a task speaks of it: with radix: 16, numbers (bytes, words) in hex, padded to width digits */
  SB.showAs = function showAs(t, v) {
    if (t && t.radix === 16) {
      if (typeof v === 'number' && Number.isInteger(v) && v >= 0) return '0x' + v.toString(16).padStart(t.width || 2, '0');
      if (typeof v === 'bigint' && v >= 0n) return '0x' + v.toString(16) + 'n';
      if (Array.isArray(v)) return '[' + v.slice(0, 16).map((x) => showAs(t, x)).join(', ') + (v.length > 16 ? ', … (' + v.length + ' items)' : '') + ']';
    }
    return SB.show(v);
  };
  const kindOf = (v) => (typeof v === 'bigint' ? 'a BigInt' : typeof v === 'number' ? 'a Number' : typeof v === 'string' ? 'a string' : isBytes(v) ? 'bytes' : Array.isArray(v) ? 'an array' : v === undefined ? 'nothing' : v === null ? 'null' : typeof v === 'object' ? 'an object' : typeof v);
  SB.same = function same(want, got) {
    if (typeof want === 'bigint' || typeof got === 'bigint') {
      if (typeof want === typeof got) return want === got ? true : 'a different value';
      // the same number in the other type is worth saying plainly
      try { if (BigInt(want) === BigInt(got)) return 'the right value, but as ' + kindOf(got) + ' where ' + kindOf(want) + ' is wanted'; } catch (e) { /* not a whole number */ }
      return 'a different value, and ' + kindOf(got) + ' where ' + kindOf(want) + ' is wanted';
    }
    if (typeof want === 'number') return typeof got === 'number' && (got === want || (Number.isNaN(got) && Number.isNaN(want))) ? true : typeof got !== 'number' ? kindOf(got) + ' where a Number is wanted' : 'a different value';
    if (isBytes(want) || Array.isArray(want)) {
      if (!(isBytes(got) || Array.isArray(got))) return kindOf(got) + ' where ' + kindOf(want) + ' is wanted';
      if (got.length !== want.length) return got.length + ' items where ' + want.length + ' are wanted';
      for (let i = 0; i < want.length; i++) { const r = same(want[i], got[i]); if (r !== true) return 'item ' + i + ' differs (' + r + ')'; }
      return true;
    }
    if (want && typeof want === 'object') {
      if (!got || typeof got !== 'object') return kindOf(got) + ' where an object is wanted';
      for (const k of Object.keys(want)) { const r = same(want[k], got[k]); if (r !== true) return 'its ' + k + ' differs (' + r + ')'; }
      return true;
    }
    return want === got ? true : 'a different value';
  };
  /** compare the raw results of a run with what is wanted */
  SB.report = function (tests, raw) {
    const cases = [];
    let firstFail = -1;
    for (let i = 0; i < tests.length; i++) {
      const t = tests[i], r = raw.cases[i];
      if (!r) { cases.push({ i, args: t.args, want: t.want, pass: false, why: raw.timedOut && i === raw.cases.length ? 'it ran past the time limit' : 'it did not run' }); if (firstFail < 0) firstFail = i; break; }
      const c = { i, args: t.args, want: t.want, got: r.got, error: r.error, ms: r.ms };
      if (r.error) { c.pass = false; c.why = 'it threw ' + r.error; }
      else {
        // a test may judge the result itself (several answers are right), or ask nothing of it (a scratch run)
        let s;
        if (t.free) s = true;
        else if (t.check) { try { s = t.check(r.got, t.args); } catch (e) { s = 'it could not be judged (' + (e.message || e) + ')'; } if (s === false) s = 'that is not right'; }
        else s = SB.same(t.want, r.got);
        c.pass = s === true; if (!c.pass) c.why = 'it returned ' + SB.show(r.got) + ': ' + s; c.diff = s;
      }
      cases.push(c);
      if (!c.pass && firstFail < 0) firstFail = i;
      if (!c.pass) break;
    }
    return { ok: !raw.compileError && firstFail < 0 && cases.length === tests.length, compileError: raw.compileError || null, cases, firstFail, timedOut: !!raw.timedOut, logs: raw.logs || [], missing: raw.missing || null };
  };

  // ---- the worker -----------------------------------------------------------------------------------------------
  // The worker's own source. It removes what a student's code must not reach, then waits for one job.
  const WORKER = function () {
    'use strict';
    const post = self.postMessage.bind(self);
    let secret = null;
    for (const k of ['fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'importScripts', 'indexedDB', 'caches', 'BroadcastChannel', 'Worker', 'SharedWorker', 'WebTransport', 'RTCPeerConnection', 'Request', 'Response', 'navigator', 'location', 'open', 'close']) {
      try { Object.defineProperty(self, k, { value: undefined, writable: false, configurable: false }); } catch (e) { try { self[k] = undefined; } catch (e2) { /* fixed */ } }
    }
    const logs = [];
    const say = (...a) => { if (logs.length < 200) logs.push(a.map((x) => (typeof x === 'bigint' ? x + 'n' : typeof x === 'string' ? x : (() => { try { return JSON.stringify(x, (k, v) => (typeof v === 'bigint' ? v + 'n' : v)); } catch (e) { return String(x); } })())).join(' ')); };
    const quiet = { log: say, info: say, warn: say, error: say, debug: say };
    self.onmessage = (e) => {
      if (secret !== null) return;
      const job = e.data;
      secret = job.secret;
      self.onmessage = null;
      const send = (m) => post(Object.assign({ secret }, m));
      let f;
      try {
        // the student's code sees a console of its own, and nothing that posts messages
        f = new Function('console', 'postMessage', 'self', 'globalThis', '"use strict";\n' + job.src + '\n;return typeof ' + job.fn + ' === "function" ? ' + job.fn + ' : undefined;')(quiet, undefined, undefined, undefined);
      } catch (err) { send({ type: 'compile', error: (err && err.name ? err.name + ': ' : '') + (err && err.message || err), logs }); return; }
      if (typeof f !== 'function') { send({ type: 'missing', logs }); return; }
      for (let i = 0; i < job.tests.length; i++) {
        send({ type: 'start', i });
        const t0 = performance.now();
        let got, error = null;
        try { got = f(...job.tests[i].args); } catch (err) { error = (err && err.name ? err.name + ': ' : '') + (err && err.message || String(err)); }
        try { send({ type: 'case', i, got, error, ms: performance.now() - t0 }); } catch (err) { send({ type: 'case', i, got: undefined, error: 'its result cannot be sent back (' + (err.message || err) + ')', ms: 0 }); }
      }
      send({ type: 'done', logs });
    };
  };

  let blobUrl = null;
  const workerUrl = () => { if (!blobUrl) blobUrl = URL.createObjectURL(new Blob(['(' + WORKER.toString() + ')();'], { type: 'text/javascript' })); return blobUrl; };
  const secretOf = () => { const a = new Uint32Array(4); crypto.getRandomValues(a); return Array.from(a, (x) => x.toString(36)).join('-'); };

  SB.available = () => typeof Worker !== 'undefined' && typeof Blob !== 'undefined' && typeof URL !== 'undefined';
  SB.run = function (job) {
    const tests = job.tests || [];
    if (!SB.available()) return Promise.resolve(SB.syncRunner ? SB.runSync(job) : SB.report(tests, { compileError: 'This browser cannot run code in a worker.', cases: [] }));
    return new Promise((resolve) => {
      let w;
      try { w = new Worker(workerUrl()); } catch (e) { resolve(SB.report(tests, { compileError: 'A worker could not be started here: ' + (e.message || e), cases: [] })); return; }
      const secret = secretOf(), cases = [];
      let done = false, timer = null, current = -1;
      const finish = (raw) => { if (done) return; done = true; clearTimeout(timer); try { w.terminate(); } catch (e) { /* gone */ } resolve(SB.report(tests, Object.assign({ cases }, raw))); };
      const limit = job.timeLimit || LIMIT;
      const arm = () => { clearTimeout(timer); timer = setTimeout(() => finish({ timedOut: true, logs: [] }), limit); };
      w.onmessage = (e) => {
        const m = e.data;
        if (!m || m.secret !== secret) return;
        if (m.type === 'start') { current = m.i; arm(); }
        else if (m.type === 'case') { cases[m.i] = { got: m.got, error: m.error, ms: m.ms }; if (job.onCase) job.onCase(m.i, cases[m.i]); }
        else if (m.type === 'compile') finish({ compileError: m.error, logs: m.logs });
        else if (m.type === 'missing') finish({ compileError: 'There is no function called ' + job.fn + ' in your code.', missing: job.fn, logs: m.logs });
        else if (m.type === 'done') { if (job.onLog) for (const l of m.logs || []) job.onLog(l); finish({ logs: m.logs }); }
      };
      w.onerror = (e) => { e.preventDefault && e.preventDefault(); finish({ compileError: String(e.message || 'the code could not run'), logs: [] }); };
      arm();
      w.postMessage({ secret, src: String(job.src || ''), fn: job.fn, tests: tests.map((t) => ({ args: t.args })) });
      void current;
    });
  };
  /** the same run, synchronously, through a runner the host provides (node's vm in tools/check-content.mjs) */
  SB.runSync = function (job) {
    if (!SB.syncRunner) throw new Error('no synchronous runner here');
    return SB.report(job.tests || [], SB.syncRunner(job));
  };
})(globalThis.Nooral = globalThis.Nooral || {});
