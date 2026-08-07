// The code kind: a function written by the student, run case by case in the sandboxed worker (sandbox-run.js) and
// compared with what each case wants; numbers from floating-point work compare to within the exercise's tolerance.
// Its fields are listed in src/app/check-index.js.
(function (S) {
  'use strict';
  const K = (S.Check = S.Check || {});
  K.kinds = K.kinds || {};
  const right = (x, extra) => ({ ok: true, message: 'Right.' + (extra ? ' ' + extra : '') + (x.why ? ' ' + x.why : '') });
  K.right = right;

  // ---- code, run in the sandbox ---------------------------------------------------------------------------------------
  /** the tests of a code exercise: given, or made from a reference function on its cases */
  K.codeTests = function (x) {
    let tests;
    if (x.tests) tests = typeof x.tests === 'function' ? x.tests() : x.tests;
    else if (x.ref && x.cases) tests = x.cases.map((args) => ({ args, want: x.ref(...args) }));
    else throw new Error('a code exercise needs tests, or ref and cases');
    // numbers from floating-point work are compared to within the exercise's tolerance
    if (x.tolerance) tests = tests.map((t) => (t.check || t.free ? t : Object.assign({}, t, { check: (got) => K.near(t.want, got, x.tolerance) })));
    return tests;
  };
  /** a number, or arrays of numbers at any depth, equal to within tol: true or a sentence saying where they differ */
  K.near = function near(want, got, tol) {
    if (typeof want === 'number') {
      if (typeof got !== 'number' || !Number.isFinite(got)) return 'it returned ' + (S.Sandbox ? S.Sandbox.show(got) : String(got)) + ' where a number is wanted';
      return Math.abs(got - want) <= tol ? true : 'it returned ' + (+got.toPrecision(6)) + ', but it should be ' + (+want.toPrecision(6)) + ' (to within ' + tol + ')';
    }
    if (Array.isArray(want) || ArrayBuffer.isView(want)) {
      if (!(Array.isArray(got) || ArrayBuffer.isView(got))) return 'it returned something that is not a list where a list of ' + want.length + ' is wanted';
      if (got.length !== want.length) return 'it returned ' + got.length + ' items where ' + want.length + ' are wanted';
      for (let i = 0; i < want.length; i++) { const r = near(want[i], got[i], tol); if (r !== true) return 'item ' + i + ': ' + r; }
      return true;
    }
    return S.Sandbox ? S.Sandbox.same(want, got) : want === got;
  };
  /** where two long strings first differ, with a few characters of each around it ('' when either is short) */
  K.stringDiff = function (want, got) {
    const LONG = (S.Sandbox && S.Sandbox.LONG) || 80;
    if (typeof want !== 'string' || typeof got !== 'string' || (want.length <= LONG && got.length <= LONG)) return '';
    let i = 0;
    while (i < want.length && i < got.length && want[i] === got[i]) i++;
    const cut = (t) => JSON.stringify(t.slice(Math.max(0, i - 12), i + 18)).slice(1, -1);
    return ' They first differ at character ' + i + ': “' + (i > 12 ? '…' : '') + cut(got) + '…” where “' + (i > 12 ? '…' : '') + cut(want) + '…” is wanted.';
  };
  /** what a sandbox report says to the student: the first failing case, with its inputs */
  K.codeResult = function (x, rep, tests) {
    const SB = S.Sandbox, show = (v) => SB.showAs(x, v), call = (args) => x.fn + '(' + args.map((v, i) => ((x.decimalArgs || []).includes(i) ? SB.show(v) : show(v))).join(', ') + ')';
    if (rep.compileError) return { ok: false, message: rep.missing ? 'Your code has no function called ' + x.fn + ': write function ' + x.fn + '(' + (x.params || []).join(', ') + ') { … }.' : 'Your code does not run: ' + rep.compileError + '.', logs: rep.logs };
    if (rep.ok) return Object.assign(right(x, 'It passes all ' + K.plural(tests.length, 'test') + '.'), { cases: rep.cases, logs: rep.logs });
    const c = rep.cases[rep.firstFail];
    const n = rep.firstFail + 1;
    let msg = 'Test ' + n + ' of ' + tests.length + ' fails: ' + call(c.args) + ' ';
    if (rep.timedOut && !('got' in c) && !c.error) msg += 'ran longer than ' + ((x.timeLimit || 2000) / 1000) + ' s and was stopped (a loop that never ends?).';
    else if (c.error) msg += 'threw ' + c.error + '.';
    else if (c.want === undefined && typeof c.diff === 'string') msg += 'returned ' + show(c.got) + ', and ' + c.diff.replace(/\.$/, '') + '.';
    else { const d = SB.same(c.want, c.got); msg += 'returned ' + show(c.got) + ', but it should return ' + show(c.want) + (d !== true && d !== 'a different value' ? ' (' + d + ')' : '') + '.' + K.stringDiff(c.want, c.got); }
    return { ok: false, message: msg, cases: rep.cases, firstFail: rep.firstFail, logs: rep.logs };
  };
  /** the code a run sees: the task's given helpers (x.prelude), then the student's own */
  K.codeSource = (x, a) => (x.prelude ? String(x.prelude).replace(/\n*$/, '\n') : '') + String(a);
  K.kinds.code = {
    check(x, a) {
      if (K.blank(a)) return { ok: false, message: 'Write the function first.' };
      if (!S.Sandbox || !S.Sandbox.syncRunner) return { ok: false, message: 'Code is checked in a worker: use answerAsync.' };
      const tests = K.codeTests(x);
      return K.codeResult(x, S.Sandbox.runSync({ src: K.codeSource(x, a), fn: x.fn, tests, timeLimit: x.timeLimit }), tests);
    },
    async checkAsync(x, a) {
      if (K.blank(a)) return { ok: false, message: 'Write the function first.' };
      const tests = K.codeTests(x);
      return K.codeResult(x, await S.Sandbox.run({ src: K.codeSource(x, a), fn: x.fn, tests, timeLimit: x.timeLimit }), tests);
    },
  };

})(globalThis.Nooral = globalThis.Nooral || {});
