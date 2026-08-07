// The answer area of the code kind: an editor whose tests run in the sandbox (the first failing case shown, with the
// console). Each answer area is S.ExerciseViews.kind = (x, host, draft, page) -> { get(), mark(result)?,
// after(result, box)? }; lessons.js calls it, and the machine-learning kinds add theirs in exercises-ml.js.
// S.ExWidgets holds the pieces other pages share (the code editor, the cases table).
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const W = (S.ExWidgets = S.ExWidgets || {});
  const V = (S.ExerciseViews = S.ExerciseViews || {});
  const K = () => S.Check;

  /** a text box for code: monospaced, keeps every space and line, Tab indents, Ctrl+Enter runs */
  W.codeArea = function (text, opts) {
    opts = opts || {};
    const ta = h('textarea.mono.code-area', { id: opts.id || 'ansCode', spellcheck: 'false', autocomplete: 'off', autocapitalize: 'off', 'aria-label': opts.label || 'Your code', placeholder: opts.placeholder || '', rows: String(opts.rows || 12), wrap: 'soft' });
    ta.value = text || '';
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Tab' && !e.shiftKey) { e.preventDefault(); if (!document.execCommand('insertText', false, '  ')) { const a = ta.selectionStart; ta.value = ta.value.slice(0, a) + '  ' + ta.value.slice(ta.selectionEnd); ta.selectionStart = ta.selectionEnd = a + 2; } }
      else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && opts.onRun) { e.preventDefault(); opts.onRun(); }
    });
    return ta;
  };
  /** the cases of a sandbox run as a table: each test, what it was given, what came back, and a mark */
  W.casesTable = function (x, cases, total) {
    const SB = S.Sandbox;
    const rows = cases.map((c) => h('tr.' + (c.pass ? 'pass' : 'fail'),
      h('td', String(c.i + 1)), h('td.mono', x.fn + '(' + c.args.map((v) => SB.show(v)).join(', ') + ')'),
      h('td.mono', c.error ? 'threw ' + c.error : 'got' in c ? SB.show(c.got) : '—'), h('td.mono', SB.show(c.want)),
      h('td.mark', c.pass ? icon('check') : icon('x'))));
    const more = total > cases.length ? h('tr.rest', h('td', { colspan: '5' }, (total - cases.length) + ' more test' + (total - cases.length === 1 ? '' : 's') + ' not run (the first failure stops the run)')) : null;
    return h('div.fit-table.cases', h('table.cases-table', h('thead', h('tr', h('th', '#'), h('th', 'Call'), h('th', 'Returned'), h('th', 'Wanted'), h('th', ''))), h('tbody', rows, more)));
  };
  W.logBox = (logs) => (logs && logs.length ? h('details.logs', { open: true }, h('summary', 'Console (' + logs.length + ' line' + (logs.length === 1 ? '' : 's') + ')'), h('pre.mono', logs.join('\n'))) : null);

  // ---- code ------------------------------------------------------------------------------------------------------------
  /** the helper code a task hands the student (x.prelude): read-only, above the editor, run before their code */
  W.preludeBox = function (x) {
    if (!x.prelude) return null;
    const src = String(x.prelude).replace(/\n+$/, '');
    const lines = src.split('\n').length;
    const names = x.preludeNames || [...new Set([...src.matchAll(/^(?:function\s+([A-Za-z_$][\w$]*)|const\s+([A-Za-z_$][\w$]*)\s*=)/gm)].map((m) => m[1] || m[2]))];
    return h('details.code-prelude', { open: lines <= 12 ? '' : null },
      h('summary', 'Given code, read-only: ' + names.join(', ') + ' (' + lines + ' line' + (lines === 1 ? '' : 's') + ', run before yours)'),
      h('pre.mono', { 'aria-label': 'Given code (read-only)' }, src));
  };
  V.code = function (x, host, draft) {
    const ta = W.codeArea(typeof draft === 'string' ? draft : x.start || 'function ' + x.fn + '(' + (x.params || []).join(', ') + ') {\n  \n}\n', { onRun: () => document.getElementById('btnCheck').click() });
    const pre = W.preludeBox(x);
    if (pre) host.appendChild(pre);
    host.appendChild(h('div.code-wrap', ta));
    host.appendChild(h('p.muted.ans-note', 'Plain JavaScript: numbers and arrays of numbers. It runs in a sealed worker with no page and no network, for at most ' + ((x.timeLimit || 2000) / 1000) + ' s. Ctrl+Enter checks.'));
    requestAnimationFrame(() => S.UI.autosize(ta, 520));
    let extra = null;
    return {
      get: () => ta.value,
      after(r, box) {
        if (extra) extra.remove();
        let tests = [];
        try { tests = K().codeTests(x); } catch (e) { /* none */ }
        extra = h('div.code-res', r.cases && r.cases.length ? W.casesTable(x, r.cases, tests.length) : null, W.logBox(r.logs));
        box.appendChild(extra);
      },
    };
  };

})(globalThis.Nooral = globalThis.Nooral || {});
