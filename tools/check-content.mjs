// Check the content: node tools/check-content.mjs [id or chapter prefix] [--verbose]
// Every exercise: its kind exists, its solution passes its own checker, every one of its wrong answers fails (with
// the message shown under --verbose; a table with none listed gets its solution with one cell changed), and it has
// three hints (the last a worked answer). Code exercises run in node's vm with the same time limit the page's
// sandboxed worker has; workflow exercises run their reference workflow (and each wrong one) through the engines,
// headless, and score it on the hidden test set. Every lesson block names things that exist (topics, exercises,
// lessons, challenges, demo scenes, workflow blocks), its predictions have an answer among their options. Every model
// challenge: its reference workflow reaches the target on the hidden set within the limits, every wrong one does not,
// and it has a story and three hints. Every course step and tour step is complete. Anything marked sample: true is
// placeholder content: a warning now, to be replaced before release.
import { files } from '../tests/load.mjs';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const ctx = vm.createContext({ console, performance, setTimeout, clearTimeout, TextEncoder, TextDecoder, crypto: globalThis.crypto });
// every file on its own, so a topic being written elsewhere that does not load yet only costs its topics
for (const f of files(['core', 'stage', 'demo', 'topics', 'app', 'content'])) {
  try { vm.runInContext(readFileSync(join(root, 'src', f), 'utf8'), ctx, { filename: f }); }
  catch (e) { if (/^(topics|demo)\//.test(f)) console.log('WARN ' + f + ' does not load: ' + e.message); else throw e; }
}
// --extra file.js loads one more content file (a scratch file to try the checker on)
if (process.argv.includes('--extra')) { const f = process.argv[process.argv.indexOf('--extra') + 1]; vm.runInContext(readFileSync(f, 'utf8'), ctx, { filename: f }); }
const S = vm.runInContext('globalThis.Nooral', ctx);
const REALM = vm.runInContext('({ Math, BigInt, Uint8Array, Uint16Array, Uint32Array, Int32Array, Float32Array, Float64Array, ArrayBuffer, DataView, Array, Object, Number, String, JSON, Map, Set, Error, TypeError, RangeError, TextEncoder, TextDecoder })', ctx);
// code runs as the page's worker runs it: a fresh context per run, no page and no network, under a time limit
S.Sandbox.syncRunner = (job) => {
  // the built-ins of the content's own realm, so arrays and bytes the code returns compare as the page compares them
  const box = vm.createContext(Object.assign({ console: { log() {}, info() {}, warn() {}, error() {} } }, REALM));
  const out = { cases: [] };
  try { vm.runInContext('"use strict";\n' + job.src + '\n;globalThis.__f = typeof ' + job.fn + ' === "function" ? ' + job.fn + ' : undefined;', box, { timeout: job.timeLimit || 2000 }); }
  catch (e) { return /timed out/.test(e.message) ? { cases: [], timedOut: true } : { compileError: (e.name ? e.name + ': ' : '') + e.message, cases: [] }; }
  if (typeof box.__f !== 'function') return { compileError: 'no function ' + job.fn, missing: job.fn, cases: [] };
  for (let i = 0; i < job.tests.length; i++) {
    box.__args = job.tests[i].args;
    try { out.cases[i] = { got: vm.runInContext('__f(...__args)', box, { timeout: job.timeLimit || 2000 }), error: null, ms: 0 }; }
    catch (e) { if (/timed out/.test(e.message)) { out.timedOut = true; break; } out.cases[i] = { got: undefined, error: (e.name ? e.name + ': ' : '') + e.message, ms: 0 }; }
  }
  return out;
};
const C = S.Content, K = S.Check;
const only = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && all[i - 1] !== '--extra')[0];
const verbose = process.argv.includes('--verbose');
const want = (x) => !only || x.id.startsWith(only) || String(x.chapter || '').startsWith(only);
let ok = 0, bad = 0, warn = 0, answers = 0, nx = 0, nl = 0, nch = 0;
const report = (what, id, why, isWarn) => {
  if (!why.length) { ok++; if (verbose) console.log('ok   ' + what + ' ' + id); return; }
  if (isWarn) { warn++; console.log('WARN ' + what + ' ' + id + '\n    ' + why.join('\n    ')); return; }
  bad++; console.log('FAIL ' + what + ' ' + id + '\n    ' + why.join('\n    '));
};
for (const e of C.errors) { if (only && !e.includes(only)) continue; bad++; console.log('FAIL registry: ' + e); }
for (const x of [...C.chapters, ...C.lessons.values(), ...C.exercises.values(), ...C.courses.values(), ...C.challenges.values(), ...(S.topics || []), ...(S.families || [])]) {
  if (x.sample && want(x)) { warn++; console.log('WARN ' + x.id + ' is marked sample: true: placeholder content, to be replaced before release'); }
}
const short = (v) => (typeof v === 'string' ? v : JSON.stringify(v, (k, x) => (typeof x === 'bigint' ? x + 'n' : x))).replace(/\s+/g, ' ').slice(0, 100);

// ---- exercises ----------------------------------------------------------------------------------------------
const t0 = Date.now();
for (const x of C.exercises.values()) {
  if (!want(x)) continue;
  nx++;
  const why = [];
  if (!K.kinds[x.kind]) why.push('unknown kind ' + x.kind + ' (the kinds: ' + (K.kindNames || Object.keys(K.kinds)).join(', ') + ')');
  else {
    if (x.solution === undefined) why.push('no solution');
    else {
      answers++;
      const r = K.answer(x, x.solution);
      if (!r.ok) why.push('the solution fails: ' + r.message);
      else if (verbose) console.log('     ' + x.id + ' solution: ' + r.message);
    }
    // a table needs no listed wrong answer: its solution with one cell changed must fail
    let wrongs = x.wrong || [];
    if (!wrongs.length && x.kind === 'table' && x.solution && typeof x.solution === 'object') {
      const k = Object.keys(x.solution)[0], v = String(x.solution[k]);
      wrongs = [Object.assign({}, x.solution, { [k]: v === '0' ? '1' : v === '1' ? '0' : v + 'x' })];
    }
    // a kind may make its own wrong answer from the solution (a matrix with one cell changed)
    if (!wrongs.length && K.kinds[x.kind].wrongFrom) wrongs = K.kinds[x.kind].wrongFrom(x);
    if (!wrongs.length) why.push('no wrong answers to try');
    for (const w of wrongs) {
      answers++;
      const r = K.answer(x, w);
      if (r.ok) why.push('a wrong answer passes: ' + short(w));
      else if (!r.message || /could not be checked|cannot be checked/.test(r.message)) why.push('a wrong answer breaks the checker: ' + short(w) + ': ' + r.message);
      else if (verbose) console.log('     ' + x.id + ' wrong ' + short(w) + '\n       → ' + r.message);
    }
    if (x.kind === 'table') { try { K.tableSpec(x); } catch (e) { why.push('the table cannot be built: ' + e.message); } }
    if (x.kind === 'code') { try { if (!K.codeTests(x).length) why.push('no tests'); } catch (e) { why.push(e.message); } }
    if (K.kinds[x.kind] && K.kinds[x.kind].lint) for (const e of K.kinds[x.kind].lint(x) || []) why.push(e);
    if (x.kind === 'choice' && [].concat(x.answer).some((i) => !(i >= 0 && i < (x.options || []).length))) why.push('the answer is not one of the options');
  }
  if (!x.text || (Array.isArray(x.text) && !x.text.length)) why.push('no text');
  if (!x.hints || x.hints.length !== 3) why.push('three hints are needed, the last a worked answer (it has ' + ((x.hints || []).length) + ')');
  report('exercise', x.id, why);
}

// ---- lessons -------------------------------------------------------------------------------------------------
for (const l of C.lessons.values()) {
  if (!want(l)) continue;
  nl++;
  const why = [];
  if (!l.blocks || !l.blocks.length) why.push('no blocks');
  (l.blocks || []).forEach((b, i) => {
    const at = 'block ' + (i + 1) + ': ';
    if (typeof b === 'string') return;
    if (b.topic && !S.byId(b.topic)) why.push(at + 'no topic ' + b.topic);
    if (b.topic && S.byId(b.topic) && b.input) { const t = S.byId(b.topic); const e = S.checkInput(t, S.inputOf(t, Object.assign(S.inputOf(t), b.input))); if (e) why.push(at + 'the input is refused: ' + e); }
    if (b.topic && S.byId(b.topic) && b.frame) { try { const t = S.byId(b.topic); const r = t.run(S.inputOf(t, Object.assign(S.inputOf(t), b.input || {}))); if (b.frame >= r.frames.length) why.push(at + 'frame ' + b.frame + ' is past the last frame (' + (r.frames.length - 1) + ')'); } catch (e) { why.push(at + 'the topic does not run: ' + e.message); } }
    if (b.exercise && !C.exercises.has(b.exercise)) why.push(at + 'no exercise ' + b.exercise);
    if (b.lesson && !C.lessons.has(b.lesson)) why.push(at + 'no lesson ' + b.lesson);
    if (b.challenge && !C.challenges.has(b.challenge)) why.push(at + 'no challenge ' + b.challenge);
    if (b.predict && !(b.predict.answer >= 0 && b.predict.answer < (b.predict.options || []).length)) why.push(at + 'the prediction\'s answer is not an option');
    if (b.scene && !(S.Demo && S.Demo.byId(b.scene))) why.push(at + 'no demo scene ' + b.scene);
    if (b.workflow) {
      try {
        const doc = typeof b.workflow === 'string' ? (S.WF.sample ? S.WF.sample(b.workflow) : null) : S.WF.fromJSON(b.workflow);
        if (!doc) why.push(at + 'no sample workflow ' + b.workflow);
        else for (const p of S.WF.problems(doc)) why.push(at + 'the workflow: ' + p.msg);
      } catch (e) { why.push(at + 'the workflow cannot be read: ' + e.message); }
    }
    if (b.textlab && b.textlab.tab && !['bpe', 'gpt', 'sample'].includes(b.textlab.tab)) why.push(at + 'the text lab has no tab ' + b.textlab.tab);
  });
  report('lesson', l.id, why);
}

// ---- model challenges ------------------------------------------------------------------------------------------
// S.Challenges.lint(x, {verbose}) -> {problems: [..], notes: [..], answers: n} runs the reference workflow and the wrong
// ones headless and scores them on the hidden set (src/app/challenges-core.js)
for (const x of C.challenges.values()) {
  if (!want(x)) continue;
  nch++;
  const why = [];
  if (!x.hints || x.hints.length !== 3) why.push('three hints are needed (it has ' + (x.hints || []).length + ')');
  if (!S.Challenges || !S.Challenges.lint) why.push('the challenge engine (S.Challenges.lint) is missing');
  else {
    try {
      const r = S.Challenges.lint(x, { verbose });
      answers += r.answers || 0;
      why.push(...r.problems);
      if (verbose) for (const n of r.notes || []) console.log('     ' + x.id + ' ' + n);
    } catch (e) { why.push('the check failed: ' + (e.stack || e.message)); }
  }
  report('challenge', x.id, why);
}

// ---- courses and tours ------------------------------------------------------------------------------------------
let ncourse = 0, ntour = 0;
for (const c of C.courses.values()) {
  if (!want(c)) continue;
  ncourse++;
  const why = [];
  for (const s of c.steps) { const st = C.stepOf(s); if (!C.stepExists(st)) why.push('no ' + st.type + ' ' + st.id); }
  report('course', c.id, why);
}
for (const t of C.tours.values()) {
  if (!want(t)) continue;
  ntour++;
  const why = [];
  t.steps.forEach((s, i) => { if (!s.text) why.push('step ' + (i + 1) + ' has no text'); });
  if (t.steps[0] && t.steps[0].route === undefined) why.push('the first step needs a route, so the tour starts in the right place');
  report('tour', t.id, why);
}
console.log(ok + ' ok, ' + bad + ' with problems, ' + warn + ' warnings (' + nx + ' exercises, ' + answers + ' answers checked in ' + ((Date.now() - t0) / 1000).toFixed(1) + 's; ' + nl + ' lessons, ' + nch + ' challenges, ' + ncourse + ' courses, ' + ntour + ' tours)');
if (bad) process.exitCode = 1;
