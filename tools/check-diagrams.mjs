// Checks every drawing: node tools/check-diagrams.mjs [--demo | --pages [groups] | --topics] [--no-build] [--shots DIR]
//   [--all-shots] [--shot-frames first,last|all] [--wide-only] [--phone-only] [--light-only] [--dark-only] [--verbose]
//   [--all] [--no-warn] [--jobs N] [filter...]
// 1. The demo page (index.html#demo): every sample scene, at every frame, at desktop size (1440×900) and at phone size
//    (390×844), in the light and the dark theme. Each is audited by tools/diagram-audit.js as a whole card: its
//    drawing (shapes apart, no wire through a block, no two wires crossing or running along each other unless one is
//    drawn as a hop, labels clear of everything and inside the drawing, text no smaller than 11px at desktop size,
//    axis ticks apart) and the HTML around it (no cut text, no sideways scrolling). A filter keeps the scenes whose id
//    holds it.
// 2. The app's pages: every scene the pages register in Nooral.Checks ({group, name, hash, ready, setup, sel}: the
//    benches, the Enigma and its Bombe, the sandbox, the CTF ladder, topics, Learn and Courses) and the home page,
//    each drawn in its page at both sizes and in both themes and audited as a whole page. --pages alone runs only
//    these; --pages enigma,ctf only those groups; a filter keeps the scenes whose name holds it.
// 3. --topics: every topic whose id (or family) holds a filter, with its default input and every preset, at its
//    first, last and two middle frames, as the whole player page.
// With no mode flag, the demo page and the pages are both checked. Pages and topics are drawn from /dev.html, so
// they need no build (and helpers working side by side never race over index.html).
// Exits 1 on any problem. Warnings (WARN lines, hidden by --no-warn) do not fail it: a picture that leaves much of
// the stage empty, and at phone size text under 10px (the stage there can be pinched and zoomed).
// --jobs N (default 4) shares the work among N headless browsers at once.
import { launch } from './cdp.mjs';
import { execSync } from 'node:child_process';
import { readFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const val = (f) => (args.includes(f) ? args[args.indexOf(f) + 1] : null);
const shotsAt = val('--shots');
const jobs = Math.max(1, Math.floor(+(val('--jobs') || 4)) || 1);
const shotFrames = val('--shot-frames') || 'all';
const pagesArg = flag('--pages') && args[args.indexOf('--pages') + 1] && !args[args.indexOf('--pages') + 1].startsWith('--') ? args[args.indexOf('--pages') + 1] : null;
const filters = args.filter((a, i) => !a.startsWith('--') && !['--shots', '--jobs', '--shot-frames'].includes(args[i - 1]) && !(args[i - 1] === '--pages' && a === pagesArg));
const doDemo = flag('--demo') || (!flag('--pages') && !flag('--topics'));
const doPages = flag('--pages') || (!flag('--demo') && !flag('--topics'));
if (doDemo && !flag('--no-build')) execSync('node build.mjs', { cwd: root, stdio: 'ignore' });
if (shotsAt) mkdirSync(shotsAt, { recursive: true });
const audit = readFileSync(join(root, 'tools/diagram-audit.js'), 'utf8');
const t0 = Date.now();
let warnings = 0, problems = 0, checked = 0;
const warnLine = (where, p) => { warnings++; if (!flag('--no-warn')) console.log('WARN ' + where + ': ' + p.kind + ': ' + p.msg); };

async function pool(tasks) {
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(jobs, tasks.length) }, async () => { while (next < tasks.length) { const i = next++; await tasks[i](); } }));
}

const report = (label, list, errs, extra) => {
  const bad = list.length + errs.length;
  problems += bad;
  const lines = [];
  if (bad || flag('--verbose')) lines.push((bad ? 'FAIL ' : 'ok   ') + label + (bad ? ': ' + list.length + ' problems' + (errs.length ? ', ' + errs.length + ' console errors' : '') : '') + (extra || ''));
  for (const p of list.slice(0, flag('--all') ? 1e9 : 10)) lines.push('     ' + p.kind.padEnd(19) + p.msg + (p.a ? '  [' + p.a + (p.b ? ' | ' + p.b : '') + ']' : ''));
  if (list.length > 10 && !flag('--all')) lines.push('     … and ' + (list.length - 10) + ' more (--all)');
  for (const e of errs.slice(0, 3)) lines.push('     console: ' + e.text.slice(0, 300));
  if (lines.length) console.log(lines.join('\n'));
  return bad;
};
const shotName = (s) => s.replace(/[^\w.-]+/g, '_').slice(0, 140);

const sizes = (flag('--phone-only') ? [] : [[1440, 900]]).concat(flag('--wide-only') ? [] : [[390, 844]]);
const themes = (flag('--dark-only') ? [] : ['light']).concat(flag('--light-only') ? [] : ['dark']);
if (flag('--topics')) await topics();
if (doDemo) await demo();
if (doPages) await pages();
process.exit(problems ? 1 : 0);

async function demo() {
// the scenes, from a first look at the page
const ids = await (async () => {
  const b = await launch({ width: 1440, height: 900 });
  try {
    await b.goto('index.html#demo');
    await b.waitFor('window.__nooralReady === true', 30000);
    return JSON.parse(await b.eval('JSON.stringify(Nooral.Demo.scenes.map((s) => s.id))')).filter((id) => !filters.length || filters.some((f) => id.includes(f)));
  } finally { await b.close(); }
})();
const tasks = [];
const per = Math.max(1, Math.ceil(ids.length / Math.max(1, Math.round(jobs / (sizes.length * themes.length)))));
for (const [W, H] of sizes) for (const theme of themes) for (let k = 0; k < ids.length; k += per) tasks.push(() => part(W, H, theme, ids.slice(k, k + per)));
await pool(tasks);
console.log(checked + ' frames checked (' + ids.length + ' demo scenes, ' + sizes.map((s) => s.join('×')).join(' and ') + ', ' + themes.join(' and ') + '): ' + (problems ? problems + ' problems' : 'no problems') + (warnings ? ', ' + warnings + ' warnings' : '') + ' in ' + ((Date.now() - t0) / 1000).toFixed(0) + ' s');

}

async function part(W, H, theme, list) {
  const phone = W < 600;
  const b = await launch({ width: W, height: H, mobile: phone });
  try {
    await b.goto('index.html#demo');
    await b.waitFor('window.__nooralReady === true', 30000);
    await b.eval(`Nooral.app.settings.animate = false; Nooral.app.setTheme('${theme}'); 1`);
    for (const id of list) {
      const n = await b.eval(`(Nooral.app.byId[${JSON.stringify(id)}].frames || []).length`);
      if (!n) { problems++; console.log('FAIL ' + id + ' @' + W + ' ' + theme + ': the scene has no frames: ' + b.logs.map((l) => l.text).join(' | ').slice(0, 400)); b.logs.length = 0; continue; }
      for (let i = 0; i < n; i++) {
        await b.eval(`(() => { const p = Nooral.app.byId[${JSON.stringify(id)}]; p.go(${i}, true); p.stage.finish(); return 1; })()`);
        const label = id + ' frame ' + i + ' @' + W + ' ' + theme;
        const list2 = JSON.parse(await b.eval(audit + `\n;JSON.stringify(NooralAudit.run(document.getElementById(${JSON.stringify(id)}), { html: true, page: true }))`)).filter((p) => !p.warn || (warnLine(label, p), false));
        const errs = b.logs.filter((l) => l.type === 'error' || l.type === 'exception');
        b.logs.length = 0;
        checked++;
        const bad = list2.length + errs.length;
        problems += bad;
        const lines = [];
        if (bad || flag('--verbose')) lines.push((bad ? 'FAIL ' : 'ok   ') + label + (bad ? ': ' + list2.length + ' problems' + (errs.length ? ', ' + errs.length + ' console errors' : '') : ''));
        for (const p of list2.slice(0, flag('--all') ? 1e9 : 8)) lines.push('     ' + p.kind.padEnd(19) + p.msg + (p.a ? '  [' + p.a + (p.b ? ' | ' + p.b : '') + ']' : ''));
        if (list2.length > 8 && !flag('--all')) lines.push('     … and ' + (list2.length - 8) + ' more (--all)');
        for (const e of errs.slice(0, 3)) lines.push('     console: ' + e.text.slice(0, 300));
        if (lines.length) console.log(lines.join('\n'));
        const want = shotFrames === 'all' || shotFrames.split(',').some((f) => (f === 'first' && i === 0) || (f === 'last' && i === n - 1) || f === String(i));
        if (shotsAt && (bad || (flag('--all-shots') && want))) {
          const box = await b.eval(`(() => { const r = document.getElementById(${JSON.stringify(id)}).getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height }; })()`);
          await b.screenshot(join(shotsAt, id + '-f' + String(i).padStart(2, '0') + '-' + W + '-' + theme + '.png'), box);
        }
      }
    }
  } finally { await b.close(); }
}

/** split a list into n parts of about the same size, keeping runs with the same key together */
function slice(list, k, n, keyOf) {
  const runs = [];
  for (const x of list) { const key = keyOf(x); if (runs.length && runs[runs.length - 1].key === key) runs[runs.length - 1].items.push(x); else runs.push({ key, items: [x] }); }
  const per = list.length / n, out = [];
  let done = 0;
  for (const r of runs) { const at = Math.floor(done / per); if (at === k) out.push(...r.items); done += r.items.length; }
  return out;
}
// ---- 2. the app's pages ----------------------------------------------------------------------------------------------
async function pages() {
  const groups = pagesArg && pagesArg !== 'all' ? pagesArg.split(',') : null;
  const t1 = Date.now();
  const before = checked, p0 = problems;
  const tasks = [];
  const parts = Math.max(1, Math.round(jobs / (sizes.length * themes.length)));
  for (const [W, H] of sizes) for (const theme of themes) for (let k = 0; k < parts; k++) tasks.push(() => pagePart(W, H, theme, k, parts));
  await pool(tasks);
  console.log((checked - before) + ' page scenes checked (' + sizes.map((s) => s.join('×')).join(' and ') + ', ' + themes.join(' and ') + '): ' + (problems - p0 ? (problems - p0) + ' problems' : 'no problems') + (warnings ? ', ' + warnings + ' warnings' : '') + ' in ' + ((Date.now() - t1) / 1000).toFixed(0) + ' s');

  async function pagePart(W, H, theme, part, nparts) {
    const phone = W < 600;
    const b = await launch({ width: W, height: H, mobile: phone });
    try {
      await b.send('Page.navigate', { url: b.url + 'dev.html#' });
      await b.waitFor('window.__nooralReady === true', 30000);
      await b.eval(`Nooral.app.settings.theme = '${theme}'; Nooral.app.settings.animate = false; Nooral.app.settings.sound = false; if (Nooral.Learner) Nooral.Learner.settings.predict = false; Nooral.app.applyTheme(); 1`);
      const all = JSON.parse(await b.eval('JSON.stringify((Nooral.Checks || []).map((c, i) => ({ i, group: c.group, name: c.name, hash: c.hash })))'))
        .concat([{ i: -1, group: 'home', name: 'home page', hash: '' }])
        .filter((x) => (!groups || groups.includes(x.group)) && (!filters.length || filters.some((f) => x.name.includes(f) || x.group.includes(f))));
      const plan = slice(all, part, nparts, (x) => x.hash);
      let lastHash = null;
      for (const x of plan) {
        let err = null;
        try {
          await b.eval(`(async () => {
            const c = ${x.i} >= 0 ? Nooral.Checks[${x.i}] : { hash: '', sel: '#main' };
            const wait = async (expr, what) => { let n = 0; while (!(await (0, eval)(expr))) { if (++n > 600) throw new Error('timed out waiting for ' + what); await new Promise((r) => setTimeout(r, 25)); } };
            if (${JSON.stringify(lastHash)} !== c.hash || c.fresh) { window.__nooralReady = false; if (location.hash.replace(/^#/, '') === c.hash) Nooral.app.route(); else location.hash = c.hash; await wait('window.__nooralReady === true', 'the page'); }
            if (c.ready) { let n = 0; while (!(await (0, eval)(c.ready)) && ++n < 60) await new Promise((r) => setTimeout(r, 25)); }
            if (c.setup) await (0, eval)('(async () => { ' + c.setup + ' })()');
            if (c.ready) await wait(c.ready, 'ready');
            await new Promise((r) => setTimeout(r, 260));
            if (Nooral.app.hero) Nooral.app.hero.stop();
          })().then(() => 1)`);
        } catch (e) { err = e.message; }
        lastHash = x.hash;
        checked++;
        const label = x.group + ' / ' + x.name + ' @' + W + ' ' + theme;
        if (err) { problems++; lastHash = null; console.log('FAIL ' + label + ': setup failed: ' + err.slice(0, 240)); continue; }
        const list = JSON.parse(await b.eval(audit + `\n;(() => { const c = ${x.i} >= 0 ? Nooral.Checks[${x.i}] : { sel: '#main' }; return JSON.stringify(NooralAudit.run(document.querySelector(c.sel || '#main') || document.body, { html: true, page: true })); })()`)).filter((p) => !p.warn || (warnLine(label, p), false));
        if (await b.eval('document.documentElement.scrollWidth > document.documentElement.clientWidth + 1')) list.push({ kind: 'page-wide', msg: 'the page scrolls sideways' });
        const errs = b.logs.filter((l) => l.type === 'error' || l.type === 'exception');
        b.logs.length = 0;
        const bad = report(label, list, errs);
        if (shotsAt && (bad || flag('--all-shots'))) await b.screenshot(join(shotsAt, shotName(label) + '.png'));
      }
    } finally { await b.close(); }
  }
}

// ---- 3. topics -------------------------------------------------------------------------------------------------------
async function topics() {
  const t1 = Date.now(), before = checked, p0 = problems;
  const waitPlayer = (id) => `await new Promise((r, j) => { let n = 0; const w = () => (Nooral.app.page && Nooral.app.page.player && Nooral.app.page.player.t.id === ${JSON.stringify(id)}) ? r() : ++n > 400 ? j(new Error('no player')) : setTimeout(w, 30); w(); });`;
  const tasks = [];
  for (const [W, H] of sizes) for (let k = 0; k < jobs; k++) tasks.push(() => part(W, H, k));
  await pool(tasks);
  console.log((checked - before) + ' topic pictures checked (' + sizes.map((s) => s.join('×')).join(' and ') + ', ' + themes.join(' and ') + '): ' + (problems - p0 ? (problems - p0) + ' problems' : 'no problems') + (warnings ? ', ' + warnings + ' warnings' : '') + ' in ' + ((Date.now() - t1) / 1000).toFixed(0) + ' s');

  async function part(W, H, k) {
    const phone = W < 600;
    const b = await launch({ width: W, height: H, mobile: phone });
    try {
      await b.send('Page.navigate', { url: b.url + 'dev.html#topics' });
      await b.waitFor('window.__nooralReady === true', 30000);
      const all = JSON.parse(await b.eval(`(() => {
        const T = Nooral, out = [], f = ${JSON.stringify(filters)};
        for (const t of T.topics || []) {
          if (f.length && !f.some((x) => t.id.includes(x) || t.family.includes(x))) continue;
          [{ label: 'default', values: {} }].concat(t.presets || []).forEach((p, k) => {
            const r = T.run(t, T.inputOf(t, Object.assign(T.inputOf(t), p.values)));
            const m = r.frames.length;
            for (const i of [...new Set([0, Math.floor(m / 3), Math.floor(2 * m / 3), m - 1])]) out.push({ id: t.id, preset: k, frame: i, name: t.id + ' / ' + p.label + ' / frame ' + i });
          });
        }
        return JSON.stringify(out);
      })()`));
      const plan = slice(all, k, jobs, (x) => x.id);
      if (!plan.length) return;
      for (const theme of themes) {
        await b.eval(`Nooral.Learner.settings.predict = false; Nooral.app.settings.theme = '${theme}'; Nooral.app.settings.animate = false; Nooral.app.applyTheme(); 1`);
        let last = null;
        for (const x of plan) {
          const setup = (last === x.id ? '' : `location.hash = 'topic=${x.id}'; ${waitPlayer(x.id)}`) + `
            const T = Nooral, t = T.byId('${x.id}'), p = T.app.page.player, pr = [{ values: {} }].concat(t.presets || [])[${x.preset}];
            p.input = T.inputOf(t, Object.assign(T.inputOf(t), pr.values)); p.buildInputs(); p.run(); p.go(${x.frame}, true); p.stage.finish && p.stage.finish();`;
          last = x.id;
          let err = null;
          try { await b.eval('(async () => { ' + setup + ' })().then(() => 1)'); } catch (e) { err = e.message; }
          await b.sleep(120);
          checked++;
          const label = x.name + ' @' + W + ' ' + theme;
          if (err) { problems++; last = null; console.log('FAIL ' + label + ': setup failed: ' + err.slice(0, 200)); continue; }
          const list = JSON.parse(await b.eval(audit + '\n;JSON.stringify(NooralAudit.run(document.querySelector(".topic") || document.body, { html: true, page: true }))')).filter((p) => !p.warn || (warnLine(label, p), false));
          if (await b.eval('document.documentElement.scrollWidth > document.documentElement.clientWidth + 1')) list.push({ kind: 'page-wide', msg: 'the page scrolls sideways' });
          const errs = b.logs.filter((l) => l.type === 'error' || l.type === 'exception');
          b.logs.length = 0;
          const bad = report(label, list, errs);
          if (shotsAt && (bad || flag('--all-shots'))) await b.screenshot(join(shotsAt, shotName(label) + '.png'));
        }
      }
    } finally { await b.close(); }
  }
}
