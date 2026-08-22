// The narration claims check shared by parts 6, 7 and 8 (tests/claims-p06.mjs, -p07.mjs, -p08.mjs). Every topic is
// run with its default input, every preset and a spread of random valid inputs. For every frame:
//  - every value the narration quotes was recorded by the topic (S.Kit.ml frames) with the engine number it shows;
//    the quoted text must be that number, formatted, and must appear in the narration in the order quoted;
//  - every number the narration quotes must be a number the frame's meta holds (its engine state) or an input;
//  - no other numeral may appear in the narration, except the formula constants a part allows (f = ±1, x²);
//  - the wording: "a" against "an", doubled words, double spaces, a space before punctuation;
// then each topic's own check recomputes the meta values independently (scikit-learn references, a second method,
// finite differences or a hand computation of every optimiser step) and compares.
import { load, files, harness, rng } from './load.mjs';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

// the engines and the stage as the page loads them; the topic files one by one, so a file another part is still
// writing does not stop this check (a broken file of the part being checked fails its topics, which go missing)
export const S = load(['core', 'stage']);
const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', 'src');
export const broken = [];
for (const f of files(['topics'])) {
  try { vm.runInThisContext(readFileSync(join(SRC, f), 'utf8'), { filename: f }); } catch (e) { broken.push(f + ': ' + e.message); }
}
const H = S.Kit.ml;

// ---- wording -------------------------------------------------------------------------------------------------------
const VOWEL_SOUND = /^(SVM|SVC|SVR|RBF|SMO|MLP|LSTM|EM|F1|L1|L2|MSE|NLL|x|f|h|l|m|n|r|s|F|H|L|M|N|R|S|X|8|11|18|80|ε.*|α.*|η.*|8\d*|11|18|(input|output|edge|error|event|example|ellipse|eigen\w*|elbow|epoch|estimate|outlier|order|odd|even|empty|answer|ideal|initial|isolation|iteration|axis|average|accuracy|update|unknown|open|obvious|optimal|optimiser|optimum|equal|instance|integer|angle|arrow|area|element|entry|inner|old|other|hour|honest|ordinary|upper|unlucky|unusual|uphill)s?\b.*)$/;
const CONSONANT = (w) => (/^[bcdfgjkpqtvwz][a-z]/.test(w) || /^(u[ns]i|use|usu|one|once|eu|uni)/.test(w) || /^[2-79]\d*$/.test(w) || (/^1\d?\d?$/.test(w) && !/^(11|18)$/.test(w))) && !VOWEL_SOUND.test(w);
export function wording(say) {
  const s = say || '', out = [];
  if (/\S {2,}\S/.test(s)) out.push('a double space');
  if (/\s[,.;:](\s|$)/.test(s)) out.push('a space before punctuation');
  for (const m of s.matchAll(/\b([A-Za-z]{2,}) \1\b/g)) if (!/^(that|had)$/i.test(m[1])) out.push('the word "' + m[1] + '" twice');
  for (const m of s.matchAll(/(^|[^\w'′])(a|an|A|An) ([\w′'α-ωΑ-Ω-]+)/g)) {
    const art = m[2].toLowerCase(), w = m[3];
    if (art === 'a' && VOWEL_SOUND.test(w)) out.push('"a ' + w + '"');
    if (art === 'an' && CONSONANT(w)) out.push('"an ' + w + '"');
  }
  if (/\b(the the|of of|a a|an an|to to|is is)\b/i.test(s)) out.push('a doubled small word');
  if (/undefined|NaN|Infinity|\[object/.test(s)) out.push('a broken value');
  return out;
}

// ---- quoted numbers ------------------------------------------------------------------------------------------------
const leaves = (o, out, depth) => {
  if (depth > 6 || o === null || o === undefined) return out;
  if (typeof o === 'number') { out.push(o); return out; }
  if (typeof o === 'boolean' || typeof o === 'string') { if (typeof o === 'string' && /^-?\d+(\.\d+)?$/.test(o)) out.push(+o); return out; }
  if (ArrayBuffer.isView(o)) { for (const v of o) out.push(v); return out; }
  if (Array.isArray(o)) { for (const v of o) leaves(v, out, depth + 1); return out; }
  if (typeof o === 'object') for (const k of Object.keys(o)) if (k !== 'quotes') leaves(o[k], out, depth + 1);
  return out;
};
const close = (a, b) => a === b || Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
// a number written in the narration: digits with a decimal point, a true minus, a ×10 power or a percent sign
const NUM = /[−-]?\d+(?:[.,]\d+)*(?:×10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?%?/g;
export function quotes(frame, input, allowed) {
  const out = [], say = frame.say || '', qs = (frame.meta && frame.meta.quotes) || [];
  const pool = leaves(frame.meta, [], 0).concat(leaves(input, [], 0));
  // quotes are found in order as whole numbers; the formula constants the part allows are then masked in the rest
  const masked = say;
  const whole = (str, s, from, signed) => { for (let k = str.indexOf(s, from); k >= 0; k = str.indexOf(s, k + 1)) { const a = str[k - 1] || ' ', b = str[k + s.length] || ' '; if (!(signed ? /[\d.]/ : /[\d.−-]/).test(a) && !/\d/.test(b) && !(b === '.' && /\d/.test(str[k + s.length + 1] || ''))) return k; } return -1; };
  let at = 0, rest = '';
  for (const q of qs) {
    const k = whole(masked, q.s, at, q.signed);
    if (k < 0) { out.push('the quoted value "' + q.s + '" is not in the narration (in order)'); continue; }
    rest += masked.slice(at, k) + ' ';
    at = k + q.s.length;
    // the text is the number, formatted
    if (q.signed) {
      const before = masked.slice(Math.max(0, k - 3), k), neg = q.v < 0 && parseFloat(q.s) !== 0;
      if (neg && !/−\s?$/.test(before)) out.push('the term ' + q.s + ' should carry a minus sign');
      if (!neg && /−\s?$/.test(before)) out.push('the term ' + q.s + ' carries a minus sign but is ' + q.v);
    }
    const av = q.signed ? Math.abs(q.v) : q.v;
    const fmts = [H.fx(av, 0), H.fx(av, 1), H.fx(av, 2), H.fx(av, 3), H.fx(av, 4), H.sg(av, 1), H.sg(av, 2), H.sg(av, 3), H.sg(av, 4), H.pct(av), String(av)];
    if (!fmts.includes(q.s)) out.push('the quoted "' + q.s + '" is not a rounding of ' + q.v);
    if (!pool.some((v) => close(v, q.v))) out.push('the quoted ' + q.s + ' (' + q.v + ') is not a value of the frame’s state or the input');
  }
  rest += masked.slice(at);
  for (const re of allowed || []) rest = rest.replace(re, ' ');
  rest = rest.replace(/\[\[[^\]]*\]\]/g, ' ');
  // numerals the topic did not quote
  const left = rest.match(NUM);
  if (left) out.push('numbers typed into the narration, not taken from the engine: ' + [...new Set(left)].join(', '));
  return out;
}

// ---- the runner ----------------------------------------------------------------------------------------------------
/**
 * run(name, prefix RegExp for the families, {topicId: {check(ctx), random(rnd) -> values, extra: [values]}},
 *     allowed: [RegExp] formula constants that may stand in the narration as typed)
 * ctx = {S, t, input, frames, fail(msg), near(a, b, tol, msg), eq(a, b, msg)}
 */
export function run(name, family, checks, allowed) {
  const T = harness(name), only = process.argv.slice(2), rnd = rng(20261018);
  const part = name.replace('claims-', '');
  for (const b of broken) if (b.startsWith('topics/' + part)) T.ok(false, 'the file does not load: ' + b);
  let runs = 0;
  for (const t of S.topics) {
    if (!family.test(t.family)) continue;
    if (only.length && !only.some((o) => t.id.startsWith(o))) continue;
    const c = checks[t.id] || {};
    if (!checks[t.id]) T.ok(false, t.id + ': no claims check is written for this topic');
    const inputs = [{}].concat((t.presets || []).map((p) => p.values), c.extra || []);
    if (c.random) for (let k = 0; k < (c.randomRuns || 6); k++) inputs.push(c.random(rnd));
    for (const v of inputs) {
      const input = S.inputOf(t, Object.assign(S.inputOf(t), v));
      if (S.checkInput(t, input)) continue;
      const tag = t.id + ' ' + JSON.stringify(v).slice(0, 80);
      const r = S.run(t, input);
      runs++;
      if (!r.ok) { T.ok(false, tag + ': ' + r.error); continue; }
      r.frames.forEach((f, i) => {
        for (const w of wording(f.say)) T.ok(false, tag + ' frame ' + i + ': ' + w + ': ' + f.say);
        if (f.predict) for (const w of wording(typeof f.predict.ask === 'string' ? f.predict.ask : '')) T.ok(false, tag + ' frame ' + i + ' prediction: ' + w);
        for (const w of quotes(f, input, allowed)) T.ok(false, tag + ' frame ' + i + ': ' + w + ' | ' + f.say);
        T.ok(true);
      });
      if (c.check) {
        const ctx = {
          S, t, input, frames: r.frames, result: r,
          fail: (msg) => T.ok(false, tag + ': ' + msg),
          ok: (cond, msg) => T.ok(!!cond, tag + ': ' + msg),
          near: (a, b, tol, msg) => T.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), tag + ': ' + msg + ': got ' + a + ', want ' + b),
          eq: (a, b, msg) => T.eq(a, b, tag + ': ' + msg),
        };
        try { c.check(ctx); } catch (e) { T.ok(false, tag + ': the claims check fails: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' ')); }
      }
    }
  }
  console.log('  (' + runs + ' runs)');
  T.done();
}
export const phase = (frames, p) => frames.filter((f) => f.meta && f.meta.phase === p);
