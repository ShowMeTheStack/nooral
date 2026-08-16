// What the narration claims checks of parts 3, 4 and 5 share. Every topic in these parts puts the values its
// narration quotes in each frame's meta; here every number in a frame's narration must be one of them (or one of
// the inputs) at the precision shown, the wording is checked (a/an, doubled words, double spaces), and each part's
// own file recomputes the meta values independently (scikit-learn references, closed forms, a second method).
import { load, harness, rng } from './load.mjs';

export function setup(name) {
  const S = load(['core', 'stage', 'topics']);
  return { S, t: harness(name), rnd: rng(20261018) };
}

// ---- wording ----
/** does a word start with a vowel sound? letters and acronyms by the names of their letters */
export function vowelSound(w) {
  if (/^(8|11|18|8\d)(\b|\D|$)/.test(w) || /^8/.test(w)) return true;
  if (/^\d/.test(w)) return false;
  if (/^[αεηιοω]/.test(w)) return true;
  if (/^[^\x00-\x7f]/.test(w)) return false;
  const lead = w.match(/^[A-Za-z]+/);
  if (!lead) return false;
  const L = lead[0];
  // a single letter, or capitals (an acronym), or a letter then digits or marks (x1, L2, R²): said letter by letter
  if (L.length === 1 || (L === L.toUpperCase() && L.length <= 5 && !/^(AND|OR|ONE)$/.test(L))) return /^[aefhilmnorsx]/i.test(L);
  if (/^(one|once|uni|use|usu|eu|ufo)/i.test(L)) return false;
  if (/^(hour|honest|honour)/i.test(L)) return true;
  return /^[aeiou]/i.test(L);
}
export function wording(say) {
  const s = String(say || ''), out = [];
  if (/\S {2,}\S/.test(s)) out.push('a double space');
  if (/\s[,.;:](\s|$)/.test(s)) out.push('a space before punctuation');
  if (/^\s|\s$/.test(s)) out.push('space at an end');
  for (const m of s.matchAll(/\b([A-Za-z]{2,}) \1\b/g)) if (!/^(that|had)$/i.test(m[1])) out.push('the word "' + m[1] + '" twice');
  for (const m of s.matchAll(/(^|[^\w'’])(a|an|A|An) ([\w′'’²α-ω-]+)/g)) {
    const art = m[2].toLowerCase(), w = m[3];
    if (m[2] === 'A' && /^[A-Z]$/.test(w)) continue; // "A B" could be names
    const v = vowelSound(w);
    if (art === 'a' && v) out.push('"a ' + w + '"');
    if (art === 'an' && !v) out.push('"an ' + w + '"');
  }
  if (/\b(the|a|an) (the|a|an|of)\b/i.test(s)) out.push('a missing word');
  if (/\.\./.test(s.replace(/\.\.\./g, ''))) out.push('two full stops');
  return out;
}

// ---- numbers ----
/** every number written in a sentence: {v, dec, pct} (a true minus counts; digits inside words like x1 do not) */
export function numbersIn(say) {
  const out = [];
  const s = String(say || '').replace(/−/g, '-');
  for (const m of s.matchAll(/(?<![\w.·^²³⁰¹⁴⁵⁶⁷⁸⁹₀-₉])(-?\d+(?:\.\d+)?)(%?)(?![\w²³¹⁰⁴-⁹])/g)) {
    const txt = m[1], dec = txt.includes('.') ? txt.split('.')[1].length : 0;
    // a number after a spaced minus ("x − 1.86") may be the size of a negative value written as a subtraction
    const before = s.slice(Math.max(0, m.index - 3), m.index);
    out.push({ v: +txt, dec, pct: m[2] === '%', txt: m[0], sig: txt.replace(/^-?0*\.?0*/, '').replace('.', '').length, signFree: / - $/.test(before) });
  }
  return out;
}
/** all numbers inside a value (arrays, objects), at most a few thousand */
export function leaves(v, out = [], depth = 0) {
  if (out.length > 20000 || depth > 6) return out;
  if (typeof v === 'number') { if (isFinite(v)) out.push(v); }
  else if (typeof v === 'boolean') { /* skip */ }
  else if (Array.isArray(v) || ArrayBuffer.isView(v)) { for (const x of v) leaves(x, out, depth + 1); }
  else if (v && typeof v === 'object') { for (const k of Object.keys(v)) leaves(v[k], out, depth + 1); }
  return out;
}
/** does a written number stand for one of the values? (rounded as written, or to its significant digits) */
export function matches(num, vals) {
  const tol = 0.5 * Math.pow(10, -num.dec) * (1 + 1e-9) + 1e-12;
  for (const raw of vals) {
    for (const v of (num.pct ? [raw * 100, raw] : [raw]).flatMap((q) => (num.signFree ? [q, -q] : [q]))) {
      if (Math.abs(v - num.v) <= tol) return true;
      // written with fewer decimals than its size allows (3 significant digits of a large number, say)
      if (num.dec === 0 && Math.abs(v) >= 1000 && Math.abs(Math.round(v) - num.v) <= 0.5) return true;
    }
  }
  return false;
}
const CONST = [0, 1, 2, 0.5, -1];

/** the generic checks of one run: wording, and every number in a frame's narration is in its meta or the inputs */
export function generic(ctx) {
  const { r, input, fail, pass, extra } = ctx;
  const inVals = leaves(input).concat(CONST, extra || []);
  r.frames.forEach((f, i) => {
    for (const w of wording(f.say)) fail('frame ' + i + ': ' + w + ': ' + f.say);
    const vals = leaves(f.meta || {}).concat(inVals);
    for (const num of numbersIn(f.say)) {
      if (matches(num, vals)) pass();
      else fail('frame ' + i + ': the narration says ' + num.txt + ', which is not in the frame\'s values: ' + f.say.slice(0, 160));
    }
    if (!f.meta) fail('frame ' + i + ': no meta');
    for (const it of f.items) for (const k of ['d', 'text', 'label']) if (typeof it[k] === 'string' && /NaN|undefined|Infinity/.test(it[k])) { fail('frame ' + i + ': ' + it.key + ' has ' + k + ' "' + it[k].slice(0, 60) + '"'); break; }
  });
}

/** close enough: relative and absolute tolerance */
export const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
export const nearAll = (a, b, tol) => a.length === b.length && a.every((v, i) => near(v, b[i], tol));

/**
 * run a part: mods is {topicId: {random(rnd) -> values, check(ctx)}}; ctx = {S, tp, input, r, fail, pass, near}
 */
export function runPart(S, t, rnd, pattern, mods, only) {
  let runs = 0;
  const seen = new Set();
  for (const tp of S.topics) {
    if (!pattern.test(tp.family)) continue;
    if (only.length && !only.some((o) => tp.id.startsWith(o) || tp.family.startsWith(o))) continue;
    seen.add(tp.id);
    const mod = mods[tp.id] || {};
    if (!mods[tp.id]) t.ok(false, tp.id + ': no claims module');
    const inputs = [{}].concat((tp.presets || []).map((p) => p.values));
    if (mod.random) for (let k = 0; k < (mod.nRandom || 6); k++) inputs.push(mod.random(rnd));
    for (const v of inputs) {
      const input = S.inputOf(tp, Object.assign(S.inputOf(tp), v));
      if (S.checkInput(tp, input)) { if (mod.random && !tp.presets.some((p) => p.values === v)) continue; t.ok(false, tp.id + ' refuses ' + JSON.stringify(v) + ': ' + S.checkInput(tp, input)); continue; }
      const js = JSON.stringify(v), tag = tp.id + ' ' + (js.length > 80 ? js.slice(0, 77) + '…}' : js);
      let r;
      try { r = S.run(tp, input); } catch (e) { t.ok(false, tag + ': ' + e.message); continue; }
      if (!r.ok) { t.ok(false, tag + ': ' + r.error); continue; }
      runs++;
      const ctx = { S, tp, input, r, near, nearAll, fail: (m) => t.ok(false, tag + ': ' + m), pass: () => t.ok(true), extra: mod.extra ? mod.extra(input) : [] };
      generic(ctx);
      if (mod.check) { try { mod.check(ctx); } catch (e) { t.ok(false, tag + ': the check threw ' + e.message + ' ' + (e.stack || '').split('\n')[1]); } }
    }
  }
  return { runs, seen };
}
