// Part 1 (data and features): every topic run with its default, every preset and a spread of random valid inputs.
// For every frame: the wording (a/an, doubled words, stray spaces), and every number the narration quotes must be a
// value the frame carries from the engines (its meta), an input, or a named constant of the topic. Then per topic
// the values themselves are checked: against scikit-learn, numpy and scipy (tests/ref/p12/sklearn.json, made by
// tests/ref/gen_p12.py) for the presets, and against a second computation here for every input.
// Prints "claims-p01 N passed, M failed". The shared machinery is exported for tests/claims-p02.mjs.
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { load, harness, rng } from './load.mjs';
import { jobsFor, refKey } from './ref/p12_jobs.mjs';

const here = dirname(fileURLToPath(import.meta.url));

// ---- wording ----
const vowelSound = (w) => {
  if (/^[A-Za-z]$/.test(w) || /^[A-Z][A-Z0-9²-]*$/.test(w)) return /^[AEFHILMNORSX]/.test(w.toUpperCase());
  if (/^\d/.test(w)) return /^(8|11|18)(\D|$)/.test(w) || /^8\d/.test(w);
  if (/^(hour|honest|heir)/i.test(w)) return true;
  return /^[aeiou]/i.test(w) && !/^(one|once|uni|use|usu|eu|ur[ai])/i.test(w);
};
export function wording(say) {
  const s = String(say || '').replace(/"[^"]*"/g, '"…"'), out = [];
  if (/\S {2,}\S/.test(s)) out.push('a double space');
  if (/\s[,.;:](\s|$)/.test(s)) out.push('a space before punctuation');
  if (/\.\./.test(s.replace(/…/g, ''))) out.push('two full stops');
  for (const m of s.matchAll(/\b([A-Za-z]{2,}) \1\b/g)) if (!/^(that|had)$/i.test(m[1])) out.push('the word "' + m[1] + '" twice');
  for (const m of s.matchAll(/(^|[^\w'’])(a|an|A|An) ([\w²’'-]+)/g)) {
    const art = m[2].toLowerCase(), w = m[3], vowel = vowelSound(w);
    if (art === 'a' && vowel && !(m[2] === 'A' && /^[A-Z]$/.test(w))) out.push('"a ' + w + '"');
    if (art === 'an' && !vowel) out.push('"an ' + w + '"');
  }
  if (/\b(the|a|an|of|to|is) (the|a|an|of|is)\b/i.test(s) && /\b(the the|of of|a a|an an|to the of)\b/i.test(s)) out.push('a missing word');
  if (/\bundefined\b|\bNaN\b|\[object/.test(s)) out.push('a broken value');
  return out;
}

// ---- numbers ----
/** every string inside a value (deep): names that may hold digits (cultivar 3), which are not quoted numbers */
export function stringsOf(v, out = []) {
  if (typeof v === 'string') out.push(v);
  else if (Array.isArray(v)) for (const x of v) stringsOf(x, out);
  else if (v && typeof v === 'object') for (const k of Object.keys(v)) stringsOf(v[k], out);
  return out;
}
/** every number inside a value (deep), and the lengths of its arrays */
export function numbersOf(v, out = []) {
  if (typeof v === 'number') { if (Number.isFinite(v)) out.push(v); }
  else if (typeof v === 'boolean') { /* no number */ }
  else if (Array.isArray(v)) { out.push(v.length); for (const x of v) numbersOf(x, out); }
  else if (v && typeof v === 'object') for (const k of Object.keys(v)) numbersOf(v[k], out);
  return out;
}
/** the numbers quoted in a sentence: [text, value, decimals, percent] (names such as Q1, x2, F1, c3 are not numbers) */
export function quoted(say) {
  const s = String(say || '').replace(/−/g, '-'), out = [];
  for (const m of s.matchAll(/(^|[^\w.²³⁰¹⁴⁵⁶⁷⁸⁹ₖᵦ_])(-?\d+(?:\.\d+)?)(%?)(?![\w²³¹⁰⁴⁵⁶⁷⁸⁹])/g)) {
    if (/[A-Za-z]$/.test(m[1])) continue;
    const t = m[2], d = t.includes('.') ? t.split('.')[1].length : 0;
    out.push([t + m[3], +t, d, !!m[3]]);
  }
  return out;
}
const matches = (tok, v) => {
  const [, x, d, pct] = tok, tol = 0.5 * Math.pow(10, -d) + 1e-9 * Math.max(1, Math.abs(x));
  const cand = pct ? [v * 100, v] : [v, v * 100];
  return cand.some((c) => Math.abs(c - x) <= tol || Math.abs(Math.abs(c) - Math.abs(x)) <= tol);
};
/** the quoted numbers of a sentence that match no allowed value */
export function untraced(say, allowed, names) {
  let s = String(say || '');
  for (const nm of (names || []).filter((x) => /\d/.test(x)).sort((a, b) => b.length - a.length)) s = s.replace(new RegExp('(?<![\\w.])' + nm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?!\\w|\\.\\d)', 'g'), ' ');
  return quoted(s).filter((tok) => !(tok[2] === 0 && Math.abs(tok[1]) <= 1) && !allowed.some((v) => matches(tok, v))).map((t) => t[0]); }

// ---- comparing values ----
export const close = (a, b, tol = 1e-6) => {
  if (Array.isArray(b)) return Array.isArray(a) && a.length === b.length && b.every((v, i) => close(a[i], v, tol));
  if (b === null || b === undefined) return a === null || a === undefined || Number.isNaN(a);
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
};

/**
 * Run the claims of the topics whose family matches `family`: CHECKS[topicId] = { constants: [n..], random(rnd) ->
 * values, check(ctx) } with ctx = { S, t, input, r, ref (the scikit-learn results of a preset, or null), near(got,
 * want, what, tol), ok(cond, what), frame(phase) }.
 */
export function runClaims(name, family, CHECKS) {
  const S = load(['core', 'stage', 'topics']);
  const t0 = harness(name);
  const refFile = join(here, 'ref', 'p12', 'sklearn.json');
  const REF = existsSync(refFile) ? JSON.parse(readFileSync(refFile, 'utf8')) : {};
  const inFile = join(here, 'ref', 'p12', 'inputs.json'), INPUTS = existsSync(inFile) ? JSON.parse(readFileSync(inFile, 'utf8')) : {};
  const only = process.argv.slice(2);
  const rnd = rng(20261018);
  let runs = 0, refs = 0;
  for (const t of S.topics) {
    if (!family.test(t.family)) continue;
    if (only.length && !only.some((o) => t.id.startsWith(o))) continue;
    const C = CHECKS[t.id] || {};
    if (!CHECKS[t.id]) t0.ok(false, t.id + ': no claims check is written for this topic');
    const inputs = [{}].concat((t.presets || []).map((p) => p.values));
    if (C.random) for (let k = 0; k < (C.nRandom || 6); k++) inputs.push(C.random(rnd));
    for (const v of inputs) {
      const input = S.inputOf(t, Object.assign(S.inputOf(t), v));
      if (S.checkInput(t, input)) { if (!C.random || inputs.indexOf(v) <= (t.presets || []).length) t0.ok(false, t.id + ' ' + JSON.stringify(v) + ': its input is refused'); continue; }
      const tag = t.id + ' ' + JSON.stringify(v).slice(0, 70);
      const r = S.run(t, input);
      runs++;
      if (!r.ok) { t0.ok(false, tag + ': ' + r.error); continue; }
      const consts = (C.constants || []).concat(numbersOf(input));
      r.frames.forEach((f, i) => {
        for (const w of wording(f.say)) t0.ok(false, tag + ' frame ' + i + ': ' + w + ': ' + f.say);
        if (f.predict) for (const w of wording(f.predict.ask)) t0.ok(false, tag + ' frame ' + i + ' prediction: ' + w);
        const allowed = numbersOf(f.meta || {}).concat(consts);
        const bad = untraced(f.say, allowed, stringsOf(f.meta || {}).concat(C.names || []));
        t0.ok(!bad.length, tag + ' frame ' + i + ': the narration quotes ' + bad.join(', ') + ', which the frame does not carry: ' + f.say);
      });
      const key = refKey(t.id, input), hasRef = !!REF[key] && JSON.stringify(INPUTS[key]) === JSON.stringify(jobsFor(S, t.id, input));
      if (REF[key] && !hasRef) t0.ok(false, tag + ': the stored reference inputs differ from this run (run tests/ref/p12_jobs.mjs and gen_p12.py again)');
      if (hasRef) refs++;
      const ctx = {
        S, t, input, r, ref: hasRef ? REF[key] : null,
        near: (got, want, what, tol) => t0.ok(close(got, want, tol === undefined ? 1e-6 : tol), tag + ': ' + what + ': got ' + JSON.stringify(got) + ', want ' + JSON.stringify(want)),
        ok: (c, what) => t0.ok(!!c, tag + ': ' + what),
        frame: (phase) => r.frames.find((f) => f.meta && f.meta.phase === phase),
        frames: (phase) => r.frames.filter((f) => f.meta && f.meta.phase === phase),
      };
      try { if (C.check) C.check(ctx); } catch (e) { t0.ok(false, tag + ': the check fails: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' ')); }
    }
  }
  console.log('  (' + runs + ' runs, ' + refs + ' with scikit-learn references)');
  t0.done();
}

// ---------------------------------------------------------------------------------------------------------------
// Part 1's checks
// ---------------------------------------------------------------------------------------------------------------
const pick = (rnd, a) => a[Math.floor(rnd() * a.length)];
const int = (rnd, a, b) => a + Math.floor(rnd() * (b - a + 1));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const sdev = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
const pctile = (a, q) => { const s = a.slice().sort((x, y) => x - y), h = (s.length - 1) * q, lo = Math.floor(h); return s[lo] + (h - lo) * ((s[Math.min(lo + 1, s.length - 1)]) - s[lo]); };
const corr = (x, y) => { const mx = mean(x), my = mean(y); let a = 0, b = 0, c = 0; x.forEach((v, i) => { a += (v - mx) * (y[i] - my); b += (v - mx) ** 2; c += (y[i] - my) ** 2; }); return a / Math.sqrt(b * c); };

export const P01 = {
  'data-table': {
    constants: [10],
    random: (rnd) => ({ set: pick(rnd, ['harvest', 'buses', 'loans', 'rents', 'penguins', 'iris', 'wine', 'digits']), rows: int(rnd, 2, 8) }),
    check({ S, input, frame, near, ok }) {
      if (input.set === 'digits') { const f = frame('image'); ok(f.meta.n === 784, 'an image has 784 pixels'); const x = S.Data.load('digits', { n: 60 }).X[0]; near(f.meta.on, Array.from(x).filter((v) => v > 0).length, 'inked pixels'); return; }
      const ds = S.Data.load(input.set), f0 = frame('table');
      near(f0.meta.nRows, ds.X.length, 'rows'); near(f0.meta.d, ds.features.length, 'features');
      const fe = frame('feature'), c = ds.X.map((r) => r[fe.meta.j]).filter((v) => Number.isFinite(v));
      near(fe.meta.min, Math.min(...c), 'min'); near(fe.meta.max, Math.max(...c), 'max'); near(fe.meta.mean, mean(c), 'mean'); near(fe.meta.miss, ds.X.length - c.length, 'missing cells');
      const lab = frame('label');
      if (lab.meta.isClass) near(lab.meta.classCounts, lab.meta.classCounts.map((_, k) => ds.y.filter((v) => v === k).length), 'class counts');
      else near([lab.meta.ysum.min, lab.meta.ysum.max, lab.meta.ysum.mean], [Math.min(...ds.y), Math.max(...ds.y), mean(ds.y)], 'label range');
      const cat = frame('category');
      if (cat) near(cat.meta.counts, cat.meta.counts.map((_, v) => ds.X.filter((r) => r[cat.meta.j] === v).length), 'category counts');
      const m = frame('matrix');
      near(m.meta.missAll, ds.X.reduce((a, r) => a + r.filter((v) => !Number.isFinite(v)).length, 0), 'missing cells in all');
    },
  },
  'data-describe': {
    constants: [1.5],
    random: (rnd) => { const set = pick(rnd, ['harvest', 'buses', 'loans', 'rents', 'pumps', 'iris', 'penguins', 'wine']); const k = { harvest: 3, buses: 4, loans: 5, rents: 4, pumps: 4, iris: 4, penguins: 5, wine: 13 }[set]; return { set, col: int(rnd, 0, k - 1), bins: int(rnd, 4, 30) }; },
    check({ S, input, frame, near, ref }) {
      const H = S.P01, ds = H.data(input.set), x = H.col(ds, H.numCols(ds)[input.col]).filter(H.isNum);
      const b = frame('box').meta, m = frame('mean').meta, h = frame('hist').meta, w = frame('whiskers').meta;
      near([b.q1, b.med, b.q3], [pctile(x, 0.25), pctile(x, 0.5), pctile(x, 0.75)], 'quartiles'); near(m.mean, mean(x), 'mean'); near(m.sd, sdev(x), 'sd');
      near(H.sum(h.counts), x.length, 'histogram total');
      const iqr = b.q3 - b.q1; near(w.out, x.filter((v) => v < b.q1 - 1.5 * iqr || v > b.q3 + 1.5 * iqr).length, 'values beyond the whiskers');
      if (ref) { const R = ref[0]; near([b.q1, b.med, b.q3, m.mean, m.sd], [R.q1, R.med, R.q3, R.mean, R.sd], 'numpy summaries', 1e-9); near(h.counts, R.counts, 'numpy histogram'); }
    },
  },
  'data-scatter-matrix': {
    random: (rnd) => { const set = pick(rnd, ['iris', 'penguins', 'wildflowers', 'wine', 'exam', 'loans']); const n = { iris: 4, penguins: 5, wildflowers: 4, wine: 13, exam: 3, loans: 5 }[set]; const k = int(rnd, 2, Math.min(4, n)); return { set, k, from: int(rnd, 0, n - k) }; },
    check({ S, input, frame, frames, near, ok, ref }) {
      const d = frame('diagonal').meta, s = frame('summary').meta;
      ok(d.eta[d.best] === Math.max(...d.eta), 'the best column has the largest η²');
      for (const f of frames('pair')) ok(Math.abs(f.meta.r) <= Math.abs(s.r) + 1e-12, 'the summary pair has the largest |r|');
      if (ref) { near(d.eta, ref[0].eta, 'η² against numpy', 1e-9); for (const f of frames('pair')) near(f.meta.r, ref[0].R[f.meta.a][f.meta.b], 'r against numpy', 1e-9); }
    },
  },
  'data-correlation': {
    check({ frame, near, ref }) {
      const r = frame('r').meta, q = frame('quarters').meta, sum = frame('sum').meta;
      near(r.r, sum.sxy / Math.sqrt(r.sxx * r.syy), 'r from its sums');
      near(sum.sPos + sum.sNeg, sum.sxy, 'positive and negative products add up');
      if (ref) { near(r.r, ref[0].r, 'Pearson against numpy', 1e-9); near(r.spearman, ref[0].spearman, 'Spearman against scipy', 1e-9); const w = frame('without'); if (w) near(w.meta.r2, ref[1].r, 'r without the outlier', 1e-9); const m = frame('matrix'); if (m) near(m.meta.R, ref[1].R, 'the matrix of r', 1e-9); }
      void q;
    },
  },
  'data-missing': {
    constants: [5],
    check({ S, input, frame, near, ref, ok }) {
      const g = frame('gaps').meta, cmp = frame('compare').meta.cmp, jc = g.jc;
      const GAP = { penguins: ['penguins', {}], pumps: ['pumps', { missing: 0.04, n: 400 }], harvest: ['harvest', { missing: 0.1, n: 200 }], rents: ['rents', { missing: 0.2, n: 150 }] }[input.set];
      const X = S.P01.data(GAP[0], GAP[1]).X, ok1 = (v) => Number.isFinite(v);
      near(g.gaps, X.reduce((a, r) => a + r.filter((v) => !ok1(v)).length, 0), 'missing cells');
      near(g.rowsWith, X.filter((r) => r.some((v) => !ok1(v))).length, 'rows with a gap');
      const present = X.map((r) => r[jc]).filter(ok1);
      near([g.mean0, g.sd0], [mean(present), sdev(present)], 'present values');
      const complete = X.filter((r) => r.every(ok1)).map((r) => r[jc]);
      const drop = cmp.find((c) => c[0] === 'drop');
      near([drop[1], drop[2], drop[3]], [complete.length, mean(complete), sdev(complete)], 'dropping rows');
      const fill = frame('fill');
      if (fill && fill.meta.fv !== null && input.how !== 'knn') near(fill.meta.fv, input.how === 'median' ? pctile(present, 0.5) : input.how === 'mean' ? mean(present) : fill.meta.fv, 'the fill value');
      if (ref) {
        const R = ref[0];
        near(drop[1], R.dropKept, 'rows kept, against pandas-free numpy');
        for (const [how, , m, sd] of cmp) { if (how === 'drop') continue; const col = R[how].map((row) => row[jc]); near([m, sd], [mean(col), sdev(col)], how + ' filling against scikit-learn', how === 'knn' ? 5e-3 : 1e-9); } // equal distances may pick different neighbours
      }
    },
  },
  'data-outliers': {
    constants: [1.5, 3, 100],
    random: (rnd) => ({ data: pick(rnd, ['pumps', 'typo', 'years', 'rain']), rule: pick(rnd, ['iqr', 'z']), k: pick(rnd, [1, 1.5, 2, 2.5, 3, 3.5, 4]), act: pick(rnd, ['clip', 'drop']) }),
    check({ S, input, frame, near, ref }) {
      const l = frame('look').meta, f = frame('flag').meta, a = frame('act').meta;
      const OUT = { pumps: ['pumps', { missing: 0 }, 0], typo: ['wildflowers', {}, 2, true], years: ['loans', {}, 2], rain: ['buses', {}, 2] }[input.data];
      const v = S.P01.col(S.P01.data(OUT[0], OUT[1]), OUT[2]).slice();
      if (OUT[3]) v[7] = +(v[7] * 100).toFixed(1);
      const q1 = pctile(v, 0.25), q3 = pctile(v, 0.75), m = mean(v), sd = sdev(v);
      const lo = input.rule === 'iqr' ? q1 - input.k * (q3 - q1) : m - input.k * sd, hi = input.rule === 'iqr' ? q3 + input.k * (q3 - q1) : m + input.k * sd;
      near([f.lo, f.hi], [lo, hi], 'fences'); near(f.nf, v.filter((x) => x < lo || x > hi).length, 'flagged values');
      const after = input.act === 'clip' ? v.map((x) => Math.min(hi, Math.max(lo, x))) : v.filter((x) => x >= lo && x <= hi);
      near([a.m1, a.s1], [mean(after), sdev(after)], 'after the action');
      if (ref) near([l.q1, l.q3, l.mean, l.sd], [ref[0].q1, ref[0].q3, ref[0].mean, ref[0].sd], 'against numpy', 1e-9);
      const tr = frame('truth');
      if (tr) { const y = S.P01.data('pumps', { missing: 0 }).y; near(tr.meta.tp, v.filter((x, i) => (x < lo || x > hi) && y[i] === 1).length, 'flagged faults'); near(tr.meta.faults, y.filter((q) => q === 1).length, 'faults'); }
    },
  },
  'data-scaling': {
    random: (rnd) => ({ data: pick(rnd, ['wine', 'loans', 'outlier']), scaler: pick(rnd, ['standard', 'minmax', 'robust']), k: int(rnd, 1, 15), q: int(rnd, 0, 40) }),
    check({ input, frame, near, ref, ok }) {
      const fit = frame('fit').meta, acc = frame('acc').meta, sc = frame('scaled').meta;
      ok(sc.count <= input.k && frame('raw').meta.count <= input.k, 'a vote count is at most k');
      if (ref) {
        const R = ref[0][input.scaler];
        near(fit.centre, R.centre, 'scaler centre against scikit-learn', 1e-9); near(fit.scale, R.scale, 'scaler scale against scikit-learn', 1e-9);
        near(acc.accSc, R.acc, 'scaled k-NN accuracy against scikit-learn', 0.02); near(acc.accRaw, ref[0].raw.acc, 'raw k-NN accuracy against scikit-learn', 0.02);
      }
    },
  },
  'data-encoding': {
    check({ frame, near, ref, ok }) {
      const raw = frame('raw').meta, of = frame('ordfit').meta, oh = frame('ohfit').meta;
      near(oh.pred, raw.means, 'one-hot predictions are the category means', 1e-8);
      ok(oh.mse <= of.mse + 1e-9, 'one-hot fits at least as well as one ordinal column');
      if (ref) { const R = ref[0]; near([of.coef, of.intercept], [R.coef, R.intercept], 'ordinal line against scikit-learn', 1e-8); near(oh.pred, R.predOh, 'one-hot fit against scikit-learn', 1e-8); near([of.mse, oh.mse], [R.mseOrd, R.mseOh], 'errors against scikit-learn', 1e-8); }
    },
  },
  'data-binning': {
    random: (rnd) => ({ data: pick(rnd, ['sine', 'rain', 'step']), how: pick(rnd, ['uniform', 'quantile']), n: int(rnd, 2, 24) }),
    check({ frame, near, ref, ok }) {
      const e = frame('edges').meta, st = frame('steps').meta, l = frame('line').meta;
      ok(e.counts.reduce((a, b) => a + b, 0) > 0, 'the bins hold the training rows');
      if (ref) { const R = ref[0]; near(e.edges, R.edges, 'bin edges against scikit-learn', 1e-9); near([l.lineTr, l.lineTe], [R.lineTr, R.lineTe], 'line errors against scikit-learn', 1e-8); if (R.edges.length === e.edges.length) near([st.binTr, st.binTe], [R.binTr, R.binTe], 'binned errors against scikit-learn', 1e-6); }
    },
  },
  'data-poly': {
    constants: [12, 4, 3],
    random: (rnd) => ({ data: pick(rnd, ['poly', 'sine', 'abs']), degree: int(rnd, 1, 12) }),
    check({ input, frame, near, ref }) {
      const d = frame('degrees').meta, f = frame('fit').meta, l = frame('line').meta;
      near([f.tr, f.te], [d.trs[input.degree - 1], d.tes[input.degree - 1]], 'the fit matches its degree');
      near([l.tr, l.te], [d.trs[0], d.tes[0]], 'the line is degree 1');
      for (let k = 1; k < d.trs.length; k++) near(Math.min(d.trs[k], d.trs[k - 1]), d.trs[k], 'the training error never rises with the degree', 1e-9);
      // past degree 10 the plain least squares of the reference loses accuracy (x¹² reaches 5·10⁵); compare up to 10
      if (ref) { near(d.trs.slice(0, 10), ref[0].tr.slice(0, 10), 'training errors against scikit-learn', 1e-5); near(d.tes.slice(0, 8), ref[0].te.slice(0, 8), 'test errors against scikit-learn', 1e-4); }
    },
  },
  'data-interactions': {
    check({ frame, near, ref }) {
      const p = frame('plain').meta, e = frame('ext').meta;
      if (ref) { near(p.acc, ref[0].acc, 'plain accuracy against scikit-learn', 0.011); near(e.acc, ref[1].acc, 'accuracy with the new columns against scikit-learn', 0.011); near(e.coef, ref[1].coef, 'weights against scikit-learn', 2e-3); }
    },
  },
  'data-select': {
    constants: [5, 8, 4],
    random: (rnd) => ({ data: pick(rnd, ['rents', 'cancer']), noise: int(rnd, 0, 8), k: int(rnd, 1, 5) }),
    nRandom: 3,
    check({ frame, frames, near, ok, ref }) {
      const f = frame('filter').meta, fw = frames('forward'), em = frame('embedded').meta;
      for (let s = 1; s < fw.length; s++) ok(fw[s].meta.score >= fw[s - 1].meta.score - 0.02, 'forward selection does not fall much');
      ok(f.pick.every((j, i) => !i || f.filt[j] <= f.filt[f.pick[i - 1]] + 1e-12), 'the filter keeps the highest scores');
      if (ref) { near(f.filt, ref[0].filt, '|r| against numpy', 1e-9); if (ref[0].lasso) near(em.emb, ref[0].lasso.map(Math.abs), 'lasso weights against scikit-learn', 1e-4); }
    },
  },
  'data-leakage': {
    constants: [10, 5, 100, 3, 400],
    check({ frame, near, ok }) {
      const l = frame('leak').meta, h = frame('honest').meta, g = frame('gap').meta;
      near(l.mean, mean(l.folds), 'leaky mean'); near(h.mean, mean(h.folds), 'honest mean'); near(g.gap, l.mean - h.mean, 'the gap');
      ok(g.gap >= -1e-9, 'the leak never scores below the honest pipeline here');
    },
  },
};

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) runClaims('claims-p01', /^p01-/, P01);
