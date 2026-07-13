// Metrics, shared pieces: label sets, safe division, number formatting and the trace step.
// Every metric in S.Metrics returns an object { value, ..., trace } where trace is a list of steps
// {kind, say, ...data}: say is one plain sentence with the actual counts, the rest is what a picture needs.
// Conventions follow the usual library definitions, so results can be checked against reference outputs:
// a division by zero gives 0 (and the trace says so), labels are the sorted union of true and predicted labels.
(function (S) {
  'use strict';
  const M = (S.Metrics = S.Metrics || {});
  M.step = (kind, say, data) => Object.assign({ kind, say }, data || {});

  /** a number for a sentence: up to d decimals, trailing zeros removed */
  M.f = (x, d = 3) => {
    if (x === Infinity) return '∞';
    if (x === -Infinity) return '−∞';
    if (typeof x !== 'number') return String(x);
    if (Number.isNaN(x)) return 'undefined';
    if (Number.isInteger(x)) return String(x);
    const s = x.toFixed(d).replace(/\.?0+$/, '');
    return s === '-0' ? '0' : s;
  };
  /** a percentage for a sentence */
  M.pct = (x, d = 1) => M.f(100 * x, d) + '%';

  /** compare labels the way a sorted unique would: numbers numerically, otherwise as text */
  M.cmp = (a, b) => (typeof a === 'number' && typeof b === 'number' ? a - b : String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0);
  /** the sorted distinct labels found in any of the arrays */
  M.labels = (...arrs) => {
    const seen = new Map();
    for (const a of arrs) for (const v of a) if (!seen.has(keyOf(v))) seen.set(keyOf(v), v);
    return [...seen.values()].sort(M.cmp);
  };
  const keyOf = (v) => (typeof v) + ':' + String(v);
  /** a lookup from label to its index in the list */
  M.index = (labels) => { const m = new Map(); labels.forEach((l, i) => m.set(keyOf(l), i)); return (v) => { const i = m.get(keyOf(v)); return i === undefined ? -1 : i; }; };

  /** a / b, or z (default 0) when b is 0 */
  M.div = (a, b, z = 0) => (b === 0 ? z : a / b);
  M.sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
  M.mean = (a) => (a.length ? M.sum(a) / a.length : NaN);

  M.check = (a, b, what) => {
    if (a.length !== b.length) throw new Error(what + ': the two lists have different lengths (' + a.length + ' and ' + b.length + ')');
  };
  /** probability rows: a 1-D list of positive-class probabilities becomes [1 - p, p] rows */
  M.probRows = (P) => (P.length && typeof P[0] === 'number' ? P.map((p) => [1 - p, p]) : P.map((r) => Array.from(r)));
})(globalThis.Nooral = globalThis.Nooral || {});
