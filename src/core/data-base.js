// Datasets, shared pieces: the seeded random generator, the registry with each set's documentation, the table
// form every set is returned in, and compact storage for images (packed bits or 4-bit grey, in base64).
//
// A dataset is { id, doc, X, y, features, target, classes?, ... } where X is a list of rows of numbers (categorical
// columns hold their category index, with the names in features[j].categories), y the target per row, and doc
// { title, kind, story, source, rule, licence, size, features, target, balance }. Image sets add
// { images, w, h, get(i) }; text sets { texts } and graphs { nodes, edges }.
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});

  /** a seeded generator of numbers in [0, 1) (mulberry32): the same seed always gives the same sequence */
  D.rng = (seed = 1) => {
    let s = seed >>> 0;
    const r = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    r.normal = (mu = 0, sd = 1) => {
      if (spare !== null) { const v = spare; spare = null; return mu + sd * v; }
      let u = 0; while (u === 0) u = r();
      const v = r(), m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v); return mu + sd * m * Math.cos(2 * Math.PI * v);
    };
    r.int = (n) => Math.floor(r() * n);
    r.range = (a, b) => a + (b - a) * r();
    r.pick = (a) => a[Math.floor(r() * a.length)];
    r.shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [a[i], a[k]] = [a[k], a[i]]; } return a; };
    return r;
  };
  D.round = (x, d = 3) => { const p = 10 ** d; return Math.round(x * p) / p; };

  // the registry: id → { doc, make(opt) }
  const REG = new Map();
  /** register a dataset: doc is its documentation, make(opt) builds it (opt.seed, opt.n and set-specific options) */
  D.register = (id, doc, make) => { REG.set(id, { doc: Object.assign({ id }, doc), make }); };
  /** the documentation of every registered set, optionally of one kind (toy, tabular, image, text, graph, grid, classic) */
  D.list = (kind) => [...REG.values()].map((r) => r.doc).filter((d) => !kind || d.kind === kind);
  D.doc = (id) => (REG.get(id) || {}).doc;
  /** build a dataset by id: D.load('moons', {n: 200, noise: 0.1, seed: 3}) */
  D.load = (id, opt = {}) => {
    const r = REG.get(id);
    if (!r) throw new Error('no dataset called ' + id);
    const ds = r.make(Object.assign({ seed: 1 }, opt));
    return Object.assign({ id, doc: r.doc }, ds);
  };

  /** a table dataset from columns of numbers: features [{name, type: 'number'|'category', unit, desc, categories}] */
  D.table = (X, y, features, target, extra) => Object.assign({ X, y, features, target, n: X.length }, extra || {});

  /** class counts of a label list, in label order */
  D.balance = (y) => { const m = new Map(); for (const v of y) m.set(v, (m.get(v) || 0) + 1); return [...m.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)); };
  /** n split as evenly as possible into k parts (the first parts take the remainder) */
  D.even = (n, k) => Array.from({ length: k }, (_, i) => Math.floor(n / k) + (i < n % k ? 1 : 0));

  /** a stratified or plain shuffle split: { train, test } index lists */
  D.split = (y, testFrac = 0.25, seed = 1, stratify = true) => {
    const r = D.rng(seed), train = [], test = [];
    const groups = stratify ? D.balance(y).map(([c]) => y.map((v, i) => (v === c ? i : -1)).filter((i) => i >= 0)) : [y.map((_, i) => i)];
    for (const g of groups) { r.shuffle(g); const k = Math.round(g.length * testFrac); test.push(...g.slice(0, k)); train.push(...g.slice(k)); }
    return { train: train.sort((a, b) => a - b), test: test.sort((a, b) => a - b) };
  };

  // base64 without relying on btoa/atob, so the same code runs in the page and in node tests
  const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  D.toBase64 = (bytes) => {
    let s = '';
    for (let i = 0; i < bytes.length; i += 3) {
      const a = bytes[i], b = i + 1 < bytes.length ? bytes[i + 1] : 0, c = i + 2 < bytes.length ? bytes[i + 2] : 0, v = (a << 16) | (b << 8) | c;
      s += B64[(v >> 18) & 63] + B64[(v >> 12) & 63] + (i + 1 < bytes.length ? B64[(v >> 6) & 63] : '=') + (i + 2 < bytes.length ? B64[v & 63] : '=');
    }
    return s;
  };
  D.fromBase64 = (s) => {
    s = s.replace(/[^A-Za-z0-9+/]/g, '');
    const out = new Uint8Array(Math.floor((s.length * 3) / 4));
    let o = 0;
    for (let i = 0; i < s.length; i += 4) {
      const v = (B64.indexOf(s[i]) << 18) | (B64.indexOf(s[i + 1]) << 12) | ((i + 2 < s.length ? B64.indexOf(s[i + 2]) : 0) << 6) | (i + 3 < s.length ? B64.indexOf(s[i + 3]) : 0);
      out[o++] = (v >> 16) & 255; if (i + 2 < s.length) out[o++] = (v >> 8) & 255; if (i + 3 < s.length) out[o++] = v & 255;
    }
    return out.slice(0, o);
  };
  /** grey pixels in [0, 1] → 4-bit levels packed two per byte, as base64 (16 grey levels) */
  D.pack4 = (px) => { const b = new Uint8Array(Math.ceil(px.length / 2)); for (let i = 0; i < px.length; i++) { const q = Math.max(0, Math.min(15, Math.round(px[i] * 15))); b[i >> 1] |= i & 1 ? q : q << 4; } return D.toBase64(b); };
  D.unpack4 = (s, n) => { const b = D.fromBase64(s), out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = ((i & 1 ? b[i >> 1] & 15 : b[i >> 1] >> 4)) / 15; return out; };
  /** black-and-white pixels (≥ 0.5 is on) packed eight per byte, as base64 */
  D.packBits = (px) => { const b = new Uint8Array(Math.ceil(px.length / 8)); for (let i = 0; i < px.length; i++) if (px[i] >= 0.5) b[i >> 3] |= 128 >> (i & 7); return D.toBase64(b); };
  D.unpackBits = (s, n) => { const b = D.fromBase64(s), out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = (b[i >> 3] >> (7 - (i & 7))) & 1; return out; };

  /** decode a compact numeric table: rows separated by ';', values by ','; '' or '?' is a missing value (NaN) */
  D.parseRows = (s) => s.split(';').map((r) => r.split(',').map((v) => (v === '' || v === '?' ? NaN : +v)));
})(globalThis.Nooral = globalThis.Nooral || {});
