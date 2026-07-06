// Reading a student's own table: a CSV parser (quotes, delimiters, header detection) with type inference and
// missing-value detection per column, and the step that turns a parsed table into a dataset. Everything stays in
// the browser.
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});
  const MISSING = new Set(['', 'na', 'n/a', 'nan', 'null', 'none', '?', '-', '--', 'missing']);
  D.isMissing = (s) => s == null || MISSING.has(String(s).trim().toLowerCase());

  /** split text into rows of fields; quoted fields may hold the delimiter, newlines and doubled quotes */
  D.splitCSV = (text, delim) => {
    const rows = []; let row = [], field = '', q = false, quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c; continue; }
      if (c === '"' && field.trim() === '') { q = true; quoted = true; field = ''; }
      else if (c === delim) { row.push(quoted ? field : field.trim()); field = ''; quoted = false; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(quoted ? field : field.trim()); rows.push(row); row = []; field = ''; quoted = false; }
      else field += c;
    }
    if (field !== '' || row.length) { row.push(quoted ? field : field.trim()); rows.push(row); }
    return rows.filter((r) => !(r.length === 1 && r[0] === ''));
  };

  /** the delimiter whose field count is most consistent over the first lines (comma, semicolon, tab or bar) */
  D.sniffDelimiter = (text) => {
    const lines = text.split(/\r?\n/).filter((l) => l.trim()).slice(0, 20);
    let best = ',', score = -1;
    for (const d of [',', ';', '\t', '|']) {
      const counts = lines.map((l) => D.splitCSV(l, d)[0].length);
      if (counts[0] < 2) continue;
      const same = counts.filter((c) => c === counts[0]).length / counts.length, s = same * 100 + counts[0];
      if (s > score) { score = s; best = d; }
    }
    return best;
  };

  const NUM = /^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$/, DATE = /^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$/, BOOL = /^(true|false|yes|no)$/i;
  const toNum = (s, decComma) => +(decComma ? s.replace(',', '.') : s);

  /** the type of one column of strings: number, boolean, date, category, text or empty */
  D.inferType = (vals, decComma) => {
    const v = vals.filter((s) => !D.isMissing(s));
    if (!v.length) return { type: 'empty' };
    if (v.every((s) => NUM.test(s) && (decComma || !s.includes(',')))) { const xs = v.map((s) => toNum(s, decComma)); return { type: 'number', integer: xs.every(Number.isInteger) }; }
    if (v.every((s) => BOOL.test(s))) return { type: 'boolean' };
    if (v.every((s) => DATE.test(s) && !Number.isNaN(Date.parse(s)))) return { type: 'date' };
    const distinct = new Set(v).size;
    return distinct <= Math.max(2, Math.min(30, Math.ceil(v.length * 0.5))) ? { type: 'category' } : { type: 'text' };
  };

  /**
   * parseCSV(text, {delimiter, header}) → { columns, n, delimiter, header, warnings, trace }
   * each column: { name, type, missing, values (numbers, booleans, strings or null when missing), categories?, min?, max?, mean? }
   */
  D.parseCSV = (text, opt = {}) => {
    text = String(text).replace(/^﻿/, '');
    const delimiter = opt.delimiter || D.sniffDelimiter(text), rows = D.splitCSV(text, delimiter), warnings = [];
    if (!rows.length) return { columns: [], n: 0, delimiter, header: false, warnings: ['the file is empty'], trace: [] };
    const width = Math.max(...rows.map((r) => r.length));
    const decComma = delimiter === ';' && rows.slice(1, 30).some((r) => r.some((s) => /^\d+,\d+$/.test(s)));
    let header = opt.header;
    if (header == null) {
      const first = rows[0], rest = rows.slice(1, 50);
      header = first.every((s) => !NUM.test(s) || s === '') && first.some((_, j) => rest.some((r) => NUM.test(r[j] || ''))) ? true : first.every((s) => !NUM.test(s)) && new Set(first).size === first.length && rows.length > 1;
    }
    const names = Array.from({ length: width }, (_, j) => (header && rows[0][j] ? rows[0][j] : 'column ' + (j + 1)));
    const body = header ? rows.slice(1) : rows;
    body.forEach((r, i) => { if (r.length !== width) warnings.push('row ' + (i + 1 + (header ? 1 : 0)) + ' has ' + r.length + ' fields, expected ' + width); });
    const seen = new Map(); names.forEach((nm, j) => { const k = seen.get(nm) || 0; if (k) names[j] = nm + ' (' + (k + 1) + ')'; seen.set(nm, k + 1); });
    const columns = names.map((name, j) => {
      const raw = body.map((r) => (r[j] === undefined ? '' : r[j])), t = D.inferType(raw, decComma);
      let missing = 0;
      const values = raw.map((s) => {
        if (D.isMissing(s)) { missing++; return null; }
        if (t.type === 'number') return toNum(s, decComma);
        if (t.type === 'boolean') return /^(true|yes)$/i.test(s);
        return s;
      });
      const col = Object.assign({ name, missing, values }, t);
      if (t.type === 'number') { const xs = values.filter((v) => v !== null); col.min = Math.min(...xs); col.max = Math.max(...xs); col.mean = xs.reduce((a, b) => a + b, 0) / xs.length; }
      if (t.type === 'category') col.categories = [...new Set(values.filter((v) => v !== null))].sort();
      return col;
    });
    const nMiss = columns.reduce((a, c) => a + c.missing, 0);
    const trace = [
      D.step('read', 'Read ' + body.length + ' rows and ' + width + ' columns separated by ' + ({ ',': 'commas', ';': 'semicolons', '\t': 'tabs', '|': 'bars' }[delimiter]) + (header ? ', with a header row.' : ', with no header row.'), { delimiter, header }),
      ...columns.map((c) => D.step('column', c.name + ': ' + c.type + (c.type === 'number' ? (c.integer ? ' (whole numbers)' : '') + ' from ' + c.min + ' to ' + c.max : c.type === 'category' ? ' with ' + c.categories.length + ' values (' + c.categories.slice(0, 5).join(', ') + (c.categories.length > 5 ? ', …' : '') + ')' : '') + (c.missing ? '; ' + c.missing + ' missing' : '') + '.', { column: c.name, type: c.type, missing: c.missing })),
      D.step('missing', nMiss ? nMiss + ' cells are missing (blank, NA, ?, null and similar); they are kept as gaps for the Clean block to drop or fill.' : 'No cells are missing.', { missing: nMiss }),
    ];
    return { columns, n: body.length, delimiter, header, decimalComma: decComma, warnings, trace };
  };
  D.step = (kind, say, data) => Object.assign({ kind, say }, data || {});

  /**
   * csvToDataset(parsed, {target}) → a dataset: number columns as they are, booleans as 0/1, categories as their
   * index, dates as days since 1970; text columns are left out (listed in .dropped). Missing cells become NaN.
   */
  D.csvToDataset = (p, opt = {}) => {
    const tName = opt.target == null ? p.columns[p.columns.length - 1].name : typeof opt.target === 'number' ? p.columns[opt.target].name : opt.target;
    const enc = (c) => c.values.map((v) => (v === null ? NaN : c.type === 'number' ? v : c.type === 'boolean' ? (v ? 1 : 0) : c.type === 'date' ? Math.floor(Date.parse(v) / 864e5) : c.type === 'category' ? c.categories.indexOf(v) : NaN));
    const tcol = p.columns.find((c) => c.name === tName);
    if (!tcol) throw new Error('no column called ' + tName);
    const used = p.columns.filter((c) => c !== tcol && ['number', 'boolean', 'date', 'category'].includes(c.type));
    const cols = used.map(enc), X = Array.from({ length: p.n }, (_, i) => cols.map((c) => c[i]));
    const y = tcol.type === 'number' ? tcol.values.map((v) => (v === null ? NaN : v)) : enc(tcol);
    const features = used.map((c) => (c.type === 'category' ? { name: c.name, type: 'category', categories: c.categories } : { name: c.name, type: c.type === 'boolean' ? 'category' : 'number', categories: c.type === 'boolean' ? ['false', 'true'] : undefined }));
    const target = tcol.type === 'number' ? { name: tName, type: 'number' } : { name: tName, type: 'category', categories: tcol.type === 'boolean' ? ['false', 'true'] : tcol.categories };
    return D.table(X, y, features, target, { classes: target.categories, dropped: p.columns.filter((c) => c !== tcol && !used.includes(c)).map((c) => c.name), uploaded: true });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
