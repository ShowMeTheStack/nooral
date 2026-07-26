// The Data blocks: a built-in dataset, the student's own CSV file and a folder of their own images, and the train/test
// split. Files are read on the page (readFile) into the block's parameters, so the workflow, its worker and a saved
// copy all hold the same rows; nothing leaves the browser.
(function (S) {
  'use strict';
  const WF = S.WF, U = () => WF.dutil;
  const MAX_CSV = 2e6;

  /** what a Data block's viewer shows of a dataset: its story, the columns, the class balance, some rows */
  WF.dataView = function (v, extra) {
    const u = U();
    const o = Object.assign({ kind: v.kind, name: v.name, n: v.y.length, task: v.task, classes: v.classes, counts: WF.counts(v), target: v.target ? { name: v.target.name, type: v.target.type } : null }, extra || {});
    if (v.task === 'regression') o.targetStats = u.stats(v.y.map((y) => [y]), 0, { name: v.target.name, type: 'number' });
    if (v.kind === 'table') {
      o.features = v.features.map((f) => ({ name: f.name, type: f.type, unit: f.unit || '', categories: f.categories ? f.categories.slice(0, 12) : null }));
      o.columns = u.columns(v, 12);
      const k = u.sample(v.y.length, 12);
      o.rows = k.map((i) => v.X[i].map((x) => u.r(x, 4)));
      o.rowY = k.map((i) => v.y[i]);
      o.rowIds = k.map((i) => v.rows[i]);
      o.missing = v.X.reduce((a, r) => a + r.filter((x) => !u.isNum(x)).length, 0);
      if (v.features.length >= 1 && v.features.length <= 40 && v.y.length) {
        const keep = u.sample(v.y.length, 900), pl = u.plane(keep.map((i) => v.X[i]).filter((r) => r.every(u.isNum)).length === keep.length ? keep.map((i) => v.X[i]) : keep.map((i) => v.X[i].map((x) => (u.isNum(x) ? x : 0))), v.features.map((f) => f.name));
        o.points = { P: u.round2(pl.P), y: keep.map((i) => v.y[i]), names: pl.names, how: pl.how, n: v.y.length };
      }
    } else if (v.kind === 'images') {
      const k = u.sample(v.y.length, 40);
      o.w = v.w; o.h = v.h;
      o.images = k.map((i) => Array.from(v.X[i], (x) => Math.round(x * 15)));
      o.imageY = k.map((i) => v.y[i]);
    } else if (v.kind === 'text') {
      const k = u.sample(v.y.length, 14);
      o.texts = k.map((i) => v.texts[i]);
      o.textY = k.map((i) => v.y[i]);
      const words = v.texts.map((t) => String(t).split(/\s+/).filter(Boolean).length);
      o.words = { mean: u.r(u.mean(words), 3), min: Math.min(...words), max: Math.max(...words) };
    }
    return o;
  };

  // the built-in sets, with a richer view than the base one
  const base = WF.def('data');
  WF.block('data', Object.assign({}, base, {
    view(r) {
      const v = r.out.out, D = S.Data, doc = D && D.doc(v.source.set);
      return WF.dataView(v, { doc: doc ? { title: doc.title, story: doc.story, rule: doc.rule, licence: doc.licence, source: doc.source, attribution: doc.attribution || '' } : null });
    },
  }));

  // ---- a CSV file of the student's own --------------------------------------------------------------------------
  const parseCached = (() => { let key = null, val = null; return (text) => { if (text === key) return val; key = text; val = S.Data.parseCSV(text); return val; }; })();
  WF.block('csv', {
    cat: 'Data', label: 'CSV file', icon: 'upload', order: 20,
    blurb: 'A table of your own from a CSV file. Columns are typed from their values (numbers or categories); blank, NA and ? are missing cells. The file stays in this browser.',
    outputs: [{ id: 'out', label: 'data', type: 'table' }],
    params: [
      { id: 'file', label: 'File', kind: 'file', def: '', accept: '.csv,.tsv,.txt,text/csv', help: 'Choose a CSV file; its first row may name the columns', describe: (v) => (v ? String(v) : 'no file yet') },
      { id: 'name', label: 'Name', kind: 'text', def: '' },
      { id: 'target', label: 'Column to predict', kind: 'choice', def: '', options: (p) => { try { return [['', 'the last column']].concat(parseCached(p.csv).columns.map((c) => [c.name, c.name + ' (' + c.type + ')'])); } catch (e) { return [['', 'the last column']]; } } },
      { id: 'asCategory', label: 'Treat the target as classes', kind: 'choice', def: 'auto', options: [['auto', 'as read'], ['yes', 'yes: classes'], ['no', 'no: a number']] },
      { id: 'csv', label: 'Contents', kind: 'hidden', def: '', show: () => false },
    ],
    /** on the page: read a chosen file into the block's parameters */
    async readFile(file) {
      if (file.size > MAX_CSV) throw new Error('This file is ' + Math.round(file.size / 1e6) + ' MB; Nooral keeps files to 2 MB so training stays quick. Use fewer rows.');
      const text = await file.text();
      const p = S.Data.parseCSV(text);
      if (!p.columns.length || p.n < 2) throw new Error('No table could be read from this file.');
      return { csv: text, name: file.name.replace(/\.[^.]*$/, ''), file: file.name, target: '' };
    },
    run(ctx) {
      const p = ctx.params;
      if (!p.csv) throw new Error('choose a CSV file: double-click the block');
      const parsed = parseCached(p.csv);
      const tName = p.target || parsed.columns[parsed.columns.length - 1].name;
      const col = parsed.columns.find((c) => c.name === tName);
      if (!col) throw new Error('the file has no column called ' + tName);
      if (p.asCategory === 'yes' && col.type === 'number') { col.type = 'category'; col.categories = [...new Set(col.values.filter((x) => x !== null))].sort((a, b) => a - b).map(String); col.values = col.values.map((x) => (x === null ? null : String(x))); }
      if (p.asCategory === 'no' && col.type === 'category') throw new Error('the column ' + tName + ' holds words, not numbers: it can only be predicted as classes');
      const ds = S.Data.csvToDataset(parsed, { target: tName });
      // rows whose target is missing cannot be learned from
      const keep = ds.y.map((y, i) => (Number.isFinite(y) && y >= (col.type === 'number' ? -Infinity : 0) ? i : -1)).filter((i) => i >= 0);
      const v = WF.fromDataset(Object.assign({}, ds, { name: p.name || p.file || 'my table', X: keep.map((i) => ds.X[i]), y: keep.map((i) => ds.y[i]) }), { set: 'csv', file: p.file });
      return { out: { out: v }, dropped: ds.dropped || [], noTarget: ds.y.length - keep.length, warnings: parsed.warnings.slice(0, 5), trace: parsed.trace.slice(0, 16).map((s) => s.say) };
    },
    view(r) { return WF.dataView(r.out.out, { file: true, dropped: r.dropped, noTarget: r.noTarget, warnings: r.warnings, trace: r.trace }); },
    summary: (r) => WF.describe(r.out.out),
  });

  // ---- a folder of images ------------------------------------------------------------------------------------------
  WF.block('imagefolder', {
    cat: 'Data', label: 'Image folder', icon: 'image', order: 30,
    blurb: 'Pictures of your own, one subfolder per class (cats/…, dogs/…). Each is made grey and shrunk to a small square. They stay in this browser.',
    outputs: [{ id: 'out', label: 'images', type: 'images' }],
    params: [
      { id: 'file', label: 'Pictures', kind: 'file', def: '', multiple: true, folder: true, accept: 'image/*', help: 'Choose all the pictures at once. The class of each is its folder’s name, or the start of its file name (cat_01.png is a cat).', describe: (v) => (v ? String(v) : 'no pictures yet') },
      { id: 'size', label: 'Size', kind: 'choice', def: 16, options: [[16, '16 × 16'], [28, '28 × 28']] },
      { id: 'name', label: 'Name', kind: 'text', def: '' },
      { id: 'pixels', label: 'Pictures', kind: 'hidden', def: '', show: () => false },
      { id: 'labels', label: 'Labels', kind: 'hidden', def: '', show: () => false },
      { id: 'classes', label: 'Classes', kind: 'hidden', def: '', show: () => false },
    ],
    /** on the page: files (a FileList from a folder picker or a drop) -> the packed pictures */
    async readFile(files, p) {
      const size = +(p && p.size) || 16;
      const set = await S.Data.ingestImages(files.length !== undefined ? files : [files], { size, maxFiles: 3000 });
      if (!set.n) throw new Error('No pictures were found in that folder.');
      if (set.classes.length < 2) throw new Error('All the pictures are in one class (' + set.classes[0] + '): put each class in its own subfolder.');
      const px = new Float32Array(set.n * size * size);
      for (let i = 0; i < set.n; i++) px.set(set.get(i), i * size * size);
      const list = Array.from(files.length !== undefined ? files : [files]), first = list[0], folder = first && first.webkitRelativePath ? first.webkitRelativePath.split('/')[0] : 'my images';
      return { pixels: S.Data.pack4(px), labels: set.y.join(','), classes: set.classes.join('|'), size, file: folder + ' (' + set.n + ' pictures)', name: folder };
    },
    run(ctx) {
      const p = ctx.params;
      if (!p.pixels) throw new Error('choose a folder of pictures: double-click the block');
      const y = String(p.labels).split(',').map(Number), size = +p.size, n = y.length;
      const all = S.Data.unpack4(p.pixels, n * size * size), images = [];
      for (let i = 0; i < n; i++) images.push(all.slice(i * size * size, (i + 1) * size * size));
      const classes = String(p.classes).split('|');
      const v = WF.fromDataset({ name: p.name || 'my images', images, X: images, w: size, h: size, y, classes, target: { name: 'class', type: 'category', categories: classes }, doc: { kind: 'image' } }, { set: 'imagefolder' });
      return { out: { out: v } };
    },
    view(r) { return WF.dataView(r.out.out, { file: true }); },
    summary: (r) => WF.describe(r.out.out),
  });

  // ---- the split, marking which side each table is (the Prepare blocks speak of leakage from it) -------------------
  const split = WF.def('split');
  WF.block('split', Object.assign({}, split, {
    run(ctx) {
      const r = split.run(ctx);
      r.out.train.part = 'train'; r.out.test.part = 'test';
      return r;
    },
    view(r, ctx) {
      const v = split.view(r, ctx), tr = r.out.train, te = r.out.test, u = U();
      if (tr.kind === 'table' && tr.features.length >= 1 && tr.features.length <= 40) {
        const all = ctx.in.in, keep = u.sample(all.y.length, 900), inTest = new Set(r.split.test);
        const ok = (row) => row.map((x) => (u.isNum(x) ? x : 0));
        const pl = u.plane(keep.map((i) => ok(all.X[i])), all.features.map((f) => f.name));
        v.points = { P: u.round2(pl.P), y: keep.map((i) => all.y[i]), test: keep.map((i) => (inTest.has(i) ? 1 : 0)), names: pl.names, how: pl.how };
      }
      v.share = ctx.params.test; v.stratified = !!ctx.params.stratify && tr.task === 'classification';
      return v;
    },
  }));
})(globalThis.Nooral = globalThis.Nooral || {});
