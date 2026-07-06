// Datasets: generators are deterministic with the documented sizes and balances, classic sets decode to the
// published values and carry their licences, and the CSV and image upload helpers read what they are given.
import { readFileSync } from 'node:fs';
import { globSync } from 'node:fs';
import { load, harness } from './load.mjs';

const S = load(['core']), D = S.Data;
const t = harness('data');
const sig = (ds) => JSON.stringify([ds.X && ds.X.slice(0, 50).map((r) => Array.from(r)), ds.y, ds.edges, ds.rows, ds.texts]);
const counts = (y) => D.balance(y).map((p) => p[1]);

// every registered set is documented and deterministic
const docs = D.list();
t.ok(docs.length >= 25, 'registered sets: ' + docs.length);
for (const d of docs) {
  for (const k of ['title', 'kind', 'story', 'rule', 'source', 'licence', 'size', 'features', 'target', 'balance']) t.ok(d[k] != null && d[k] !== '', d.id + ' documents ' + k);
  const small = d.kind === 'image' ? { n: 40, size: 16 } : {};
  const a = D.load(d.id, Object.assign({ seed: 5 }, small)), b = D.load(d.id, Object.assign({ seed: 5 }, small));
  t.ok(sig(a) === sig(b), d.id + ' is deterministic');
  if (['toy', 'tabular', 'image', 'graph'].includes(d.kind)) t.ok(sig(a) !== sig(D.load(d.id, Object.assign({ seed: 6 }, small))), d.id + ' changes with the seed');
  if (a.X && a.y) t.ok(a.X.length === a.y.length, d.id + ' rows and labels match');
}

// documented default sizes and balances
const sizeOf = (id) => +String(D.doc(id).size).match(/\d+/)[0];
for (const id of ['moons', 'circles', 'blobs', 'spirals', 'xor', 'checkerboard', 'linear', 'curve', 'harvest', 'buses', 'wildflowers', 'loans', 'pumps', 'rents', 'exam']) t.eq(D.load(id).X.length, sizeOf(id), id + ' default size');
for (const [id, k] of [['moons', 2], ['circles', 2], ['blobs', 3], ['spirals', 2], ['checkerboard', 2], ['linear', 2], ['wildflowers', 3]]) {
  for (const n of [7, 100, 301]) { const c = counts(D.load(id, { n, noise: 0 }).y); t.eq(c, D.even(n, k), id + ' balance n=' + n); }
}
t.eq(counts(D.load('xor', { n: 200 }).y), [100, 100], 'xor balance');
t.eq(counts(D.load('pumps').y), [950, 50], 'pumps: 5% faults');
{ const L = D.load('loans'); t.eq(counts(L.X.map((x) => x[5])), [360, 240], 'loans: groups 60/40'); t.ok(L.sensitive === 5, 'loans marks its sensitive column'); }
t.ok(D.load('pumps').X.flat().filter(Number.isNaN).length > 50, 'pumps has missing cells');
t.ok(D.load('moons', { noise: 0 }).X.every(([x, y]) => Math.abs(Math.hypot(x, y) - 1) < 1e-3 || Math.abs(Math.hypot(x - 1, y - 0.5) - 1) < 1e-3), 'moons lie on their half circles');
t.ok(D.load('checkerboard').X.every((p, i) => (Math.floor(p[0]) + Math.floor(p[1])) % 2 === D.load('checkerboard').y[i] || Math.abs(p[0] - Math.round(p[0])) < 1e-3 || Math.abs(p[1] - Math.round(p[1])) < 1e-3), 'checkerboard rule');
for (const sh of D.curveShapes) t.eq(D.load('curve', { shape: sh, n: 30 }).X.length, 30, 'curve ' + sh);

// images
{
  const a = D.load('digits', { n: 100, size: 28, seed: 9 }), b = D.load('digits', { n: 100, size: 28, seed: 9 });
  t.ok(a.images.length === 100 * 784 && a.images.every((v, i) => v === b.images[i]), 'digits identical per seed');
  t.eq(counts(a.y), new Array(10).fill(10), 'digits balanced');
  t.ok(a.images.every((v) => v >= 0 && v <= 1), 'digit pixels in [0, 1]');
  for (let d = 0; d < 10; d++) { const img = D.drawDigit(d, { size: 16, seed: d }), ink = img.reduce((s, v) => s + v, 0); t.ok(ink > 12 && ink < 120, 'digit ' + d + ' has ink ' + ink.toFixed(1)); }
  t.ok(D.drawDigit(3, { seed: 1 }).some((v, i) => v !== D.drawDigit(3, { seed: 2 })[i]), 'different seeds jitter differently');
  const sh = D.load('shapes', { n: 40 }), ic = D.load('icons', { n: 60, size: 16 });
  t.eq(counts(sh.y), [10, 10, 10, 10], 'shapes balanced'); t.eq(counts(ic.y), new Array(6).fill(10), 'icons balanced');
  const big = D.drawShape('square', { seed: 1 }).reduce((s, v) => s + v, 0), outline = D.drawShape('square', { seed: 1, outline: true }).reduce((s, v) => s + v, 0);
  t.ok(outline < big, 'outlined shape has less ink than filled');
  // compact storage
  const px = a.get(3), p4 = D.unpack4(D.pack4(px), px.length), pb = D.unpackBits(D.packBits(px), px.length);
  t.ok(px.every((v, i) => Math.abs(v - p4[i]) <= 1 / 30 + 1e-6), '4-bit grey round trip');
  t.ok(px.every((v, i) => pb[i] === (v >= 0.5 ? 1 : 0)), 'bit packing round trip');
  t.eq(D.pack4(px).length, Math.ceil(392 / 3) * 4, '4-bit storage is 392 bytes as base64');
  for (const len of [0, 1, 2, 3, 4, 5, 31]) { const bytes = Uint8Array.from({ length: len }, (_, i) => (i * 37 + 11) & 255); t.eq(D.toBase64(bytes), Buffer.from(bytes).toString('base64'), 'base64 ' + len); t.eq(Array.from(D.fromBase64(D.toBase64(bytes))), Array.from(bytes), 'base64 back ' + len); }
}

// classic sets: sizes, class counts, values and licences
{
  const want = { iris: [150, 4, [50, 50, 50]], wine: [178, 13, [59, 71, 48]], breastCancer: [569, 30, [212, 357]], penguins: [344, 7, [152, 68, 124]] };
  for (const [id, [n, d, c]] of Object.entries(want)) {
    const ds = D.load(id);
    t.eq([ds.X.length, ds.X[0].length], [n, d], id + ' shape'); t.eq(counts(ds.y), c, id + ' class counts');
    const lic = D.licences[id], doc = D.doc(id);
    t.ok(lic && /CC/.test(lic.licence) && lic.attribution.length > 60 && /^https:/.test(lic.source), id + ' licence recorded');
    t.ok(doc.licence === lic.licence && doc.attribution === lic.attribution, id + ' doc carries licence');
    t.ok(D.notice().includes(lic.attribution), id + ' in NOTICE text');
  }
  t.ok(D.list('classic').length === 4 && D.list('classic').every((d) => D.licences[d.id]), 'every classic set has a licence entry');
  t.eq(D.load('iris').X[0], [5.1, 3.5, 1.4, 0.2], 'iris first row');
  t.eq(D.load('breastCancer').X[0].slice(0, 5), [17.99, 10.38, 122.8, 1001, 0.1184], 'breast cancer first row');
  t.eq(D.load('wine').X[177][12], 560, 'wine last proline');
  const P = D.load('penguins');
  t.eq(P.X[0], [2, 39.1, 18.7, 181, 3750, 1, 2007], 'penguins first row');
  t.eq(P.X.filter((r) => Number.isNaN(r[1])).length, 2, 'penguins missing bill lengths');
  t.eq(P.X.filter((r) => Number.isNaN(r[5])).length, 11, 'penguins missing sex');
  // the sums of every column agree with the bundled sources
  const sk = globSync(process.env.HOME + '/.nooral-venv/lib/python3*/site-packages/sklearn/datasets/data')[0];
  if (sk) for (const [id, f] of [['iris', 'iris.csv'], ['wine', 'wine_data.csv'], ['breastCancer', 'breast_cancer.csv']]) {
    const rows = readFileSync(sk + '/' + f, 'utf8').trim().split('\n').slice(1).map((l) => l.split(',').map(Number)), ds = D.load(id);
    t.ok(rows.every((r, i) => r.slice(0, -1).every((v, j) => v === ds.X[i][j]) && r[r.length - 1] === ds.y[i]), id + ' equals the reference copy exactly');
  }
  t.ok(/licen/i.test(D.excluded.handwrittenDigits), 'excluded set has a reason');
}

// text, graphs and grid worlds
{
  t.eq(D.load('spam').n, 90, 'spam messages'); t.eq(counts(D.load('spam').y), [50, 40], 'spam balance');
  t.eq(counts(D.load('reviews').y), [30, 30], 'reviews balance'); t.eq(counts(D.load('topics').y), [10, 10, 10, 10], 'topics balance');
  const words = D.proseText().split(/\s+/).length; t.ok(words > 3000, 'story corpus has ' + words + ' words');
  const g = D.load('communities'); t.eq(counts(g.y), [10, 10, 10], 'communities balanced');
  const nb = D.neighbours(g); t.ok(nb.every((l) => l.length > 0), 'no isolated nodes');
  const inside = g.edges.filter(([a, b]) => g.y[a] === g.y[b]).length; t.ok(inside > 0.7 * g.edges.length, 'most edges inside communities');
  const c = D.load('citations'); t.ok(c.edges.every(([a, b]) => b < a), 'papers cite older papers'); t.eq(counts(c.y), [20, 20, 20], 'citation fields balanced');
  for (const name of D.gridNames) { const w = D.gridWorld({ name, seed: 3 }); t.ok(w.start && w.goals.length && D.gridDistance(w) > 0, name + ' goal reachable'); }
  t.ok(D.gridWorld({ name: 'maze', seed: 4, w: 15, h: 11 }).rows.length === 11, 'maze size');
}

// CSV reading
{
  const p = D.parseCSV('name,age,score,passed,city\n"Lee, Ana",21,88.5,yes,Leeds\nBo,NA,72,no,York\nCy,19,,yes,Leeds\n"Di ""DJ""",22,91,no,York\n');
  t.eq(p.columns.map((c) => c.type), ['text', 'number', 'number', 'boolean', 'category'], 'csv types');
  t.eq(p.columns[0].values, ['Lee, Ana', 'Bo', 'Cy', 'Di "DJ"'], 'csv quotes');
  t.eq(p.columns.map((c) => c.missing), [0, 1, 1, 0, 0], 'csv missing counts');
  t.ok(p.header && p.columns[1].integer, 'csv header and integers');
  t.ok(p.trace.some((s) => /2 cells are missing/.test(s.say)), 'csv trace counts missing cells');
  const ds = D.csvToDataset(p, { target: 'passed' });
  t.eq(ds.dropped, ['name'], 'text column dropped'); t.eq(ds.y, [1, 0, 1, 0], 'boolean target');
  t.ok(Number.isNaN(ds.X[1][0]) && ds.X[0][2] === 0, 'missing as NaN, categories as index');
  const q = D.parseCSV('a;b;c\n1,5;2;x\n3,25;4;y\n');
  t.ok(q.delimiter === ';' && q.decimalComma && q.columns[0].values[1] === 3.25, 'semicolons with decimal commas');
  const r = D.parseCSV('1\t2\t3\r\n4\t5\t6\r\n');
  t.ok(r.delimiter === '\t' && !r.header && r.n === 2 && r.columns[2].values[1] === 6, 'tabs, no header, CRLF');
  t.ok(D.parseCSV('x,y\n1,2\n3\n').warnings.length === 1, 'short row warned');
  t.eq(D.inferType(['2024-01-05', '2023-12-31']).type, 'date', 'dates');
}

// image uploads (the part that runs without a browser)
{
  const rgba = new Uint8ClampedArray(4 * 4 * 4).fill(255);
  for (const [x, y] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const i = 4 * (y * 4 + x); rgba[i] = rgba[i + 1] = rgba[i + 2] = 0; }
  t.eq(Array.from(D.resizeGrey(rgba, 4, 4, 2)), [1, 0, 0, 0], 'resize: dark corner becomes ink');
  t.eq(Array.from(D.resizeGrey(rgba, 4, 4, 2, { invert: false })), [0, 1, 1, 1], 'resize without inverting');
  const wide = new Uint8ClampedArray(4 * 4 * 2).map((_, i) => (i % 4 === 3 ? 255 : 0)); // a black 4×2 picture, fitted into a square with white bands
  t.eq(Array.from(D.resizeGrey(wide, 4, 2, 2, { invert: false })), [0.5, 0.5, 0.5, 0.5], 'contain fit pads with white');
  t.eq([D.labelOf('pets/cat/img1.png'), D.labelOf('dog_03.jpg'), D.labelOf('7.png')], ['cat', 'dog', 'unlabelled'], 'labels from paths');
  const set = D.imagesFromPixels([{ path: 'a/x.png', rgba, w: 4, h: 4 }, { path: 'b/y.png', rgba, w: 4, h: 4 }], { size: 2 });
  t.eq([set.classes, set.y, set.n], [['a', 'b'], [0, 1], 2], 'image folder set');
}

// bundle size of the data files
{
  // the long text corpus (data-text-more-*.js) has its own budget: it is plain prose the tiny GPT needs
  const all = globSync(new URL('../src/core/data-*.js', import.meta.url).pathname), size = (fs) => fs.reduce((s, f) => s + readFileSync(f).length, 0);
  const corpus = all.filter((f) => /data-text-more-/.test(f)), files = all.filter((f) => !corpus.includes(f)), bytes = size(files), cb = size(corpus);
  console.log('data files: ' + files.length + ', ' + (bytes / 1024).toFixed(1) + ' KiB; text corpus: ' + corpus.length + ' files, ' + (cb / 1024).toFixed(1) + ' KiB');
  t.ok(bytes < 260 * 1024, 'data bundle under 260 KiB');
  t.ok(cb < 200 * 1024, 'text corpus under 200 KiB');
  const long = D.corpusText(), short = D.proseText();
  t.ok(long.length >= 120000 && long.includes(short.slice(0, 200)), 'the long corpus has ' + long.length + ' characters and contains the short one');
  t.ok(/^[A-Za-z \n.,'"!?:-]+$/.test(long), 'the corpus uses only letters, spaces, newlines and plain punctuation');
  const titles = D.corpusPieces().map((p) => p.title);
  t.eq(titles.filter((x, i) => titles.indexOf(x) !== i), [], 'no two pieces share a title');
  const tenth = D.corpusPieces().slice(-Math.ceil(titles.length / 10)), dlg = tenth.filter((p) => /^[A-Z][A-Za-z ]*: /.test(p.text)).length;
  t.ok(dlg > 0 && dlg < tenth.length, 'the last tenth (the GPT validation text) mixes stories and dialogues (' + dlg + ' of ' + tenth.length + ' are dialogues)');
  const op = D.load('opinions', { seed: 1 }), op2 = D.load('opinions', { seed: 1 });
  t.eq(counts(op.y), [300, 300], 'opinions balance');
  t.ok(new Set(op.texts).size === 600 && JSON.stringify(op.texts) === JSON.stringify(op2.texts), 'opinions: 600 different reviews, the same for a seed');
  const turn = op.texts.filter((x, i) => op.y[i] === 1 && /\b(not|but|although|nothing)\b/i.test(x)).length;
  t.ok(turn >= 60, 'opinions: ' + turn + ' positive reviews use not, but, although or nothing');
}
t.done();
