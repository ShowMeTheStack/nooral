// The data, prepare, split and unsupervised blocks on the canvas: each sample of theirs runs in the worker, every
// viewer opens and draws, k-means can be stepped back to its start, the dendrogram is cut with its slider and the cut
// is kept, the leakage note tells the two orders apart, a CSV file read on the page becomes a table, and the viewers
// fit a phone.
import { assert } from '../lib.mjs';

const P = 'Nooral.app.page';
const idle = (p) => p.ev(`${P}.whenIdle().then(() => 1)`);
const sample = async (p, id) => { await p.route('canvas'); await p.ev(`${P}.loadSample(${JSON.stringify(id)}).then(() => 1)`); await idle(p); };
const open = async (p, id) => { await p.ev(`(${P}.open(${JSON.stringify(id)}), 1)`); await p.wait("!!document.querySelector('.wf-viewer .wfd svg, .wf-viewer .wfd table')", 8000); await p.sleep(150); };
const states = (p) => p.ev(`JSON.stringify(Object.fromEntries(${P}.doc.blocks.map((b) => [b.id, (${P}.remote.states[b.id] || {}).state])))`).then(JSON.parse);

export default function (t) {
  t.test('every data sample runs to the end in the worker', async (p) => {
    for (const id of ['prepare-pumps', 'leakage-loans', 'clean-pumps', 'features-moons', 'cv-wildflowers', 'text-spam-bow', 'clusters-kmeans', 'clusters-moons', 'maps-wine']) {
      await sample(p, id);
      const st = await states(p);
      for (const [b, s] of Object.entries(st)) assert.equal(s, 'done', id + ': ' + b + ' is ' + s);
    }
  }, { timeout: 120000 });

  t.test('each viewer opens and draws', async (p) => {
    const plan = { 'prepare-pumps': ['data1', 'split1', 'impute1', 'scale1'], 'leakage-loans': ['encode1', 'select1'], 'clean-pumps': ['clean1', 'clean2'], 'features-moons': ['poly1'], 'cv-wildflowers': ['cv1'], 'text-spam-bow': ['data1', 'vectorise1'], 'clusters-kmeans': ['kmeans1', 'gmm1'], 'clusters-moons': ['dbscan1', 'hierarchical1'], 'maps-wine': ['pca1', 'tsne1', 'umap1'] };
    for (const [id, blocks] of Object.entries(plan)) {
      await sample(p, id);
      for (const b of blocks) {
        await open(p, b);
        assert.ok(!(await p.ev("[...document.querySelectorAll('.wf-viewer .wfd')].some((e) => /could not be drawn/.test(e.innerText))")), b + ' viewer draws');
      }
    }
  }, { timeout: 120000 });

  t.test('k-means replays its rounds: back to the start, forward to convergence', async (p) => {
    await sample(p, 'clusters-kmeans');
    await open(p, 'kmeans1');
    const pos = () => p.text('.wf-viewer .wfd-controls .pos');
    const [, total] = (await pos()).split(' / ').map(Number);
    assert.ok(total >= 4, 'several half-rounds');
    await p.click('.wf-viewer .wfd-controls button[aria-label="Back to the start"]');
    assert.equal(await pos(), '1 / ' + total);
    assert.match(await p.text('.wf-viewer .wfd-say'), /first centres/);
    await p.click('.wf-viewer .wfd-controls button[aria-label="Step forward"]');
    assert.match(await p.text('.wf-viewer .wfd-say'), /assign: every row joins its nearest centre/);
    await p.setValue('.wf-viewer .wfd-controls input[type=range]', String(total - 1));
    assert.match(await p.text('.wf-viewer .wfd-say'), /converged/);
  });

  t.test('the dendrogram is cut with the slider, and the cut can be kept', async (p) => {
    await sample(p, 'clusters-moons');
    await open(p, 'hierarchical1');
    const k0 = await p.text('.wf-viewer .wfd-facts');
    await p.setValue('.wf-viewer .wfd-row input[type=range]', '20');
    const k1 = await p.text('.wf-viewer .wfd-facts');
    assert.notEqual(k0, k1, 'a lower cut makes more clusters');
    assert.match(await p.text('.wf-viewer .wfd-facts'), /clusters \d+/);
    await p.click('.wf-viewer .wfd-row button', 'Use this cut');
    await p.wait(`${P}.doc.blocks.find((b) => b.id === 'hierarchical1').params.cut === 'height'`);
    await idle(p);
    assert.equal(await p.ev(`${P}.remote.states.hierarchical1.state`), 'done');
  });

  t.test('the leakage note tells the two orders apart', async (p) => {
    await sample(p, 'prepare-pumps');
    await open(p, 'scale1');
    assert.match(await p.text('.wf-viewer .wfd-note.good'), /training rows only/);
    await sample(p, 'leakage-loans');
    await open(p, 'scale1');
    assert.match(await p.text('.wf-viewer .wfd-note.bad'), /leak into training/);
  });

  t.test('a CSV file read on the page becomes a table that a model learns from', async (p) => {
    await p.route('canvas');
    const r = await p.ev(`(async () => {
      const WF = Nooral.WF, csv = 'size,weight,kind\\n' + Array.from({ length: 60 }, (_, i) => (1 + (i % 7)) + ',' + (i % 2 ? 10 + i % 5 : '') + ',' + (i % 3 ? 'big' : 'small')).join('\\n');
      const patch = await WF.def('csv').readFile(new File([csv], 'mine.csv', { type: 'text/csv' }));
      const doc = WF.build({ blocks: { c: ['csv', patch], i: ['impute', {}], m: ['logistic', {}], t: ['train', {}] }, wires: [['c.out', 'i.in'], ['i.out', 't.data'], ['m.model', 't.model']] });
      await ${P}.loadDoc(doc); await ${P}.whenIdle();
      return JSON.stringify({ c: ${P}.remote.states.c, t: ${P}.remote.states.t.state, name: patch.name });
    })()`);
    const o = JSON.parse(r);
    assert.equal(o.name, 'mine');
    assert.equal(o.c.state, 'done');
    assert.match(o.c.summary, /60 rows × 2 features · 2 classes/);
    assert.equal(o.t, 'done');
  });

  t.test('the viewers fit a phone', async (p) => {
    await p.viewport(390, 844, { mobile: true, touch: true });
    for (const [s, b] of [['clusters-kmeans', 'kmeans1'], ['maps-wine', 'pca1'], ['prepare-pumps', 'impute1'], ['cv-wildflowers', 'cv1']]) {
      await sample(p, s);
      await open(p, b);
      assert.ok(await p.noHScroll(), b + ' fits the phone');
    }
    await p.viewport(1400, 900);
  }, { timeout: 60000 });
}
