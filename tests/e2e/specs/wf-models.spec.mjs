// The model and Train viewers on the canvas: a trained classical model's insides, a network's training controls
// (reset, one batch, one epoch, play, pause), its curves, and the pictures of the networks.
import { assert } from '../lib.mjs';

const load = async (p, sample, patch) => {
  await p.route('canvas');
  await p.ev('(async () => { const P = Nooral.app.page; await P.loadDoc(Nooral.WF.newDoc()); await P.whenIdle(); return 1; })()');
  await p.ev(`(async () => { const P = Nooral.app.page; const d = Nooral.WF.sample(${JSON.stringify(sample)}); ${patch ? 'Object.assign(Nooral.WF.byId(d, "train1").params, ' + JSON.stringify(patch) + ');' : ''} await P.loadDoc(d); await P.whenIdle(); return 1; })()`);
};
const where = (p) => p.text('#wfmWhere');

export default function (t) {
  t.test('a decision tree: the tree, the plane and the steps of its growth', async (p) => {
    await load(p, 'tree-moons');
    await p.ev("Nooral.app.page.open('train1')");
    await p.wait("!!document.querySelector('.wfm .stage-svg .it-state.tnode')");
    assert.match(await p.text('.wfm-title'), /Trained: Decision tree/);
    assert.ok((await p.count('.wfm .it-raster')) >= 1, 'the plane is shaded');
    const steps = await p.ev("document.querySelector('.wfm-controls .pos').textContent");
    assert.match(steps, /^(\d+) \/ \1$/, 'it opens at the grown tree');
    await p.click('.wfm-controls button[aria-label="Back to the start"]');
    assert.match(await p.text('.wfm-say'), /root holds every training row/);
    assert.equal(await p.count('.wfm .it-state.tnode'), 1, 'the root alone');
    await p.click('.wfm-controls button[aria-label="Step forward"]');
    assert.equal(await p.count('.wfm .it-state.tnode'), 3, 'one split: three nodes');
    assert.match(await p.text('.wfm-say'), /splits on x\d <= /);
  });

  t.test('k-NN answers a click with the neighbours of that point', async (p) => {
    await load(p, 'knn-blobs');
    await p.ev("Nooral.app.page.open('train1')");
    await p.wait("!!document.querySelector('.wfm-pick .stage-svg')");
    const before = await p.text('.wfm-say');
    const r = await p.rect('.wfm-pick');
    await p.clickAt(r.x + r.w * 0.3, r.y + r.h * 0.7);
    await p.wait(`document.querySelector('.wfm-say').innerText !== ${JSON.stringify(before)}`);
    assert.match(await p.text('.wfm-say'), /asks its 7 nearest neighbours/);
  });

  t.test('training controls: reset, one batch, one epoch, play, pause', async (p) => {
    await load(p, 'mlp-spirals', { epochs: 4000 });
    await p.ev("Nooral.app.page.open('train1')");
    await p.wait("!!document.getElementById('wfmReset')");
    await p.click('#wfmReset');
    await p.wait("/^epoch 0 of 4000 · batch 0/.test((document.getElementById('wfmWhere') || {}).textContent || '')", 15000);
    await p.click('#wfmBatch');
    await p.wait("/^epoch 0 of 4000 · batch 1 of/.test((document.getElementById('wfmWhere') || {}).textContent || '')", 15000);
    await p.click('#wfmEpoch');
    await p.wait("/^epoch 1 of 4000 · batch 0 of/.test((document.getElementById('wfmWhere') || {}).textContent || '')", 15000);
    assert.ok((await p.count('.wfm-curves .it-path.series')) >= 1, 'the curves have their first points');
    await p.click('#wfmPlay');
    await p.wait("(() => { const m = /^epoch (\\d+)/.exec((document.getElementById('wfmWhere') || {}).textContent || ''); return m && +m[1] > 5; })()", 20000);
    await p.click('#wfmPause');
    await p.wait("Nooral.app.page.remote.states.train1 && Nooral.app.page.remote.states.train1.state === 'paused'", 15000);
    const e1 = +(/^epoch (\d+)/.exec(await where(p)) || [])[1];
    await p.sleep(600);
    const e2 = +(/^epoch (\d+)/.exec(await where(p)) || [])[1];
    assert.ok(e1 > 5 && e1 < 4000 && e1 === e2, 'paused stays put at epoch ' + e1);
    assert.equal(await p.ev("Nooral.app.page.remote.states.evaluate1.state"), 'done', 'the paused model is evaluated');
  });

  t.test('play runs to the end; the network, its regions and the curves are drawn', async (p) => {
    await load(p, 'mlp-spirals', { epochs: 30 });
    await p.ev("Nooral.app.page.open('train1')");
    await p.wait("/^epoch 30 of 30/.test((document.getElementById('wfmWhere') || {}).textContent || '')", 20000);
    assert.ok((await p.count('.wfm .it-state.neuron')) >= 12, 'the neurons');
    assert.ok((await p.count('.wfm .it-path.weight')) >= 40, 'the weights');
    assert.ok((await p.count('.wfm .it-raster')) >= 9, 'the plane and every hidden neuron’s map');
    assert.match(await p.text('.wfm'), /Train again/);
  });

  t.test('the text networks show their insides', async (p) => {
    // the viewer is drawn while training runs and again when it ends, and the viewer of the sample before may still
    // be on the page: wait for the words of the finished network itself
    // (its probe is filled in a moment after each redraw, so the words are waited for, twice, rather than read once)
    const done = "Nooral.app.page.remote.states.train1 && Nooral.app.page.remote.states.train1.state === 'done'";
    const words = (re) => `(() => { const el = document.querySelector('.wfm'); return !!el && (${done}) && !!document.querySelector('.wfm-probe') && ${re}.test(el.textContent); })()`;
    const settle = async (re) => { await p.wait(words(re), 12000); await p.sleep(300); await p.wait(words(re), 4000); };
    await load(p, 'lstm-spam', { epochs: 3 });
    await p.ev("Nooral.app.page.open('train1')");
    // a viewer near the phone width says the short form, 'the gates (0 shut, 1 open)': both are right
    await settle('/the gates(, averaged over the units)? .0 shut, 1 open./');
    await p.ev('Nooral.app.page.closeViewer ? Nooral.app.page.closeViewer() : 1');
    await load(p, 'transformer-spam', { epochs: 3 });
    await p.ev("Nooral.app.page.open('train1')");
    await settle('/head 2/');
  });

  t.test('an untrained model block explains itself; the viewer fits a phone', async (p) => {
    await load(p, 'mlp-spirals', { epochs: 5 });
    await p.ev("Nooral.app.page.open('mlp1')");
    await p.wait("!!document.querySelector('.wfm-table')");
    assert.match(await p.text('.wfm-title'), /not trained yet/);
    assert.ok(await p.fitsPhone(), 'no sideways scroll at phone width');
  });
}
