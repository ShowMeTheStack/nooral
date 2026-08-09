// The playground: the network trains and pauses, the controls rebuild it, inputs switch, a weight reads out, the
// setup survives a share link, and the page fits a phone.
import { assert } from '../lib.mjs';

export default function (t) {
  t.test('the default network draws every neuron as a map', async (p) => {
    await p.route('playground');
    await p.wait("document.querySelectorAll('#pgNetStage .it-raster').length >= 7 + 4 + 2 + 1");
    assert.match(await p.text('#pgEpoch'), /Epoch 000/);
    assert.ok(await p.exists('#topNav button[data-route="playground"].on'), 'the nav lights');
  });

  t.test('play trains, pause stops, one epoch steps', async (p) => {
    await p.route('playground');
    await p.click('#pgPlay');
    await p.wait("Nooral.app.page.lab.epoch >= 3", 15000);
    await p.click('#pgPlay');
    const e = await p.page('P.lab.epoch');
    assert.equal(await p.page('P.running'), false);
    await p.sleep(300);
    assert.equal(await p.page('P.lab.epoch'), e, 'paused: no more epochs');
    await p.click('#pgStep');
    assert.equal(await p.page('P.lab.epoch'), e + 1, 'one epoch');
    assert.match(await p.text('#pgEpoch'), new RegExp('Epoch 0*' + (e + 1)));
    await p.click('#pgReset');
    assert.equal(await p.page('P.lab.epoch'), 0, 'reset');
  });

  t.test('the loss falls on circles', async (p) => {
    await p.route('playground');
    const before = await p.page('P.lab.losses().train');
    await p.ev('Nooral.app.page.train(40)');
    const after = await p.page('P.lab.losses().train');
    assert.ok(after < before * 0.6, before + ' → ' + after);
    assert.match(await p.text('#pgLoss'), /Test loss \d\.\d{3}/);
  });

  t.test('layers and neurons, inputs and the dataset', async (p) => {
    await p.route('playground');
    await p.click('#pgLayerPlus');
    assert.equal(JSON.stringify(await p.page('P.cfg.layers')), '[4,2,2]');
    await p.click('#pgPlus0');
    assert.equal(JSON.stringify(await p.page('P.cfg.layers')), '[5,2,2]');
    await p.click('#pgMinus2');
    await p.click('#pgMinus2');
    assert.equal(JSON.stringify(await p.page('P.cfg.layers')), '[5,2]');
    assert.equal(await p.page('P.lab.sizes.join()'), '2,5,2,1');
    await p.ev("Nooral.app.page.pick('pg:f:x1sq')");
    assert.equal((await p.page('P.cfg.features')).join(), 'x1,x2,x1sq');
    assert.equal(await p.page('P.lab.sizes[0]'), 3);
    await p.click('.pg-sets .chip[data-set="xor"]');
    assert.equal(await p.page('P.cfg.data'), 'xor');
    await p.setValue('#pgAct', 'relu', 'change');
    assert.equal(await p.page('P.cfg.act'), 'relu');
  });

  t.test('a weight reads out, a neuron shows on the plane', async (p) => {
    await p.route('playground');
    await p.ev("Nooral.app.page.pick('pg:w0_0_0')");
    assert.match(await p.text('#pgRead'), /Weight from x₁ to neuron 1\.1: −?\d/);
    await p.ev("Nooral.app.page.pick('pg:n1_0')");
    assert.match(await p.text('#pgWhat'), /neuron 1\.1/);
    assert.ok(await p.exists('#pgNetStage .it-raster.act'));
  });

  t.test('a share link brings the setup back', async (p) => {
    await p.route('playground');
    await p.ev("Nooral.app.page.set({ data: 'moons', layers: [3], act: 'relu', lr: 0.3 })");
    const url = await p.ev("Nooral.Shell.share('playground', Nooral.app.page.cfg)");
    await p.closeModals();
    await p.route('');
    await p.ev(`(() => { window.__nooralReady = false; location.hash = ${JSON.stringify(url)}.replace(/^[^#]*#/, ''); })()`);
    await p.wait("location.hash === '#playground' && window.__nooralReady === true && !!Nooral.app.page && !!Nooral.app.page.cfg", 10000);
    assert.equal(await p.page('P.cfg.data'), 'moons');
    assert.equal(JSON.stringify(await p.page('P.cfg.layers')), '[3]');
    assert.equal(await p.page('P.cfg.lr'), 0.3);
  });

  t.test('the playground fits a phone', async (p) => {
    await p.route('playground');
    assert.ok(await p.fitsPhone(), 'no sideways scroll at 390 px');
  });
}
