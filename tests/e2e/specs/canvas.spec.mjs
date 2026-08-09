// The workflow canvas: the first pipeline running in the worker, blocks added from the palette by click and by drag,
// wires drawn between ports (and one refused, with its reason), a parameter edited in its dialog and what follows it
// running again, undo and redo, tidy-up, the viewer, save and open, a share link, autosave, and the phone layout.
import { assert } from '../lib.mjs';

const P = 'Nooral.app.page';
const idle = (p) => p.ev(`${P}.whenIdle().then(() => 1)`);
/** the middle of a port on the screen */
const portXY = (p, id, side, port) => p.ev(`(() => { const e = document.querySelector('.wf-block[data-id="${id}"] .wf-port[data-side="${side}"][data-port="${port}"]'); const r = e.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()`);
const blockXY = (p, id) => p.ev(`(() => { const r = document.querySelector('.wf-block[data-id="${id}"] .wf-bbox').getBoundingClientRect(); return [r.left + r.width / 2, r.top + 22]; })()`);
const svgText = (p, sel) => p.ev(`(document.querySelector(${JSON.stringify(sel)}) || {}).textContent || ''`);
const empty = async (p) => { await p.ev(`${P}.loadDoc(Nooral.WF.newDoc()).then(() => 1)`); };

export default function (t) {
  t.test('the canvas opens with a first pipeline and runs it in a worker', async (p) => {
    await p.route('canvas');
    await idle(p);
    assert.equal(await p.ev(`${P}.remote.mode`), 'worker', 'runs in a Web Worker');
    assert.equal(await p.count('.wf-block'), 5);
    assert.equal(await p.ev(`${P}.remote.states.evaluate1.state`), 'done');
    assert.match(await svgText(p, '.wf-block[data-id="evaluate1"]'), /accuracy 0\.\d+/);
    assert.match(await p.text('#wfStatus'), /Up to date/);
    assert.equal(await p.ev('Nooral.WF.crossings(Nooral.app.page.doc)'), 0, 'no crossings');
  });

  t.test('a block is added from the palette by a click and by a drag', async (p) => {
    await p.route('canvas');
    await empty(p);
    await p.click('.wf-pal-item[data-type="data"]');
    assert.equal(await p.ev(`${P}.doc.blocks.length`), 1, 'click adds');
    const [x, y] = await p.center('.wf-pal-item[data-type="split"]');
    const r = await p.rect('.wf-wrap');
    await p.drag(x, y, r.x + r.w * 0.7, r.y + r.h * 0.5, { steps: 12 });
    assert.equal(await p.ev(`${P}.doc.blocks.length`), 2, 'drag adds');
    const b = await p.rect('.wf-block[data-id="split1"] .wf-bbox');
    assert.ok(Math.abs(b.x + b.w / 2 - (r.x + r.w * 0.7)) < 40, 'dropped where it was let go');
  });

  t.test('a wire joins two ports; a wire between the wrong kinds is refused with the reason', async (p) => {
    await p.route('canvas');
    await empty(p);
    await p.ev(`(() => { const e = ${P}.editor; e.addBlock('data', null, { x: 0, y: 0 }); e.addBlock('split', null, { x: 300, y: 0 }); e.addBlock('train', null, { x: 600, y: 160 }); e.fit(); return 1; })()`);
    const [ax, ay] = await portXY(p, 'data1', 'out', 'out');
    const [bx, by] = await portXY(p, 'split1', 'in', 'in');
    await p.drag(ax, ay, bx, by, { steps: 10 });
    assert.deepEqual(await p.ev(`${P}.doc.wires.map(w => w.from.join('.') + '>' + w.to.join('.'))`), ['data1.out>split1.in']);
    const [cx, cy] = await portXY(p, 'train1', 'in', 'model');
    await p.drag(ax, ay, cx, cy, { steps: 10 });
    await p.waitToast(/Cannot wire: the model input of Train takes model, not table/);
    assert.equal(await p.ev(`${P}.doc.wires.length`), 1, 'refused');
    await idle(p);
    assert.match(await svgText(p, '.wf-block[data-id="split1"]'), /train · \d+ test/);
  });

  t.test('a parameter edited in its dialog runs what follows again; undo and redo', async (p) => {
    await p.route('canvas');
    await p.ev(`${P}.loadSample('first-pipeline').then(() => 1)`);
    const before = await p.ev(`${P}.remote.states.evaluate1.summary`);
    const [x, y] = await blockXY(p, 'logistic1');
    await p.clickAt(x, y, { double: true });
    await p.waitModal();
    assert.ok(await p.exists('#pd-C'));
    await p.setValue('#pd-C', '0.002');
    await p.modalButton('OK');
    assert.equal(await p.ev(`${P}.doc.blocks.find(b => b.id === 'logistic1').params.C`), 0.002);
    await idle(p);
    const after = await p.ev(`${P}.remote.states.evaluate1.summary`);
    assert.notEqual(after, before, 'the evaluation changed: ' + before + ' → ' + after);
    await p.click('#wfUndo');
    assert.equal(await p.ev(`${P}.doc.blocks.find(b => b.id === 'logistic1').params.C`), 1, 'undo');
    await p.click('#wfRedo');
    assert.equal(await p.ev(`${P}.doc.blocks.find(b => b.id === 'logistic1').params.C`), 0.002, 'redo');
    // a value out of range is refused in the dialog
    await p.clickAt(x, y, { double: true });
    await p.waitModal();
    await p.setValue('#pd-C', '-3');
    assert.match(await p.text('#pd-C-err'), /At least/);
    await p.key('Escape');
  });

  t.test('tidy-up lays out a messy workflow with no crossings; delete and the viewer', async (p) => {
    await p.route('canvas');
    await p.ev(`${P}.loadSample('first-pipeline').then(() => 1)`);
    await p.ev(`(() => { const d = ${P}.doc; d.blocks.forEach((b, i) => { b.x = (i * 137) % 400; b.y = (i * 211) % 300; }); ${P}.editor.commit('mess'); return 1; })()`);
    await p.click('#wfTidy');
    assert.equal(await p.ev(`Nooral.WF.crossings(${P}.doc)`), 0);
    const [x, y] = await blockXY(p, 'split1');
    await p.clickAt(x, y);
    await p.wait("document.querySelector('.wf-viewer[data-block=\"split1\"]') && !document.querySelector('.wf-viewer').hidden");
    assert.match(await p.text('.wf-viewer'), /Train\/test split/);
    await p.key('Escape');
    await p.wait("document.querySelector('.wf-viewer').hidden");
    await p.clickAt(x, y);
    await p.key('Escape');
    await p.ev(`${P}.editor.select(['evaluate1']); document.querySelector('.wf-wrap').focus(); 1`);
    await p.key('Delete');
    assert.equal(await p.ev(`${P}.doc.blocks.length`), 4, 'deleted');
    assert.equal(await p.ev(`${P}.doc.wires.length`), 3, 'with its wires');
  });

  t.test('save to a file and open it again', async (p) => {
    await p.route('canvas');
    await p.ev(`${P}.loadSample('two-strengths').then(() => 1)`);
    await p.ev("window.__saved = null; Nooral.UI.download = (name, data) => { window.__saved = { name, data }; }; 1");
    await p.click('#wfSave');
    const saved = await p.ev('window.__saved');
    assert.match(saved.name, /\.nooral\.json$/);
    await empty(p);
    await p.ev(`(() => { const f = new File([window.__saved.data], 'w.nooral.json', { type: 'application/json' }); Nooral.UI.pickFiles = () => Promise.resolve([f]); return 1; })()`);
    await p.click('#wfOpen');
    await p.wait(`${P}.doc.blocks.length === 8`);
    assert.equal(await p.ev(`JSON.stringify(${P}.doc.wires.map(w => w.id).sort())`), await p.ev(`JSON.stringify(JSON.parse(window.__saved.data).wires.map(w => w.id).sort())`));
  });

  t.test('a share link opens the same workflow; the canvas is kept between visits', async (p) => {
    await p.route('canvas');
    await p.ev(`${P}.loadSample('train-vs-test').then(() => 1)`);
    await p.ev(`${P}.editor.addBlock('evaluate'); 1`);
    await p.ev("window.__url = null; Nooral.Share.copy = (u) => { window.__url = u; }; 1");
    await p.sleep(500);
    await p.click('#wfShare');
    await p.wait('!!window.__url');
    const url = await p.ev('window.__url');
    assert.match(url, /#share=N1\./);
    // autosave: a reload brings the same workflow back
    await p.reload();
    await p.route('canvas');
    assert.equal(await p.ev(`${P}.doc.blocks.length`), 7, 'kept in this browser');
    await empty(p);
    await p.ev(`location.hash = ${JSON.stringify(url.slice(url.indexOf('#') + 1))}; 1`);
    await p.wait("location.hash === '#canvas' && window.__nooralReady === true && Nooral.app.page && Nooral.app.page.doc && Nooral.app.page.doc.blocks.length === 7", 10000);
    assert.equal(await p.ev(`${P}.doc.name`), 'Training rows against test rows');
  });

  t.test('the canvas fits a phone; the palette is a drawer there', async (p) => {
    await p.route('canvas');
    assert.ok(await p.fitsPhone(), 'no sideways scroll');
    await p.viewport(390, 844, { mobile: true, touch: true });
    await p.sleep(400);
    assert.ok(!(await p.visible('.wf-pal-search')), 'the palette is tucked away');
    await p.click('#wfPalBtn');
    await p.sleep(300);
    assert.ok(await p.visible('.wf-pal-search'), 'and slides in');
    await p.viewport(1400, 900);
  });
}
