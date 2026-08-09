// The evaluation, comparison, inference and explanation viewers on the canvas: a confusion cell clicked shows its
// rows, the threshold slider recounts the matrix and moves the point, a curve click moves the threshold, Compare marks
// the best, the inference panel answers a typed row, a click on the plane and a drawn digit, and the explanations draw.
import { assert } from '../lib.mjs';

const P = 'Nooral.app.page';
const open = async (p, sample, block) => {
  await p.route('canvas');
  await p.ev(`${P}.loadSample(${JSON.stringify(sample)}).then(() => ${P}.whenIdle()).then(() => 1)`);
  await p.ev(`(${P}.open(${JSON.stringify(block)}), 1)`);
  await p.wait("!!document.querySelector('.wf-viewer .wf-ev')", 20000);
  await p.sleep(150);
};
const clickKey = (p, key) => p.ev(`(() => { const e = document.querySelector('[data-key="${key}"]'); e.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: 0, clientY: 0 })); return 1; })()`);
const cellText = (p, key) => p.ev(`(() => { const e = document.querySelector('[data-key="${key}"]'); return e ? [...e.querySelectorAll('text')].map((t) => t.textContent).join(' ') : ''; })()`);

export default function (t) {
  t.test('clicking a confusion cell lists the rows in it', async (p) => {
    await open(p, 'evaluate-loans', 'evaluate1');
    assert.ok(await p.exists('[data-key="cm:c0_0"]'), 'the matrix is drawn');
    // the largest cell of mistakes is open at first
    assert.match(await p.text('.wf-examples h4'), /^Wrong: true \w+, predicted \w+/);
    await clickKey(p, 'cm:c0_0');
    await p.wait("/^Right: true repaid, predicted repaid/.test((document.querySelector('.wf-examples h4') || {}).innerText || '')");
    const n = +(await cellText(p, 'cm:c0_0'));
    const rows = await p.count('.wf-examples tbody tr');
    assert.ok(rows > 0 && rows === Math.min(24, n), 'its rows are listed (' + rows + ' of ' + n + ')');
    assert.ok(await p.exists('.wf-examples [data-key], .wf-examples table'));
    // shares of each true class: the row adds to 1
    await p.click('.wf-seg button', 'share of each true class');
    const a = parseFloat(String(await cellText(p, 'cm:c0_0')).replace(/^\./, '0.')), b = parseFloat(String(await cellText(p, 'cm:c0_1')).replace(/^\./, '0.'));
    assert.ok(Math.abs(a + b - 1) < 0.02, 'shares of a row add up: ' + a + ' + ' + b);
  });

  t.test('the threshold slider recounts the matrix and moves the point on both curves', async (p) => {
    await open(p, 'evaluate-loans', 'roc1');
    const tp0 = await cellText(p, 'mx:c0_0'), pt0 = await p.ev("document.querySelector('[data-key=\"roc:pt\"] circle').getAttribute('cx')");
    await p.setValue('#wfThreshold', '0.15');
    await p.sleep(120);
    const tp1 = await cellText(p, 'mx:c0_0'), pt1 = await p.ev("document.querySelector('[data-key=\"roc:pt\"] circle').getAttribute('cx')");
    assert.notEqual(tp1, tp0, 'the true positives change');
    assert.ok(+/\d+/.exec(tp1)[0] > +/\d+/.exec(tp0)[0], 'a lower threshold finds more of the class');
    assert.notEqual(pt1, pt0, 'the point moves');
    assert.match(await p.text('.wf-threshold output'), /0\.15/);
    // a click on the ROC curve puts the threshold there
    const xy = await p.ev("(() => { const r = document.querySelector('[data-key=\"roc:hit\"] rect, [data-key=\"roc:hit\"] .cell-box').getBoundingClientRect(); return [r.left + r.width * 0.8, r.top + r.height * 0.1]; })()");
    await p.clickAt(xy[0], xy[1]);
    await p.sleep(120);
    assert.notEqual(await p.text('.wf-threshold output'), '0.15', 'the threshold moved to the clicked point');
  });

  t.test('Compare marks the best model of each score', async (p) => {
    await open(p, 'compare-three', 'compare1');
    assert.equal(await p.count('.wf-tab.compare tbody tr'), 3);
    assert.ok((await p.count('.wf-tab.compare .best')) >= 1, 'the best is marked');
    assert.ok(await p.exists('[data-key="p0:field"]'), 'boundaries are drawn');
  });

  t.test('the inference panel answers a typed row and a click on the plane', async (p) => {
    await open(p, 'predict-exam', 'predict1');
    await p.wait("/The model says/.test((document.getElementById('wfAnswer') || {}).innerText || '')", 8000);
    await p.setValue('#wfAsk input[type=range]', '19');
    await p.wait("/The model says (pass|fail)/.test(document.getElementById('wfAnswer').innerText)", 8000);
    assert.ok((await p.count('.wf-bar')) === 2, 'a bar per class');
    await open(p, 'predict-moons', 'predict1');
    const xy = await p.ev("(() => { const r = document.querySelector('[data-key=\"pl:hit\"] .cell-box, [data-key=\"pl:hit\"] rect').getBoundingClientRect(); return [r.left + r.width * 0.15, r.top + r.height * 0.2]; })()");
    const before = await p.ev("document.querySelectorAll('#wfAsk input[type=text]')[0].value");
    await p.clickAt(xy[0], xy[1]);
    await p.sleep(400);
    const after = await p.ev("document.querySelectorAll('#wfAsk input[type=text]')[0].value");
    assert.notEqual(after, before, 'the click set the fields');
    assert.match(await p.text('#wfAnswer'), /The model says/);
  });

  t.test('a digit drawn on the pad is read', async (p) => {
    await open(p, 'predict-digits', 'predict1');
    const r = await p.rect('#wfPad');
    // a vertical stroke: a one
    await p.drag(r.x + r.w * 0.5, r.y + r.h * 0.2, r.x + r.w * 0.52, r.y + r.h * 0.8, { steps: 12 });
    await p.wait("/The model says/.test((document.getElementById('wfAnswer') || {}).innerText || '')", 10000);
    assert.equal(await p.count('#wfAnswer .wf-bar'), 10, 'ten probability bars');
    await p.click('#wfPadClear');
    assert.match(await p.text('#wfAnswer'), /Draw a digit/);
  });

  t.test('the explanations draw', async (p) => {
    for (const [s, b, sel] of [['explain-harvest', 'importance1', '[data-key="imp:b0"]'], ['explain-harvest', 'pdp1', '[data-key="pd:avg"]'], ['explain-harvest', 'shap1', '[data-key="wf:b0"]'], ['explain-harvest', 'lime1', '[data-key="lm:b0"]'], ['predict-digits', 'saliency1', '[data-key="s0:sal"]'], ['regression-residuals', 'residuals1', '[data-key="r1:0"]'], ['learning-curve-tree', 'learningcurve1', '[data-key="vc:v"]'], ['evaluate-loans', 'calibration1', '[data-key="cal:line"]']]) {
      await open(p, s, b);
      assert.ok(await p.exists(sel), b + ' draws ' + sel);
    }
  }, { timeout: 120000 });

  t.test('the viewers fit a phone', async (p) => {
    await p.viewport(390, 844, { mobile: true, touch: true });
    for (const [s, b] of [['evaluate-loans', 'roc1'], ['explain-harvest', 'shap1']]) {
      await open(p, s, b);
      assert.ok(await p.noHScroll(), b + ' fits');
      const w = await p.ev("(() => { const f = document.querySelector('.wf-viewer .wf-fig-stage'); return f.getBoundingClientRect().width <= document.querySelector('.wf-viewer').getBoundingClientRect().width + 1; })()");
      assert.ok(w, b + ': its figure fits its panel');
    }
    await p.viewport(1400, 900);
  }, { timeout: 60000 });
}
