// The machine-learning exercise kinds in the page, on exercises of the course: a confusion matrix and attention weights
// to fill, points to click on a plot (a place, a threshold, a point to pick), a curve to sketch by dragging, and a
// workflow to build and score.
import { assert } from '../lib.mjs';

/** the screen position of a point of the exercise's plot (in its own units) */
const screen = (p, x, y) => p.fn((x, y) => {
  const svg = document.querySelector('#exPlot svg'), m = svg.getScreenCTM(), id = location.hash.split('=')[1];
  const P = Nooral.Check.plotOf(Nooral.Content.exercises.get(id));
  const phone = innerWidth < 600, B = phone ? (P.strip ? { x: 30, y: 30, w: 310, h: 110 } : { x: 48, y: 30, w: 290, h: 250 }) : { x: 66, y: 34, w: 520, h: P.strip ? 130 : 300 };
  const ux = B.x + ((x - P.xr[0]) / (P.xr[1] - P.xr[0])) * B.w, uy = B.y + B.h - ((y - P.yr[0]) / (P.yr[1] - P.yr[0])) * B.h;
  return [m.a * ux + m.c * uy + m.e, m.b * ux + m.d * uy + m.f];
}, x, y);
const check = async (p) => { await p.ev("(() => { const r = document.getElementById('exResult'); if (r) { r.className = 'ex-result'; r.textContent = ''; } return 1; })()"); await p.click('#btnCheck'); await p.wait("!!document.querySelector('#exResult.ok, #exResult.bad')", 60000); return p.text('#exResult'); };

export default function (t) {
  t.test('a confusion matrix: wrong cells marked red, then all right; the diagonal tinted', async (p) => {
    await p.route('exercise=ev-x-cm-fill');
    assert.equal(await p.count('#answer .mx-table input'), 4);
    assert.ok(await p.exists('#answer .mx-table.mx-diag'), 'a confusion matrix tints its diagonal');
    await p.ev("document.querySelectorAll('#answer input').forEach((i) => { i.value = '5'; }), 1");
    let r = await check(p);
    assert.match(r, /Not yet/);
    assert.ok((await p.count('#answer input.wrong')) >= 2);
    const sol = await p.ev("Nooral.Content.exercises.get('ev-x-cm-fill').solution");
    await p.ev(`(() => { const s = ${JSON.stringify(sol)}; document.querySelectorAll('#answer input').forEach((i) => { i.value = s[+i.dataset.i][+i.dataset.j]; }); return 1; })()`);
    r = await check(p);
    assert.match(r, /Correct/);
    assert.equal(await p.count('#answer input.right'), 4);
  });

  t.test('a grid that is not a confusion matrix has no diagonal tint', async (p) => {
    await p.route('exercise=cn-x-conv-map');
    assert.ok(await p.exists('#answer .mx-table'));
    assert.ok(!(await p.exists('#answer .mx-table.mx-diag')));
  });

  t.test('attention weights: the row sums follow the typing', async (p) => {
    await p.route('exercise=tf-x-weights');
    const first = '#answer input[data-i="0"]';
    assert.equal((await p.text('.mx-sum[data-row="0"]')).trim(), '0', 'an empty row sums to 0');
    await p.setValue(first + '[data-j="0"]', '0.5');
    await p.setValue(first + '[data-j="1"]', '0.4');
    assert.equal((await p.text('.mx-sum[data-row="0"]')).trim(), '0.9');
    assert.ok(!(await p.exists('.mx-sum.off[data-row="0"]')), 'not red while the row is still being filled');
    await p.setValue(first + '[data-j="2"]', '0.3');
    assert.equal((await p.text('.mx-sum[data-row="0"]')).trim(), '1.2');
    assert.ok(await p.exists('.mx-sum.off[data-row="0"]'), 'a full row that misses 1 is red');
    await p.setValue(first + '[data-j="2"]', '0.1');
    assert.ok(!(await p.exists('.mx-sum.off[data-row="0"]')));
    assert.ok(await p.exists('.mx-sum.hit[data-row="0"]'), 'a full row that adds to 1 is marked');
    const r = await check(p);
    assert.match(r, /not right yet|Not yet/);
  });

  t.test('click the next centroid: far, then on it', async (p) => {
    await p.route('exercise=un-x-update');
    await p.wait("!!document.querySelector('#exPlot svg')");
    let [x, y] = await screen(p, 6, 1);
    await p.clickAt(x, y);
    assert.match(await p.text('#pointReadout'), /Your point: \(/);
    let r = await check(p);
    assert.doesNotMatch(r, /Correct/);
    const a = await p.ev("Nooral.Content.exercises.get('un-x-update').answer");
    [x, y] = await screen(p, a[0], a[1]);
    await p.clickAt(x, y);
    r = await check(p);
    assert.match(r, /Correct/);
  });

  t.test('a split threshold and a point to pick', async (p) => {
    await p.route('exercise=tr-x-best-threshold');
    const th = await p.ev("Nooral.Content.exercises.get('tr-x-best-threshold').answer");
    let [x, y] = await screen(p, th + 0.02, 0.5);
    await p.clickAt(x, y);
    assert.match(await p.text('#pointReadout'), /Your line: /);
    assert.match(await check(p), /Correct/);
    await p.route('exercise=nn-x-nearest-pick');
    const pts = await p.ev("Nooral.Content.exercises.get('nn-x-nearest-pick').plot.points");
    [x, y] = await screen(p, pts[0][0], pts[0][1]);
    await p.clickAt(x + 3, y - 2);
    assert.match(await p.text('#pointReadout'), /You chose the point at \(/);
    assert.doesNotMatch(await check(p), /Correct/);
  });

  t.test('sketch a curve by dragging', async (p) => {
    await p.route('exercise=li-x-gd-curve');
    await p.wait("!!document.querySelector('#exPlot svg')");
    const [x1, y1] = await screen(p, 0, 80), [x2, y2] = await screen(p, 30, 80);
    await p.drag(x1, y1, x2, y2, { steps: 16 });
    const got = await p.page('P.answer.get()');
    assert.ok(got.length > 8, 'the drag was sampled into points: ' + got.length);
    const r = await check(p);
    assert.doesNotMatch(r, /Correct/);
    // a sketch that follows the curve passes
    await p.ev("(() => { const x = Nooral.Content.exercises.get('li-x-gd-curve'); Nooral.app.page.answer.set(x.answer.map(([a, b]) => [a, b + 0.2])); return 1; })()");
    assert.match(await check(p), /Correct/);
    await p.click('#btnClearSketch');
    assert.equal((await p.page('P.answer.get()')).length, 0);
  });

  t.test('a workflow: the task, the canvas, and scoring on hidden rows', async (p) => {
    await p.route('exercise=tr-x-wf-moons');
    assert.match(await p.text('#wfTask'), /accuracy ≥ 0\.84/);
    await p.wait("!!document.querySelector('#exCanvas svg')", 10000);
    let r = await check(p);
    assert.match(r, /could not be scored/);
    await p.ev("(() => { Nooral.app.page.answer.set(Nooral.Content.exercises.get('tr-x-wf-moons').wrong[0]); return 1; })()");
    r = await check(p);
    assert.doesNotMatch(r, /Correct/);
    await p.ev("(() => { Nooral.app.page.answer.set(Nooral.Content.exercises.get('tr-x-wf-moons').solution); return 1; })()");
    r = await check(p);
    assert.match(r, /Correct/);
  }, { timeout: 90000 });

  t.test('the exercise pages fit a phone', async (p) => {
    for (const id of ['tf-x-weights', 'un-x-update', 'li-x-gd-curve', 'tr-x-wf-moons']) {
      await p.route('exercise=' + id);
      assert.ok(await p.fitsPhone(), id);
    }
  });
}
