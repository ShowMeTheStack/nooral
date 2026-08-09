// Topics, Learn and Courses: the catalogue, a topic's player (steps, a slider that keeps the step, the quiz),
// clickable items, the list of lessons, a lesson, and a course with its strip.
import { assert } from '../lib.mjs';

export default function (t) {
  t.test('the topics catalogue lists the parts and opens a topic', async (p) => {
    await p.route('topics');
    assert.equal(await p.count('#topicsView section.part'), 18);
    const id = await p.ev('Nooral.topics[0].id');
    await p.route('topic=' + id);
    await p.wait("!!document.querySelector('.topic #stageWrap svg')");
    assert.match(await p.text('#say'), /\S/);
    await p.click('#btnFwd');
    assert.match(await p.text('#pos'), /^2\b|2 \//);
  });

  t.test('a slider keeps the step it was on', async (p) => {
    const id = await p.ev("(Nooral.topics.find((t) => (t.inputs || []).some((i) => i.kind === 'slider')) || {}).id || ''");
    if (!id) return;
    await p.ev('Nooral.Learner.settings.predict = false');
    await p.route('topic=' + id);
    await p.wait("!!document.querySelector('.topic input[type=range][id$=\"-r\"]')");
    await p.ev('Nooral.app.page.player.go(2, true)');
    const n = await p.ev('Nooral.app.page.player.frames.length');
    await p.ev("(() => { const r = document.querySelector('.topic input[type=range][id$=\"-r\"]'); r.value = String((+r.min + +r.max) / 2); r.dispatchEvent(new Event('input', { bubbles: true })); })()");
    await p.sleep(500);
    assert.equal(await p.ev('Nooral.app.page.player.i'), Math.min(2, n - 1), 'still on the third step');
  });

  t.test('a clickable item shows its details under the narration', async (p) => {
    const id = await p.ev('Nooral.topics[0].id');
    await p.route('topic=' + id);
    await p.ev("Nooral.app.page.player.showInfo({ title: 'Cell (1, 0)', rows: [[1, 2]], cols: ['a', 'b'] }, 'x')");
    assert.ok(await p.exists('.pick-info.on'));
    assert.match(await p.text('.pick-info'), /Cell \(1, 0\)/);
    await p.click('.pick-info .pick-x');
    assert.ok(!(await p.exists('.pick-info.on')));
  });

  t.test('the quiz of a topic', async (p) => {
    const id = await p.ev('Nooral.topics[0].id');
    await p.route('topic=' + id);
    await p.click('#btnQuiz');
    await p.waitModal();
    await p.closeModals();
  });

  t.test('learn lists chapters and a lesson opens and is marked read', async (p) => {
    await p.route('learn');
    assert.ok((await p.count('.chap-card')) > 0);
    const id = await p.ev('[...Nooral.Content.lessons.keys()][0]');
    await p.route('lesson=' + id);
    await p.wait("!!document.querySelector('#lessonView')");
    await p.click('#lessonDone');
    assert.ok(await p.ev(`Nooral.Learn.isDone('lesson', ${JSON.stringify(id)})`));
  });

  t.test('a course: the overview, a step with its strip, next and back', async (p) => {
    await p.route('courses');
    const id = await p.ev('[...Nooral.Content.courses.keys()].find((k) => !k.startsWith("app-")) || [...Nooral.Content.courses.keys()][0]');
    await p.route('course=' + id);
    await p.click('#courseStart');
    await p.wait("!!document.querySelector('#courseStrip')");
    // Start resumes at the first step not yet done (an earlier test marked a lesson read)
    const at = await p.ev("+location.hash.split('/').pop()");
    await p.click('#courseNext');
    await p.wait(`location.hash.endsWith('/${at + 1}')`);
    await p.click('#coursePrev');
    await p.wait(`location.hash.endsWith('/${at}')`);
  });
}
