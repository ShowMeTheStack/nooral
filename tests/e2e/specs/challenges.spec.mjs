// The Challenges pages: the list, a challenge's story, task and canvas, an attempt that cannot be scored, the reference
// workflow reaching the target with a medal, attempts kept across a reload, storage that is off, and the phone layout.
import { assert } from '../lib.mjs';

const ID = 'nn-ch-pumps';
const scored = "document.querySelector('#chResult.ok, #chResult.bad')";

export default function (t) {
  t.test('the list shows every challenge with its goal', async (p) => {
    await p.route('challenges');
    const ids = await p.ev("[...document.querySelectorAll('.ch-card')].map(a => a.dataset.challenge)");
    assert.deepEqual(ids, await p.ev('Nooral.Content.ladder().map((x) => x.id)'));
    assert.ok(ids.length >= 3);
    assert.match(await p.text('.ch-card[data-challenge="' + ID + '"]'), /F1 ≥ 0\.9/);
    assert.match(await p.text('.ch-card[data-challenge="' + ID + '"]'), /not tried yet/);
    assert.equal(await p.ev("document.querySelector('#topNav button.on').dataset.route"), 'challenges');
    await p.click('.ch-card[data-challenge="' + ID + '"]');
    await p.wait("location.hash === '#challenge/" + ID + "'");
    await p.wait("!!document.querySelector('#challengeView')");
  });

  t.test('a challenge: story, data, goal, limits and a canvas with only the allowed blocks', async (p) => {
    await p.route('challenge/' + ID);
    assert.match(await p.text('.ch-story-full'), /water works/);
    const task = await p.text('#wfTask');
    assert.match(task, /Pump sensors/);
    assert.match(task, /F1 ≥ 0\.9/);
    assert.match(task, /Models you may train: /);
    assert.match(task, /600 hidden rows/);
    await p.wait("!!document.querySelector('#chCanvas svg')", 10000);
    const doc = await p.page('P.editor().doc()');
    assert.equal(doc.blocks.length, 1, 'the starter holds the challenge data alone');
    assert.equal(doc.blocks[0].type, 'challenge-data');
  });

  t.test('an attempt with nothing to score says why', async (p) => {
    await p.route('challenge/' + ID);
    await p.wait("!!(Nooral.app.page.editor && Nooral.app.page.editor())", 10000);
    await p.click('#btnSubmit');
    await p.wait(scored, 30000);
    assert.match(await p.text('#chResult'), /could not be scored/);
    assert.match(await p.text('#chResult'), /no Train block/);
  });

  t.test('the reference workflow reaches the target, earns a medal and is remembered', async (p) => {
    await p.route('challenge/' + ID);
    await p.wait("!!(Nooral.app.page.editor && Nooral.app.page.editor())", 10000);
    await p.ev("Nooral.app.page.load(Nooral.Content.challenges.get('" + ID + "').solution), 1");
    await p.click('#btnSubmit');
    await p.wait("!!document.querySelector('#chResult.ok')", 60000);
    const r = await p.text('#chResult');
    assert.match(r, /Target reached/);
    assert.match(r, /0\.929/);
    assert.match(r, /first scored attempt/);
    assert.ok(await p.exists('#chResult .medal.bronze'));
    assert.match(await p.text('#chHistory'), /Your attempts \(\d+\)/);
    // a second attempt that breaks the limits does not take the top place
    await p.ev("(() => { const d = Nooral.app.page.editor().doc(); d.blocks.push({ id: 'g1', type: 'gboost', params: {}, x: 900, y: 40 }); Nooral.app.page.load(d); return 1; })()");
    await p.ev("document.getElementById('chResult').className = 'ch-result', 1");
    await p.click('#btnSubmit');
    await p.wait("document.querySelector('#chResult.bad')", 60000);
    assert.match(await p.text('#chResult'), /breaks the limits/);
    await p.route('challenges');
    const card = await p.text('.ch-card[data-challenge="' + ID + '"]');
    assert.match(card, /best 0\.929/);
    assert.ok(await p.exists('.ch-card.solved[data-challenge="' + ID + '"] .ch-ok'));
    await p.reload();
    await p.route('challenges');
    assert.match(await p.text('.ch-card[data-challenge="' + ID + '"]'), /best 0\.929/, 'kept across a reload');
  }, { timeout: 120000 });

  t.test('with storage off the page still works', async (p) => {
    await p.ev("(() => { const no = () => { throw new Error('storage is off'); }; Storage.prototype.getItem = no; Storage.prototype.setItem = no; return 1; })()");
    await p.route('challenges');
    assert.ok((await p.count('.ch-card')) >= 3);
    await p.route('challenge/' + ID);
    await p.wait("!!(Nooral.app.page.editor && Nooral.app.page.editor())", 10000);
    await p.click('#btnSubmit');
    await p.wait(scored, 30000);
    assert.match(await p.text('#chResult'), /could not be scored/);
  }, { fresh: true });

  t.test('hints come one at a time; a missing challenge says so', async (p) => {
    await p.route('challenge/' + ID);
    await p.click('#btnHint');
    assert.equal(await p.count('#hints .hint'), 1);
    await p.click('#btnHint'); await p.click('#btnHint');
    assert.equal(await p.count('#hints .hint'), 3);
    assert.ok(await p.ev("document.getElementById('btnHint').disabled"));
    await p.route('challenge/nowhere');
    assert.match(await p.text('.notice'), /no challenge called/);
  });

  t.test('the pages fit a phone', async (p) => {
    await p.route('challenges');
    assert.ok(await p.fitsPhone(), 'the list');
    await p.route('challenge/' + ID);
    assert.ok(await p.fitsPhone(), 'a challenge');
  });
}
