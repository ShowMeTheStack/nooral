// The frame of the app: the top bar and its routes, the home page and its live picture, themes and projector mode,
// settings, the shortcuts sheet, About and the author, a missing page, share links, the demo page, and every page at
// phone width.
import { assert } from '../lib.mjs';

const SECTIONS = ['topics', 'learn', 'courses', 'canvas', 'playground', 'textlab', 'challenges'];
const SIBLINGS = ['See', 'Strux', 'Baysik', 'Lohgo', 'Kwery', 'Tooring', 'Sirkit', 'Sifer'];

export default function (t) {
  t.test('the top bar routes to every section and lights the one open', async (p) => {
    const have = await p.ev(`${JSON.stringify(SECTIONS)}.filter((id) => !!document.querySelector('#topNav button[data-route="' + id + '"]'))`);
    assert.ok(have.length >= 3, 'sections in the bar: ' + have.join(', '));
    for (const id of have) {
      await p.route(id);
      assert.equal(await p.ev("(document.querySelector('#topNav button.on') || {}).dataset?.route || ''"), id, 'nav for ' + id);
    }
    await p.route('lesson=nowhere');
    assert.equal(await p.ev("document.querySelector('#topNav button.on').dataset.route"), 'learn', 'an alias lights its page');
    await p.click('#brand');
    await p.wait("location.hash === '' || location.hash === '#'");
    assert.ok(await p.exists('.hero'));
    await p.click('#topNav button[data-route="topics"]');
    await p.wait("location.hash === '#topics'");
  });

  t.test('the home page: the live workflow picture, the workspaces and the eighteen parts', async (p) => {
    await p.route('');
    await p.wait("!!document.querySelector('#heroArt svg')");
    assert.match(await p.text('#heroCap'), /k-nearest neighbours with k = \d+/);
    assert.match(await p.ev("[...document.querySelectorAll('#heroArt text')].map((t) => t.textContent).join(' ')"), /% of test rows right/);
    assert.equal(await p.count('.part-card'), 18);
    assert.ok((await p.count('.tool-card')) >= 5);
    await p.click('#toScenes');
    await p.wait("location.hash === '#demo'");
    await p.route('');
    await p.click('#btnToTopics');
    await p.wait("location.hash === '#topics'");
  });

  t.test('a page that is not there says so', async (p) => {
    await p.route('nowhere-at-all');
    assert.match(await p.text('.notice'), /no page called/);
  });

  t.test('light and dark, projector mode, and settings', async (p) => {
    await p.route('topics');
    const t0 = await p.theme();
    await p.click('#btnTheme');
    assert.notEqual(await p.theme(), t0);
    await p.key('t');
    assert.equal(await p.theme(), t0);
    await p.click('#btnProjector');
    assert.equal(await p.ev('document.documentElement.dataset.projector'), 'on');
    await p.key('P', { shift: true });
    assert.equal(await p.ev('document.documentElement.dataset.projector'), 'off');
    await p.click('#btnAbout');
    await p.click('.menu-pop .menu-item', 'Settings');
    assert.equal(await p.waitModal(), 'Settings');
    assert.equal(await p.ev("document.getElementById('set-autosave').checked"), true, 'autosave is on by default');
    await p.ev("(() => { const c = document.getElementById('set-autosave'); c.checked = false; c.dispatchEvent(new Event('change')); })()");
    assert.equal((await p.settings()).autosave, false, 'and can be turned off');
    await p.ev('Nooral.Shell.autosave()').then((v) => assert.equal(v, false));
    await p.setValue('#set-theme', 'dark', 'change');
    assert.equal(await p.theme(), 'dark');
    await p.setValue('#set-theme', 'light', 'change');
    assert.match(await p.text('#engineStatus'), /Engines: Tensors and autodiff/);
    await p.closeModals();
    assert.equal((await p.settings()).theme, 'light');
  });

  t.test('the shortcuts sheet and About, with the author and every sister tool', async (p) => {
    await p.route('');
    await p.key('?');
    assert.equal(await p.waitModal(), 'Keyboard shortcuts');
    assert.match(await p.modalText(), /Light or dark/);
    await p.closeModals();
    await p.click('#btnAbout');
    await p.click('.menu-pop .menu-item', 'About Nooral');
    assert.equal(await p.waitModal(), 'About Nooral');
    const about = await p.modalText();
    assert.match(about, /GNU General Public License/);
    assert.match(about, /Based on Nooral by Jyotiprakash Mishra/);
    for (const name of SIBLINGS) assert.match(about, new RegExp(name), name);
    await p.closeModals();
    await p.click('#btnAbout');
    await p.click('.menu-pop .menu-item', 'About the author');
    assert.equal(await p.waitModal(), 'About the author');
    const text = await p.modalText();
    assert.match(text, /Jyotiprakash Mishra/);
    for (const name of SIBLINGS) assert.match(text, new RegExp(name), name);
    assert.ok(await p.ev("!!document.querySelector('.author-photo img').getAttribute('src').startsWith('data:image/')"), 'the photo');
    await p.closeModals();
  });

  t.test('a damaged share link says so and lands on the home page', async (p) => {
    await p.ev("location.hash = 'share=N1.zAAAA'");
    await p.waitToast(/damaged|incomplete|does not look/);
    await p.wait("location.hash === '' || location.hash === '#'");
  });

  t.test('a share link round trip opens what was shared', async (p) => {
    const kinds = await p.ev("['canvas', 'playground', 'textlab'].filter((k) => Nooral.Pages.get(k) && Nooral.Pages.get(k).load)");
    if (!kinds.includes('canvas')) return;
    const code = await p.ev("(async () => Nooral.Share.pack({ kind: 'canvas', data: Nooral.WF.sample('first-pipeline') || Nooral.WF.newDoc('shared') }))()");
    await p.ev(`location.hash = 'share=' + ${JSON.stringify(code)}`);
    await p.waitToast(/Opened the shared workflow/);
    await p.wait("location.hash === '#canvas'");
  });

  t.test('the demo page shows every sample scene, and its players step', async (p) => {
    await p.route('demo');
    const n = await p.ev('Nooral.Demo.scenes.length');
    assert.ok(n >= 6, 'scenes: ' + n);
    assert.equal(await p.count('section.scene'), n);
    const id = await p.ev('Nooral.Demo.scenes[0].id');
    await p.ev(`Nooral.app.byId[${JSON.stringify(id)}].go(1, true)`);
    assert.match(await p.text('#' + id + ' .say'), /Step 2 of/i);
  });

  t.test('the menu on a phone, and every page fits the phone width', async (p) => {
    await p.viewport(390, 844, { mobile: true, touch: true });
    await p.route('');
    assert.ok(await p.visible('#btnMenu'), 'the menu button');
    assert.ok(!(await p.visible('#topNav button[data-route="learn"]')), 'the sections are folded away');
    await p.click('#btnMenu');
    assert.ok(await p.visible('#topNav button[data-route="learn"]'));
    await p.click('#topNav button[data-route="learn"]');
    await p.wait("location.hash === '#learn'");
    for (const id of [''].concat(SECTIONS, ['demo'])) {
      if (id && !(await p.ev(`!!Nooral.Pages.get(${JSON.stringify(id)})`))) continue;
      await p.route(id);
      await p.sleep(150);
      assert.ok(await p.noHScroll(), 'the page ' + (id || 'home') + ' scrolls sideways at 390 px');
    }
    await p.viewport(1400, 900);
  });
}
