// The text lab: BPE steps merge by merge, the tiny GPT trains in its worker and pauses, sampling follows its rule,
// a share link brings the setup back, and every tab fits a phone.
import { assert } from '../lib.mjs';

const trained = (n) => `new Promise((r) => { const L = Nooral.TL.link(); const go = () => { if (!L.ready) return setTimeout(go, 50); L.send({ cmd: 'steps', n: ${n} }); const off = L.on((m) => { if (m.type === 'state' && !m.running) { off(); r(L.step); } }); }; go(); })`;

export default function (t) {
  t.test('BPE: merges step forward and back, and the sentence gets fewer tokens', async (p) => {
    await p.route('textlab/bpe');
    await p.wait("document.querySelectorAll('#bpeStage .it-cell.tok').length > 10");
    await p.ev('Nooral.app.page.view.go(0)');
    const n0 = await p.page("P.view.result && document.querySelectorAll('#bpeStage .it-cell.tok').length");
    assert.match(await p.text('#bpeSay'), /Every character is its own token/);
    await p.click('#bpeStep');
    assert.equal(await p.page('P.view.k'), 1);
    assert.match(await p.text('#bpeSay'), /Merged “.*” and “.*” into “.*”, the most frequent pair/);
    await p.ev('Nooral.app.page.view.go(80)');
    const n80 = await p.count('#bpeStage .it-cell.tok');
    assert.ok(n80 < n0, 'fewer tokens: ' + n0 + ' → ' + n80);
    await p.click('#bpeBack');
    assert.equal(await p.page('P.view.k'), 79);
    assert.equal(await p.count('#bpeList li'), 14);
    await p.setValue('#bpeTotal', '100', 'change');
    assert.equal(await p.page('P.view.result.steps.length'), 100);
  });

  t.test('BPE on the reader’s own text', async (p) => {
    await p.route('textlab/bpe');
    await p.ev("document.querySelector('.tl-own').open = true");
    await p.setValue('#bpeOwn', 'banana bandana banana bandana cabana banana bandana cabana banana bandana');
    await p.click('#bpeUseOwn');
    await p.ev('Nooral.app.page.view.go(1)');
    assert.match(await p.text('#bpeSay'), /“a” and “n” into “an”|“n” and “a” into “na”|into/);
    await p.click('#bpeUseStories');
  });

  t.test('GPT: it trains in a worker, steps, pauses, and writes samples', async (p) => {
    await p.route('textlab/gpt');
    await p.wait("Nooral.TL.link().ready", 15000);
    assert.equal(await p.ev('Nooral.TL.link().mode'), 'worker');
    await p.click('#gptStep');
    await p.wait('Nooral.TL.link().step >= 10', 30000);
    await p.click('#gptRun');
    await p.wait('Nooral.TL.link().step >= 40', 60000);
    await p.click('#gptRun');
    await p.wait('!Nooral.TL.link().running', 10000);
    const s = await p.ev('Nooral.TL.link().step');
    await p.sleep(400);
    assert.equal(await p.ev('Nooral.TL.link().step'), s, 'paused');
    assert.match(await p.text('#gptStatus'), /parameters/);
    await p.ev(trained(160));
    await p.wait("document.querySelectorAll('#gptSamples .tl-sample').length >= 1", 20000);
    assert.ok((await p.count('#gptAttn .it-cell')) > 50, 'attention cells drawn');
    const v = await p.ev('Nooral.TL.link().val');
    assert.ok(v[v.length - 1].loss < v[0].loss - 0.5, 'validation loss falls: ' + v[0].loss + ' → ' + v[v.length - 1].loss);
  }, { timeout: 150000 });

  t.test('Sampling: greedy is the top bar; top-k cuts the rest', async (p) => {
    await p.route('textlab/sample');
    await p.wait("document.querySelectorAll('#smpBars .it-cell.tok').length === 10");
    await p.ev("Nooral.app.page.view.set({ greedy: true })");
    await p.click('#smpNext');
    await p.wait("Nooral.app.page.view.out.length === 1");
    const probs = await p.page('P.view.probs');
    const top = probs.indexOf(Math.max(...probs));
    assert.equal(await p.page('P.view.out[0]'), top, 'greedy took the most likely');
    await p.ev("Nooral.app.page.view.set({ greedy: false, topK: 3 })");
    assert.match(await p.text('#smpNote'), /^3 of \d+ characters survive/);
    await p.click('#smpGen');
    await p.wait("Nooral.app.page.view.out.length === 41", 30000);
    await p.click('#smpClear');
    await p.wait("Nooral.app.page.view.out.length === 0");
  }, { timeout: 60000 });

  t.test('a share link brings the setup back', async (p) => {
    await p.route('textlab/sample');
    await p.ev("Nooral.app.page.view.set({ topK: 7, temperature: 0.6, prompt: 'Under the bridge' })");
    const url = await p.ev("Nooral.Shell.share('textlab', Nooral.app.page.state)");
    await p.closeModals();
    await p.route('');
    await p.ev(`(() => { window.__nooralReady = false; location.hash = ${JSON.stringify(url)}.replace(/^[^#]*#/, ''); })()`);
    await p.wait("location.hash === '#textlab/sample' && window.__nooralReady === true && !!(Nooral.app.page && Nooral.app.page.state)", 10000);
    assert.equal(await p.page('P.state.sample.topK'), 7);
    assert.equal(await p.ev("document.getElementById('smpPrompt').value"), 'Under the bridge');
  });

  t.test('every tab fits a phone', async (p) => {
    for (const tab of ['bpe', 'gpt', 'sample']) { await p.route('textlab/' + tab); assert.ok(await p.fitsPhone(), tab + ': no sideways scroll at 390 px'); }
  });
}
