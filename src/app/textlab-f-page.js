// The text lab page (#textlab, #textlab/bpe, #textlab/gpt, #textlab/sample): tokens and a tiny language model.
// BPE learns merges on a text; GPT trains a small transformer on the stories; Sample writes with it, one character at
// a time, under a sampling rule. The tabs share one model (TL.link), so what was trained can be sampled.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const store = S.UI.storage;
  const TL = (S.TL = S.TL || {});
  const TABS = [['bpe', 'Byte-pair encoding', 'grammar'], ['gpt', 'Train a tiny GPT', 'run'], ['sample', 'Sampling', 'shuffle']];
  let pending = null;
  /** open the lab next with this setup ({tab, bpe, gpt, sample}): the share link, an exercise card */
  TL.openWith = (o) => { pending = clean(o); };
  const clean = (o) => { o = o && typeof o === 'object' ? o : {}; return { tab: TABS.some((t) => t[0] === o.tab) ? o.tab : null, bpe: o.bpe || {}, gpt: o.gpt || {}, sample: o.sample || {} }; };

  function open(app, main, rest) {
    const sub = String(rest || '').replace(/^textlab\/?/, '');
    const saved = S.Shell && S.Shell.autosave() ? store.get('textlab', null) : null;
    const state = clean(pending || saved);
    pending = null;
    const tabId = TABS.some((t) => t[0] === sub) ? sub : state.tab || 'bpe';
    state.tab = tabId;
    const box = h('div.tl', { id: 'tlView', dataset: { tab: tabId } });
    const share = h('button.btn', { type: 'button', id: 'tlShare', title: 'Copy a link to this setup' }, icon('link'), h('span.lbl', 'Share'));
    share.addEventListener('click', () => S.Shell && S.Shell.share('textlab', state));
    const tabs = h('div.tl-tabs', { role: 'tablist', 'aria-label': 'Text lab' }, ...TABS.map(([id, label, ic]) => h('a.tl-tab' + (id === tabId ? '.on' : ''), { href: '#textlab/' + id, role: 'tab', id: 'tlTab-' + id, 'aria-selected': String(id === tabId) }, icon(ic), h('span', label))));
    box.append(h('div.tl-head', h('div', h('h1.page-title', 'Text lab'), h('p.lead', 'How a language model reads and writes: text cut into tokens by byte-pair merges, a small transformer trained here on our own stories, and the rules that turn its probabilities into words.')), share), tabs);
    const host = h('div.tl-body', { id: 'tlBody' });
    box.appendChild(host);
    main.appendChild(box);
    const page = { state, save() { if (S.Shell && S.Shell.autosave()) store.set('textlab', state); }, reopen() { if (view && view.dispose) view.dispose(); host.innerHTML = ''; view = TL.tabs[tabId](host, page); } };
    let view = TL.tabs[tabId](host, page);
    page.save();
    return {
      get tab() { return tabId; }, get view() { return view; }, state,
      keys: [['Text lab', [['→  ←', 'Next or previous merge (BPE); the next character (Sampling)'], ['Space', 'Play or pause']]]],
      onKey(e, typing) { if (typing || e.metaKey || e.ctrlKey || e.altKey) return false; const tag = (e.target && e.target.tagName) || ''; if (tag === 'BUTTON' || tag === 'A' || tag === 'SELECT') return false; if (view && view.onKey && view.onKey(e)) { e.preventDefault(); return true; } return false; },
      redraw() { if (view && view.redraw) view.redraw(); },
      resized() { if (view && view.redraw) view.redraw(); },
      dispose() { if (view && view.dispose) view.dispose(); },
    };
  }
  S.Pages.register('textlab', { nav: 'textlab', wide: true, title: 'Text lab', open, load(data) { const s = clean(data); TL.openWith(s); return 'textlab/' + (s.tab || 'bpe'); } });

  const P = 'Nooral.app.page';
  const waitTrained = (n) => `await new Promise((r, j) => { const L = Nooral.TL.link(); const go = () => { if (!L.ready) return setTimeout(go, 50); L.send({ cmd: 'steps', n: ${n} }); const off = L.on((m) => { if (m.type === 'state' && !m.running) { off(); setTimeout(r, 200); } }); }; go(); setTimeout(() => j(new Error('training took too long')), 90000); });`;
  S.Checks.push(
    { group: 'textlab', name: 'BPE, 40 merges in', hash: 'textlab/bpe', ready: "document.querySelector('#bpeStage .it-cell')", setup: P + '.view.go(40);' },
    { group: 'textlab', name: 'BPE, the first merge', hash: 'textlab/bpe', setup: P + '.view.go(1);', ready: "document.querySelector('#bpeStage .it-cell')" },
    { group: 'textlab', name: 'GPT before training', hash: 'textlab/gpt', fresh: true, ready: "document.querySelector('#gptLoss .it-path') && document.querySelector('#gptAttn .it-cell')" },
    { group: 'textlab', name: 'GPT after 60 steps', hash: 'textlab/gpt', setup: waitTrained(60) + ' ' + P + '.view.redraw(); await new Promise((r) => setTimeout(r, 600));', ready: "document.querySelector('#gptAttn .it-cell')" },
    { group: 'textlab', name: 'sampling with top-k', hash: 'textlab/sample', setup: 'const v = ' + P + '.view; v.set({ topK: 5, temperature: 0.8 }); await v.advance(3);', ready: "document.querySelector('#smpBars .it-cell')" },
  );
})(globalThis.Nooral = globalThis.Nooral || {});
