// The text lab's GPT tab: a tiny decoder-only transformer trained here, character by character, on the original
// stories. Play, pause or step; the training and validation loss fall (perplexity is e to the loss), a sample is
// written every 150 steps so the text can be seen improving, and the attention of each head is drawn on a prompt.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const TL = (S.TL = S.TL || {});
  TL.tabs = TL.tabs || {};
  const C = () => S.Kit.curve, HM = () => S.Kit.heat;
  const show = (t) => String(t).replace(/ /g, '·').replace(/\n/g, '⏎');
  TL.ATT_PROMPT = 'the lights come on';

  /** a moving average of the training loss, so the curve reads through the noise of small batches */
  const smooth = (hs, k) => hs.map((x, i) => { const a = hs.slice(Math.max(0, i - k + 1), i + 1); return [x.step, a.reduce((s, y) => s + y.loss, 0) / a.length]; });
  TL.lossFrame = function (L, narrow) {
    const W = narrow ? 330 : 560, H = narrow ? 190 : 230, n = Math.max(100, L.step), top = Math.max(1, (L.val[0] && L.val[0].loss) || Math.log((L.info && L.info.vocab.length) || 60)) * 1.05;
    const ax = C().axes({ prefix: 'gl', x: 0, y: 0, w: W, h: H, xr: [0, n], yr: [0, top], xticks: C().ticks(0, n, narrow ? 3 : 5), yticks: C().ticks(0, top, 4), xname: 'training steps', yname: 'loss (nats a character)', tickSize: 13 });
    const items = ax.items.slice();
    items.push(C().line(ax, 'gl:tr', smooth(L.history, 20), 'c1'));
    items.push(C().line(ax, 'gl:va', L.val.map((x) => [x.step, x.loss]), 'c0'));
    for (const [i, v] of L.val.entries()) if (i === L.val.length - 1) items.push(C().dot(ax, 'gl:vd', v.step, v.loss, 'c0', 4.5));
    items.push(...C().legend('gl:lg', narrow ? 0 : W - 230, narrow ? H + 62 : -16, [{ text: 'training', cls: 'c1' }, { text: 'validation', cls: 'c0' }], { size: 13 }));
    return { items, view: { x: -48, y: -40, w: W + 64, h: H + (narrow ? 112 : 86) } };
  };
  TL.attnFrame = function (A, chars, head, narrow) {
    const n = chars.length, cw = narrow ? 17 : 26;
    const g = HM().grid({ prefix: 'ga', x: 0, y: 0, cw, ch: cw, gap: 1, vals: A[head], scheme: 'seq', max: 1, fmt: null, rows: chars.map(show), cols: chars.map(show), labelSize: 13 });
    const items = g.items.slice();
    items.push({ key: 'ga:q', kind: 'text', x: -34, y: n * (cw + 1) + 30, text: 'each row: where that character looks', size: 13, cls: 'axname' });
    return { items, view: { x: -40, y: -30, w: n * (cw + 1) + 50, h: n * (cw + 1) + 70 } };
  };

  /** one sentence: which size and backend the model got, and why the graphics card was not used */
  TL.sizeSay = function (info, mode) {
    const c = info.cfg, gpu = info.backend === 'webgpu';
    const what = info.size === 'webgpu' ? 'The GPU size' : info.size === 'cpu' ? 'The CPU size' : info.size === 'small' ? 'The small size' : 'A custom size';
    return (info.auto ? 'Auto chose: ' : '') + what + ' (' + c.layers + ' layers, width ' + c.d + ', ' + c.heads + ' heads, context ' + c.ctx + '), ' + info.params.toLocaleString('en') + ' parameters, ' + info.vocab.length + ' characters in the vocabulary, trained ' +
      (gpu ? 'on the graphics card (WebGPU)' : 'on the CPU') + (mode === 'worker' ? ' in a worker' : ' on the page') + '; the learning rate falls over ' + c.steps + ' planned steps.' + (!gpu && info.why ? ' No WebGPU here: ' + info.why + '.' : '');
  };

  TL.tabs.gpt = function (host, page) {
    const st = Object.assign({ size: 'auto', prompt: TL.ATT_PROMPT, layer: 0, head: 0 }, page.state.gpt || {});
    if (!TL.SIZE_NAMES.includes(st.size)) st.size = 'auto';
    delete st.layers; delete st.d; delete st.ctx;
    page.state.gpt = st;
    const L = TL.link();
    const btn = (id, ic, label, fn, cls) => { const b = h('button.btn' + (cls ? '.' + cls : ''), { type: 'button', id, title: label }, icon(ic), h('span.lbl', label)); b.addEventListener('click', fn); return b; };
    const bRun = btn('gptRun', 'run', 'Train', () => { if (L.running) L.send({ cmd: 'pause' }); else { if (!L.ready) start(); L.send({ cmd: 'run' }); L.running = true; } upd(); }, 'primary');
    const bStep = btn('gptStep', 'stepOver', '10 steps', () => { if (!L.ready) start(); L.send({ cmd: 'pause' }); L.send({ cmd: 'steps', n: 10 }); });
    const bReset = btn('gptReset', 'restart', 'New model', () => { start(); upd(); });
    const sel = (id, label, opts, v, fn) => { const s = h('select', { id, 'aria-label': label }, ...opts.map(([x, l]) => h('option', { value: x, selected: String(x) === String(v) }, l))); s.addEventListener('change', () => fn(s.value)); return h('label.tl-field.inline', h('span', label), s); };
    const sizeName = (k) => { const z = TL.modelSize(k); return z.layers + ' layers, width ' + z.d + ', context ' + z.ctx; };
    const SIZES = [['auto', 'Auto: the GPU size with WebGPU, else the CPU size'], ['webgpu', 'GPU size (' + sizeName('webgpu') + ')'], ['cpu', 'CPU size (' + sizeName('cpu') + ')'], ['small', 'Small and quick (' + sizeName('small') + ')']];
    const layerSel = h('select', { id: 'gptLayer', 'aria-label': 'Layer' });
    layerSel.addEventListener('change', () => { st.layer = +layerSel.value; attn(); });
    const fillLayers = () => { const n = (L.cfg && L.cfg.layers) || 1; st.layer = Math.min(st.layer, n - 1); layerSel.replaceChildren(...Array.from({ length: n }, (_, i) => h('option', { value: i, selected: i === st.layer }, String(i + 1)))); };
    const status = h('div.tl-status', { id: 'gptStatus', 'aria-live': 'polite' });
    const lossWrap = h('div.stage-wrap.tl-loss', { id: 'gptLoss' });
    const lossStage = new S.Stage(lossWrap, { label: 'Training and validation loss', minText: () => (innerWidth < 600 ? 10 : 11) });
    const samples = h('div.tl-samples', { id: 'gptSamples' });
    const prompt = h('input', { type: 'text', id: 'gptPrompt', value: st.prompt, maxlength: 24, 'aria-label': 'A prompt to see the attention on', spellcheck: 'false' });
    prompt.addEventListener('change', () => { st.prompt = prompt.value.slice(0, 24) || TL.ATT_PROMPT; page.save(); attn(); });
    const heads = h('div.seg', { id: 'gptHeads', role: 'group', 'aria-label': 'Head' });
    const attWrap = h('div.stage-wrap.tl-att', { id: 'gptAttn' });
    const attStage = new S.Stage(attWrap, { label: 'Attention of one head', minText: () => (innerWidth < 600 ? 10 : 11) });
    const attNote = h('p.muted.tl-note', { id: 'gptAttNote' });
    host.append(
      h('div.tl-bar', bRun, bStep, bReset,
        sel('gptSize', 'Model', SIZES, st.size, (v) => { st.size = v; start(); upd(); })),
      status,
      h('div.tl-grid.gpt', h('div.card.tl-main', h('h3', 'Loss'), lossWrap), h('div.card.tl-side', h('h3', 'What it writes'), samples)),
      h('div.card.tl-attcard', h('div.tl-attpic', h('h3', 'Attention'), attWrap), h('div.tl-attside', h('label.tl-field', h('span', 'Prompt (up to 24 characters)'), prompt),
        h('div.tl-row', h('label.tl-field.inline', h('span', 'Layer'), layerSel), heads), attNote)));
    function start() {
      L.init({ size: st.size }, TL.trainText());
      page.save();
    }
    if (!L.ready && !L.cfg) start();
    fillLayers();
    let lastAttn = -1;
    function attn() {
      if (!L.ready) return;
      const text = (st.prompt || TL.ATT_PROMPT).slice(0, Math.min(24, L.cfg.ctx));
      const ids = Array.from(text, (c) => Math.max(0, L.info.vocab.indexOf(c)));
      lastAttn = L.step;
      L.ask('attention', { ids, layer: st.layer }).then((A) => {
        heads.innerHTML = '';
        A.forEach((_, i) => { const b = h('button' + (i === st.head ? '.on' : ''), { type: 'button', 'aria-pressed': String(i === st.head) }, 'head ' + (i + 1)); b.addEventListener('click', () => { st.head = i; attn(); }); heads.appendChild(b); });
        st.head = Math.min(st.head, A.length - 1);
        attStage.show(TL.attnFrame(A, Array.from(text), st.head, attWrap.clientWidth < 520), { instant: true });
        const row = A[st.head][text.length - 1], j = row.indexOf(Math.max(...row));
        attNote.textContent = 'Layer ' + (st.layer + 1) + ', head ' + (st.head + 1) + ' after ' + L.step + ' steps. Above the diagonal is empty: a character may only look back. The last character, “' + show(text[text.length - 1]) + '”, looks most at “' + show(text[j]) + '” (weight ' + row[j].toFixed(2) + ').';
      }).catch(() => {});
    }
    function upd() {
      bRun.replaceChildren(icon(L.running ? 'pause' : 'run'), h('span.lbl', L.running ? 'Pause' : L.step ? 'Train on' : 'Train'));
      const tr = L.history.length ? L.history.slice(-20).reduce((s, x) => s + x.loss, 0) / Math.min(20, L.history.length) : null, va = L.val.length ? L.val[L.val.length - 1].loss : null;
      status.innerHTML = '';
      if (L.error) status.append(h('b.bad', 'Training stopped: ' + L.error));
      else if (!L.ready) status.append(h('span', 'Building the model…'));
      else status.append(h('span', TL.sizeSay(L.info, L.mode) + ' '),
        h('span', 'Step ', h('b', String(L.step)), va !== null ? [' · validation loss ', h('b', va.toFixed(3)), ' (perplexity ', h('b', Math.exp(va).toFixed(1)), ': as unsure as a choice among that many characters)'] : '', tr !== null ? [' · training loss ', h('b', tr.toFixed(3))] : ''));
      if (L.ready) lossStage.show(TL.lossFrame(L, lossWrap.clientWidth < 520), { instant: true });
      samples.innerHTML = '';
      if (!L.samples.length) samples.append(h('p.muted', 'Every 150 steps the model writes 120 characters after “The ” (temperature 0.8). Before training they are noise; watch words appear.'));
      for (const s of L.samples.slice(0, 6)) samples.append(h('div.tl-sample', h('span.tl-at', 'step ' + s.step), h('code', h('b', s.prompt), s.text)));
    }
    let last = 0, pend = null;
    const off = L.on((m) => {
      if (m.type === 'ready') { fillLayers(); upd(); attn(); return; }
      if (m.type === 'state' && !m.running) { upd(); attn(); return; }
      const now = performance.now();
      if (now - last > 300) { last = now; upd(); if (L.step - lastAttn >= 100) attn(); } else if (!pend) pend = setTimeout(() => { pend = null; last = performance.now(); upd(); }, 320);
    });
    upd(); attn();
    return { redraw() { upd(); attn(); }, dispose() { off(); clearTimeout(pend); if (L.running) L.send({ cmd: 'pause' }); },
      onKey(e) { if (e.key === ' ') { bRun.click(); return true; } return false; } };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
