// The text lab's sampling tab: a prompt, then the next character chosen one at a time. The bars show the model's
// ten most likely next characters before and after the sampling rule: a temperature reshapes them, top-k keeps the
// k most likely, top-p keeps the fewest whose probabilities add to p, and what survives is renormalised.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const TL = (S.TL = S.TL || {});
  TL.tabs = TL.tabs || {};
  const show = (t) => (t === ' ' ? '·' : t === '\n' ? '⏎' : t);
  const r2 = (v) => Math.round(v * 100) / 100;

  /** the bars: for the ten most likely characters, the model's probability and the one after the rule */
  TL.barsFrame = function (probs, cut, vocab, chosen, narrow) {
    const items = [], top = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a] || a - b).slice(0, 10);
    const W = narrow ? 170 : 460, rowH = 34, lab = 50, max = Math.max(...top.map((i) => Math.max(probs[i], cut.probs[i])), 1e-9);
    items.push({ key: 'sb:h1', kind: 'text', x: lab, y: -12, text: 'the model, then the rule', size: 13, cls: 'axname' });
        top.forEach((i, r) => {
      const y = r * rowH, kept = cut.kept[i], isC = i === chosen;
      items.push({ key: 'sb:t' + i, kind: 'cell', x: 0, y: y + 2, w: 36, h: rowH - 6, rx: 5, text: show(vocab[i]), size: 15, cls: 'tok mono ' + (isC ? 'new' : kept ? '' : 'soft') });
      const w1 = r2(Math.max(1.5, (W * probs[i]) / max)), w2 = r2(Math.max(kept ? 1.5 : 0, (W * cut.probs[i]) / max));
      items.push({ key: 'sb:p' + i, kind: 'cell', x: lab, y: y + 4, w: w1, h: 10, rx: 2, text: '', cls: 'bar soft' });
      if (kept) items.push({ key: 'sb:q' + i, kind: 'cell', x: lab, y: y + 16, w: w2, h: 12, rx: 2, text: '', cls: 'bar' + (isC ? ' c1' : '') });
      items.push({ key: 'sb:v' + i, kind: 'text', x: lab + W + 14, y: y + 23, text: (probs[i] * 100).toFixed(1) + '% → ' + (kept ? (cut.probs[i] * 100).toFixed(1) + '%' : 'cut'), size: 13, cls: kept ? (isC ? 'strong' : '') : 'tick' });
    });
    return { items, view: { x: -10, y: -34, w: lab + W + 170, h: 10 * rowH + 44 } };
  };

  TL.tabs.sample = function (host, page) {
    const st = Object.assign({ prompt: 'At six o’clock the', greedy: false, temperature: 1, topK: 0, topP: 1, seed: 1 }, page.state.sample || {});
    page.state.sample = st;
    const L = TL.link();
    let bigram = null, out = [], probs = null, chosen = -1, busy = false, n = 0;
    const useGPT = () => L.ready && L.step > 0;
    const model = () => (useGPT() ? null : (bigram = bigram || TL.Bigram(TL.corpus())));
    const vocab = () => (useGPT() ? L.info.vocab : model().tok.vocab);
    const enc = (s) => { const v = vocab(); return Array.from(s, (c) => Math.max(0, v.indexOf(c) >= 0 ? v.indexOf(c) : v.indexOf(' '))); };
    const rule = () => ({ greedy: st.greedy, temperature: st.temperature, topK: st.topK, topP: st.topP });
    const nextProbs = (ids) => (useGPT() ? L.ask('next', { ids }) : Promise.resolve(model().next(ids)));
    const btn = (id, ic, label, fn, cls) => { const b = h('button.btn' + (cls ? '.' + cls : ''), { type: 'button', id, title: label }, icon(ic), h('span.lbl', label)); b.addEventListener('click', fn); return b; };
    const which = h('div.tl-status', { id: 'smpModel' });
    const prompt = h('input', { type: 'text', id: 'smpPrompt', value: st.prompt, 'aria-label': 'Prompt', spellcheck: 'false' });
    prompt.addEventListener('change', () => { st.prompt = prompt.value; out = []; page.save(); look(); });
    const bNext = btn('smpNext', 'stepOver', 'Next character', () => advance(1), 'primary');
    const bGen = btn('smpGen', 'run', 'Write 40', () => advance(40));
    const bClear = btn('smpClear', 'restart', 'Start again', () => { out = []; look(); });
    const mode = h('div.seg', { role: 'group', 'aria-label': 'Rule', id: 'smpMode' });
    const sliders = [];
    const slider = (id, label, min, max, step, key, fmt) => { const o = h('output', fmt(st[key])), r = h('input', { type: 'range', id, min, max, step, value: st[key], 'aria-label': label }); r.addEventListener('input', () => { st[key] = +r.value; o.textContent = fmt(st[key]); redraw(); page.save(); }); sliders.push(() => { r.value = st[key]; o.textContent = fmt(st[key]); }); return h('label.pg-slider.tl-slider', h('span', label, o), r); };
    const sT = slider('smpTemp', 'Temperature', 0.1, 2, 0.05, 'temperature', (v) => v.toFixed(2));
    const sK = slider('smpTopK', 'Top-k (0: off)', 0, 20, 1, 'topK', (v) => (v ? String(v) : 'off'));
    const sP = slider('smpTopP', 'Top-p (1: off)', 0.05, 1, 0.05, 'topP', (v) => (v >= 1 ? 'off' : v.toFixed(2)));
    const textEl = h('div.tl-text', { id: 'smpText', 'aria-live': 'polite' });
    const wrap = h('div.stage-wrap.tl-bars', { id: 'smpBars' });
    const stage = new S.Stage(wrap, { label: 'The next character: probabilities before and after the rule', minText: () => (innerWidth < 600 ? 10 : 11) });
    const note = h('p.muted.tl-note', { id: 'smpNote' });
    host.append(which, h('div.tl-grid',
      h('div.card.tl-main', h('label.tl-field', h('span', 'Prompt'), prompt), h('div.tl-bar.plain', bNext, bGen, bClear), textEl, h('h3', 'The next character'), wrap, note),
      h('aside.card.tl-side', h('h3', 'The rule'), mode, sT, sK, sP, h('p.muted.tl-note', 'Temperature divides the model’s scores before the softmax: below 1 the likely get likelier, above 1 the choice flattens. Top-k and top-p then cut the unlikely tail, and what is left is scaled back up to add to 1.'))));
    const fillMode = () => {
      mode.innerHTML = '';
      for (const [g, label] of [[true, 'Greedy'], [false, 'Sample']]) { const b = h('button' + (st.greedy === g ? '.on' : ''), { type: 'button', 'aria-pressed': String(st.greedy === g), dataset: { greedy: String(g) } }, label); b.addEventListener('click', () => { st.greedy = g; fillMode(); redraw(); page.save(); }); mode.appendChild(b); }
      for (const el of [sT, sK, sP]) el.classList.toggle('off', st.greedy);
    };
    function redraw() {
      which.innerHTML = '';
      which.append(useGPT() ? h('span', 'Using the transformer trained in the GPT tab (', h('b', 'step ' + L.step), ').') : h('span', 'Using a bigram model counted from the stories: each character depends only on the one before. ', h('a', { href: '#textlab/gpt' }, 'Train the transformer'), ' to sample from it instead.'));
      textEl.innerHTML = '';
      textEl.append(h('span.tl-prompt', st.prompt), h('span.tl-gen', TL.decodeWith(vocab(), out)), h('span.tl-caret', '▍'));
      if (!probs) return;
      const cut = TL.cut(probs, rule());
      stage.show(TL.barsFrame(probs, cut, vocab(), chosen, wrap.clientWidth < 480), { instant: true });
      const kept = cut.kept.filter(Boolean).length;
      note.textContent = st.greedy ? 'Greedy always takes the most likely character, so the same prompt always gives the same text.' : kept + ' of ' + probs.length + ' characters survive the rule' + (chosen >= 0 ? '; the draw (seed ' + st.seed + ', draw ' + n + ') took “' + show(vocab()[chosen]) + '”.' : '.');
    }
    function look() { const ids = enc(st.prompt).concat(out); nextProbs(ids).then((p) => { probs = p; chosen = -1; redraw(); }); }
    async function advance(k) {
      if (busy) return;
      busy = true; bNext.disabled = bGen.disabled = true;
      try {
        for (let i = 0; i < k; i++) {
          const ids = enc(st.prompt).concat(out);
          probs = await nextProbs(ids);
          const cut = TL.cut(probs, rule());
          n++;
          chosen = TL.pick(cut.probs, S.ML.rng(st.seed * 7919 + n));
          redraw();
          out.push(chosen);
          if (k > 1) await new Promise((r) => setTimeout(r, 60));
        }
      } finally { busy = false; bNext.disabled = bGen.disabled = false; }
    }
    fillMode(); look();
    /** change the rule (or the prompt) from outside: the controls follow */
    const set = (o) => { Object.assign(st, o || {}); prompt.value = st.prompt; for (const f of sliders) f(); fillMode(); redraw(); page.save(); };
    return { advance, redraw, set, get out() { return out; }, get probs() { return probs; }, dispose() {},
      onKey(e) { if (e.key === 'ArrowRight') { bNext.click(); return true; } return false; } };
  };
  TL.decodeWith = (vocab, ids) => ids.map((i) => vocab[i] || '').join('');
})(globalThis.Nooral = globalThis.Nooral || {});
