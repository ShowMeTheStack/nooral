// The text lab's BPE tab: byte-pair merges learned one by one on a text. At each step: the pairs counted (the most
// frequent becomes the next merge), the length of the text in tokens falling as merges grow, and a sentence cut
// into tokens with the merges so far.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const TL = (S.TL = S.TL || {});
  TL.tabs = TL.tabs || {};
  const C = () => S.Kit.curve, TK = () => S.Kit.tokens;
  const show = (t) => String(t).replace(/ /g, '·').replace(/\n/g, '⏎');
  const num = (n) => Number(n).toLocaleString('en');
  const r2 = (v) => Math.round(v * 100) / 100;
  TL.SENTENCE = 'The night market opens at six, and the first vans arrive before the lights come on.';

  /** the picture at k merges */
  TL.bpeFrame = function (R, k, sentence, narrow) {
    const N = R.steps.length, items = [];
    const next = R.steps[k] || null, last = k ? R.steps[k - 1] : null;
    const top = (next || last || { top: [] }).top;
    // the pairs counted now: the most frequent is merged next
    const W = narrow ? 400 : 430, lab = 150, bw = W - lab - 64, rowH = 30;
    const cmax = Math.max(1, ...top.map((x) => x[2]));
    items.push({ key: 'bp:t', kind: 'text', x: 0, y: 0, text: next ? 'Pairs counted after ' + k + ' merge' + (k === 1 ? '' : 's') : 'The last pairs counted', size: 14, cls: 'axname' });
    top.forEach(([a, b, c], i) => {
      const y = 18 + i * rowH, win = next && i === 0;
      items.push({ key: 'bp:l' + i + ':' + a + '|' + b, kind: 'text', x: lab - 10, y: y + 18, text: show(a) + ' + ' + show(b), anchor: 'end', size: 14, mono: true, cls: win ? 'strong' : '' });
      items.push({ key: 'bp:b' + i, kind: 'cell', x: lab, y: y + 4, w: r2(Math.max(2, (bw * c) / cmax)), h: rowH - 10, rx: 3, text: '', cls: 'bar ' + (win ? '' : 'soft') });
      items.push({ key: 'bp:c' + i, kind: 'text', x: r2(lab + Math.max(2, (bw * c) / cmax) + 8), y: y + 18, text: num(c), size: narrow ? 14 : 13, cls: win ? 'strong' : 'tick' });
    });
    if (next) items.push({ key: 'bp:nx', kind: 'text', x: 0, y: 18 + top.length * rowH + 24, text: 'next: “' + show(next.pair[0]) + '” + “' + show(next.pair[1]) + '” → “' + show(next.token) + '”', size: 14, mono: true, cls: 'strong' });
    // the text's length in tokens against the merges
    const cx = narrow ? 50 : W + 110, cy = narrow ? 18 + top.length * rowH + 90 : 30, cw = narrow ? 340 : 400, chh = narrow ? 180 : 210;
    const lo = Math.min(...R.steps.map((s) => s.tokens).concat([R.start]));
    const ax = C().axes({ prefix: 'bc', x: cx, y: cy, w: cw, h: chh, xr: [0, N], yr: [Math.floor(lo * 0.9 / 1000) * 1000, Math.ceil(R.start / 1000) * 1000], xticks: C().ticks(0, N, 4), yticks: C().ticks(lo * 0.9, R.start, 3), xname: 'merges', yname: 'tokens in the text', nameSize: narrow ? 14 : 13, fmtY: (v) => (v >= 1000 ? +(v / 1000).toFixed(1) + 'k' : String(v)), tickSize: narrow ? 14 : 13 });
    items.push(...ax.items);
    const pts = [[0, R.start]].concat(R.steps.map((s, i) => [i + 1, s.tokens]));
    items.push(C().line(ax, 'bc:l', pts, 'act'));
    items.push(C().dot(ax, 'bc:d', k, k ? R.steps[k - 1].tokens : R.start, '', 5.5));
    // a sentence cut into tokens with the merges so far
    const sy = narrow ? cy + chh + 84 : 330;
    const pieces = TL.tokenizer(R.base, R.steps.slice(0, k).map((s) => s.pair)).pieces(sentence);
    items.push({ key: 'bs:t', kind: 'text', x: 0, y: sy, text: 'The sentence in ' + pieces.length + ' tokens (' + Array.from(sentence).length + ' characters)', size: 14, cls: 'axname' });
    const st = TK().strip({ prefix: 'bs', x: 0, y: sy + 14, width: narrow ? 400 : W + 110 + cw, tokens: pieces, size: 15, cls: (i) => (last && pieces[i] === last.token ? 'new' : ''), keyOf: (i) => i + ':' + pieces[i] });
    items.push(...st.items);
    const vw = narrow ? 420 : W + 120 + cw;
    return { items, view: { x: -14, y: -26, w: vw + 20, h: sy + 14 + st.h + 30 }, pieces };
  };

  TL.tabs.bpe = function (host, page) {
    const st = Object.assign({ total: 200, k: 40, own: '', sentence: TL.SENTENCE }, page.state.bpe || {});
    page.state.bpe = st;
    let R = null, timer = null;
    const text = () => (st.own && st.own.trim().length > 40 ? st.own.slice(0, 20000) : TL.corpus());
    const btn = (id, ic, label, fn, cls) => { const b = h('button.btn' + (cls ? '.' + cls : ''), { type: 'button', id, title: label }, icon(ic), h('span.lbl', label)); b.addEventListener('click', fn); return b; };
    const bBack = btn('bpeBack', 'stepBack', 'Back', () => { stop(); go(st.k - 1); });
    const bPlay = btn('bpePlay', 'run', 'Play', () => (timer ? stop() : play()), 'primary');
    const bNext = btn('bpeStep', 'stepOver', 'Next merge', () => { stop(); go(st.k + 1); });
    const range = h('input', { type: 'range', id: 'bpeK', min: 0, max: st.total, value: st.k, 'aria-label': 'Merges' });
    range.addEventListener('input', () => { stop(); go(+range.value); });
    const pos = h('span.tl-pos', { id: 'bpePos' });
    const total = h('select', { id: 'bpeTotal', 'aria-label': 'Merges to learn' }, ...[100, 200, 300, 500].map((v) => h('option', { value: v, selected: v === st.total }, v + ' merges')));
    total.addEventListener('change', () => { st.total = +total.value; train(); page.save(); });
    const sent = h('input', { type: 'text', id: 'bpeSentence', value: st.sentence, 'aria-label': 'A sentence to cut into tokens', spellcheck: 'false' });
    sent.addEventListener('change', () => { st.sentence = sent.value || TL.SENTENCE; draw(); page.save(); });
    const own = h('textarea', { id: 'bpeOwn', rows: 5, 'aria-label': 'Your own text', placeholder: 'Paste a text of your own (at least a few sentences), then press Learn from this text.' });
    own.value = st.own;
    const useOwn = btn('bpeUseOwn', 'refresh', 'Learn from this text', () => { st.own = own.value; train(); page.save(); });
    const useStories = btn('bpeUseStories', 'book', 'Back to the stories', () => { st.own = ''; own.value = ''; train(); page.save(); });
    const wrap = h('div.stage-wrap.tl-stage', { id: 'bpeStage' });
    const stage = new S.Stage(wrap, { label: 'Byte-pair merges', minText: () => (innerWidth < 600 ? 10 : 11) });
    const say = h('div.say.tl-say', { id: 'bpeSay', 'aria-live': 'polite' });
    const list = h('ol.tl-merges', { id: 'bpeList' });
    host.append(
      h('div.tl-bar', bBack, bPlay, bNext, range, pos, total),
      h('div.tl-grid', h('div.card.tl-main', wrap, say), h('aside.card.tl-side', h('h3', 'Merges so far'), list,
        h('label.tl-field', h('span', 'A sentence to cut'), sent),
        h('details.tl-own', st.own ? { open: '' } : {}, h('summary', 'Learn from your own text'), own, h('div.tl-row', useOwn, useStories)))));
    function train() {
      stop();
      try { R = TL.bpeTrain(text(), st.total); } catch (e) { S.UI.toast(e.message, 'err'); return; }
      range.max = String(R.steps.length);
      go(Math.min(st.k, R.steps.length));
    }
    function go(k) { if (!R) return; st.k = Math.max(0, Math.min(R.steps.length, k)); draw(); }
    function draw() {
      if (!R) return;
      const k = st.k, narrow = wrap.clientWidth < 560, f = TL.bpeFrame(R, k, st.sentence, narrow);
      stage.show(f, { instant: true });
      range.value = String(k);
      pos.textContent = k + ' / ' + R.steps.length;
      bBack.disabled = k === 0; bNext.disabled = k === R.steps.length;
      const tokens = k ? R.steps[k - 1].tokens : R.start, words = TL.pretokenize(text()).length;
      const next = R.steps[k], last = k ? R.steps[k - 1] : null;
      say.innerHTML = '';
      say.append(h('span.step', k ? 'Merge ' + k + ' of ' + R.steps.length : 'Before any merge'), ' ',
        last ? h('span', 'Merged “', h('span.v', show(last.pair[0])), '” and “', h('span.v', show(last.pair[1])), '” into “', h('span.v', show(last.token)), '”, the most frequent pair (', h('span.v', num(last.count)), ' times). ') : h('span', 'Every character is its own token. '),
        h('span', 'The text is now ', h('span.v', num(tokens)), ' tokens for ', h('span.v', num(words)), ' words and spaces (', h('span.v', (tokens / words).toFixed(2)), ' tokens each), with a vocabulary of ', h('span.v', String(R.base.length + k)), ' symbols.'),
        next ? h('span', ' Next comes “', h('span.v', show(next.token)), '”.') : null);
      list.innerHTML = '';
      R.steps.slice(Math.max(0, k - 14), k).forEach((s, i, arr) => list.appendChild(h('li' + (i === arr.length - 1 ? '.now' : ''), { value: Math.max(0, k - 14) + i + 1 }, h('code', show(s.pair[0])), ' + ', h('code', show(s.pair[1])), ' → ', h('code', show(s.token)), h('span.muted', ' ' + num(s.count)))));
      if (!k) list.appendChild(h('li.muted', { style: { listStyle: 'none' } }, 'No merges yet.'));
    }
    function play() { if (st.k >= R.steps.length) go(0); timer = setInterval(() => { if (st.k >= R.steps.length) stop(); else go(st.k + 1); }, 450); bPlay.replaceChildren(icon('pause'), h('span.lbl', 'Pause')); }
    function stop() { clearInterval(timer); timer = null; bPlay.replaceChildren(icon('run'), h('span.lbl', 'Play')); page.save(); }
    train();
    return { go, play, stop, get k() { return st.k; }, get result() { return R; }, redraw: draw, dispose() { clearInterval(timer); },
      onKey(e) { if (e.key === 'ArrowRight') { bNext.click(); return true; } if (e.key === 'ArrowLeft') { bBack.click(); return true; } if (e.key === ' ') { bPlay.click(); return true; } return false; } };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
