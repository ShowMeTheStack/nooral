// The playground page (#playground): a small network on 2-D data. Pick the data and the inputs, add layers and
// neurons, choose the activation, learning rate, regularisation and batch size, and press play: every neuron's map
// changes as it learns, the weights thicken and change colour, and the boundary and the loss curves move. Training
// runs on the page in short slices, so the page never stops answering.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const store = S.UI.storage;
  const PG = (S.PG = S.PG || {});
  let pending = null;
  if (S.UI.addIcons) S.UI.addIcons({ minus: '<path d="M5 10h10"/>' });
  const MAX_LAYERS = 6, MAX_N = 8;
  const clone = (o) => JSON.parse(JSON.stringify(o));
  /** a setup checked and filled in, from a share link or storage */
  PG.clean = function (c) {
    const d = PG.defaults(), o = Object.assign(d, c || {});
    if (!PG.DATASETS.some(([id]) => id === o.data)) o.data = d.data;
    o.features = (Array.isArray(o.features) ? o.features : d.features).filter((f) => PG.FEATURES.some((x) => x.id === f));
    if (!o.features.length) o.features = ['x1', 'x2'];
    o.layers = (Array.isArray(o.layers) ? o.layers : d.layers).slice(0, MAX_LAYERS).map((n) => Math.max(1, Math.min(MAX_N, n | 0)));
    if (!PG.ACTS[o.act]) o.act = d.act;
    if (!['none', 'l1', 'l2'].includes(o.reg)) o.reg = 'none';
    if (!['sgd', 'momentum', 'adam'].includes(o.optimizer)) o.optimizer = 'sgd';
    for (const [k, lo, hi] of [['n', 50, 600], ['noise', 0, 0.5], ['testShare', 0.1, 0.9], ['lr', 1e-5, 10], ['regRate', 0, 10], ['batch', 1, 300], ['seed', 1, 99999]]) { const v = +o[k]; o[k] = Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : d[k]; }
    o.seed = Math.round(o.seed); o.n = Math.round(o.n); o.batch = Math.round(o.batch);
    return o;
  };
  PG.preset = (cfg) => { pending = PG.clean(cfg); };

  function open(app, main) {
    let cfg = PG.clean(pending || (S.Shell && S.Shell.autosave() ? store.get('playground', null) : null));
    pending = null;
    let lab = null, running = false, raf = 0, lastDraw = 0, sel = { neuron: null, weight: null }, disposed = false, showTest = true;
    const box = h('div.pg', { id: 'pgView' });
    // ---- the controls along the top ----
    const btn = (id, ic, label, fn, cls) => { const b = h('button.btn' + (cls ? '.' + cls : ''), { type: 'button', id, title: label }, icon(ic), h('span.lbl', label)); b.addEventListener('click', fn); return b; };
    const bPlay = btn('pgPlay', 'run', 'Play', () => (running ? pause() : play()), 'primary');
    const bStep = btn('pgStep', 'stepOver', 'One epoch', () => { pause(); lab.epochStep(); draw(true); });
    const bReset = btn('pgReset', 'restart', 'Reset', () => { pause(); rebuild(); });
    const bShare = btn('pgShare', 'link', 'Share', () => S.Shell && S.Shell.share('playground', cfg));
    const epochEl = h('span.pg-epoch', { id: 'pgEpoch', 'aria-live': 'off' });
    const sel_ = (id, label, opts, val, fn) => { const s = h('select', { id, 'aria-label': label }, ...opts.map(([v, l]) => h('option', { value: String(v), selected: String(v) === String(val) }, l))); s.addEventListener('change', () => fn(s.value)); return h('label.pg-field', h('span', label), s); };
    const fmtN = (v) => String(v);
    const top = h('div.pg-top.card');
    const fill = () => {
      top.innerHTML = '';
      top.append(h('div.pg-run', bPlay, bStep, bReset, epochEl),
        sel_('pgLr', 'Learning rate', PG.LRS.map((v) => [v, fmtN(v)]), cfg.lr, (v) => { cfg.lr = +v; if (lab) lab.setLr(cfg.lr); save(); }),
        sel_('pgOpt', 'Optimiser', [['sgd', 'SGD'], ['momentum', 'Momentum'], ['adam', 'Adam']], cfg.optimizer, (v) => { cfg.optimizer = v; changed(); }),
        sel_('pgAct', 'Activation', [['relu', 'ReLU'], ['tanh', 'tanh'], ['sigmoid', 'sigmoid'], ['linear', 'linear']], cfg.act, (v) => { cfg.act = v; changed(); }),
        sel_('pgReg', 'Regularisation', [['none', 'none'], ['l1', 'L1'], ['l2', 'L2']], cfg.reg, (v) => { cfg.reg = v; if (v !== 'none' && !cfg.regRate) { cfg.regRate = 0.001; fill(); } save(); }),
        sel_('pgRate', 'Its rate', PG.RATES.map((v) => [v, fmtN(v)]), cfg.regRate, (v) => { cfg.regRate = +v; save(); }),
        sel_('pgBatch', 'Batch size', [1, 5, 10, 20, 30, 50, 100].map((v) => [v, String(v)]), cfg.batch, (v) => { cfg.batch = +v; changed(); }),
        bShare);
    };
    // ---- data ----
    const dataCard = h('aside.pg-data.card', { id: 'pgData' });
    const slider = (id, label, min, max, step, val, show, fn) => {
      const out = h('output', show(val)), r = h('input', { type: 'range', id, min, max, step, value: val, 'aria-label': label });
      r.addEventListener('input', () => { out.textContent = show(+r.value); });
      r.addEventListener('change', () => fn(+r.value));
      return h('label.pg-slider', h('span', label, out), r);
    };
    const fillData = () => {
      dataCard.innerHTML = '';
      const chips = h('div.pg-sets', { role: 'group', 'aria-label': 'Dataset' });
      for (const [id, label] of PG.DATASETS) {
        const c = h('button.chip' + (cfg.data === id ? '.on' : ''), { type: 'button', dataset: { set: id }, 'aria-pressed': String(cfg.data === id) }, label);
        c.addEventListener('click', () => { cfg.data = id; fillData(); changed(); });
        chips.appendChild(c);
      }
      const seed = h('input', { type: 'number', id: 'pgSeed', min: 1, max: 99999, value: cfg.seed, 'aria-label': 'Seed', style: { width: '84px' } });
      seed.addEventListener('change', () => { cfg.seed = Math.max(1, Math.round(+seed.value || 1)); changed(); });
      const dice = h('button.icon-btn', { type: 'button', title: 'Another seed: new points and new starting weights', 'aria-label': 'Another seed' }, icon('shuffle'));
      dice.addEventListener('click', () => { cfg.seed = 1 + Math.floor(Math.random() * 9999); fillData(); changed(); });
      dataCard.append(h('h3', 'Data'), chips,
        slider('pgNoise', 'Noise', 0, 0.5, 0.05, cfg.noise, (v) => v.toFixed(2), (v) => { cfg.noise = v; changed(); }),
        slider('pgN', 'Points', 50, 600, 50, cfg.n, (v) => String(v), (v) => { cfg.n = v; changed(); }),
        slider('pgTest', 'Held out for testing', 0.1, 0.9, 0.1, cfg.testShare, (v) => Math.round(v * 100) + '%', (v) => { cfg.testShare = v; changed(); }),
        h('label.pg-seed', h('span', 'Seed'), seed, dice),
        h('p.muted.pg-note', 'Click an input in the network to switch it on or off. Filled dots are training points, hollow ones are held out for testing.'));
    };
    // ---- the network ----
    const netCard = h('section.pg-net.card');
    const layersRow = h('div.pg-layers', { id: 'pgLayers' });
    const netWrap = h('div.stage-wrap.pg-stage', { id: 'pgNetStage' });
    const read = h('p.pg-read', { id: 'pgRead', 'aria-live': 'polite' });
    netCard.append(h('div.pg-head', h('h3', 'Network'), layersRow), netWrap, read);
    const netStage = new S.Stage(netWrap, { label: 'The network: every neuron a map of the plane', minText: () => (innerWidth < 600 ? 10 : 11) });
    const fillLayers = () => {
      layersRow.innerHTML = '';
      const mini = (ic, label, fn, dis, id) => { const b = h('button.icon-btn.pg-pm', { type: 'button', title: label, 'aria-label': label, disabled: !!dis, id }, icon(ic)); b.addEventListener('click', fn); return b; };
      layersRow.append(mini('minus', 'Remove the last hidden layer', () => { cfg.layers.pop(); changed(); }, !cfg.layers.length, 'pgLayerMinus'),
        h('span.pg-count', { id: 'pgLayerCount' }, cfg.layers.length + ' hidden layer' + (cfg.layers.length === 1 ? '' : 's')),
        mini('plus', 'Add a hidden layer', () => { cfg.layers.push(cfg.layers[cfg.layers.length - 1] || 4); changed(); }, cfg.layers.length >= MAX_LAYERS, 'pgLayerPlus'));
      cfg.layers.forEach((n, l) => layersRow.append(h('span.pg-layer', { dataset: { layer: String(l) } }, 'layer ' + (l + 1) + ':',
        mini('minus', 'One neuron fewer in layer ' + (l + 1), () => { cfg.layers[l]--; if (!cfg.layers[l]) cfg.layers.splice(l, 1); changed(); }, false, 'pgMinus' + l), h('b', String(n)),
        mini('plus', 'One more neuron in layer ' + (l + 1), () => { cfg.layers[l]++; changed(); }, n >= MAX_N, 'pgPlus' + l))));
    };
    netStage.onPick = (key) => {
      if (key.startsWith('pg:f:')) { const id = key.slice(5), on = new Set(cfg.features); if (on.has(id)) { if (on.size > 1) on.delete(id); } else on.add(id); cfg.features = PG.FEATURES.map((f) => f.id).filter((f) => on.has(f)); changed(); return; }
      if (key.startsWith('pg:w')) { sel.weight = sel.weight === key.slice(4) ? null : key.slice(4); draw(true); return; }
      if (key.startsWith('pg:n') || key === 'pg:out') { sel.neuron = sel.neuron === key ? null : key; draw(true); }
    };
    // ---- the output ----
    const outCard = h('aside.pg-out.card');
    const lossEl = h('div.pg-loss', { id: 'pgLoss' });
    const plotWrap = h('div.stage-wrap.pg-plot', { id: 'pgPlot' }), curveWrap = h('div.stage-wrap.pg-curve', { id: 'pgCurve' });
    const plotStage = new S.Stage(plotWrap, { label: 'What the network says over the plane', minText: () => (innerWidth < 600 ? 10 : 11) }), curveStage = new S.Stage(curveWrap, { label: 'Loss against epochs', minText: () => (innerWidth < 600 ? 10 : 11) });
    const whatEl = h('p.muted.pg-note', { id: 'pgWhat' });
    const testBox = h('input', { type: 'checkbox', checked: true, id: 'pgShowTest' });
    testBox.addEventListener('change', () => { showTest = testBox.checked; draw(true); });
    outCard.append(h('h3', 'Output'), lossEl, plotWrap, whatEl, h('label.pg-check', testBox, ' Show the test points'), curveWrap);

    box.append(top, h('div.pg-cols', dataCard, netCard, outCard));
    main.appendChild(box);

    function save() { if (S.Shell && S.Shell.autosave()) store.set('playground', cfg); }
    function rebuild() {
      try { lab = PG.make(cfg); } catch (e) { S.UI.toast(e.message, 'err'); return; }
      sel = { neuron: null, weight: null };
      draw(true);
    }
    function changed() { const was = running; pause(); save(); fillLayers(); rebuild(); if (was) play(); }
    function draw(force) {
      if (!lab || disposed) return;
      const now = performance.now();
      if (!force && now - lastDraw < 140) return;
      lastDraw = now;
      const wk = lab.weights();
      const nf = PG.netFrame(lab, wk, Object.assign({ narrow: netWrap.clientWidth < 560 }, sel));
      netStage.show(nf, { instant: true });
      const narrow = innerWidth < 700;
      plotStage.show(PG.plotFrame(lab, wk, { neuron: sel.neuron, n: narrow ? 48 : 64, showTest }), { instant: true });
      curveStage.show(PG.curveFrame(lab, {}), { instant: true });
      const L = lab.history[lab.history.length - 1];
      epochEl.textContent = 'Epoch ' + String(lab.epoch).padStart(3, '0');
      const acc = lab.losses(wk);
      lossEl.innerHTML = '';
      lossEl.append(h('span', 'Test loss ', h('b', L.test.toFixed(3))), h('span', 'Training loss ', h('b', L.train.toFixed(3))), h('span.muted', Math.round(acc.testAcc * 100) + '% of test points right'));
      whatEl.textContent = sel.neuron && sel.neuron !== 'pg:out' ? 'Shown: neuron ' + sel.neuron.slice(4).replace('_', '.').replace(/\.(\d+)$/, (m, i) => '.' + (+i + 1)) + ' over the plane (blue positive, vermilion negative). Click it again for the output.' : 'Shown: the network’s probability of orange over the plane, and its boundary p = 0.5. Click a neuron to see its own map here.';
      if (sel.weight) { const [l, i, j] = sel.weight.split('_').map(Number); const w = wk.W[l][i][j]; const from = l === 0 ? lab.feats[j].label : 'neuron ' + l + '.' + (j + 1), to = l + 1 === wk.W.length ? 'the output' : 'neuron ' + (l + 1) + '.' + (i + 1); read.textContent = 'Weight from ' + from + ' to ' + to + ': ' + PG.fmtW(w) + (w >= 0 ? ' (blue: it pushes the next neuron up)' : ' (vermilion: it pushes the next neuron down)') + '.'; }
      else read.textContent = lab.numParams() + ' weights and biases. Line colour is a weight’s sign, thickness its size; click a line to read it, or a neuron to see its map on the right.';
    }
    function tick() {
      if (!running || disposed) return;
      const t0 = performance.now();
      try { while (performance.now() - t0 < 14) lab.step(); } catch (e) { pause(); S.UI.toast('Training stopped: ' + e.message, 'err'); return; }
      draw(false);
      raf = requestAnimationFrame(tick);
    }
    function play() { if (running || !lab) return; running = true; bPlay.replaceChildren(icon('pause'), h('span.lbl', 'Pause')); bPlay.title = 'Pause'; raf = requestAnimationFrame(tick); }
    function pause() { running = false; cancelAnimationFrame(raf); bPlay.replaceChildren(icon('run'), h('span.lbl', 'Play')); bPlay.title = 'Play'; draw(true); }
    fill(); fillData(); fillLayers(); rebuild();
    return {
      get lab() { return lab; }, get cfg() { return cfg; }, get running() { return running; },
      play, pause, set(c) { cfg = PG.clean(Object.assign({}, cfg, c)); fill(); fillData(); fillLayers(); rebuild(); save(); },
      train(epochs) { pause(); for (let e = 0; e < epochs; e++) lab.epochStep(); draw(true); },
      pick(key) { netStage.onPick(key); },
      keys: [['Playground', [['Space', 'Play or pause'], ['→', 'One epoch'], ['R', 'Reset the weights']]]],
      onKey(e, typing) {
        if (typing || e.metaKey || e.ctrlKey || e.altKey) return false;
        const tag = (e.target && e.target.tagName) || '';
        if (e.key === ' ' && tag !== 'BUTTON') { e.preventDefault(); bPlay.click(); return true; }
        if (e.key === 'ArrowRight') { e.preventDefault(); bStep.click(); return true; }
        if (e.key === 'r' || e.key === 'R') { e.preventDefault(); bReset.click(); return true; }
        return false;
      },
      redraw() { draw(true); },
      resized() { draw(true); },
      dispose() { disposed = true; running = false; cancelAnimationFrame(raf); },
    };
  }
  S.Pages.register('playground', { nav: 'playground', wide: true, title: 'Playground', open, load(data) { PG.preset(data); return 'playground'; } });

  // what the diagram checker looks at
  const P = 'Nooral.app.page';
  S.Checks.push(
    { group: 'playground', name: 'the default network', hash: 'playground', fresh: true, ready: "document.querySelector('#pgNetStage .it-raster')" },
    { group: 'playground', name: 'after 60 epochs', hash: 'playground', setup: P + '.set({ data: "circles", layers: [4, 2], act: "tanh", seed: 1 }); ' + P + '.train(60);', ready: "document.querySelector('#pgNetStage .it-raster')" },
    { group: 'playground', name: 'a neuron shown on the plane', hash: 'playground', setup: P + '.pick("pg:n1_0"); ' + P + '.pick("pg:w1_0_1");', ready: "document.querySelector('#pgNetStage .it-raster.act')" },
    { group: 'playground', name: 'spirals, deep ReLU, all inputs', hash: 'playground', setup: P + '.set({ data: "spirals", layers: [8, 8, 6, 4], act: "relu", features: ["x1", "x2", "x1sq", "x2sq", "x1x2", "sin1", "sin2"], lr: 0.1, seed: 3 }); ' + P + '.train(120);', ready: "document.querySelector('#pgNetStage .it-raster')" },
  );
})(globalThis.Nooral = globalThis.Nooral || {});
