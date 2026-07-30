// The workflow canvas demo page (#flow): a palette of blocks, the canvas, and the output of the selected block drawn
// below it. Double-click a block for its parameters; any change runs the pipeline again at once. The confusion
// matrix's cells can be clicked to list the examples in them and ring them on the plot.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const W = S.Flow;
  const NAMES = ['blue', 'orange'];

  S.Pages.register('flow', {
    title: 'A first workflow',
    nav: 'canvas',
    wide: true,
    open(app, main) {
      const doc = W.starter();
      const page = { doc, sel: { block: 'b4' }, cell: null, keys: [['Canvas', [['Delete', 'Remove the selected block or wire'], ['Enter', 'Edit the selected block']]]] };
      const head = h('div.cv-head', h('div', h('h1', 'A first workflow'), h('p', 'Four blocks to start with, before the full canvas. Blocks pass data along typed wires: a table of rows (blue ports) or a trained model (berry ports). Drag from an output on a block’s right to an input on another’s left; double-click a block to change its parameters. Everything runs again as soon as anything changes.')));
      const reset = h('button.btn.small', { type: 'button', id: 'cvReset' }, icon('restart'), 'Start again');
      const full = h('a.btn.small.primary', { href: '#canvas', id: 'cvFull' }, icon('layout'), 'Open the full canvas');
      head.append(h('div.cv-tools', reset, full));
      const pal = h('div.cv-pal', { 'aria-label': 'Blocks' }, h('h3', 'Blocks'));
      for (const cat of ['Data', 'Transform', 'Model', 'Evaluate']) {
        pal.append(h('div.cv-cat', cat));
        for (const [kind, K] of Object.entries(W.KINDS)) if (K.cat === cat) {
          const b = h('button.cv-item', { type: 'button', dataset: { kind }, title: 'Add a ' + K.name + ' block' }, h('span.cv-dot.' + (K.outs[0] ? W.TYPES[K.outs[0].type].cls : 't-none')), K.name);
          b.addEventListener('click', () => { page.canvas.add(kind); });
          pal.append(b);
        }
      }
      const host = h('div.cv-host');
      const panel = h('section.cv-panel', { id: 'cvPanel' });
      const ptitle = h('h2.cv-ptitle'), psum = h('p.cv-psum'), pwrap = h('div.stage-wrap.cv-stage'), pextra = h('div.cv-extra');
      panel.append(h('div.cv-phead', ptitle, psum), pwrap, pextra);
      main.append(head, h('div.cv-main', pal, h('div', h('div.cv-work', { dataset: { scrollAffordance: '1' } }, host), h('p.cv-swipe', 'Swipe sideways to see the whole workflow.'))), panel);
      const stage = new S.Stage(pwrap, { label: 'The output of the selected block', minText: 11 });
      page.stage = stage;
      stage.onPick = (key) => { const m = /^cm:c(\d)_(\d)$/.exec(key); if (m) page.pickCell(+m[1], +m[2]); };

      const run = () => { page.state = W.run(page.doc); page.canvas.setState(page.state); };
      page.canvas = new S.FlowCanvas(host, {
        doc,
        onChange: () => { run(); page.show(); },
        onSelect: (sel) => { if (sel && sel.block) { if (!page.sel || page.sel.block !== sel.block) page.cell = null; page.sel = sel; page.show(); } },
        onEdit: (id) => page.edit(id),
      });
      reset.addEventListener('click', () => { page.doc = W.starter(); page.canvas.doc = page.doc; page.canvas.fit(); page.sel = { block: 'b4' }; page.cell = null; run(); page.canvas.select(page.sel); });

      /** the selected block's output, drawn with the kits */
      page.show = () => {
        const b = page.sel && page.doc.blocks.find((x) => x.id === page.sel.block);
        pextra.innerHTML = '';
        if (!b) { ptitle.textContent = 'Nothing selected'; psum.textContent = 'Click a block to see what it makes.'; stage.show({ items: [] }, { instant: true }); pwrap.style.height = '120px'; return; }
        const K = W.KINDS[b.kind], st = page.state, out = st.out[b.id];
        ptitle.textContent = K.name;
        psum.textContent = st.err[b.id] ? 'Error: ' + st.err[b.id] : st.missing[b.id] ? 'Waiting for its ' + st.missing[b.id].join(' and ') + ' input: wire one in.' : K.sum(out);
        const narrow = (pwrap.clientWidth || innerWidth) < 600;
        const frame = out ? S.FlowViews[b.kind](out, { narrow, cell: page.cell }) : { items: [], view: { x: 0, y: 0, w: 600, h: 120 } };
        const PW = pwrap.clientWidth || 800;
        // wide screens: the picture keeps its own scale, centred
        if (!narrow && PW > frame.view.w) frame.view = { x: frame.view.x - (PW - frame.view.w) / 2, y: frame.view.y, w: PW, h: frame.view.h };
        pwrap.style.height = Math.round(Math.max(160, PW * (frame.view.h / frame.view.w))) + 'px';
        stage.show(frame, { instant: true });
        if (b.kind === 'confusion' && out) pextra.append(S.FlowViews.examples(out, page.cell));
      };
      page.pickCell = (i, j) => { page.sel = { block: page.doc.blocks.find((b) => b.kind === 'confusion').id }; page.cell = page.cell && page.cell[0] === i && page.cell[1] === j ? null : [i, j]; page.canvas.select(page.sel); };
      page.selectBlock = (id) => page.canvas.select({ block: id });
      /** the parameters dialog of a block */
      page.edit = (id) => {
        const b = page.doc.blocks.find((x) => x.id === id);
        if (!b) return;
        const K = W.KINDS[b.kind], fields = {};
        const body = h('div.cv-form', ...K.params.map((p) => {
          let ctl;
          if (p.kind === 'choice') ctl = h('select', { id: 'cvp-' + p.id }, ...p.choices.map(([v, l]) => h('option', { value: v, selected: String(b.params[p.id]) === String(v) }, l)));
          else ctl = h('input', { type: 'number', id: 'cvp-' + p.id, min: p.min, max: p.max, step: p.step, value: b.params[p.id] });
          fields[p.id] = ctl;
          return h('label.cv-field', { for: 'cvp-' + p.id }, h('span', p.name), ctl, p.hint ? h('small', p.hint) : null);
        }));
        S.UI.modal({ title: K.name + ': parameters', body, cls: 'cv-modal', buttons: [{ label: 'Cancel', value: false }, { label: 'Apply', value: true, primary: true }], enterValue: true }).then((ok) => {
          if (!ok) return;
          for (const p of K.params) {
            const v = fields[p.id].value;
            if (p.kind === 'choice') b.params[p.id] = typeof p.choices[0][0] === 'number' ? +v : v;
            else if (v !== '' && isFinite(+v)) b.params[p.id] = Math.max(p.min, Math.min(p.max, +v));
          }
          page.cell = null;
          run(); page.canvas.select({ block: id });
        });
      };
      page.onKey = (e, typing) => { if (!typing && e.key === 'Enter' && page.sel && page.sel.block && document.activeElement === page.canvas.svg) { e.preventDefault(); page.edit(page.sel.block); return true; } return false; };
      page.redraw = () => page.show();
      page.resized = () => { page.canvas.fit(); page.show(); };
      page.dispose = () => S.UI.closeAllModals();
      run();
      page.canvas.select(page.sel);
      requestAnimationFrame(() => { page.canvas.fit(); page.show(); });
      return page;
    },
  });

  // the scenes the diagram checker draws on this page
  S.Checks.push(
    { group: 'flow', name: 'the starter pipeline, confusion matrix shown', hash: 'flow', ready: 'document.querySelector(".cv-block.ok")', setup: 'Nooral.app.page.selectBlock("b4")', sel: '#main' },
    { group: 'flow', name: 'a cell of the confusion matrix picked', hash: 'flow', ready: 'document.querySelector(".cv-block.ok")', setup: 'Nooral.app.page.pickCell(1, 0)', sel: '#main' },
    { group: 'flow', name: 'the data block', hash: 'flow', ready: 'document.querySelector(".cv-block.ok")', setup: 'Nooral.app.page.selectBlock("b1")', sel: '#main' },
    { group: 'flow', name: 'the split block', hash: 'flow', ready: 'document.querySelector(".cv-block.ok")', setup: 'Nooral.app.page.selectBlock("b2")', sel: '#main' },
    { group: 'flow', name: 'the logistic regression block', hash: 'flow', ready: 'document.querySelector(".cv-block.ok")', setup: 'Nooral.app.page.selectBlock("b3")', sel: '#main' },
    { group: 'flow', name: 'the parameters dialog', hash: 'flow', ready: 'document.querySelector(".cv-block.ok")', setup: 'Nooral.app.page.edit("b3"); await new Promise((r) => setTimeout(r, 100))', sel: '.modal' },
  );
})(globalThis.Nooral = globalThis.Nooral || {});
