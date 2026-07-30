// The canvas page (#canvas): the workspace with a toolbar around it: the workflow's name, New, Open, Save, Samples,
// Share, Undo and Redo, Tidy, and on a phone the palette in a drawer. The workflow is kept in this browser as it is
// changed (when the reader keeps work), opened again next time, and can be shared as a link or saved as a file.
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const { h, icon } = S.UI;
  const store = S.UI.storage;
  let pending = null;
  /** the document the canvas opens with next (a lesson's card, a share link) */
  WF.preset = (doc) => { pending = WF.adopt(doc); };
  const KEY = 'canvas.doc';
  const autosave = () => !S.Shell || !S.Shell.autosave || S.Shell.autosave();

  function open(app, main) {
    let doc = null;
    if (pending) { doc = pending; pending = null; }
    else { try { const saved = store.get(KEY, null); if (saved) doc = WF.adopt(saved); } catch (e) { doc = null; } }
    if (!doc) doc = WF.sample('first-pipeline') || WF.newDoc();
    const box = h('div.wf-page', { id: 'canvasView' });
    main.appendChild(box);
    const bar = h('div.wf-toolbar', { role: 'toolbar', 'aria-label': 'Workflow' });
    const body = h('div.wf-page-body');
    box.append(bar, body);
    const save = S.UI.debounce((d) => { if (autosave()) try { store.set(KEY, d); } catch (e) { /* full */ } }, 300);
    const ws = new WF.Workspace(body, { doc, page: true, onChange: (d) => { save(d); syncBar(); }, onEvent: (e) => { if (e.type === 'state' || e.type === 'idle') status(); } });

    // ---- the toolbar ----
    const btn = (id, ic, label, title, fn, cls) => { const b = h('button.btn.small' + (cls ? '.' + cls : ''), { type: 'button', id, title }, icon(ic), h('span.lbl', label)); b.addEventListener('click', fn); return b; };
    const name = h('input.wf-name', { type: 'text', id: 'wfName', 'aria-label': 'The workflow’s name', value: ws.doc.name || '', spellcheck: 'false' });
    name.addEventListener('change', () => { ws.editor.rename(name.value.trim() || 'Untitled workflow'); });
    name.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); name.blur(); } });
    const palBtn = btn('wfPalBtn', 'plus', 'Blocks', 'The blocks (a drawer on a phone)', () => ws.el.classList.toggle('pal-open'), 'wf-pal-btn');
    const bNew = btn('wfNew', 'newFile', 'New', 'A new, empty workflow', async () => {
      if (ws.doc.blocks.length && !(await S.UI.confirm('Start a new workflow?', 'The canvas will be emptied. Save this one first if you want to keep it.', 'Start anew', 'Keep it'))) return;
      await ws.loadDoc(WF.newDoc());
      syncBar();
    });
    const bOpen = btn('wfOpen', 'folderOpen', 'Open', 'Open a workflow saved as a file', async () => {
      const files = await S.UI.pickFiles('.json,application/json');
      if (!files.length) return;
      try { const d = WF.adopt(JSON.parse(await files[0].text())); await ws.loadDoc(d); syncBar(); S.UI.toast('Opened ' + files[0].name + '.', 'ok'); }
      catch (e) { S.UI.toast('That file could not be opened: ' + e.message, 'err', 5000); }
    });
    const bSave = btn('wfSave', 'download', 'Save', 'Save the workflow as a file', () => page.saveFile());
    const bSamples = btn('wfSamples', 'layers', 'Samples', 'Open a sample workflow', () => {
      const r = bSamples.getBoundingClientRect();
      const list = [...WF.samples.values()].sort((a, b) => (a.cat || '').localeCompare(b.cat || '') || (a.order || 50) - (b.order || 50));
      const items = [];
      let cat = null;
      for (const s of list) { if (s.cat !== cat) { cat = s.cat; items.push({ header: true, label: cat || 'Samples' }); } items.push({ label: s.title, action: async () => { if (ws.doc.blocks.length > 1 && JSON.stringify(ws.doc) !== store.get('canvas.clean', '') && !(await S.UI.confirm('Open the sample?', 'It replaces the workflow on the canvas. Save this one first if you want to keep it.', 'Open the sample', 'Keep mine'))) return; await page.loadSample(s.id); } }); }
      S.UI.showMenu(items, r.left, r.bottom + 4, { anchor: bSamples, keyboard: true });
    });
    const bShare = btn('wfShare', 'link', 'Share', 'Copy a link that opens this workflow', () => page.share());
    const bUndo = btn('wfUndo', 'undo', 'Undo', 'Undo (Ctrl+Z)', () => ws.editor.undo());
    const bRedo = btn('wfRedo', 'redo', 'Redo', 'Redo (Ctrl+Y)', () => ws.editor.redo());
    const bTidy = btn('wfTidy', 'tidy', 'Tidy', 'Lay the workflow out neatly (L)', () => ws.editor.tidy());
    const stat = h('span.wf-status', { id: 'wfStatus', role: 'status', 'aria-live': 'polite' });
    bar.append(palBtn, name, h('span.wf-tb-sep'), bNew, bOpen, bSave, bSamples, bShare, h('span.wf-tb-sep'), bUndo, bRedo, bTidy, stat);
    function syncBar() {
      if (document.activeElement !== name) name.value = ws.doc.name || '';
      bUndo.disabled = ws.editor.hi <= 0;
      bRedo.disabled = ws.editor.hi >= ws.editor.hist.length - 1;
    }
    function status() {
      const sts = Object.entries(ws.remote.states).filter(([id]) => WF.byId(ws.doc, id)).map(([, s]) => s.state);
      const busy = sts.filter((s) => s === 'running' || s === 'training').length, bad = sts.filter((s) => s === 'error').length;
      stat.className = 'wf-status' + (busy ? ' busy' : bad ? ' bad' : '');
      stat.textContent = busy ? (busy === 1 ? 'Running 1 block…' : 'Running ' + busy + ' blocks…') : bad ? bad + ' block' + (bad === 1 ? '' : 's') + ' failed' : ws.doc.blocks.length ? 'Up to date' + (ws.remote.mode === 'page' ? ' (on the page)' : '') : '';
    }
    syncBar();
    const origChanged = ws.changed.bind(ws);
    ws.changed = (d, why) => { origChanged(d, why); syncBar(); };

    const page = {
      editor: ws.editor, remote: ws.remote, workspace: ws,
      get doc() { return ws.doc; },
      loadDoc: async (d) => { const p = ws.loadDoc(WF.adopt(d)); syncBar(); await p; status(); },
      loadSample: async (id) => { const p = ws.loadSample(id); syncBar(); try { store.set('canvas.clean', JSON.stringify(ws.doc)); } catch (e) { /* no room */ } await p; status(); },
      whenIdle: () => ws.whenIdle(),
      open: (id) => ws.open(id),
      closeViewer: () => ws.closeViewer(),
      editParams: (id) => ws.editParams(id),
      setParams: (id, patch) => ws.setParams(id, patch),
      saveFile() { const d = ws.doc; S.UI.download(String(d.name || 'workflow').replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '').toLowerCase() + '.nooral.json', JSON.stringify(d, null, 1), 'application/json'); },
      async share() {
        const json = JSON.stringify(ws.doc);
        if (json.length > 60000) { S.UI.toast('This workflow is too big for a link (it holds a file’s data). Save it as a file and send that instead.', 'err', 6000); return null; }
        return app.share ? app.share('canvas', JSON.parse(json)) : null;
      },
      keys: [['Canvas', [['Drag', 'Move a block; from a port, draw a wire'], ['Double-click  Enter', 'The block’s parameters'], ['Click  O', 'Open the block’s viewer'], ['Delete', 'Delete what is selected'], ['Ctrl  Z  Ctrl  Y', 'Undo, redo'], ['Ctrl  C  Ctrl  V', 'Copy, paste'], ['Ctrl  A', 'Select every block'], ['L', 'Tidy the layout'], ['F', 'Fit the workflow'], ['+  −', 'Zoom in, out'], ['Wheel  Ctrl+wheel', 'Pan, zoom'], ['Arrows', 'Nudge what is selected'], ['Esc', 'Close the viewer']]]],
      onKey(e, typing) {
        if (e.__wf) return true;
        if (typing) return false;
        if (e.key === 'Escape' && ws.viewer.open) { ws.closeViewer(); return true; }
        if (e.key === 'Escape' && ws.el.classList.contains('pal-open')) { ws.el.classList.remove('pal-open'); return true; }
        const tag = (e.target && e.target.tagName) || '';
        if (tag === 'BUTTON' || tag === 'A' || tag === 'SELECT') return false;
        // keys that belong to the canvas work when nothing else has the focus
        if (ws.editor.key(e, true)) { e.preventDefault(); return true; }
        return false;
      },
      redraw() { ws.redraw(); },
      dispose() { ws.dispose(); },
    };
    status();
    return page;
  }

  S.Pages.register('canvas', {
    nav: 'canvas', wide: true, title: 'Workflow canvas',
    open,
    load(data) { WF.preset(data); return 'canvas'; },
  });

  // ---- what the diagram checker looks at -------------------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.group === 'canvas')) {
    const ready = "document.querySelector('#canvasView .wf-svg') && Nooral.app.page && Nooral.app.page.whenIdle";
    S.Checks.push({ group: 'canvas', name: 'empty canvas', hash: 'canvas', fresh: true, ready, setup: 'await Nooral.app.page.loadDoc(Nooral.WF.newDoc()); await Nooral.app.page.whenIdle();', sel: '#main' });
    // every sample, and every workflow in a lesson: wf-z-checks.js (it loads after every file that adds samples)
    S.Checks.push({ group: 'canvas', name: 'sample with a viewer open', hash: 'canvas', ready, setup: "await Nooral.app.page.loadSample('first-pipeline'); await Nooral.app.page.whenIdle(); Nooral.app.page.open('split1'); await new Promise((r) => setTimeout(r, 200));", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
