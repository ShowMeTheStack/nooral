// The parameter dialog of a block, made from its params schema: whole numbers and numbers (a slider on a log scale
// where the parameter asks for one), choices (whose options may depend on the other parameters or on what flows in),
// switches, seeds with a dice button, text, layer sizes, columns of the table that flows in, and files (the block's own
// readFile parses them). Each field says what is wrong with it as it is typed; OK applies the change as one undo step.
//   S.WF.paramDialog(block, {doc, upstream: {features: [names], view}}) -> Promise<patch | null>
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const { h, icon } = S.UI;
  const fmtNum = (v) => (v === null || v === undefined || v === '' ? '' : String(+(+v).toPrecision(6)));

  /** every file of a folder the reader picks (where the browser can pick folders; else several files) */
  const pickFolder = () => new Promise((resolve) => {
    const inp = h('input', { type: 'file', multiple: true, style: { display: 'none' } });
    try { inp.webkitdirectory = true; inp.setAttribute('webkitdirectory', ''); } catch (e) { /* several files instead */ }
    inp.addEventListener('change', () => { resolve(Array.from(inp.files || [])); inp.remove(); });
    document.body.appendChild(inp);
    inp.click();
  });
  WF.pickFolder = pickFolder;

  WF.paramDialog = function (block, ctx) {
    ctx = ctx || {};
    const def = WF.def(block.type);
    if (!def) return Promise.resolve(null);
    const vals = Object.assign({}, WF.defaults(block.type, block.params));
    const rows = {}, errs = {}, readers = {};
    const form = h('div.wf-pd-form');
    const optionsOf = (p) => { let o = typeof p.options === 'function' ? p.options(vals, ctx) : p.options; return (o || []).map((c) => (Array.isArray(c) ? c : [c, String(c)])); };
    const build = () => {
      form.innerHTML = '';
      for (const p of def.params) {
        const id = 'pd-' + p.id, kind = p.kind || 'text';
        if (kind === 'hidden') continue;
        let ctl;
        const v = vals[p.id];
        if (kind === 'bool') {
          ctl = h('input', { type: 'checkbox', id, checked: !!v });
          ctl.addEventListener('change', () => { vals[p.id] = ctl.checked; refresh(); });
        } else if (kind === 'choice') {
          const opts = optionsOf(p);
          ctl = h('select', { id }, ...opts.map(([cv, cl]) => h('option', { value: String(cv), selected: String(cv) === String(v) }, cl)));
          ctl.addEventListener('change', () => { const c = opts.find(([cv]) => String(cv) === ctl.value); vals[p.id] = c ? c[0] : ctl.value; rebuild(); });
        } else if (kind === 'columns') {
          const names = (ctx.upstream && ctx.upstream.features) || [];
          const chosen = new Set([].concat(v || []));
          if (!names.length) ctl = h('div.wf-pd-cols.muted', { id }, 'The columns appear here once a table flows in. Leave it empty for all of them.');
          else {
            ctl = h('div.wf-pd-cols', { id, role: 'group', 'aria-label': p.label });
            for (const n of names) {
              const c = h('input', { type: 'checkbox', checked: chosen.has(n), value: n });
              c.addEventListener('change', () => { if (c.checked) chosen.add(n); else chosen.delete(n); vals[p.id] = names.filter((x) => chosen.has(x)); refresh(); });
              ctl.appendChild(h('label.wf-pd-col', c, h('span.mono', n)));
            }
          }
        } else if (kind === 'file') {
          const note = h('span.muted.wf-pd-file-note', v ? (p.describe ? p.describe(v, vals) : 'a file is loaded') : 'no file yet');
          const btn = h('button.btn.small', { type: 'button', id }, icon(p.folder ? 'folderOpen' : 'upload'), p.folder ? 'Choose a folder…' : p.multiple ? 'Choose files…' : 'Choose a file…');
          btn.addEventListener('click', async () => {
            const files = p.folder ? await pickFolder() : await S.UI.pickFiles(p.accept || '', !!p.multiple);
            if (!files.length) return;
            const read = def.readFile || p.read;
            if (!read) { errs[p.id].textContent = 'This block cannot read files.'; return; }
            note.textContent = 'reading…';
            try {
              const patch = await read(p.multiple || p.folder ? files : files[0], vals, p.id);
              readers[p.id] = patch;
              Object.assign(vals, patch);
              note.textContent = p.describe ? p.describe(vals[p.id], vals) : files.map((f) => f.name).join(', ');
              errs[p.id].textContent = '';
              refresh();
            } catch (e) { note.textContent = 'not read'; errs[p.id].textContent = String(e.message || e); }
          });
          ctl = h('div.wf-pd-file', btn, note);
        } else if (kind === 'layers') {
          ctl = h('input.mono', { type: 'text', id, value: [].concat(v || []).join(', '), spellcheck: 'false', autocomplete: 'off', placeholder: '16, 8' });
          ctl.addEventListener('input', () => { vals[p.id] = ctl.value; refresh(); });
        } else if (kind === 'number' && p.log && p.min > 0 && p.max > p.min) {
          const toPos = (x) => Math.log10(x), toVal = (q) => +Math.pow(10, q).toPrecision(3);
          const num = h('input', { type: 'text', id, inputmode: 'decimal', value: fmtNum(v), spellcheck: 'false', autocomplete: 'off' });
          const r = h('input', { type: 'range', min: toPos(p.min), max: toPos(p.max), step: 0.02, value: toPos(Math.max(p.min, Math.min(p.max, +v || p.min))), 'aria-label': p.label + ' (slider)' });
          r.addEventListener('input', () => { vals[p.id] = toVal(+r.value); num.value = fmtNum(vals[p.id]); refresh(); });
          num.addEventListener('input', () => { vals[p.id] = num.value; if (+num.value > 0) r.value = toPos(+num.value); refresh(); });
          ctl = h('div.wf-pd-slide', r, num);
        } else if (kind === 'seed') {
          const num = h('input', { type: 'text', id, inputmode: 'numeric', value: fmtNum(v), spellcheck: 'false', autocomplete: 'off' });
          const dice = h('button.icon-btn', { type: 'button', title: 'A new seed', 'aria-label': 'A new seed' }, icon('shuffle'));
          dice.addEventListener('click', () => { num.value = String(1 + Math.floor(Math.random() * 9999)); vals[p.id] = num.value; refresh(); });
          num.addEventListener('input', () => { vals[p.id] = num.value; refresh(); });
          ctl = h('div.wf-pd-seed', num, dice);
        } else {
          const numeric = kind === 'int' || kind === 'number';
          ctl = h('input' + (kind === 'text' ? '' : '.mono'), { type: 'text', id, inputmode: numeric ? (kind === 'int' ? 'numeric' : 'decimal') : null, value: numeric ? fmtNum(v) : v === undefined || v === null ? '' : String(v), spellcheck: 'false', autocomplete: 'off' });
          ctl.addEventListener('input', () => { vals[p.id] = ctl.value; refresh(); });
        }
        const err = h('div.wf-pd-err', { id: id + '-err', role: 'alert' });
        errs[p.id] = err;
        const row = h('div.wf-pd-row' + (kind === 'bool' ? '.bool' : '') + (kind === 'columns' ? '.wide' : ''), { dataset: { param: p.id } }, h('label', { for: id }, p.label || p.id), ctl, p.help ? h('div.wf-pd-help.muted', p.help) : null, err);
        rows[p.id] = row;
        form.appendChild(row);
      }
      if (!def.params.some((p) => p.kind !== 'hidden')) form.appendChild(h('p.muted', 'This block has nothing to set.'));
      refresh();
    };
    const rebuild = () => { const f = document.activeElement && document.activeElement.id; build(); if (f) { const el = document.getElementById(f); if (el) el.focus(); } };
    /** the values as the block wants them, and a sentence for each field that is wrong */
    function parsed() {
      const out = {}, bad = {};
      for (const p of def.params) {
        const kind = p.kind || 'text';
        let v = vals[p.id];
        if (kind === 'hidden') { out[p.id] = v; continue; }
        if (kind === 'int' || kind === 'number' || kind === 'seed') {
          const t = String(v === undefined || v === null ? '' : v).trim().replace(/−/g, '-');
          const n = t === '' ? NaN : Number(t);
          if (!Number.isFinite(n)) bad[p.id] = 'Give a number.';
          else if ((kind === 'int' || kind === 'seed') && !Number.isInteger(n)) bad[p.id] = 'Give a whole number.';
          else if (p.min !== undefined && n < p.min) bad[p.id] = 'At least ' + p.min + '.';
          else if (p.max !== undefined && n > p.max) bad[p.id] = 'At most ' + p.max + '.';
          v = n;
        } else if (kind === 'layers') {
          const parts = String(Array.isArray(v) ? v.join(',') : v || '').split(/[\s,;]+/).filter(Boolean);
          const ns = parts.map(Number);
          if (ns.some((n) => !Number.isInteger(n) || n < 1)) bad[p.id] = 'Layer sizes are whole numbers of at least 1, like 16, 8.';
          else if (p.max !== undefined && ns.some((n) => n > p.max)) bad[p.id] = 'At most ' + p.max + ' units in a layer.';
          else if (p.maxLayers !== undefined && ns.length > p.maxLayers) bad[p.id] = 'At most ' + p.maxLayers + ' layers.';
          else if (!ns.length && !p.allowEmpty) bad[p.id] = 'Give at least one layer.';
          v = ns;
        }
        if (!bad[p.id] && p.validate) { try { const m = p.validate(v, vals); if (m) bad[p.id] = m; } catch (e) { /* no word */ } }
        out[p.id] = v;
      }
      return { out, bad };
    }
    function refresh() {
      const { bad } = parsed();
      for (const p of def.params) {
        if (p.kind === 'hidden') continue;
        const shown = !p.show || (() => { try { return p.show(vals, ctx); } catch (e) { return true; } })();
        if (rows[p.id]) rows[p.id].style.display = shown ? '' : 'none';
        if (errs[p.id]) errs[p.id].textContent = shown ? bad[p.id] || errs[p.id].dataset.keep || '' : '';
      }
      return !def.params.some((p) => bad[p.id] && (!p.show || p.show(vals, ctx)));
    }
    build();
    const head = def.blurb ? h('p.wf-pd-blurb.muted', def.blurb) : null;
    const body = h('div', head, form);
    return S.UI.modal({
      title: (block.label || def.label) + ' · ' + block.id, wide: def.params.length > 5 || def.params.some((p) => p.kind === 'columns'), cls: 'wf-pd-modal', body,
      buttons: [{ label: 'Cancel', value: null }, { label: 'OK', primary: true, keepOpen: true, value: () => (refresh() ? true : undefined) }],
      enterValue: () => (refresh() ? true : undefined), cancelValue: null,
    }).then((yes) => {
      if (!yes) return null;
      const { out } = parsed();
      const patch = {};
      for (const [k, v] of Object.entries(out)) if (JSON.stringify(v) !== JSON.stringify(block.params[k])) patch[k] = v;
      for (const r of Object.values(readers)) for (const [k, v] of Object.entries(r)) if (JSON.stringify(v) !== JSON.stringify(block.params[k])) patch[k] = v;
      return Object.keys(patch).length ? patch : null;
    });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
