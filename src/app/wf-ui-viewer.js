// The viewer panel of a block: its title and category, what state it is in (waiting and why, running with its
// progress, the error), its summary, and below them the block's own viewer (S.WF.viewers[type], or one for its whole
// category), which draws what the block made: a table, a tree growing, a confusion matrix to click. A block with no
// viewer yet shows its parameters and summary.
//   new S.WF.ViewerPanel(host, page) ; panel.open(id) ; panel.close() ; panel.event(e) ; panel.redraw()
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const { h, icon } = S.UI;
  const WORDS = { waiting: 'Waiting', queued: 'Queued', running: 'Running', training: 'Training', paused: 'Paused', done: 'Done', error: 'Error', idle: 'Not run yet' };

  class ViewerPanel {
    constructor(host, page) {
      this.host = host;
      this.page = page;
      this.id = null;
      this.inst = null;
      this.curveFns = new Set();
      this.el = h('section.wf-viewer', { 'aria-label': 'Block viewer', hidden: true });
      host.appendChild(this.el);
    }
    get open() { return !!this.id; }
    /** open the viewer of a block (again, if it is open: it is rebuilt) */
    show(id) {
      const doc = this.page.doc, b = WF.byId(doc, id);
      if (!b) return;
      this.close(true);
      this.id = id;
      const def = WF.def(b.type);
      this.el.hidden = false;
      this.el.dataset.block = id;
      this.el.className = 'wf-viewer ' + WF.Draw.catCls(def && def.cat);
      const edit = h('button.btn.small', { type: 'button', id: 'wfViewEdit', title: 'Change its parameters (Enter)', disabled: !!this.page.readOnly }, icon('edit'), h('span.lbl', 'Parameters'));
      edit.addEventListener('click', () => this.page.editParams(id));
      const close = h('button.icon-btn', { type: 'button', id: 'wfViewClose', title: 'Close the viewer (Esc)', 'aria-label': 'Close the viewer' }, icon('x'));
      close.addEventListener('click', () => this.page.closeViewer());
      this.stateEl = h('div.wf-v-state', { role: 'status', 'aria-live': 'polite' });
      this.head = h('div.wf-v-head', h('div.wf-v-title', h('span.wf-v-cat', def ? def.cat : 'Unknown'), h('h2', b.label || (def ? def.label : b.type), h('span.wf-v-id.mono', b.id))), h('div.wf-v-tools', edit, close));
      this.body = h('div.wf-v-body');
      this.el.replaceChildren(this.head, this.stateEl, this.body);
      this.mountViewer();
      this.paintState();
    }
    api() {
      const page = this.page, id = this.id, self = this;
      const block = () => WF.byId(page.doc, id);
      return {
        id, type: block() && block().type, get block() { return block(); },
        doc: () => page.doc,
        call: (method, args) => page.remote.call(id, method, args),
        control: (action) => page.remote.control(id, action),
        onCurve: (fn) => { self.curveFns.add(fn); return () => self.curveFns.delete(fn); },
        setParams: (patch) => page.setParams(id, patch),
        theme: () => document.documentElement.dataset.theme,
        narrow: () => (self.body.clientWidth || innerWidth) < 600,
        open: (other) => page.open(other),
        state: () => page.remote.states[id] || null,
        view: () => page.remote.views[id],
      };
    }
    mountViewer() {
      const b = WF.byId(this.page.doc, this.id), def = b && WF.def(b.type);
      if (this.inst && this.inst.dispose) try { this.inst.dispose(); } catch (e) { /* gone */ }
      this.inst = null;
      this.body.innerHTML = '';
      const view = this.page.remote.views[this.id];
      const make = def && (WF.viewers[b.type] || WF.viewers['cat:' + def.cat]);
      if (make && view !== undefined && view !== null && !view.error) {
        const host = h('div.wf-v-host');
        this.body.appendChild(host);
        try { this.inst = make(host, view, this.api()) || null; this.mountedFor = view; return; }
        catch (e) { host.remove(); this.body.appendChild(h('div.wf-v-err', icon('warn'), h('span', 'This viewer could not draw: ' + (e.message || e)))); if (typeof console !== 'undefined') console.error(e); }
      }
      this.mountedFor = view;
      this.fallback(b, def, view);
    }
    /** no viewer for this block (yet): its summary and parameters */
    fallback(b, def, view) {
      if (!b) return;
      const st = this.page.remote.states[b.id] || {};
      const box = h('div.wf-v-fallback');
      if (view && view.error) box.appendChild(h('div.wf-v-err', icon('warn'), h('span', view.error)));
      if (def && def.blurb) box.appendChild(h('p.muted', def.blurb));
      if (st.summary) box.appendChild(h('p.wf-v-sum', h('b', 'Result: '), st.summary));
      const params = def ? def.params.filter((p) => p.kind !== 'hidden' && (!p.show || p.show(b.params, {}))) : [];
      if (params.length) {
        const show = (p, v) => { if (p.kind === 'choice') { const o = (typeof p.options === 'function' ? p.options(b.params, {}) : p.options || []).find((c) => String(Array.isArray(c) ? c[0] : c) === String(v)); return o ? (Array.isArray(o) ? o[1] : String(o)) : String(v); } if (p.kind === 'bool') return v ? 'yes' : 'no'; if (p.kind === 'file') return v ? 'loaded' : 'none'; if (Array.isArray(v)) return v.join(', ') || '—'; return v === '' || v === null || v === undefined ? '—' : String(v); };
        box.appendChild(h('div.fit-table', h('table.wf-v-params', h('tbody', ...params.map((p) => h('tr', h('th', p.label || p.id), h('td.mono', show(p, b.params[p.id]))))))));
      }
      if (!st.summary && !(view && view.error)) box.appendChild(h('p.muted', st.state === 'waiting' ? 'Nothing to show until its inputs are wired.' : 'Its picture appears here once it has run.'));
      this.body.appendChild(box);
    }
    paintState() {
      if (!this.id) return;
      const st = this.page.remote.states[this.id] || { state: 'idle' };
      const s = st.state || 'idle';
      const kids = [h('span.wf-v-dot.st-' + s), h('b', WORDS[s] || s)];
      if (s === 'waiting' && st.words) kids.push(h('span', ': ' + st.words));
      else if ((s === 'running' || s === 'training') && st.words) kids.push(h('span', ': ' + st.words));
      else if (s === 'error') kids.push(h('span', ': ' + (st.error || 'it failed')));
      else if (s === 'paused' && st.words) kids.push(h('span', ': ' + st.words));
      else if (st.summary) kids.push(h('span', ' · ' + st.summary));
      if ((s === 'running' || s === 'training' || s === 'paused') && st.progress !== undefined) kids.push(h('span.wf-v-bar', h('i', { style: { width: Math.round(100 * Math.max(0, Math.min(1, st.progress))) + '%' } })));
      this.stateEl.className = 'wf-v-state st-' + s;
      this.stateEl.replaceChildren(...kids);
    }
    /** an event of the run */
    event(e) {
      if (!this.id || e.id !== this.id) return;
      if (e.type === 'state') {
        this.paintState();
        if (e.state === 'error' || e.state === 'waiting') { if (this.inst || this.mountedFor !== undefined) this.mountViewer(); }
      } else if (e.type === 'view') {
        if (e.live) { if (this.inst && this.inst.update) try { this.inst.update(e.view, true); } catch (x) { /* a live frame is a courtesy */ } return; }
        if (this.inst && this.inst.update && e.view && !e.view.error) { try { this.inst.update(e.view, false); this.mountedFor = e.view; } catch (x) { this.mountViewer(); } }
        else this.mountViewer();
      } else if (e.type === 'curve') for (const f of this.curveFns) try { f(e.point); } catch (x) { /* a listener's slip */ }
    }
    /** the block's parameters changed or the doc was replaced: the header follows */
    refreshHead() {
      if (!this.id) return;
      const b = WF.byId(this.page.doc, this.id);
      if (!b) { this.page.closeViewer(); return; }
      const t = this.head.querySelector('h2');
      if (t) t.firstChild.textContent = b.label || (WF.def(b.type) ? WF.def(b.type).label : b.type);
      if (!this.inst) this.mountViewer();
    }
    redraw() { if (this.inst && this.inst.redraw) try { this.inst.redraw(); } catch (e) { /* next update */ } }
    close(keep) {
      if (this.inst && this.inst.dispose) try { this.inst.dispose(); } catch (e) { /* gone */ }
      this.inst = null;
      this.curveFns.clear();
      this.id = null;
      this.mountedFor = undefined;
      if (!keep) { this.el.hidden = true; this.el.replaceChildren(); delete this.el.dataset.block; }
    }
  }
  WF.ViewerPanel = ViewerPanel;
})(globalThis.Nooral = globalThis.Nooral || {});
