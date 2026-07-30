// Drawing the canvas as SVG: a block (its category stripe, title, one-line summary, typed ports with their names, a
// status mark and a progress bar) and a wire (a curve from an output to an input, coloured by what it carries, with a
// halo so a wire that must cross another reads as passing over it). Strings, so a whole canvas redraws in one go.
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const esc = (t) => String(t === undefined || t === null ? '' : t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const D = (WF.Draw = { esc });
  const CAT_CLS = (cat) => 'cat-' + String(cat || 'x').toLowerCase();
  D.catCls = CAT_CLS;
  /** the type a port shows: an input that takes several kinds shows the first, or 'data' for the three data kinds */
  D.typeCls = (t) => { const ts = [].concat(t); if (ts.length > 1) return ts.every((x) => ['table', 'images', 'text'].includes(x)) ? 't-data' : 't-' + ts[0]; return 't-' + (ts[0] === 'same:in' ? 'data' : ts[0]); };
  const STATE_WORDS = { waiting: 'waiting', queued: 'queued', running: 'running', training: 'training', paused: 'paused', done: 'done', error: 'error' };
  D.STATE_WORDS = STATE_WORDS;
  /** cut a text to about n characters, with an ellipsis */
  D.cut = (t, n) => { t = String(t || ''); return t.length > n ? t.slice(0, n - 1).trimEnd() + '…' : t; };

  /** one block: st = {state, summary, words, progress, error} or null */
  D.block = function (b, doc, st, opts) {
    opts = opts || {};
    const def = WF.def(b.type), g = WF.geom(b, doc), W = g.w, H = g.h, named = !WF.dims(doc).compact;
    const state = (st && st.state) || 'idle';
    const cls = 'wf-block ' + CAT_CLS(def && def.cat) + ' st-' + state + (opts.sel ? ' sel' : '') + (def ? '' : ' unknown');
    const title = D.cut(b.label || (def ? def.label : b.type), 30);
    let sum = '';
    if (state === 'error') sum = 'error: ' + ((st && st.error) || '');
    else if (state === 'waiting') sum = (st && st.words) || 'waiting';
    else if (state === 'running' || state === 'training') sum = (st && st.words) || (state === 'training' ? 'training…' : 'running…');
    else if (st && st.summary) sum = st.summary;
    else if (!def) sum = 'unknown block ' + b.type;
    else sum = '';
    let s = '<g class="' + cls + '" data-id="' + esc(b.id) + '" transform="translate(' + b.x + ',' + b.y + ')" role="group" aria-label="' + esc(title) + '">';
    s += '<rect class="wf-bbox" data-audit="node" data-audit-ends="strict" x="0" y="0" width="' + W + '" height="' + H + '" rx="10"/>';
    s += '<path class="wf-bstripe" data-audit="ignore" d="M10,0 H' + (W - 10) + ' A10,10 0 0 1 ' + W + ',10 V5 H0 V10 A10,10 0 0 1 10,0 Z"/>';
    s += '<text class="wf-btitle" x="14" y="' + (named ? 25 : 24) + '">' + esc(title) + '</text>';
    s += '<text class="wf-bsum" x="14" y="42">' + esc(D.cut(sum, 44)) + '</text>';
    s += '<circle class="wf-bstate" data-audit="ignore" cx="' + (W - 16) + '" cy="20" r="4.5"><title>' + esc(STATE_WORDS[state] || '') + '</title></circle>';
    for (const p of g.ins) {
      s += '<circle class="wf-port in ' + D.typeCls(p.type) + (p.optional ? ' opt' : '') + '" data-audit="ignore" data-port="' + esc(p.id) + '" data-side="in" cx="0" cy="' + p.y + '" r="6"><title>' + esc(p.label + ': takes ' + [].concat(p.type).map((t) => (WF.TYPES[t] ? WF.TYPES[t].label : t)).join(' or ')) + '</title></circle>';
      if (named) s += '<text class="wf-plabel" x="12" y="' + (p.y + 4) + '">' + esc(D.cut(p.label, 18)) + '</text>';
    }
    for (const p of g.outs) {
      s += '<circle class="wf-port out ' + D.typeCls(p.type) + '" data-audit="ignore" data-port="' + esc(p.id) + '" data-side="out" cx="' + W + '" cy="' + p.y + '" r="6"><title>' + esc(p.label + ': gives ' + (WF.TYPES[p.type] ? WF.TYPES[p.type].label : 'data')) + '</title></circle>';
      if (named) s += '<text class="wf-plabel out" x="' + (W - 12) + '" y="' + (p.y + 4) + '" text-anchor="end">' + esc(D.cut(p.label, 18)) + '</text>';
    }
    const prog = st && (state === 'running' || state === 'training' || state === 'paused') && st.progress !== undefined ? Math.max(0, Math.min(1, st.progress)) : null;
    if (prog !== null) s += '<rect class="wf-prog-track" data-audit="ignore" x="10" y="' + (H - 7) + '" width="' + (W - 20) + '" height="3" rx="1.5"/><rect class="wf-prog" data-audit="ignore" x="10" y="' + (H - 7) + '" width="' + Math.max(2, (W - 20) * prog) + '" height="3" rx="1.5"/>';
    return s + '</g>';
  };

  /** every wire: crossing wires are marked edge-x (drawn with a halo so the crossing reads as one passing over) */
  D.wires = function (doc, selWire, crossing) {
    let s = '';
    for (const g of WF.wireGeom(doc)) {
      const w = g.w, d = WF.wirePath(g.a, g.b, g.via);
      const t = WF.outType(doc, w.from[0], w.from[1]) || 'data';
      const x = crossing && crossing.has(w.id);
      s += '<g class="wf-wire-g' + (w.id === selWire ? ' sel' : '') + '" data-wire="' + esc(w.id) + '">';
      s += '<path class="wf-halo" data-audit="ignore" d="' + d + '"/>';
      s += '<path class="wf-wire ' + D.typeCls(t) + '" data-audit="' + (x ? 'edge-x' : 'edge') + '" d="' + d + '"/>';
      s += '<path class="wf-hit" data-audit="ignore" d="' + d + '"/>';
      s += '</g>';
    }
    return s;
  };
  /** the ids of wires that cross some other wire as drawn */
  D.crossingWires = function (doc) {
    const out = new Set();
    const ws = WF.wireGeom(doc).map((g) => Object.assign(g, { pts: WF.wirePoints(g.a, g.b, g.via, 20) }));
    const segX = (p1, p2, p3, p4) => {
      const d = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x);
      if (Math.abs(d) < 1e-9) return false;
      const t = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / d, u = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / d;
      return t > 1e-6 && t < 1 - 1e-6 && u > 1e-6 && u < 1 - 1e-6;
    };
    for (let i = 0; i < ws.length; i++) for (let j = i + 1; j < ws.length; j++) {
      const A = ws[i], B = ws[j];
      if (A.w.from[0] === B.w.from[0] && A.w.from[1] === B.w.from[1]) continue;
      let hit = false;
      for (let a = 1; a < A.pts.length && !hit; a++) for (let b = 1; b < B.pts.length && !hit; b++) if (segX(A.pts[a - 1], A.pts[a], B.pts[b - 1], B.pts[b])) hit = true;
      if (hit) { out.add(A.w.id); out.add(B.w.id); }
    }
    return out;
  };
  /** a palette glyph: a small block with its ports in their colours */
  D.glyph = function (def) {
    const ports = { inputs: typeof def.inputs === 'function' ? def.inputs(WF.defaults(def.type)) : def.inputs, outputs: typeof def.outputs === 'function' ? def.outputs(WF.defaults(def.type)) : def.outputs };
    const n = Math.max(1, ports.inputs.length, ports.outputs.length), H = 8 + n * 9;
    let s = '<svg class="wf-glyph ' + CAT_CLS(def.cat) + '" viewBox="-5 -2 42 ' + (H + 4) + '" width="30" height="' + Math.round(((H + 4) * 30) / 42) + '" aria-hidden="true"><rect class="wf-gbox" x="0" y="0" width="32" height="' + H + '" rx="4"/><rect class="wf-gstripe" x="0" y="0" width="32" height="3" rx="1.5"/>';
    ports.inputs.forEach((p, i) => { s += '<circle class="wf-gport ' + D.typeCls(p.type) + '" cx="0" cy="' + (6 + (i + 0.5) * ((n * 9) / ports.inputs.length)) + '" r="3"/>'; });
    ports.outputs.forEach((p, i) => { s += '<circle class="wf-gport ' + D.typeCls(p.type) + '" cx="32" cy="' + (6 + (i + 0.5) * ((n * 9) / ports.outputs.length)) + '" r="3"/>'; });
    return s + '</svg>';
  };
})(globalThis.Nooral = globalThis.Nooral || {});
