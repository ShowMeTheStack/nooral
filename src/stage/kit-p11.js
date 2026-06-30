// Pieces of picture for part 11 (sequences and embeddings): word labels placed beside their points without
// covering one another, horizontal probability bars with their names and values, a vector drawn as a strip of
// heat cells, and an unrolled recurrent cell (a box per time step with its input below and its output above).
//   S.Kit.p11.place(pts [{x, y, text, size, r}], {avoid: [boxes], bounds}) -> [{x, y, anchor} | null] a spot for each
//     label (null when no spot is free: the point stays unlabelled)
//   S.Kit.p11.bars({prefix, x, y, w, rows: [{label, v, cls, value}], max, rowH, labelW, size}) -> items
//   S.Kit.p11.strip({prefix, x, y, vals, cw, ch, scheme, max, fmt, dir: 'row' | 'col', cls(i), names, title}) -> {items, w, h}
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const P = (K.p11 = {});
  const r2 = (v) => Math.round(v * 100) / 100;
  const hit = (a, b, pad) => a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;

  /** greedy label placement: points in order, each tries eight spots round its point */
  P.place = function (pts, o) {
    o = o || {};
    const taken = (o.avoid || []).slice(), out = [];
    for (const p of pts) taken.push({ x: p.x - (p.r || 5), y: p.y - (p.r || 5), w: 2 * (p.r || 5), h: 2 * (p.r || 5) });
    for (const p of pts) {
      const fs = p.size || 12, w = S.geom.textWidth(p.text, fs) + 6, h = fs * 1.25 + 2;
      let got = null;
      // a label that must be shown tries farther rings of spots when the near ones are taken
      for (const far of p.must ? [1, 2, 3, 4] : [1]) {
      const d = ((p.r || 5) + 3) * far + (far - 1) * 6;
      const spots = [
        { x: p.x + d, y: p.y + fs * 0.35, anchor: 'start', bx: p.x + d - 3, by: p.y - h / 2 },
        { x: p.x - d, y: p.y + fs * 0.35, anchor: 'end', bx: p.x - d - w + 3, by: p.y - h / 2 },
        { x: p.x, y: p.y - d - 2, anchor: 'middle', bx: p.x - w / 2, by: p.y - d - h + 1 },
        { x: p.x, y: p.y + d + fs, anchor: 'middle', bx: p.x - w / 2, by: p.y + d + 1 },
        { x: p.x + d - 2, y: p.y - d, anchor: 'start', bx: p.x + d - 5, by: p.y - d - h + fs * 0.35 + 1 },
        { x: p.x - d + 2, y: p.y - d, anchor: 'end', bx: p.x - d - w + 5, by: p.y - d - h + fs * 0.35 + 1 },
        { x: p.x + d - 2, y: p.y + d + fs * 0.7, anchor: 'start', bx: p.x + d - 5, by: p.y + d + fs * 0.7 - h + fs * 0.35 },
        { x: p.x - d + 2, y: p.y + d + fs * 0.7, anchor: 'end', bx: p.x - d - w + 5, by: p.y + d + fs * 0.7 - h + fs * 0.35 },
      ];
      for (const s of spots) {
        const b = { x: s.bx, y: s.by, w, h };
        if (o.bounds && (b.x < o.bounds.x || b.y < o.bounds.y || b.x + b.w > o.bounds.x + o.bounds.w || b.y + b.h > o.bounds.y + o.bounds.h)) continue;
        if (taken.some((t) => hit(t, b, 1.5))) continue;
        got = { x: r2(s.x), y: r2(s.y), anchor: s.anchor };
        taken.push(b);
        break;
      }
      if (got) break;
      }
      out.push(got);
    }
    return out;
  };

  /** horizontal bars: a name at the left, a bar as long as v / max, the value after the bar */
  P.bars = function (o) {
    const items = [], rowH = o.rowH || 24, lw = o.labelW || 60, fs = o.size || 13, bw = o.w - lw - 56, max = o.max || 1;
    o.rows.forEach((r, i) => {
      const y = o.y + i * rowH, len = Math.max(1.5, (bw * Math.max(0, r.v)) / max);
      items.push({ key: o.prefix + ':n' + i, kind: 'text', x: r2(o.x + lw - 8), y: r2(y + rowH / 2 + fs * 0.35 - 1), text: r.label, anchor: 'end', size: fs, cls: 'mono ' + (r.labelCls || 'legend') });
      items.push({ key: o.prefix + ':b' + i, kind: 'cell', x: r2(o.x + lw), y: r2(y + 4), w: r2(len), h: rowH - 8, rx: 3, text: '', cls: 'bar p11bar ' + (r.cls || ''), audit: 'ignore' });
      items.push({ key: o.prefix + ':v' + i, kind: 'text', x: r2(o.x + lw + len + 6), y: r2(y + rowH / 2 + fs * 0.35 - 1), text: r.value, size: fs - 1, cls: 'tick' });
    });
    return items;
  };

  /** a vector as a strip of heat cells (a row, or a column), with optional names beside the cells */
  P.strip = function (o) {
    const vals = Array.from(o.vals), col = o.dir === 'col';
    const g = S.Kit.heat.grid({ prefix: o.prefix, x: o.x, y: o.y, cw: o.cw || 30, ch: o.ch || o.cw || 24, vals: col ? vals.map((v) => [v]) : [vals], scheme: o.scheme || 'div', max: o.max || 1, fmt: o.fmt === undefined ? null : o.fmt, rows: col ? o.names || null : o.name ? [o.name] : null, cols: col ? (o.name ? [o.name] : null) : o.names || null, size: o.size || 12, labelSize: o.labelSize || 12, gap: o.gap === undefined ? 2 : o.gap, cls: o.cls ? (i, j) => o.cls(col ? i : j) : null, title: o.title, titleCls: o.titleCls || 'strong', titleSize: o.titleSize || 13 });
    return g;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
// An unrolled recurrent network: a box per time step, the input under it, the output over it, and the hidden
// state passed from box to box.
//   S.Kit.p11.chain({prefix, x, y, n, upto, bw, bh, gap, inputs: [text], outputs: [text | null], label(t), inner(t, box) -> items,
//                    cls(t), outCls(t), inCls(t), carry: 'h' | 'h, c'}) -> {items, box(t)}
(function (S) {
  'use strict';
  const P = S.Kit.p11, r2 = (v) => Math.round(v * 100) / 100;
  P.chain = function (o) {
    const items = [], bw = o.bw || 64, bh = o.bh || 70, gap = o.gap || 22, pre = o.prefix || 'ch';
    const box = (t) => ({ x: r2(o.x + t * (bw + gap)), y: o.y, w: bw, h: bh });
    const inY = o.y + bh + (o.inGap || 40), outY = o.y - (o.outGap || 66);
    for (let t = 0; t < o.n; t++) {
      const b = box(t), on = t < o.upto, cx = r2(b.x + bw / 2);
      const iw = o.inW || 32, ow = o.outW || 36;
      items.push({ key: pre + ':in' + t, kind: 'cell', x: r2(cx - iw / 2), y: inY, w: iw, h: 26, rx: 5, text: o.inputs[t], size: o.inSize || 14, cls: 'tok mono ' + (o.inCls ? o.inCls(t) || '' : '') });
      items.push({ key: pre + ':ia' + t, kind: 'edge', from: { x: cx, y: inY - 2 }, to: { x: cx, y: o.y + bh + 2 }, head: true, cls: on ? '' : 'faint', audit: 'edge-x' });
      items.push({ key: pre + ':b' + t, kind: 'cell', x: b.x, y: b.y, w: bw, h: bh, rx: 8, text: '', cls: 'p11box ' + (o.cls ? o.cls(t) || '' : ''), audit: 'group' });
      items.push({ key: pre + ':bl' + t, kind: 'text', x: cx, y: r2(b.y + 17), text: o.label ? o.label(t) : 'h' + (t + 1), anchor: 'middle', size: 12, cls: 'strong' });
      if (on && o.inner) items.push(...o.inner(t, b));
      if (t + 1 < o.n) items.push({ key: pre + ':h' + t, kind: 'edge', from: { x: r2(b.x + bw + 2), y: r2(b.y + bh / 2) }, to: { x: r2(b.x + bw + gap - 2), y: r2(b.y + bh / 2) }, head: true, cls: t + 1 < o.upto ? 'act' : 'faint', audit: 'edge-x' });
      if (on && o.outputs && o.outputs[t] !== null && o.outputs[t] !== undefined) {
        items.push({ key: pre + ':oa' + t, kind: 'edge', from: { x: cx, y: b.y - 2 }, to: { x: cx, y: outY + 28 }, head: true, audit: 'edge-x' });
        items.push({ key: pre + ':o' + t, kind: 'cell', x: r2(cx - ow / 2), y: outY, w: ow, h: 26, rx: 5, text: o.outputs[t], size: 14, cls: 'tok mono ' + (o.outCls ? o.outCls(t) || '' : '') });
      }
    }
    return { items, box, inY, outY };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
