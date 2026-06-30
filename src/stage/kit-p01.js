// Pictures of data for parts 1 and 2: a data table whose columns are coloured by their type, histograms and box
// plots on the axes of S.Kit.curve, and horizontal bars with their names and values (feature scores, metrics).
//   S.Kit.dt.table({prefix, x, y, cols: [{name, w, cls, sub}], rows: [[text..]..], rowH, size, cls(i, j), headH})
//     -> {items, w, h, cell(i, j) -> {x, y, w, h}, colX(j)}: header cells 'dthead <col cls>', body cells 'dt'
//   S.Kit.dt.hist(ax, key, counts, edges, cls, opts {only}) -> bars standing on the x axis
//   S.Kit.dt.box(ax, key, s {lo, q1, med, q3, hi}, y, h, cls) -> a horizontal box plot at height y (stage units)
//   S.Kit.dt.hbars({prefix, x, y, w, rowH, vals, names, max, fmt, labels, cls(i), nameW, size}) -> bars from a left edge
//     (labels: the text after each bar, in place of fmt(value))
//   S.Kit.dt.vbars({prefix, x, y, w, h, vals, names, max, fmt, cls(i), size}) -> upright bars with names under
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const DT = (K.dt = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  DT.table = function (o) {
    const P = o.prefix || 'dt', rh = o.rowH || 30, hh = o.headH || rh + 4, size = o.size || 13, gap = o.gap === undefined ? 2 : o.gap, items = [];
    const xs = [];
    let cx = o.x;
    for (const c of o.cols) { xs.push(cx); cx += c.w + gap; }
    const w = cx - gap - o.x;
    o.cols.forEach((c, j) => {
      items.push({ key: P + ':h' + j, kind: 'cell', x: r2(xs[j]), y: r2(o.y), w: c.w, h: hh, rx: 4, text: c.name, size: c.size || size, cls: 'dthead ' + (c.cls || '') });
    });
    const cell = (i, j) => ({ x: r2(xs[j]), y: r2(o.y + hh + gap + i * (rh + gap)), w: o.cols[j].w, h: rh });
    o.rows.forEach((row, i) => row.forEach((v, j) => {
      const b = cell(i, j), t = v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v)) ? '—' : String(v);
      const extra = o.cls ? o.cls(i, j) || '' : '';
      items.push({ key: P + ':c' + i + '_' + j, kind: 'cell', x: b.x, y: b.y, w: b.w, h: rh, rx: 3, text: t, size, cls: 'dt ' + (o.cols[j].cls || '') + (t === '—' ? ' miss' : '') + ' ' + extra });
    }));
    return { items, w, h: hh + gap + o.rows.length * (rh + gap) - gap, cell, colX: (j) => xs[j], headH: hh };
  };

  DT.hist = function (ax, key, counts, edges, cls, opts) {
    opts = opts || {};
    const items = [], y0 = ax.Y(Math.max(ax.o.yr[0], 0));
    counts.forEach((c, k) => {
      if (opts.only && !opts.only(k)) return;
      const x1 = ax.X(edges[k]), x2 = ax.X(edges[k + 1]), top = ax.Y(c);
      const h = Math.max(0, y0 - top);
      if (h < 0.5) return;
      items.push({ key: key + ':' + k, kind: 'cell', x: r2(x1 + 0.8), y: r2(top), w: r2(Math.max(0.6, x2 - x1 - 1.6)), h: r2(h), rx: 1.5, text: '', cls: 'hbin ' + (typeof cls === 'function' ? cls(k) || '' : cls || ''), audit: 'ignore', layer: 'lines' });
    });
    return items;
  };

  DT.box = function (ax, key, s, y, h, cls) {
    const X = ax.X, c = cls || '', items = [];
    const top = r2(y - h / 2), bot = r2(y + h / 2);
    items.push({ key: key + ':wh', kind: 'path', d: `M${X(s.lo)},${r2(y)} H${X(s.q1)} M${X(s.q3)},${r2(y)} H${X(s.hi)} M${X(s.lo)},${r2(y - h / 4)} V${r2(y + h / 4)} M${X(s.hi)},${r2(y - h / 4)} V${r2(y + h / 4)}`, cls: 'whisk ' + c, audit: 'ignore', layer: 'lines' });
    items.push({ key: key + ':bx', kind: 'cell', x: X(s.q1), y: top, w: r2(Math.max(1, X(s.q3) - X(s.q1))), h: r2(h), rx: 2, text: '', cls: 'boxq ' + c, audit: 'ignore', layer: 'lines' });
    items.push({ key: key + ':md', kind: 'path', d: `M${X(s.med)},${top} V${bot}`, cls: 'med ' + c, audit: 'ignore', layer: 'items' });
    return items;
  };

  DT.hbars = function (o) {
    const P = o.prefix || 'hb', rh = o.rowH || 26, size = o.size || 13, nameW = o.nameW === undefined ? 120 : o.nameW, items = [];
    const max = o.max || Math.max(1e-12, ...o.vals.map((v) => Math.abs(v)));
    const fmt = o.fmt || ((v) => String(+v.toPrecision(3)));
    const bw = o.w - nameW - (o.valW === undefined ? 56 : o.valW);
    o.vals.forEach((v, i) => {
      const y = o.y + i * rh, len = Math.max(0, (Math.abs(v) / max) * bw);
      items.push({ key: P + ':n' + i, kind: 'text', x: r2(o.x + nameW - 8), y: r2(y + rh / 2 + size * 0.35), text: o.names[i], anchor: 'end', size, cls: 'hbname ' + (o.nameCls ? o.nameCls(i) || '' : '') });
      if (len > 0.5) items.push({ key: P + ':b' + i, kind: 'cell', x: r2(o.x + nameW), y: r2(y + 4), w: r2(len), h: rh - 8, rx: 2, text: '', cls: 'bar ' + (o.cls ? o.cls(i) || '' : ''), audit: 'ignore' });
      if (o.fmt !== null) items.push({ key: P + ':v' + i, kind: 'text', x: r2(o.x + nameW + len + 6), y: r2(y + rh / 2 + size * 0.35), text: o.labels ? o.labels[i] : fmt(v), size, cls: 'hbval mono' });
    });
    return items;
  };

  DT.vbars = function (o) {
    const P = o.prefix || 'vb', n = o.vals.length, slot = o.w / n, bw = Math.min(o.bw || 40, slot * 0.7), size = o.size || 13, items = [];
    const max = o.max || Math.max(1e-12, ...o.vals.map((v) => Math.abs(v)));
    const fmt = o.fmt || ((v) => String(+v.toPrecision(3)));
    items.push({ key: P + ':base', kind: 'path', d: `M${r2(o.x)},${r2(o.y + o.h)} h${r2(o.w)}`, cls: 'axis', audit: 'ignore', layer: 'back' });
    o.vals.forEach((v, i) => {
      const cx = o.x + slot * (i + 0.5), bh = Math.max(0, (Math.max(0, v) / max) * (o.h - size - 8));
      if (bh > 0.5) items.push({ key: P + ':b' + i, kind: 'cell', x: r2(cx - bw / 2), y: r2(o.y + o.h - bh), w: r2(bw), h: r2(bh), rx: 2, text: '', cls: 'bar ' + (o.cls ? o.cls(i) || '' : ''), audit: 'ignore' });
      if (o.fmt !== null) items.push({ key: P + ':v' + i, kind: 'text', x: r2(cx), y: r2(o.y + o.h - bh - 6), text: fmt(v), anchor: 'middle', size, cls: 'hbval mono' });
      if (o.names) items.push({ key: P + ':n' + i, kind: 'text', x: r2(cx), y: r2(o.y + o.h + size + 6), text: o.names[i], anchor: 'middle', size, cls: 'hbname ' + (o.nameCls ? o.nameCls(i) || '' : '') });
    });
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
