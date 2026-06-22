// Shape ops. transpose, permute, slice, squeeze, unsqueeze and expand are views (they share the data and only
// change shape, strides and offset); reshape is a view when the data is contiguous. All of them pass gradients back.
//   reshape(x, shape)  (one entry may be -1)   flatten(x, startAxis = 1)   transpose(x, a = -2, b = -1)   permute(x, dims)
//   slice(x, ...specs)  spec per axis: an index, [start, end, step?], or null for all   contiguous(x)
//   squeeze(x, axis)  unsqueeze(x, axis)  concat(list, axis)  stack(list, axis)  split(x, sizes, axis)
//   take(x, indices, axis = 0)   gather(x, axis, index)   embedding(weight, ids)
(function (S) {
  'use strict';
  const T = S.T;
  const F = (t) => t.data.constructor;
  const idxValues = (ix) => (ix instanceof T.Tensor ? ix.values() : ArrayBuffer.isView(ix) ? ix : T.tensor(ix).values());
  const idxShape = (ix) => (ix instanceof T.Tensor ? ix.shape : ArrayBuffer.isView(ix) ? [ix.length] : T.tensor(ix).shape);

  /** add (or write) contiguous values into a strided view */
  T.writeView = (view, vals, add) => {
    const sh = view.shape, st = view.strides, n = sh.length, d0 = view.data;
    if (!view.size) return;
    if (!n) { if (add) d0[view.offset] += vals[0]; else d0[view.offset] = vals[0]; return; }
    // runs along the last axis, then an odometer over the others
    const len = sh[n - 1], s1 = st[n - 1], idx = new Array(n).fill(0);
    let off = view.offset;
    for (let i = 0; i < view.size; i += len) {
      if (add) for (let j = 0, o = off; j < len; j++, o += s1) d0[o] += vals[i + j];
      else for (let j = 0, o = off; j < len; j++, o += s1) d0[o] = vals[i + j];
      for (let d = n - 2; d >= 0; d--) { idx[d]++; off += st[d]; if (idx[d] < sh[d]) break; off -= st[d] * sh[d]; idx[d] = 0; }
    }
  };

  T.def('contiguous', (x) => {
    if (x.isContiguous()) return x;
    return T.record(T.make(new (F(x))(x.values()), x.shape), 'contiguous', [x], (g) => [g]);
  });
  T.def('reshape', (x, shape) => {
    shape = shape.slice();
    const k = shape.indexOf(-1);
    if (k >= 0) shape[k] = x.size / T.prod(shape.filter((d, i) => i !== k));
    if (T.prod(shape) !== x.size) throw new Error('cannot reshape [' + x.shape + '] into [' + shape + ']');
    const out = x.isContiguous() ? T.make(x.data, shape) : T.make(new (F(x))(x.values()), shape);
    return T.record(out, 'reshape', [x], (g) => [T.make(g.values(), x.shape)], { from: x.shape.slice(), to: shape });
  });
  T.def('flatten', (x, start = 1) => T.reshape(x, x.shape.slice(0, start).concat([T.prod(x.shape.slice(start))])));
  T.def('permute', (x, dims) => {
    dims = dims.map((d) => T.normAxis(d, x.ndim));
    const out = T.make(x.data, dims.map((d) => x.shape[d]), dims.map((d) => x.strides[d]), x.offset);
    const inv = new Array(dims.length); dims.forEach((d, i) => { inv[d] = i; });
    return T.record(out, 'permute', [x], (g) => [T.permute(g, inv)], { dims });
  });
  T.def('transpose', (x, a = -2, b = -1) => {
    const n = x.ndim, d = [...Array(n).keys()], i = T.normAxis(a, n), j = T.normAxis(b, n);
    d[i] = j; d[j] = i;
    const out = T.permute(x, d);
    if (out.ctx) out.ctx.op = 'transpose';
    return out;
  });
  T.Tensor.prototype.T = function () { return T.transpose(this); };
  T.def('squeeze', (x, axis) => {
    const keep = x.shape.map((s, i) => (axis === undefined ? s !== 1 : i !== T.normAxis(axis, x.ndim)));
    const out = T.make(x.data, x.shape.filter((_, i) => keep[i]), x.strides.filter((_, i) => keep[i]), x.offset);
    return T.record(out, 'squeeze', [x], (g) => [T.make(g.values(), x.shape)]);
  });
  T.def('unsqueeze', (x, axis) => {
    const a = axis < 0 ? axis + x.ndim + 1 : axis, sh = x.shape.slice(), st = x.strides.slice();
    sh.splice(a, 0, 1); st.splice(a, 0, a < x.ndim ? x.strides[a] * x.shape[a] : 1);
    return T.record(T.make(x.data, sh, st, x.offset), 'unsqueeze', [x], (g) => [T.make(g.values(), x.shape)]);
  });
  /** a broadcast view (stride 0 along stretched axes); its gradient sums back */
  T.def('expand', (x, shape) => {
    const out = T.make(x.data, shape, T.broadcastStrides(x.shape, x.strides, shape), x.offset);
    return T.record(out, 'expand', [x], (g) => [T.sumTo(g, x.shape)]);
  });

  /** the view geometry of a slice */
  function sliceView(x, specs) {
    let off = x.offset;
    const shape = [], strides = [];
    for (let d = 0; d < x.ndim; d++) {
      const s = specs[d], n = x.shape[d];
      if (s === undefined || s === null) { shape.push(n); strides.push(x.strides[d]); continue; }
      if (typeof s === 'number') { const k = s < 0 ? s + n : s; if (k < 0 || k >= n) throw new Error('index ' + s + ' out of range for axis ' + d); off += k * x.strides[d]; continue; }
      const step = s[2] || 1;
      let a = s[0] === undefined || s[0] === null ? (step > 0 ? 0 : n - 1) : s[0] < 0 ? s[0] + n : s[0];
      let b = s[1] === undefined || s[1] === null ? (step > 0 ? n : -1) : s[1] < 0 ? s[1] + n : s[1];
      a = Math.max(step > 0 ? 0 : -1, Math.min(a, step > 0 ? n : n - 1));
      b = Math.max(-1, Math.min(b, n));
      const len = Math.max(0, Math.ceil((b - a) / step));
      shape.push(len); strides.push(x.strides[d] * step); off += a * x.strides[d];
    }
    return { shape, strides, offset: off };
  }
  T.def('slice', (x, ...specs) => {
    if (specs.length === 1 && Array.isArray(specs[0]) && specs[0].some((s) => Array.isArray(s) || s === null)) specs = specs[0];
    const v = sliceView(x, specs), out = T.make(x.data, v.shape, v.strides, v.offset);
    // the gradient only touches the sliced part: it is added straight into the input's gradient (see t-01)
    return T.record(out, 'slice', [x], (g) => [{ addTo: (dst) => { const zv = sliceView(dst, specs); T.writeView(T.make(dst.data, zv.shape, zv.strides, zv.offset), g.values(), true); } }], { specs });
  });

  T.concat = (list, axis = 0) => {
    const a = T.normAxis(axis, list[0].ndim), shape = list[0].shape.slice();
    shape[a] = list.reduce((s, t) => s + t.shape[a], 0);
    const { outer, inner } = T.splitAxis(shape, a), n = shape[a], out = new (F(list[0]))(T.prod(shape));
    let at = 0;
    const offs = [];
    for (const t of list) {
      const v = t.values(), m = t.shape[a];
      for (let o = 0; o < outer; o++) out.set(v.subarray(o * m * inner, (o + 1) * m * inner), (o * n + at) * inner);
      offs.push(at); at += m;
    }
    return T.record(T.make(out, shape), 'concat', list, (g) => {
      const gv = g.values();
      return list.map((t, j) => {
        if (!t.requiresGrad) return null;
        const m = t.shape[a], r = new (F(g))(t.size);
        for (let o = 0; o < outer; o++) r.set(gv.subarray((o * n + offs[j]) * inner, (o * n + offs[j] + m) * inner), o * m * inner);
        return T.make(r, t.shape);
      });
    }, { axis: a });
  };
  T.stack = (list, axis = 0) => T.concat(list.map((t) => T.unsqueeze(t, axis)), axis);
  T.def('split', (x, sizes, axis = 0) => {
    const a = T.normAxis(axis, x.ndim);
    if (typeof sizes === 'number') sizes = Array(x.shape[a] / sizes).fill(sizes);
    let at = 0;
    return sizes.map((s) => { const sp = new Array(x.ndim).fill(null); sp[a] = [at, at + s]; at += s; return T.slice(x, sp); });
  });

  /** rows (or slices along an axis) picked by integer indices; out shape = x.shape[:axis] + idx.shape + x.shape[axis+1:] */
  T.def('take', (x, indices, axis = 0, opName = 'take') => {
    const a = T.normAxis(axis, x.ndim), iv = idxValues(indices), ish = idxShape(indices);
    const { outer, n, inner } = T.splitAxis(x.shape, a), m = iv.length, v = x.values(), out = new (F(x))(outer * m * inner);
    for (let o = 0; o < outer; o++) for (let j = 0; j < m; j++) {
      const k = iv[j] | 0;
      if (k < 0 || k >= n) throw new Error('index ' + iv[j] + ' out of range 0..' + (n - 1));
      out.set(v.subarray((o * n + k) * inner, (o * n + k + 1) * inner), (o * m + j) * inner);
    }
    const shape = x.shape.slice(0, a).concat(ish, x.shape.slice(a + 1));
    return T.record(T.make(out, shape), opName, [x], (g) => {
      const gv = g.values(), r = new (F(g))(x.size);
      for (let o = 0; o < outer; o++) for (let j = 0; j < m; j++) {
        const dst = (o * n + (iv[j] | 0)) * inner, src = (o * m + j) * inner;
        for (let i = 0; i < inner; i++) r[dst + i] += gv[src + i];
      }
      return [T.make(r, x.shape)];
    }, { axis: a });
  });
  /** an embedding lookup: ids of any shape -> ids.shape + [dim] */
  T.embedding = (weight, ids) => T.take(weight, ids, 0, 'embedding');
  /** out[..i..] = x[..index[..i..]..] along axis; index has x's rank */
  T.def('gather', (x, axis, index) => {
    const a = T.normAxis(axis, x.ndim), iv = idxValues(index), ish = idxShape(index);
    const { outer, n, inner } = T.splitAxis(x.shape, a), m = ish[a], v = x.values(), out = new (F(x))(iv.length);
    if (T.prod(ish) !== outer * m * inner) throw new Error('gather: index shape [' + ish + '] must match [' + x.shape + '] off the axis');
    for (let o = 0; o < outer; o++) for (let j = 0; j < m; j++) for (let i = 0; i < inner; i++) {
      const p = (o * m + j) * inner + i;
      out[p] = v[(o * n + (iv[p] | 0)) * inner + i];
    }
    return T.record(T.make(out, ish.slice()), 'gather', [x], (g) => {
      const gv = g.values(), r = new (F(g))(x.size);
      for (let o = 0; o < outer; o++) for (let j = 0; j < m; j++) for (let i = 0; i < inner; i++) {
        const p = (o * m + j) * inner + i;
        r[(o * n + (iv[p] | 0)) * inner + i] += gv[p];
      }
      return [T.make(r, x.shape)];
    }, { axis: a });
  });
  /** one-hot rows for integer labels */
  T.oneHot = (labels, k) => { const lv = idxValues(labels), t = T.zeros([lv.length, k]); for (let i = 0; i < lv.length; i++) t.data[i * k + (lv[i] | 0)] = 1; return t; };
})(globalThis.Nooral = globalThis.Nooral || {});
