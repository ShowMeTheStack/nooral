// Scaled dot-product attention, and the losses. Attention is built from recorded ops, so its gradient comes from
// theirs and every intermediate (scores, weights) can be inspected.
//   attention(q, k, v, {mask, causal, scale, dropout, rng, training}) -> out, with out.attn = {scores, weights}
//      q [..., Tq, d], k [..., Tk, d], v [..., Tk, dv]; mask broadcasts to [..., Tq, Tk], non-zero = may attend
//   causalMask(T) -> [T, T] with ones on and below the diagonal
//   mseLoss(pred, target)  crossEntropy(logits, target, {labelSmoothing, ignoreIndex})  (target: class ids or probabilities)
//   bceLoss(prob, target)  bceWithLogits(logits, target)  hingeLoss(scores, target ±1)  multiHinge(scores, ids, margin)
//   klDiv(logQ, P)  Σ P (log P − log Q) / batch     nllLoss(logProbs, ids)
// Every loss returns the mean over the batch (reduction 'sum' or 'none' through opts.reduction where offered).
(function (S) {
  'use strict';
  const T = S.T;

  T.causalMask = (n, m = n) => { const t = T.zeros([n, m]); for (let i = 0; i < n; i++) for (let j = 0; j <= Math.min(i + m - n, m - 1); j++) t.data[i * m + j] = 1; return t; };
  T.attention = (q, k, v, opts = {}) => {
    const d = q.shape[q.ndim - 1], scale = opts.scale || 1 / Math.sqrt(d);
    let scores = T.scale(T.matmul(q, T.transpose(k)), scale).named(opts.name ? opts.name + '.scores' : null);
    let mask = opts.mask ? T.as(opts.mask) : null;
    if (opts.causal) { const cm = T.causalMask(q.shape[q.ndim - 2], k.shape[k.ndim - 2]); mask = mask ? T.mul(mask, cm) : cm; }
    if (mask) scores = T.maskedFill(scores, T.eq(mask, 0), -1e9);
    let w = T.softmax(scores, -1).named(opts.name ? opts.name + '.weights' : null);
    const weights = w;
    if (opts.dropout && opts.training !== false) w = T.dropout(w, opts.dropout, opts.rng, true);
    const out = T.matmul(w, v);
    out.attn = { scores, weights, mask };
    return out;
  };

  const reduceWith = (t, opts) => (opts && opts.reduction === 'none' ? t : opts && opts.reduction === 'sum' ? T.sum(t) : T.mean(t));
  T.mseLoss = (pred, target, opts) => { const d = T.sub(pred, T.as(target)); return reduceWith(T.mul(d, d), opts); };
  T.l1Loss = (pred, target, opts) => reduceWith(T.abs(T.sub(pred, T.as(target))), opts);
  T.huberLoss = (pred, target, delta = 1, opts) => {
    const a = T.abs(T.sub(pred, T.as(target))), q = T.minimum(a, delta);
    return reduceWith(T.add(T.scale(T.mul(q, q), 0.5), T.scale(T.sub(a, q), delta)), opts);
  };
  /** negative log-likelihood of class ids under log-probabilities [N, C] (or [..., C] with ids [...]) */
  T.nllLoss = (logp, ids, opts = {}) => {
    const C = logp.shape[logp.ndim - 1], lp = T.reshape(logp, [-1, C]);
    const iv = ids instanceof T.Tensor ? ids.values() : Float64Array.from(ids.flat ? ids.flat(Infinity) : ids);
    const N = iv.length, keep = [], idx = new Float64Array(N);
    for (let i = 0; i < N; i++) { const ign = opts.ignoreIndex !== undefined && iv[i] === opts.ignoreIndex; idx[i] = ign ? 0 : iv[i]; if (!ign) keep.push(i); }
    let picked = T.reshape(T.gather(lp, 1, T.make(idx, [N, 1])), [N]);
    if (keep.length < N) {
      const w = T.zeros([N]); for (const i of keep) w.data[i] = 1;
      picked = T.mul(picked, w);
      const s = T.neg(T.sum(picked));
      return opts.reduction === 'sum' ? s : T.scale(s, 1 / Math.max(1, keep.length));
    }
    return opts.reduction === 'none' ? T.neg(picked) : opts.reduction === 'sum' ? T.neg(T.sum(picked)) : T.neg(T.mean(picked));
  };
  /** cross-entropy from logits: target class ids, or a probability row per example */
  T.crossEntropy = (logits, target, opts = {}) => {
    const logp = T.logSoftmax(logits, -1), C = logits.shape[logits.ndim - 1];
    const tt = target instanceof T.Tensor ? target : null;
    if (tt && tt.shape[tt.ndim - 1] === C && tt.ndim === logits.ndim) {
      const per = T.neg(T.sum(T.mul(logp, tt), -1));
      return reduceWith(per, opts);
    }
    const nll = T.nllLoss(logp, target, opts);
    if (!opts.labelSmoothing) return nll;
    const e = opts.labelSmoothing, smooth = T.neg(T.mean(T.mean(logp, -1)));
    return T.add(T.scale(nll, 1 - e), T.scale(smooth, e));
  };
  T.bceLoss = (p, y, opts) => {
    y = T.as(y);
    const eps = 1e-7, pc = T.clamp(p, eps, 1 - eps);
    return reduceWith(T.neg(T.add(T.mul(y, T.log(pc)), T.mul(T.sub(1, y), T.log(T.sub(1, pc))))), opts);
  };
  /** stable: max(z, 0) − z·y + log(1 + e^(−|z|)) */
  T.bceWithLogits = (z, y, opts) => {
    y = T.as(y);
    return reduceWith(T.add(T.sub(T.relu(z), T.mul(z, y)), T.softplus(T.neg(T.abs(z)))), opts);
  };
  /** binary hinge: mean(max(0, 1 − y·s)) with y in {−1, +1} */
  T.hingeLoss = (s, y, opts) => reduceWith(T.relu(T.sub(1, T.mul(s, T.as(y)))), opts);
  /** multi-class hinge: Σ_{j≠y} max(0, s_j − s_y + margin), averaged over the batch */
  T.multiHinge = (scores, ids, margin = 1) => {
    const N = scores.shape[0], C = scores.shape[1], iv = ids instanceof T.Tensor ? ids.values() : ids;
    const sy = T.gather(scores, 1, T.make(Float64Array.from(iv), [N, 1]));
    const m = T.relu(T.addScalar(T.sub(scores, sy), margin)), notY = T.ones([N, C]);
    for (let i = 0; i < N; i++) notY.data[i * C + iv[i]] = 0;
    return T.scale(T.sum(T.mul(m, notY)), 1 / N);
  };
  /** KL(P ‖ Q) with Q given as log-probabilities, averaged over the batch (rows) */
  T.klDiv = (logQ, P, opts = {}) => {
    P = T.as(P);
    const pv = P.values(), lp = new (P.data.constructor)(pv.length);
    for (let i = 0; i < pv.length; i++) lp[i] = pv[i] > 0 ? Math.log(pv[i]) : 0;
    const term = T.mul(P, T.sub(T.make(lp, P.shape), logQ));
    const rows = logQ.ndim > 1 ? logQ.shape[0] : 1;
    return opts.reduction === 'sum' ? T.sum(term) : T.scale(T.sum(term), 1 / rows);
  };
  /** accuracy of logits [N, C] against ids */
  T.accuracy = (logits, ids) => {
    const am = T.argmax(logits, -1).values(), iv = ids instanceof T.Tensor ? ids.values() : ids;
    let c = 0; for (let i = 0; i < am.length; i++) if (am[i] === +iv[i]) c++;
    return c / am.length;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
