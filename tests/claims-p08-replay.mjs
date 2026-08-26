// Part 8's independent working-out: every optimiser's update rule written out by hand in plain JavaScript (no
// engine), the schedules from their formulas, gradients and Hessians by finite differences, and the eigenvalues of
// a 2 × 2 matrix from its characteristic polynomial. The claims check replays every run with these and compares.

/** one optimiser, by its rule: returns step(w, g, lr) -> new w, and exposes its state */
export function rule(name, o = {}) {
  const st = { t: 0 };
  const mu = o.momentum || 0, nest = !!o.nesterov;
  const steps = {
    sgd(w, g, lr) {
      if (!mu) return w.map((x, i) => x - lr * g[i]);
      st.v = st.v ? st.v.map((v, i) => mu * v + g[i]) : g.slice();
      return w.map((x, i) => x - lr * (nest ? g[i] + mu * st.v[i] : st.v[i]));
    },
    adagrad(w, g, lr) {
      st.sumSq = (st.sumSq || [0, 0]).map((h, i) => h + g[i] * g[i]);
      return w.map((x, i) => x - (lr * g[i]) / (Math.sqrt(st.sumSq[i]) + 1e-10));
    },
    rmsprop(w, g, lr) {
      st.sqAvg = (st.sqAvg || [0, 0]).map((s, i) => 0.99 * s + (1 - 0.99) * g[i] * g[i]);
      return w.map((x, i) => x - lr * (g[i] / (Math.sqrt(st.sqAvg[i]) + 1e-8)));
    },
    adam(w, g, lr) {
      st.t++;
      st.m = (st.m || [0, 0]).map((m, i) => 0.9 * m + (1 - 0.9) * g[i]);
      st.v = (st.v || [0, 0]).map((v, i) => 0.999 * v + (1 - 0.999) * g[i] * g[i]);
      const c1 = 1 - Math.pow(0.9, st.t), c2 = 1 - Math.pow(0.999, st.t);
      return w.map((x, i) => x - (lr * (st.m[i] / c1)) / (Math.sqrt(st.v[i] / c2) + 1e-8));
    },
  };
  const id = name === 'momentum' || name === 'nesterov' ? 'sgd' : name;
  if (name === 'momentum' && !o.momentum) o.momentum = 0.9;
  return { st, step: (w, g, lr) => steps[id](w, g, lr) };
}
/** replay: f, grad(p, k), start, steps, lr or sched(t) -> {pts, loss, grads, state: [copies], lrs, diverged} */
export function replay(name, opts, f, grad, start, steps, sched) {
  const R = rule(name, Object.assign({}, opts)), pts = [start.slice()], loss = [f(start)], grads = [], state = [], lrs = [];
  let w = start.slice(), diverged = false;
  for (let k = 0; k < steps; k++) {
    const g = grad(w, k), lr = sched ? sched(k) : opts.lr;
    grads.push(g); lrs.push(lr);
    w = R.step(w, g, lr);
    state.push(JSON.parse(JSON.stringify(R.st)));
    pts.push(w.slice()); loss.push(f(w));
    if (!w.every(Number.isFinite) || Math.abs(w[0]) > 1e6 || Math.abs(w[1]) > 1e6) { diverged = true; break; }
  }
  return { pts, loss, grads, state, lrs, diverged };
}
// ---- schedules by formula ----
export const SCHED = {
  constant: (lr) => () => lr,
  step: (lr, every, gamma) => (t) => lr * Math.pow(gamma, Math.floor(t / every)),
  exp: (lr, gamma) => (t) => lr * Math.pow(gamma, t),
  cosine: (lr, total) => (t) => 0.5 * lr * (1 + Math.cos((Math.PI * Math.min(t, total)) / total)),
  warmup: (lr, W) => (t) => (t < W ? (lr * (t + 1)) / W : lr),
};
// ---- calculus by finite differences ----
export const fdGrad = (f, p, h = 1e-6) => [0, 1].map((j) => { const a = p.slice(), b = p.slice(); a[j] += h; b[j] -= h; return (f(a) - f(b)) / (2 * h); });
export const fdHess = (g, p, h = 1e-5) => [0, 1].map((j) => { const a = p.slice(), b = p.slice(); a[j] += h; b[j] -= h; const ga = g(a), gb = g(b); return [(ga[0] - gb[0]) / (2 * h), (ga[1] - gb[1]) / (2 * h)]; });
/** eigenvalues of a symmetric 2 × 2 matrix as the roots of λ² − (trace)λ + det */
export const eigs = (m) => { const tr = m[0][0] + m[1][1], det = m[0][0] * m[1][1] - m[0][1] * m[1][0], disc = Math.sqrt(Math.max(0, tr * tr - 4 * det)); return [(tr + disc) / 2, (tr - disc) / 2]; };
export const rel = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(Math.abs(a), Math.abs(b)) + 1e-12;
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
