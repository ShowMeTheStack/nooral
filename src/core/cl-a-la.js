// S.ML.la: dense linear algebra on arrays of rows, in float64. Small and exact enough to match
// reference results to about 1e-10 on the sizes the course uses (up to a few hundred columns).
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const la = (ML.la = {});

  la.zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  la.eye = (n) => { const A = la.zeros(n, n); for (let i = 0; i < n; i++) A[i][i] = 1; return A; };
  la.copy = (A) => A.map((r) => r.slice());
  la.T = (A) => { const r = A.length, c = A[0].length, B = la.zeros(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) B[j][i] = A[i][j]; return B; };
  la.dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  la.norm = (a) => Math.sqrt(la.dot(a, a));
  la.matvec = (A, x) => A.map((r) => la.dot(r, x));
  la.matmul = function (A, B) {
    const n = A.length, m = B[0].length, k = B.length, C = la.zeros(n, m);
    for (let i = 0; i < n; i++) {
      const Ai = A[i], Ci = C[i];
      for (let t = 0; t < k; t++) { const a = Ai[t]; if (a === 0) continue; const Bt = B[t]; for (let j = 0; j < m; j++) Ci[j] += a * Bt[j]; }
    }
    return C;
  };
  // X^T W X (W diagonal weights, optional)
  la.gram = function (X, w) {
    const d = X[0].length, G = la.zeros(d, d);
    for (let i = 0; i < X.length; i++) {
      const r = X[i], wi = w ? w[i] : 1;
      for (let a = 0; a < d; a++) { const v = wi * r[a]; if (v === 0) continue; const Ga = G[a]; for (let b = a; b < d; b++) Ga[b] += v * r[b]; }
    }
    for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
    return G;
  };

  // LU with partial pivoting; returns null when singular
  la.lu = function (A) {
    const n = A.length, L = la.copy(A), piv = [...Array(n).keys()];
    let sign = 1;
    for (let k = 0; k < n; k++) {
      let m = k;
      for (let i = k + 1; i < n; i++) if (Math.abs(L[i][k]) > Math.abs(L[m][k])) m = i;
      if (L[m][k] === 0) return null;
      if (m !== k) { [L[m], L[k]] = [L[k], L[m]]; [piv[m], piv[k]] = [piv[k], piv[m]]; sign = -sign; }
      for (let i = k + 1; i < n; i++) {
        const f = (L[i][k] /= L[k][k]);
        if (f !== 0) for (let j = k + 1; j < n; j++) L[i][j] -= f * L[k][j];
      }
    }
    return { L, piv, sign };
  };
  la.luSolve = function (F, b) {
    const n = F.L.length, x = F.piv.map((i) => b[i]);
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) x[i] -= F.L[i][j] * x[j];
    for (let i = n - 1; i >= 0; i--) { for (let j = i + 1; j < n; j++) x[i] -= F.L[i][j] * x[j]; x[i] /= F.L[i][i]; }
    return x;
  };
  la.solve = function (A, b) {
    const F = la.lu(A);
    if (!F) throw new Error('singular matrix');
    if (Array.isArray(b[0])) { const B = la.T(b); return la.T(B.map((col) => la.luSolve(F, col))); }
    return la.luSolve(F, b);
  };
  la.inv = (A) => la.solve(A, la.eye(A.length));
  la.det = (A) => { const F = la.lu(A); if (!F) return 0; let d = F.sign; for (let i = 0; i < A.length; i++) d *= F.L[i][i]; return d; };

  // Cholesky A = L L^T (lower); null when not positive definite
  la.chol = function (A) {
    const n = A.length, L = la.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) { if (s <= 0) return null; L[i][i] = Math.sqrt(s); } else L[i][j] = s / L[j][j];
      }
    }
    return L;
  };
  la.cholSolve = function (L, b) {
    const n = L.length, x = b.slice();
    for (let i = 0; i < n; i++) { for (let k = 0; k < i; k++) x[i] -= L[i][k] * x[k]; x[i] /= L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { for (let k = i + 1; k < n; k++) x[i] -= L[k][i] * x[k]; x[i] /= L[i][i]; }
    return x;
  };
  la.logdetChol = (L) => { let s = 0; for (let i = 0; i < L.length; i++) s += Math.log(L[i][i]); return 2 * s; };
  // inverse of a lower-triangular matrix
  la.triInv = function (L) {
    const n = L.length, M = la.zeros(n, n);
    for (let i = 0; i < n; i++) {
      M[i][i] = 1 / L[i][i];
      for (let j = 0; j < i; j++) { let s = 0; for (let k = j; k < i; k++) s -= L[i][k] * M[k][j]; M[i][j] = s / L[i][i]; }
    }
    return M;
  };

  // Householder QR of an n x d matrix (n >= d). Returns R (d x d) and a function applying Q^T.
  la.qr = function (A) {
    const n = A.length, d = A[0].length, R = la.copy(A), vs = [];
    for (let k = 0; k < Math.min(n - 1, d); k++) {
      let nrm = 0;
      for (let i = k; i < n; i++) nrm += R[i][k] * R[i][k];
      nrm = Math.sqrt(nrm);
      if (nrm === 0) { vs.push(null); continue; }
      const alpha = R[k][k] > 0 ? -nrm : nrm, v = new Array(n).fill(0);
      v[k] = R[k][k] - alpha;
      for (let i = k + 1; i < n; i++) v[i] = R[i][k];
      const vv = la.dot(v, v);
      if (vv === 0) { vs.push(null); continue; }
      for (let j = k; j < d; j++) {
        let s = 0;
        for (let i = k; i < n; i++) s += v[i] * R[i][j];
        const f = (2 * s) / vv;
        for (let i = k; i < n; i++) R[i][j] -= f * v[i];
      }
      vs.push(v);
    }
    const Qt = (b) => {
      const x = b.slice();
      vs.forEach((v, k) => { if (!v) return; let s = 0; for (let i = k; i < n; i++) s += v[i] * x[i]; const f = (2 * s) / la.dot(v, v); for (let i = k; i < n; i++) x[i] -= f * v[i]; });
      return x;
    };
    return { R: R.slice(0, d).map((r) => r.slice()), Qt };
  };
  // least squares min ||A x - b|| through QR
  la.lstsq = function (A, b) {
    const d = A[0].length, { R, Qt } = la.qr(A), c = Qt(b), x = new Array(d).fill(0);
    for (let i = d - 1; i >= 0; i--) {
      let s = c[i];
      for (let j = i + 1; j < d; j++) s -= R[i][j] * x[j];
      x[i] = Math.abs(R[i][i]) < 1e-12 * Math.abs(R[0][0] || 1) ? 0 : s / R[i][i];
    }
    return { x, R };
  };

  // the least-norm least-squares solution through the SVD: of all x that fit b best (there are many when A has more
  // columns than rows, or dependent columns), the shortest; singular values below rcond * the largest count as zero
  la.lstsqMinNorm = function (A, b, rcond) {
    const d = A[0].length, { U, s, V } = la.svd(A), tol = (rcond || Math.max(A.length, d) * 2.220446049250313e-16) * (s[0] || 0);
    const x = new Array(d).fill(0);
    let rank = 0;
    for (let k = 0; k < s.length; k++) {
      if (!(s[k] > tol)) continue;
      rank++;
      let c = 0;
      for (let i = 0; i < A.length; i++) c += U[i][k] * b[i];
      c /= s[k];
      for (let j = 0; j < d; j++) x[j] += c * V[j][k];
    }
    return { x, s, rank };
  };

  // symmetric eigen-decomposition by cyclic Jacobi; values descending, vectors as columns of V
  la.symEig = function (A, opts) {
    const n = A.length, a = la.copy(A), V = la.eye(n), maxSweeps = (opts && opts.sweeps) || 100;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-30) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
        for (let k = 0; k < n; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
        for (let k = 0; k < n; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
      }
    }
    const order = [...Array(n).keys()].sort((i, j) => a[j][j] - a[i][i]);
    return { values: order.map((i) => a[i][i]), vectors: V.map((r) => order.map((i) => r[i])) };
  };

  // thin SVD by one-sided Jacobi: A (n x d) = U diag(s) V^T, s descending
  la.svd = function (A) {
    const n = A.length, d = A[0].length;
    if (n < d) { const r = la.svd(la.T(A)); return { U: r.V, s: r.s, V: r.U }; }
    const U = la.copy(A), V = la.eye(d);
    for (let sweep = 0; sweep < 80; sweep++) {
      let rot = 0;
      for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) {
        let al = 0, be = 0, ga = 0;
        for (let i = 0; i < n; i++) { al += U[i][p] * U[i][p]; be += U[i][q] * U[i][q]; ga += U[i][p] * U[i][q]; }
        if (Math.abs(ga) <= 1e-15 * Math.sqrt(al * be) || ga === 0) continue;
        rot++;
        const z = (be - al) / (2 * ga), t = Math.sign(z || 1) / (Math.abs(z) + Math.sqrt(1 + z * z)), c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let i = 0; i < n; i++) { const x = U[i][p], y = U[i][q]; U[i][p] = c * x - s * y; U[i][q] = s * x + c * y; }
        for (let i = 0; i < d; i++) { const x = V[i][p], y = V[i][q]; V[i][p] = c * x - s * y; V[i][q] = s * x + c * y; }
      }
      if (!rot) break;
    }
    const s = [...Array(d).keys()].map((j) => { let t = 0; for (let i = 0; i < n; i++) t += U[i][j] * U[i][j]; return Math.sqrt(t); });
    const order = [...Array(d).keys()].sort((i, j) => s[j] - s[i]);
    return {
      U: U.map((r) => order.map((j) => (s[j] > 0 ? r[j] / s[j] : 0))),
      s: order.map((j) => s[j]),
      V: V.map((r) => order.map((j) => r[j])),
    };
  };

  // covariance of rows (ddof 0 or 1), optionally weighted by w and around a given mean
  la.cov = function (X, ddof = 0, mean) {
    const d = X[0].length, m = mean || ML.colMeans(X), C = la.zeros(d, d);
    for (const r of X) for (let a = 0; a < d; a++) { const u = r[a] - m[a]; for (let b = a; b < d; b++) C[a][b] += u * (r[b] - m[b]); }
    const k = X.length - ddof;
    for (let a = 0; a < d; a++) for (let b = a; b < d; b++) { C[a][b] /= k; C[b][a] = C[a][b]; }
    return C;
  };
})();
