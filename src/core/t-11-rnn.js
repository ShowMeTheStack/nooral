// Recurrent layers over sequences x [B, T, I] -> outputs [B, T, H]. Each forward keeps, per time step, the hidden
// state and the gate values in layer.steps (tensors in the graph), so the pictures can show a cell opening and closing.
//   new nn.RNN(I, H, {act: 'tanh', rng})   new nn.LSTM(I, H, {rng})   new nn.GRU(I, H, {rng})
//   layer.forward(x, state?) -> out [B, T, H], with out.h (and out.c for the LSTM) the final state
//   LSTM gate order in the weights: input, forget, cell, output (forget bias starts at 1). GRU: reset, update, new.
(function (S) {
  'use strict';
  const T = S.T, nn = T.nn;
  const rngOf = (o) => T.asRng(o && o.rng !== undefined ? o.rng : 1);
  const at = (t, k) => T.slice(t, null, k, null); // [B, T, X] -> [B, X] at step k
  const part = (t, j, H) => T.slice(t, null, [j * H, (j + 1) * H]); // gate j of [B, G·H]

  class Recurrent extends nn.Module {
    constructor(I, H, G, o) {
      super();
      const g = rngOf(o), a = 1 / Math.sqrt(H);
      this.I = I; this.H = H;
      this.Wx = this.param('Wx', T.uniform([I, G * H], -a, a, g));
      this.Wh = this.param('Wh', T.uniform([H, G * H], -a, a, g));
      this.b = this.param('b', T.zeros([G * H]));
    }
    zeros(B) { return T.zeros([B, this.H]); }
  }
  nn.RNN = class RNN extends Recurrent {
    constructor(I, H, o = {}) { super(I, H, 1, o); this.act = o.act || 'tanh'; }
    forward(x, h0) {
      const [B, n] = x.shape, xw = T.add(T.matmul(x, this.Wx), this.b);
      let h = h0 || this.zeros(B);
      const hs = [];
      this.steps = [];
      for (let k = 0; k < n; k++) {
        h = T[this.act](T.add(at(xw, k), T.matmul(h, this.Wh)));
        hs.push(h); this.steps.push({ h });
      }
      const out = T.stack(hs, 1);
      out.h = h;
      return out;
    }
  };
  nn.LSTM = class LSTM extends Recurrent {
    constructor(I, H, o = {}) { super(I, H, 4, o); for (let i = H; i < 2 * H; i++) this.b.data[i] = 1; }
    forward(x, state) {
      const [B, n] = x.shape, H = this.H, xw = T.add(T.matmul(x, this.Wx), this.b);
      let h = state ? state.h : this.zeros(B), c = state ? state.c : this.zeros(B);
      const hs = [];
      this.steps = [];
      for (let k = 0; k < n; k++) {
        const z = T.add(at(xw, k), T.matmul(h, this.Wh));
        const i = T.sigmoid(part(z, 0, H)), f = T.sigmoid(part(z, 1, H)), g = T.tanh(part(z, 2, H)), o = T.sigmoid(part(z, 3, H));
        c = T.add(T.mul(f, c), T.mul(i, g));
        h = T.mul(o, T.tanh(c));
        hs.push(h); this.steps.push({ i, f, g, o, c, h });
      }
      const out = T.stack(hs, 1);
      out.h = h; out.c = c;
      return out;
    }
  };
  nn.GRU = class GRU extends Recurrent {
    constructor(I, H, o = {}) { super(I, H, 3, o); this.bh = this.param('bh', T.zeros([3 * H])); }
    forward(x, h0) {
      const [B, n] = x.shape, H = this.H, xw = T.add(T.matmul(x, this.Wx), this.b);
      let h = h0 || this.zeros(B);
      const hs = [];
      this.steps = [];
      for (let k = 0; k < n; k++) {
        const xk = at(xw, k), hh = T.add(T.matmul(h, this.Wh), this.bh);
        const r = T.sigmoid(T.add(part(xk, 0, H), part(hh, 0, H)));
        const u = T.sigmoid(T.add(part(xk, 1, H), part(hh, 1, H)));
        const cand = T.tanh(T.add(part(xk, 2, H), T.mul(r, part(hh, 2, H))));
        h = T.add(T.mul(T.sub(1, u), cand), T.mul(u, h));
        hs.push(h); this.steps.push({ r, z: u, n: cand, h });
      }
      const out = T.stack(hs, 1);
      out.h = h;
      return out;
    }
  };
})(globalThis.Nooral = globalThis.Nooral || {});
