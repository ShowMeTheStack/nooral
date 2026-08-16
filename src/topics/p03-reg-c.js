// Part 3, elastic net: a mix of the lasso and ridge penalties. Its ball keeps the diamond's corners on the axes
// (so weights can still be exactly zero) but bulges between them (so correlated features share the weight instead
// of one taking it all). The paths (ML.enetPath) of the same data under the lasso and under the mix, side by side.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, C = () => S.Kit.curve;

  S.topic({
    id: 'elastic-net', family: 'p03-reg', order: 30, title: 'Elastic net: lasso and ridge mixed',
    blurb: 'Elastic net penalises λ·(r·Σ|w| + (1 − r)/2·Σw²): r = 1 is the lasso, r = 0 is ridge. Its ball still has corners on the axes, so it can set weights to exactly zero and select features. Between the corners it bulges like a disc, so when two features carry the same information it spreads the weight over both instead of picking one at random, as the lasso does.',
    inputs: [
      { id: 'r', kind: 'slider', label: 'mix r (1 = lasso)', def: 0.5, min: 0.1, max: 0.9, step: 0.05 },
      { id: 'steps', kind: 'number', label: 'values of λ', def: 16, min: 6, max: 30 },
      { id: 'seed', kind: 'number', label: 'seed', def: 3, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Half and half', values: { r: 0.5, steps: 16, seed: 3 } },
      { label: 'Mostly lasso (r = 0.9)', values: { r: 0.9, steps: 16, seed: 3 } },
      { label: 'Mostly ridge (r = 0.1)', values: { r: 0.1, steps: 16, seed: 3 } },
      { label: 'Other twins', values: { r: 0.5, steps: 20, seed: 8 } },
    ],
    counters: [{ id: 'nz', label: 'non-zero (elastic net)' }],
    run(inp) {
      const raw = H.regData('pair', inp.seed), st = H.standardise(raw.X), X = st.X, y = raw.y, names = raw.names, d = names.length, r = inp.r, K = inp.steps;
      const pl = ML().enetPath(X, y, { l1Ratio: 1, nAlphas: K, eps: 1e-2 }), pe = ML().enetPath(X, y, { l1Ratio: r, nAlphas: K, eps: 1e-2 });
      const F = new S.Frames(['nz']);
      // the balls: the diamond (r = 1), the disc (r = 0) and the mix, all through (1, 0) and (0, 1)
      const bax = H.axes({ prefix: 'eb', x: 40, y: 70, w: 250, h: 250, xr: [-1.3, 1.3], yr: [-1.3, 1.3], xticks: [-1, 0, 1], yticks: [-1, 0, 1], xname: 'w₁', yname: 'w₂', grid: true });
      const ball = (rr, key, cls) => {
        const c = rr + (1 - rr) / 2, pts = [];
        for (let i = 0; i <= 180; i++) {
          const th = (2 * Math.PI * i) / 180, s = Math.abs(Math.cos(th)) + Math.abs(Math.sin(th));
          const rho = rr >= 1 ? c / s : rr <= 0 ? Math.sqrt(2 * c) : (-rr * s + Math.sqrt(rr * rr * s * s + 2 * (1 - rr) * c)) / (1 - rr);
          pts.push([bax.X(rho * Math.cos(th)), bax.Y(rho * Math.sin(th))]);
        }
        return { key, kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + H.r2(p[0]) + ',' + H.r2(p[1])).join(' ') + ' Z', cls, audit: 'ignore', layer: 'back', z: 6 };
      };
      const balls = [ball(1, 'eb:l1', 'pl-seg dash'), ball(0, 'eb:l2', 'pl-seg dash'), ball(r, 'eb:mix', 'pl-ball')];
      const legend = C().legend('eb:lg', 40, 392, [{ text: 'lasso (r = 1), ridge (r = 0)', cls: 'faint' }, { text: 'mix r = ' + n$(r), cls: 'act' }], { dir: 'col' });
      // the two paths
      const lg = (v) => Math.log10(v);
      const allW = pl.coefs.flat().concat(pe.coefs.flat()).map(Math.abs), wm = Math.max(0.5, ...allW) * 1.1, yr = H.nice([-0.3, wm], 0);
      const mk = (prefix, y0, path, title) => {
        const xs = path.alphas.map(lg), xr = [Math.floor(Math.min(...xs) * 2) / 2, Math.ceil(Math.max(...xs) * 2) / 2];
        const ax = H.axes({ prefix, x: 400, y: y0, w: 350, h: 170, xr, yr, xticks: H.tickList(xr[0], xr[1], xr[1] - xr[0] > 3 ? 1 : 0.5).filter(Number.isInteger), yticks: H.tickList(0, yr[1], yr[1] > 3 ? 1 : 0.5), fmtX: (v) => '10' + (v < 0 ? '⁻' : '') + H.sup(Math.abs(v)), xname: 'λ, stronger →', grid: true });
        return { ax, items: [...ax.items, H.title(prefix + ':t', 400, y0 - 26, title), C().hline(ax, prefix + ':z', 0, '')] };
      };
      const A = mk('ea', 50, pl, 'Lasso path'), B = mk('ee', 300, pe, 'Elastic net path, mix ' + n$(r));
      const cls = (j) => 'f' + j;
      const series = (P, path, k) => names.map((nm, j) => C().line(P.ax, P.ax.o.x + ':' + nm + j + (P === A ? 'a' : 'e'), path.alphas.slice(0, k + 1).map((a, q) => [lg(a), path.coefs[q][j]]), cls(j) + ' thin'));
      const keyed = (P, path, k, tag) => series(P, path, k).map((it, j) => Object.assign(it, { key: tag + ':s' + j }));
      const names2 = C().legend('en:lg', 400, 520, names.map((nm, j) => ({ text: nm, cls: cls(j) })), {});
      const head = () => [...bax.items, H.title('eb:t', 40, 30, 'The balls'), ...balls, ...legend, ...A.items, ...B.items, ...names2];
      const meta0 = { r, d, n: X.length };
      F.add({ items: head(), say: 'Two of the ' + d + ' features are near copies of each other (twin 1 and twin 2), and both matter; the other four are noise. Left: the elastic-net ball for r = ' + n$(r) + ' (shaded) between the lasso diamond and the ridge disc. It keeps the corners on the axes but bulges between them.', meta: Object.assign({ phase: 'balls' }, meta0) });
      for (let k = 0; k < K; k++) {
        const wl = pl.coefs[k], we = pe.coefs[k], nzl = wl.filter((v) => v !== 0).length, nze = we.filter((v) => v !== 0).length;
        F.set('nz', nze);
        const items = [...head(), ...keyed(A, pl, k, 'ea'), ...keyed(B, pe, k, 'ee'), C().vline(A.ax, 'ea:at', lg(pl.alphas[k]), 'act'), C().vline(B.ax, 'ee:at', lg(pe.alphas[k]), 'act')];
        items.push(...H.eqs('eb:eq', 40, 450, [['lasso: twins ' + n$(wl[0]) + ', ' + n$(wl[1]), 'muted'], ['mix:   twins ' + n$(we[0]) + ', ' + n$(we[1]), 'act']]));
        const share = (w) => (Math.abs(w[0]) + Math.abs(w[1]) > 0 ? Math.min(Math.abs(w[0]), Math.abs(w[1])) / Math.max(Math.abs(w[0]), Math.abs(w[1])) : 1);
        let say = 'Step ' + (k + 1) + ' of ' + K + ': lasso at λ = ' + n$(pl.alphas[k]) + ' gives the twins ' + n$(wl[0]) + ' and ' + n$(wl[1]) + ' (' + nzl + ' non-zero in all); elastic net at λ = ' + n$(pe.alphas[k]) + ' gives ' + n$(we[0]) + ' and ' + n$(we[1]) + ' (' + nze + ' non-zero).';
        if (k === 0) say = 'At the strongest λ both models set every weight to 0: λ = ' + n$(pl.alphas[k]) + ' for the lasso and ' + n$(pe.alphas[k]) + ' for elastic net (the mix needs a larger λ because only the share r of it is L1).';
        else if ((wl[0] === 0) !== (wl[1] === 0) && we[0] !== 0 && we[1] !== 0) say += ' The lasso has picked one twin; the mix uses both.';
        F.add({ items, say, meta: Object.assign({ phase: 'path', k, step: k + 1, K, al: pl.alphas[k], ae: pe.alphas[k], wl, we, nzl, nze, share: share(we) }, meta0) });
      }
      const wl = pl.coefs[K - 1], we = pe.coefs[K - 1];
      const firstOne = pl.coefs.findIndex((w) => (w[0] === 0) !== (w[1] === 0)), bothIn = pl.coefs.findIndex((w) => w[0] !== 0 && w[1] !== 0), lag = firstOne >= 0 ? (bothIn >= 0 ? bothIn : K) - firstOne : 0;
      F.add({ items: F.last().items, say: 'Along the whole path, ' + (firstOne >= 0 ? (bothIn >= 0 ? 'the lasso let one twin in ' + lag + ' steps before the other' : 'the lasso kept one twin out all the way') : 'the lasso let both twins in together') + ', while elastic net moved them together. At the weakest λ shown: lasso ' + n$(wl[0]) + ' and ' + n$(wl[1]) + ', elastic net ' + n$(we[0]) + ' and ' + n$(we[1]) + '. The noise features stay near zero in both.', meta: Object.assign({ phase: 'end', wl, we, firstOne, lag }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Elastic net', text: ['(1/2n)‖y − Xw‖² + λ (r Σ|wⱼ| + (1 − r)/2 Σ wⱼ²)', 'r = 1: lasso   r = 0: ridge', 'coordinate step: wⱼ = S(ρⱼ, nλr) / (‖xⱼ‖² + nλ(1 − r))', 'corners on the axes: exact zeros', 'curved edges: correlated features share'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'balls', pick: 'option', ask: 'Two features are near copies and both matter. What will the lasso do with them?', options: ['bring one in well before the other', 'bring both in together with equal weights', 'drop both', 'make one negative'], answer: (fr) => (fr.some((f) => f.meta.phase === 'path' && (f.meta.wl[0] === 0) !== (f.meta.wl[1] === 0)) ? 'bring one in well before the other' : 'bring both in together with equal weights'), explain: 'Once one twin is in, the other has almost nothing left to explain, so the L1 penalty keeps it out.' },
    ],
    quiz: [
      { q: 'Elastic net with r = 1 is…', options: ['the lasso', 'ridge', 'least squares', 'none of these'], answer: 0, why: 'The L2 part has weight 1 − r = 0.' },
      { q: 'Why can elastic net still give exact zeros?', options: ['its ball keeps the corners on the axes', 'it rounds small weights', 'its ball is a disc', 'it cannot'], answer: 0, why: 'Any r > 0 leaves a kink at each axis.' },
      { q: 'With two highly correlated useful features, elastic net tends to…', options: ['give both similar weights', 'keep only one', 'drop both', 'make them opposite'], answer: 0, why: 'The squared part of the penalty is smaller when the weight is split.' },
      { q: 'In the coordinate step, what does the L2 part of the penalty change?', options: ['it adds nλ(1 − r) to the denominator, shrinking every weight', 'the threshold', 'the sign', 'nothing'], answer: 0, why: 'wⱼ = S(ρⱼ, nλr) / (‖xⱼ‖² + nλ(1 − r)).' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
