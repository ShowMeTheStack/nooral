// Part 14, the variational autoencoder run live from its trained weights: the latent map of the test digits with
// what the decoder draws across the plane, and a walk across it decoded point by point; then the reparameterisation
// trick taken apart for one digit.
(function (S) {
  'use strict';
  const P = S.P14, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const R = 3, TILES = 7;
  P.FAR = [2.6, -2.6];

  /** the average test image of each digit, for saying what a decoded image looks most like */
  P.vaeMeans = () => P.once('vaeMeans', () => {
    const v = P.vae(), out = {};
    for (const d of P.VAE_CLASSES) { const m = new Float64Array(256); let n = 0; v.test.y.forEach((y, i) => { if (y !== d) return; n++; for (let k = 0; k < 256; k++) m[k] += v.test.X[i * 256 + k]; }); out[d] = Float64Array.from(m, (s) => s / n); }
    return out;
  });
  /** the digit whose average test image correlates best with an image (the shape counts, not how dark it is) */
  P.corr = (a, b) => { const n = a.length; let ma = 0, mb = 0; for (let k = 0; k < n; k++) { ma += a[k]; mb += b[k]; } ma /= n; mb /= n; let sab = 0, saa = 0, sbb = 0; for (let k = 0; k < n; k++) { const x = a[k] - ma, y = b[k] - mb; sab += x * y; saa += x * x; sbb += y * y; } return sab / Math.sqrt(saa * sbb || 1); };
  P.looksLike = (px) => { const M = P.vaeMeans(); let best = -1, bc = -Infinity; for (const d of P.VAE_CLASSES) { const c = P.corr(px, M[d]); if (c > bc) { bc = c; best = d; } } return { digit: best, corr: bc }; };
  P.nearestCentre = (z) => { const c = P.vae().centres, ds = P.VAE_CLASSES.map((d) => [d, Math.hypot(z[0] - c[d][0], z[1] - c[d][1])]).sort((a, b) => a[1] - b[1]); return { digit: ds[0][0], dist: ds[0][1], second: ds[1][0], dist2: ds[1][1], tie: ds[1][1] - ds[0][1] < 1e-3 }; };
  /** the tiles: what the decoder draws at the centre of each of 7 × 7 squares covering [−3, 3]² */
  P.tiles = () => P.once('tiles', () => {
    const z = []; for (let a = 0; a < TILES; a++) for (let b = 0; b < TILES; b++) z.push([-R + ((b + 0.5) * 2 * R) / TILES, R - ((a + 0.5) * 2 * R) / TILES]);
    return { z, images: P.vae().decode(z) };
  });
  const mapDots = (ax, key, map, faint) => map.map((m, i) => ({ key: key + ':' + i, kind: 'dot', x: ax.X(m.z[0]), y: ax.Y(m.z[1]), r: faint ? 2.6 : 3.6, cls: 'sp ' + P.cOf(m.y) + (faint ? ' faded' : ''), audit: 'overlap-ok', layer: 'items' }));
  const legend = (key, x, y) => C().legend(key, x, y, P.VAE_CLASSES.map((d) => ({ text: 'digit ' + d, cls: P.cOf(d), dot: true })), { dir: 'row', size: 13 });
  const placeOf = (z) => (z === 'far' ? P.FAR : z === 'centre' ? [0, 0] : P.vae().centres[+z]);
  const PLACE_OPTS = [{ value: '0', label: 'the 0s' }, { value: '1', label: 'the 1s' }, { value: '3', label: 'the 3s' }, { value: '7', label: 'the 7s' }, { value: 'centre', label: 'the centre (0, 0)' }, { value: 'far', label: 'far out (2.6, −2.6)' }];

  S.topic({
    id: 'gen-vae-map', family: 'p14-ae', order: 20, title: 'Walking the VAE’s latent map',
    blurb: 'A variational autoencoder is trained so that its codes fill a known region: each digit is encoded as a small cloud around a point, and the clouds are pulled towards a standard normal. The result is a map. Every point on it decodes to an image, and nearby points decode to similar images, so a straight walk from one digit to another passes through shapes in between. This VAE, with a code of 2 numbers, was trained on the digits 0, 1, 3 and 7.',
    inputs: [
      { id: 'from', kind: 'choice', label: 'Walk from', def: '0', options: PLACE_OPTS },
      { id: 'to', kind: 'choice', label: 'Walk to', def: '1', options: PLACE_OPTS },
      { id: 'n', kind: 'number', label: 'Points on the walk', def: 7, min: 3, max: 9 },
    ],
    check: (inp) => (inp.from === inp.to ? 'Choose two different places' : null),
    presets: [
      { label: 'From the 0s to the 1s', values: { from: '0', to: '1', n: 7 } },
      { label: 'From the 3s to the 7s', values: { from: '3', to: '7', n: 7 } },
      { label: 'Through the centre', values: { from: '1', to: 'centre', n: 5 } },
      { label: 'Off the map', values: { from: '7', to: 'far', n: 9 } },
    ],
    counters: [{ id: 'pt', label: 'points decoded' }],
    run(inp) {
      const v = P.vae(), F = H().frames(['pt']), q = F.q, tiles = P.tiles(), a = placeOf(inp.from), b = placeOf(inp.to), n = inp.n;
      const ax = P.latentAxes('vm', 50, 56, 320, R), bx = P.latentAxes('vt', 450, 56, 320, R, { grid: false });
      const tw = 320 / TILES, base = [...ax.items, ...bx.items, ...P.rings(ax, 'vm:p'), ...mapDots(ax, 'vm:d', v.map), ...legend('lg', 50, 436)];
      tiles.images.forEach((im, k) => base.push(P.img('vt:i' + k, 450 + (k % TILES) * tw + 1, 56 + Math.floor(k / TILES) * tw + 1, tw - 2, im, 'tile')));
      base.push(H().text('vm:h', 50, 18, 'The test digits’ codes μ', 'strong', 14), H().text('vt:h', 450, 18, 'What the decoder draws there', 'strong', 14));
      const walk = [...Array(n).keys()].map((j) => [a[0] + ((b[0] - a[0]) * j) / (n - 1), a[1] + ((b[1] - a[1]) * j) / (n - 1)]), imgs = v.decode(walk);
      const path = (k, upto) => { const pts = walk.slice(0, upto + 1).map((z) => [k.X(z[0]), k.Y(z[1])]); return pts.length > 1 ? { key: k.o.prefix + ':walk', kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ',' + p[1]).join(' '), cls: 'p14walk', pathAudit: 'edge-x', layer: 'heads', box: H().boxOf(pts) } : null; };
      const sw = Math.min(56, (720 - (n - 1) * 8) / n);
      const counts = {}; for (const m of v.map) counts[m.y] = (counts[m.y] || 0) + 1;
      F.push(base.concat(), `The encoder places each of the ${q.n(v.map.length)} test digits at its mean code μ: ${H().list(P.VAE_CLASSES.map((d) => q.n(counts[d]) + ' ' + q.n(d) + 's'))}, each digit in its own region, most within the prior’s 2σ circle. On the right, what the decoder draws at the centres of ${q.n(TILES * TILES)} squares of the same plane: every point decodes to something, and neighbours look alike.`, { phase: 'map', digits: P.VAE_CLASSES, n: v.map.length, counts: P.VAE_CLASSES.map((d) => counts[d]), tiles: TILES * TILES });
      const looks = imgs.map(P.looksLike), near = walk.map(P.nearestCentre);
      walk.forEach((z, j) => {
        F.set('pt', j + 1);
        const items = base.concat([path(ax, j), path(bx, j)].filter(Boolean));
        items.push({ key: 'vm:z', kind: 'dot', x: ax.X(z[0]), y: ax.Y(z[1]), r: 7, cls: 'p14z', layer: 'heads', audit: 'overlap-ok' }, { key: 'vt:z', kind: 'dot', x: bx.X(z[0]), y: bx.Y(z[1]), r: 7, cls: 'p14z', layer: 'heads', audit: 'overlap-ok' });
        items.push(...P.row('w', 50, 462, sw, imgs.slice(0, j + 1), 8), P.ring('w:ring', 50 + j * (sw + 8), 462, sw, 'act'));
        const r = Math.hypot(z[0], z[1]), lk = looks[j], nc = near[j];
        F.push(items, `Point ${q.n(j + 1)} of ${q.n(n)}: z = (${q(z[0], 2)}, ${q(z[1], 2)}), ${q(r, 2)} from the centre. ` + (nc.tie ? `It is halfway between the regions of the ${q.n(Math.min(nc.digit, nc.second))}s and the ${q.n(Math.max(nc.digit, nc.second))}s (${q(nc.dist, 2)} from each centre), and` : `The nearest region is that of the ${q.n(nc.digit)}s (${q(nc.dist, 2)} from their centre), and`) + ` the decoded image, with ink ${q(P.ink(imgs[j]), 1)}, looks most like the average ${q.n(lk.digit)} (correlation ${q(lk.corr, 2)}).`, { phase: 'walk', j: j + 1, n, z, r, near: nc.digit, near2: nc.second, tie: nc.tie, nearDist: nc.dist, looks: lk.digit, corr: lk.corr, ink: P.ink(imgs[j]) });
      });
      const changes = looks.map((l, j) => (j && l.digit !== looks[j - 1].digit ? j : -1)).filter((j) => j > 0), far = walk.map((z) => Math.hypot(z[0], z[1])), fi = far.indexOf(Math.max(...far));
      const last = F.last();
      // beyond the 2σ circle: compare the farthest image with the last one drawn inside it, and say what the numbers show
      const ins = far.map((r, j) => (r <= 2.2 ? j : -1)).filter((j) => j >= 0), j0 = ins.length ? ins[ins.length - 1] : 0, i0 = P.ink(imgs[j0]), i1 = P.ink(imgs[fi]);
      const faded = i1 < 0.8 * i0 || looks[fi].corr < 0.7;
      const offMap = () => {
        return faded
          ? `out there, beyond the 2σ circle where few codes were trained, the images grow faint and blurred: ink ${q(i0, 1)} at point ${q.n(j0 + 1)}, ${q(i1, 1)} at point ${q.n(fi + 1)}, and a correlation of only ${q(looks[fi].corr, 2)} with the average ${q.n(looks[fi].digit)}.`
          : `beyond the 2σ circle no codes were trained, yet the decoder keeps drawing a sharp ${q.n(looks[fi].digit)}: the ink stays near ${q(i1, 1)} (${q(i0, 1)} at point ${q.n(j0 + 1)}) and the correlation with the average ${q.n(looks[fi].digit)} only slips to ${q(looks[fi].corr, 2)}. It simply extends the nearest region; nothing warns that the code is off the map.`;
      };
      F.push(last.items, `Along the walk the decoded image changes from looking like ${q.n(looks[0].digit)} to ${q.n(looks[n - 1].digit)}` + (changes.length ? ` (first between points ${q.n(changes[0])} and ${q.n(changes[0] + 1)})` : '') + `. The walk goes out to ${q(far[fi], 2)} from the centre; ` + (far[fi] > 2.2 ? offMap() : 'it stays within the region the codes fill, so every image is a plausible digit.'), { phase: 'end', from: looks[0].digit, to: looks[n - 1].digit, change: changes, changeAt: changes.length ? [changes[0], changes[0] + 1] : [], farthest: far[fi], inside: { point: j0 + 1, ink: i0 }, outside: { point: fi + 1, ink: i1, corr: looks[fi].corr, digit: looks[fi].digit }, faded });
      for (const f of F.list) f.narrowView = { x: 20, y: 0, w: 400, h: 540 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The VAE’s map', text: ['encoder: x → μ(x), log σ²(x)   (2 numbers each)', 'prior: z ~ N(0, I)', 'walk: z = (1 − t) a + t b, t = 0 … 1', 'decoder: z → an image'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'map', pick: 'option', ask: 'What will the image at the far end of the walk look like most?', options: ['0', '1', '3', '7'], answer: (fr) => String(fr[fr.length - 1].meta.to), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Why do neighbouring points of a VAE’s latent map decode to similar images?', options: ['each image is encoded as a cloud of codes, so the decoder must draw the same thing across a neighbourhood', 'the decoder is linear', 'the codes are sorted', 'the map is drawn by hand'], answer: 0, why: 'The noise in z teaches the decoder that small moves in the code should not change the picture much.' },
      { q: 'What does the prior N(0, I) do to the map?', options: ['pulls the codes into one region round the origin, so random draws land where the decoder was trained', 'labels the digits', 'spreads the codes as far apart as possible', 'nothing during training'], answer: 0, why: 'The KL term pulls each code cloud towards the standard normal.' },
      { q: 'How would you make a brand-new digit with a trained VAE?', options: ['draw z from N(0, I) and decode it', 'encode a digit and copy it', 'average two images pixel by pixel', 'run the encoder backwards'], answer: 0, why: 'The decoder alone is the generator; the prior says where to sample.' },
      { q: 'A walk goes far outside the 2σ circle. What can you say about the images there?', options: ['nothing in training checked them: few codes fell there', 'they are always sharper digits', 'they are always blank', 'the decoder refuses to draw them'], answer: 0, why: 'The decoder only learned the region where codes fell during training. Out there it may draw faint or odd images, or, as this one does, simply go on extending the nearest digit.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'gen-reparam', family: 'p14-ae', order: 30, title: 'The reparameterisation trick, step by step',
    blurb: 'A VAE’s code is random: the encoder gives a mean μ and a spread σ, and z is drawn from the normal they describe. A random draw has no gradient, so training could not reach the encoder. The trick is to draw the randomness separately, as ε from a standard normal, and to build z = μ + σ ⊙ ε from it: now z is a plain sum and product of μ and σ, and the gradients flow.',
    inputs: [
      { id: 'img', kind: 'number', label: 'Test digit (1–128)', def: 2, min: 1, max: 128 },
      { id: 'seed', kind: 'number', label: 'Seed for ε', def: 5, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A 3', values: { img: 2, seed: 5 } },
      { label: 'The same 3, another ε', values: { img: 2, seed: 11 } },
      { label: 'A 1', values: { img: 11, seed: 5 } },
      { label: 'A 7', values: { img: 3, seed: 8 } },
    ],
    counters: [{ id: 'draws', label: 'draws of ε' }],
    run(inp) {
      const v = P.vae(), i = inp.img - 1, tr = v.explain(i, inp.seed), st = Object.fromEntries(tr.map((s) => [s.kind, s]));
      const F = H().frames(['draws']), q = F.q, digit = v.test.y[i], img = v.test.X.subarray(i * 256, (i + 1) * 256);
      const mu = st.encode.mu, lv = st.encode.logvar, sd = st.sigma.sigma, eps = st.noise.eps, z = st.reparam.z, kl = st.kl.kl, klSum = kl[0] + kl[1];
      const rec = st.decode.recon, elbo = st.elbo.elbo, out = st.decode.image;
      // the graph across the top
      const node = (key, x, y, w, text, cls) => ({ key, kind: 'state', x, y, w, h: 36, rx: 8, text, cls: 'block small ' + (cls || '') });
      const edge = (key, a, b, cls, label, lx, ly) => Object.assign({ key, kind: 'edge', from: a, to: b, head: true, cls: 'p14e ' + (cls || '') }, label ? { label, lx, ly, size: 13 } : {});
      const lit = (on) => (k) => (on.includes(k) ? 'act' : '');
      const graph = (on, grad) => {
        const L = lit(on);
        return [
          P.img('g:x', 16, 96, 64, img, on.includes('x') ? 'act' : ''), node('g:enc', 148, 128, 92, 'encoder', L('enc')), node('g:mu', 260, 78, 60, 'μ', L('mu')), node('g:lv', 260, 178, 70, 'log σ²', L('lv')),
          node('g:sd', 360, 178, 50, 'σ', L('sd')), node('g:eps', 360, 262, 104, 'ε ~ N(0, I)', 'rand ' + L('eps')), { key: 'g:mul', kind: 'state', x: 450, y: 178, r: 17, text: '×', cls: 'block p14op ' + L('mul') },
          { key: 'g:add', kind: 'state', x: 530, y: 128, r: 17, text: '+', cls: 'block p14op ' + L('add') }, node('g:z', 610, 128, 50, 'z', L('z')), node('g:dec', 712, 128, 92, 'decoder', L('dec')), P.img('g:out', 780, 96, 64, on.includes('out') ? out : new Float32Array(256).fill(0), on.includes('out') ? 'act' : ''),
          { key: 'g:e0', kind: 'edge', from: { x: 80, y: 128 }, to: 'g:enc', head: true, cls: 'p14e' }, edge('g:e1', 'g:enc', 'g:mu', L('mu')), edge('g:e2', 'g:enc', 'g:lv', L('lv')), edge('g:e3', 'g:lv', 'g:sd', L('sd')),
          edge('g:e4', 'g:sd', 'g:mul', grad ? 'grad' : L('mul'), grad ? '∂z/∂σ = ε' : null, 408, 146), edge('g:e5', 'g:eps', 'g:mul', L('mul')), edge('g:e6', 'g:mul', 'g:add', L('add')),
          edge('g:e7', 'g:mu', 'g:add', grad ? 'grad' : L('add'), grad ? '∂z/∂μ = 1' : null, 420, 84), edge('g:e8', 'g:add', 'g:z', L('z')), edge('g:e9', 'g:z', 'g:dec', L('dec')), { key: 'g:e10', kind: 'edge', from: 'g:dec', to: { x: 778, y: 128 }, head: true, cls: 'p14e ' + L('out') },
        ];
      };
      // the latent plane: the whole map first, then zoomed in on μ so the encoder's spread can be seen
      const axF = P.latentAxes('rl', 60, 300, 240, R), hw = Math.max(0.35, 4 * Math.max(sd[0], sd[1]));
      const tk = S.Kit.curve.ticks(0, 2 * hw, 3), axZ = S.Kit.curve.axes({ prefix: 'rl', x: 60, y: 300, w: 240, h: 240, xr: [mu[0] - hw, mu[0] + hw], yr: [mu[1] - hw, mu[1] + hw], xticks: tk[1] - tk[0], yticks: tk[1] - tk[0], xname: 'z₁', yname: 'z₂' });
      const plane = (ax, zoom) => [...ax.items, ...(zoom ? [] : P.rings(ax, 'rl:p')), ...mapDots(ax, 'rl:d', v.map.filter((m) => ax.inside(m.z[0], m.z[1])), !zoom), H().text('rl:h', 100, 290, zoom ? 'Zoomed in on μ' : 'The latent map', 'strong', 14)];
      const sg = (x) => q.sg(x, 3), neg = (x) => (x < 0 ? '(' + sg(x) + ')' : sg(x));
      const table = (rows, hot) => S.Kit.dt.table({ prefix: 'tb', x: 350, y: 300, cols: [{ name: '', w: 76 }, { name: 'z₁', w: 76 }, { name: 'z₂', w: 76 }], rows, rowH: 26, size: 13, cls: (r) => (r === hot ? 'act' : '') }).items;
      const T = (k) => {
        const rows = [['μ'], ['log σ²'], ['σ'], ['ε'], ['z'], ['KL']], fill = [mu, lv, sd, eps, z, kl];
        return rows.map((r, j) => (j <= k ? [r[0], H().sg(fill[j][0], 3), H().sg(fill[j][1], 3)] : [r[0], '', '']));
      };
      const ell = (ax, k) => H().ellipse(ax, 'rl:q' + k, mu, [[sd[0] ** 2, 0], [0, sd[1] ** 2]], k, 'p14q');
      const dot = (ax, key, p, cls, r) => ({ key, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: r || 6, cls, layer: 'heads', audit: 'overlap-ok' });
      const muDot = (ax) => dot(ax, 'rl:mu', mu, 'p14mu'), zDot = (ax) => dot(ax, 'rl:z', z, 'p14z');
      const arrow = (ax) => H().arrow('rl:ar', [ax.X(mu[0]), ax.Y(mu[1])], [ax.X(z[0]), ax.Y(z[1])], 'p14arr');
      const zoomed = (extra) => [...plane(axZ, true), ell(axZ, 1), ell(axZ, 2), muDot(axZ)].concat(extra || []);
      const meta = { mu, lv, sd, eps, z, kl, klSum, rec, elbo, img: inp.img, digit };
      const imgsAt = (list) => list.map((p, k) => P.img('im' + k, p[0], p[1], 56, p[2], p[3] || ''));
      F.push([...graph(['x', 'enc']), ...plane(axF), ...table(T(-1), -1)], `Test digit ${q.n(inp.img)} is ${P.aQ(q, digit)}. The encoder does not give one code for it: it gives the mean μ and the log-variance log σ² of a normal distribution over codes, ${q.n(2)} numbers each.`, Object.assign({ phase: 'input', k: 2 }, meta));
      F.push([...graph(['enc', 'mu', 'lv']), ...plane(axF), muDot(axF), ...table(T(1), 0)], `The encoder’s outputs: μ = (${sg(mu[0])}, ${sg(mu[1])}) and log σ² = (${sg(lv[0])}, ${sg(lv[1])}). μ sits among the other ${q.n(digit)}s on the map.`, Object.assign({ phase: 'encode' }, meta));
      F.push([...graph(['lv', 'sd']), ...zoomed(), ...table(T(2), 2)], `σ = exp(½ log σ²) = (${sg(sd[0])}, ${sg(sd[1])}). Zoomed in on μ, the ellipses are one and two σ around it: the encoder says the code is somewhere in there.`, Object.assign({ phase: 'sigma' }, meta));
      F.set('draws', 1);
      F.push([...graph(['eps']), ...zoomed(), ...table(T(3), 3)], `Draw ε from a standard normal, independently of the network: ε = (${sg(eps[0])}, ${sg(eps[1])}). All the randomness lives here, outside the path from the encoder to z.`, Object.assign({ phase: 'noise' }, meta));
      F.push([...graph(['sd', 'eps', 'mul', 'mu', 'add', 'z']), ...zoomed([arrow(axZ), zDot(axZ)]), ...table(T(4), 4)], `z = μ + σ ⊙ ε, one coordinate at a time: z₁ = ${sg(mu[0])} + ${sg(sd[0])} × ${neg(eps[0])} = ${sg(z[0])} and z₂ = ${sg(mu[1])} + ${sg(sd[1])} × ${neg(eps[1])} = ${sg(z[1])}.`, Object.assign({ phase: 'reparam' }, meta));
      F.push([...graph(['mu', 'sd', 'z'], true), ...zoomed([arrow(axZ), zDot(axZ)]), ...table(T(4), 4)], `Now the gradient can pass: ∂z/∂μ = 1 and ∂z/∂σ = ε = (${sg(eps[0])}, ${sg(eps[1])}). Had z been drawn directly from N(μ, σ²), there would be nothing to differentiate.`, Object.assign({ phase: 'grad' }, meta));
      // more draws of ε for the same digit
      const g = S.T.rng(inp.seed * 7 + 1), more = [...Array(6).keys()].map(() => [g.normal(), g.normal()]), mz = more.map((e) => [mu[0] + sd[0] * e[0], mu[1] + sd[1] * e[1]]), mimg = v.decode(mz);
      F.set('draws', 7);
      const md = mz.map((p, k) => dot(axZ, 'rl:m' + k, p, 'p14z soft', 4.5));
      const spread = mimg.map(P.ink), lks = mimg.map((im) => P.looksLike(im).digit), same = lks.filter((d) => d === digit).length;
      F.push([...graph(['eps', 'z', 'dec']), ...zoomed([zDot(axZ), ...md]), ...table(T(4), 4), ...imgsAt(mimg.map((im, k) => [600 + (k % 3) * 64, 316 + Math.floor(k / 3) * 64, im])), H().text('im:h', 600, 302, 'Six more draws', 'strong', 14)], `Six more draws of ε give six more codes inside the ellipses, and six decoded images with ink from ${q(Math.min(...spread), 1)} to ${q(Math.max(...spread), 1)}; ${q.n(same)} of the ${q.n(6)} look most like the average ${q.n(digit)}. The noise teaches the decoder that the whole cloud stands for this digit.`, Object.assign({ phase: 'many', more: mz, inks: spread, same, six: 6 }, meta));
      const pair = () => [...imgsAt([[600, 316, img], [680, 316, out, 'act']]), H().text('im:a', 628, 392, 'input', 'axname', 13, 'middle'), H().text('im:b', 708, 392, 'rebuilt', 'axname', 13, 'middle')];
      F.push([...graph(['z', 'dec', 'out']), ...zoomed([zDot(axZ)]), ...table(T(4), 4), ...pair()], `The decoder turns z into an image. Compared pixel by pixel with the input, the reconstruction loss (binary cross-entropy summed over the pixels) is ${q.sg(rec, 3)} nats.`, Object.assign({ phase: 'decode' }, meta));
      F.push([...graph([]), ...zoomed(), ...table(T(5), 5), ...pair()], `The KL term measures how far the encoder’s normal is from the prior N(0, I): ½ (μ² + σ² − 1 − log σ²) per coordinate is ${q.sg(kl[0], 3)} and ${q.sg(kl[1], 3)}, ${q.sg(klSum, 3)} nats in all. It pulls μ towards the centre and σ towards ${q.n(1)}.`, Object.assign({ phase: 'kl', one: 1 }, meta));
      F.push([...graph(['out']), ...zoomed([zDot(axZ)]), ...table(T(5), -1), ...pair()], `The training objective, the ELBO, is −(reconstruction + KL) = −(${q.sg(rec, 3)} + ${q.sg(klSum, 3)}) = ${q.sg(elbo, 3)}. Training raises it, through z = μ + σ ⊙ ε, into both the decoder and the encoder.`, Object.assign({ phase: 'elbo' }, meta));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Reparameterisation', text: ['σ = exp(½ log σ²)', 'ε ~ N(0, I)', 'z = μ + σ ⊙ ε', '∂z/∂μ = 1,  ∂z/∂σ = ε', 'KL = ½ Σ (μ² + σ² − 1 − log σ²)', 'ELBO = −(reconstruction + KL)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'noise', pick: 'option', ask: 'Will z land closer to the centre (0, 0) than μ does?', options: ['closer', 'further'], answer: (fr) => { const m = fr[0].meta; return Math.hypot(...m.z) < Math.hypot(...m.mu) ? 'closer' : 'further'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'reparam').say },
    ],
    quiz: [
      { q: 'μ = 0.5, σ = 0.2 and ε = −1.5. What is z?', options: ['0.2', '0.8', '−0.3', '0.5'], answer: 0, why: '0.5 + 0.2 × (−1.5) = 0.2.' },
      { q: 'Why is z written as μ + σ ⊙ ε instead of drawn straight from N(μ, σ²)?', options: ['so z is a differentiable function of μ and σ, and the encoder gets gradients', 'it is faster to draw', 'it makes z non-random', 'it removes the KL term'], answer: 0, why: 'The draw moves to ε, which needs no gradient.' },
      { q: 'log σ² = 0 and μ = 0 in one coordinate. What is its KL to N(0, 1)?', options: ['0', '0.5', '1', '−0.5'], answer: 0, why: '½ (0 + 1 − 1 − 0) = 0: the encoder’s normal is the prior itself.' },
      { q: 'What does ∂z/∂σ = ε tell you?', options: ['a draw far from the mean pushes σ’s gradient harder', 'σ never changes', 'the gradient of σ is always 1', 'ε is learned'], answer: 0, why: 'σ scales ε, so its gradient is proportional to ε.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
