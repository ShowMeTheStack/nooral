// Part 14, generative models: the exercises. Hand computations are worked out here from the formulas the engines use
// (S.Gen: the reparameterisation and KL of gen-b-ae.js, the schedule and forward jump of gen-d-diffusion.js, the
// coverage count of gen-a-base.js); figures quoted from the topics (the β runs of gen-kl, the code sizes of gen-ae)
// were printed by the topics themselves.
(function (S) {
  'use strict';
  const C = S.Content, G = S.Gen;
  if (!C || !G || C.exercises.has('gm-x-ratio')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => mn(r2(v).toFixed(2)), n3 = (v) => mn(r3(v).toFixed(3));
  const mn = (v) => String(v).replace(/^-/, '−');
  const k2 = (v) => String(r2(v)), k3 = (v) => String(r3(v));
  const ln = Math.log, sq = Math.sqrt, ex = Math.exp;
  const klDim = (m, lv) => 0.5 * (m * m + ex(lv) - 1 - lv);

  // =========================================================================================================
  // Chapter 1: autoencoders and VAEs
  // =========================================================================================================
  C.exercise({ id: 'gm-x-ratio', chapter: 'gm-ae', title: 'How hard is the squeeze?', level: 1, kind: 'number',
    text: ['An autoencoder takes 16 × 16 greyscale images and squeezes each through a code of 4 numbers. How many input numbers does each code number stand for?'],
    answer: 64, tolerance: 0,
    whyNot: { 4: 'That is the size of the code; divide the number of pixels by it.', 256: 'That is the number of pixels; divide it by the size of the code.', 16: 'An image has 16 × 16 pixels, not 16.', '*': 'Divide the pixels by the code size.' },
    hints: ['Count the pixels of one image.', '16 × 16 = 256 pixels, squeezed into 4 numbers.', '256 / 4 = 64: each code number stands for 64 pixels.'],
    solution: '64', wrong: ['4', '256', '16'] });

  const wopts = ['The code of 8 numbers', 'The code of 4 numbers', 'The code of 2 numbers', 'All three end the same: the decoder makes up the difference'];
  C.exercise({ id: 'gm-x-ae-width', chapter: 'gm-ae', title: 'Which bottleneck rebuilds best?', level: 1, kind: 'choice',
    text: ['In [The autoencoder’s bottleneck](#topic=gen-ae) the same network is trained for 600 steps with codes of 2, 4 and 8 numbers. Which ends with the lowest reconstruction loss on the 128 test digits?'],
    options: wopts, answer: 0,
    whyNot: { 1: '4 numbers end at 42.2 nats, behind 8 numbers.', 2: '2 numbers keep the least: 50.2 nats, the highest loss of the three.', 3: 'The decoder sees only the code. What the code leaves out, it cannot rebuild.' },
    hints: ['The code is everything the decoder gets.', 'More numbers in the code let more detail through.', 'The topic’s last frame: 50.2 nats with 2 numbers, 42.2 with 4 and 34.7 with 8. The widest code wins.'],
    solution: 0, wrong: [2, 3] });

  const sopts = ['Nothing says where its codes lie, so a made-up code may land where the decoder learned nothing', 'Its decoder cannot take a code as input', 'Its codes are random, so it never rebuilds the same image twice', 'It has no encoder'];
  C.exercise({ id: 'gm-x-ae-sample', chapter: 'gm-ae', title: 'Why not sample from an autoencoder?', level: 2, kind: 'choice',
    text: ['A plain autoencoder rebuilds digits well. Why is drawing a random code and decoding it a poor way to make new digits?'],
    options: sopts, answer: 0,
    whyNot: { 1: 'The decoder takes exactly a code: that is how it rebuilds.', 2: 'A plain autoencoder’s code is a fixed function of the image; there is no randomness in it.', 3: 'It has one: the encoder makes the codes.' },
    hints: ['Where do the codes of the training images lie?', 'Nothing in the training pulls them into a known region, so their layout can be anything.', 'A random code can fall in a gap the decoder never saw, and decode to nonsense. A VAE fixes this by pulling every code towards N(0, I).'],
    solution: 0, wrong: [1, 2] });

  const wa = [-1, 2], wb = [2, -1], wt = [0, 1 / 3, 2 / 3, 1], walk = wt.map((t) => [wa[0] + t * (wb[0] - wa[0]), wa[1] + t * (wb[1] - wa[1])]);
  const walkCells = {}; walk.forEach((z, j) => { walkCells['point ' + (j + 1) + '|z₁'] = r2(z[0]); walkCells['point ' + (j + 1) + '|z₂'] = r2(z[1]); });
  C.exercise({ id: 'gm-x-walk', chapter: 'gm-ae', title: 'Points on a walk through the latent map', level: 1, kind: 'table',
    text: ['A VAE with a code of 2 numbers. A walk goes in a straight line from a = (−1, 2) to b = (2, −1) in 4 evenly spaced points, z = (1 − t) a + t b with t = 0, ⅓, ⅔, 1. Write the code of each point; the decoder would draw an image at each.'],
    rows: ['point 1', 'point 2', 'point 3', 'point 4'], cols: ['z₁', 'z₂'], corner: '', cells: walkCells, tolerance: 0.01,
    hints: ['The first point is a and the last is b.', 'Each step adds (b − a) / 3 = (1, −1).', 'Points: ' + walk.map((z) => '(' + z.map(n2).join(', ') + ')').join(', ') + '.'],
    solution: Object.fromEntries(Object.entries(walkCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(walkCells).map(([k, v]) => [k, String(v)])), { 'point 2|z₁': '0.5', 'point 2|z₂': '1.5' })] });

  const RM = [0.4, -1.1], RLV = [-0.5, -2.0], RE = [1.3, -0.6], RSD = RLV.map((v) => ex(0.5 * v)), RZ = RM.map((m, k) => m + RSD[k] * RE[k]);
  const rpCells = { 'coordinate 1|σ': r3(RSD[0]), 'coordinate 1|z': r3(RZ[0]), 'coordinate 2|σ': r3(RSD[1]), 'coordinate 2|z': r3(RZ[1]) };
  const rpWrongSd = RLV.map((v) => ex(v)), rpWrong = { 'coordinate 1|σ': n3(rpWrongSd[0]), 'coordinate 1|z': n3(RM[0] + rpWrongSd[0] * RE[0]), 'coordinate 2|σ': n3(rpWrongSd[1]), 'coordinate 2|z': n3(RM[1] + rpWrongSd[1] * RE[1]) };
  C.exercise({ id: 'gm-x-reparam', chapter: 'gm-ae', title: 'The reparameterisation by hand', level: 1, kind: 'table',
    text: ['An encoder gives μ = (0.4, −1.1) and log σ² = (−0.5, −2.0). The draw from a standard normal is ε = (1.3, −0.6). Find σ = exp(½ log σ²) and the code z = μ + σ ε for each coordinate. Three decimal places.'],
    rows: ['coordinate 1', 'coordinate 2'], cols: ['σ', 'z'], corner: '', cells: rpCells, tolerance: 0.002,
    hints: ['σ is the square root of the variance: exp(½ log σ²), not exp(log σ²).', 'σ₁ = exp(−0.25) = ' + n3(RSD[0]) + ' and σ₂ = exp(−1) = ' + n3(RSD[1]) + '.', 'z₁ = 0.4 + ' + n3(RSD[0]) + ' × 1.3 = ' + n3(RZ[0]) + '; z₂ = −1.1 + ' + n3(RSD[1]) + ' × (−0.6) = ' + n3(RZ[1]) + '.'],
    solution: Object.fromEntries(Object.entries(rpCells).map(([k, v]) => [k, String(v)])),
    wrong: [rpWrong] });

  const gopts = ['∂z/∂μ = 1 and ∂z/∂σ = ε, so the loss’s gradient reaches μ and σ', '∂z/∂μ = 0: the code is random, so nothing flows back', '∂z/∂σ = 1 and ∂z/∂μ = ε', 'There is no gradient; the encoder is trained by trying many draws'];
  C.exercise({ id: 'gm-x-reparam-grad', chapter: 'gm-ae', title: 'Where does the gradient go?', level: 2, kind: 'choice',
    text: ['With z = μ + σ ε and ε drawn apart from the network, what are the derivatives of z with respect to μ and σ, and what do they make possible?'],
    options: gopts, answer: 0,
    whyNot: { 1: 'ε is random, but once drawn it is a fixed number: z is then an ordinary sum of μ and σε.', 2: 'μ enters z with weight 1; σ is the one multiplied by ε.', 3: 'Backpropagation does it in one pass, precisely because z is written with ε outside.' },
    hints: ['Treat ε as a constant for this one step.', 'z = μ + σ ε is linear in μ and in σ.', '∂z/∂μ = 1, ∂z/∂σ = ε: the gradient of the loss with respect to z passes on to μ and σ, and from them into the encoder.'],
    solution: 0, wrong: [1, 2] });

  const reSol = 'function reparam(mu, logvar, eps) {\n  return mu.map((m, i) => m + Math.exp(0.5 * logvar[i]) * eps[i]);\n}';
  C.exercise({ id: 'gm-x-code-reparam', chapter: 'gm-ae', title: 'Write the reparameterisation', level: 2, kind: 'code',
    text: ['Write `reparam(mu, logvar, eps)`: three arrays of the same length, the encoder’s means, its log-variances log σ², and a draw from a standard normal. Return the code z, an array with z[i] = μ[i] + σ[i] · ε[i] where σ[i] = exp(½ log σ²[i]).'],
    fn: 'reparam', params: ['mu', 'logvar', 'eps'], start: 'function reparam(mu, logvar, eps) {\n  \n}', tolerance: 1e-9,
    ref: (mu, lv, e) => mu.map((m, i) => m + ex(0.5 * lv[i]) * e[i]),
    cases: [[RM, RLV, RE], [[0], [0], [1]], [[-1.54, 0.467], [-3.17, -4.03], [0.123, -0.853]], [[2, -3, 0.5], [1, -1, 0], [0, 0.5, -2]], [[0.1, 0.2, 0.3, 0.4], [-8, 6, 0, 2], [1, 1, -1, -1]]],
    hints: ['Work coordinate by coordinate with map.', 'σ = Math.exp(0.5 * logvar[i]): half the log-variance, then the exponential.', reSol],
    solution: reSol,
    wrong: ['function reparam(mu, logvar, eps) {\n  return mu.map((m, i) => m + Math.exp(logvar[i]) * eps[i]);\n}', 'function reparam(mu, logvar, eps) {\n  return mu.map((m, i) => m + logvar[i] * eps[i]);\n}', 'function reparam(mu, logvar, eps) {\n  return mu.map((m, i) => m * Math.exp(0.5 * logvar[i]) + eps[i]);\n}'] });

  const K1m = 0.8, K1s = 0.5, K1 = 0.5 * (K1m * K1m + K1s * K1s - 1 - ln(K1s * K1s));
  C.exercise({ id: 'gm-x-kl-1d', chapter: 'gm-ae', title: 'The KL term for one coordinate', level: 2, kind: 'number',
    text: ['One coordinate of a VAE’s code has μ = 0.8 and σ = 0.5. What is its KL term against the prior N(0, 1), ½ (μ² + σ² − 1 − ln σ²)? Three decimal places (natural logarithms).'],
    answer: r3(K1), tolerance: 0.002,
    whyNot: { [k3(0.5 * (K1m * K1m + K1s - 1 - ln(K1s)))]: 'That uses σ where the formula has σ²: square σ first, σ² = 0.25.', [k3(K1m * K1m + K1s * K1s - 1 - ln(K1s * K1s))]: 'Halve it: the formula starts with ½.', [k3(0.5 * (K1m * K1m + K1s * K1s - 1))]: 'The −ln σ² term is missing; ln 0.25 is negative, so it adds 1.386.', '*': 'σ² = 0.25 and ln 0.25 = −1.386.' },
    hints: ['μ² = 0.64 and σ² = 0.25.', 'ln σ² = ln 0.25 = −1.386, so −ln σ² = +1.386.', '½ (0.64 + 0.25 − 1 + 1.386) = ½ × ' + n3(2 * K1) + ' = ' + n3(K1) + ' nats.'],
    solution: n3(K1), wrong: [k3(0.5 * (K1m * K1m + K1s - 1 - ln(K1s))), n3(2 * K1), n3(0.5 * (K1m * K1m + K1s * K1s - 1))] });

  const K2m = [1.2, -0.5], K2lv = [-1, 0.4], K2d = K2m.map((m, k) => klDim(m, K2lv[k])), K2 = K2d[0] + K2d[1];
  C.exercise({ id: 'gm-x-kl-2d', chapter: 'gm-ae', title: 'KL from the encoder’s outputs', level: 3, kind: 'number',
    text: ['An encoder outputs μ = (1.2, −0.5) and log σ² = (−1, 0.4) for an image. What is the KL term, summed over both coordinates, ½ Σ (μ² + σ² − 1 − log σ²)? Three decimal places.'],
    answer: r3(K2), tolerance: 0.002,
    whyNot: { [k3(K2d[0])]: 'That is only the first coordinate; add the second.', [k3(0.5 * K2m.reduce((s, m, k) => s + m * m + K2lv[k] - 1 - K2lv[k], 0))]: 'σ² is exp(log σ²), not log σ² itself.', '*': 'σ² = exp(log σ²): 0.368 and 1.492.' },
    hints: ['σ² = exp(log σ²): exp(−1) = 0.368 and exp(0.4) = 1.492.', 'First coordinate: ½ (1.44 + 0.368 − 1 + 1) = ' + n3(K2d[0]) + '. Second: ½ (0.25 + 1.492 − 1 − 0.4) = ' + n3(K2d[1]) + '.', 'Sum: ' + n3(K2) + ' nats.'],
    solution: n3(K2), wrong: [k3(K2d[0]), n3(0.5 * K2m.reduce((s, m, k) => s + m * m + K2lv[k] - 1 - K2lv[k], 0))] });

  const zopts = ['μ = 0 and σ = 1 in every coordinate', 'μ = 0 and σ = 0 in every coordinate', 'μ = 1 and σ = 1', 'whenever the reconstruction is perfect'];
  C.exercise({ id: 'gm-x-kl-zero', chapter: 'gm-ae', title: 'When is the KL zero?', level: 1, kind: 'choice',
    text: ['The KL term of a VAE is ½ Σ (μ² + σ² − 1 − ln σ²). When is it exactly zero?'],
    options: zopts, answer: 0,
    whyNot: { 1: 'σ = 0 makes ln σ² = −∞: the KL is infinite, not zero. A code with no spread is as far from the prior as can be.', 2: 'μ = 1 adds μ² = 1 to the sum: KL ½ for that coordinate.', 3: 'The KL does not look at the reconstruction; it compares the code’s normal with N(0, 1).' },
    hints: ['Try each coordinate on its own.', 'μ² is zero only at μ = 0; σ² − 1 − ln σ² is zero only at σ = 1.', 'μ = 0 and σ = 1: the code’s normal is exactly the prior. An encoder that always outputs this carries no information about the image, which is why a KL near zero can be a warning.'],
    solution: 0, wrong: [1, 2] });

  C.exercise({ id: 'gm-x-elbo', chapter: 'gm-ae', title: 'The ELBO from its parts', level: 1, kind: 'number',
    text: ['On a batch of digits, a VAE’s reconstruction loss is 52.3 nats per image and its KL term 6.1 nats. What is the ELBO per image?'],
    answer: -58.4, tolerance: 0.01,
    whyNot: { 58.4: 'The ELBO is the negative of the loss: −(reconstruction + KL).', '-46.2': 'Both terms are added inside the bracket, not subtracted.', '46.2': 'Add the two terms, then take the negative.', '*': 'ELBO = −(reconstruction + KL).' },
    hints: ['The VAE’s loss is reconstruction + KL.', 'The ELBO is the negative of that loss; training raises it.', '−(52.3 + 6.1) = −58.4.'],
    solution: '-58.4', wrong: ['58.4', '-46.2'] });

  const BETAS = [0, 1, 4, 16], BREC = [39.4, 40.1, 44.8, 55.1], BKL = [31.2, 3.39, 1.56, 0.00383];
  const bopts = ['The reconstruction loss rises and the KL falls', 'Both fall', 'The reconstruction loss falls and the KL rises', 'Both rise'];
  C.exercise({ id: 'gm-x-beta', chapter: 'gm-ae', title: 'Turning up β', level: 2, kind: 'choice',
    text: ['Four VAEs are trained alike except for the weight β in loss = reconstruction + β · KL. At the end, on the test digits:', { table: [['β', ...BETAS.map(String)], ['reconstruction (nats)', ...BREC.map(String)], ['KL (nats)', ...BKL.map((v) => String(v))]] }, 'What happens as β grows?'],
    options: bopts, answer: 0,
    whyNot: { 1: 'Read the reconstruction row: it goes up, from 39.4 to 55.1.', 2: 'That is the reverse: a larger β makes the KL cost more, so the encoder gives it up.', 3: 'The KL row goes down, from 31.2 to almost 0.' },
    hints: ['A larger β makes each nat of KL cost more.', 'The encoder answers by moving its codes towards N(0, I), giving up detail.', 'Reconstruction rises (39.4 → 55.1), KL falls (31.2 → 0.004). At β = 16 the codes crowd at the centre and the images blur.'],
    solution: 0, wrong: [1, 2] });

  const bPts = BETAS.map((b, i) => [b, BREC[i]]);
  C.exercise({ id: 'gm-x-beta-curve', chapter: 'gm-ae', title: 'Sketch reconstruction against β', level: 2, kind: 'predict-curve',
    text: ['The four VAEs of [The KL term against reconstruction](#topic=gen-kl), trained with β = 0, 1, 4 and 16. With β = 0, the test reconstruction ends at ' + BREC[0] + ' nats. Sketch the final reconstruction loss from β = 0 to β = 16.'],
    plot: { xr: [0, 16], yr: [30, 60], xname: 'β, the weight on the KL term', yname: 'reconstruction (nats)', given: [bPts[0]], note: 'final reconstruction loss on the test digits' },
    answer: bPts, tolerance: 2.5, checks: ['rising'],
    hints: ['A larger β pulls the codes towards the centre.', 'Codes crowded together leave the decoder less to tell images apart: reconstruction gets worse.', 'The topic’s values: ' + bPts.map(([b, v]) => 'β = ' + b + ': ' + v).join(', ') + '. Little change up to β = 1, then a steady climb.'],
    solution: bPts, wrong: [BETAS.map((b) => [b, BREC[0]]), BETAS.map((b, i) => [b, BREC[3] - (BREC[i] - BREC[0])])] });

  const klSol = 'function gaussKL(mu, logvar) {\n  let s = 0;\n  for (let i = 0; i < mu.length; i++) s += 0.5 * (mu[i] * mu[i] + Math.exp(logvar[i]) - 1 - logvar[i]);\n  return s;\n}';
  C.exercise({ id: 'gm-x-code-kl', chapter: 'gm-ae', title: 'Write the Gaussian KL', level: 2, kind: 'code',
    text: ['Write `gaussKL(mu, logvar)`: the encoder’s means and log-variances for one image, arrays of the same length. Return the KL from N(μ, σ²) to N(0, I), summed over the coordinates: Σ ½ (μ² + σ² − 1 − log σ²), with σ² = exp(log σ²).'],
    fn: 'gaussKL', params: ['mu', 'logvar'], start: 'function gaussKL(mu, logvar) {\n  \n}', tolerance: 1e-9,
    ref: (mu, lv) => mu.reduce((s, m, i) => s + klDim(m, lv[i]), 0),
    cases: [[K2m, K2lv], [[0], [0]], [[K1m], [ln(K1s * K1s)]], [[-1.54, 0.467], [-3.17, -4.03]], [[0, 0, 0], [1, -1, 2]], [[3, -2, 0.5, 0], [0, 0, 0, -6]]],
    hints: ['Loop over the coordinates and add one term each.', 'Each term is 0.5 * (μ² + Math.exp(logvar) − 1 − logvar). Mind the sign of the last part.', klSol],
    solution: klSol,
    wrong: ['function gaussKL(mu, logvar) {\n  let s = 0;\n  for (let i = 0; i < mu.length; i++) s += 0.5 * (mu[i] * mu[i] + logvar[i] - 1 - Math.exp(logvar[i]));\n  return s;\n}', 'function gaussKL(mu, logvar) {\n  let s = 0;\n  for (let i = 0; i < mu.length; i++) s += mu[i] * mu[i] + Math.exp(logvar[i]) - 1 - logvar[i];\n  return s;\n}', 'function gaussKL(mu, logvar) {\n  let s = 0;\n  for (let i = 0; i < mu.length; i++) s += 0.5 * (mu[i] * mu[i] + Math.exp(logvar[i]) - 1 + logvar[i]);\n  return s;\n}'] });

  // =========================================================================================================
  // Chapter 2: GANs
  // =========================================================================================================
  const DR = [0.9, 0.6], DF = [0.2, 0.4], dTerms = DR.map((r, i) => -ln(r) - ln(1 - DF[i])), DL = (dTerms[0] + dTerms[1]) / 2;
  const dWrongSum = dTerms[0] + dTerms[1], dWrongSign = DR.reduce((s, r, i) => s - ln(r) - ln(DF[i]), 0) / 2;
  C.exercise({ id: 'gm-x-d-loss', chapter: 'gm-gan', title: 'The discriminator’s loss on a batch', level: 2, kind: 'number',
    text: ['A batch holds two real images and two generated ones. The discriminator gives the real ones D = 0.9 and 0.6, and the generated ones D = 0.2 and 0.4. Its loss is the mean over the two pairs of −ln D(real) − ln(1 − D(fake)). What is it? Three decimal places.'],
    answer: r3(DL), tolerance: 0.002,
    whyNot: { [k3(dWrongSum)]: 'That is the sum over the two pairs; take the mean.', [k3(dWrongSign)]: 'For a generated image the discriminator wants D near 0, so its term is −ln(1 − D), not −ln D.', '*': 'Pair 1: −ln 0.9 − ln 0.8; pair 2: −ln 0.6 − ln 0.6; then the mean.' },
    hints: ['A real image costs −ln D(real); a generated one costs −ln(1 − D(fake)).', 'Pair 1: −ln 0.9 − ln(1 − 0.2) = 0.105 + 0.223 = ' + n3(dTerms[0]) + '. Pair 2: −ln 0.6 − ln(1 − 0.4) = ' + n3(dTerms[1]) + '.', 'Mean: (' + n3(dTerms[0]) + ' + ' + n3(dTerms[1]) + ') / 2 = ' + n3(DL) + '. A pure guess, D = 0.5 everywhere, would score 2 ln 2 = 1.386.'],
    solution: n3(DL), wrong: [k3(dWrongSum), n3(dWrongSign)] });

  C.exercise({ id: 'gm-x-g-loss', chapter: 'gm-gan', title: 'The generator’s loss for one fake', level: 1, kind: 'number',
    text: ['The discriminator gives a generated image D = 0.2. In the non-saturating form, the generator’s loss for it is −ln D(G(z)). What is it? Three decimal places.'],
    answer: r3(-ln(0.2)), tolerance: 0.002,
    whyNot: { [k3(ln(1 - 0.2))]: 'That is ln(1 − D), the original minimax form, which flattens out exactly when the generator is losing. The non-saturating loss is −ln D.', [k3(-ln(0.8))]: 'That is −ln(1 − D), the discriminator’s term for a fake. The generator wants D large: its loss is −ln D.', '*': '−ln 0.2.' },
    hints: ['The generator wants the discriminator to say “real”, D near 1.', 'Its loss −ln D is large when D is small.', '−ln 0.2 = ' + n3(-ln(0.2)) + '.'],
    solution: n3(-ln(0.2)), wrong: [k3(ln(1 - 0.2)), n3(-ln(0.8))] });

  const satOpts = ['Early on D(G(z)) is near 0, where ln(1 − D) is almost flat, so the generator would get almost no gradient; −ln D is steep there', 'It makes the discriminator learn faster', 'It removes the need for a discriminator', 'The two forms have the same gradients everywhere'];
  const satD = 0.01, gMin = 1 / (1 - satD), gNs = 1 / satD;
  C.exercise({ id: 'gm-x-saturate', chapter: 'gm-gan', title: 'Why the non-saturating loss?', level: 3, kind: 'choice',
    text: ['The original GAN objective has the generator minimise ln(1 − D(G(z))). In practice it minimises −ln D(G(z)) instead. At D(G(z)) = ' + satD + ' the slope of ln(1 − D) with respect to D has size ' + n3(gMin) + ', and that of −ln D has size ' + mn(r2(gNs)) + '. Why the switch?'],
    options: satOpts, answer: 0,
    whyNot: { 1: 'The change is in the generator’s loss only; the discriminator trains as before.', 2: 'The generator still learns only through the discriminator’s gradient.', 3: 'Both push D(G(z)) up, but the slopes differ hugely when D(G(z)) is small: ' + n3(gMin) + ' against ' + mn(r2(gNs)) + '.' },
    hints: ['Early in training the discriminator easily spots fakes: D(G(z)) is close to 0.', 'd/dD ln(1 − D) = −1 / (1 − D) is about −1 near D = 0; d/dD (−ln D) = −1 / D is huge there.', 'With ln(1 − D) the generator gets a weak signal exactly when it is worst; −ln D gives it a strong one. Both have the same aim, so the balance point does not change.'],
    solution: 0, wrong: [1, 3] });

  const copts = ['Run B: its samples land almost all near one cluster', 'Run A: it covers every cluster, so it has not learned the data', 'Neither: the discriminator gives both about 0.5', 'Both: every GAN collapses'];
  C.exercise({ id: 'gm-x-collapse-spot', chapter: 'gm-gan', title: 'Spot the collapse', level: 1, kind: 'choice',
    text: ['Two GANs are trained on 8 clusters arranged on a ring. After 1500 steps:', { table: [['run', 'D on real points', 'D on generated points', 'clusters covered', 'effective number of clusters'], ['A', '0.52', '0.43', '8 of 8', '7.9'], ['B', '0.49', '0.50', '1 of 8', '1.3']] }, 'Which run has collapsed?'],
    options: copts, answer: 0,
    whyNot: { 1: 'Covering every cluster is the goal: run A’s samples match the data.', 2: 'Being fooled is not the same as being matched: run B fools D on the one cluster it covers.', 3: 'Run A covers all 8 clusters, spread over the equivalent of 7.9.' },
    hints: ['The discriminator’s scores alone cannot show collapse: both runs fool it.', 'Look at how many clusters receive samples.', 'Run B covers 1 of 8, spread over the equivalent of 1.3: a mode collapse. These are the two runs of [Mode collapse](#topic=gen-mode-collapse).'],
    solution: 0, wrong: [1, 2] });

  const CN = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], CC = [70, 3, 0, 88, 5, 61, 2, 7], CTOT = 256, FAIR = 0.2 * (CTOT / CN.length);
  const covered = CN.filter((_, i) => CC[i] >= FAIR), covText = '{' + covered.join(', ') + '}';
  C.exercise({ id: 'gm-x-covered', chapter: 'gm-gan', title: 'Which clusters are covered?', level: 2, kind: 'set',
    text: ['A generator draws ' + CTOT + ' samples for data made of 8 clusters A to H. The number of samples within reach of each cluster:', { table: [['cluster', ...CN], ['samples near it', ...CC.map(String)]] }, 'A cluster counts as covered when it gets at least a fifth of an equal share of the samples. Which clusters are covered? Write a set, like {A, B}.'],
    answer: covText,
    hints: ['An equal share is ' + CTOT + ' / 8 = ' + CTOT / 8 + ' samples.', 'A fifth of it is ' + FAIR + ': a cluster needs at least that many.', 'Clusters with ' + FAIR + ' or more: ' + covText + ' (H just makes it with 7). That is ' + covered.length + ' of 8.'],
    solution: covText, wrong: ['{A, D, F}', '{A, B, D, E, F, G, H}'] });

  const EN = [64, 0, 0, 128, 0, 64, 0, 0], ESH = EN.filter((c) => c).map((c) => c / 256), EH = -ESH.reduce((s, p) => s + p * ln(p), 0), EFF = ex(EH);
  C.exercise({ id: 'gm-x-eff-modes', chapter: 'gm-gan', title: 'The effective number of modes', level: 3, kind: 'number',
    text: ['256 generated samples are each assigned to their nearest of 8 clusters: 64 go to cluster A, 128 to D, 64 to F and none to the rest. The effective number of modes is e raised to the entropy of the shares, exp(−Σ p ln p) over the clusters that get samples. What is it? Two decimal places.'],
    answer: r2(EFF), tolerance: 0.01,
    whyNot: { 3: 'Three clusters get samples, but not equally: D gets twice the others, so the effective number is less than 3.', 8: 'Clusters with no samples add nothing to the entropy.', [k2(EH)]: 'That is the entropy itself; raise e to it.', '*': 'Shares 0.25, 0.5 and 0.25.' },
    hints: ['The shares are 64/256 = 0.25, 128/256 = 0.5 and 0.25.', 'Entropy: −(0.25 ln 0.25 + 0.5 ln 0.5 + 0.25 ln 0.25) = ' + n3(EH) + ' nats.', 'exp(' + n3(EH) + ') = ' + n2(EFF) + ': between 2 and 3 clusters’ worth of variety. Equal shares over 8 clusters would give 8; all in one, 1.'],
    solution: n2(EFF), wrong: ['3', n2(EH)] });

  const fopts = ['Give the two networks comparable learning rates, or more discriminator steps', 'Use larger batches, so a batch that lacks variety stands out', 'Measure coverage of the data, not only how real single samples look', 'Raise the generator’s learning rate well above the discriminator’s', 'Stop training as soon as the discriminator says 0.5'];
  C.exercise({ id: 'gm-x-collapse-fix', chapter: 'gm-gan', title: 'Guarding against collapse', level: 2, kind: 'choice',
    text: ['A GAN’s samples look real but nearly all show the same digit. Which of these help? Choose all that do.'],
    options: fopts, answer: [0, 1, 2],
    whyNot: { 3: 'A generator that learns much faster than its judge is the unbalanced setting that collapsed in [Mode collapse](#topic=gen-mode-collapse).', 4: 'D near 0.5 says it is fooled, not that the data are covered: the collapsed run also reached 0.5.' },
    hints: ['Collapse came from an unbalanced duel.', 'The discriminator judges one sample at a time unless it can see a batch’s variety.', 'Balance the learning (comparable rates, more D steps), larger batches, and always check coverage. A faster generator and stopping at D = 0.5 do not help.'],
    solution: [0, 1, 2], wrong: [[0, 1, 2, 3], [4]] });

  // =========================================================================================================
  // Chapter 3: diffusion
  // =========================================================================================================
  const FX0 = [1.2, -0.4], FE = [0.5, 1.5], FAB = 0.64, sch1 = { alphaBar: [1, FAB] }, FXT = G.noise(FX0, 1, sch1, FE);
  const fxCells = { 'x₁|x_t': r3(FXT[0]), 'x₂|x_t': r3(FXT[1]) };
  const fxWrong = FX0.map((v, i) => FAB * v + (1 - FAB) * FE[i]);
  C.exercise({ id: 'gm-x-forward-xt', chapter: 'gm-diff', title: 'Noise a point in one jump', level: 1, kind: 'table',
    text: ['A clean point x₀ = (1.2, −0.4), a noise draw ε = (0.5, 1.5), and a step t with ᾱ_t = 0.64. Compute x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε.'],
    rows: ['x₁', 'x₂'], cols: ['x_t'], corner: '', cells: fxCells, tolerance: 0.002,
    hints: ['√0.64 = 0.8 and √(1 − 0.64) = √0.36 = 0.6.', 'x₁: 0.8 × 1.2 + 0.6 × 0.5; x₂: 0.8 × (−0.4) + 0.6 × 1.5.', 'x_t = (' + FXT.map(n3).join(', ') + ').'],
    solution: { 'x₁|x_t': String(r3(FXT[0])), 'x₂|x_t': String(r3(FXT[1])) },
    wrong: [{ 'x₁|x_t': n3(fxWrong[0]), 'x₂|x_t': n3(fxWrong[1]) }] });

  const BS = [0.1, 0.2, 0.3], sch3 = G.schedule({ steps: 3, b0: BS[0], b1: BS[2] }), AB3 = sch3.alphaBar[3];
  C.exercise({ id: 'gm-x-alphabar', chapter: 'gm-diff', title: 'ᾱ from a schedule', level: 2, kind: 'number',
    text: ['A tiny schedule of 3 steps has β₁ = 0.1, β₂ = 0.2 and β₃ = 0.3. What is ᾱ₃, the share of the signal’s power left after the third step? Three decimal places.'],
    answer: r3(AB3), tolerance: 0.001,
    whyNot: { 0.7: 'That is 1 − β₃ alone; ᾱ multiplies all three.', 0.4: 'That is 1 minus the sum of the βs; ᾱ is a product of (1 − β).', 0.006: 'That multiplies the βs; multiply the (1 − β)s instead.', '*': 'ᾱ₃ = (1 − β₁)(1 − β₂)(1 − β₃).' },
    hints: ['α_t = 1 − β_t: 0.9, 0.8 and 0.7.', 'ᾱ₃ is their product.', '0.9 × 0.8 × 0.7 = ' + n3(AB3) + '. So x₃ = √' + n3(AB3) + ' · x₀ + √' + n3(1 - AB3) + ' · ε, about ' + n2(sq(AB3)) + ' of the signal.'],
    solution: n3(AB3), wrong: ['0.7', '0.4', '0.006'] });

  const schL = G.schedule({ steps: 100, kind: 'linear' }), abPts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((t) => [t, r3(schL.alphaBar[t])]);
  C.exercise({ id: 'gm-x-ab-curve', chapter: 'gm-diff', title: 'Sketch ᾱ through a linear schedule', level: 2, kind: 'predict-curve',
    text: ['The linear schedule of [Diffusion: dissolving data into noise](#topic=gen-diff-forward), 100 steps with β rising from ' + n3(schL.beta[1]) + ' to ' + n2(schL.beta[100]) + '. At t = 0, ᾱ = 1. Sketch ᾱ_t, the share of the signal’s power left, from t = 0 to t = 100.'],
    plot: { xr: [0, 100], yr: [0, 1], xname: 'step t', yname: 'ᾱ_t', given: [abPts[0]], note: 'the product of (1 − β) up to t' },
    answer: abPts, tolerance: 0.08, checks: ['falling'],
    hints: ['Each step multiplies ᾱ by 1 − β_t, a little under 1.', 'Early βs are tiny, so ᾱ falls slowly at first; as β grows the fall speeds up, then ᾱ flattens near 0.', 'The engine’s values: ' + abPts.filter((p, i) => i % 2 === 0).map(([t, v]) => 't = ' + t + ': ' + v).join(', ') + '. By t = 50 less than a tenth is left.'],
    solution: abPts, wrong: [abPts.map(([t]) => [t, 1 - t / 100]), abPts.map(([t]) => [t, 1])] });

  const fwSol = 'function forwardSample(x0, alphaBar, eps) {\n  const a = Math.sqrt(alphaBar), b = Math.sqrt(1 - alphaBar);\n  return x0.map((v, i) => a * v + b * eps[i]);\n}';
  C.exercise({ id: 'gm-x-code-forward', chapter: 'gm-diff', title: 'Write the forward-diffusion sample', level: 2, kind: 'code',
    text: ['Write `forwardSample(x0, alphaBar, eps)`: a clean data point x0 (an array of numbers), the value ᾱ_t of the schedule at the step wanted, and a draw eps from a standard normal of the same length. Return x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · eps, as an array.'],
    fn: 'forwardSample', params: ['x0', 'alphaBar', 'eps'], start: 'function forwardSample(x0, alphaBar, eps) {\n  \n}', tolerance: 1e-9,
    ref: (x0, ab, e) => G.noise(x0, 1, { alphaBar: [1, ab] }, e),
    cases: [[FX0, FAB, FE], [[1, 2, 3], 1, [5, 5, 5]], [[1, 2, 3], 0, [0.1, -0.2, 0.3]], [[-1.38, 0.97], schL.alphaBar[50], [0.4, -1.2]], [[0.3], 0.5, [-1]], [new Array(6).fill(0).map((_, i) => i / 5 - 0.5), 0.9, [1, -1, 0.5, -0.5, 2, 0]]],
    hints: ['The signal is scaled by the square root of ᾱ, not by ᾱ itself.', 'a = Math.sqrt(alphaBar), b = Math.sqrt(1 − alphaBar); then x0[i] times a plus eps[i] times b.', fwSol],
    solution: fwSol,
    wrong: ['function forwardSample(x0, alphaBar, eps) {\n  return x0.map((v, i) => alphaBar * v + (1 - alphaBar) * eps[i]);\n}', 'function forwardSample(x0, alphaBar, eps) {\n  return x0.map((v, i) => Math.sqrt(alphaBar) * v + Math.sqrt(alphaBar) * eps[i]);\n}', 'function forwardSample(x0, alphaBar, eps) {\n  return x0.map((v, i) => Math.sqrt(alphaBar) * v + eps[i]);\n}'] });

  const HX = [0.9, -0.3], HAB = 0.36, HEP = [0.5, -0.75], HX0 = HX.map((v, i) => (v - sq(1 - HAB) * HEP[i]) / sq(HAB));
  const hxWrong = HX.map((v, i) => v - HEP[i]);
  C.exercise({ id: 'gm-x-x0hat', chapter: 'gm-diff', title: 'The denoiser’s guess at the clean point', level: 2, kind: 'table',
    text: ['A noisy point x_t = (0.9, −0.3) at a step where ᾱ_t = 0.36. The denoiser predicts the noise ε̂ = (0.5, −0.75). Estimate the clean point, x̂₀ = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t. Three decimal places.'],
    rows: ['x₁', 'x₂'], cols: ['x̂₀'], corner: '', cells: { 'x₁|x̂₀': r3(HX0[0]), 'x₂|x̂₀': r3(HX0[1]) }, tolerance: 0.002,
    hints: ['√0.36 = 0.6 and √(1 − 0.36) = 0.8.', 'x₁: (0.9 − 0.8 × 0.5) / 0.6; x₂: (−0.3 − 0.8 × (−0.75)) / 0.6.', 'x̂₀ = (' + HX0.map(n3).join(', ') + ').'],
    solution: { 'x₁|x̂₀': n3(HX0[0]), 'x₂|x̂₀': n3(HX0[1]) },
    wrong: [{ 'x₁|x̂₀': n3(hxWrong[0]), 'x₂|x̂₀': n3(hxWrong[1]) }] });

  const LE = [-0.22, 1.97], LEH = [0.64, 1.01], LSQ = LE.reduce((s, v, i) => s + (v - LEH[i]) ** 2, 0);
  C.exercise({ id: 'gm-x-eps-loss', chapter: 'gm-diff', title: 'The denoiser’s loss on one example', level: 1, kind: 'number',
    text: ['On one training example the true noise was ε = (−0.22, 1.97) and the denoiser predicted ε̂ = (0.64, 1.01). What is the squared error ‖ε − ε̂‖², summed over both coordinates? Two decimal places.'],
    answer: r2(LSQ), tolerance: 0.01,
    whyNot: { [k2(sq(LSQ))]: 'That is the length ‖ε − ε̂‖; the loss squares it.', [k2(LSQ / 2)]: 'That is the mean over the coordinates; this asks for the sum.', '*': 'Square each difference and add.' },
    hints: ['Differences: −0.22 − 0.64 = −0.86 and 1.97 − 1.01 = 0.96.', 'Squares: 0.7396 and 0.9216.', 'Sum: ' + n2(LSQ) + '. This is the example in [Learning to predict the noise](#topic=gen-diff-denoise), which reports 1.66.'],
    solution: n2(LSQ), wrong: [k2(sq(LSQ)), n2(LSQ / 2)] });

  C.exercise({ id: 'gm-x-steps', chapter: 'gm-diff', title: 'How many denoiser calls?', level: 1, kind: 'number',
    text: ['A diffusion model was trained with a chain of T = 250 steps. Sampling runs the whole reverse chain, from x₂₅₀ down to x₀, one image at a time. How many calls of the denoiser does it take to make 12 images?'],
    answer: 3000, tolerance: 0,
    whyNot: { 250: 'That is for one image; there are 12.', 12: 'Each image needs a call at every reverse step.', 2988: 'From x₂₅₀ to x₀ there are 250 steps, t = 250 down to t = 1, each one call.', 3012: 'From x₂₅₀ to x₀ is 250 steps, not 251.', '*': 'One call per reverse step, per image.' },
    hints: ['Each reverse step asks the denoiser for ε̂ once.', 'Going from x₂₅₀ to x₀ takes 250 steps.', '250 × 12 = 3000 calls. A VAE or a GAN would need 12.'],
    solution: '3000', wrong: ['250', '12', '2988'] });

  const RT = { xt: 0.5, beta: 0.02, ab: 0.3, eh: 0.4, z: -1 }, Ral = 1 - RT.beta, Rabp = RT.ab / Ral;
  const Rmean = (RT.xt - (RT.beta / sq(1 - RT.ab)) * RT.eh) / sq(Ral), Rsig = sq((RT.beta * (1 - Rabp)) / (1 - RT.ab)), Rx = Rmean + Rsig * RT.z;
  const Rnodiv = RT.xt - (RT.beta / sq(1 - RT.ab)) * RT.eh + Rsig * RT.z;
  C.exercise({ id: 'gm-x-reverse-step', chapter: 'gm-diff', title: 'One reverse step by hand', level: 3, kind: 'number',
    text: ['A 1-D diffusion model at step t: x_t = 0.5, β_t = 0.02 (so α_t = 0.98), ᾱ_t = 0.3, and ᾱ_{t−1} = ᾱ_t / α_t. The denoiser predicts ε̂ = 0.4, and the fresh draw is z = −1. Take one reverse step:', { math: 'x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε̂) / √α_t + σ_t · z\nσ_t² = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t)' }, 'What is x_{t−1}? Three decimal places.'],
    answer: r3(Rx), tolerance: 0.002,
    whyNot: { [k3(Rmean)]: 'That is the mean of the step; add σ_t · z, with z = −1.', [k3(Rnodiv)]: 'That skips the division by √α_t that rescales the mean.', '*': 'First the mean, then σ_t, then add σ_t · z.' },
    hints: ['β_t / √(1 − ᾱ_t) = 0.02 / √0.7 = ' + n3(RT.beta / sq(1 - RT.ab)) + '; times ε̂ = 0.4 that removes ' + mn((RT.beta / sq(1 - RT.ab) * RT.eh).toFixed(4)) + '.', 'Mean: (0.5 − ' + mn((RT.beta / sq(1 - RT.ab) * RT.eh).toFixed(4)) + ') / √0.98 = ' + n3(Rmean) + '. ᾱ_{t−1} = 0.3 / 0.98 = ' + n3(Rabp) + ', so σ_t² = 0.02 × ' + n3(1 - Rabp) + ' / 0.7 and σ_t = ' + n3(Rsig) + '.', 'x_{t−1} = ' + n3(Rmean) + ' + ' + n3(Rsig) + ' × (−1) = ' + n3(Rx) + '.'],
    solution: n3(Rx), wrong: [k3(Rmean), n3(Rnodiv)] });

  // =========================================================================================================
  // Chapter 4: choosing a generative model
  // =========================================================================================================
  const faOpts = ['A VAE makes a sample with one network call', 'A GAN makes a sample with one network call', 'Diffusion makes a sample with one call per step of its chain', 'A GAN’s training is a plain squared error with no duel', 'A plain autoencoder samples by drawing z from N(0, I)'];
  C.exercise({ id: 'gm-x-families', chapter: 'gm-choose', title: 'What each family costs', level: 2, kind: 'choice',
    text: ['Which of these are true? Choose all that are.'],
    options: faOpts, answer: [0, 1, 2],
    whyNot: { 3: 'That is diffusion. A GAN trains by a duel between generator and discriminator.', 4: 'That is a VAE. A plain autoencoder has no prior: nothing says where its codes lie.' },
    hints: ['Count the network calls between a draw of noise and a finished sample.', 'VAE and GAN: one decoder or generator call. Diffusion: one denoiser call per reverse step.', 'The first three are true. Diffusion trains by a squared error; the VAE, not the plain autoencoder, samples from N(0, I).'],
    solution: [0, 1, 2], wrong: [[0, 1, 2, 3], [0, 1, 4]] });

  const pkOpts = ['A diffusion model: steady training and good coverage; sampling time is not a constraint', 'A GAN: one call per sample, so it is fastest', 'A plain autoencoder: it rebuilds best', 'A VAE with β = 16: the codes stay closest to the prior'];
  C.exercise({ id: 'gm-x-family-pick', chapter: 'gm-choose', title: 'Choose a family', level: 3, kind: 'choice',
    text: ['A team needs a model to make varied synthetic images of a rare part for an inspection system. Training runs once; images are generated overnight in a batch, so speed of sampling hardly matters. Covering every kind of part matters most, and they cannot afford long rounds of tuning an unstable training. Which family fits best?'],
    options: pkOpts, answer: 0,
    whyNot: { 1: 'One call per sample is fast, but speed was not the need; GANs train unstably and can lose variety, which is what matters most here.', 2: 'A plain autoencoder only rebuilds; it cannot safely invent new images.', 3: 'A heavy KL makes every code look alike: the images blur into averages.' },
    hints: ['List what matters: variety, stable training. What does not: sampling time.', 'Which family pays only in sampling time?', 'Diffusion: a plain squared error to train, good coverage of the data, and many calls per sample, which an overnight batch can afford.'],
    solution: 0, wrong: [1, 3] });
})(globalThis.Nooral = globalThis.Nooral || {});
