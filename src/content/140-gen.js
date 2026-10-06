// Part 14, generative models: the chapters and the lessons. The exercises are in 141-gen-ex.js; the tour, the course
// and the pages the diagram checker looks at are in 142-gen-ch.js. Each lesson opens with a question, runs the topic
// that shows the idea at a frame chosen for the point being made, asks for a prediction, sums up, and leads to its
// exercises. Every value quoted here was printed by the topics themselves (S.Gen's stored runs and live models). The
// digit GAN and the image sampler are described by what they show, not by sample counts, so their captions hold
// whatever the samples look like.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('gm-ae')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'gm-ae', part: 'generative', n: 1, title: 'Autoencoders and VAEs', blurb: 'An image squeezed through a few numbers and rebuilt; a variational autoencoder whose codes form a map you can walk; the reparameterisation trick; and the KL term pulling against reconstruction.' });
  C.chapter({ id: 'gm-gan', part: 'generative', n: 2, title: 'Generative adversarial networks', blurb: 'A generator and a discriminator in a duel, their two losses, the standoff they aim for, and the mode collapse that comes when the duel is unbalanced.' });
  C.chapter({ id: 'gm-diff', part: 'generative', n: 3, title: 'Diffusion', blurb: 'Data dissolved into noise by a schedule, a network that learns to predict that noise, and sampling that runs the chain backwards, on points and on small images.' });
  C.chapter({ id: 'gm-choose', part: 'generative', n: 4, title: 'Choosing a generative model', blurb: 'Autoencoders, VAEs, GANs and diffusion side by side: what each learns, how it draws a sample, and what it costs.' });

  // =========================================================================================================
  // Chapter 1: autoencoders and VAEs
  // =========================================================================================================
  C.lesson({
    id: 'gm-ae', chapter: 'gm-ae', title: 'Squeezing an image through a few numbers', minutes: 6,
    summary: 'An autoencoder has two halves: an encoder that turns an image into a short code, and a decoder that rebuilds the image from the code alone. Trained to make the rebuilt image match the input, the narrow code is forced to keep what matters most. A wider code keeps more detail.',
    blocks: [
      'A 16 × 16 digit has 256 pixels, yet most of them are blank and the rest follow a few strokes. Could four numbers say enough about a digit to draw it again?',
      { topic: 'gen-ae', frame: 0, caption: 'Before training: every image is squeezed through a code of 4 numbers, and every rebuilt image is grey noise around a mean pixel of 0.50. The loss on the eight test digits is 180 nats each.' },
      { def: 'Autoencoder', text: 'An encoder x → c and a decoder c → x̂, trained together so that x̂ matches x. The code c is the bottleneck: much shorter than x.' },
      { topic: 'gen-ae', frame: 3, caption: 'Step 100: the test loss has fallen to 49.5 nats. Digit 1, an 8, is coded as four numbers, and its rebuilt pixels are off by 0.092 on average.' },
      { predict: { q: 'The same network is trained with a code of 2, 4 or 8 numbers. Which ends with the lowest loss on the test digits?', options: ['2 numbers', '4 numbers', '8 numbers', 'all the same: the decoder does the work'], answer: 2, explain: 'A wider bottleneck lets more detail through: 50.2 nats with 2 numbers, 42.2 with 4 and 34.7 with 8.' } },
      { topic: 'gen-ae', frame: 7, caption: 'After 600 steps, on all 128 test digits: 50.2 nats with a code of 2 numbers, 42.2 with 4 and 34.7 with 8.' },
      { topic: 'gen-ae', frame: 6, input: { k: '8', img: 1 }, caption: 'With 4 numbers the 8 above came back looking like a 6. With 8 numbers it comes back as an 8, its pixels off by 0.068 on average at step 600 against 0.080 with 4.' },
      { warn: 'An autoencoder rebuilds what it is given. It is not yet a generator: nothing says which codes are safe to decode, and a code made up at random may land where the decoder never learned anything.' },
      { h: 'Recap' },
      { list: ['Encoder to a short code, decoder back to the image, trained to rebuild.', 'The narrow code keeps what matters most; a wider code keeps more detail.', 'A plain autoencoder has no map of where its codes lie, so it cannot safely invent new images.'] },
      X('gm-x-ratio'), X('gm-x-ae-width'), X('gm-x-ae-sample'),
    ],
  });

  C.lesson({
    id: 'gm-vae-map', chapter: 'gm-ae', title: 'A map of codes you can walk', minutes: 7,
    summary: 'A variational autoencoder encodes each image as a small cloud of codes, and pulls every cloud towards a standard normal. The codes then fill one known region round the origin, so any point there decodes to something plausible, and a straight walk from one digit to another passes through shapes in between.',
    blocks: [
      'To invent a new digit we need a code to decode. Where should it come from? A VAE makes the answer simple: from a standard normal, because it trains its codes to sit there.',
      { topic: 'gen-vae-map', frame: 0, caption: 'Left: the mean codes μ of 128 test digits, 0s, 1s, 3s and 7s, each digit in its own region, most inside the prior’s 2σ circle. Right: what the decoder draws at 49 points of the same plane. Neighbours look alike.' },
      { p: 'The code has just 2 numbers here, so the whole map can be drawn. Every point on the right is a decoded image; none of them was in the training set.' },
      { predict: { q: 'A walk goes in a straight line from the centre of the 0s to the centre of the 1s. What does the image halfway look like?', options: ['A 0 or a 1, whichever is nearer', 'Something in between, neither a clean 0 nor a clean 1', 'Pure noise'], answer: 1, explain: 'Halfway, at (0.17, 0.14), the image looks most like the average 3 (correlation 0.71), not like either end.' } },
      { topic: 'gen-vae-map', frame: 4, caption: 'Point 4 of 7, halfway between the 0s and the 1s (1.32 from each centre): the decoded image is neither, and looks most like the average 3.' },
      { topic: 'gen-vae-map', frame: 8, caption: 'Along the walk the image changes from a 0 to a 1, first between points 3 and 4. The walk never goes further than 1.40 from the centre, so every image is a plausible digit.' },
      { topic: 'gen-vae-map', frame: 10, input: { from: '7', to: 'far', n: 9 }, caption: 'Walking from the 7s off the map, out to (2.6, −2.6), 3.68 from the centre and far beyond the 2σ circle. Few training codes lie out there; this decoder simply goes on drawing 1s, but nothing in training checked what it draws so far out, and a draw from N(0, I) almost never lands there.' },
      { h: 'Recap' },
      { list: ['A VAE’s codes fill a known region, pulled towards N(0, I).', 'Nearby codes decode to similar images, so walks are smooth.', 'Draw z from N(0, I) and decode it: a new image.'] },
      X('gm-x-walk'),
    ],
  });

  C.lesson({
    id: 'gm-reparam', chapter: 'gm-ae', title: 'The reparameterisation trick', minutes: 7,
    summary: 'The encoder of a VAE gives a mean μ and a log-variance log σ² for each image, and the code is a random draw from N(μ, σ²). A draw has no gradient, so z is built instead as μ + σ ⊙ ε, with ε drawn from N(0, I) apart from the network. Then ∂z/∂μ = 1 and ∂z/∂σ = ε, and training reaches the encoder.',
    blocks: [
      'Training follows gradients back through every step from the loss to the weights. But a VAE’s code is drawn at random. What is the derivative of “a random number near μ” with respect to μ?',
      { topic: 'gen-reparam', frame: 1, caption: 'Test digit 2, a 3. The encoder gives μ = (−1.54, 0.467) and log σ² = (−3.17, −4.03): not one code, but a normal distribution over codes.' },
      { topic: 'gen-reparam', frame: 2, caption: 'σ = exp(½ log σ²) = (0.205, 0.133). The ellipses mark one and two σ around μ.' },
      { topic: 'gen-reparam', frame: 3, caption: 'ε = (0.123, −0.853) is drawn from a standard normal, outside the network. All the randomness lives here.' },
      { predict: { q: 'With μ₁ = −1.54, σ₁ = 0.205 and ε₁ = 0.123, what is z₁ = μ₁ + σ₁ ε₁?', options: ['−1.52', '−1.54', '−1.29'], answer: 0, explain: '−1.54 + 0.205 × 0.123 = −1.54 + 0.025 = −1.52.' } },
      { topic: 'gen-reparam', frame: 4, caption: 'z₁ = −1.54 + 0.205 × 0.123 = −1.52 and z₂ = 0.467 + 0.133 × (−0.853) = 0.353.' },
      { topic: 'gen-reparam', frame: 5, caption: 'Now ∂z/∂μ = 1 and ∂z/∂σ = ε = (0.123, −0.853): plain derivatives of a sum and a product, so the gradient of the loss reaches the encoder. Had z been drawn directly from N(μ, σ²), there would be nothing to differentiate.' },
      { math: 'z = μ + σ ⊙ ε,   ε ~ N(0, I),   σ = exp(½ · log σ²)\n∂z/∂μ = 1        ∂z/∂σ = ε        ∂σ/∂(log σ²) = ½ σ' },
      { topic: 'gen-reparam', frame: 6, caption: 'Six more draws of ε give six more codes inside the ellipses, and six images that all look like a 3: the noise teaches the decoder that the whole cloud stands for this digit.' },
      { tip: 'The encoder outputs log σ² rather than σ so that any real number is allowed: σ = exp(½ log σ²) is always positive.' },
      { h: 'Recap' },
      { list: ['The encoder outputs μ and log σ²; σ = exp(½ log σ²).', 'z = μ + σ ⊙ ε, with ε drawn apart from the network.', 'The gradient of z is 1 for μ and ε for σ.'] },
      X('gm-x-reparam'), X('gm-x-reparam-grad'), X('gm-x-code-reparam'),
    ],
  });

  C.lesson({
    id: 'gm-kl', chapter: 'gm-ae', title: 'The KL term against reconstruction', minutes: 8,
    summary: 'A VAE’s loss is reconstruction plus a KL term, ½ Σ (μ² + σ² − 1 − log σ²), which measures how far each code’s normal is from N(0, I). The two pull against each other: reconstruction wants precise, spread-out codes; the KL wants them crowded at the origin. A weight β on the KL moves the balance.',
    blocks: [
      'If the KL term pulls every code to the centre, and reconstruction pushes them apart, which one wins? It depends on how much each is weighed.',
      { math: 'loss = reconstruction + β · KL\nKL = ½ Σ over code coordinates (μ² + σ² − 1 − log σ²)\nELBO = −(reconstruction + KL)   (for β = 1)' },
      { topic: 'gen-reparam', frame: 8, caption: 'For the 3 of the last lesson: ½ (μ² + σ² − 1 − log σ²) is 2.29 for the first coordinate and 1.63 for the second, 3.93 nats in all.' },
      { topic: 'gen-reparam', frame: 9, caption: 'With a reconstruction loss of 45 nats, the ELBO is −(45 + 3.93) = −49. Training raises it.' },
      { predict: { q: 'For which μ and σ is the KL of one coordinate zero?', options: ['μ = 0 and σ = 0', 'μ = 0 and σ = 1', 'μ = 1 and σ = 1'], answer: 1, explain: '½ (0 + 1 − 1 − log 1) = 0: the code’s normal is then exactly the prior.' } },
      { topic: 'gen-kl', frame: 6, caption: 'Four VAEs differing only in β. At the end, reconstruction is 39.4, 40.1, 44.8 and 55.1 nats for β = 0, 1, 4 and 16; the KL is 31.2, 3.39, 1.56 and 0.00383.' },
      { topic: 'gen-kl', frame: 5, input: { beta: '0' }, caption: 'β = 0: nothing pulls the codes in. They spread out, a mean 6.61 from the centre, far beyond the region a draw from N(0, I) would reach.' },
      { topic: 'gen-kl', frame: 5, input: { beta: '16' }, caption: 'β = 16: the KL wins. The codes crowd within 0.03 of the centre, the KL is almost 0, and the decoder can only draw vague averages: reconstruction 55.1 nats.' },
      { warn: 'A KL of almost zero is not a success: an encoder that outputs μ = 0 and σ = 1 for every image pays no KL at all, but its code then says nothing about the image. This is called posterior collapse.' },
      { h: 'Recap' },
      { list: ['KL = ½ Σ (μ² + σ² − 1 − log σ²): zero only when μ = 0 and σ = 1.', 'Raising β lowers the KL and raises the reconstruction loss.', 'β = 1 is the standard VAE; ELBO = −(reconstruction + KL).'] },
      X('gm-x-kl-1d'), X('gm-x-kl-2d'), X('gm-x-kl-zero'), X('gm-x-elbo'), X('gm-x-beta'), X('gm-x-beta-curve'), X('gm-x-code-kl'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: GANs
  // =========================================================================================================
  C.lesson({
    id: 'gm-gan', chapter: 'gm-gan', title: 'A generator and a discriminator in a duel', minutes: 8,
    summary: 'A GAN trains two networks against each other. The generator turns noise into samples; the discriminator gives each sample the probability that it is real. The discriminator is trained to tell them apart, the generator to fool it. At the balance point the samples sit on the data and the discriminator can only guess: ½ everywhere.',
    blocks: [
      'A forger and an inspector: the inspector learns from every fake caught, the forger from every fake that is caught. Neither ever sees a rule for what a good banknote is. Can the two together still learn to make one?',
      { topic: 'gen-gan-duel', frame: 0, caption: 'Eight clusters on a ring are the real data. Before training the generator squeezes its noise into a blob in the middle and reaches none of them.' },
      { math: 'D(x): the probability that x is real\nD minimises  −log D(x_real) − log(1 − D(G(z)))\nG minimises  −log D(G(z))      (the non-saturating form)' },
      { topic: 'gen-gan-duel', frame: 1, caption: 'Step 100: on its last batch the discriminator gives real points 0.48 on average and generated ones 0.47. Its loss is 1.37, close to 2 ln 2 = 1.386, the loss of a pure guess.' },
      { predict: { q: 'The discriminator gives a fake point 0.2. What is that point’s term in the generator’s loss, −log D(G(z))?', options: ['0.22', '1.61', '0.80'], answer: 1, explain: '−ln 0.2 = 1.61. The less real the fake looks, the larger the generator’s loss.' } },
      { topic: 'gen-gan-duel', frame: 6, caption: 'Step 800: the generator’s points now cover all 8 clusters. The shading is the discriminator’s view of the plane: where it believes points are real.' },
      { topic: 'gen-gan-duel', frame: 10, caption: 'After 1500 steps: a standoff. D gives real and generated points 0.52 and 0.43, near the 0.5 of a coin toss, because the generator’s points sit on the clusters.' },
      { topic: 'gen-gan-digits', frame: 0, caption: 'The same duel on 16 × 16 digits, trained on 0s, 1s and 7s. Before training every generated image is a grey fog, and the untrained discriminator gives the fakes 0.50 and the real digits 0.48 on average.' },
      { topic: 'gen-gan-digits', frame: 4, caption: 'Step 1000: the fog has become strokes. A digit reader, a separate classifier trained on real digits, takes the 16 fakes for 6 × 0, 3 × 1 and 7 × 7; the discriminator gives them 0.42 on average against 0.49 for real digits.' },
      { topic: 'gen-gan-digits', frame: 8, caption: '64 fresh samples after 4000 steps. The digit reader, which reads 999 of 1000 new real digits correctly, takes them for 16 × 0, 21 × 1 and 22 × 7, with 5 strays read as other digits, and is sure of 56 of them. 64 real training digits read as 24, 22 and 18.' },
      { p: 'On images the duel needed help to stay balanced, and this run has four kinds: real images are labelled 0.9 instead of 1, so the discriminator never becomes certain; it learns four times faster than the generator; it is told how spread out each batch is, so a batch of near copies gives itself away; and each of its weight matrices is scaled so it stretches no direction by more than 1. The next lesson shows the same duel without them.' },
      { tip: 'The generator never sees a real image. Everything it learns comes through the discriminator’s gradient: where to move each sample to look more real.' },
      { h: 'Recap' },
      { list: ['The discriminator learns to tell real from generated; the generator learns to fool it.', 'D’s loss is −log D(real) − log(1 − D(fake)); G’s is −log D(fake).', 'At the balance D = ½ everywhere, and D’s loss is 2 ln 2 ≈ 1.386.'] },
      X('gm-x-d-loss'), X('gm-x-g-loss'), X('gm-x-saturate'),
    ],
  });

  C.lesson({
    id: 'gm-collapse', chapter: 'gm-gan', title: 'Mode collapse', minutes: 7,
    summary: 'A generator only has to fool the discriminator, not to cover the data. If it finds a few places the discriminator believes, it may put every sample there. The samples look real but the variety is gone. Counting how many clusters receive a fair share of samples, and how spread the samples are, makes this visible.',
    blocks: [
      'A generator that makes one perfect digit, over and over, would fool a judge who looks at one image at a time. What stops a GAN from doing exactly that?',
      { topic: 'gen-mode-collapse', frame: 0, caption: 'Two generators on the same ring: a balanced run and one whose generator learns much faster than its discriminator. A cluster counts as covered when at least a fifth of an equal share of samples, 6.4 of 256, lands near it.' },
      { topic: 'gen-mode-collapse', frame: 2, caption: 'Step 200: the unbalanced run already spreads over the equivalent of only 2.8 clusters.' },
      { predict: { q: 'At the end, how many of the 8 clusters will the unbalanced generator cover?', options: ['8', 'about 4', '1'], answer: 2, explain: 'It covers 1 cluster, with its samples spread over the equivalent of 1.3: it found one place the discriminator believes and stayed.' } },
      { topic: 'gen-mode-collapse', frame: 10, caption: 'The balanced generator ends covering 8 clusters, the unbalanced one 1.' },
      { topic: 'gen-gan-duel', frame: 10, input: { run: 'collapse', field: 'on' }, caption: 'The unbalanced duel from inside: D gives real points 0.49 and generated ones 0.50. The discriminator is fooled, yet 7 of the 8 clusters receive no samples.' },
      { def: 'Effective number of modes', text: 'e raised to the entropy of the shares of samples nearest each cluster: 8 when the shares are equal, 1 when every sample goes to one cluster.' },
      { topic: 'gen-gan-digits', frame: 7, input: { run: 'plain', pick: 1 }, caption: 'On images the same thing happens. The plain duel on digits, with none of the four remedies of the last lesson, after 1500 steps: the digit reader takes all 64 of its samples for 0, though it was trained on 0s, 1s and 7s. The steadied duel ends with 16 × 0, 21 × 1 and 22 × 7.' },
      { p: 'What helps against it:' },
      { list: ['Balance the two networks: comparable learning rates, or more discriminator steps.', 'Larger batches, so the discriminator sees whether a batch is varied, or a feature that tells it the batch’s spread.', 'Softer targets for real data (0.9 rather than 1) and a cap on how steep the discriminator can be, such as spectral normalisation.', 'Losses and tricks that reward variety, and always checking coverage, not only realism.'] },
      { h: 'Recap' },
      { list: ['Fooling the discriminator does not require covering the data.', 'Mode collapse: realistic samples with little variety.', 'Measure it: clusters covered, effective number of modes.'] },
      X('gm-x-collapse-spot'), X('gm-x-covered'), X('gm-x-eff-modes'), X('gm-x-collapse-fix'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: diffusion
  // =========================================================================================================
  C.lesson({
    id: 'gm-forward', chapter: 'gm-diff', title: 'Dissolving data into noise', minutes: 7,
    summary: 'The forward process of a diffusion model adds a little Gaussian noise at each of T steps, following a schedule β₁ … β_T. Any step can be reached in one jump: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε, where ᾱ_t is the product of (1 − β) up to t. By the last step almost nothing of the data is left.',
    blocks: [
      'Turning noise into data is hard. Turning data into noise is easy: just keep adding noise. Diffusion learns the hard direction by watching the easy one, step by step.',
      { topic: 'gen-diff-forward', frame: 0, caption: 'At t = 0: 256 points on two moons, and a 16 × 16 circle. A linear schedule over 100 steps sets β from 0.001 at the first step to 0.2 at the last.' },
      { math: 'ᾱ_t = (1 − β₁)(1 − β₂) … (1 − β_t)\nx_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)' },
      { topic: 'gen-diff-forward', frame: 4, caption: 't = 20: ᾱ = 0.665, so x_t = 0.816 · x₀ + 0.578 · ε. Though x₀ still carries 0.816 of the scale, noise of 0.578 is enough to smear the two moons into one cloud.' },
      { predict: { q: 'At t = 50, ᾱ = 0.0742. How much of x₀ is left in x_t?', options: ['0.0742 of it', '0.272 of it', '0.926 of it'], answer: 1, explain: '√0.0742 = 0.272: x_t = 0.272 · x₀ + 0.962 · ε. The scale of the signal is the square root of ᾱ.' } },
      { topic: 'gen-diff-forward', frame: 6, caption: 't = 50: x_t = 0.272 · x₀ + 0.962 · ε. Now noise is almost all there is.' },
      { topic: 'gen-diff-forward', frame: 8, caption: 't = 100: ᾱ = 2.04 × 10⁻⁵, so x_t = 0.005 · x₀ + 1.000 · ε. This is what sampling starts from.' },
      { topic: 'gen-diff-forward', frame: 6, input: { kind: 'cosine', T: 100, data: 'moons', shape: 'circle' }, caption: 'A cosine schedule spends its noise more evenly: at t = 50, ᾱ = 0.494, so half the signal’s power is still there.' },
      { tip: 'Because x_t can be drawn in one jump from x₀, training never has to run the chain: pick a t, draw ε, and compute x_t directly.' },
      { h: 'Recap' },
      { list: ['A schedule of small β_t adds noise step by step.', 'ᾱ_t is the product of the (1 − β); x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε in one jump.', 'At t = T the data are pure noise.'] },
      X('gm-x-forward-xt'), X('gm-x-alphabar'), X('gm-x-ab-curve'), X('gm-x-code-forward'),
    ],
  });

  C.lesson({
    id: 'gm-denoise', chapter: 'gm-diff', title: 'Learning to predict the noise', minutes: 7,
    summary: 'The denoiser is trained on noisy points made by the forward process: given x_t and t, it predicts the noise ε that was added, and the loss is the squared error. From a prediction ε̂ the clean point can be estimated as x̂₀ = (x_t − √(1 − ᾱ_t) ε̂) / √ᾱ_t. With little noise the estimate is sharp; with much noise it gathers towards the middle of the data, and with almost pure noise the division by a tiny √ᾱ_t magnifies every error and the estimate flies far off.',
    blocks: [
      'If a network could look at a noisy point and say exactly which noise was added, we could subtract it and get the data back. What should such a network be trained on?',
      { topic: 'gen-diff-denoise', frame: 0, caption: 'One training example: a clean point x₀ = (1.47, −0.82) from the moons, a step t = 30 drawn at random, and fresh noise ε = (−0.22, 1.97).' },
      { topic: 'gen-diff-denoise', frame: 1, caption: 'At t = 30, ᾱ = 0.397: x_t = 0.630 · x₀ + 0.776 · ε = (0.76, 1.01). The network sees only x_t and t.' },
      { predict: { q: 'With x_t = (0.76, 1.01), ᾱ = 0.397 and a predicted ε̂, how do we estimate the clean point?', options: ['x_t − ε̂', '(x_t − √(1 − ᾱ) · ε̂) / √ᾱ', 'x_t / ᾱ'], answer: 1, explain: 'Undo the forward jump: subtract the scaled noise, then divide by the scale of the signal.' } },
      { topic: 'gen-diff-denoise', frame: 2, caption: 'The denoiser predicts ε̂ = (0.64, 1.01) against the true (−0.22, 1.97): a squared error of 1.66. Its guess at the clean point is x̂₀ = (0.41, 0.36).' },
      { topic: 'gen-diff-denoise', frame: 3, caption: 'At t = 90 the denoiser is asked about 121 places on a grid. So little of x₀ is left that x̂₀ divides by √ᾱ = 0.013, magnifying every small error in ε̂ 77 times: the guesses lie a mean 7.53 from the data, far past it.' },
      { topic: 'gen-diff-denoise', frame: 4, caption: 'At t = 50 the guesses gather towards the middle of the data: half of them lie within 0.65 of its centre, where half the data lie beyond 1.29.' },
      { topic: 'gen-diff-denoise', frame: 5, caption: 'At t = 20 each place is pulled to the nearest part of a moon: the guesses lie a mean 0.18 from the data.' },
      { topic: 'gen-diff-denoise', frame: 7, caption: 'Training ran 3000 steps of 128 examples. The loss fell from 0.495 to about 0.232; it cannot reach 0, since at small t the noise is a tiny nudge that cannot be told apart from the data’s own spread. At large t, ε is easy to read off.' },
      { h: 'Recap' },
      { list: ['Training: draw x₀, t and ε; make x_t; predict ε; loss ‖ε − ε̂‖².', 'x̂₀ = (x_t − √(1 − ᾱ_t) ε̂) / √ᾱ_t.', 'The loss never reaches 0: at small t the tiny noise cannot be told from the data’s own spread; at large t, x_t is almost ε and ε is easy to read off.'] },
      X('gm-x-x0hat'), X('gm-x-eps-loss'),
    ],
  });

  C.lesson({
    id: 'gm-sample', chapter: 'gm-diff', title: 'Sampling: running the chain backwards', minutes: 8,
    summary: 'To sample, start from pure noise at t = T and take T reverse steps. Each step asks the denoiser for ε̂, removes a little of it, rescales, and adds a little fresh noise (none at the last step). The price is one network call per step: a hundred steps cost a hundred calls per sample.',
    blocks: [
      'We have a network that guesses the noise in a noisy point. How do we turn that into a brand-new point, starting from nothing at all?',
      { topic: 'gen-diff-sample', frame: 0, caption: 'Start: 200 points of pure noise, x₁₀₀ ~ N(0, I).' },
      { math: 'x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε̂) / √α_t + σ_t · z\nσ_t² = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t),   z ~ N(0, I)\nat the last step (t = 1) no fresh noise is added' },
      { topic: 'gen-diff-sample', frame: 1, caption: 't = 90: in the step from 91 the chain removed 0.182 × ε̂, rescaled by 1/√α = 1.106 and added fresh noise with σ = 0.427.' },
      { predict: { q: 'Halfway, at t = 50, will the points already look like two moons?', options: ['Yes, clearly', 'Not yet: they lie about as far from the data as at the start', 'They will have collapsed onto one point'], answer: 1, explain: 'At t = 50 they lie a mean 0.39 from the data, as at the start; the shape forms late, in the last 30 steps.' } },
      { topic: 'gen-diff-sample', frame: 5, caption: 't = 50: the points still lie a mean 0.39 from the data, as far as at the start. The early steps set the coarse layout; the detail comes later.' },
      { topic: 'gen-diff-sample', frame: 10, caption: 't = 0: the last step adds no noise. The points lie a mean 0.09 from the data.' },
      { topic: 'gen-diff-sample', frame: 11, caption: 'The sliced Wasserstein distance from the samples to the data is 0.085, against 0.291 for plain noise.' },
      { topic: 'gen-diff-images', frame: 0, caption: 'The same chain on 16 × 16 images: eight images of pure noise, 256 pixels each, and 60 reverse steps ahead.' },
      { topic: 'gen-diff-images', frame: 5, caption: 't = 30, halfway: ᾱ = 0.494, so the chain still treats the images as 71% noise by scale, yet dark blobs have begun to gather where the circles will be. A small convolutional denoiser makes every step, one call for all eight images.' },
      { topic: 'gen-diff-images', frame: 11, caption: 'After 60 steps: 6 of the 8 samples are circles with dark areas inside the 40 to 116 pixels of the training circles, and 2 faded to a blank ground. Of 64 more samples, 60 come out in that range.' },
      { p: 'The image denoiser was trained for 1000 steps on 2048 circles, and each reverse step’s fresh noise was scaled by 0.7, a lower temperature than the formula gives: a little less variety for steadier shapes. Even so, a sample now and then never commits to a circle and fades away, a reminder that a small model’s chain does not always end on the data.' },
      { tip: 'Fewer steps sample faster but take bigger jumps. Faster samplers skip steps of the chain; the trade between speed and quality is the main cost of diffusion.' },
      { h: 'Recap' },
      { list: ['Sampling starts from pure noise and takes T reverse steps.', 'Each step: predict ε̂, remove some of it, rescale, add a little fresh noise.', 'Cost: one denoiser call per step, for every sample.'] },
      X('gm-x-steps'), X('gm-x-reverse-step'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: choosing a generative model
  // =========================================================================================================
  C.lesson({
    id: 'gm-families', chapter: 'gm-choose', title: 'Four families side by side', minutes: 6,
    summary: 'An autoencoder only rebuilds; a VAE samples in one call but its images are soft; a GAN samples in one call with sharp images but trains unstably and may lose variety; diffusion trains steadily and covers the data, but needs a network call for every step of the chain.',
    blocks: [
      'You need new images. Which model should you train? Each family pays somewhere: in sharpness, in variety, in stable training or in sampling time.',
      { topic: 'gen-compare', frame: 0, caption: 'An autoencoder rebuilds its input, off by 0.067 per pixel on average, but its code space has no known shape: there is no safe place to draw a new code from.' },
      { topic: 'gen-compare', frame: 1, caption: 'A VAE draws z from its prior and decodes: one network call. Its samples are varied but soft-edged.' },
      { topic: 'gen-compare', frame: 2, caption: 'A GAN also needs one call, and its samples are sharp. A digit reader takes its 64 samples for 16 × 0, 21 × 1 and 22 × 7, with 5 strays: the steadied duel kept all three digits it was trained on, where the plain duel made only 0s.' },
      { predict: { q: 'Which family needs the most network calls to make one sample?', options: ['VAE', 'GAN', 'Diffusion'], answer: 2, explain: 'Each diffusion sample here took 60 calls of the denoiser, one per reverse step; the VAE and the GAN need 1.' } },
      { topic: 'gen-compare', frame: 3, caption: 'Diffusion pays at sampling time: 60 denoiser calls per image. Its training is the steadiest: a plain squared error, with no prior to fit and no critic to balance.' },
      { table: [['Family', 'Trained by', 'Calls per sample', 'Typical weakness'], ['Autoencoder', 'reconstruction', 'cannot sample', 'no map of its codes'], ['VAE', 'reconstruction + KL', '1', 'soft, blurred images'], ['GAN', 'a duel with a discriminator', '1', 'unstable training, mode collapse'], ['Diffusion', 'predicting the noise', 'one per step (60 to 1000)', 'slow sampling']], caption: 'Each family trades something away.' },
      { h: 'Recap' },
      { list: ['VAE: a well-shaped code space, softer images.', 'GAN: sharp images in one call, at the risk of instability and lost variety.', 'Diffusion: steady training and good coverage, at many calls per sample.'] },
      X('gm-x-families'), X('gm-x-family-pick'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
