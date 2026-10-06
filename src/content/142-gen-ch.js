// Part 14, generative models: the guided tour, the course, and the pages the diagram checker looks at. The workflow
// canvas has no generative blocks and no score for generated samples on a hidden set, so this part sets no model
// challenge; its exercises work the pieces by hand and in code.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.courses.has('gm-course')) return;

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'gm-tour', title: 'Tour: generative models', blurb: 'Autoencoders, the VAE’s map and its KL term, a GAN duel, mode collapse and diffusion, in eight steps.',
    steps: [
      { route: 'part=generative', target: '#main', text: 'Part 14 of the catalogue: generative models. Every topic runs the engine’s own networks, trained here on fixed seeds, with their codes, losses and samples drawn.' },
      { route: 'topic=gen-ae', target: '#stageWrap', text: 'An autoencoder squeezes each digit through a few numbers and rebuilds it. Change the code size and compare the losses.' },
      { target: '#btnPlay', text: 'Play runs training snapshot by snapshot; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=gen-vae-map', target: '#inputs', text: 'A VAE’s codes form a map. Choose where a walk starts and ends, and watch each point decoded.' },
      { route: 'topic=gen-gan-duel', target: '#stageWrap', text: 'A generator and a discriminator in a duel on eight clusters, with the discriminator’s view of the plane shaded.' },
      { route: 'topic=gen-mode-collapse', target: '#stageWrap', text: 'A balanced and an unbalanced duel side by side: the unbalanced generator ends on one cluster.' },
      { route: 'lesson=gm-forward', target: '#lessonView', text: 'Lessons run the topics at the frames that matter and ask for a prediction on the way: here, data dissolving into noise.' },
      { route: 'learn', target: '[data-chapter="gm-ae"]', text: 'The lessons and exercises of this part start here: the reparameterisation and the KL by hand and in code, the GAN losses, coverage, and a diffusion step.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'gm-course', title: 'Generative models', blurb: 'All twelve topics of part 14, from an autoencoder’s bottleneck to sampling images by diffusion, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'gm-ae', note: 'Squeeze and rebuild.' },
      { topic: 'gen-ae', note: 'The bottleneck, at three widths.' },
      { lesson: 'gm-vae-map', note: 'Codes that form a map.' },
      { topic: 'gen-vae-map', note: 'Walk across it.' },
      { topic: 'gen-reparam', note: 'z = μ + σ ⊙ ε, step by step.' },
      { exercise: 'gm-x-reparam', note: 'The trick by hand.' },
      { topic: 'gen-kl', note: 'β moves the balance.' },
      { exercise: 'gm-x-kl-1d', note: 'One KL term by hand.' },
      { exercise: 'gm-x-code-kl', note: 'The KL in code.' },
      { lesson: 'gm-gan', note: 'The duel.' },
      { topic: 'gen-gan-duel', note: 'Generator against discriminator on a ring.' },
      { exercise: 'gm-x-d-loss', note: 'The discriminator’s loss.' },
      { topic: 'gen-mode-collapse', note: 'When the duel is unbalanced.' },
      { topic: 'gen-gan-digits', note: 'The duel on digits.' },
      { exercise: 'gm-x-covered', note: 'Count the covered clusters.' },
      { lesson: 'gm-forward', note: 'Into noise.' },
      { topic: 'gen-diff-forward', note: 'Schedules and ᾱ.' },
      { exercise: 'gm-x-forward-xt', note: 'One jump by hand.' },
      { topic: 'gen-diff-denoise', note: 'Predicting the noise.' },
      { topic: 'gen-diff-sample', note: 'Back from noise to moons.' },
      { topic: 'gen-diff-images', note: 'Back from noise to images.' },
      { exercise: 'gm-x-steps', note: 'The cost of sampling.' },
      { topic: 'gen-compare', note: 'The four families side by side.' },
      { exercise: 'gm-x-family-pick', note: 'Choose one.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson gm-ae')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'generative').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['gm-x-reparam', 'gm-x-code-reparam', 'gm-x-kl-2d', 'gm-x-beta', 'gm-x-beta-curve', 'gm-x-d-loss', 'gm-x-covered', 'gm-x-forward-xt', 'gm-x-ab-curve', 'gm-x-reverse-step', 'gm-x-families'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course gm-course', hash: 'course=gm-course', ready: "document.querySelector('#courseView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
