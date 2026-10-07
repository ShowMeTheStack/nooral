// Part 15, reinforcement learning: the guided tour, the course, and the pages the diagram checker looks at. The
// workflow canvas has no environments, agents or reward scores, so this part sets no model challenge; its exercises
// work the updates by hand and in code.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.courses.has('re-course')) return;

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 're-tour', title: 'Tour: reinforcement learning', blurb: 'Bandits, an MDP, value iteration, Q-learning against SARSA, a policy gradient and a deep Q-network, in eight steps.',
    steps: [
      { route: 'part=rl', target: '#main', text: 'Part 15 of the catalogue: reinforcement learning. Every topic runs the engine’s own agents on bandits, grid worlds and a cart with a pole, with every value and update shown.' },
      { route: 'topic=rl-bandit', target: '#inputs', text: 'Choose a strategy here, greedy, ε-greedy, UCB1 or Thompson sampling, and watch every pull and every estimate.' },
      { target: '#btnPlay', text: 'Play runs pull after pull; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=rl-value-iteration', target: '#stageWrap', text: 'Value iteration on a grid: values spread out from the goal sweep by sweep, and the arrows form the best policy.' },
      { route: 'topic=rl-cliff', target: '#stageWrap', text: 'Q-learning and SARSA side by side on the cliff: one learns the edge, the other a safer route.' },
      { route: 'topic=rl-reinforce', target: '#stageWrap', text: 'A policy network learning to balance a pole, update by update.' },
      { route: 'lesson=re-vi', target: '#lessonView', text: 'Lessons run the topics at the frames that matter and ask for a prediction on the way: here, a Bellman backup.' },
      { route: 'learn', target: '[data-chapter="re-bandits"]', text: 'The lessons and exercises of this part start here: bandit estimates and UCB scores, returns, Bellman backups, Q-learning updates, and value iteration in code.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 're-course', title: 'Reinforcement learning', blurb: 'All ten topics of part 15, from a bandit’s first pull to a deep Q-network, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 're-bandit', note: 'Explore or exploit.' },
      { topic: 'rl-bandit', note: 'Pull by pull, four strategies.' },
      { exercise: 're-x-estimates', note: 'Estimates by hand.' },
      { topic: 'rl-regret', note: 'Regret curves, averaged.' },
      { exercise: 're-x-ucb', note: 'UCB scores by hand.' },
      { lesson: 're-mdp', note: 'States, actions, rewards, γ.' },
      { topic: 'rl-mdp', note: 'An MDP on a slippery grid.' },
      { exercise: 're-x-return', note: 'A discounted return.' },
      { topic: 'rl-value-iteration', note: 'Sweep by sweep.' },
      { exercise: 're-x-vi-sweep', note: 'Two sweeps by hand.' },
      { topic: 'rl-policy-iteration', note: 'Evaluate, improve, repeat.' },
      { lesson: 're-mc', note: 'No model: learn from episodes.' },
      { topic: 'rl-mc', note: 'Monte Carlo control.' },
      { topic: 'rl-cliff', note: 'Q-learning against SARSA.' },
      { exercise: 're-x-targets', note: 'Their two targets.' },
      { exercise: 're-x-code-q', note: 'The update in code.' },
      { topic: 'rl-explore', note: 'How much to explore.' },
      { topic: 'rl-reinforce', note: 'A policy gradient on cart-pole.' },
      { exercise: 're-x-pg-sign', note: 'Which way it pushes.' },
      { topic: 'rl-dqn', note: 'Replay and a target network.' },
      { exercise: 're-x-dqn-why', note: 'Why both.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson re-bandit')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'rl').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['re-x-estimates', 're-x-ucb', 're-x-regret-curve', 're-x-return', 're-x-bellman', 're-x-vi-sweep', 're-x-code-vi', 're-x-mc-return', 're-x-targets', 're-x-pg-sign', 're-x-dqn-why'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course re-course', hash: 'course=re-course', ready: "document.querySelector('#courseView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
