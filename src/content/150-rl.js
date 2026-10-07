// Part 15, reinforcement learning: the chapters and the lessons. The exercises are in 151-rl-ex.js; the tour, the
// course and the pages the diagram checker looks at are in 152-rl-ch.js. Each lesson opens with a question, runs the
// topic that shows the idea at a frame chosen for the point being made, asks for a prediction, sums up, and leads to
// its exercises. Every value quoted here was printed by the topics themselves (S.RL on the topics' own worlds).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('re-bandit')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 're-bandits', part: 'rl', n: 1, title: 'Bandits: explore or exploit', blurb: 'Slot machines with unknown payouts: running estimates, ε-greedy, the regret of every pull, and the two cleverer strategies, UCB and Thompson sampling.' });
  C.chapter({ id: 're-plan', part: 'rl', n: 2, title: 'Planning with a known model', blurb: 'States, actions, rewards and the discount; the return and the value of a state; the Bellman backup; and value and policy iteration on a grid.' });
  C.chapter({ id: 're-learn', part: 'rl', n: 3, title: 'Learning from experience', blurb: 'No model, only episodes: Monte Carlo averages, Q-learning and SARSA updating one step at a time, and how much to explore.' });
  C.chapter({ id: 're-deep', part: 'rl', n: 4, title: 'Policies and networks', blurb: 'A policy network trained by REINFORCE to balance a pole, and a Q-network kept steady by a replay buffer and a target network.' });

  // =========================================================================================================
  // Chapter 1: bandits
  // =========================================================================================================
  C.lesson({
    id: 're-bandit', chapter: 're-bandits', title: 'Pulling arms: estimates and ε-greedy', minutes: 7,
    summary: 'A bandit has several arms, each paying out at an unknown average. The agent keeps a running mean of the rewards from each arm. Greedy play always pulls the best-looking arm, and can lock onto a poor one for ever; ε-greedy pulls a random arm a fraction ε of the time, so every estimate keeps improving.',
    blocks: [
      'Five slot machines, each paying out with a different, unknown chance. You have 500 coins. Play the machine that has paid best so far, or try the others in case one is better?',
      { topic: 'rl-bandit', frame: 0, caption: 'Five arms pay out with probabilities 0.68, 0.13, 0.46, 0.16 and 0.71: the best is arm 4. The agent knows none of this; every estimate starts at 0.' },
      { math: 'Q(a) ← Q(a) + (r − Q(a)) / N(a)        (the running mean of arm a’s rewards)\nε-greedy: with probability ε a random arm, else argmax Q(a)' },
      { topic: 'rl-bandit', frame: 4, caption: 'Pull 4: arm 0 pays 1, so its estimate jumps from 0 to 1 after 1 pull. Now it looks best by far.' },
      { predict: { q: 'Plain greedy play (ε = 0) now sticks with arm 0, which pays 0.68. Will it ever find arm 4, which pays 0.71?', options: ['Yes, soon', 'Only by luck, if arm 0’s estimate falls below the others’', 'No: the other estimates stay at 0, below arm 0’s'], answer: 2, explain: 'After 500 pulls greedy has pulled arm 0 497 times and arm 4 never: the other estimates stay at 0 and never get a chance to rise.' } },
      { topic: 'rl-bandit', frame: 14, input: { algo: 'greedy', eps: 0.1, c: 1, k: 5, steps: 500, seed: 3 }, caption: 'Greedy after 500 pulls: arm 0 has had 497 of them, the best arm none. It has settled on the wrong arm.' },
      { topic: 'rl-bandit', frame: 14, caption: 'ε-greedy with ε = 0.1 after 500 pulls: the estimates are 0.66, 0.00, 0.46, 0.14 and 0.64, close to the truth, but on this run it still favours arm 0, which has had 455 pulls. Exploring finds the arms; it takes long to tell two close ones apart.' },
      { def: 'Regret', text: 'The reward lost by not always pulling the best arm: after each pull, add the best arm’s mean minus the mean of the arm pulled.' },
      { h: 'Recap' },
      { list: ['Each arm’s estimate is the running mean of its rewards.', 'Greedy can lock onto a poor arm and never learn better.', 'ε-greedy keeps trying everything a fraction ε of the time.'] },
      X('re-x-estimates'), X('re-x-incremental'), X('re-x-eps-prob'), X('re-x-greedy-stuck'),
    ],
  });

  C.lesson({
    id: 're-regret', chapter: 're-bandits', title: 'Regret, UCB and Thompson sampling', minutes: 8,
    summary: 'ε-greedy explores blindly, at the same rate for ever. UCB1 adds to each estimate a bonus that is large for arms pulled rarely and shrinks as they are tried. Thompson sampling draws a plausible payout for each arm from its posterior and pulls the highest draw. Both explore where it is still worth it, and their regret grows more slowly.',
    blocks: [
      'ε-greedy keeps pulling arms it already knows are bad, a tenth of the time, for ever. Could an agent explore only where it is still unsure?',
      { math: 'UCB1:   pick argmax  Q(a) + c · √(2 ln t / N(a))\nThompson (0/1 rewards):   draw θ(a) ~ Beta(1 + wins, 1 + losses),  pick argmax θ(a)' },
      { topic: 'rl-bandit', frame: 5, input: { algo: 'ucb', eps: 0.1, c: 1, k: 5, steps: 500, seed: 3 }, caption: 'UCB1 first tries each arm once: an arm with no pulls has an unlimited bonus.' },
      { topic: 'rl-bandit', frame: 6, input: { algo: 'ucb', eps: 0.1, c: 1, k: 5, steps: 500, seed: 3 }, caption: 'Pull 6: arm 4’s estimate 1.00 plus a bonus of 1.89 gives 2.89, the highest upper bound.' },
      { predict: { q: 'Arm A has been pulled 4 times, arm B 40 times, and both estimates are 0.6. Which one gets the larger UCB bonus?', options: ['A', 'B', 'The same: their estimates are equal'], answer: 0, explain: 'The bonus has N(a) under the square root in the denominator: fewer pulls, more uncertainty, larger bonus.' } },
      { topic: 'rl-bandit', frame: 14, input: { algo: 'ucb', eps: 0.1, c: 1, k: 5, steps: 500, seed: 3 }, caption: 'UCB1 after 500 pulls: arm 4 has had the most, 222. It has found the best arm.' },
      { topic: 'rl-bandit', frame: 14, input: { algo: 'thompson', eps: 0.1, c: 1, k: 5, steps: 500, seed: 3 }, caption: 'Thompson sampling: arm 4 has had 294 pulls, and the regret is 22.3, against 30.7 for ε-greedy on the same bandit.' },
      { topic: 'rl-regret', frame: 6, caption: 'Averaged over 30 bandits of 10 arms: after 500 pulls the regret is 115.6 for greedy, 39.1 for ε-greedy, 78.9 for UCB1 and 30.0 for Thompson. Greedy’s curve stays a straight line.' },
      { tip: 'UCB1’s c sets how much to trust the bonus. With c = 1 on these 0-or-1 arms it keeps exploring for a long time; a smaller c explores less. Try it in [Regret: greedy, ε-greedy, UCB and Thompson](#topic=rl-regret).' },
      { h: 'Recap' },
      { list: ['Regret adds up what each pull lost against the best arm.', 'UCB1: estimate plus a bonus that shrinks with the pulls.', 'Thompson sampling: pull the arm whose plausible payout, drawn from its posterior, is highest.'] },
      X('re-x-regret'), X('re-x-ucb'), X('re-x-ucb-untried'), X('re-x-thompson'), X('re-x-regret-curve'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: planning with a known model
  // =========================================================================================================
  C.lesson({
    id: 're-mdp', chapter: 're-plan', title: 'A Markov decision process', minutes: 8,
    summary: 'An MDP has states, actions, a transition model P(s′ | s, a), rewards and a discount γ. The return adds the rewards, each later one multiplied by one more γ. A policy says what to do in each state, and the value V^π(s) is the expected return from s when following it.',
    blocks: [
      'A robot on a slippery floor wants to reach a charger and avoid two holes. Each move costs a little battery. To plan, what exactly does it need to know, and what is it trying to make large?',
      { topic: 'rl-mdp', frame: 0, caption: 'The states are the 14 open cells; the goal and the 2 pits end an episode. In every state there are 4 actions: north, east, south and west.' },
      { topic: 'rl-mdp', frame: 2, caption: 'P(s′ | s, east) for cell (2, 2): 0.80 to (2, 3), and 0.10 to each side. Only the current cell and action matter, not the path that led here: the Markov property.' },
      { topic: 'rl-mdp', frame: 3, caption: 'Rewards: −0.04 for every move, 1 at the goal, −1 in a pit. The discount γ = 0.90 makes a reward k steps away worth γᵏ of itself now.' },
      { math: 'G = r₀ + γ r₁ + γ² r₂ + …\nV^π(s) = the expected G from s, following π' },
      { predict: { q: 'The same long run of −0.04 rewards is discounted with γ = 0.5 and with γ = 0.99. Which return is closer to 0?', options: ['γ = 0.5', 'γ = 0.99', 'The same'], answer: 0, explain: 'With γ = 0.5 later rewards fade fast: −0.08 in all, against −1.32 with γ = 0.99.' } },
      { topic: 'rl-mdp', frame: 5, caption: 'A random episode: G = −0.394 with γ = 0.9. With γ = 0.5, 0.9 or 0.99 the same rewards would be worth −0.08, −0.394 and −1.32.' },
      { topic: 'rl-mdp', frame: 6, caption: 'For the policy “always go east”, V at the start is −0.758, averaged over every way the slips can go.' },
      { topic: 'rl-mdp', frame: 7, caption: 'The best policy does better: V* at the start is 0.161. Finding it, when the model is known, is planning.' },
      { h: 'Recap' },
      { list: ['An MDP: states, actions, P(s′ | s, a), rewards and γ.', 'G = Σ γᵏ r_k; a smaller γ cares less about the far future.', 'V^π(s) is the expected return from s under π; V* is the best possible.'] },
      X('re-x-return'), X('re-x-code-return'), X('re-x-markov'), X('re-x-gamma'),
    ],
  });

  C.lesson({
    id: 're-vi', chapter: 're-plan', title: 'Value iteration', minutes: 8,
    summary: 'The Bellman backup sets a state’s value to the best, over actions, of the expected reward plus γ times the value of where the action leads. Value iteration applies it to every state, sweep after sweep, starting from zero. Values spread out from the goal one step per sweep, and the arrows of the best actions form the optimal policy.',
    blocks: [
      'If you knew how good every neighbouring cell was, choosing a move would be easy: go to the best one. But how good a cell is depends on its neighbours. Where do we start?',
      { math: 'V(s) ← max over a of  Σ P(s′ | s, a) · [ r + γ V(s′) ]\nπ*(s) = the action that achieves the max' },
      { topic: 'rl-value-iteration', frame: 0, caption: 'Start with V = 0 in every cell. The goal and the pits keep 0: an episode ends there.' },
      { topic: 'rl-value-iteration', frame: 1, caption: 'Sweep 1: next to the goal, at cell (2, 4), moving south earns 1, so V becomes 1. The look-ahead values were north −1, east −0.04, south 1 and west −0.04.' },
      { predict: { q: 'Sweep 2, at cell (2, 3), next to (2, 4). With γ = 0.95 and a step cost of −0.04, what is the best look-ahead value?', options: ['1', '0.91', '−0.04'], answer: 1, explain: 'East to (2, 4): −0.04 + 0.95 × 1 = 0.91.' } },
      { topic: 'rl-value-iteration', frame: 2, caption: 'Sweep 2: at (2, 3), east gives −0.04 + 0.95 × 1 = 0.91. Each sweep the values reach one cell further from the goal.' },
      { topic: 'rl-value-iteration', frame: 9, caption: 'Converged after 9 sweeps. V at the start is 0.523, and following the arrows reaches the goal in 7 moves.' },
      { topic: 'rl-value-iteration', frame: 13, input: { world: 'pits', gamma: 0.95, slip: 0.2 }, caption: 'On slippery ground every move may slide sideways, so it takes 33 sweeps, and V at the start falls to 0.333: the same route is now riskier, near the pits.' },
      { h: 'Recap' },
      { list: ['A Bellman backup: the best expected r + γ V(next).', 'Value iteration sweeps every state until nothing changes.', 'The arrows of the best actions are the optimal policy.'] },
      X('re-x-bellman'), X('re-x-vi-sweep'), X('re-x-code-vi'),
    ],
  });

  C.lesson({
    id: 're-pi', chapter: 're-plan', title: 'Policy iteration', minutes: 6,
    summary: 'Policy iteration alternates two steps: evaluate the current policy exactly, then improve it by choosing in each state the action that is best under those values. It stops when no state wants to change, and then the policy is optimal: the same values value iteration finds.',
    blocks: [
      'Value iteration improves its guesses a little everywhere at once. Another way: fix a plan, work out exactly how good it is, then fix every place where a different move would do better. Does that end, and where?',
      { math: 'evaluate:  V^π(s) = Σ P(s′ | s, π(s)) [ r + γ V^π(s′) ]   (a fixed point, found by sweeps)\nimprove:   π(s) ← argmax over a of Σ P(s′ | s, a) [ r + γ V^π(s′) ]' },
      { topic: 'rl-policy-iteration', frame: 0, caption: 'Start with “always north” in every cell.' },
      { topic: 'rl-policy-iteration', frame: 1, caption: 'Round 1: evaluating “always north” takes 298 sweeps, and V at the start is −0.8.' },
      { topic: 'rl-policy-iteration', frame: 2, caption: 'Improvement: acting greedily on those values changes the arrow in 3 cells.' },
      { predict: { q: 'Policy iteration stops when an improvement step changes no arrow. Will its value at the start match value iteration’s 0.5232?', options: ['Yes', 'No, it is a little lower', 'Only on firm ground'], answer: 0, explain: 'A policy that no backup can improve is optimal: both methods find the same V*.' } },
      { topic: 'rl-policy-iteration', frame: 16, caption: 'Round 8: no cell wants a different action, so the policy is stable, and therefore optimal.' },
      { topic: 'rl-policy-iteration', frame: 17, caption: 'Policy iteration needed 8 rounds; its value at the start, 0.5232, matches value iteration’s.' },
      { h: 'Recap' },
      { list: ['Evaluate the policy, then improve it greedily; repeat.', 'It stops when no state changes its action: that policy is optimal.', 'Few rounds, but each evaluation can take many sweeps.'] },
      X('re-x-policy-eval'), X('re-x-pi-vs-vi'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: learning from experience
  // =========================================================================================================
  C.lesson({
    id: 're-mc', chapter: 're-learn', title: 'Learning from whole episodes', minutes: 7,
    summary: 'Without a model there is nothing to sweep. Monte Carlo control plays whole episodes, and afterwards walks back from the end, computing the return that followed each first visit to a state and action and averaging those returns into Q(s, a). Acting ε-greedily on Q, the agent improves as the averages settle.',
    blocks: [
      'Now the robot does not know the floor. It cannot compute P(s′ | s, a); it can only try, and see what happens. How can it learn the value of a move from the episodes it plays?',
      { topic: 'rl-mc', frame: 0, caption: 'Every Q starts at 0. The first episode, played ε-greedily with ε = 0.20, is a random walk: 132 moves, returning −6.24.' },
      { topic: 'rl-mc', frame: 1, caption: 'Walking back from the end: the return after the first visit to (1, 3) going east is G = −1, so Q moves from 0 to −1.' },
      { predict: { q: 'One step earlier the reward was −0.04 and γ = 0.95. The return after that step is G = −1 one step later. What is G there?', options: ['−1.04', '−0.99', '−0.95'], answer: 1, explain: 'G = r + γ G_next = −0.04 + 0.95 × (−1) = −0.99, as the topic shows for (1, 2) going east.' } },
      { topic: 'rl-mc', frame: 2, caption: 'The return after the first visit to (1, 2) going east: G = −0.04 + 0.95 × (−1) = −0.99.' },
      { topic: 'rl-mc', frame: 7, caption: 'After 100 episodes the greedy arrows from the start reach the goal in 7 moves.' },
      { topic: 'rl-mc', frame: 9, caption: 'After 300 episodes the last 20 average a return of 0.43, though ε = 0.2 still makes it wander now and then.' },
      { h: 'Recap' },
      { list: ['Monte Carlo waits for the end of an episode.', 'Returns are computed backwards: G = r + γ G_next.', 'Q(s, a) is the average of the returns that followed (s, a).'] },
      X('re-x-mc-return'),
    ],
  });

  C.lesson({
    id: 're-cliff', chapter: 're-learn', title: 'Q-learning and SARSA on the cliff', minutes: 8,
    summary: 'Temporal-difference methods update after every step, towards r + γ times a value of the next state. Q-learning uses the best next action, max Q(s′, ·): it learns the values of the greedy policy, whatever it actually does. SARSA uses the next action it really takes: it learns the values of its own exploring policy, and so learns to keep away from danger.',
    blocks: [
      'A path runs along the edge of a cliff. The shortest route hugs the edge; one random step and you fall. Should the agent learn the shortest route, or a safer one?',
      { topic: 'rl-cliff', frame: 0, caption: 'Start at the bottom left, the goal at the bottom right, the cliff between. Every move costs −1; a fall costs −100 and sends the agent back. Both agents explore with ε = 0.10 and learn with α = 0.20.' },
      { math: 'Q(s, a) ← Q(s, a) + α [ target − Q(s, a) ]\nQ-learning:  target = r + γ · max over a′ of Q(s′, a′)\nSARSA:       target = r + γ · Q(s′, a′),  a′ the action it will take next' },
      { topic: 'rl-cliff', frame: 1, caption: 'The first updates: from (3, 0) going north, reward −1, target −1, so Q moves from 0 to −0.2 for both.' },
      { predict: { q: 'Which agent will learn the route right along the edge?', options: ['Q-learning', 'SARSA', 'Both'], answer: 0, explain: 'Q-learning backs up the best next action, as if it never stepped at random: the edge is shortest. SARSA backs up its real, exploring next action, which sometimes falls.' } },
      { topic: 'rl-cliff', frame: 7, caption: 'After 300 episodes: Q-learning’s greedy route takes 13 moves, SARSA’s 17.' },
      { topic: 'rl-cliff', frame: 9, caption: 'Q-learning learns the shortest route, along the edge; SARSA keeps a safer distance. While still exploring, SARSA’s episodes average −29.6 against Q-learning’s −51.3.' },
      { topic: 'rl-cliff', frame: 9, input: { eps: 0.02, alpha: 0.2, episodes: 500, seed: 1 }, caption: 'With hardly any exploration, ε = 0.02, falls are rare: SARSA’s route shortens to 15 moves, and the two averages are close.' },
      { h: 'Recap' },
      { list: ['Both update after every step, towards r + γ × a value of s′.', 'Q-learning uses max Q(s′, ·): off-policy, it learns the greedy policy.', 'SARSA uses Q(s′, a′) for the action it takes: on-policy, it learns what its exploring self can do safely.'] },
      X('re-x-q-update'), X('re-x-targets'), X('re-x-cliff-paths'), X('re-x-code-q'),
    ],
  });

  C.lesson({
    id: 're-explore', chapter: 're-learn', title: 'How much to explore', minutes: 6,
    summary: 'An agent learns only about what it tries. Some exploration is needed to find a good route; but a fixed ε keeps paying for random moves long after the route is known. Decaying ε, high at first and shrinking every episode, explores early and exploits later.',
    blocks: [
      'Exploring costs reward now and may pay later. When should an agent stop?',
      { topic: 'rl-explore', frame: 5, caption: 'Q-learning crossing two rooms with ε fixed at 0.3. The greedy path takes 14 moves, but the last 20 episodes average only 0.118: a third of the moves are still random.' },
      { predict: { q: 'Now ε starts at 1 and is multiplied by 0.98 after every episode. How large is ε after 50 episodes?', options: ['about 0.36', 'about 0.02', '0.5'], answer: 0, explain: '0.98⁵⁰ = 0.364: halfway to small. After 200 episodes it is 0.02.' } },
      { topic: 'rl-explore', frame: 2, input: { eps: 1, decay: '0.98', episodes: 300, seed: 1 }, caption: 'Decaying from ε = 1: after 50 episodes ε = 0.36, every cell has been visited, and the greedy path already reaches the goal.' },
      { topic: 'rl-explore', frame: 5, input: { eps: 1, decay: '0.98', episodes: 300, seed: 1 }, caption: 'After 300 episodes ε is down to 0.01 and the last 20 average 0.466, against 0.118 with ε fixed at 0.3.' },
      { topic: 'rl-explore', frame: 5, input: { eps: 0, decay: 'none', episodes: 300, seed: 1 }, caption: 'With ε = 0 this agent still visits every cell, because its estimates all start at 0 and it breaks ties at random; then it exploits at once, averaging 0.48. On a world with a tempting short-cut, such luck runs out.' },
      { warn: 'Never judge an agent only by its returns while it is still exploring. Watch the greedy policy too: on the cliff, Q-learning’s greedy route was best even while its exploring episodes fared worse.' },
      { h: 'Recap' },
      { list: ['No exploration risks never finding the good route.', 'Fixed exploration keeps paying for random moves.', 'Decaying ε: explore early, exploit later.'] },
      X('re-x-eps-decay'), X('re-x-explore-choice'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: policies and networks
  // =========================================================================================================
  C.lesson({
    id: 're-reinforce', chapter: 're-deep', title: 'Policy gradients: REINFORCE', minutes: 8,
    summary: 'Instead of learning values, a policy network outputs the probability of each action. REINFORCE plays episodes, then nudges up the log-probability of every action taken, in proportion to how much better than expected the return after it was (the advantage). Actions followed by good returns become likelier; others less likely.',
    blocks: [
      'A pole balanced on a cart: the state is four real numbers, so there is no table to fill. Could a network learn to push left or right directly, from nothing but how long the pole stayed up?',
      { topic: 'rl-reinforce', frame: 0, caption: 'Each step the policy pushes the cart with 10 N, left or right. An episode ends when the pole leans 12°, the cart leaves 2.4 m, or after 500 steps.' },
      { math: 'θ ← θ + lr · Σ over steps of  A_t · ∇θ log π(a_t | s_t)\nA_t = G_t − b(s_t)   (the return after step t, minus a baseline)' },
      { topic: 'rl-reinforce', frame: 1, caption: 'The first update plays 4 episodes with the untrained policy. The first lasted 46 steps, so with γ = 0.99 its return from the start is 37.02. The baseline network predicted −0.716; after the advantages are centred and scaled, that first action’s advantage is 1.18, positive, so its probability is pushed up.' },
      { predict: { q: 'An action is followed by a return lower than the baseline predicted. What does REINFORCE do to its probability?', options: ['Pushes it up', 'Pushes it down', 'Leaves it: only the best action changes'], answer: 1, explain: 'A negative advantage times ∇ log π lowers the log-probability of that action.' } },
      { topic: 'rl-reinforce', frame: 2, caption: 'Before any update the greedy policy keeps the pole up for 10 steps.' },
      { topic: 'rl-reinforce', frame: 7, caption: 'After 10 updates (40 episodes) it holds 279 steps.' },
      { topic: 'rl-reinforce', frame: 11, caption: 'After 25 updates it balances the pole for all 500 steps, leaning at most 0.4°.' },
      { tip: 'The baseline does not change the direction of the gradient on average, but it makes each update far less noisy: actions are judged against what was expected from that state, not against zero.' },
      { h: 'Recap' },
      { list: ['A policy network gives the probability of each action.', 'REINFORCE raises log π(a | s) in proportion to the advantage A = G − b(s).', 'A baseline cuts the noise of the update.'] },
      X('re-x-pg-sign'), X('re-x-pg-step'), X('re-x-baseline'),
    ],
  });

  C.lesson({
    id: 're-dqn', chapter: 're-deep', title: 'Deep Q-learning', minutes: 8,
    summary: 'A deep Q-network replaces the table with a network that outputs a value for each action. Two devices keep it stable: a replay buffer, from which past transitions are drawn at random so that updates are not dominated by the last few correlated steps, and a target network, a frozen copy that supplies the targets y = r + γ max Q_target(s′, ·) and is refreshed only now and then.',
    blocks: [
      'Q-learning with a table works when the states can be counted. With a network in place of the table, every update also moves the values of other states, including the very target it is chasing. How do we keep that from spiralling?',
      { topic: 'rl-dqn', frame: 0, caption: 'The untrained network gives every state 4 small, arbitrary action values. ε falls from 1 to 0.05 over the first 1200 steps, and every transition goes into the replay buffer.' },
      { topic: 'rl-dqn', frame: 4, caption: 'Step 1000: the buffer holds 1000 transitions and the target network has been refreshed 10 times. The greedy arrows do not reach the goal yet.' },
      { predict: { q: 'For a stored transition with reward −0.04, γ = 0.95 and max Q_target(s′) = 0.421, what target y does the online network learn towards?', options: ['0.421', '0.36', '−0.04'], answer: 1, explain: 'y = −0.04 + 0.95 × 0.421 = 0.36.' } },
      { topic: 'rl-dqn', frame: 5, caption: 'A minibatch replayed at step 1000: 6 stored transitions drawn at random. For the first, y = −0.04 + 0.95 × 0.421 = 0.36, while the online network says 0.293: the loss pulls it towards y.' },
      { topic: 'rl-dqn', frame: 7, caption: 'Step 2000: the greedy arrows lead from the start to the goal in 7 moves.' },
      { topic: 'rl-dqn', frame: 8, caption: 'After 2000 steps and 103 episodes, the last 10 returned 0.74 on average.' },
      { table: [['Device', 'The problem it solves'], ['replay buffer', 'consecutive steps are strongly correlated; drawing old transitions at random makes a batch look more like independent data, and reuses each experience many times'], ['target network', 'with one network the target moves every time the network does; a frozen copy holds the targets still between refreshes']], caption: 'Two devices that keep deep Q-learning from chasing its own tail.' },
      { h: 'Recap' },
      { list: ['A network outputs Q(s, ·) for every action.', 'Targets y = r + γ max Q_target(s′, ·), or just r when the episode ends.', 'Replay breaks correlations; a target network holds the targets still.'] },
      X('re-x-dqn-target'), X('re-x-dqn-terminal'), X('re-x-dqn-why'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
