// Q-learning in a small grid world: each open cell holds four values, one per move, drawn as triangles (blue good,
// vermilion bad) with their numbers; the arrow is the best move. The values fill in backwards from the goal as
// episodes go by, and the return of each episode climbs.
(function (S) {
  'use strict';
  const D = S.Demo, C = () => S.Kit.curve, GW = () => S.Kit.grid, box = D.box;
  D.add({
    id: 'qlearn-grid', order: 120, tag: 'Reinforcement learning', short: 'Q-learning grid', title: 'Q-learning in a grid world: values and greedy arrows filling in',
    blurb: 'Reaching the goal pays +1, the pit −1, and every move costs 0.04. One move in ten slips sideways. After each move, Q(cell, move) moves halfway toward the reward plus 0.9 times the best value of the next cell.',
    frames({ narrow }) {
      const T = S.Toy, w = T.world, run = D.once('qlearn', () => T.qlearn(w, { keep: [1, 3, 10, 30, 100, 300] }));
      const R = w.map.length, Cn = w.map[0].length;
      const L = narrow
        ? { view: box(-4, -14, 396, 720), g: { x: 14, y: 30, cell: 90 }, rc: { x: 50, y: 470, w: 310, h: 130 } }
        : { view: box(-10, -14, 1180, 492), g: { x: 40, y: 30, cell: 104 }, rc: { x: 580, y: 60, w: 540, h: 200 } };
      // the return of each episode, averaged over the ten before it
      const avg = run.rets.map((_, i) => { const a = run.rets.slice(Math.max(0, i - 9), i + 1); return [i + 1, a.reduce((x, y) => x + y, 0) / a.length]; });
      const rax = C().axes(Object.assign({ prefix: 'qr', xr: [0, 300], yr: [-2, 1], xticks: [0, 100, 200, 300], yticks: [-2, -1, 0, 1], xname: 'episode', yname: 'return per episode (average of the last ten)' }, L.rc));
      const kinds = (r, c) => ({ S: 'start', G: 'goal', X: 'pit', '#': 'wall' }[w.map[r][c]] || '');
      return run.snaps.map((sn, k) => {
        const items = [];
        const qmax = 1;
        items.push(...GW().world({ prefix: 'qw', x: L.g.x, y: L.g.y, cell: L.g.cell, rows: R, cols: Cn, kinds, q: (r, c) => sn.Q[r][c], qmax, values: !narrow, valueSize: 12, best: (r, c) => (sn.Q[r][c].some((v) => v !== 0) ? run.best(sn.Q[r][c]) : -1), names: { goal: '+1', pit: '−1', start: 'start' } }));
        items.push(...rax.items, C().line(rax, 'qr:l', avg.slice(0, sn.episode), 'act'), C().dot(rax, 'qr:d', sn.episode, avg[sn.episode - 1][1], ''));
        const ty = narrow ? L.rc.y + L.rc.h + 66 : L.rc.y + L.rc.h + 80;
        items.push({ key: 'qr:t1', kind: 'text', x: L.rc.x, y: ty, text: `episode ${sn.episode}: ${sn.steps} moves, return ${D.fix(sn.ret)}`, size: 14, cls: 'strong' });
        items.push({ key: 'qr:t2', kind: 'text', x: L.rc.x, y: ty + 24, text: `exploring: a random move with chance ε = ${sn.eps.toFixed(2)}`, size: 13, cls: 'axname' });
        if (!narrow) items.push({ key: 'qr:t3', kind: 'text', x: L.rc.x, y: ty + 48, text: 'triangles: the value of each move (blue good, vermilion bad); arrow: the best move', size: 13, cls: 'axname' });
        const lastCell = sn.path[sn.path.length - 1], end = w.map[lastCell[0]][lastCell[1]];
        const s0 = sn.Q[run.start[0]][run.start[1]], b0 = run.best(s0);
        const say = k === 0
          ? `Episode 1: the agent wanders for \`${sn.steps}\` moves until it ${end === 'G' ? 'finds the goal' : end === 'X' ? 'falls in the pit' : 'gives up'}. Every move it made is nudged down by the \`0.04\` cost, and the last by the ${end === 'G' ? '\`+1\`' : '\`−1\`'}: the values are still almost all \`0\`.`
          : `After ${sn.episode} episodes the values have spread back from the goal and the pit. At the start the best move is ${S.Kit.grid.NAMES[b0]}, worth \`${D.fix(s0[b0])}\`` + (k === run.snaps.length - 1 ? `. The arrows now lead from the start round the wall to the goal; this last episode took ${sn.steps} moves.` : '.');
        return { view: L.view, items, say };
      });
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
