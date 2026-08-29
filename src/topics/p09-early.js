// Part 9, early stopping: watch the loss on held-out points as training goes, remember the best weights so far, and
// stop once it has not improved for a number of checks in a row (the patience); then go back to the best weights.
// Run on the overfitting experiment, check by check, with the counter shown.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;

  /** the rule run over a list of validation losses: {best, stop, events: [{i, kind: 'best' | 'wait' | 'stop', wait}]} */
  P.earlyStop = function (losses, patience) {
    let best = 0, wait = 0, stop = null;
    const events = [{ i: 0, kind: 'best', wait: 0 }];
    for (let i = 1; i < losses.length; i++) {
      if (losses[i] < losses[best]) { best = i; wait = 0; events.push({ i, kind: 'best', wait }); }
      else { wait++; events.push({ i, kind: wait >= patience ? 'stop' : 'wait', wait }); if (wait >= patience) { stop = i; break; } }
    }
    return { best, stop, events };
  };

  S.topic({
    id: 'p09-early-stop', family: 'p09-reg', order: 40, title: 'Early stopping: quit while you are ahead',
    blurb: 'The simplest cure for overfitting needs no change to the network: check the loss on held-out points every few steps, keep a copy of the best weights so far, and stop when it has not improved for a while. The waiting time is the patience: too short and noise stops training early, too long and time is wasted, but the best copy is kept either way.',
    inputs: [{ id: 'patience', kind: 'number', label: 'Patience (checks without improvement)', def: 5, min: 1, max: 30 }, { id: 'size', kind: 'choice', label: 'Network', def: 'big', options: [{ value: 'big', label: 'big: 32 and 32' }, { value: 'medium', label: 'medium: 8 and 8' }] }],
    presets: [
      { label: 'Patience 5', values: { patience: 5, size: 'big' } },
      { label: 'Impatient: 1 check', values: { patience: 1, size: 'big' } },
      { label: 'Patient: 20 checks', values: { patience: 20, size: 'big' } },
      { label: 'A medium network', values: { patience: 5, size: 'medium' } },
    ],
    counters: [{ id: 'check', label: 'checks' }, { id: 'wait', label: 'checks since the best' }],
    run(inp) {
      const F = H().frames(['check', 'wait']), q = F.q, pat = inp.patience, r = P.regRun({ hidden: inp.size === 'big' ? [32, 32] : [8, 8], best: true });
      const ev = r.evals, L = ev.map((e) => e.valLoss), rule = P.earlyStop(L, pat), d = r.data, every = P.REG.evalEvery;
      const cb = { x: 400, y: 50, w: 460, h: 240 }, pb = { x: 40, y: 50, w: 300, h: 200 };
      const lg = (v) => Math.log10(Math.max(1e-4, v));
      const ax = P.ax(Object.assign({ prefix: 'es', xr: [0, P.REG.steps], yr: [-1, 0.5], xticks: [0, 100, 200, 300, 400, 500], yticks: [-1, -0.5, 0, 0.5], fmtY: (v) => H().sg(Math.pow(10, v), 2), xname: 'step', yname: 'loss on held-out points' }, cb));
      const plot = (upto, bestI, stopI) => {
        const use = ev.slice(0, upto + 1), items = [...ax.items, C().line(ax, 'es:v', use.map((e) => [e.step, lg(e.valLoss)]), 'p09val'), ...use.map((e, i) => C().dot(ax, 'es:d' + i, e.step, lg(e.valLoss), i === bestI ? 'p09train' : 'p09dim', i === bestI ? 6 : 2.5))];
        if (stopI !== null && stopI !== undefined) items.push(C().vline(ax, 'es:stop', ev[stopI].step, 'act'));
        items.push(C().vline(ax, 'es:best', ev[bestI].step, ''));
        return items;
      };
      const read = (rows) => H().lines('es:r', 40, 320, rows, 14, 24);
      const fieldAt = (i) => { const s = r.snaps.slice().reverse().find((x) => x.step <= ev[i].step); return s.field; };
      F.push([...P.regPanel('es:p', pb, r.snaps[0].field, d, { title: 'before training' }), ...plot(0, 0, null), ...read([['patience: ' + pat + ' checks', 'math'], ['a check every ' + every + ' steps', 'math']])],
        `Every ${q.n(every)} steps the loss on the ${q.n(d.val.X.length)} held-out points is checked. Rule: keep a copy of the weights whenever that loss is the lowest so far, and stop after ${P.qpl(q, pat, 'check')} in a row without a new lowest. It starts at ${q(L[0], 3)}.`, { phase: 'start', pat, every, n: d.val.X.length, L0: L[0] }, { hl: { eq: [0] } });
      let bestI = 0;
      // the checks, shown as runs: each new best, and the waiting after the last one
      const keyEvents = rule.events.filter((e, k) => k > 0 && (e.kind === 'best' || e.kind === 'stop' || (e.kind === 'wait' && (e.wait === 1 || e.wait === pat - 1))));
      for (const e of keyEvents) {
        F.set('check', e.i).set('wait', e.wait);
        if (e.kind === 'best') bestI = e.i;
        const s = ev[e.i];
        const items = [...P.regPanel('es:p', pb, e.kind === 'best' ? fieldAt(e.i) : fieldAt(bestI), d, { title: 'best weights so far: step ' + ev[bestI].step }), ...plot(e.i, bestI, e.kind === 'stop' ? e.i : null), ...read([['check ' + e.i + ' (step ' + s.step + '): ' + H().fx(s.valLoss, 3), 'math'], ['best: ' + H().fx(ev[bestI].valLoss, 3) + ' at step ' + ev[bestI].step, 'math'], ['checks without improvement: ' + e.wait, 'math']])];
        const say = e.kind === 'best' ? `Check ${q.n(e.i)}, step ${q.n(s.step)}: held-out loss ${q(s.valLoss, 3)}, a new lowest. Copy the weights and reset the counter.`
          : e.kind === 'stop' ? `Check ${q.n(e.i)}, step ${q.n(s.step)}: ${P.qpl(q, e.wait, 'check')} in a row without beating ${q(ev[bestI].valLoss, 3)}. The patience is used up: stop here.`
            : `Check ${q.n(e.i)}, step ${q.n(s.step)}: ${q(s.valLoss, 3)} is no better than the best, ${q(ev[bestI].valLoss, 3)}. The counter stands at ${q.n(e.wait)} of ${q.n(pat)}.`;
        F.push(items, say, { phase: e.kind, i: e.i, step: s.step, valLoss: s.valLoss, best: ev[bestI].valLoss, bestStep: ev[bestI].step, wait: e.wait, pat }, { hl: { eq: [e.kind === 'best' ? 1 : 2] } });
      }
      const last = ev[ev.length - 1], stopI = rule.stop;
      const bf = r.bests.filter((b) => b.step <= ev[stopI === null ? ev.length - 1 : stopI].step).pop();
      F.set('wait', 0);
      const rest = stopI === null ? [] : [C().line(ax, 'es:rest', ev.slice(stopI).map((e) => [e.step, lg(e.valLoss)]), 'p09alt thin'), H().text('es:restl', ax.X(P.REG.steps) - 4, ax.Y(lg(last.valLoss)) - 10, 'had it kept going', 'legend', 13, 'end')];
      F.push([...P.regPanel('es:p', pb, bf.field, d, { title: 'the weights kept: step ' + bf.step }), ...plot(stopI === null ? ev.length - 1 : stopI, rule.best, stopI), ...rest, ...read([['kept: step ' + bf.step + ', loss ' + H().fx(bf.valLoss, 3), 'math'], ['full run: step ' + last.step + ', loss ' + H().fx(last.valLoss, 3), 'math']])],
        (stopI === null ? `The patience never ran out within ${q.n(last.step)} steps. ` : `Training stopped at step ${q.n(ev[stopI].step)} instead of ${q.n(last.step)}, saving ${q.pct(1 - ev[stopI].step / last.step)} of the work. `) + `The weights kept are those of step ${q.n(bf.step)}: held-out loss ${q(bf.valLoss, 3)} and ${q.pct(bf.valAcc)} right, against ${q(last.valLoss, 3)} and ${q.pct(last.valAcc)} for the weights at the end of the full run.`,
        { phase: 'end', bestStep: bf.step, bestLoss: bf.valLoss, bestAcc: bf.valAcc, stopStep: stopI === null ? null : ev[stopI].step, lastStep: last.step, lastLoss: last.valLoss, lastAcc: last.valAcc, saved: stopI === null ? 0 : 1 - ev[stopI].step / last.step }, { hl: { eq: [3] } });
      P.narrow(F.list, (m) => (m.phase === 'end' ? { x: 10, y: 20, w: 400, h: 470 } : { x: 380, y: 20, w: 500, h: 330 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Early stopping', text: ['every k steps: measure the held-out loss', 'if it is the lowest so far: save the weights, wait = 0', 'else: wait = wait + 1; stop when wait = patience', 'at the end: restore the saved weights'].join('\n') }] }, result: { losses: L } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'start', once: true, pick: 'option',
      ask: 'Will training stop before the last step?',
      options: ['yes', 'no, it runs to the end'],
      answer: (fr) => (fr.find((f) => f.meta.phase === 'end').meta.stopStep === null ? 'no, it runs to the end' : 'yes'),
      explain: (fr) => fr.find((f) => f.meta.phase === 'end').say,
    }],
    quiz: [
      { q: 'With early stopping, which weights does the final model use?', options: ['the copy saved at the lowest held-out loss', 'the weights at the moment training stopped', 'the initial weights', 'an average of all of them'], answer: 0, why: 'Training goes on a little past the best point, so the saved copy is restored.' },
      { q: 'A patience of 1 is risky because…', options: ['one noisy check can stop training long before the best point', 'training never stops', 'it uses more memory', 'it changes the learning rate'], answer: 0, why: 'Held-out loss wobbles; one bad check is not a trend.' },
      { q: 'Early stopping acts like a regulariser because…', options: ['it limits how far the weights can travel from their small starting values', 'it removes neurons', 'it adds noise to the data', 'it changes the loss function'], answer: 0, why: 'Fewer steps mean weights that stay closer to where they began.' },
      { q: 'Can the held-out set used for early stopping also report the final score?', options: ['not honestly: it helped choose the model, so a separate test set is needed', 'yes, always', 'only with patience 1', 'only if it is large'], answer: 0, why: 'It was used for a decision, so its score is optimistic.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
