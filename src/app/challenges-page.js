// The Challenges pages: #challenges lists the model challenges with the reader's best score and medal; #challenge/id
// sets one: its story, the data, the goal and the limits, a canvas to build the workflow on (only the blocks the task
// allows), and Submit, which scores the trained workflow on the hidden rows and keeps the attempt in this browser.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const C = () => S.Challenges;
  const LEVEL = ['', 'easy', 'medium', 'hard'];
  const MEDAL = { gold: 'Gold', silver: 'Silver', bronze: 'Target reached' };
  const inline = (t) => S.UI.inline(t);
  const medalEl = (m) => (m ? h('span.medal.' + m, { title: MEDAL[m] }, icon('trophy'), MEDAL[m]) : null);

  function listPage(main, notice) {
    document.title = 'Challenges · Nooral';
    const box = h('div.home.challenges', { id: 'challengesView' });
    if (notice) box.appendChild(h('div.notice', icon('info'), h('span', notice)));
    box.append(h('h1.page-title', 'Challenges'), h('p.lead', 'Each challenge is a dataset with a story, a target score and limits on what you may use. Build a workflow, train it, and submit: it is scored on rows it has never seen. Your attempts and best scores stay in this browser.'));
    const list = S.Content ? S.Content.ladder() : [];
    if (!list.length) { box.appendChild(S.UI.empty('trophy', 'No challenges yet', 'They are being written. Meanwhile the workflow canvas trains any model with its insides drawn.', [{ label: 'Open the canvas', icon: 'layout', primary: true, action: () => { location.hash = 'canvas'; } }])); main.appendChild(box); return; }
    const grid = h('div.ch-grid');
    for (const x of list) {
      const best = C().best(x.id), solved = C().solved(x.id), medal = best ? C().medal(x, best.score) : null;
      const card = h('a.ch-card' + (solved ? '.solved' : ''), { href: '#challenge/' + x.id, dataset: { challenge: x.id } },
        h('div.ch-top', h('span.ch-n', String(x.n)), h('span.ch-level', { title: LEVEL[x.level || 1] }, '●'.repeat(x.level || 1) + ' ' + LEVEL[x.level || 1]), solved ? h('span.ch-ok', { title: 'Target reached' }, icon('check')) : null),
        h('h3', x.title),
        h('p.ch-story', String(x.story).split(/(?<=\.)\s/)[0]),
        h('div.ch-meta', h('span.ch-goal', icon('target'), C().goal(x)), best ? h('span.ch-best', 'best ' + C().fmt(best.score)) : h('span.muted', 'not tried yet'), medalEl(medal)));
      grid.appendChild(card);
    }
    box.appendChild(grid);
    main.appendChild(box);
  }

  function challengePage(main, x) {
    document.title = x.title + ' · Challenges · Nooral';
    const page = { editors: [], dispose() { for (const e of page.editors) if (e.dispose) e.dispose(); } };
    const box = h('article.challenge', { id: 'challengeView', dataset: { challenge: x.id } });
    box.append(h('div.crumbs', h('a', { href: '#challenges' }, 'Challenges'), ' › ' + x.n + '. ' + x.title),
      h('h1.page-title', x.title),
      h('div.ls-meta', h('span.level', '●'.repeat(x.level || 1) + ' ' + LEVEL[x.level || 1]), C().solved(x.id) ? h('span.solved', icon('check'), 'Target reached') : null),
      h('div.ch-story-full', ...String(x.story).split(/\n\s*\n/).map((p) => h('p', inline(p)))),
      S.ExerciseViews.taskCard(x));
    // the canvas: the reader's saved attempt, or the starter
    const host = h('div.ch-canvas', { id: 'chCanvas' });
    let ed = null, doc = null;
    try { doc = C().draft(x.id) ? S.WF.fromJSON(C().draft(x.id)) : C().starter(x); } catch (e) { doc = C().starter(x); }
    const allowed = x.limits && x.limits.blocks ? ['challenge-data'].concat(x.limits.blocks) : null;
    const save = S.UI.debounce(() => { if (ed && (!S.Shell || S.Shell.autosave())) C().saveDraft(x.id, ed.doc()); }, 500);
    const submit = h('button.btn.primary', { type: 'button', id: 'btnSubmit' }, icon('trophy'), 'Submit for scoring');
    const reset = h('button.btn', { type: 'button', id: 'btnReset' }, icon('restart'), 'Start again');
    const hintBtn = h('button.btn', { type: 'button', id: 'btnHint' }, icon('hint'), 'A hint');
    const result = h('div.ch-result', { id: 'chResult', 'aria-live': 'polite' });
    const hints = h('div.hints', { id: 'hints' });
    box.append(host, h('div.ex-tools.ch-tools', submit, reset, hintBtn), result, hints, history());
    main.appendChild(box);
    const open = (d) => {
      host.innerHTML = '';
      if (ed && ed.dispose) ed.dispose();
      page.editors = [];
      if (!S.WF.embed) { host.appendChild(S.UI.empty('layout', 'The canvas is on its way', 'This challenge is built on the workflow canvas, which is not in this build yet.')); ed = null; return; }
      ed = S.WF.embed(host, d, { allowed, height: 520, onChange: save });
      page.editors.push(ed);
    };
    open(doc);
    reset.addEventListener('click', async () => {
      if (!(await S.UI.confirm('Start again?', 'Your workflow for this challenge is replaced by the starting one. Your scores are kept.', 'Start again', 'Keep mine'))) return;
      C().saveDraft(x.id, null);
      open(C().starter(x));
    });
    let shown = 0;
    hintBtn.addEventListener('click', () => {
      const hs = x.hints || [];
      if (shown >= hs.length) return;
      hints.appendChild(h('div.ls-note.tip.hint', icon('hint'), h('div', h('b', 'Hint ' + (shown + 1) + '. '), inline(hs[shown]))));
      shown++;
      if (shown >= hs.length) { hintBtn.disabled = true; hintBtn.lastChild.textContent = 'No more hints'; }
    });
    submit.addEventListener('click', async () => {
      if (submit.disabled) return;
      submit.disabled = true;
      result.className = 'ch-result busy';
      result.innerHTML = '';
      result.appendChild(h('div.ex-v', icon('history'), h('b', 'Training and scoring… '), h('span', 'the workflow runs to the end, then meets the hidden rows.')));
      let r;
      try { r = await score(x, ed ? ed.doc() : doc, ed); } catch (e) { r = { ok: false, score: null, problems: [String(e.message || e)], limits: [] }; }
      finally { submit.disabled = false; }
      const rank = C().record(x.id, r);
      showResult(x, r, rank, result);
      // the result can bring a scroll bar that narrows the canvas: fit the workflow to it again
      requestAnimationFrame(() => { if (ed && ed.editor && ed.editor.fit) ed.editor.fit(); });
      const hist = document.getElementById('chHistory');
      if (hist) hist.replaceWith(history());
    });
    function history() {
      const at = C().attempts(x.id);
      if (!at.length) return h('div', { id: 'chHistory' });
      const rows = at.slice(-8).reverse().map((a) => h('tr' + (a.reached ? '.ok' : ''), h('td', new Date(a.t).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })), h('td.mono', a.limits ? 'broke the limits' : C().fmt(a.score)), h('td', medalEl(a.medal) || (a.reached ? 'reached' : '—'))));
      return h('details.ch-history', { id: 'chHistory' }, h('summary', 'Your attempts (' + at.length + ')'), h('div.fit-table', h('table.ch-hist', h('thead', h('tr', h('th', 'When'), h('th', C().metricLabel(x)), h('th', ''))), h('tbody', rows))));
    }
    page.submit = () => submit.click();
    page.load = (d) => { const doc2 = S.WF.fromJSON(d); if (ed && ed.loadDoc) ed.loadDoc(doc2); else open(doc2); };
    page.editor = () => ed;
    return page;
  }

  /** score through the canvas's own worker when it has one (the workflow is trained already), else a fresh run */
  async function score(x, doc, ed) {
    const limits = C().limitsProblems(S.WF.fromJSON(doc), x);
    const remote = ed && ed.remote;
    if (remote && remote.fn) {
      await remote.idle();
      return C().result(x, await remote.fn('scoreHidden', { data: x.data, hidden: x.hidden, metric: x.metric, constraints: x.constraints }), limits);
    }
    return C().evaluateAsync(doc, x);
  }

  function showResult(x, r, rank, box) {
    box.innerHTML = '';
    box.className = 'ch-result ' + (r.reached ? 'ok' : 'bad');
    const head = r.limits && r.limits.length ? 'The workflow breaks the limits.' : r.score === null || r.score === undefined ? 'The workflow could not be scored.' : r.reached ? 'Target reached.' : 'Not there yet.';
    box.appendChild(h('div.ex-v', icon(r.reached ? 'check' : 'x'), h('b', head)));
    if (r.score !== null && r.score !== undefined && !(r.limits && r.limits.length)) {
      box.appendChild(h('div.ch-score', h('span.ch-big.mono', C().fmt(r.score)), h('span', C().metricLabel(x) + ' on ' + r.n + ' hidden rows · the target is ' + C().goal(x)), medalEl(r.medal)));
      if (rank && rank.rank) box.appendChild(h('p.muted', rank.rank === 1 ? (rank.of > 1 ? 'Your best of ' + rank.of + ' attempts.' : 'Your first scored attempt.') : 'Number ' + rank.rank + ' of your ' + rank.of + ' attempts.'));
      const m = x.marks || {};
      if (r.reached && r.medal !== 'gold' && m.gold !== undefined) box.appendChild(h('p.muted', 'Silver at ' + (m.silver !== undefined ? m.silver : '—') + ', gold at ' + m.gold + '.'));
    }
    for (const c of r.constraints || []) {
      const rates = (c.rates || []).map((q) => q.group + ' ' + C().fmt(c.kind === 'odds' ? q.tpr : q.rate)).join(', ');
      box.appendChild(h('div.ch-constraint' + (c.ok ? '.ok' : '.bad'), { dataset: { kind: c.kind } }, icon(c.ok ? 'check' : 'x'), h('span', h('b', c.value === null ? 'not measured' : C().fmt(c.value)), ' ' + c.label + (c.ok ? ': kept.' : ': broken, so the target is not reached.') + (rates ? ' (' + (c.kind === 'odds' ? 'true positive rates' : 'positive rates') + ': ' + rates + ')' : ''))));
    }
    for (const p of r.problems || []) box.appendChild(h('div.ex-d', p.replace(/^./, (c) => c.toUpperCase()) + (/[.!?]$/.test(p) ? '' : '.')));
  }

  S.Pages.register('challenges', {
    nav: 'challenges', title: 'Challenges',
    open(app, main, rest) {
      const m = /^challenges?[=/](.+)$/.exec(rest || '');
      if (m) {
        const x = S.Content && S.Content.challenges.get(m[1]);
        if (x) return challengePage(main, x);
        listPage(main, 'There is no challenge called "' + m[1] + '". Pick one from the list.');
        return { dispose() {} };
      }
      listPage(main);
      return { dispose() {} };
    },
  });

  // ---- what the diagram checker looks at ------------------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.group === 'challenges')) {
    S.Checks.push({ group: 'challenges', name: 'challenges list', hash: 'challenges', ready: "document.querySelector('#challengesView')", sel: '#main' });
    S.Checks.push({ group: 'challenges', name: 'a challenge', hash: 'challenge/nn-ch-pumps', ready: "document.querySelector('#challengeView')", sel: '#main' });
    S.Checks.push({ group: 'challenges', name: 'a scored attempt', hash: 'challenge/nn-ch-pumps', fresh: true, setup: "const x = Nooral.Content.challenges.get('nn-ch-pumps'); Nooral.app.page.load(x.solution); Nooral.app.page.submit();", ready: "document.querySelector('#chResult.ok, #chResult.bad')", sel: '#chResult' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
