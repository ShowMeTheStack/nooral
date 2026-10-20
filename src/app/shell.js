// The frame around Nooral's pages, as in the sister tools: the sections of the top bar (Topics, Learn, Courses, the
// workflow Canvas, the Playground, the Text lab, Challenges), the home page, the settings dialog, About, share links
// (#share=code opens a shared workflow, network or text-lab setup) and autosave. S.Shell holds it all; the app
// (app.js) calls it.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const NAV = [['topics', 'Topics', 'library'], ['learn', 'Learn', 'learn'], ['courses', 'Courses', 'route'], ['canvas', 'Canvas', 'layout'], ['playground', 'Playground', 'grid'], ['textlab', 'Text lab', 'terminal'], ['challenges', 'Challenges', 'trophy']];
  // other names a link may use for a page
  const ALIAS = { topic: 'topics', part: 'topics', lesson: 'learn', exercise: 'learn', tour: 'learn', course: 'courses', challenge: 'challenges', scenes: 'demo', workflow: 'canvas', text: 'textlab', lab: 'playground' };
  const frame = () => document.getElementById('app') || document.body;
  const SIBLINGS = [['See', 'https://see.jyotiprakash.org'], ['Strux', 'https://strux.jyotiprakash.org'], ['Baysik', 'https://baysik.jyotiprakash.org'], ['Lohgo', 'https://lohgo.jyotiprakash.org'], ['Kwery', 'https://kwery.jyotiprakash.org'], ['Tooring', 'https://tooring.jyotiprakash.org'], ['Sirkit', 'https://sirkit.jyotiprakash.org'], ['Sifer', 'https://sifer.jyotiprakash.org'], ['Proses', 'https://proses.jyotiprakash.org'], ['Arkitek', 'https://arkitek.jyotiprakash.org']];

  const Shell = (S.Shell = {
    NAV, ALIAS,
    /** is autosave on? (it is unless the reader turned it off) */
    autosave: () => !S.app || S.app.settings.autosave !== false,

    buildTop(app) {
      const top = document.getElementById('top');
      let menu = document.getElementById('btnMenu');
      if (!menu) { menu = h('button.icon-btn', { id: 'btnMenu', type: 'button', 'aria-label': 'Sections', title: 'Sections' }); top.insertBefore(menu, top.firstChild); }
      let nav = document.getElementById('topNav');
      if (!nav) { nav = h('nav.top-nav', { id: 'topNav', 'aria-label': 'Sections' }); document.getElementById('brand').after(nav); }
      menu.innerHTML = '';
      menu.appendChild(icon('menu'));
      menu.setAttribute('aria-expanded', 'false');
      menu.addEventListener('click', () => { const on = frame().classList.toggle('menu-open'); menu.setAttribute('aria-expanded', String(on)); });
      document.getElementById('brand').addEventListener('click', (e) => { e.preventDefault(); location.hash = ''; });
      nav.innerHTML = '';
      for (const [id, label, ic] of NAV) {
        if (!S.Pages.get(id)) continue;
        const b = h('button', { type: 'button', dataset: { route: id }, title: label }, icon(ic), h('span', label));
        b.addEventListener('click', () => { frame().classList.remove('menu-open'); location.hash = id; });
        nav.appendChild(b);
      }
      const tools = document.getElementById('topTools');
      tools.innerHTML = '';
      app.btnProj = h('button.icon-btn', { type: 'button', title: 'Projector mode: large type and strong lines for a classroom (Shift+P)', 'aria-label': 'Projector mode', id: 'btnProjector' }, icon('screen'));
      app.btnProj.addEventListener('click', () => app.setProjector(!app.settings.projector));
      app.btnTheme = h('button.icon-btn', { type: 'button', title: 'Light or dark (T)', 'aria-label': 'Light or dark', id: 'btnTheme' });
      app.btnTheme.addEventListener('click', () => app.toggleTheme());
      const about = h('button.icon-btn', { type: 'button', title: 'Help, settings and About', 'aria-label': 'Help and About', id: 'btnAbout' }, icon('help'));
      about.addEventListener('click', () => Shell.aboutMenu(app, about));
      tools.append(app.btnProj, app.btnTheme, about);
      app.applyTheme();
    },
    aboutMenu(app, anchor) {
      const r = anchor.getBoundingClientRect();
      S.UI.showMenu([
        { label: 'Keyboard shortcuts', sc: '?', action: () => app.shortcuts() },
        { label: 'Settings…', action: () => Shell.settings(app) },
        { label: 'The sample scenes', action: () => { location.hash = 'demo'; } },
        { sep: true },
        { label: 'About Nooral', action: () => Shell.about(app) },
        { label: 'About the author', action: () => S.authorDialog && S.authorDialog() },
      ], r.right - 220, r.bottom + 4, { anchor, keyboard: true });
    },
    pageOf(hash) {
      const m = /^([a-z]+)(?:[=/?](.*))?$/i.exec(hash);
      if (!m) return [null, ''];
      return [ALIAS[m[1]] || m[1], hash];
    },

    // ---- home ----------------------------------------------------------------------------------------------
    renderHome(app, main, notice) {
      const box = h('div.home', { id: 'homeView' });
      if (notice) box.appendChild(h('div.notice', icon('info'), h('span', notice)));
      const go = (id, ic, label, hash, primary) => { const b = h('button.btn' + (primary ? '.primary' : ''), { type: 'button', id }, icon(ic), label); b.addEventListener('click', () => { location.hash = hash; }); return b; };
      const art = h('div.hero-art', { id: 'heroArt' }), cap = h('div.hero-cap', { id: 'heroCap', 'aria-live': 'off' });
      box.appendChild(h('section.hero.home-hero',
        h('div.hero-text', h('h1', 'Machine learning you can see inside.'),
          h('p.lead', 'Build a workflow from blocks, then open any block: the boundary forming as a model trains, a tree growing split by split, the weights of a network, attention between words, and the very examples a model gets wrong.'),
          h('div.cta', go('btnStart', 'layout', 'Open the workflow canvas', 'canvas', true), go('btnToTopics', 'library', 'Browse the topics', 'topics')),
          h('p.hero-stats', 'Every model is written for Nooral, so nothing is a black box, and every algorithm is checked against reference results.')),
        h('div.hero-show', art, cap)));
      const tool = (id, ic, title, text, hash) => { const b = h('button.tool-card', { type: 'button', id }, h('span.tc-icon', icon(ic)), h('span.tc-text', h('b', title), h('span', text))); b.addEventListener('click', () => { location.hash = hash; }); return b; };
      box.appendChild(h('h2.sub-title', 'Build, train and look inside'));
      box.appendChild(h('section.home-tools',
        tool('toCanvas', 'layout', 'Workflow canvas', 'Drag blocks, wire data to a model and an evaluation, and open any block to see inside it. Everything reruns as you change it.', 'canvas'),
        tool('toPlayground', 'grid', 'Playground', 'A small network on 2-D data: add layers and neurons and watch every neuron carve out its region as it learns.', 'playground'),
        tool('toTextlab', 'terminal', 'Text lab', 'Byte-pair encoding merge by merge, a tiny language model trained here, and sampling with the probability bars.', 'textlab'),
        tool('toChallenges', 'trophy', 'Challenges', 'Reach a target score on a hidden test set within limits on blocks, parameters and epochs.', 'challenges'),
        tool('toScenes', 'layers', 'Sample scenes', 'The pictures at a glance: boundaries, trees, margins, clusters, optimisers, networks and attention.', 'demo')));
      const parts = S.parts || [];
      if (parts.length) box.appendChild(h('h2.sub-title', 'The course, in ' + parts.length + ' parts'));
      const grid = h('div.part-grid');
      parts.forEach((p) => {
        const n = (S.topics || []).filter((t) => { const f = S.familyOf(t); return f && f.part === p.id; }).length;
        grid.appendChild(h('section.part.part-card', { id: 'part-' + p.n }, h('div.part-head', h('span.num', String(p.n)), h('h3', h('a', { href: '#part=' + p.id }, p.title)), n ? h('span.part-n', n + ' topic' + (n === 1 ? '' : 's')) : null), h('p.part-blurb', p.blurb)));
      });
      box.appendChild(grid);
      main.appendChild(box);
      requestAnimationFrame(() => { if (!art.isConnected || !S.HomeHero) return; const hero = S.HomeHero(art, cap); if (hero) { app.hero = hero; app.players.push(hero); } });
    },

    // ---- settings and About ------------------------------------------------------------------------------------
    settings(app) {
      const s = app.settings;
      const row = (title, sub, ctl) => h('div.settings-row', h('div.what', h('b', title), h('span', sub)), ctl);
      const chk = (key, after, dflt) => { const c = h('input', { type: 'checkbox', checked: s[key] === undefined ? !!dflt : !!s[key], id: 'set-' + key }); c.addEventListener('change', () => { s[key] = c.checked; app.saveSettings(); app.applyTheme(); if (after) after(); }); return c; };
      const sel = (key, opts, after) => { const c = h('select', { id: 'set-' + key, 'aria-label': key }, ...opts.map(([v, l]) => h('option', { value: v, selected: String(s[key]) === String(v) }, l))); c.addEventListener('change', () => { s[key] = c.value; app.saveSettings(); app.applyTheme(); if (after) after(); }); return c; };
      const L = S.Learner;
      const pr = h('input', { type: 'checkbox', checked: !!(L && L.settings.predict), id: 'set-predict' });
      pr.addEventListener('change', () => { if (L) { L.settings.predict = pr.checked; L.saveSettings(); } });
      const gpu = S.T && S.T.compute ? S.T.compute.name : 'cpu';
      S.UI.modal({ title: 'Settings', wide: true, body: h('div',
        row('Theme', 'Light, dark, or whatever the system uses', sel('theme', [['auto', 'Like the system'], ['light', 'Light'], ['dark', 'Dark']], () => app.redraw())),
        row('Animate', 'Values glide between steps; off shows each step at once', chk('animate', null, true)),
        row('Predictions', 'The topic player stops to ask what comes next before key moments', pr),
        row('Keep my work', 'Save the canvas, the playground and the text lab in this browser as I go', chk('autosave', null, true)),
        row('Projector mode', 'Large type and strong lines for a classroom (Shift+P)', chk('projector', () => app.setProjector(s.projector))),
        h('p.muted.engines', { id: 'engineStatus' }, 'Engines: ' + Shell.engines().map((x) => x.name + (x.live ? '' : ' (coming)')).join(', ') + '. Heavy arithmetic runs on: ' + (gpu === 'webgpu' ? 'the graphics card (WebGPU)' : 'the processor') + '.')),
      buttons: [{ label: 'Close', value: true, primary: true }] });
    },
    engines() {
      const has = (p) => p.split('.').reduce((o, k) => (o && o[k] !== undefined ? o[k] : undefined), S) !== undefined;
      return [['Tensors and autodiff', 'T.tensor'], ['Networks', 'T.nn'], ['Classical models', 'ML.LogisticRegression'], ['Trees and ensembles', 'ML.RandomForestClassifier'], ['Support vector machines', 'ML.SVC'],
        ['Clustering', 'ML.KMeans'], ['Metrics', 'Metrics.confusion'], ['Explanations', 'Explain'], ['Datasets', 'Data.load'], ['Workflows', 'WF.Session']].map(([name, path]) => ({ name, live: has(path) }));
    },
    about() {
      const more = h('a', { href: '#' }, 'About the author');
      more.addEventListener('click', (e) => { e.preventDefault(); S.UI.closeAllModals(); setTimeout(() => S.authorDialog && S.authorDialog(), 50); });
      const sib = (name, url) => h('a', { href: url, target: '_blank', rel: 'noopener' }, name);
      const sibs = [];
      SIBLINGS.forEach(([n, u], i) => { if (i) sibs.push(i === SIBLINGS.length - 1 ? ' and ' : ', '); sibs.push(sib(n, u)); });
      S.UI.modal({
        title: 'About Nooral',
        body: h('div', h('p', h('b', 'Nooral'), ' @@VERSION@@ is machine learning you can see inside: workflows built from blocks, and every model opened up while it trains, tests and predicts, from linear models, trees and support vector machines to networks, transformers and a tiny language model. Every model is written for the tool, so any intermediate value can be drawn, and every algorithm is checked against reference results. Every example, dataset, story and exercise in it was made for Nooral, apart from a few classic open datasets, credited where they are used.'),
          h('p', 'It runs entirely in your browser from one file: no server, no account, nothing sent anywhere. Your progress, settings and workflows stay in this browser.'),
          h('p', 'Its sister tools: ', ...sibs, '.'),
          h('p', 'Free software under the GNU General Public License, version 3 or later. Every copy and every derived work must keep this attribution: Based on Nooral by Jyotiprakash Mishra (https://nooral.jyotiprakash.org).'),
          h('p', 'Copyright © 2026 Jyotiprakash Mishra. The name Nooral is not licensed for derived works.'), h('p', more)),
        buttons: [{ label: 'OK', value: true, primary: true }],
      });
    },

    // ---- share links -------------------------------------------------------------------------------------------
    /** copy a link that opens this state: kind is the page that opens it (canvas, playground, textlab, topics) */
    async share(kind, data) {
      const code = await S.Share.pack({ kind, data });
      const url = location.href.replace(/#.*$/, '') + '#share=' + code;
      S.Share.copy(url, url.length > 6000 ? 'Copied. The link is long (' + Math.round(url.length / 1000) + ' KB): some apps cut long links, so for a big workflow save it as a file instead.' : 'Link copied: whoever opens it gets this exact setup.');
      return url;
    },
    async openShared(app, code) {
      try {
        const { kind, data } = await S.Share.unpack(code);
        const pg = S.Pages.get(kind);
        if (!pg || !pg.load) throw new Error('This link is for a part of Nooral this version does not have.');
        const where = pg.load(data) || kind;
        history.replaceState(null, '', '#' + where);
        app.route();
        S.UI.toast('Opened the shared ' + ({ canvas: 'workflow', playground: 'network', textlab: 'text-lab setup' }[kind] || 'setup') + '.', 'ok');
      } catch (e) {
        history.replaceState(null, '', '#');
        app.route();
        S.UI.toast(e.message, 'err', 5000);
      }
    },
  });

})(globalThis.Nooral = globalThis.Nooral || {});
