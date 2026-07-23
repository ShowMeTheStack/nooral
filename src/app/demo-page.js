// The demo page (#demo): every sample scene of src/demo in its own small player (S.Player, player.js). The diagram
// checker walks these scenes frame by frame through Nooral.app.byId.
(function (S) {
  'use strict';
  const { h } = S.UI;
  S.Pages.register('demo', {
    nav: 'demo',
    title: 'The sample scenes',
    open(app, main) {
      const scenes = (S.Demo && S.Demo.scenes) || [];
      main.append(h('div.demo-head', h('h1', 'Nooral: the pictures'), h('p', 'Sample scenes drawn from the engines: every boundary, weight and value shown is computed live, on our own datasets and texts. Step through a scene, or press play.'), h('p', h('a', { href: '#flow' }, 'Open the workflow canvas'), ': data, split, logistic regression and a confusion matrix, wired and running live.'),
        h('nav.demo-nav', { 'aria-label': 'Scenes' }, scenes.map((s) => h('a', { href: '#demo', onclick: (e) => { e.preventDefault(); const el = document.getElementById(s.id); if (el) el.scrollIntoView({ behavior: 'smooth' }); } }, s.short || s.title)))));
      for (const sc of scenes) { const p = new S.Player(sc); app.players.push(p); main.append(p.el); }
      for (const p of app.players) { try { p.build(); } catch (e) { console.error(p.scene.id + ': ' + (e && e.stack || e)); } }
      app.byId = Object.fromEntries(app.players.map((p) => [p.scene.id, p]));
      return { dispose() {} };
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
