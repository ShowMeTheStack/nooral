// Lesson blocks beyond the basic ones (lessons.js lists them all): a scene of the demo page, a workflow (a card that
// opens it on the canvas, or the canvas itself, live), and cards that open the playground or the text lab set up
// for the lesson.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const LB = (S.LessonBlocks = S.LessonBlocks || {});
  const inline = (t) => S.UI.inline(t);
  const figure = (el, caption, cls) => h('figure.ls-fig' + (cls ? '.' + cls : ''), el, caption ? h('figcaption', inline(caption)) : null);
  const card = (href, ic, kind, title, meta) => h('a.ls-link', { href }, h('span.tc-icon', icon(ic)), h('span', h('span.kind', kind), h('b', title), meta ? h('span.meta', meta) : null), h('span.go', '→'));

  /** {scene: 'demo scene id'}: the scene with its own small player */
  LB.scene = function (b, page) {
    const sc = S.Demo && S.Demo.byId && S.Demo.byId(b.scene);
    if (!sc) return figure(S.UI.empty('watch', 'A figure is missing here', 'It shows the scene "' + b.scene + '", which is not in this build.'), b.caption);
    const p = new S.Player(sc);
    const box = h('div.ls-scene', p.el);
    box._redraw = () => p.resized();
    requestAnimationFrame(() => { try { p.build(); } catch (e) { /* drawn on the next resize */ } });
    if (page) (page.scenes = page.scenes || []).push(p);
    if (S.app) S.app.players.push(p);
    return figure(box, b.caption, 'scene');
  };
  /** {workflow: {blocks, wires} | 'sample id', title, text, live, height, caption}: a card that opens the workflow on
   * the canvas, or with live: true the canvas itself, running, inside the lesson */
  LB.workflow = function (b, page) {
    const doc = typeof b.workflow === 'string' ? (S.WF && S.WF.sample ? S.WF.sample(b.workflow) : null) : b.workflow;
    if (b.live && S.WF && S.WF.embed && doc) {
      const host = h('div.ls-wf', { style: { height: (b.height || 420) + 'px' } });
      requestAnimationFrame(() => { try { const ed = S.WF.embed(host, doc, { readOnly: !!b.readOnly }); if (page) (page.editors = page.editors || []).push(ed); } catch (e) { host.appendChild(S.UI.empty('warn', 'The workflow could not open', String(e.message || e))); } });
      return figure(h('div', host, h('div.ls-fig-tools', h('span.muted', b.title || 'A workflow'), openBtn(doc))), b.caption, 'workflow');
    }
    const a = card('#canvas', 'layout', 'On the workflow canvas', b.title || 'Open this workflow', b.text || '');
    a.addEventListener('click', () => { if (doc && S.WF && S.WF.preset) S.WF.preset(doc); });
    return a;
  };
  const openBtn = (doc) => { const a = h('a.btn.small', { href: '#canvas', title: 'Open this workflow on the canvas, to change it' }, 'Open on the canvas'); a.addEventListener('click', () => { if (S.WF && S.WF.preset) S.WF.preset(doc); }); return a; };
  /** {playground: {data, layers, act, lr, reg, ...}, title, text}: a card that opens the playground set up like this */
  LB.playground = function (b) {
    const a = card('#playground', 'grid', 'In the playground', b.title || 'A network to train', b.text || '');
    a.addEventListener('click', () => { if (S.Playground && S.Playground.preset) S.Playground.preset(b.playground); });
    return a;
  };
  /** {textlab: {tab: 'bpe' | 'gpt' | 'sample', ...}, title, text}: a card that opens the text lab */
  LB.textlab = function (b) {
    const o = b.textlab || {};
    const a = card('#textlab' + (o.tab ? '/' + o.tab : ''), 'terminal', 'In the text lab', b.title || 'Tokens and a tiny language model', b.text || '');
    a.addEventListener('click', () => { if (S.TL && S.TL.openWith) S.TL.openWith(o); });
    return a;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
