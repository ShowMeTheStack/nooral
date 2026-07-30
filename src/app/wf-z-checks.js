// What the diagram checker looks at on the canvas, registered after every file that adds samples: every sample on the
// canvas page as the tidy-up lays it out, every sample as a lesson shows it (the compact canvas, folded to the width
// of a column of text), and every lesson that holds a live workflow. The audit treats each block as a shape and
// each wire as an edge: no wire may run through or under a block, its own ends included.
(function (S) {
  'use strict';
  const WF = S.WF;
  if (!S.Checks || !WF || S.Checks.some((c) => c.group === 'canvas-embed')) return;
  const ready = "document.querySelector('#canvasView .wf-svg') && Nooral.app.page && Nooral.app.page.whenIdle";
  for (const id of WF.samples.keys()) S.Checks.push({ group: 'canvas', name: 'sample ' + id, hash: 'canvas', ready, setup: 'await Nooral.app.page.loadSample(' + JSON.stringify(id) + '); await Nooral.app.page.whenIdle();', sel: '#main' });
  // a sample inside a lesson's column: drawn into the learn page's main area in place of the list
  const embed = (id) => `
    if (window.__wfEmb) window.__wfEmb.dispose();
    const m = document.querySelector('#main'), page = { editors: [] };
    m.innerHTML = '';
    const art = document.createElement('article');
    art.className = 'lesson';
    art.appendChild(Nooral.LessonBlocks.workflow({ workflow: ${JSON.stringify(id)}, live: true, height: 420 }, page));
    m.appendChild(art);
    let n = 0;
    while (!page.editors.length && ++n < 200) await new Promise((r) => setTimeout(r, 25));
    window.__wfEmb = { dispose() { for (const e of page.editors) e.dispose(); } };
    await Promise.all(page.editors.map((e) => e.whenIdle()));
    await new Promise((r) => setTimeout(r, 120));`;
  for (const id of WF.samples.keys()) S.Checks.push({ group: 'canvas-embed', name: 'embedded ' + id, hash: 'learn', ready: "document.querySelector('#learnView, .lesson')", setup: embed(id), sel: '#main' });
  // every lesson with a live workflow, once the content has loaded (it loads after the app)
  const lessons = () => {
    const C = S.Content;
    if (!C || !C.lessons) return;
    // every lesson, its figures drawn as the reader sees them (a filter picks some: --pages lessons da-)
    if (!S.Checks.some((c) => c.group === 'lessons')) for (const l of C.lessons.values()) S.Checks.push({ group: 'lessons', name: 'lesson ' + l.id, hash: 'lesson=' + l.id, ready: "document.querySelector('#lessonView')", setup: 'await new Promise((r) => setTimeout(r, 300));', sel: '#main' });
    for (const l of C.lessons.values()) {
      const n = (l.blocks || []).filter((b) => b && b.workflow && b.live).length;
      if (!n) continue;
      S.Checks.push({
        group: 'lesson-wf', name: 'lesson ' + l.id, hash: 'lesson=' + l.id,
        ready: "document.querySelectorAll('.ls-wf .wf-svg .wf-block').length && document.querySelectorAll('.ls-wf .wf-svg').length === " + n + " && Nooral.app.page && Nooral.app.page.editors && Nooral.app.page.editors.length === " + n,
        setup: 'await Promise.all(Nooral.app.page.editors.map((e) => e.whenIdle())); await new Promise((r) => setTimeout(r, 150));',
        sel: '#main',
      });
    }
  };
  if (typeof document !== 'undefined' && document.readyState === 'loading') document.addEventListener('DOMContentLoaded', lessons);
  else lessons();
})(globalThis.Nooral = globalThis.Nooral || {});
