// The screenshots in the README: node tools/readme-shots.mjs [name ...] [--built] [--base=url] [--list]
// Drives the page in headless Chrome and writes docs/images/<name>.png: 1440×900 in the light theme, one shot in
// the dark theme and one on a phone at 390×844. Each topic stops on a frame part of the way through, where the
// picture has something to say. With names, only those shots are taken again; --list prints the names.
// By default it uses dev.html (every source file on its own, no build needed); --built builds index.html first and
// shoots that, as the site serves it; --base= points it at a page served elsewhere (such as a running
// node tools/devpage.mjs). Prints console errors per shot.
import { launch } from './cdp.mjs';
import { execSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const built = process.argv.includes('--built');
if (built) execSync('node build.mjs', { cwd: root, stdio: 'ignore' });
mkdirSync(join(root, 'docs/images'), { recursive: true });
const out = (n) => join(root, 'docs/images', n + '.png');
const only = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const base = (process.argv.find((a) => a.startsWith('--base=')) || '').slice(7);
const W = [1440, 900], PHONE = [390, 844];

const wait = (ms) => `await new Promise((r) => setTimeout(r, ${ms}));`;
// wait for an expression to come true inside the page
const until = (expr, ms = 30000) => `await new Promise((r, j) => { const t0 = Date.now(); const w = () => { let ok = false; try { ok = (${expr}); } catch (e) { /* not yet */ } if (ok) r(); else if (Date.now() - t0 > ${ms}) j(new Error('timed out: ' + ${JSON.stringify(expr)})); else setTimeout(w, 40); }; w(); });`;
// the box that scrolls the page, whichever it is
const SC = `[document.getElementById('main'), document.scrollingElement].find((x) => x && x.scrollHeight > x.clientHeight + 2)`;
// scroll the page so an element sits just under the top bar
const scrollTo = (sel, pad = 12) => `{ const e = document.querySelector(${JSON.stringify(sel)}); if (e) { const bar = document.getElementById('top'); const off = (bar ? bar.getBoundingClientRect().bottom : 56) + ${pad}; const sc = ${SC}; if (sc) sc.scrollTop += e.getBoundingClientRect().top - off; } }`;
const P = 'Nooral.app.page';
// open a topic, pick a preset (its index among the chips, or none), and stop on a frame part of the way through
// (frac, or a function of the frames that picks the index)
const topic = (id, frac, preset, extra) => ({
  hash: 'topic=' + id,
  js: until(`${P} && ${P}.player && ${P}.player.frames.length > 1`, 60000)
    + (preset != null ? `${P}.player.pause(); document.querySelectorAll('.presets .chip')[${preset}].click(); ${until(`${P}.player.frames.length > 1 && document.querySelectorAll('.presets .chip.on').length`, 60000)}` : '')
    + ` const p = ${P}.player; p.pause(); const fr = p.frames; p.go(${typeof frac === 'function' ? `(${frac})(fr)` : `Math.round(${frac} * (fr.length - 1))`}, true); ${wait(300)} ${scrollTo('.stage-card', 4)} ${extra || ''}`,
});
// a sample workflow on the canvas with one block's viewer open
const canvas = (sample, block, then) => ({
  hash: 'canvas',
  js: until(`${P} && ${P}.loadSample`) + `await ${P}.loadSample(${JSON.stringify(sample)}); await ${P}.whenIdle(); ${P}.open(${JSON.stringify(block)}); ${wait(600)} ${then || ''}`,
  wait: 400,
});
// the text lab's GPT trained for n steps in its worker
const trained = (n) => `await new Promise((r) => { const L = Nooral.TL.link(); const go = () => { if (!L.ready) return setTimeout(go, 50); L.send({ cmd: 'steps', n: ${n} }); const off = L.on((m) => { if (m.type === 'state' && !m.running) { off(); r(L.step); } }); }; go(); });`;

// name: { hash, js (after the route), size, wait ms, theme }
const SHOTS = {
  // the home page with its live picture
  home: { hash: '', js: until(`document.querySelector('#heroArt svg, #heroArt canvas')`) + wait(2500), wait: 100 },

  // the canvas: small loans through a split into a logistic regression, the confusion matrix open, one cell clicked
  canvas: canvas('evaluate-loans', 'evaluate1', `{ const e = document.querySelector('[data-key="cm:c1_0"]'); if (e) e.dispatchEvent(new MouseEvent('click', { bubbles: true })); } ${wait(300)}`),
  // the canvas: a network on the spirals, its training viewer open
  canvasmlp: canvas('mlp-spirals', 'train1', wait(1500)),
  // the whole canvas with no viewer open: a comparison of three models
  workflow: { hash: 'canvas', js: until(`${P} && ${P}.loadSample`) + `await ${P}.loadSample('compare-three'); await ${P}.whenIdle(); ${P}.editor.fit(); ${wait(400)}`, wait: 300 },

  boundary: topic('logit-boundary', 0.6, 3),
  tree: topic('cart-grow', 0.55),
  svm: topic('svm-kernel-lift', 0.8),
  kmeans: topic('km-steps', 0.5),
  race: topic('opt-race', 0.6, 0),
  backprop: topic('p09-backprop', 0.7),
  cnn: topic('cnn-flow', 1),
  gradcam: topic('cnn-gradcam', 0.8),
  attention: topic('attention-sdpa', 0.85),
  heads: topic('attention-multihead', 0.8),
  vae: topic('gen-vae-map', 0.6),
  rl: topic('rl-value-iteration', 1),
  gnn: topic('message-passing', 0.6),
  shap: topic('shap-waterfall', 0.9),

  // the playground: spirals learned by a deep ReLU network, every neuron's region drawn
  playground: {
    hash: 'playground',
    js: until(`${P} && ${P}.set && document.querySelector('#pgNetStage')`) + `${P}.set({ data: 'spirals', layers: [8, 8, 6, 4], act: 'relu', features: ['x1', 'x2', 'x1sq', 'x2sq', 'x1x2', 'sin1', 'sin2'], lr: 0.03, seed: 3 }); ${P}.train(300); ${until(`document.querySelector('#pgNetStage .it-raster')`)} ${wait(800)}`,
    wait: 400,
  },
  // the text lab: the tiny GPT after some training, its loss, a sample and the attention of a head
  gpt: { hash: 'textlab/gpt', js: until('Nooral.TL.link().ready', 30000) + trained(900) + until(`document.querySelectorAll('#gptSamples .tl-sample').length >= 1`) + wait(800), wait: 400, timeout: 600000 },
  bpe: { hash: 'textlab/bpe', js: until(`document.querySelectorAll('#bpeStage .it-cell.tok').length > 10`) + `${P}.view.go(60); ${wait(400)}`, wait: 300 },
  sampling: { hash: 'textlab/sample', js: until(`document.querySelectorAll('#smpBars .it-cell.tok').length === 10`) + `${P}.view.set({ greedy: false, topK: 5, temperature: 0.8, prompt: 'The night market' }); ${wait(300)} document.getElementById('smpNext').click(); ${wait(600)}`, wait: 300 },
  challenge: { hash: 'challenges', js: until(`document.querySelector('#challengesView')`) + wait(400), wait: 200 },
  learn: { hash: 'learn', js: until(`document.querySelector('#learnView')`) + wait(400), wait: 200 },

  dark: Object.assign(topic('attention-multihead', 0.8), { theme: 'dark' }),
  phone: { hash: 'playground', size: PHONE, js: until(`${P} && ${P}.set && document.querySelector('#pgNetStage')`) + `${P}.set({ data: 'circles', layers: [4, 2], act: 'tanh', seed: 1 }); ${P}.train(80); ${until(`document.querySelector('#pgNetStage .it-raster')`)} ${wait(800)}`, wait: 400 },
};

if (process.argv.includes('--list')) { console.log(Object.keys(SHOTS).join('\n')); process.exit(0); }
for (const n of only) if (!SHOTS[n]) { console.log('no shot called ' + n + ' (--list)'); process.exit(1); }
for (const [name, s] of Object.entries(SHOTS)) {
  if (only.length && !only.includes(name)) continue;
  const [w, hgt] = s.size || W;
  const b = await launch({ width: w, height: hgt, mobile: w < 600 });
  try {
    const url = (base || b.url) + (built ? '' : 'dev.html');
    await b.send('Page.navigate', { url: url + '#' });
    await b.waitFor('window.__nooralReady === true', 30000);
    await b.eval(`Nooral.app.settings.theme = ${JSON.stringify(s.theme || 'light')}; Nooral.app.settings.animate = false; if (Nooral.Learner) Nooral.Learner.settings.predict = false; Nooral.app.applyTheme(); 1`);
    await b.eval(`(async () => { ${s.hash ? `window.__nooralReady = false; location.hash = ${JSON.stringify(s.hash)}; ${until('window.__nooralReady === true')}` : 'Nooral.app.route();'} ${s.js || ''} })().then(() => 1)`, s.timeout);
    await b.sleep(s.wait || 300);
    await b.screenshot(out(name));
    const errs = b.logs.filter((l) => l.type === 'exception' || l.type === 'error');
    console.log(name + (errs.length ? '  console: ' + errs.map((e) => e.text.slice(0, 200)).join(' | ') : ''));
  } catch (e) {
    console.log(name + ' FAILED: ' + e.message.slice(0, 300));
  } finally { await b.close(); }
}
