// The canvas's tidy-up: every sample and random workflows that can be drawn without crossings are laid out with none,
// blocks never overlap, and every wire runs left to right (node tests/wf-layout.mjs).
import { files, harness, rng } from './load.mjs';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
// the engine, the blocks and the canvas's layout and samples, each file on its own (a block file being written
// elsewhere that does not load yet costs only its own blocks)
const ctx = vm.createContext({ console, performance, setTimeout, clearTimeout, structuredClone, TextEncoder, TextDecoder });
for (const f of files(['core', 'app']).filter((f) => f.startsWith('core/') || /^app\/(wf-[ab]-|wf-ui-(layout|samples))/.test(f))) {
  try { vm.runInContext(readFileSync(new URL('../src/' + f, import.meta.url), 'utf8'), ctx, { filename: f }); } catch (e) { if (f.startsWith('core/') || /wf-a-|wf-ui-/.test(f)) throw e; console.log('(skipped ' + f + ': ' + e.message + ')'); }
}
const S = ctx.Nooral;
const WF = S.WF, t = harness('wf-layout');
const overlaps = (doc) => {
  let n = 0;
  for (let i = 0; i < doc.blocks.length; i++) for (let j = i + 1; j < doc.blocks.length; j++) {
    const a = doc.blocks[i], b = doc.blocks[j], ga = WF.geom(a, doc), gb = WF.geom(b, doc);
    if (a.x < b.x + gb.w + 8 && b.x < a.x + ga.w + 8 && a.y < b.y + gb.h + 8 && b.y < a.y + ga.h + 8) n++;
  }
  return n;
};
// a wire runs left to right, unless it is folded down to the next row of a long workflow
const forwards = (doc) => doc.wires.every((w) => (w.via && w.via[0] && w.via[0].c) || WF.byId(doc, w.to[0]).x > WF.byId(doc, w.from[0]).x);
// where one output feeds several blocks that each need a second input, some crossing cannot be helped (one output
// fanning out to k blocks, each with its own other source, makes k − 1): those samples are allowed exactly that many
// so a sample may have the crossings of the best order of its layers (the layout tries every order of small ones)
for (const [id] of WF.samples) {
  const doc = WF.sample(id);
  // every sample is tidied when it opens on the canvas
  WF.tidy(doc);
  t.ok(WF.crossings(doc) <= WF.tidy.last.order, 'sample ' + id + ' crossings: ' + WF.crossings(doc) + ', the best order has ' + WF.tidy.last.order);
  if (!WF.tidy.last.order) t.eq(WF.crossings(doc), 0, 'sample ' + id + ' has no crossing');
  t.eq(overlaps(doc), 0, 'sample ' + id + ' overlaps');
  t.ok(forwards(doc), 'sample ' + id + ' wires run left to right');
}
// random trees of blocks (each block fed by one earlier block): always drawable without crossings
const r = rng(7);
const types = ['split', 'evaluate', 'train'];
let trees = 0;
for (let k = 0; k < 60; k++) {
  const doc = WF.newDoc();
  WF.add(doc, 'data', {});
  const n = 3 + Math.floor(r() * 9);
  for (let i = 0; i < n; i++) {
    // a split fed by any earlier output that is a table
    const outs = [];
    for (const b of doc.blocks) for (const p of WF.ports(b).outputs) { const ty = WF.outType(doc, b.id, p.id); if (ty === 'table' || ty === 'data') outs.push([b.id, p.id]); }
    const src = outs[Math.floor(r() * outs.length)];
    const sp = WF.add(doc, 'split', {});
    WF.connect(doc, src, [sp.id, 'in']);
  }
  void types;
  WF.tidy(doc);
  trees++;
  t.eq(WF.crossings(doc), 0, 'random tree ' + k + ' crossings');
  t.eq(overlaps(doc), 0, 'random tree ' + k + ' overlaps');
}
// random DAGs of m pipelines sharing one split: the two fans (train rows to m trains, test rows to m evaluations)
// force some crossings; the drawing must have no more than the best possible order of the layers
for (let k = 0; k < 40; k++) {
  const doc = WF.newDoc();
  const d = WF.add(doc, 'data', {}), sp = WF.add(doc, 'split', {});
  WF.connect(doc, [d.id, 'out'], [sp.id, 'in']);
  const m = 1 + Math.floor(r() * 3);
  for (let i = 0; i < m; i++) {
    const mo = WF.add(doc, 'logistic', {}), tr = WF.add(doc, 'train', {}), ev = WF.add(doc, 'evaluate', {});
    WF.connect(doc, [mo.id, 'model'], [tr.id, 'model']); WF.connect(doc, [sp.id, 'train'], [tr.id, 'data']);
    WF.connect(doc, [tr.id, 'model'], [ev.id, 'model']); WF.connect(doc, [sp.id, 'test'], [ev.id, 'data']);
  }
  // shuffle the block order so the layout cannot lean on the order they were added in
  doc.blocks.sort(() => r() - 0.5);
  WF.tidy(doc);
  // every order of every layer was tried for these (they are small): the drawing has no more crossings than the best order
  t.ok(WF.crossings(doc) <= WF.tidy.last.order, 'pipelines ×' + m + ' (' + k + ') crossings: ' + WF.crossings(doc) + ', the best order has ' + WF.tidy.last.order);
  t.eq(overlaps(doc), 0, 'pipelines ×' + m + ' (' + k + ') overlaps');
}
t.ok(trees === 60, 'trees tried');
// compact (inside lessons and challenges): every sample, folded to 3 columns, with no overlap and no more crossings
// than the best order
for (const [id] of WF.samples) {
  const doc = WF.setCompact(WF.sample(id), true);
  WF.tidy(doc, { maxLayers: 3 });
  t.ok(WF.crossings(doc) <= WF.tidy.last.order, 'compact ' + id + ' crossings: ' + WF.crossings(doc));
  t.eq(overlaps(doc), 0, 'compact ' + id + ' overlaps');
}
// no wire runs through or under a block, its own ends included: every sample, full size and compact at every fold
for (const [id] of WF.samples) for (const m of [0, 2, 3, 4, 5]) {
  const doc = WF.setCompact(WF.sample(id), !!m);
  WF.tidy(doc, m ? { maxLayers: m } : undefined);
  t.eq(WF.wireHits(doc), [], 'sample ' + id + (m ? ' compact in ' + m + ' columns' : '') + ': wires through blocks');
  // no two wires from different ports share a straight run
  const runs = [];
  for (const g of WF.wireGeom(doc)) if (g.via && g.via[0] && g.via[0].c) { const ps = [g.a].concat(g.via, [g.b]); for (let i = 1; i < ps.length; i++) runs.push({ src: g.w.from.join('.'), p: ps[i - 1], q: ps[i] }); }
  let shared = 0;
  for (let i = 0; i < runs.length; i++) for (let j = i + 1; j < runs.length; j++) {
    const A = runs[i], B = runs[j];
    if (A.src === B.src) continue;
    const vert = (r) => Math.abs(r.p.x - r.q.x) < 0.5, hor = (r) => Math.abs(r.p.y - r.q.y) < 0.5;
    const ov = (a1, a2, b1, b2) => Math.min(Math.max(a1, a2), Math.max(b1, b2)) - Math.max(Math.min(a1, a2), Math.min(b1, b2)) > 2;
    if (vert(A) && vert(B) && Math.abs(A.p.x - B.p.x) < 0.5 && ov(A.p.y, A.q.y, B.p.y, B.q.y)) shared++;
    if (hor(A) && hor(B) && Math.abs(A.p.y - B.p.y) < 0.5 && ov(A.p.x, A.q.x, B.p.x, B.q.x)) shared++;
  }
  t.eq(shared, 0, 'sample ' + id + (m ? ' compact in ' + m + ' columns' : '') + ': wires sharing a run');
}
// blocks dragged anywhere (clear of each other): the router takes every wire round them
{
  const r2 = rng(11);
  for (const [id] of WF.samples) for (let k = 0; k < 3; k++) {
    const doc = WF.sample(id); WF.tidy(doc);
    const b = doc.blocks[Math.floor(r2() * doc.blocks.length)], g = WF.geom(b, doc);
    for (let tries = 0; tries < 60; tries++) {
      const x = Math.round((r2() * 1400 - 200) / 12) * 12, y = Math.round((r2() * 800 - 100) / 12) * 12;
      if (doc.blocks.some((o) => o !== b && x < o.x + WF.geom(o, doc).w + 40 && o.x < x + g.w + 40 && y < o.y + WF.geom(o, doc).h + 40 && o.y < y + g.h + 40)) continue;
      b.x = x; b.y = y; break;
    }
    t.eq(WF.wireHits(doc), [], 'sample ' + id + ' with ' + b.id + ' dragged: wires through blocks');
  }
}
t.done();
