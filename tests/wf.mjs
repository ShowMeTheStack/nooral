// The workflow engine: documents, wiring rules, hashing and the session that runs them (node tests/wf.mjs).
import { load, harness } from './load.mjs';
const S = load(['core', 'app']);
const WF = S.WF, t = harness('workflow');

const doc = WF.newDoc('test');
const d = WF.add(doc, 'data', { set: 'moons', n: 200, noise: 0.2 });
const sp = WF.add(doc, 'split', { test: 0.3 });
const m = WF.add(doc, 'logistic', {});
const tr = WF.add(doc, 'train', {});
const ev = WF.add(doc, 'evaluate', {});
WF.connect(doc, [d.id, 'out'], [sp.id, 'in']);
WF.connect(doc, [m.id, 'model'], [tr.id, 'model']);
WF.connect(doc, [sp.id, 'train'], [tr.id, 'data']);
WF.connect(doc, [tr.id, 'model'], [ev.id, 'model']);
WF.connect(doc, [sp.id, 'test'], [ev.id, 'data']);
t.eq(WF.order(doc), ['data1', 'logistic1', 'split1', 'train1', 'evaluate1'], 'order');
t.ok(WF.canWire(doc, [ev.id, 'pred'], [d.id, 'out']) !== true, 'no wire into an output');
t.ok(/takes/.test(WF.canWire(doc, [m.id, 'model'], [ev.id, 'data'])), 'type check: ' + WF.canWire(doc, [m.id, 'model'], [ev.id, 'data']));
t.ok(/loop/.test(String(WF.canWire(doc, [ev.id, 'pred'], [sp.id, 'in']))) || WF.canWire(doc, [ev.id, 'pred'], [sp.id, 'in']) !== true, 'no loops');
t.eq(WF.outType(doc, sp.id, 'train'), 'table', 'a split of a table is a table');

const events = [];
const s = new WF.Session({ onEvent: (e) => events.push(e) }).load(doc).runSync();
for (const b of doc.blocks) t.eq(s.res[b.id].state, 'done', b.id + ' ran: ' + (s.res[b.id].error || ''));
const pred = s.output(ev.id, 'pred');
const acc = WF.score('accuracy', pred);
t.ok(acc > 0.8 && acc < 1, 'logistic regression on moons: accuracy ' + acc);
t.eq(pred.y.length, 60, 'test rows');
const v = s.res[ev.id].view;
t.eq(v.counts.flat().reduce((a, b) => a + b, 0), 60, 'confusion counts add up');
t.ok(JSON.stringify(v).length < 200000, 'the view is small enough to send');

// only what changed runs again
const n0 = events.filter((e) => e.type === 'state' && e.state === 'done').length;
const doc2 = WF.clone(doc); WF.byId(doc2, m.id).params.C = 0.01;
s.load(doc2).runSync();
const ran = events.slice().filter((e) => e.type === 'state' && e.state === 'done').slice(n0).map((e) => e.id);
t.eq(ran, ['logistic1', 'train1', 'evaluate1'], 'a changed parameter runs only what is downstream');
t.ok(WF.score('accuracy', s.output(ev.id, 'pred')) !== acc, 'and changes the result');

// raw rows go through the model's pipe
const model = s.output(tr.id, 'model');
t.eq(model.predictRaw([[0, 0.5]]).length, 1, 'predictRaw');
// a missing input waits, it does not fail
const doc3 = WF.clone(doc); doc3.wires = doc3.wires.filter((w) => w.to[0] !== tr.id || w.to[1] !== 'data');
const s3 = WF.runSync(doc3);
t.eq(s3.res[tr.id].state, 'waiting', 'unwired input waits');
t.eq(s3.res[ev.id].state, 'waiting', 'and so does what it feeds');
// JSON round trip
t.eq(WF.fromJSON(JSON.stringify(doc)).wires.length, 5, 'fromJSON');

// a trainer, driven: paused at the start, one epoch, one batch, then play to the end
WF.model('dummynet', { label: 'Dummy net', task: 'classification', iterative: true, params: [] });
WF.trainers.dummynet = (ctx, spec, data) => {
  const epochs = 3, batches = 4;
  const tr = { epochs, epoch: 0, batch: 0, steps: 0,
    step() { this.steps++; this.batch++; if (this.batch === batches) { this.batch = 0; this.epoch++; } return { loss: 1 / this.steps }; },
    done() { return this.epoch >= epochs; }, progress() { return (this.epoch * batches + this.batch) / (epochs * batches); },
    current() { const steps = this.steps; return { out: { model: WF.fitted({ family: 'dummynet', label: 'Dummy net', params: {}, predict: (X) => X.map(() => 0), steps }, data) }, epoch: this.epoch, epochs }; },
    snapshot() { return { live: true, epoch: this.epoch }; } };
  return tr;
};
const d4 = WF.clone(doc); WF.byId(d4, m.id).type = 'dummynet'; WF.byId(d4, m.id).params = {};
const ev4 = [];
const s4 = new WF.Session({ autoTrain: false, onEvent: (e) => ev4.push(e) }).load(d4);
await s4.run();
t.eq(s4.res[tr.id].state, 'paused', 'a trainer starts paused when asked');
t.eq(s4.output(tr.id, 'model').steps, 0, 'and has taken no step');
t.eq(s4.res[ev.id].state, 'done', 'the untrained model is still evaluated');
await s4.control(tr.id, 'epoch');
t.eq(s4.output(tr.id, 'model').steps, 4, 'one epoch is four batches');
await s4.control(tr.id, 'batch');
t.eq(s4.output(tr.id, 'model').steps, 5, 'then one batch');
const evalRuns = ev4.filter((e) => e.type === 'state' && e.id === ev.id && e.state === 'done').length;
t.eq(evalRuns, 3, 'evaluate ran after each step of training');
await s4.control(tr.id, 'play');
t.eq(s4.res[tr.id].state, 'done', 'play runs to the end');
t.eq(s4.output(tr.id, 'model').steps, 12, 'all twelve batches');
t.ok(ev4.some((e) => e.type === 'curve' && e.id === tr.id), 'curve points were sent');
await s4.control(tr.id, 'reset');
t.eq(s4.output(tr.id, 'model').steps, 0, 'reset goes back to the untrained model');
t.done();
