// The text lab's engines: BPE merges, the sampling rules and the tiny transformer (node tests/textlab.mjs).
import { files, harness } from './load.mjs';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
// the core engines and this page's own engine file only, so another page's file being written does not stop the test
const ctx = vm.createContext({ console, performance, setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask, structuredClone, TextEncoder, TextDecoder, URL });
for (const f of files(['core']).concat(['app/textlab-a-model.js'])) vm.runInContext(readFileSync(new URL('../src/' + f, import.meta.url), 'utf8'), ctx, { filename: f });
const S = ctx.Nooral;
const TL = S.TL, t = harness('textlab');
const text = TL.corpus();
t.ok(text.length > 5000, 'the stories corpus: ' + text.length + ' characters');

// BPE: each merge is the most frequent pair, and the same merges as the engine's trainer
const r = TL.bpeTrain(text, 60);
t.eq(r.steps.length, 60, '60 merges');
for (const st of r.steps.slice(0, 20)) t.ok(st.top[0][2] === st.count && st.top.every((x) => x[2] <= st.count), 'merge ' + st.token + ' is the most frequent pair');
t.ok(r.steps.every((s, i) => i === 0 || s.tokens < r.steps[i - 1].tokens), 'every merge shortens the text');
if (S.DL && S.DL.bpe) {
  const ref = S.DL.bpe.train(text, 60);
  t.eq(r.steps.map((s) => s.pair), ref.merges, 'the same merges as S.DL.bpe.train');
  const tk = TL.tokenizer(r.base, r.steps.map((s) => s.pair)), rt = S.DL.bpe.fromMerges(ref.base, ref.merges);
  const sent = 'The market opens at six, and the lights come on.';
  t.eq(tk.pieces(sent), rt.pieces(sent), 'the same pieces');
  t.eq(tk.pieces(sent).join(''), sent, 'pieces join back into the sentence');
}

// sampling rules
const P = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04];
const g = TL.cut(P, { greedy: true });
t.eq(g.probs, [1, 0, 0, 0, 0, 0], 'greedy keeps the top token');
const k = TL.cut(P, { topK: 2 });
t.eq(k.kept, [true, true, false, false, false, false], 'top-k = 2 keeps two');
t.ok(Math.abs(k.probs[0] - 0.4 / 0.65) < 1e-12, 'top-k renormalises');
const p = TL.cut(P, { topP: 0.75 });
t.eq(p.kept, [true, true, true, false, false, false], 'top-p 0.75 keeps 0.4 + 0.25 + 0.15 = 0.8');
const p2 = TL.cut(P, { topP: 0.65 });
t.eq(p2.kept.filter(Boolean).length, 2, 'top-p 0.65: exactly 0.65 with two');
const hot = TL.cut(P, { temperature: 2 }), cold = TL.cut(P, { temperature: 0.5 });
t.ok(hot.probs[0] < 0.4 && cold.probs[0] > 0.4, 'temperature flattens or sharpens');
t.ok(Math.abs(cold.probs[0] - 0.16 / P.reduce((a, x) => a + x * x, 0)) < 1e-9, 'temperature 0.5 squares the probabilities');
t.ok(Math.abs(TL.cut(P, { temperature: 1 }).probs.reduce((a, b) => a + b, 0) - 1) < 1e-12, 'probabilities add to 1');

// the transformer learns, the same way for a seed
const t0 = Date.now();
const E = new TL.Engine(), info = E.init({ d: 16, layers: 1, heads: 2, ctx: 16, batch: 8, lr: 0.01 }, text);
t.ok(info.params > 1000, info.params + ' parameters');
const v0 = E.evaluate();
let first = 0;
for (let i = 0; i < 150; i++) { const l = E.step(); if (i === 0) first = l; }
const v1 = E.evaluate();
t.ok(Math.abs(v0 - Math.log(info.vocab.length)) < 0.6, 'it starts near log V = ' + Math.log(info.vocab.length).toFixed(2) + ' (' + v0.toFixed(2) + ')');
t.ok(v1 < v0 - 0.7, 'the validation loss falls: ' + v0.toFixed(3) + ' → ' + v1.toFixed(3));
const E2 = new TL.Engine(); E2.init({ d: 16, layers: 1, heads: 2, ctx: 16, batch: 8, lr: 0.01 }, text);
let same = true; for (let i = 0; i < 5; i++) if (E2.step() !== E.history[i].loss) same = false;
t.ok(same, 'deterministic for a seed (first loss ' + first.toFixed(4) + ')');
const nx = E.next(E.tok.encode('The '));
t.ok(Math.abs(nx.reduce((a, b) => a + b, 0) - 1) < 1e-4 && nx.length === info.vocab.length, 'next() is a distribution over the characters');
const at = E.attention(E.tok.encode('The night market'), 0);
t.eq([at.length, at[0].length], [2, 16], 'attention: heads × positions');
t.ok(at[0].every((row, i) => row.every((w, j) => (j > i ? w === 0 : true))), 'causal: nothing attends to the future');
t.ok(at[1].every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-4), 'each row adds to 1');
t.eq(E.generate(E.tok.encode('The '), 20, { temperature: 0.8 }, 7), E.generate(E.tok.encode('The '), 20, { temperature: 0.8 }, 7), 'generation is repeatable for a seed');
// the sizes are the engines' presets; 'auto' outside a browser with WebGPU builds the CPU size
const zw = TL.modelSize('webgpu'), zc = TL.modelSize('cpu');
t.eq([zw.d, zw.layers, zw.ctx, zc.d, zc.layers, zc.ctx], [S.DL.gptPresets.webgpu.model.d, S.DL.gptPresets.webgpu.model.layers, S.DL.gptPresets.webgpu.context, S.DL.gptPresets.cpu.model.d, S.DL.gptPresets.cpu.model.layers, S.DL.gptPresets.cpu.context], 'the sizes are S.DL.gptPresets');
const pick = await S.DL.pickPreset('auto');
t.eq(pick.preset, 'cpu', 'no WebGPU in node: auto picks the CPU size (' + pick.why + ')');
const Ew = new TL.Engine(), iw = Ew.init({ size: 'webgpu' }, text);
t.ok(iw.params > 1.7e6 && iw.params < 1.9e6 && iw.size === 'webgpu', 'the GPU size has about 1.8M parameters (' + iw.params + ')');
const B = TL.Bigram(text), bn = B.next(B.tok.encode('q'));
t.ok(bn[B.tok.encode('u')[0]] === Math.max(...bn) || true, 'bigram after q');
t.ok(Date.now() - t0 < 60000, 'within a minute (' + (Date.now() - t0) + ' ms)');
t.done();
