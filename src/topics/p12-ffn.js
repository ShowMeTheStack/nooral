// Part 12, the two other pieces of a transformer block, in the trained tiny transformer: the position-wise
// feed-forward layer (widen, GELU, narrow back), and layer normalisation with the residual stream it sits on.
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml, C = () => S.Kit.curve;
  /** a point on a plot; points may sit on one another */
  const dotOk = (...a) => Object.assign(C().dot(...a), { audit: 'overlap-ok', point: false });
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  const common = [
    { id: 'task', kind: 'choice', label: 'Trained model', def: 'reverse', options: null },
    { id: 'digits', kind: 'string', label: 'Five digits (0 to 7)', def: '27405', alphabet: '01234567', min: 5, max: 5 },
    { id: 'pos', kind: 'number', label: 'Position to follow (0 to 9)', def: 6, min: 0, max: 9 },
    { id: 'layer', kind: 'choice', label: 'Block', def: '0', options: [{ value: '0', label: 'block 0' }, { value: '1', label: 'block 1' }] },
  ];
  const inputs = () => common.map((i) => (i.id === 'task' ? Object.assign({}, i, { options: P.TASK_OPTS }) : Object.assign({}, i)));
  /** one row as a strip of heat cells with a caption */
  const row = (key, x, y, v, title, o) => P.mat(key, x, y, [v], Object.assign({ rows: false, cols: false, cw: 20, ch: 22, fmt: null, title }, o || {})).items;

  S.topic({
    id: 'transformer-ffn', family: 'p12-block', order: 10, title: 'The feed-forward layer',
    blurb: 'After attention has mixed information between positions, each position is processed on its own by a small two-layer network: widen the vector (here from 16 to 32 numbers), apply GELU, and narrow it back. The same weights are used at every position. Attention moves information around; the feed-forward layer transforms it where it is, and it holds most of a transformer’s weights.',
    inputs: inputs(),
    presets: [
      { label: 'Reverse, block 0', values: { task: 'reverse', digits: '27405', pos: 6, layer: '0' } },
      { label: 'Reverse, block 1', values: { task: 'reverse', digits: '27405', pos: 6, layer: '1' } },
      { label: 'Sort, the separator', values: { task: 'sort', digits: '62517', pos: 5, layer: '0' } },
      { label: 'Copy, first digit', values: { task: 'copy', digits: '04471', pos: 0, layer: '1' } },
    ],
    counters: [{ id: 'steps', label: 'steps' }],
    run(inp) {
      const F = H().frames(['steps']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), ids = P.seqOf(inp.task, x), tr = P.trace(net, ids), l = +inp.layer, k = inp.pos, b = tr.blocks[l];
      const blk = net.blocks[l], d = b.ln2[0].length, ff = b.pre[0].length, inV = b.ln2[k], pre = b.pre[k], act = b.act[k], out = b.ff[k];
      const mx = (v) => Math.max(...v.map(Math.abs));
      const lin = row('fi', 70, 70, inV, 'input at position ' + k + ' (after layer norm): ' + d + ' numbers', { max: mx(inV) });
      const hid = row('fh', 70, 160, pre, 'hidden: W₁ x + b₁, ' + ff + ' numbers', { max: mx(pre) });
      const gel = row('fg', 70, 250, act, 'after GELU', { max: mx(pre) });
      const outR = row('fo', 70, 340, out, 'output: W₂ h + b₂, back to ' + d, { max: mx(out) });
      const neg = pre.filter((v) => v < 0).length, big = P.argmax(pre), low = pre.indexOf(Math.min(...pre));
      F.count('steps');
      F.push([...lin], `Position ${q.n(k)} (“${P.qtok(q, ids[k])}”) arrives at block ${q.n(l)}’s feed-forward layer as ${P.qpl(q, d, 'number')}, already mixed by attention and normalised. Its largest entry is ${q(Math.max(...inV), 2)}.`, { phase: 'in', k, l, d, maxIn: Math.max(...inV), tok: ids[k] });
      F.count('steps');
      F.push([...lin, ...hid], `The first layer widens it to ${q.n(ff)} hidden numbers, each a weighted sum of all ${q.n(d)} inputs plus a bias: ${q.n(neg)} of them come out negative. The largest is unit ${q.n(big)} at ${q(pre[big], 2)}, the most negative unit ${q.n(low)} at ${q(pre[low], 2)}.`, { phase: 'widen', ff, d, neg, big, low, pb: pre[big], pl: pre[low] });
      const A = C().axes({ prefix: 'fa', x: 470, y: 380, w: 280, h: 110, xr: [-3, 3], yr: [-0.5, 3], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [0, 1, 2, 3], xname: 'z', yname: 'GELU(z)' });
      const curve = C().line(A, 'fa:c', Array.from({ length: 61 }, (_, i) => { const z = -3 + i / 10; return [z, gelu(z)]; }), 'p12a');
      const dots = pre.map((z, j) => (Math.abs(z) <= 3 ? dotOk(A, 'fa:d' + j, z, gelu(z), 'p12c', 3.2) : null)).filter(Boolean);
      F.count('steps');
      F.push([...lin, ...hid, ...gel, ...A.items, curve, ...dots], `GELU keeps positive numbers almost as they are and squeezes negative ones towards 0 (it dips only to about −0.17): unit ${q.n(big)} goes from ${q(pre[big], 2)} to ${q(act[big], 2)}, unit ${q.n(low)} from ${q(pre[low], 2)} to ${q(act[low], 3)}. Only the units pushed well above zero pass much on: this is where the layer chooses which features to use.`, { phase: 'gelu', big, low, pb: pre[big], ab: act[big], pl: pre[low], al: act[low] });
      F.count('steps');
      F.push([...lin, ...hid, ...gel, ...outR, ...A.items, curve, ...dots], `The second layer narrows the ${q.n(ff)} back to ${q.n(d)}. The output, of size ${q(P.norm(out), 2)} (its length), is added to the residual stream, which was of size ${q(P.norm(b.mid[k]), 2)} at this position.`, { phase: 'narrow', ff, d, outNorm: P.norm(out), midNorm: P.norm(b.mid[k]) });
      // every position, the same weights
      const n = ids.length, active = b.act.map((r) => r.filter((v) => v > 0.1).length);
      const allH = P.mat('fx', 110, 70, b.act, { rows: ids.map((t, i) => i + ' ' + P.tok(t)), cols: false, cw: 18, ch: 24, gap: 2, fmt: null, scheme: 'div', max: mx(b.act.flat()), title: 'GELU outputs at every position (rows) for the ' + ff + ' hidden units', rowCls: (i) => (i === k ? 'strong' : '') }).items;
      const pff = blk.ff1.W.size + blk.ff1.b.size + blk.ff2.W.size + blk.ff2.b.size, pat = ['Wq', 'Wk', 'Wv', 'Wo'].reduce((s, w) => s + blk.attn[w].W.size + blk.attn[w].b.size, 0);
      const most = active.indexOf(Math.max(...active));
      F.count('steps');
      F.push([...allH], `The same two layers run at all ${P.qpl(q, n, 'position')} at once, with no exchange between them. Each position lights its own set of units: position ${q.n(most)} has the most above ${q(0.1, 1)}, ${q.n(active[most])} of ${q.n(ff)}. In this block the feed-forward layer has ${q.n(pff)} weights against attention’s ${q.n(pat)}; in large models it is usually about two thirds of all weights.`, { phase: 'all', n, most, act: active[most], ff, pff, pat, thr: 0.1 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Feed-forward layer (per position)', text: ['h = GELU(x W₁ + b₁)     (d → 4d, here 16 → 32)', 'y = h W₂ + b₂          (back to d)', 'GELU(z) = z · Φ(z) ≈ ½ z (1 + tanh(√(2/π)(z + 0.044715 z³)))', 'the same W₁, W₂ at every position'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'widen', once: true, pick: 'option',
      ask: 'After GELU, what happens to the most negative hidden number?',
      options: ['it ends up close to 0', 'it stays as negative', 'it becomes large and positive'],
      answer: 'it ends up close to 0',
      explain: 'GELU(z) is near 0 for clearly negative z; its lowest value is only about −0.17.',
    }],
    quiz: [
      { q: 'In a transformer with width 512 and a hidden width of 2048, the feed-forward layer has about how many weights?', options: ['2 × 512 × 2048 ≈ 2.1 million', '512 × 512', '2048', '512 + 2048'], answer: 0, why: 'Two matrices, 512 × 2048 and 2048 × 512 (plus small biases).' },
      { q: 'Does the feed-forward layer let positions exchange information?', options: ['no: each position is processed on its own', 'yes, like attention', 'only neighbouring positions', 'only in the decoder'], answer: 0, why: 'It is applied to every row separately with the same weights.' },
      { q: 'Why is there a nonlinearity (GELU) between the two layers?', options: ['two linear layers in a row would collapse into one', 'to keep the numbers positive', 'to normalise the vector', 'to make it causal'], answer: 0, why: 'A product of matrices is again one matrix.' },
      { q: 'GELU differs from ReLU mainly in that it…', options: ['is smooth and lets small negative values through slightly', 'is always positive', 'is linear', 'saturates at 1'], answer: 0, why: 'It curves gently around zero instead of bending sharply.' },
    ],
  });

  S.topic({
    id: 'transformer-residual-norm', family: 'p12-block', order: 20, title: 'Residual connections and layer normalisation',
    blurb: 'Every sublayer of a transformer reads a normalised copy of the running vector and adds its result back: x + attention(norm(x)), then x + feed-forward(norm(x)). The running vector, the residual stream, carries everything from the embedding to the output, so gradients have a direct path back. Layer normalisation rescales each vector to mean 0 and spread 1 before a sublayer sees it, then applies a learned scale and shift. Both are traced here in the trained tiny transformer.',
    inputs: inputs(),
    presets: [
      { label: 'Reverse, block 0', values: { task: 'reverse', digits: '27405', pos: 6, layer: '0' } },
      { label: 'Reverse, block 1', values: { task: 'reverse', digits: '27405', pos: 6, layer: '1' } },
      { label: 'Sort', values: { task: 'sort', digits: '62517', pos: 7, layer: '0' } },
      { label: 'Copy, the separator', values: { task: 'copy', digits: '04471', pos: 5, layer: '1' } },
    ],
    counters: [{ id: 'steps', label: 'steps' }],
    run(inp) {
      const F = H().frames(['steps']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), ids = P.seqOf(inp.task, x), tr = P.trace(net, ids), l = +inp.layer, k = inp.pos, b = tr.blocks[l];
      const blk = net.blocks[l], v = b.inp[k], d = v.length, mean = v.reduce((a, c) => a + c, 0) / d, va = v.reduce((a, c) => a + (c - mean) ** 2, 0) / d, sd = Math.sqrt(va + 1e-5);
      const cen = v.map((c) => c - mean), z = cen.map((c) => c / sd), g = Array.from(blk.ln1.gamma.values()), be = Array.from(blk.ln1.beta.values()), y = b.ln1[k];
      const zm = z.reduce((a, c) => a + c, 0) / d, zs = Math.sqrt(z.reduce((a, c) => a + (c - zm) ** 2, 0) / d);
      const mxv = Math.max(...v.map(Math.abs), ...z.map(Math.abs), ...y.map(Math.abs));
      const strip = (key, yy, vals, title, lit) => P.mat(key, 70, yy, [vals], { rows: false, cols: false, cw: 42, ch: 24, d: 2, size: 11, max: mxv, title, cls: (i, j) => (j === lit ? 'act' : '') }).items;
      const rows = [['rx', 70, v, 'x: the residual stream at position ' + k + ' entering block ' + l], ['rc', 150, cen, 'x − mean'], ['rz', 230, z, '÷ √(variance + ε)'], ['ry', 310, y, '× γ + β: what attention reads']];
      const upto = (m, lit) => rows.slice(0, m).flatMap((r) => strip(r[0], r[1], r[2], r[3], lit));
      F.count('steps');
      F.push(upto(1), `Position ${q.n(k)} enters block ${q.n(l)} as a vector of ${q.n(d)} numbers whose mean is ${q(mean, 3)} and whose spread (standard deviation) is ${q(Math.sqrt(va), 3)}. Its size depends on everything added to the stream so far.`, { phase: 'in', k, l, d, mean, sd: Math.sqrt(va) });
      F.count('steps');
      F.push(upto(3), `Layer normalisation subtracts the mean and divides by √(variance + ε), with ε = 10⁻⁵ to avoid dividing by zero. The result has mean ${q(zm, 3)} and spread ${q(zs, 3)}: whatever the stream looked like, the sublayer always sees numbers of the same scale.`, { phase: 'norm', zm, zs });
      F.count('steps');
      const gi = g.indexOf(Math.max(...g)), gl = g.indexOf(Math.min(...g));
      F.push(upto(4, g.indexOf(Math.max(...g))), `Then a learned scale γ and shift β per dimension give the layer back some freedom. Here γ runs from ${q(g[gl], 2)} (dimension ${q.n(gl)}) to ${q(g[gi], 2)} (dimension ${q.n(gi)}). Dimension ${q.n(gi)} (outlined): ${q(z[gi], 3)} × ${q(g[gi], 3)}${q.t(be[gi], 3)} = ${q(y[gi], 3)}.`, { phase: 'affine', gl, gi, gmin: g[gl], gmax: g[gi], z: z[gi], gam: g[gi], beta: be[gi], y: y[gi] });
      // the residual stream across the whole model
      const stages = [['embedding + position', tr.x0[k]]];
      tr.blocks.forEach((bb, i) => { stages.push(['+ attention ' + i, bb.mid[k]]); stages.push(['+ feed-forward ' + i, bb.out[k]]); });
      const deltas = [];
      tr.blocks.forEach((bb, i) => { deltas.push(P.norm(bb.att[k])); deltas.push(P.norm(bb.ff[k])); });
      const norms = stages.map((s) => P.norm(s[1])), top = Math.max(...norms, ...deltas);
      const sx = 300, sw = 330, items = [];
      // the stream drawn as a wire with the sublayers hanging off it
      items.push({ key: 'rs:w', kind: 'wire', pts: [[110, 80], [110, 470]], cls: 'p12res', audit: 'ignore' });
      stages.forEach((s, i) => {
        const yy = 92 + i * 76;
        items.push(H().text('rs:n' + i, 128, yy + 5, s[0], i ? 'legend' : 'strong', 13));
        items.push({ key: 'rs:b' + i, kind: 'cell', x: sx, y: yy - 10, w: P.r2(Math.max(2, (sw * norms[i]) / top)), h: 18, rx: 3, text: '', cls: 'p12bar', audit: 'ignore' });
        items.push(H().text('rs:v' + i, sx + (sw * norms[i]) / top + 8, yy + 4, H().fx(norms[i], 2), 'tick', 12));
        if (i) {
          items.push({ key: 'rs:d' + i, kind: 'cell', x: sx, y: yy - 36, w: P.r2(Math.max(2, (sw * deltas[i - 1]) / top)), h: 10, rx: 2, text: '', cls: 'p12bar alt', audit: 'ignore' });
          items.push(H().text('rs:dv' + i, sx + (sw * deltas[i - 1]) / top + 8, yy - 27, 'added ' + H().fx(deltas[i - 1], 2), 'p12hint', 12));
        }
        items.push({ key: 'rs:p' + i, kind: 'dot', x: 110, y: yy, r: 5, cls: 'mark p12a' });
      });
      items.push(H().text('rs:h', sx, 56, 'length of the stream vector (and of what each sublayer adds)', 'strong', 13));
      const grow = norms[norms.length - 1] / norms[0], dmax = deltas.indexOf(Math.max(...deltas));
      F.count('steps');
      F.push(items, `The residual stream at position ${q.n(k)}, from the embedding to the last block. Each sublayer adds its result rather than replacing the vector: lengths go ${H().list(norms.map((nv) => q(nv, 2)))}. The biggest addition, ${q(deltas[dmax], 2)}, comes from ${dmax % 2 ? 'the feed-forward layer' : 'attention'} in block ${q.n(Math.floor(dmax / 2))}.`, { phase: 'stream', k, norms, deltas, dmax, blockOf: Math.floor(dmax / 2) });
      F.count('steps');
      F.push(items, `Why add instead of replace? The output is x + f(x), so the derivative with respect to x is 1 + f′(x): there is always a direct path of 1 back to the embedding, and a deep stack trains as easily as a shallow one. Layer normalisation inside each branch (pre-norm, as here) keeps every sublayer’s input at the same scale, while the stream itself is free to grow, here by a factor of ${q(grow, 2)}.`, { phase: 'why', grow });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Layer norm and the residual stream', text: ['LN(x) = γ ⊙ (x − mean) / √(var + ε) + β', 'over the d numbers of one position', 'x ← x + Attention(LN₁(x))', 'x ← x + FFN(LN₂(x))'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'in', once: true, pick: 'option',
      ask: 'After subtracting the mean and dividing by the spread, what will the mean of the numbers be?',
      options: ['0', '1', 'the old mean', 'the old spread'],
      answer: '0',
      explain: 'Subtracting the mean leaves numbers that average to zero (up to rounding).',
    }],
    quiz: [
      { q: 'Layer normalisation computes its mean and variance over…', options: ['the numbers of one position', 'one number across the batch', 'the whole sequence', 'the weights'], answer: 0, why: 'Unlike batch normalisation, it works on each vector by itself.' },
      { q: 'In a pre-norm block, the attention sublayer’s output is…', options: ['added to the residual stream', 'used instead of the stream', 'normalised and thrown away', 'multiplied into the stream'], answer: 0, why: 'x ← x + Attention(LN(x)).' },
      { q: 'Why do residual connections help deep networks train?', options: ['the gradient has a direct path through the additions', 'they remove the need for weights', 'they make every layer linear', 'they shrink the vectors'], answer: 0, why: 'The derivative of x + f(x) contains the identity.' },
      { q: 'What are γ and β in layer normalisation?', options: ['a learned scale and shift per dimension', 'the mean and variance', 'the learning rate and momentum', 'fixed constants 1 and 0, never trained'], answer: 0, why: 'They let the model undo or reshape the normalisation where useful.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
