// Part 12, the whole transformer block traced one position at a time through the trained tiny transformer (norm,
// attention, add, norm, feed-forward, add), checked against the engine's own block, and then what the last block
// hands to the output layer. Second topic: the same block as an encoder (every position sees every other) or a
// decoder (causal), and how an encoder–decoder joins the two with cross-attention.
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml;
  const BX = 170, BW = 150;
  /** the block drawn as a diagram: the stream on the left, the two branches on the right; lit = the part in focus */
  const diagram = (lit, L) => {
    const box = (key, y, h, text, cls) => ({ key: 'bd:' + key, kind: 'cell', x: BX, y, w: BW, h, rx: 6, text, size: 13, cls: 'p12box ' + cls + (lit === key ? ' act' : '') });
    const items = [
      { key: 'bd:in', kind: 'cell', x: 40, y: 18, w: 120, h: 30, rx: 6, text: 'x  (10 × 16)', size: 13, cls: 'p12box ' + (lit === 'in' ? 'act' : '') },
      { key: 'bd:out', kind: 'cell', x: 40, y: 470, w: 120, h: 30, rx: 6, text: L ? 'to block ' + L : 'to the output', size: 13, cls: 'p12box ' + (lit === 'out' ? 'act' : '') },
      box('ln1', 82, 30, 'LayerNorm₁', 'norm'), box('attn', 136, 44, 'multi-head attention', 'attn'), box('ln2', 286, 30, 'LayerNorm₂', 'norm'), box('ffn', 340, 44, 'feed-forward', 'ffn'),
      { key: 'bd:s1', kind: 'wire', pts: [[100, 48], [100, 219]], cls: 'p12res', head: true },
      { key: 'bd:s2', kind: 'wire', pts: [[100, 241], [100, 423]], cls: 'p12res', head: true },
      { key: 'bd:s3', kind: 'wire', pts: [[100, 445], [100, 470]], cls: 'p12res', head: true },
      { key: 'bd:b1', kind: 'wire', pts: [[100, 62], [245, 62], [245, 82]], cls: 'p12side', head: true },
      { key: 'bd:b2', kind: 'wire', pts: [[245, 112], [245, 136]], cls: 'p12side', head: true },
      { key: 'bd:b3', kind: 'wire', pts: [[245, 180], [245, 230], [111, 230]], cls: 'p12side', head: true },
      { key: 'bd:b4', kind: 'wire', pts: [[100, 262], [245, 262], [245, 286]], cls: 'p12side', head: true },
      { key: 'bd:b5', kind: 'wire', pts: [[245, 316], [245, 340]], cls: 'p12side', head: true },
      { key: 'bd:b6', kind: 'wire', pts: [[245, 384], [245, 434], [111, 434]], cls: 'p12side', head: true },
      { key: 'bd:a1', kind: 'op', x: 100, y: 230, r: 11, op: 'add', cls: lit === 'add1' ? 'act' : '' },
      { key: 'bd:a2', kind: 'op', x: 100, y: 434, r: 11, op: 'add', cls: lit === 'add2' ? 'act' : '' },
    ];
    return items;
  };
  const fmt1 = (v) => H().fx(v, 1).replace(/^0\./, '.').replace(/^−0\./, '−.');
  /** a 16-number vector as two rows of eight heat cells */
  const vecStrip = (key, y, v, title, max) => P.mat(key, 372, y, [v.slice(0, 8), v.slice(8)], { rows: false, cols: false, cw: 46, ch: 24, gap: 2, d: 2, size: 12, max, title }).items;

  S.topic({
    id: 'transformer-block', family: 'p12-block', order: 30, title: 'The full transformer block, traced',
    blurb: 'A transformer block is two sublayers on a residual stream: normalise, attend, add; normalise, feed forward, add. Its input and output have the same shape, so blocks stack. Follow one position of the trained tiny transformer through block 0 or 1, with the vector shown at every stage, then see what the last block hands to the output layer: a score for every token that could come next.',
    inputs: [
      { id: 'task', kind: 'choice', label: 'Trained model', def: 'reverse', options: P.TASK_OPTS },
      { id: 'digits', kind: 'string', label: 'Five digits (0 to 7)', def: '27405', alphabet: '01234567', min: 5, max: 5 },
      { id: 'pos', kind: 'number', label: 'Position to follow (0 to 9)', def: 7, min: 0, max: 9 },
      { id: 'layer', kind: 'choice', label: 'Block', def: '0', options: [{ value: '0', label: 'block 0' }, { value: '1', label: 'block 1' }] },
    ],
    presets: [
      { label: 'Reverse, block 0', values: { task: 'reverse', digits: '27405', pos: 7, layer: '0' } },
      { label: 'Reverse, block 1', values: { task: 'reverse', digits: '27405', pos: 7, layer: '1' } },
      { label: 'Sort, the separator', values: { task: 'sort', digits: '62517', pos: 5, layer: '1' } },
      { label: 'Copy', values: { task: 'copy', digits: '04471', pos: 8, layer: '0' } },
    ],
    counters: [{ id: 'stage', label: 'stages passed' }],
    run(inp) {
      const T = S.T, F = H().frames(['stage']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), ids = P.seqOf(inp.task, x), tr = P.trace(net, ids), l = +inp.layer, k = inp.pos, b = tr.blocks[l], n = ids.length;
      const L = net.blocks.length, nextL = l + 1 < L ? l + 1 : 0, all = [b.inp[k], b.ln1[k], b.att[k], b.mid[k], b.ln2[k], b.ff[k], b.out[k]], max = Math.max(...all.flat().map(Math.abs));
      const D = (lit) => diagram(lit, nextL);
      const toks = ids.map(P.tok);
      const shown = (list) => list.flatMap(([key, y, v, t]) => vecStrip(key, y, v, t, max));
      const ln = (v) => P.norm(v);
      const s0 = ['v0', 40, b.inp[k], 'x at position ' + k + ' (“' + toks[k] + '”)'];
      F.count('stage');
      F.push([...D('in'), ...shown([s0])], `Block ${q.n(l)} takes the whole ${q.n(n)} × ${q.n(16)} matrix; follow position ${q.n(k)} (“${P.qtok(q, ids[k])}”). It enters with length ${q(ln(b.inp[k]), 2)}.`, { phase: 'in', l, n, d: 16, k, len: ln(b.inp[k]), tok: ids[k] });
      F.count('stage');
      F.push([...D('ln1'), ...shown([s0, ['v1', 130, b.ln1[k], 'after LayerNorm₁']])], `First a normalised copy is made (the stream itself is left alone): mean about zero, spread about one, then scaled by the learned γ and shifted by β. Its length is now ${q(ln(b.ln1[k]), 2)}.`, { phase: 'ln1', len: ln(b.ln1[k]) });
      const w0 = b.w[0][k], w1 = b.w[1][k], j0 = P.argmax(w0), j1 = P.argmax(w1);
      const wrow = (key, y, w, title) => P.mat(key, 372, y, [w], { rows: false, cols: toks, cw: 30, ch: 22, gap: 2, scheme: 'seq', max: 1, fmt: (v) => (v < 0.05 ? '' : fmt1(v)), size: 12, title }).items;
      F.count('stage');
      F.push([...D('attn'), ...shown([s0, ['v2', 330, b.att[k], 'attention output (after Wo)']]), ...wrow('w0', 140, w0, 'head 0 weights'), ...wrow('w1', 236, w1, 'head 1 weights')],
        `Attention: position ${q.n(k)} queries every position up to itself. Head ${q.n(0)} gives most weight to position ${q.n(j0)} (${q(w0[j0], 2)}), head ${q.n(1)} to position ${q.n(j1)} (${q(w1[j1], 2)}). The two heads’ outputs are joined and multiplied by Wo into a vector of length ${q(ln(b.att[k]), 2)}.`, { phase: 'attn', h0: 0, h1: 1, k, j0, j1, a0: w0[j0], a1: w1[j1], len: ln(b.att[k]) });
      F.count('stage');
      F.push([...D('add1'), ...shown([s0, ['v2', 130, b.att[k], '+ attention output'], ['v3', 220, b.mid[k], '= the stream after attention']])], `The attention output is added to the stream: length ${q(ln(b.inp[k]), 2)} plus a vector of length ${q(ln(b.att[k]), 2)} gives ${q(ln(b.mid[k]), 2)}. Whatever attention gathered from the other positions is now part of position ${q.n(k)}.`, { phase: 'add1', a: ln(b.inp[k]), b: ln(b.att[k]), c: ln(b.mid[k]), k });
      F.count('stage');
      F.push([...D('ln2'), ...shown([['v3', 40, b.mid[k], 'the stream after attention'], ['v4', 130, b.ln2[k], 'after LayerNorm₂']])], `A second normalised copy, with its own γ and β, feeds the feed-forward layer: length ${q(ln(b.ln2[k]), 2)}.`, { phase: 'ln2', len: ln(b.ln2[k]) });
      F.count('stage');
      F.push([...D('ffn'), ...shown([['v3', 40, b.mid[k], 'the stream after attention'], ['v4', 130, b.ln2[k], 'after LayerNorm₂'], ['v5', 220, b.ff[k], 'feed-forward output (16 → 32 → 16)']])], `The feed-forward layer works on this position alone: widen to ${q.n(b.pre[k].length)}, GELU, back to ${q.n(16)}. Its output has length ${q(ln(b.ff[k]), 2)}.`, { phase: 'ffn', ff: b.pre[k].length, d: 16, len: ln(b.ff[k]) });
      // the engine's own block on the same input, compared
      const xin = T.tensor([b.inp]), eng = T.noGrad(() => net.blocks[l].forward(xin)).toArray()[0], diff = Math.max(...eng.flatMap((r, i) => r.map((v, j) => Math.abs(v - b.out[i][j]))));
      const params = net.blocks[l].numParams();
      F.count('stage');
      F.push([...D('add2'), ...shown([['v3', 40, b.mid[k], 'the stream after attention'], ['v5', 130, b.ff[k], '+ feed-forward output'], ['v6', 220, b.out[k], '= the block’s output']])], `Added to the stream again: the block’s output at position ${q.n(k)} has length ${q(ln(b.out[k]), 2)}. Run as one call, the engine’s block gives ${diff ? 'the same' : 'exactly the same'} ${q.n(n)} × ${q.n(16)} output${diff ? ' to within ' + q.sg(diff, 2) : ''}. The block holds ${q.n(params)} weights, and because it maps a ${q.n(n)} × ${q.n(16)} matrix to another of the same shape, blocks stack.`, { phase: 'add2', k, len: ln(b.out[k]), diff, params, n, d: 16 });
      // the output layer
      const pr = tr.probs[k], best = P.argmax(pr), full = x.concat([P.SEP], P.answer(inp.task, x)), want = full[k + 1];
      const bars = pr.map((p, j) => [{ key: 'po:b' + j, kind: 'cell', x: P.r2(380 + j * 44), y: P.r2(300 - 160 * p), w: 32, h: P.r2(Math.max(1, 160 * p)), rx: 3, text: '', cls: 'p12bar ' + (j === best ? '' : 'soft'), audit: 'ignore' }, H().text('po:t' + j, 396 + j * 44, 322, ['0', '1', '2', '3', '4', '5', '6', '7', '='][j], 'tick', 13, 'middle'), H().text('po:v' + j, 396 + j * 44, P.r2(292 - 160 * p), p >= 0.01 ? H().pct(p) : '', 'tick', 12, 'middle')]).flat();
      F.count('stage');
      F.push([...diagram('out', 0), H().text('po:h', 380, 110, 'after both blocks: chances for the token after position ' + k, 'strong', 13), { key: 'po:base', kind: 'line', x1: 376, y1: 301, x2: 776, y2: 301, cls: 'faint', audit: 'ignore' }, ...bars],
        `After the last block, a final layer norm and a product with the embedding matrix (the same one used for the input) give a score for each of the ${q.n(9)} tokens, and a softmax turns them into chances. At position ${q.n(k)} the model gives “${P.qtok(q, best)}” ${q.pct(pr[best])}` + (want === undefined ? '.' : want === best ? `, the right next token.` : `; the right next token was “${P.qtok(q, want)}”.`), { phase: 'output', k, best, p: pr[best], want: want === undefined ? -1 : want, vocab: 9 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'One pre-norm block', text: ['a = MultiHeadAttention(LN₁(x))   (causal here)', 'x = x + a', 'f = W₂ GELU(W₁ LN₂(x) + b₁) + b₂', 'x = x + f', 'shape in = shape out: (positions × d)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'ffn', once: true, pick: 'option',
      ask: 'Will the token this position predicts next be the right one?',
      options: ['yes', 'no'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'output').meta; return m.want === m.best ? 'yes' : 'no'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'output').say,
    }],
    quiz: [
      { q: 'In a pre-norm block, where does layer normalisation sit?', options: ['at the start of each branch, before attention and before the feed-forward layer', 'only at the very end of the model', 'after each addition, on the stream', 'nowhere'], answer: 0, why: 'x + Attention(LN(x)): the stream itself is not normalised.' },
      { q: 'Why can transformer blocks be stacked any number of times?', options: ['each maps (positions × d) to (positions × d)', 'each halves the width', 'they share all their weights', 'the softmax rescales them'], answer: 0, why: 'Same shape in and out.' },
      { q: 'Which part of the block moves information between positions?', options: ['attention', 'the feed-forward layer', 'layer normalisation', 'the residual addition'], answer: 0, why: 'Every other part works on each position separately.' },
      { q: 'In this model the output scores come from multiplying by the embedding matrix. This is called…', options: ['tying the input and output embeddings', 'attention', 'a residual', 'dropout'], answer: 0, why: 'One matrix serves both ends and saves weights.' },
    ],
  });

  S.topic({
    id: 'encoder-decoder', family: 'p12-block', order: 40, title: 'Encoder versus decoder',
    blurb: 'The same block makes two kinds of model. In an encoder every position attends to every other, so each output depends on the whole input: good for understanding a finished text. In a decoder the causal mask lets each position see only the past, so it can generate one token after another. An encoder–decoder model uses both: the decoder adds cross-attention, its queries reading the encoder’s keys and values.',
    inputs: [
      { id: 'task', kind: 'choice', label: 'Trained model', def: 'reverse', options: P.TASK_OPTS },
      { id: 'digits', kind: 'string', label: 'Five digits (0 to 7)', def: '27405', alphabet: '01234567', min: 5, max: 5 },
      { id: 'head', kind: 'choice', label: 'Head of layer 0', def: '0', options: [{ value: '0', label: 'head 0' }, { value: '1', label: 'head 1' }] },
      { id: 'change', kind: 'number', label: 'New value for the last token', def: 3, min: 0, max: 7 },
    ],
    presets: [
      { label: 'Reverse, head 0', values: { task: 'reverse', digits: '27405', head: '0', change: 3 } },
      { label: 'Reverse, head 1', values: { task: 'reverse', digits: '27405', head: '1', change: 6 } },
      { label: 'Sort', values: { task: 'sort', digits: '62517', head: '0', change: 0 } },
      { label: 'Copy', values: { task: 'copy', digits: '04471', head: '1', change: 5 } },
    ],
    counters: [{ id: 'runs', label: 'passes' }],
    run(inp) {
      const T = S.T, F = H().frames(['runs']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), ids = P.seqOf(inp.task, x), n = ids.length, hd = +inp.head, blk = net.blocks[0], toks = ids.map(P.tok);
      const run = (seq, causal) => T.noGrad(() => { const h = T.add(net.emb.forward(T.tensor([seq])), T.slice(net.pos.P, [0, n], null)), y = blk.attn.forward(blk.ln1.forward(h), { causal }); return { out: y.toArray()[0], w: blk.attn.weights.toArray()[0][hd] }; });
      const dec = run(ids, true), enc = run(ids, false), last = n - 1;
      const ids2 = ids.slice(0, last).concat([inp.change === ids[last] ? (inp.change + 1) % 8 : inp.change]);
      const dec2 = run(ids2, true), enc2 = run(ids2, false);
      const change = (a, b) => a.map((r, i) => Math.sqrt(r.reduce((s, v, j) => s + (v - b[i][j]) ** 2, 0)));
      const cD = change(dec.out, dec2.out), cE = change(enc.out, enc2.out);
      const map = (key, x0, w, title) => P.mat(key, x0, 74, w, { rows: toks.map((t, i) => i + ' ' + t), cols: toks, cw: 26, ch: 24, scheme: 'seq', max: 1, fmt: (v) => (v < 0.05 ? '' : H().fx(v, 1).replace(/^0\./, '.')), size: 11, title }).items;
      const future = enc.w.map((r, i) => r.slice(i + 1).reduce((a, b) => a + b, 0)), r0 = 0;
      F.count('runs');
      F.push([...map('ed', 70, dec.w, 'decoder: causal (as trained)')], `The trained model is a decoder: in the first layer, head ${q.n(hd)}, every row is zero to the right of the diagonal. Position ${q.n(r0)} can only read itself, and position ${q.n(last)} reads all ${q.n(n)}. That is what lets it be trained to predict every next token at once.`, { phase: 'dec', hd, r0, last, n });
      F.count('runs');
      F.push([...map('ed', 70, dec.w, 'decoder: causal (as trained)'), ...map('ee', 440, enc.w, 'encoder: no mask, same weights')], `Remove the mask and the same weights make an encoder layer: now position ${q.n(r0)} spreads ${q(future[r0], 2)} of its weight over the positions after it. (This model was trained with the mask, so its encoder pattern is only for comparison; a real encoder is trained without one.)`, { phase: 'enc', r0, fut: future[r0] });
      const chg = (key, y, v, title) => P.mat(key, 120, y, [v], { rows: false, cols: toks.map((_, i) => String(i)), cw: 44, ch: 26, scheme: 'seq', max: Math.max(...cE, ...cD, 1e-6), d: 2, size: 11, title }).items;
      F.count('runs');
      F.push([...chg('cd', 120, cD, 'decoder: how much each output moves'), ...chg('ce', 260, cE, 'encoder: how much each output moves'), ...P.strip('cs', 120, 380, ids2.map(P.tok), { size: 13, gap: 8, cls: (i) => (i === last ? 'p12ask' : '') }).items, H().text('cs:h', 120, 368, 'the input with its last token changed', 'strong', 13)],
        `Change only the last token (to “${P.qtok(q, ids2[last])}”) and run both again. In the decoder the outputs at positions ${q.n(0)} to ${q.n(last - 1)} do not move at all${Math.max(...cD.slice(0, last)) ? ' (beyond rounding of ' + q.sg(Math.max(...cD.slice(0, last)), 2) + ')' : ''}. In the encoder every position moves, position ${q.n(0)} by ${q(cE[0], 3)}: each output depends on the whole sequence.`, { phase: 'change', tok: ids2[last], last, first: 0, upto: last - 1, maxD: Math.max(...cD.slice(0, last)), e0: cE[0] });
      // cross-attention: queries from the decoder side, keys and values from the encoder side
      const ne = 5, nd = n - ne, src = ids.slice(0, ne), tgt = ids.slice(ne);
      const cross = T.noGrad(() => {
        const he = blk.ln1.forward(T.add(net.emb.forward(T.tensor([src])), T.slice(net.pos.P, [0, ne], null))), hdv = blk.ln1.forward(T.add(net.emb.forward(T.tensor([tgt])), T.slice(net.pos.P, [ne, n], null)));
        const a = blk.attn, qq = a.split(a.Wq.forward(hdv), 1, nd), kk = a.split(a.Wk.forward(he), 1, ne), vv = a.split(a.Wv.forward(he), 1, ne);
        return T.attention(qq, kk, vv).attn.weights.toArray()[0][hd];
      });
      const boxes = [
        { key: 'xc:e', kind: 'cell', x: 60, y: 300, w: 220, h: 60, rx: 8, text: 'encoder: reads the input', size: 13, cls: 'p12box attn' },
        { key: 'xc:d', kind: 'cell', x: 360, y: 300, w: 260, h: 60, rx: 8, text: 'decoder: causal + cross-attention', size: 13, cls: 'p12box ffn' },
        { key: 'xc:w', kind: 'wire', pts: [[280, 330], [360, 330]], cls: 'p12side', head: true },
        H().text('xc:wl', 320, 318, 'K, V', 'p12hint', 12, 'middle'),
      ];
      F.count('runs');
      F.push([...P.mat('xm', 120, 80, cross, { rows: tgt.map(P.tok).map((t, i) => (ne + i) + ' ' + t), cols: src.map(P.tok), cw: 40, ch: 26, scheme: 'seq', max: 1, d: 2, size: 11, title: 'cross-attention: ' + nd + ' decoder queries × ' + ne + ' encoder keys' }).items, ...boxes],
        `An encoder–decoder model, as used for translation, keeps the input and the output apart. The encoder reads the ${P.qpl(q, ne, 'input token')} with no mask; the decoder generates the ${P.qpl(q, nd, 'output token')} causally and, in an extra cross-attention layer, its queries read the encoder’s keys and values: a ${q.n(nd)} × ${q.n(ne)} weight matrix in which each row adds up to 1. (Our model has no such layer: its first-layer weights are borrowed here only to show the shape.)`, { phase: 'cross', ne, nd });
      const tab = [['', 'sees', 'used for'], ['encoder only', 'the whole input', 'classifying, tagging, search'], ['decoder only', 'the past only', 'generating text, like this model'], ['encoder–decoder', 'input fully, output causally', 'translating, summarising']];
      const items = tab.flatMap((r, i) => r.map((t, j) => ({ key: 'tb:' + i + '_' + j, kind: 'cell', x: [60, 220, 450][j], y: 120 + i * 50, w: [150, 220, 300][j], h: 40, rx: 4, text: t, size: 13, align: 'left', cls: i === 0 ? 'p12box norm' : i === 2 ? 'p12box act' : 'p12box' })));
      F.push(items, `In short: an encoder looks both ways and understands; a decoder looks back and generates; an encoder–decoder does one then the other. The model in these topics is decoder-only, ${q.n(net.blocks.length)} causal blocks, as are the large language models of the next part.`, { phase: 'table', blocks: net.blocks.length });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Three ways to use the block', text: ['encoder: mask = all ones (sees everything)', 'decoder: causal mask (sees the past)', 'cross-attention: Q from the decoder,', '                 K, V from the encoder'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'enc', once: true, pick: 'option',
      ask: 'If only the last token changes, will the decoder’s output at position 0 change?',
      options: ['no', 'yes, a little', 'yes, a lot'],
      answer: 'no',
      explain: 'Under the causal mask position 0 cannot see any later token.',
    }],
    quiz: [
      { q: 'Which kind of model can generate text one token at a time?', options: ['a decoder', 'an encoder only', 'neither', 'only an encoder–decoder'], answer: 0, why: 'Its causal mask means each position depends only on the past.' },
      { q: 'In cross-attention, the queries come from…', options: ['the decoder', 'the encoder', 'both, averaged', 'the output layer'], answer: 0, why: 'The decoder asks; the encoder’s keys and values answer.' },
      { q: 'A model that labels each word of a finished sentence (noun, verb…) is best built as…', options: ['an encoder', 'a decoder', 'neither', 'a feed-forward layer only'], answer: 0, why: 'Each label can use the words on both sides.' },
      { q: 'The cross-attention weights for 7 output tokens over 12 input tokens form a matrix of shape…', options: ['7 × 12', '12 × 12', '7 × 7', '19 × 19'], answer: 0, why: 'One row per decoder query, one column per encoder key.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
