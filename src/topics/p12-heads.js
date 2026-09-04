// Part 12, multi-head attention in a trained model: the width split into heads, each head's weights on one input
// drawn as a heat map, where each answer position looks, and what is lost when one head is switched off (the model
// is run again over a set of new sequences with that head's output set to zero).
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml;
  /** n random digit sequences of length 5 (seeded), as full inputs for a task */
  P.probeSet = (task, n, seed) => { const g = S.T.rng(seed), out = []; for (let i = 0; i < n; i++) { const x = Array.from({ length: 5 }, () => g.int(8)); out.push({ x, ids: P.seqOf(task, x), want: P.answer(task, x) }); } return out; };
  /**
   * the share of answer digits predicted right (teacher forced: each answer position sees the true digits before it)
   * over a set, with one head of one layer switched off (off = {layer, head}) or none
   */
  P.ablate = (net, set, off) => S.T.noGrad(() => {
    const T = S.T, B = set.length, n = set[0].ids.length;
    let h = T.add(net.emb.forward(T.tensor(set.map((s) => s.ids))), T.slice(net.pos.P, [0, n], null));
    net.blocks.forEach((blk, l) => {
      const a = blk.attn, x1 = blk.ln1.forward(h), Hh = a.heads;
      const qq = a.split(a.Wq.forward(x1), B, n), kk = a.split(a.Wk.forward(x1), B, n), vv = a.split(a.Wv.forward(x1), B, n);
      let o = T.attention(qq, kk, vv, { causal: true });
      if (off && off.layer === l) { const keep = T.ones([1, Hh, 1, 1]); keep.data[off.head] = 0; o = T.mul(o, keep); }
      h = T.add(h, a.Wo.forward(T.reshape(T.permute(o, [0, 2, 1, 3]), [B, n, a.d])));
      h = T.add(h, blk.ff2.forward(T[blk.act](blk.ff1.forward(blk.ln2.forward(h)))));
    });
    const lg = T.matmul(net.lnf.forward(h), T.transpose(net.emb.W)), am = T.argmax(lg, -1).values();
    let right = 0, total = 0;
    set.forEach((s, b) => { for (let k = 0; k < 5; k++) { total++; if (am[b * n + 5 + k] === s.want[k]) right++; } });
    return right / total;
  });

  S.topic({
    id: 'attention-multihead', family: 'p12-attn', order: 60, title: 'Multi-head attention: what each head looks at',
    blurb: 'One attention pattern can answer only one question at a time, so a transformer splits its width into several heads, each with its own queries, keys and values, runs them side by side and joins their outputs. Here are the real heads of a tiny trained transformer (two layers, two heads each, sixteen numbers wide) as it reverses, copies or sorts five digits. Some heads turn out to do the essential pointing; switching them off shows which.',
    inputs: [
      { id: 'task', kind: 'choice', label: 'Trained model', def: 'reverse', options: P.TASK_OPTS },
      { id: 'digits', kind: 'string', label: 'Five digits (0 to 7)', def: '27405', alphabet: '01234567', min: 5, max: 5 },
      { id: 'row', kind: 'number', label: 'Answer position to follow (5 to 9)', def: 6, min: 5, max: 9 },
    ],
    presets: [
      { label: 'Reverse', values: { task: 'reverse', digits: '27405', row: 6 } },
      { label: 'Copy', values: { task: 'copy', digits: '27405', row: 7 } },
      { label: 'Sort', values: { task: 'sort', digits: '62517', row: 5 } },
      { label: 'Reverse, repeated digits', values: { task: 'reverse', digits: '33113', row: 8 } },
    ],
    counters: [{ id: 'heads', label: 'heads shown' }],
    run(inp) {
      const F = H().frames(['heads']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), ids = P.seqOf(inp.task, x), full = x.concat([P.SEP], P.answer(inp.task, x));
      const tr = P.trace(net, ids), n = ids.length, toks = ids.map(P.tok), r = inp.row, L = net.blocks.length, Hh = net.blocks[0].attn.heads, d = tr.x0[0].length, dk = d / Hh;
      const maps = []; tr.blocks.forEach((b, l) => b.w.forEach((w, h) => maps.push({ l, h, w })));
      const src = (i) => P.sourceOf(full, 5, i);
      const hits = (w) => { let k = 0; for (let i = 5; i < n; i++) if (src(i) >= 0 && P.argmax(w[i]) === src(i)) k++; return k; };
      const strip = (w, lit) => {
        const st = P.strip('hs', 70, 466, toks, { size: 13, gap: 8, cls: (i) => (i === r ? 'p12ask' : lit && i === src(r) ? 'p12src' : '') }), items = st.items.slice();
        if (w) {
          const cx = (k) => st.boxes[k].x + st.boxes[k].w / 2, y0 = st.boxes[0].y - 3;
          w[r].forEach((a, k) => { if (k === r || a < 0.04) return; const x1 = cx(r) - 6, x2 = cx(k), hh = 14 + Math.abs(k - r) * 6; items.push({ key: 'hs:a' + k, kind: 'path', d: `M${P.r2(x1)},${y0} C${P.r2(x1)},${y0 - hh} ${P.r2(x2)},${y0 - hh} ${P.r2(x2)},${y0}`, cls: 'p12arc', paint: { sw: P.r2(1 + 8 * a) }, pathAudit: 'edge-x', layer: 'lines', box: { x: Math.min(x1, x2), y: y0 - hh, w: Math.abs(x2 - x1), h: hh } }); });
        }
        return items;
      };
      const fmt = (v) => (v < 0.05 ? '' : H().fx(v, 1).replace(/^0\./, '.'));
      const bigMap = (m) => P.mat('hm', 70, 96, m.w, { rows: toks.map((t, i) => i + ' ' + t), cols: toks, cw: 30, ch: 25, scheme: 'seq', max: 1, fmt, size: 11, title: 'layer ' + m.l + ', head ' + m.h + ': row = who asks, column = who is read', rowCls: (i) => (i === r ? 'strong' : ''), cls: (i, j) => (i === r && j === P.argmax(m.w[r]) ? 'act' : '') }).items;
      // 1. the split
      F.push([...strip(null, false), H().text('sp:t', 70, 56, 'width d = ' + d + ', split into ' + Hh + ' heads of ' + dk, 'strong', 13),
        ...P.mat('sq', 70, 110, tr.blocks[0].q[0], { rows: toks.map((t, i) => i + ' ' + t), cols: Array.from({ length: dk }, (_, j) => String(j)), cw: 22, ch: 25, fmt: null, title: 'head 0 queries' }).items,
        ...P.mat('sk', 330, 110, tr.blocks[0].q[1], { rows: false, cols: Array.from({ length: dk }, (_, j) => String(dk + j)), cw: 22, ch: 25, fmt: null, title: 'head 1 queries' }).items],
        `The model is ${q.n(d)} numbers wide. Each layer makes queries, keys and values of that width and cuts them into ${q.n(Hh)} heads of ${q.n(dk)}: each head scores with its own slice, so each can learn its own pattern, for the same cost as one wide head.`, { phase: 'split', d, Hh, dk });
      maps.forEach((m) => {
        F.count('heads');
        const k = hits(m.w), j = P.argmax(m.w[r]);
        F.push([...bigMap(m), ...strip(m.w, true)], `Layer ${q.n(m.l)}, head ${q.n(m.h)}. Answer position ${q.n(r)} (holding “${P.qtok(q, ids[r])}”) must produce the digit that sits at position ${q.n(src(r))}; this head gives most weight, ${q(m.w[r][j], 2)}, to position ${q.n(j)}. Over the ${q.n(5)} answer positions, its largest weight lands on the needed digit ${q.n(k)} times.`, { phase: 'head', l: m.l, h: m.h, row: r, j, wj: m.w[r][j], src: src(r), hits: k, answers: 5, tok: ids[r] });
      });
      // 2. all four, small, with their hit counts
      const all = maps.map((m, k) => P.mat('hx' + k, 90 + (k % 2) * 360, 110 + Math.floor(k / 2) * 190, m.w, { rows: false, cols: false, cw: 14, ch: 14, gap: 1, fmt: null, scheme: 'seq', max: 1, title: 'layer ' + m.l + ' head ' + m.h + ': ' + hits(m.w) + ' of 5 point right' }).items).flat();
      const best = maps.reduce((b, m, k) => (hits(m.w) > hits(maps[b].w) ? k : b), 0);
      F.push([...all, ...strip(null, false)], `Side by side, the heads differ. The clearest pointer is layer ${q.n(maps[best].l)}, head ${q.n(maps[best].h)}, right ${q.n(hits(maps[best].w))} times out of ${q.n(5)}; others spread their weight, mix in the separator or look at the answer so far.`, { phase: 'all', bl: maps[best].l, bh: maps[best].h, hits: hits(maps[best].w), answers: 5 });
      // 3. switching heads off
      const set = P.probeSet(inp.task, 150, 7), base = P.ablate(net, set, null), drops = maps.map((m) => P.ablate(net, set, { layer: m.l, head: m.h }));
      const worst = drops.indexOf(Math.min(...drops));
      const bars = maps.map((m, k) => { const y = 140 + k * 54, w = 360 * drops[k]; return [H().text('ab:n' + k, 250, y + 18, 'without layer ' + m.l + ' head ' + m.h, 'legend', 13, 'end'), { key: 'ab:b' + k, kind: 'cell', x: 262, y, w: P.r2(Math.max(2, w)), h: 26, rx: 3, text: '', cls: 'p12bar ' + (k === worst ? 'hot' : 'soft'), audit: 'ignore' }, H().text('ab:v' + k, 270 + w, y + 18, H().pct(drops[k]), 'tick', 13)]; }).flat();
      F.push([H().text('ab:t', 70, 70, 'answer digits right on ' + set.length + ' new sequences', 'strong', 13), H().text('ab:f', 250, 104, 'all heads on', 'legend', 13, 'end'), { key: 'ab:full', kind: 'cell', x: 262, y: 86, w: P.r2(360 * base), h: 26, rx: 3, text: '', cls: 'p12bar', audit: 'ignore' }, H().text('ab:fv', 270 + 360 * base, 104, H().pct(base), 'tick', 13), ...bars, ...strip(null, false)],
        `To see what a head is for, switch it off. On ${q.n(set.length)} new sequences the full model gets ${q.pct(base)} of the answer digits right. Without layer ${q.n(maps[worst].l)}, head ${q.n(maps[worst].h)} it falls to ${q.pct(drops[worst])}; without each of the other heads it stays at ${H().list(drops.filter((_, k) => k !== worst).map((v) => q.pct(v)))}.`, { phase: 'ablate', count: set.length, base, drops, wl: maps[worst].l, wh: maps[worst].h, worst: drops[worst] });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Multi-head attention', text: ['split Q, K, V of width d into h heads of d/h', 'headᵢ = softmax(Qᵢ Kᵢᵀ / √(d/h)) Vᵢ', 'join the heads side by side (width d)', 'output = joined · Wo'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'all', once: true, pick: 'option',
      ask: 'Which head, switched off, will hurt the model most?',
      options: () => ['layer 0 head 0', 'layer 0 head 1', 'layer 1 head 0', 'layer 1 head 1'],
      answer: (fr) => { const a = fr.find((f) => f.meta.phase === 'ablate').meta; return 'layer ' + a.wl + ' head ' + a.wh; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'ablate').say,
    }],
    quiz: [
      { q: 'A model 512 wide with 8 heads gives each head queries and keys of size…', options: ['64', '512', '8', '4096'], answer: 0, why: '512 / 8 = 64.' },
      { q: 'How are the outputs of the heads combined?', options: ['joined side by side, then multiplied by Wo', 'averaged', 'only the best head is kept', 'multiplied together'], answer: 0, why: 'Concatenation restores the width; Wo mixes the heads.' },
      { q: 'Why use several small heads instead of one wide one?', options: ['each can attend to a different place for a different reason', 'they need far fewer weights in total', 'they remove the need for a softmax', 'they make attention causal'], answer: 0, why: 'One softmax row is one pattern; several heads give several patterns at once.' },
      { q: 'Switching one head off and seeing accuracy collapse tells you…', options: ['that head does work the others cannot cover', 'the model was badly trained', 'the head has the largest weights', 'nothing at all'], answer: 0, why: 'This kind of test (ablation) measures what a part contributes.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
