// Part 4, multinomial naive Bayes on our own short messages: a message scored word by word (each word adds the log
// of how much likelier it is in spam than in an ordinary message), and smoothing: what a word never seen in one
// class does to the verdict when α is 0, and how adding α to every count fixes it.
(function (S) {
  'use strict';
  const H = S.PL, P = S.PL4, ML = () => S.ML;
  const n$ = H.n, lp = P.lp;
  const NAMES = ['not spam', 'spam'];
  /** a probability for a sentence: very small or very large ones as bounds */
  const pr = (v) => (v < 0.001 ? 'below 0.001' : v > 0.999 ? 'above 0.999' : n$(v));

  const msgCheck = (max) => (v) => { const w = P.words(v); if (!w.length) return 'message: type some words'; if (w.length > max) return 'message: at most ' + max + ' words, so the picture stays readable'; return null; };
  /** the model on every message except the one being scored */
  const fit = (text, alpha) => {
    const bag = P.bag(S.Data.textSets.spam, text.trim());
    const m = ML().MultinomialNB({ alpha }).fit(bag.X, bag.y);
    return { bag, m, V: bag.vocab.length, N: m.featureCount.map((c) => ML().sum(c)), nDocs: [0, 1].map((k) => bag.y.filter((v) => v === k).length) };
  };
  /** the words of a message as chips, wrapped */
  const chips = (key, words, x, y, w, cls) => {
    const it = [];
    let cx = x, cy = y;
    words.forEach((t, i) => {
      const cw = Math.max(34, S.geom.textWidth(t, 14, { mono: true }) + 16);
      if (cx + cw > x + w) { cx = x; cy += 36; }
      it.push({ key: key + i, kind: 'cell', x: H.r2(cx), y: cy, w: H.r2(cw), h: 28, rx: 6, text: t, size: 14, cls: 'pl-word ' + (cls ? cls(i) || '' : '') });
      cx += cw + 8;
    });
    return { items: it, bottom: cy + 28 };
  };

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'nb-spam', family: 'p04-nb', order: 20, title: 'A spam filter, word by word',
    blurb: 'Multinomial naive Bayes treats a message as a bag of words. For each class it counts how often every word appears in that class’s training messages, and turns the counts into word probabilities (with a little smoothing). A new message starts from the log prior of each class, and every word adds the log of its probability in that class. The difference between the two running totals, the log-odds, says spam or not.',
    inputs: [
      { id: 'text', kind: 'text', label: 'message', def: 'Are you free for lunch? Reply today', check: msgCheck(16) },
      { id: 'alpha', kind: 'slider', label: 'smoothing α', def: 1, min: 0.1, max: 3, step: 0.1 },
    ],
    presets: [
      { label: 'Free for lunch?', values: { text: 'Are you free for lunch? Reply today', alpha: 1 } },
      { label: 'A fake prize', values: { text: 'You have won a free prize, claim it now', alpha: 1 } },
      { label: 'An ordinary note', values: { text: 'Can you send me the photos from the trip tomorrow', alpha: 1 } },
      { label: 'A training message (left out first)', values: { text: 'Cheap loans approved in minutes, no credit check, apply now!', alpha: 1 } },
      { label: 'Words never seen', values: { text: 'Zebra marmalade quantum free', alpha: 1 } },
      { label: 'Heavy smoothing', values: { text: 'You have won a free prize, claim it now', alpha: 3 } },
    ],
    counters: [{ id: 'words', label: 'words scored' }],
    run(inp) {
      const { bag, m, V, N, nDocs } = fit(inp.text, inp.alpha), a = inp.alpha;
      const toks = P.words(inp.text), known = toks.map((t) => bag.at.has(t));
      const F = new S.Frames(['words']);
      const prior = m.classLogPrior.slice(), lo0 = prior[1] - prior[0];
      const steps = [];
      let run = lo0;
      toks.forEach((t, i) => {
        if (!known[i]) return;
        const j = bag.at.get(t), c = [m.featureCount[0][j], m.featureCount[1][j]], l = [m.featureLogProb[0][j], m.featureLogProb[1][j]], d = l[1] - l[0];
        run += d;
        steps.push({ i, t, j, c, p: l.map(Math.exp), l, d, run });
      });
      const post = m.predictProba([P.vector(bag, inp.text)])[0], joint = m.jointLogLikelihood([P.vector(bag, inp.text)])[0];
      // the log-odds chart
      const cbx = chips('x', toks, 30, 32, 720).bottom, cx = { x: 70, y: cbx + 180, w: 640, h: 200 }, nS = Math.max(1, steps.length);
      const lim = Math.max(2, ...steps.map((s) => Math.abs(s.d)), ...steps.map((s) => Math.abs(s.run)), Math.abs(lo0)) * 1.1, st = lim > 12 ? 5 : lim > 6 ? 2 : 1, top = Math.ceil(lim / st) * st;
      const cax = H.axes({ prefix: 'lo', x: cx.x, y: cx.y, w: cx.w, h: cx.h, xr: [0, nS + 1], yr: [-top, top], xticks: 0, yticks: st, yname: 'log-odds of spam', grid: true });
      const slot = cx.w / (nS + 1);
      const chart = (upto) => {
        const it = [...cax.items, H.seg('lo:zero', cx.x, cax.Y(0), cx.x + cx.w, cax.Y(0), 'pl-seg', 'ignore')];
        it.push(H.text('lo:sp', cx.x + cx.w + 6, cx.y + 12, 'spam ↑', 'pl-lab c1', 12), H.text('lo:hm', cx.x + cx.w + 6, cx.y + cx.h - 4, 'not ↓', 'pl-lab c0', 12));
        const pts = [[cax.X(0.5), cax.Y(lo0)]];
        it.push(H.text('lo:pl', cax.X(0.5), cx.y + cx.h + 18, 'prior', 'tick', 12, 'middle'));
        steps.forEach((s, k) => {
          const xc = cax.X(k + 1.5);
          it.push(H.text('lo:w' + k, xc, cx.y + cx.h + 18, s.t.length > 8 ? s.t.slice(0, 7) + '…' : s.t, k === upto ? 'pl-lab act' : 'tick', 12, 'middle'));
          if (k > upto) return;
          const y0 = cax.Y(s.run - s.d), y1 = cax.Y(s.run);
          it.push({ key: 'lo:b' + k, kind: 'cell', x: H.r2(xc - slot * 0.28), y: H.r2(Math.min(y0, y1)), w: H.r2(slot * 0.56), h: H.r2(Math.max(1, Math.abs(y1 - y0))), rx: 2, text: '', cls: 'pl-pbar ' + (s.d >= 0 ? 'c1' : 'c0') + (k === upto ? '' : ' dim'), audit: 'ignore' });
          pts.push([xc, y1]);
        });
        it.push({ key: 'lo:run', kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + H.r2(p[0]) + ',' + H.r2(p[1])).join(' '), cls: 'series thin', pathAudit: 'edge-x', layer: 'lines' });
        pts.forEach((p, k) => it.push({ key: 'lo:d' + k, kind: 'dot', x: p[0], y: p[1], r: 3.5, cls: 'mark', audit: 'overlap-ok', layer: 'items' }));
        return it;
      };
      let ord = -1;
      const chipCls = (cur) => (i) => (!known[i] ? 'unseen' : i === cur ? 'act' : '');
      const head = (cur) => [H.title('nb:t', 30, 20, 'the message'), ...chips('nb:c', toks, 30, 32, 720, chipCls(cur)).items];
      const cb = chips('x', toks, 30, 32, 720).bottom;
      const block = (lines) => P.eqs('nb:eq', 30, cb + 34, lines, 14, 25);
      const unknown = toks.filter((t, i) => !known[i]);
      F.add({ items: [...head(-1), ...chart(-1), ...block([['trained on ' + bag.y.length + ' messages: ' + nDocs[1] + ' spam, ' + nDocs[0] + ' not', ''], [V + ' different words; ' + N[1] + ' word tokens in spam, ' + N[0] + ' in the others', 'muted'], ['prior log-odds = ln(' + nDocs[1] + ' / ' + nDocs[0] + ') = ' + lp(lo0), 'act']])], say: 'The filter was trained on ' + bag.y.length + ' messages (' + nDocs[1] + ' spam, ' + nDocs[0] + ' not)' + (bag.y.length < S.Data.textSets.spam.length ? ', leaving out the one being scored' : '') + ': ' + V + ' different words. Before reading any word, the prior log-odds of spam are ln(' + nDocs[1] + ' / ' + nDocs[0] + ') = ' + lp(lo0) + '.' + (unknown.length ? ' ' + H.plural(unknown.length, 'word') + ' of this message never appeared in training (dashed) and will be ignored.' : ''), meta: { phase: 'prior', nDocs, V, N, lo0, nTrain: bag.y.length, unknown: unknown.length } });
      steps.forEach((s, k) => {
        F.set('words', k + 1);
        ord = k;
        const lines = [['"' + s.t + '": ' + s.c[1] + ' times in spam, ' + s.c[0] + ' in the others', ''], ['P(' + s.t + ' | spam) = (' + s.c[1] + ' + ' + n$(a) + ') / (' + N[1] + ' + ' + n$(a) + '·' + V + ') = ' + n$(s.p[1]), 'c1'], ['P(' + s.t + ' | not) = (' + s.c[0] + ' + ' + n$(a) + ') / (' + N[0] + ' + ' + n$(a) + '·' + V + ') = ' + n$(s.p[0]), 'c0'], ['ln ratio = ' + lp(s.l[1]) + ' − (' + lp(s.l[0]) + ') = ' + (s.d >= 0 ? '+' : '') + lp(s.d), s.d >= 0 ? 'c1' : 'c0']];
        F.add({ items: [...head(s.i), ...chart(k), ...block(lines)], say: '"' + s.t + '" appears ' + s.c[1] + ' times in spam and ' + s.c[0] + ' times in the other messages. Smoothed, P(' + s.t + ' | spam) = ' + n$(s.p[1]) + ' and P(' + s.t + ' | not spam) = ' + n$(s.p[0]) + ', so the word adds ' + lp(s.d) + ' to the log-odds' + (s.d >= 0 ? ' (toward spam)' : ' (away from spam)') + ', which now stand at ' + lp(s.run) + '.', meta: { phase: 'word', c: s.c, p: s.p, l: s.l, d: s.d, run: s.run, a, N, V } });
      });
      const verdict = post[1] >= 0.5 ? 'spam' : 'not spam', final = steps.length ? steps[steps.length - 1].run : lo0;
      F.add({ items: [...head(-1), ...chart(steps.length - 1), ...block([['log-odds ' + lp(final) + ' → P(spam) ' + (post[1] < 0.001 || post[1] > 0.999 ? pr(post[1]).replace('below', '<').replace('above', '>') : '= ' + pr(post[1])), post[1] >= 0.5 ? 'c1' : 'c0'], ['verdict: ' + verdict, 'act'], ['scores: spam ' + lp(joint[1]) + ', not ' + lp(joint[0]), 'muted']])], say: 'After ' + H.plural(steps.length, 'word') + ' the log-odds are ' + lp(final) + ': P(spam) = 1 / (1 + e^(' + lp(-final) + ')), ' + (post[1] < 0.001 || post[1] > 0.999 ? '' : 'which is ') + pr(post[1]) + ', so the filter says ' + verdict + '. The order of the words never mattered; only their counts did.', meta: { phase: 'verdict', final, negFinal: -final, post: post[1], joint, bounds: [0.001, 0.999], nWords: steps.length } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Multinomial naive Bayes', text: ['P(w | c) = (count(w, c) + α) / (N_c + α·V)', 'log score(c) = log P(c) + Σ_words log P(w | c)', 'log-odds = score(spam) − score(not)', 'P(spam) = 1 / (1 + e^(−log-odds))', 'unknown words are skipped'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'prior', pick: 'option', ask: 'Will the filter call this message spam?', options: ['spam', 'not spam'], answer: (fr) => (fr[fr.length - 1].meta.post >= 0.5 ? 'spam' : 'not spam'), explain: (fr) => fr[fr.length - 1].say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'word' && f.meta.phase === 'word', once: true, pick: 'option', ask: 'Will the next word push toward spam or away from it?', options: ['toward spam', 'away from spam'], answer: (fr, i) => (fr[i + 1].meta.d >= 0 ? 'toward spam' : 'away from spam'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'Spam has 200 word tokens, "prize" appears 7 times, the vocabulary has 300 words and α = 1. P(prize | spam) = ?', options: ['8 / 500', '7 / 200', '8 / 201', '7 / 500'], answer: 0, why: '(7 + 1) / (200 + 1·300).' },
      { q: 'Why does the order of the words not matter?', options: ['the model only uses each word’s count', 'messages are sorted first', 'the log-odds are averaged', 'it does matter'], answer: 0, why: 'A bag of words keeps counts and forgets order; sums do not depend on order.' },
      { q: 'A word with the same probability in both classes adds what to the log-odds?', options: ['0', '1', 'its count', 'the prior'], answer: 0, why: 'ln(p / p) = 0.' },
      { q: 'What happens to a word the training data never contained?', options: ['it is skipped: it has no column', 'it counts as spam', 'it makes the probability zero', 'the filter fails'], answer: 0, why: 'The vocabulary is fixed when the model is trained.' },
      { q: 'Log-odds of 2 mean P(spam) is about…', options: ['0.88', '0.5', '0.67', '0.98'], answer: 0, why: '1 / (1 + e^(−2)) ≈ 0.881.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  const ALPHA0 = 1e-10;
  S.topic({
    id: 'nb-smoothing', family: 'p04-nb', order: 30, title: 'Smoothing: the word never seen',
    blurb: 'Without smoothing, a word that never appeared in spam has P(word | spam) = 0, and one zero wipes out the whole product: the message cannot be spam, however many other words shout that it is. Additive smoothing adds α to every count, so no probability is zero. Small α trusts the counts; large α flattens every word toward the same probability.',
    inputs: [
      { id: 'text', kind: 'text', label: 'message', def: 'Your lunch for today', check: (v) => { const e = msgCheck(12)(v); if (e) return e; const b = P.bag(S.Data.textSets.spam, v.trim()); const k = [...new Set(P.words(v).filter((w) => b.at.has(w)))]; if (!k.length) return 'message: use at least one word from the training messages'; if (k.length > 9) return 'message: at most 9 different known words'; return null; } },
      { id: 'alphas', kind: 'list', label: 'α values', def: ['0', '0.01', '0.1', '0.5', '1', '2', '5'], min: 2, max: 8, check: (v) => { for (const s of v) { const a = +s; if (!(a >= 0 && a <= 20)) return 'α values: numbers from 0 to 20'; } return null; } },
    ],
    presets: [
      { label: 'Spammy words and one ordinary word', values: { text: 'Your lunch for today', alphas: ['0', '0.01', '0.1', '0.5', '1', '2', '5'] } },
      { label: 'Unseen words on both sides', values: { text: 'Claim your free prize at the station tomorrow', alphas: ['0', '0.01', '0.1', '0.5', '1', '2', '5'] } },
      { label: 'An ordinary note with one spam word', values: { text: 'Lunch tomorrow at the station, my treat, it is free', alphas: ['0', '0.1', '1', '5'] } },
      { label: 'Every word seen in both classes', values: { text: 'Is it for you today', alphas: ['0', '0.5', '1', '10'] } },
      { label: 'Large α flattens everything', values: { text: 'Win a free prize now', alphas: ['0.1', '1', '5', '20'] } },
    ],
    counters: [{ id: 'alpha', label: 'α' }],
    run(inp) {
      const toks = P.words(inp.text), alphas = [...new Set(inp.alphas.map(Number))].sort((a, b) => a - b);
      const base = fit(inp.text, 1), bag = base.bag, V = base.V, N = base.N, words = [...new Set(toks.filter((t) => bag.at.has(t)))];
      const counts = words.map((w) => [base.m.featureCount[0][bag.at.get(w)], base.m.featureCount[1][bag.at.get(w)]]);
      const zero = words.map((w, i) => counts[i][0] === 0 || counts[i][1] === 0);
      const zs = words.filter((w, i) => counts[i][1] === 0), zh = words.filter((w, i) => counts[i][0] === 0);
      const crit = zs.length ? words.indexOf(zs[0]) : zero.indexOf(true);
      const x = P.vector(bag, inp.text);
      const runs = alphas.map((a) => { const m = ML().MultinomialNB({ alpha: a }).fit(bag.X, bag.y); return { a, m, post: m.predictProba([x])[0][1], p: words.map((w) => [Math.exp(m.featureLogProb[0][bag.at.get(w)]), Math.exp(m.featureLogProb[1][bag.at.get(w)])]) }; });
      // the bar chart of word probabilities, on a log scale
      const bx = { x: 64, y: 130, w: 430, h: 300 }, floor = -13;
      const bax = H.axes({ prefix: 'sb', x: bx.x, y: bx.y, w: bx.w, h: bx.h, xr: [0, words.length], yr: [floor, 0], xticks: 0, yticks: [-12, -10, -8, -6, -4, -2, 0], fmtY: (v) => (v === 0 ? '1' : '10' + H.sup(-v).replace(/^/, '⁻')), yname: 'P(word | class), log scale', grid: true });
      const slot = bx.w / words.length;
      const bars = (r) => {
        const it = [];
        words.forEach((w, i) => {
          it.push(H.text('sb:w' + i, bax.X(i + 0.5), bx.y + bx.h + 18, w.length > 8 ? w.slice(0, 7) + '…' : w, zero[i] ? 'pl-lab hot' : 'tick', 12, 'middle'));
          [0, 1].forEach((k) => {
            const v = Math.max(floor, Math.log10(r.p[i][k])), top = bax.Y(v), bw = Math.min(22, slot * 0.34);
            it.push({ key: 'sb:b' + i + '_' + k, kind: 'cell', x: H.r2(bax.X(i + 0.5) + (k ? 2 : -2 - bw)), y: H.r2(top), w: H.r2(bw), h: H.r2(Math.max(1, bx.y + bx.h - top)), rx: 2, text: '', cls: 'pl-pbar c' + k, audit: 'ignore' });
          });
        });
        return it;
      };
      const lx = { x: 560, y: 130, w: 190, h: 170 }, la = (a) => Math.log10(a === 0 ? ALPHA0 : a);
      const ax0 = Math.min(-2, Math.floor(la(alphas.find((a) => a > 0) || 0.01)) - 0.5), ax1 = Math.max(1, Math.ceil(la(Math.max(...alphas)) + 0.2));
      const pax = H.axes({ prefix: 'sp', x: lx.x, y: lx.y, w: lx.w, h: lx.h, xr: [ax0, ax1], yr: [0, 1], xticks: [-2, -1, 0, 1].filter((v) => v >= ax0 && v <= ax1), yticks: 0.5, fmtX: (v) => String(+Math.pow(10, v).toPrecision(1)), xname: 'α (log)', yname: 'P(spam | message)', grid: true });
      const cl = (a) => (a === 0 ? ax0 : la(a));
      const F = new S.Frames(['alpha']);
      const head = [H.title('sm:t', 30, 20, 'the message'), ...chips('sm:c', toks, 30, 32, 720, (i) => (!bag.at.has(toks[i]) ? 'unseen' : zero[words.indexOf(toks[i])] ? 'act' : '')).items];
      const legend = S.Kit.curve.legend('sm:lg', lx.x, lx.y + lx.h + 64, [{ text: 'P(w | not spam)', cls: 'c0', dot: true }, { text: 'P(w | spam)', cls: 'c1', dot: true }], { dir: 'col' });
      const cw = crit >= 0 ? words[crit] : null, cc = crit >= 0 ? counts[crit] : null, zc = crit >= 0 ? (cc[1] === 0 ? 1 : 0) : -1;
      F.add({ items: [...head, ...bax.items, ...pax.items, ...legend, ...P.eqs('sm:eq', lx.x, lx.y + lx.h + 128, [[V + ' words in the vocabulary', 'muted']], 13, 22)], say: 'The message has ' + H.plural(words.length, 'known word') + '. ' + (crit >= 0 ? '"' + cw + '" appears ' + cc[1] + ' times in spam and ' + cc[0] + ' times in the other messages: with no smoothing, P(' + cw + ' | ' + NAMES[zc] + ') would be exactly 0.' : 'Every one of them appears in both classes, so even α = 0 leaves no zero probability.'), meta: { phase: 'start', words: words.length, V, c: cc || [], N } });
      runs.forEach((r, s) => {
        F.set('alpha', r.a);
        const pts = runs.slice(0, s + 1).map((o) => [cl(o.a), o.post]);
        const items = [...head, ...bax.items, ...bars(r), ...pax.items, S.Kit.curve.line(pax, 'sp:l', pts, 'act'), ...pts.map((p, i) => S.Kit.curve.dot(pax, 'sp:d' + i, p[0], p[1], i === s ? '' : 'muted', i === s ? 5 : 3.5)), ...legend];
        const lines = [['α = ' + (r.a === 0 ? '0' : n$(r.a)) + ':  P(spam) ' + (r.post < 0.001 || r.post > 0.999 ? pr(r.post).replace('below', '<').replace('above', '>') : '= ' + pr(r.post)), 'act']];
        if (crit >= 0) lines.push(['ln P(' + cw + ' | ' + (zc ? 'spam' : 'not') + ') = ' + lp(Math.log(r.p[crit][zc])), 'c' + zc]);
        items.push(...P.eqs('sm:eq', lx.x, lx.y + lx.h + 128, lines, 13, 22));
        const a$ = r.a === 0 ? '0' : n$(r.a);
        const say = 'α = ' + a$ + (r.a === 0 ? ' (computed with 10⁻¹⁰, as libraries do, to avoid the logarithm of 0)' : '') + ': ' +
          (crit >= 0 ? 'P(' + cw + ' | ' + NAMES[zc] + ') = (0 + α) / (' + N[zc] + ' + α·' + V + '), whose logarithm is ' + lp(Math.log(r.p[crit][zc])) + '. ' : '') +
          'The message is spam with probability ' + pr(r.post) + '.' +
          (r.a === 0 && crit >= 0 ? (zs.length && zh.length ? ' Words unseen in each class drag that class’s score down by about 23 each (ln 10⁻¹⁰), so the verdict hangs on which class has more of them: ' + zs.length + ' unseen in spam, ' + zh.length + ' in the others.' : zs.length ? ' ' + (zs.length === 1 ? 'One word' : zs.length + ' words') + ' never seen in spam ' + (zs.length === 1 ? 'vetoes' : 'veto') + ' spam, whatever the other words say.' : ' ' + (zh.length === 1 ? 'One word' : zh.length + ' words') + ' never seen in ordinary messages ' + (zh.length === 1 ? 'forces' : 'force') + ' spam, whatever the other words say.') : s === runs.length - 1 && r.a >= 2 ? ' With this much smoothing all word probabilities drift toward 1 / ' + V + ' and the words count for less.' : '');
        F.add({ items, say, meta: { phase: 'alpha', a: r.a, post: r.post, bounds: [0.001, 0.999], lnPz: crit >= 0 ? Math.log(r.p[crit][zc]) : 0, N, V, zs: zs.length, zh: zh.length, tiny: [10, -Math.log(ALPHA0)] } });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Additive smoothing', text: ['P(w | c) = (count(w, c) + α) / (N_c + α·V)', 'α = 0: maximum likelihood; an unseen word gives 0', 'α = 1: Laplace smoothing', 'α → ∞: every word gets 1 / V'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'With α = 0 (or the smallest α here), will the message be judged spam?', options: ['yes', 'no'], answer: (fr) => (fr[1].meta.post >= 0.5 ? 'yes' : 'no'), explain: (fr) => fr[1].say },
    ],
    quiz: [
      { q: 'Without smoothing, a word never seen in spam makes P(spam | message)…', options: ['exactly 0', 'smaller, but not 0', 'unchanged', 'exactly 1'], answer: 0, why: 'One factor of 0 makes the whole product 0.' },
      { q: 'With α = 1, a word seen 0 times in a class with 50 tokens and a vocabulary of 100 gets…', options: ['1 / 150', '0', '1 / 50', '1 / 100'], answer: 0, why: '(0 + 1) / (50 + 1·100).' },
      { q: 'As α grows very large, every P(w | c) tends to…', options: ['1 / V', '0', '1', 'the word’s count'], answer: 0, why: 'The added α·V swamps the real counts.' },
      { q: 'How should α be chosen in practice?', options: ['by validation, like any hyperparameter', 'always 0', 'always 1', 'as the vocabulary size'], answer: 0, why: 'It trades trust in the counts against caution about rare words.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
