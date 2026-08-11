// Part 1, missing values and outliers. Missing cells drawn red in the table, then dropped or filled (mean, median,
// most frequent, or the mean of the k most similar rows) with the histogram of the column showing what the filling
// did to its spread; and outliers found by the box-plot rule or by z-scores, with what clipping or dropping does.
(function (S) {
  'use strict';
  const H = S.P01, C = () => S.Kit.curve, DT = () => S.Kit.dt;

  // ---- imputers (fitted on rows X, applied to any rows) ----
  /** the fill value of one column: mean, median or most frequent of the present values */
  H.fillValue = (vals, how) => {
    const ok = vals.filter(H.isNum);
    if (how === 'median') return H.median(ok);
    if (how === 'mode') { const m = new Map(); for (const v of ok) m.set(v, (m.get(v) || 0) + 1); let best = null; for (const [v, c] of m) if (best === null || c > m.get(best) || (c === m.get(best) && v < best)) best = v; return best; }
    return H.mean(ok);
  };
  /** the distance used for rows with gaps: squared differences over the columns both have, scaled up for the rest */
  H.nanDist = (a, b) => {
    let s = 0, k = 0;
    for (let j = 0; j < a.length; j++) if (H.isNum(a[j]) && H.isNum(b[j])) { s += (a[j] - b[j]) ** 2; k++; }
    return k ? Math.sqrt((a.length / k) * s) : NaN;
  };
  /** k-nearest-neighbour filling: each gap gets the mean of that column over the k nearest rows that have it */
  H.knnFill = (Xfit, row, j, k) => {
    const cand = [];
    Xfit.forEach((r, i) => { if (r === row || !H.isNum(r[j])) return; const d = H.nanDist(row, r); if (Number.isFinite(d)) cand.push([d, i]); });
    cand.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const use = cand.slice(0, k);
    return { value: H.mean(use.map(([, i]) => Xfit[i][j])), rows: use.map(([, i]) => i), dist: use.map(([d]) => d) };
  };
  H.impute = (X, feats, how, k) => X.map((r) => r.map((v, j) => {
    if (H.isNum(v)) return v;
    if (feats[j].type === 'category') return H.fillValue(X.map((q) => q[j]), 'mode');
    return how === 'knn' ? H.knnFill(X, r, j, k || 5).value : H.fillValue(X.map((q) => q[j]), how);
  }));

  const GAPSETS = {
    penguins: { set: 'penguins', opt: {}, label: 'Penguins (real gaps)' },
    pumps: { set: 'pumps', opt: { missing: 0.04, n: 400 }, label: 'Pump sensors, 4% lost' },
    harvest: { set: 'harvest', opt: { missing: 0.1, n: 200 }, label: 'Harvest, 10% blanked' },
    rents: { set: 'rents', opt: { missing: 0.2, n: 150 }, label: 'Rents, 20% blanked' },
  };
  const HOW = { drop: 'drop rows with a gap', mean: 'fill with the mean', median: 'fill with the median', mode: 'fill with the most frequent value', knn: 'fill from the 5 most similar rows' };

  S.topic({
    id: 'data-missing', family: 'p01-clean', order: 10, title: 'Missing values and imputation',
    blurb: 'Real tables have holes: a sensor dropped out, a field was left blank. Most models cannot take a hole, so either drop the rows that have one (and lose data, perhaps not at random) or fill each hole with a guess: the column\'s mean or median, its most frequent value, or the average of the rows most like this one. Filling with one constant piles values up at that point and shrinks the column\'s spread; filling from similar rows keeps more of its shape.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Dataset', def: 'harvest', options: Object.entries(GAPSETS).map(([value, g]) => ({ value, label: g.label })) },
      { id: 'how', kind: 'choice', label: 'Strategy', def: 'mean', options: Object.entries(HOW).map(([value, label]) => ({ value, label })) },
    ],
    presets: [
      { label: 'Harvest: mean filling', values: { set: 'harvest', how: 'mean' } },
      { label: 'Harvest: similar rows', values: { set: 'harvest', how: 'knn' } },
      { label: 'Rents: drop rows (20% gaps)', values: { set: 'rents', how: 'drop' } },
      { label: 'Penguins: median', values: { set: 'penguins', how: 'median' } },
      { label: 'Pumps: most frequent', values: { set: 'pumps', how: 'mode' } },
    ],
    counters: [{ id: 'gaps', label: 'missing cells' }, { id: 'rows', label: 'rows kept' }],
    run(inp) {
      const g = GAPSETS[inp.set], ds = H.data(g.set, g.opt), X = ds.X, feats = ds.features, n = X.length, d = feats.length;
      const missCol = feats.map((_, j) => X.filter((r) => !H.isNum(r[j])).length), gaps = H.sum(missCol);
      const rowsWith = X.map((r, i) => (r.some((v) => !H.isNum(v)) ? i : -1)).filter((i) => i >= 0);
      const numJ = feats.map((f, j) => (f.type === 'category' ? -1 : j)).filter((j) => j >= 0);
      const jc = numJ.reduce((b, j) => (missCol[j] > missCol[b] ? j : b), numJ[0]), cf = feats[jc], cname = H.short(cf);
      // the table: up to 5 columns (the chosen one first), 8 rows (gappy rows first)
      const showJ = [jc].concat(feats.map((_, j) => j).filter((j) => j !== jc).sort((a, b) => missCol[b] - missCol[a] || a - b)).slice(0, 5).sort((a, b) => a - b);
      const showI = rowsWith.slice(0, 6).concat(X.map((_, i) => i).filter((i) => !rowsWith.includes(i)).slice(0, 2)).sort((a, b) => a - b).slice(0, 8);
      const filled = inp.how === 'drop' ? null : H.impute(X, feats, inp.how, 5);
      const fmtCell = (j, v) => (!H.isNum(v) ? NaN : feats[j].type === 'category' ? feats[j].categories[v] : H.g(v, 4));
      const cols = [{ name: '#', w: 44, cls: 'idx' }].concat(showJ.map((j) => ({ name: H.short(feats[j]), w: Math.max(74, S.textWidth(H.short(feats[j]), 13, { bold: true }) + 18), cls: feats[j].type === 'category' ? 'cat' : 'num' })));
      const table = (mode) => {
        const rows = showI.map((i) => [String(i)].concat(showJ.map((j) => (mode === 'fill' && !H.isNum(X[i][j]) ? fmtCell(j, filled[i][j]) : fmtCell(j, X[i][j])))));
        const t = DT().table({ prefix: 'tb', x: 0, y: 34, cols, rows, rowH: 26, size: 13, cls: (r, c) => { const i = showI[r], j = showJ[c - 1]; if (c === 0) return mode === 'drop' && rowsWith.includes(i) ? 'off' : ''; if (mode === 'drop' && rowsWith.includes(i)) return 'off'; if (mode === 'fill' && !H.isNum(X[i][j])) return 'fill'; return ''; } });
        return t;
      };
      const t0 = table('gaps'), tw = t0.w;
      // the histogram of the chosen column: present values, and the filled ones on top
      const present = X.map((r) => r[jc]).filter(H.isNum), lo = Math.min(...present), hi = Math.max(...present);
      const after = inp.how === 'drop' ? X.filter((r) => r.every(H.isNum)).map((r) => r[jc]) : filled.map((r) => r[jc]);
      const fillVals = inp.how === 'drop' ? [] : X.map((r, i) => (H.isNum(r[jc]) ? null : filled[i][jc])).filter((v) => v !== null);
      const bins = 14, h0 = H.histogram(present, bins, lo, hi), hf = H.histogram(fillVals.length ? fillVals : [lo], bins, lo, hi);
      if (!fillVals.length) hf.counts.fill(0);
      const hAfter = inp.how === 'drop' ? H.histogram(after, bins, lo, hi) : null;
      const top = Math.max(...h0.counts.map((c, k) => c + hf.counts[k]));
      const yt = C().ticks(0, top * 1.08, 4);
      const hx = tw + 64, hw = 770 - hx;
      const ax = C().axes({ prefix: 'hx', x: hx, y: 64, w: hw, h: 230, xr: [lo, hi], yr: [0, yt[yt.length - 1]], xticks: C().ticks(lo, hi, 4).filter((u) => u >= lo && u <= hi), yticks: yt, xname: cname + (cf.unit ? ' (' + cf.unit + ')' : ''), yname: 'rows', tickSize: 12 });
      const stats = (a) => ({ mean: H.mean(a), sd: H.sd(a), n: a.length });
      const s0 = stats(present), s1 = stats(after);
      const F = new S.Frames(['gaps', 'rows']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, ds.doc.title + ': ' + n + ' rows');
      const statLines = (lines) => lines.map((l, k) => H.text('st' + k, hx, 350 + k * 24, l[0], l[1] || 'strong', 14));
      F.set('gaps', gaps).set('rows', n);
      const perCol = feats.map((f, j) => [H.short(f), missCol[j]]).filter(([, m]) => m > 0);
      F.add({ items: [title, ...t0.items, ...ax.items, H.title('ht', hx, 30, 'the column ' + cname), ...DT().hist(ax, 'h0', h0.counts, h0.edges, 'soft'), ...statLines([['present: ' + s0.n + ' values', 'tick'], ['mean ' + H.g(s0.mean) + ',  sd ' + H.g(s0.sd), 'strong']])], say: `${gaps} of the ${n * d} cells are missing (red), spread over ${rowsWith.length} of the ${n} rows: ${H.list(perCol.map(([nm, m]) => nm + ' ' + m))}. Over its ${s0.n} present values, ${cname} has mean ${H.g(s0.mean)} and sd ${H.g(s0.sd)}.`, meta: { phase: 'gaps', gaps, rowsWith: rowsWith.length, missCol, mean0: s0.mean, sd0: s0.sd, n0: s0.n, jc, cells: n * d }, view });
      if (inp.how === 'drop') {
        const kept = n - rowsWith.length;
        F.set('rows', kept);
        F.add({ items: [title, ...table('drop').items, ...ax.items, H.title('ht', hx, 30, 'the column ' + cname), ...DT().hist(ax, 'h0', hAfter.counts, hAfter.edges, 'soft'), ...statLines([['kept: ' + kept + ' of ' + n + ' rows', 'bad'], ['mean ' + H.g(s1.mean) + ',  sd ' + H.g(s1.sd), 'strong']])], say: `Dropping every row with a gap keeps ${kept} of ${n} rows (${H.pct(kept / n)}): ${rowsWith.length} rows go, and with them ${rowsWith.length * d - gaps} perfectly good cells. With ${d} columns, even a small share of gaps per column removes many rows.`, meta: { phase: 'drop', kept, lost: rowsWith.length, mean1: s1.mean, sd1: s1.sd, share: kept / n }, view });
      } else {
        const fv = inp.how === 'knn' ? null : H.fillValue(X.map((r) => r[jc]), inp.how);
        const howSay = inp.how === 'knn' ? 'each gap gets the mean of that column over the 5 rows nearest to its row (measured on the columns both rows have)' : `every gap in ${cname} gets the ${inp.how === 'mode' ? 'most frequent value' : inp.how} of the present values, ${H.g(fv, 4)}`;
        F.add({ items: [title, ...table('fill').items, ...ax.items, H.title('ht', hx, 30, 'the column ' + cname), ...DT().hist(ax, 'h0', h0.counts, h0.edges, 'soft'), ...statLines([['filled: ' + fillVals.length + ' cells', 'good']])], say: `Filling (green): ${howSay}.` + (feats.some((f, j) => f.type === 'category' && missCol[j] > 0) ? ' Category columns always take their most frequent value.' : ''), meta: { phase: 'fill', fv, nFill: fillVals.length }, view });
        // the filled values stacked on the histogram
        const stacked = h0.counts.map((c, k) => c + hf.counts[k]);
        const fillBars = DT().hist(ax, 'hf', stacked, h0.edges, 'good', { only: (k) => hf.counts[k] > 0 });
        const spike = hf.counts.indexOf(Math.max(...hf.counts));
        F.add({ items: [title, ...table('fill').items, ...ax.items, H.title('ht', hx, 30, 'the column ' + cname), ...fillBars, ...DT().hist(ax, 'h0', h0.counts, h0.edges, 'soft'), ...statLines([['before: mean ' + H.g(s0.mean) + ',  sd ' + H.g(s0.sd), 'strong'], ['after:  mean ' + H.g(s1.mean) + ',  sd ' + H.g(s1.sd), 'good']])], say: `The filled values (green, stacked on top) ` + (inp.how === 'knn' ? `spread over ${hf.counts.filter((c) => c > 0).length} bins, following the rows they came from.` : `all land in one bin, ${H.g(h0.edges[spike])} to ${H.g(h0.edges[spike + 1])}: a spike.`) + ` The sd goes from ${H.g(s0.sd)} to ${H.g(s1.sd)}` + (s1.sd < s0.sd ? ': the column looks more certain than it is.' : '.'), meta: { phase: 'effect', mean1: s1.mean, sd1: s1.sd, spike, bins: hf.counts.filter((c) => c > 0).length }, view });
      }
      // every strategy side by side for this column
      const cmp = ['drop', 'mean', 'median', 'knn'].map((h) => { const a = h === 'drop' ? X.filter((r) => r.every(H.isNum)).map((r) => r[jc]) : H.impute(X, feats, h, 5).map((r) => r[jc]); return [h, stats(a)]; });
      const cRows = cmp.map(([h, s]) => [h === 'knn' ? '5 similar rows' : h, String(s.n), H.g(s.mean), H.g(s.sd)]);
      const ct = DT().table({ prefix: 'cmp', x: 0, y: t0.h + 60, cols: [{ name: 'strategy', w: 130 }, { name: 'rows', w: 64, cls: 'num' }, { name: 'mean', w: 76, cls: 'num' }, { name: 'sd', w: 70, cls: 'num' }], rows: cRows, rowH: 22, size: 13, headH: 24, cls: (r) => (cmp[r][0] === inp.how ? 'act' : '') });
      F.add({ items: [title, ...(inp.how === 'drop' ? table('drop') : table('fill')).items, ...ax.items, H.title('ht', hx, 30, 'the column ' + cname), ...DT().hist(ax, 'h0', h0.counts, h0.edges, 'soft'), ...ct.items], say: `The four strategies on ${cname}: the sd is ${H.list(cmp.map(([h, s]) => H.g(s.sd) + ' (' + (h === 'knn' ? 'similar rows' : h) + ')'))}, against ${H.g(s0.sd)} for the values that were there. Whatever the strategy, its fill values are learned from the training rows only; filling with numbers from the test rows leaks them.`, meta: { phase: 'compare', cmp: cmp.map(([h, s]) => [h, s.n, s.mean, s.sd]) }, view });
      const ctx = { n, d, sd0: s0.sd, mean0: s0.mean, lostCells: rowsWith.length * d - gaps, binEdges: h0.edges };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Strategies', text: ['drop: lose every row with a gap', 'mean / median: one constant per column', 'most frequent: for categories', 'k nearest rows: mean of their values,', '  distance over the columns both rows have', 'fit fill values on the training rows only'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'fill', pick: 'option', ask: 'After filling, will the spread (sd) of the column be smaller, larger, or the same as before?', options: ['smaller', 'larger', 'about the same'], answer: (fr) => { const a = fr[0].meta.sd0, b = fr.find((f) => f.meta.phase === 'effect').meta.sd1; return b < a * 0.995 ? 'smaller' : b > a * 1.005 ? 'larger' : 'about the same'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'effect').say },
      { at: (f) => f.meta.phase === 'gaps', pick: 'option', ask: 'How many rows have at least one gap?', options: (fr) => { const r = fr[0].meta.rowsWith; return [...new Set([r, fr[0].meta.gaps, Math.max(0, r - 3), r + 5])].map(String); }, answer: (fr) => String(fr[0].meta.rowsWith), explain: (fr) => fr[0].say },
    ],
    quiz: [
      { q: 'A column\'s gaps are all filled with its mean. Its mean…', options: ['stays the same, but its spread shrinks', 'rises', 'falls', 'becomes zero'], answer: 0, why: 'Adding values equal to the mean leaves the mean alone and adds no spread.' },
      { q: 'Each of 10 columns has 5% of its cells missing, at random. Dropping rows with a gap keeps about…', options: ['60% of the rows', '95%', '50%', '5%'], answer: 0, why: 'A row survives with probability 0.95¹⁰ ≈ 0.60.' },
      { q: 'Why fit the fill values on the training rows only?', options: ['statistics of the test rows would leak into training', 'it is faster', 'test rows have no gaps', 'the median needs sorting'], answer: 0, why: 'The test set must stay unseen, including its averages.' },
      { q: 'Which fill is most robust to a few huge values in the column?', options: ['the median', 'the mean', 'the maximum', 'zero'], answer: 0, why: 'The median ignores how far out the extreme values are.' },
      { q: 'Values are missing because the sensor fails when it overheats. Dropping those rows…', options: ['hides exactly the hot cases: the gaps are not random', 'is always safe', 'improves accuracy on hot days', 'changes nothing'], answer: 0, why: 'When missingness depends on the value, dropping or filling with the mean biases the data.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Outliers
  // ---------------------------------------------------------------------------------------------------------------
  const OUT = {
    pumps: { set: 'pumps', opt: { missing: 0 }, col: 0, label: 'Pump temperature (faults inside)' },
    typo: { set: 'wildflowers', opt: {}, col: 2, typo: true, label: 'Wildflower stem height, one typo' },
    years: { set: 'loans', opt: {}, col: 2, label: 'Loan years employed (a long tail)' },
    rain: { set: 'buses', opt: {}, col: 2, label: 'Bus trips: rain (mostly 0)' },
  };
  function outData(key) {
    const o = OUT[key], ds = H.data(o.set, o.opt), v = H.col(ds, o.col).slice();
    if (o.typo) v[7] = +(v[7] * 100).toFixed(1);
    return { v, f: ds.features[o.col], ds, truth: o.set === 'pumps' ? ds.y : null };
  }
  S.topic({
    id: 'data-outliers', family: 'p01-clean', order: 20, title: 'Outliers',
    blurb: 'An outlier is a value far from the rest. Two common rules flag them: beyond 1.5 interquartile ranges outside the box of a box plot, or more than 3 standard deviations from the mean. The second is fooled by the outliers themselves, which stretch the standard deviation. A flagged value may be a typing error (fix or drop it), or the very thing you are looking for (a faulty pump); clipping pulls it in to the fence instead.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Column', def: 'typo', options: Object.entries(OUT).map(([value, o]) => ({ value, label: o.label })) },
      { id: 'rule', kind: 'choice', label: 'Rule', def: 'iqr', options: [{ value: 'iqr', label: 'box plot: k × IQR beyond the box' }, { value: 'z', label: 'z-score: k sd from the mean' }] },
      { id: 'k', kind: 'slider', label: 'k', def: 1.5, min: 1, max: 4, step: 0.5, fmt: (v) => String(v) },
      { id: 'act', kind: 'choice', label: 'Then', def: 'clip', options: [{ value: 'clip', label: 'clip to the fences' }, { value: 'drop', label: 'drop the rows' }] },
    ],
    presets: [
      { label: 'A typo, box-plot rule', values: { data: 'typo', rule: 'iqr', k: 1.5, act: 'drop' } },
      { label: 'A typo, 3 sd rule', values: { data: 'typo', rule: 'z', k: 3, act: 'drop' } },
      { label: 'Pump faults found', values: { data: 'pumps', rule: 'iqr', k: 1.5, act: 'clip' } },
      { label: 'A long tail is not an error', values: { data: 'years', rule: 'iqr', k: 1.5, act: 'clip' } },
      { label: 'Mostly zero: everything else flagged', values: { data: 'rain', rule: 'iqr', k: 1.5, act: 'clip' } },
    ],
    counters: [{ id: 'flag', label: 'flagged' }],
    run(inp) {
      const D = outData(inp.data), v = D.v, n = v.length, name = H.short(D.f), unit = D.f.unit ? ' ' + D.f.unit : '';
      const st = H.five(v, inp.k), mean = H.mean(v), sd = H.sd(v);
      const lo = inp.rule === 'iqr' ? st.q1 - inp.k * st.iqr : mean - inp.k * sd, hi = inp.rule === 'iqr' ? st.q3 + inp.k * st.iqr : mean + inp.k * sd;
      const flag = v.map((x) => x < lo || x > hi), nf = flag.filter(Boolean).length;
      const [x0, x1] = H.span(Math.min(...v, lo), Math.max(...v, hi), 0.03);
      const bx = C().axes({ prefix: 'bx', x: 40, y: 60, w: 720, h: 190, xr: [x0, x1], yr: [0, 1], yticks: 0, xname: name + unit, grid: false, tickSize: 12 });
      const yOf = (i) => +(200 + H.jit(i) * 38).toFixed(2);
      const dots = (cls) => v.map((x, i) => ({ key: 'pt' + i, kind: 'dot', x: bx.X(Math.min(x1, Math.max(x0, x))), y: yOf(i), r: n > 400 ? 2.2 : 3, cls: cls(i), audit: 'overlap-ok' }));
      const fence = (key, x, text, side) => [{ key, kind: 'path', d: `M${bx.X(x)},64 V250`, cls: 'guide act', audit: 'ignore', layer: 'back', z: 3 }, { key: key + 't', kind: 'text', x: bx.X(x) + (side < 0 ? -4 : 4), y: 56, text, anchor: side < 0 ? 'end' : 'start', size: 12, cls: 'act' }];
      const fences = () => [...(lo > x0 ? fence('flo', lo, H.g(lo), -1) : []), ...fence('fhi', hi, H.g(hi), 1)];
      const boxIt = () => DT().box(bx, 'bp', { lo: st.lo, q1: st.q1, med: st.med, q3: st.q3, hi: st.hi }, 110, 28, '');
      const F = new S.Frames(['flag']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, D.ds.doc.title + ': ' + name + ', ' + n + ' values');
      const lines = (ls) => ls.map((l, k) => H.text('ln' + k, 40, 330 + k * 26, l[0], l[1] || 'strong', 14));
      F.add({ items: [title, ...bx.items, ...dots(() => 'pt0'), ...(inp.rule === 'iqr' ? boxIt() : [])], say: `${n} values of ${name}, from ${H.g(st.min)} to ${H.g(st.max)}${unit}. ` + (inp.rule === 'iqr' ? `The box runs from Q1 = ${H.g(st.q1)} to Q3 = ${H.g(st.q3)}, so IQR = ${H.g(st.iqr)}.` : `Their mean is ${H.g(mean)} and their sd ${H.g(sd)}.`), meta: { phase: 'look', min: st.min, max: st.max, q1: st.q1, q3: st.q3, iqr: st.iqr, mean, sd }, view });
      const ruleSay = inp.rule === 'iqr' ? `The fences sit ${inp.k} × IQR beyond the box: ${H.g(st.q1)} − ${H.g(inp.k * st.iqr)} = ${H.g(lo)} and ${H.g(st.q3)} + ${H.g(inp.k * st.iqr)} = ${H.g(hi)}.` : `The fences sit ${inp.k} sd from the mean: ${H.g(mean)} ± ${H.g(inp.k * sd)}, from ${H.g(lo)} to ${H.g(hi)}.`;
      F.set('flag', nf);
      F.add({ items: [title, ...bx.items, ...fences(), ...dots((i) => (flag[i] ? 'outl' : 'pt0')), ...(inp.rule === 'iqr' ? boxIt() : [])], say: ruleSay + ` ${H.plural(nf, 'value lies', 'values lie')} outside (red).`, meta: { phase: 'flag', lo, hi, nf }, view });
      if (D.truth) {
        const tp = v.filter((_, i) => flag[i] && D.truth[i] === 1).length, faults = D.truth.filter((y) => y === 1).length;
        F.add({ items: [title, ...bx.items, ...fences(), ...dots((i) => (D.truth[i] === 1 ? (flag[i] ? 'outl' : 'sp c1') : flag[i] ? 'sp c0 wrong' : 'pt0')), ...lines([['flagged and really faulty: ' + tp + ' of ' + nf, 'bad'], ['faulty but not flagged (orange): ' + (faults - tp), 'c1'], ['faults of other kinds hide in the other sensors', 'tick']])], say: `Checking against the log of faults: ${tp} of the ${nf} flagged minutes really were faults. ${faults - tp} of the ${faults} faulty minutes (orange) look normal on temperature, because their fault shows in vibration or pressure instead.`, meta: { phase: 'truth', tp, faults }, view });
      }
      const kept = v.filter((_, i) => !flag[i]), after = inp.act === 'clip' ? v.map((x) => Math.min(hi, Math.max(lo, x))) : kept;
      const m1 = H.mean(after), s1 = H.sd(after), med1 = H.median(after);
      const actSay = inp.act === 'clip' ? `Clipping pulls the ${H.plural(nf, 'flagged value')} in to the fences and keeps every row` : `Dropping removes the ${H.plural(nf, 'flagged row')}, leaving ${kept.length}`;
      F.add({ items: [title, ...bx.items, ...fences(), ...dots((i) => (flag[i] ? (inp.act === 'clip' ? 'fillpt' : 'pt0 faded') : 'pt0')).map((d, i) => (flag[i] && inp.act === 'clip' ? Object.assign(d, { x: bx.X(after[i]) }) : d)), ...lines([['before: mean ' + H.g(mean) + ',  sd ' + H.g(sd) + ',  median ' + H.g(st.med), 'strong'], ['after:  mean ' + H.g(m1) + ',  sd ' + H.g(s1) + ',  median ' + H.g(med1), 'good']])], say: `${actSay}. The mean moves from ${H.g(mean)} to ${H.g(m1)} and the sd from ${H.g(sd)} to ${H.g(s1)}, while the median barely moves (${H.g(st.med)} to ${H.g(med1)}).` + (inp.data === 'years' ? ' But these long careers are real: clipping them throws away true information.' : inp.data === 'rain' ? ' With most trips dry, every rainy trip is "unusual": the rule is the wrong tool for a column like this.' : ''), meta: { phase: 'act', m1, s1, med1, med0: st.med, kept: kept.length }, view });
      const ctx = { n, q1: st.q1, q3: st.q3, kIqr: inp.k * st.iqr, kSd: inp.k * sd, mean, sd, nf, missed: D.truth ? D.truth.filter((y, i) => y === 1 && !flag[i]).length : 0 };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Two rules', text: ['box plot: outside [Q1 − k·IQR, Q3 + k·IQR]', 'z-score: |x − x̄| / sd > k', 'z uses x̄ and sd, which outliers inflate', 'clip: move to the fence; drop: remove the row', 'an outlier may be an error, or the signal'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'look', pick: 'option', ask: 'How many values will the rule flag?', options: (fr) => { const k = fr[1].meta.nf; return [...new Set([k, k + 3, Math.max(0, k - 1), 0, k * 2 + 1])].slice(0, 4).map(String); }, answer: (fr) => String(fr[1].meta.nf), explain: (fr) => fr[1].say },
    ],
    quiz: [
      { q: 'Q1 = 10, Q3 = 20. With k = 1.5 the upper fence is…', options: ['35', '30', '21.5', '50'], answer: 0, why: 'IQR = 10, so 20 + 1.5 × 10 = 35.' },
      { q: 'Why can the 3-sd rule miss a single enormous value in a small sample?', options: ['the value inflates the sd it is measured with', 'the sd ignores large values', 'it uses the median', 'it only looks at negatives'], answer: 0, why: 'One huge value raises both the mean and the sd, so its own z-score stays modest.' },
      { q: 'A pump temperature far above the rest is…', options: ['probably the fault we want to detect: keep it', 'certainly a sensor error', 'always clipped', 'always dropped'], answer: 0, why: 'In anomaly detection the outliers are the point.' },
      { q: 'Clipping instead of dropping…', options: ['keeps every row but bounds the extreme values', 'removes rows', 'changes the median a lot', 'fills missing values'], answer: 0, why: 'Values beyond a fence are set to the fence; the row stays.' },
      { q: 'Which summary is least affected by outliers?', options: ['the median', 'the mean', 'the standard deviation', 'the range'], answer: 0, why: 'It depends only on the middle of the sorted values.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
