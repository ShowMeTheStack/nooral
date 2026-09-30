// Part 10, convolutional networks: the exercises. Every number is worked out here by the engines (the tensor engine's
// convolution, pooling, output-size rule and autodiff, S.T, in double precision, and the network S.DL builds for the
// canvas), so the text, the answers and the hints agree.
(function (S) {
  'use strict';
  const C = S.Content, T = S.T, WF = S.WF;
  if (!C || !T || !WF || !S.DL || C.exercises.has('cn-x-conv-cell')) return;
  const r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const f64 = (fn) => T.float64(fn);
  const z0 = (v) => (Math.abs(v) < 1e-12 ? 0 : v);
  const rows = (v, h, w) => Array.from({ length: h }, (_, i) => Array.from(v.slice(i * w, i * w + w), z0));
  const str = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));
  const sm = (M) => M.map((r) => r.map(String));
  const sign = (v) => String(v).replace(/^-/, '−');
  const grid = (M, name) => ({ table: [[name].concat(M[0].map((_, j) => 'column ' + (j + 1)))].concat(M.map((r, i) => ['row ' + (i + 1)].concat(r.map((v) => sign(v))))) });
  const ex = (x) => C.exercise(x);
  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });
  /** a valid, stride-1 (or any opts) convolution of one picture with one kernel, by the engine: rows of the map */
  const conv = (P, K, b, opts) => f64(() => { const y = T.conv2d(T.tensor([[P]]), T.tensor([[K]]), b === undefined ? undefined : T.tensor([b]), opts); return rows(y.values(), y.shape[2], y.shape[3]); });
  const pool = (M, k, s) => f64(() => { const y = T.maxPool2d(T.tensor([[M]]), k, s); return rows(y.values(), y.shape[2], y.shape[3]); });
  const outSize = (n, k, o) => T.convOut(n, n, k, k, o)[0];
  const one1 = (t) => t.values()[0];
  const L = (s) => s.split('\n').map((r) => r.trim().split(/\s+/).map(Number));

  // =========================================================================================================
  // Chapter 1: the convolution
  // =========================================================================================================
  const PIC = L(`0 0 1 0 0
                 0 1 1 1 0
                 1 1 0 1 1
                 0 1 1 1 0
                 0 0 1 0 0`);
  const KSPOT = [[0, 1, 0], [1, -4, 1], [0, 1, 0]], spot = conv(PIC, KSPOT);
  ex({ id: 'cn-x-conv-cell', chapter: 'cn-conv', title: 'One output cell', level: 1, kind: 'number',
    text: ['A 5 × 5 picture (1 is ink) and a 3 × 3 kernel:', grid(PIC, 'picture'), grid(KSPOT, 'kernel'), 'The kernel slides over the picture with no padding and stride 1, giving a 3 × 3 map. What is the value in the middle of the map, row 2, column 2?'],
    answer: spot[1][1], tolerance: 0, whyNot: { [String(spot[0][0])]: 'That is the top-left cell; the middle cell’s window is the middle 3 × 3 of the picture.', '-4': 'The centre pixel is paper (0), so the −4 contributes nothing.' },
    hints: ['The middle cell’s window is rows 2 to 4 and columns 2 to 4 of the picture.', 'Multiply each window pixel by the kernel weight on top of it: the corners are multiplied by 0, the four edge pixels by 1, the centre by −4.', 'The four edge pixels are all ink and the centre is paper: 1 + 1 + 1 + 1 + 0 · (−4) = ' + spot[1][1] + '. This kernel answers to a hole surrounded by ink.'],
    solution: String(spot[1][1]), wrong: ['-4', String(spot[0][0])] });

  const KSTEP = [[1, -1, 0], [1, -1, 0], [1, -1, 0]], step = conv(PIC, KSTEP);
  ex({ id: 'cn-x-conv-map', chapter: 'cn-conv', title: 'The whole map', level: 2, kind: 'matrix',
    text: ['The same 5 × 5 picture, and a new kernel:', grid(PIC, 'picture'), grid(KSTEP, 'kernel'), 'Fill in the whole 3 × 3 output map (no padding, stride 1). Row 1, column 1 is the window in the picture’s top-left corner.'],
    rows: ['row 1', 'row 2', 'row 3'], cols: ['column 1', 'column 2', 'column 3'], corner: 'output', answer: step, tolerance: 0,
    hints: ['Each output is the sum of (column 1 of its window) minus (column 2 of its window): the kernel’s third column is zero.', 'Row 1, column 1: the window’s first column holds 0, 0, 1 and its second 0, 1, 1, so 1 − 2 = −1.', 'The map is ' + step.map((r) => r.map(sign).join(', ')).join(' / ') + '. It is negative where ink starts after paper, going right.'],
    solution: sm(step), wrong: [sm(step.map((r) => r.map((v) => -v))), sm(step.map((r, i) => r.map((v, j) => (i === 1 && j === 1 ? v + 1 : v))))] });

  const relu = conv(PIC, KSPOT, -2), reluCell = relu[1][1], reluA = Math.max(0, reluCell), reluCorner = conv(PIC, KSPOT, -2)[0][0];
  ex({ id: 'cn-x-conv-relu', chapter: 'cn-conv', title: 'Bias and ReLU', level: 2, kind: 'table',
    text: ['The spot kernel [[0, 1, 0], [1, −4, 1], [0, 1, 0]] on the 5 × 5 picture of “One output cell”, now with a bias of −2 and a ReLU after it. Fill in the value before and after the ReLU at the middle cell and at the top-left cell of the map.'],
    rows: ['middle (row 2, column 2)', 'top left (row 1, column 1)'], cols: ['before ReLU', 'after ReLU'], corner: 'cell',
    cells: { 'middle (row 2, column 2)|before ReLU': reluCell, 'middle (row 2, column 2)|after ReLU': reluA, 'top left (row 1, column 1)|before ReLU': reluCorner, 'top left (row 1, column 1)|after ReLU': Math.max(0, reluCorner) }, tolerance: 0,
    hints: ['Before ReLU: the convolution’s sum plus the bias. After: max(0, that).', 'Without the bias the middle cell is ' + spot[1][1] + ' and the top-left cell ' + spot[0][0] + '.', 'Middle: ' + spot[1][1] + ' − 2 = ' + reluCell + ', kept by ReLU. Top left: ' + spot[0][0] + ' − 2 = ' + sign(reluCorner) + ', cut to 0. The bias sets how strong a match must be before the unit fires.'],
    solution: { 'middle (row 2, column 2)|before ReLU': String(reluCell), 'middle (row 2, column 2)|after ReLU': String(reluA), 'top left (row 1, column 1)|before ReLU': String(reluCorner), 'top left (row 1, column 1)|after ReLU': '0' },
    wrong: [{ 'middle (row 2, column 2)|before ReLU': String(spot[1][1]), 'middle (row 2, column 2)|after ReLU': String(spot[1][1]), 'top left (row 1, column 1)|before ReLU': String(spot[0][0]), 'top left (row 1, column 1)|after ReLU': '0' }, { 'middle (row 2, column 2)|before ReLU': String(reluCell), 'middle (row 2, column 2)|after ReLU': String(reluA), 'top left (row 1, column 1)|before ReLU': String(reluCorner), 'top left (row 1, column 1)|after ReLU': String(reluCorner) }] });

  const convSol = 'function conv2d(img, ker) {\n  const H = img.length, W = img[0].length, k = ker.length, m = ker[0].length;\n  const out = [];\n  for (let i = 0; i + k <= H; i++) {\n    const row = [];\n    for (let j = 0; j + m <= W; j++) {\n      let s = 0;\n      for (let u = 0; u < k; u++) for (let v = 0; v < m; v++) s += img[i + u][j + v] * ker[u][v];\n      row.push(s);\n    }\n    out.push(row);\n  }\n  return out;\n}';
  ex({ id: 'cn-x-code-conv', chapter: 'cn-conv', title: 'Write a 2-D convolution', level: 2, kind: 'code',
    text: ['Write `conv2d(img, ker)`: slide the kernel over the picture with no padding and stride 1, and return the output map as an array of rows. Output row i, column j is Σᵤ Σᵥ img[i + u][j + v] · ker[u][v] (no flipping of the kernel, as networks do it). The kernel need not be square.'],
    fn: 'conv2d', params: ['img', 'ker'], start: 'function conv2d(img, ker) {\n  \n}', tolerance: 1e-9, ref: (img, ker) => conv(img, ker),
    cases: [[PIC, KSPOT], [PIC, KSTEP], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]]], [[[1, 2, 3, 4]], [[1, -1]]], [[[0.5, -1], [2, 1], [1, 0]], [[2], [-1]]], [[[3]], [[2]]]],
    hints: ['The output has H − k + 1 rows and W − m + 1 columns.', 'Four loops: output row, output column, then kernel row and kernel column, adding img[i + u][j + v] · ker[u][v].', convSol],
    solution: convSol,
    wrong: ['function conv2d(img, ker) { const k = ker.length, m = ker[0].length, out = []; for (let i = 0; i + k <= img.length; i++) { const r = []; for (let j = 0; j + m <= img[0].length; j++) { let s = 0; for (let u = 0; u < k; u++) for (let v = 0; v < m; v++) s += img[i + u][j + v] * ker[k - 1 - u][m - 1 - v]; r.push(s); } out.push(r); } return out; }',
      'function conv2d(img, ker) { const k = ker.length, out = []; for (let i = 0; i + k <= img.length; i++) { const r = []; for (let j = 0; j + k <= img[0].length; j++) { let s = 0; for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) s += img[i + u][j + v] * ker[u][v]; r.push(s); } out.push(r); } return out; }'] });

  const SV = outSize(28, 5);
  ex({ id: 'cn-x-size-valid', chapter: 'cn-conv', title: 'How much smaller?', level: 1, kind: 'number',
    text: ['A 28 × 28 image goes through a 5 × 5 convolution with no padding and stride 1. How many pixels wide is the output?'],
    answer: SV, tolerance: 0, whyNot: { '23': 'Count the places: the kernel’s left edge can sit at columns 1 to 24.', '28': 'Without padding the kernel cannot be centred on the edge pixels.', '5.6': 'That divides; the size goes down by k − 1, it is not divided by k.' },
    hints: ['out = n − k + 1 with no padding and stride 1.', 'The kernel needs 5 columns, so its left edge has 28 − 5 + 1 places.', '28 − 5 + 1 = ' + SV + '.'], solution: String(SV), wrong: ['23', '28'] });

  const ST = [['A', 32, 3, 1, 1, 1], ['B', 32, 3, 1, 2, 1], ['C', 32, 5, 0, 1, 2], ['D', 31, 3, 0, 2, 1]], stcells = {};
  for (const [id, n, k, p, s, d] of ST) stcells[id + '|output'] = outSize(n, k, { padding: p, stride: s, dilation: d });
  ex({ id: 'cn-x-size-table', chapter: 'cn-conv', title: 'Output sizes', level: 2, kind: 'table',
    text: ['Fill in the output width for each square input, using out = ⌊(n + 2p − d(k − 1) − 1) / s⌋ + 1.', { table: [['', 'n', 'kernel k', 'padding p', 'stride s', 'dilation d']].concat(ST.map((r) => r.map(String))) }],
    rows: ST.map((r) => r[0]), cols: ['output'], corner: 'layer', cells: stcells, tolerance: 0,
    hints: ['Work out the span d(k − 1) + 1 first, then (n + 2p − span) / s, rounded down, plus one.', 'C: the span is 2 · 4 + 1 = 9, so (32 − 9) / 1 + 1. D: (31 − 3) / 2 + 1.', 'A: ' + stcells['A|output'] + ' (same padding). B: ' + stcells['B|output'] + ' (stride 2 halves it). C: ' + stcells['C|output'] + '. D: ' + stcells['D|output'] + '.'],
    solution: str(stcells), wrong: [Object.assign(str(stcells), { 'C|output': '28' }), Object.assign(str(stcells), { 'B|output': '15', 'D|output': '14' })] });

  const SP = 2 * (5 - 1) / 2, spCheck = outSize(20, 5, { padding: SP, dilation: 2 });
  ex({ id: 'cn-x-same-pad', chapter: 'cn-conv', title: 'Padding to keep the size', level: 2, kind: 'number',
    text: ['A 5 × 5 kernel with dilation 2 and stride 1 must keep a 20 × 20 map at 20 × 20. How many rings of zero padding does it need?'],
    answer: SP, tolerance: 0, whyNot: { '2': 'That is right for an undilated 5 × 5 kernel; with dilation 2 it spans 9 pixels.', '8': 'Padding goes on both sides: half the extra span on each.', '9': 'That is the kernel’s span; the padding is half of span − 1.' },
    hints: ['At stride 1 the size stays the same when 2p = d(k − 1).', 'The span is d(k − 1) + 1 = 9, so the map would lose 8 pixels without padding.', 'p = 8 / 2 = ' + SP + ' on each side: (20 + 8 − 8 − 1) / 1 + 1 = ' + spCheck + '.'], solution: String(SP), wrong: ['2', '8'] });

  const SPAN = 3 * (3 - 1) + 1;
  ex({ id: 'cn-x-span', chapter: 'cn-conv', title: 'A dilated kernel’s span', level: 1, kind: 'number',
    text: ['A 3 × 3 kernel has dilation 3: its taps are 3 pixels apart. How many pixels wide is the patch it covers?'],
    answer: SPAN, tolerance: 0, whyNot: { '9': 'That is 3 × 3, the number of taps; the question asks how wide they reach.', '6': 'Count the pixels from the first tap to the last, inclusive: d(k − 1) + 1.', '3': 'That is the width without dilation.' },
    hints: ['The taps sit at 0, d and 2d from the first one.', 'With d = 3: at 0, 3 and 6.', 'From pixel 0 to pixel 6 inclusive: d(k − 1) + 1 = ' + SPAN + ' pixels, with only 9 weights.'], solution: String(SPAN), wrong: ['9', '6'] });

  const PM = L(`1 3 0 2
                4 2 1 0
                0 1 5 2
                2 0 3 6`), pm = pool(PM, 2, 2);
  ex({ id: 'cn-x-pool-map', chapter: 'cn-conv', title: 'Max pooling by hand', level: 1, kind: 'matrix',
    text: ['A 4 × 4 map after ReLU:', grid(PM, 'map'), 'Max-pool it with a 2 × 2 window and stride 2. Fill in the 2 × 2 result.'],
    rows: ['row 1', 'row 2'], cols: ['column 1', 'column 2'], corner: 'pooled', answer: pm, tolerance: 0,
    hints: ['The four windows are the four 2 × 2 corners of the map; they do not overlap.', 'Top-left window: 1, 3, 4, 2.', 'The largest of each window: ' + pm.map((r) => r.join(', ')).join(' / ') + '.'],
    solution: sm(pm), wrong: [sm([[2.5, 0.75], [0.75, 4]]), sm([[pm[0][0], pm[0][1]], [pm[1][0], 5]])] });

  const po = pool(PM, 3, 1);
  ex({ id: 'cn-x-pool-overlap', chapter: 'cn-conv', title: 'Overlapping windows', level: 2, kind: 'matrix',
    text: ['The same 4 × 4 map:', grid(PM, 'map'), 'Max-pool it with a 3 × 3 window moving 1 at a time. The output is 2 × 2; fill it in.'],
    rows: ['row 1', 'row 2'], cols: ['column 1', 'column 2'], corner: 'pooled', answer: po, tolerance: 0,
    hints: ['(4 − 3) / 1 + 1 = 2 places each way; each window is 3 × 3 and they overlap.', 'Window at row 1, column 1 covers rows 1 to 3 and columns 1 to 3.', 'The largest values: ' + po.map((r) => r.join(', ')).join(' / ') + '. The 5 in the middle sits in all four windows, so it can win more than once.'],
    solution: sm(po), wrong: [sm(pm), sm([[po[0][0], po[0][1]], [po[1][0], 5]])] });

  const PG = f64(() => { const x = T.param(T.tensor([[PM]])), y = T.maxPool2d(x, 3, 1); T.sum(y).backward(); return rows(x.grad.values(), 4, 4); });
  const pgSet = [];
  PG.forEach((r, i) => r.forEach((v, j) => { if (v) pgSet.push('(' + (i + 1) + ', ' + (j + 1) + ')'); }));
  const pgAns = pgSet.join(', ');
  ex({ id: 'cn-x-pool-grad', chapter: 'cn-conv', title: 'Where the gradient goes', level: 3, kind: 'set',
    text: ['Backpropagation through the overlapping max pooling of “Overlapping windows” (3 × 3, stride 1) on the same map, with a gradient of 1 arriving at each of the four pooled outputs. Which cells of the 4 × 4 map receive a non-zero gradient? Write them as (row, column), like (1, 2), (3, 4).', grid(PM, 'map')],
    answer: pgAns, placeholder: 'like (1, 2), (3, 4)',
    hints: ['Max pooling passes each output’s gradient only to the input that won its window.', 'Find the winner of each of the four windows; a cell that wins several windows collects several gradients.', 'The winners are ' + pgAns + '. Every other cell gets zero: changing a loser a little does not change the output.'],
    solution: pgAns, wrong: ['(3, 3)', '(1, 1), (1, 2), (2, 1), (2, 2)'] });

  const poolRef = (M, k, s) => pool(M, k, s);
  const poolSol = 'function maxPool(map, k, s) {\n  const out = [];\n  for (let i = 0; i + k <= map.length; i += s) {\n    const row = [];\n    for (let j = 0; j + k <= map[0].length; j += s) {\n      let m = -Infinity;\n      for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) m = Math.max(m, map[i + u][j + v]);\n      row.push(m);\n    }\n    out.push(row);\n  }\n  return out;\n}';
  ex({ id: 'cn-x-code-pool', chapter: 'cn-conv', title: 'Write max pooling', level: 2, kind: 'code',
    text: ['Write `maxPool(map, k, s)`: slide a k × k window over the map (an array of rows) with stride s and no padding, and return the largest value of each window, as an array of rows. Windows that would run off the map are not used.'],
    fn: 'maxPool', params: ['map', 'k', 's'], start: 'function maxPool(map, k, s) {\n  \n}', tolerance: 1e-9, ref: poolRef,
    cases: [[PM, 2, 2], [PM, 3, 1], [[[1, -2, 3], [-4, 5, -6], [7, -8, 9]], 2, 1], [[[-1, -2], [-3, -4]], 2, 2], [L(`1 2 3 4 5
      6 7 8 9 1
      2 3 4 5 6`), 2, 2], [[[0.5]], 1, 1]],
    hints: ['The output has ⌊(n − k) / s⌋ + 1 rows and as many columns, worked out the same way.', 'Start each window’s largest at −Infinity, not 0: a map can hold negative numbers.', poolSol],
    solution: poolSol,
    wrong: ['function maxPool(map, k, s) { const out = []; for (let i = 0; i + k <= map.length; i += s) { const r = []; for (let j = 0; j + k <= map[0].length; j += s) { let m = 0; for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) m = Math.max(m, map[i + u][j + v]); r.push(m); } out.push(r); } return out; }',
      'function maxPool(map, k, s) { const out = []; for (let i = 0; i + k <= map.length; i += k) { const r = []; for (let j = 0; j + k <= map[0].length; j += k) { let m = -Infinity; for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) m = Math.max(m, map[i + u][j + v]); r.push(m); } out.push(r); } return out; }'] });

  const CH2 = [L(`1 0 2
                  0 1 0
                  2 0 1`), L(`0 1 0
                              1 1 1
                              0 1 0`)], KC = [[[1, 0], [0, 1]], [[0, 2], [1, -1]]];
  const chv = f64(() => { const y = T.conv2d(T.tensor([CH2]), T.tensor([KC]), T.tensor([0.5])); return rows(y.values(), 2, 2); });
  const chA = f64(() => rows(T.conv2d(T.tensor([[CH2[0]]]), T.tensor([[KC[0]]])).values(), 2, 2)), chB = f64(() => rows(T.conv2d(T.tensor([[CH2[1]]]), T.tensor([[KC[1]]])).values(), 2, 2));
  ex({ id: 'cn-x-channels', chapter: 'cn-conv', title: 'A filter over two channels', level: 2, kind: 'number',
    text: ['A picture has two channels, each 3 × 3:', grid(CH2[0], 'channel 1'), grid(CH2[1], 'channel 2'), 'A filter has a 2 × 2 kernel for each channel, [[1, 0], [0, 1]] for channel 1 and [[0, 2], [1, −1]] for channel 2, and a bias of 0.5. What is its output at row 1, column 1 (the top-left window of both channels)?'],
    answer: chv[0][0], tolerance: 0.0005, whyNot: { [String(chA[0][0])]: 'That is channel 1’s part alone; add channel 2’s part and the bias.', [String(chA[0][0] + chB[0][0])]: 'Add the filter’s bias, 0.5.', [String(chA[0][0] + chB[0][0] + 1)]: 'A filter has one bias for all its channels, not one per channel.' },
    hints: ['Convolve each channel with its own kernel at the same place, then add the results and the one bias.', 'Channel 1: 1 · 1 + 0 · 0 + 0 · 0 + 1 · 1 = ' + chA[0][0] + '. Channel 2: 0 · 0 + 2 · 1 + 1 · 1 + (−1) · 1 = ' + chB[0][0] + '.', chA[0][0] + ' + ' + chB[0][0] + ' + 0.5 = ' + chv[0][0] + '.'],
    solution: String(chv[0][0]), wrong: [String(chA[0][0]), String(chA[0][0] + chB[0][0])] });

  const CNN = S.DL.build('cnn', { size: 16, channels: 1, classes: 10, filters: [6, 12], kernel: 5, hidden: [32], seed: 1 });
  const NP = CNN.namedParameters().map(([n, p]) => [n, p.size]), npOf = (i) => NP[2 * i][1] + NP[2 * i + 1][1];
  const PC = npOf(1);
  ex({ id: 'cn-x-params-conv', chapter: 'cn-conv', title: 'Weights of a convolution layer', level: 1, kind: 'number',
    text: ['A convolution layer reads 6 channels and has 12 filters, each with a 5 × 5 kernel per channel and one bias. How many weights and biases does it have?'],
    answer: PC, tolerance: 0, whyNot: { '312': 'Each filter has a 5 × 5 kernel for every one of the 6 input channels, not just one.', '1800': 'Those are the weights only; each of the 12 filters also has a bias.', '1806': 'One bias per filter: 12, not 6.' },
    hints: ['Per filter: C · k · k weights and one bias.', '6 · 5 · 5 = 150 weights, plus 1 bias: 151 per filter.', '12 · 151 = ' + PC + '. The second convolution of the network in the lessons is exactly this layer.'],
    solution: String(PC), wrong: ['1800', '312'] });

  const PD = npOf(2);
  ex({ id: 'cn-x-params-dense', chapter: 'cn-conv', title: 'Weights of a dense layer', level: 1, kind: 'number',
    text: ['After the last pooling, twelve 2 × 2 maps are laid out in one list and fed to a dense layer of 32 neurons. How many weights and biases does that layer have?'],
    answer: PD, tolerance: 0, whyNot: { '1536': 'Those are the weights only; add one bias per neuron.', '416': 'Every neuron reads all 48 numbers, not one per map.', '80': 'That adds the sizes; a dense layer has a weight for every pair of input and neuron.' },
    hints: ['The list has 12 · 2 · 2 = 48 numbers.', 'Each of the 32 neurons has a weight for each of the 48, and a bias.', '48 · 32 + 32 = ' + PD + '.'], solution: String(PD), wrong: ['1536', '416'] });

  ex({ id: 'cn-x-share-choice', chapter: 'cn-conv', title: 'Why so few weights?', level: 1, kind: 'choice',
    text: ['A layer turns a 16 × 16 grey image into six 16 × 16 maps. As a convolution with 5 × 5 kernels it has 156 weights and biases; as a dense layer making the same 1536 outputs it would have 394,752. Where does the saving come from?'],
    options: ['Each kernel is used at every place, so its 25 weights are shared by all 256 outputs of its map', 'Convolutions round their weights to fewer digits', 'Pooling removes most of the weights', 'A convolution ignores most of the image'], answer: 0,
    whyNot: { 1: 'The weights are ordinary numbers; the saving is in how many there are.', 2: 'Pooling has no weights at all, and comes after the convolution.', 3: 'Every pixel is read by some window; each output just reads a small patch.' },
    hints: ['Count the weights one output of a map uses, and how many different outputs share them.', 'A dense output has 256 weights of its own; a convolution output uses the kernel’s 25, the same 25 as every other output of its map.', 'Weight sharing (and small windows): 6 · (25 + 1) = 156, against 1536 · (256 + 1) = 394,752.'], solution: 0, wrong: [2, 3] });

  // =========================================================================================================
  // Chapter 2: inside a trained network
  // =========================================================================================================
  const s1 = outSize(16, 5, { padding: 2 }), p1 = outSize(s1, 2, { stride: 2 }), s2 = outSize(p1, 5), p2 = outSize(s2, 2, { stride: 2 });
  const fs = { 'first convolution|channels': 6, 'first convolution|size': s1, 'first pooling|channels': 6, 'first pooling|size': p1, 'second convolution|channels': 12, 'second convolution|size': s2, 'second pooling|channels': 12, 'second pooling|size': p2 };
  ex({ id: 'cn-x-flow-shapes', chapter: 'cn-inside', title: 'Shapes through the network', level: 2, kind: 'table',
    text: ['The network of the lessons reads a 1 × 16 × 16 image: a convolution with 6 filters of 5 × 5 and padding 2; max pooling 2 × 2, stride 2; a convolution with 12 filters of 5 × 5 and no padding; max pooling 2 × 2, stride 2. Fill in the number of maps (channels) and the side of each square map after each step.'],
    rows: ['first convolution', 'first pooling', 'second convolution', 'second pooling'], cols: ['channels', 'size'], corner: 'after', cells: fs, tolerance: 0,
    hints: ['A convolution makes as many maps as it has filters; pooling keeps the number of maps.', 'Sizes: (16 + 4 − 5) + 1 = 16; then 16 / 2; then 8 − 5 + 1; then that over 2.', '6 × 16 × 16, 6 × 8 × 8, 12 × 4 × 4, 12 × 2 × 2: 48 numbers for the dense layer.'],
    solution: str(fs), wrong: [Object.assign(str(fs), { 'second convolution|size': '8', 'second pooling|size': '4' }), Object.assign(str(fs), { 'first pooling|channels': '3' })] });

  const ptc = { 'first convolution|weights and biases': npOf(0), 'second convolution|weights and biases': npOf(1), 'dense 48 → 32|weights and biases': npOf(2), 'output 32 → 10|weights and biases': npOf(3), 'the whole network|weights and biases': CNN.numParams() };
  ex({ id: 'cn-x-params-table', chapter: 'cn-inside', title: 'Every weight of the network', level: 3, kind: 'table',
    text: ['Count the weights and biases of the whole network of the lessons, layer by layer: a convolution of 6 filters of 5 × 5 on 1 channel; a convolution of 12 filters of 5 × 5 on 6 channels; a dense layer from 48 numbers to 32; and an output layer from 32 to 10 classes. Pooling and ReLU have none.'],
    rows: Object.keys(ptc).map((k) => k.split('|')[0]), cols: ['weights and biases'], corner: 'layer', cells: ptc, tolerance: 0,
    hints: ['Convolution: filters · (channels · k · k + 1). Dense: inputs · outputs + outputs.', '6 · (25 + 1) and 12 · (150 + 1); 48 · 32 + 32 and 32 · 10 + 10.', npOf(0) + ' + ' + npOf(1) + ' + ' + npOf(2) + ' + ' + npOf(3) + ' = ' + CNN.numParams() + '. The two dense layers hold half of it, though the convolutions do most of the work.'],
    solution: str(ptc), wrong: [Object.assign(str(ptc), { 'first convolution|weights and biases': '150', 'the whole network|weights and biases': String(CNN.numParams() - 6) }), Object.assign(str(ptc), { 'second convolution|weights and biases': '312', 'the whole network|weights and biases': String(CNN.numParams() - npOf(1) + 312) })] });

  ex({ id: 'cn-x-filter-read', chapter: 'cn-inside', title: 'Reading a filter', level: 1, kind: 'choice',
    text: ['A learned first-layer filter, drawn as a picture, has a band of blue (positive weights) across its top rows and a band of vermilion (negative weights) across its bottom rows. Where is its map largest?'],
    options: ['Along level edges with ink above and paper below', 'Along upright strokes', 'On blank paper', 'Everywhere equally'], answer: 0,
    whyNot: { 1: 'Its weights change from top to bottom, not from left to right: an upright stroke under it adds and takes away about equally.', 2: 'Blank paper gives zero products everywhere.', 3: 'A kernel with weights of both signs responds very differently to different patches.' },
    hints: ['Blue adds the pixel, vermilion takes it away.', 'The sum is largest when ink lies under the blue and paper under the vermilion.', 'Ink above, paper below: the lower edge of a level stroke. Filter 3 of the lesson is of this kind.'], solution: 0, wrong: [1, 2] });

  /** receptive field and jump through layers [k, s, d] */
  const field = (layers) => layers.reduce((a, [k, s, d]) => ({ r: a.r + (k - 1) * (d || 1) * a.j, j: a.j * s }), { r: 1, j: 1 });
  const F4 = field([[3, 1], [3, 1], [3, 1], [3, 1]]).r;
  ex({ id: 'cn-x-field', chapter: 'cn-inside', title: 'Four small kernels', level: 2, kind: 'number',
    text: ['Four 3 × 3 convolutions, stride 1, no dilation, one after another. How many pixels wide is the receptive field of a unit in the fourth layer?'],
    answer: F4, tolerance: 0, whyNot: { '12': 'Each layer adds k − 1 = 2, not k = 3; start from 1 pixel.', '81': 'Fields grow by adding, not multiplying: 1 + 2 + 2 + 2 + 2.', '8': 'Start from the single pixel the first layer reads round: 1 + 4 · 2.' },
    hints: ['Start with r = 1; each layer adds (k − 1) · d · jump, and the jump stays 1 at stride 1.', 'Each layer adds 2.', '1 + 2 + 2 + 2 + 2 = ' + F4 + ': four 3 × 3 layers see as far as one 9 × 9 kernel, with 36 weights per channel pair instead of 81.'], solution: String(F4), wrong: ['12', '81'] });

  const FT = [['3 × 3 conv, stride 1', 3, 1], ['3 × 3 conv, stride 2', 3, 2], ['3 × 3 conv, stride 1', 3, 1], ['2 × 2 pool, stride 2', 2, 2], ['3 × 3 conv, stride 1', 3, 1]], ftc = {};
  FT.reduce((a, [name, k, s], i) => { const b = { r: a.r + (k - 1) * a.j, j: a.j * s }; ftc['layer ' + (i + 1) + '|field'] = b.r; ftc['layer ' + (i + 1) + '|jump'] = b.j; return b; }, { r: 1, j: 1 });
  ex({ id: 'cn-x-field-table', chapter: 'cn-inside', title: 'Field and jump, layer by layer', level: 3, kind: 'table',
    text: ['Five layers, no dilation: ' + FT.map((r, i) => (i + 1) + '. ' + r[0]).join('; ') + '. Starting from r₀ = 1 and j₀ = 1, fill in the receptive field and the jump after each layer, with rₗ = rₗ₋₁ + (kₗ − 1) · jₗ₋₁ and jₗ = jₗ₋₁ · sₗ.'],
    rows: FT.map((_, i) => 'layer ' + (i + 1)), cols: ['field', 'jump'], corner: 'after', cells: ftc, tolerance: 0,
    hints: ['A layer’s growth uses the jump from before it; its own stride only affects the layers after.', 'Layer 2 adds 2 · 1 = 2 (field 5), then the jump becomes 2. Layer 3 adds 2 · 2 = 4.', 'Fields ' + FT.map((_, i) => ftc['layer ' + (i + 1) + '|field']).join(', ') + '; jumps ' + FT.map((_, i) => ftc['layer ' + (i + 1) + '|jump']).join(', ') + '.'],
    solution: str(ftc), wrong: [Object.assign(str(ftc), { 'layer 2|field': '7', 'layer 3|field': '11', 'layer 4|field': '13', 'layer 5|field': '21' }), Object.assign(str(ftc), { 'layer 5|jump': '8' })] });

  const FD = field([[3, 1, 1], [3, 1, 1], [3, 1, 2], [3, 1, 4]]).r;
  ex({ id: 'cn-x-field-dilated', chapter: 'cn-inside', title: 'Dilation instead of pooling', level: 2, kind: 'number',
    text: ['Four 3 × 3 convolutions at stride 1 with dilations 1, 1, 2 and 4. How wide is the receptive field after the fourth?'],
    answer: FD, tolerance: 0, whyNot: { '9': 'That is four 3 × 3 layers without dilation; dilation d multiplies what a layer adds by d.', '25': 'Each layer adds (k − 1) · d, not k · d.', '15': 'That is three layers with dilations 1, 2 and 4; here there are four.' },
    hints: ['Each layer adds (k − 1) · d with the jump at 1.', 'The layers add 2, 2, 4 and 8.', '1 + 2 + 2 + 4 + 8 = ' + FD + ', and the map never shrank.'], solution: String(FD), wrong: ['9', '25'] });

  // =========================================================================================================
  // Chapter 3: training, and asking why
  // =========================================================================================================
  const SH = ['circle', 'square', 'triangle', 'star'], CMS = [[72, 1, 2, 0], [3, 64, 9, 1], [0, 4, 70, 1], [1, 0, 2, 71]];
  const recall = CMS[1][1] / CMS[1].reduce((a, b) => a + b, 0), prec = CMS[1][1] / CMS.reduce((a, r) => a + r[1], 0);
  const cmTable = { table: [['true \\ predicted', ...SH]].concat(CMS.map((r, i) => [SH[i], ...r.map(String)])) };
  ex({ id: 'cn-x-cm-recall', chapter: 'cn-train', title: 'How many squares are found?', level: 1, kind: 'number',
    text: ['A CNN’s confusion matrix on 300 validation shapes (rows: the true shape; columns: the prediction):', cmTable, 'What share of the true squares does it call squares (the recall of square)? Give three decimals.'],
    answer: r3(recall), tolerance: 0.0015, whyNot: { [String(r3(prec))]: 'That divides by the square column, the precision; recall divides by the square row.', [String(r3(64 / 300))]: 'Divide by the number of true squares, not all 300 shapes.' },
    hints: ['Recall of a class: the right ones on its row, divided by the row’s total.', 'The square row is 3, 64, 9, 1: 77 true squares.', '64 / 77 = ' + r3(recall) + '. Most of the misses went to triangle.'], solution: String(r3(recall)), wrong: [String(r3(prec)), String(r3(64 / 300))] });

  ex({ id: 'cn-x-cm-worst', chapter: 'cn-train', title: 'The confusion to look at first', level: 1, kind: 'choice',
    text: ['The same confusion matrix:', cmTable, 'Which mistake is the most common?'],
    options: ['Squares taken for triangles', 'Triangles taken for squares', 'Circles taken for triangles', 'Stars taken for triangles'], answer: 0,
    whyNot: { 1: 'That cell holds 4: row triangle, column square.', 2: 'That cell holds 2.', 3: 'That cell holds 2.' },
    hints: ['Look only at the cells off the diagonal.', 'Row is the truth, column the prediction.', 'The largest off-diagonal cell is 9, row square, column triangle. In the app, clicking it shows those 9 images.'], solution: 0, wrong: [1, 2] });

  ex({ id: 'cn-x-augment-choice', chapter: 'cn-train', title: 'A harmful augmentation', level: 1, kind: 'choice',
    text: ['A CNN is trained on images of the digits 0 to 9. Which of these random changes should not be used to augment them?'],
    options: ['Turning images by up to 180°', 'Shifting images by up to 2 pixels', 'Resizing them by up to 10%', 'Thickening or thinning the strokes a little'], answer: 0,
    whyNot: { 1: 'A digit two pixels to the left is the same digit: a good augmentation.', 2: 'A slightly larger or smaller digit is the same digit.', 3: 'Pens differ: stroke width does not change the class.' },
    hints: ['A good augmentation keeps the label true.', 'Which change can turn one digit into another?', 'A 6 turned by 180° is a 9: the label would be wrong for those copies. Small turns are fine.'], solution: 0, wrong: [1, 2] });

  const fewDigits = { data: { set: 'digits', n: 120, seed: 4 }, hidden: { seed: 44, n: 400 }, metric: 'accuracy', target: 0.95 };
  C.exercise(Object.assign({ id: 'cn-x-wf-few', chapter: 'cn-train', title: 'Ninety digits are not many', level: 2, kind: 'workflow',
    text: ['Only 120 drawn digits, a quarter of them held back for testing, leave about ninety to train on: nine of each digit. Build a workflow whose CNN, trained for at most 60 epochs, reads 400 hidden digits with accuracy at least 0.95.'],
    limits: { blocks: ['split', 'cnn', 'train', 'evaluate'], models: ['cnn'], epochs: 60 },
    hints: ['Run a plain CNN for 60 epochs first: it reads only 92% of the hidden digits. Nine examples of each digit are too few to learn every way a digit can be drawn.', 'Make the ninety images go further: the CNN block can augment them, shifting and turning each a little every time it is seen.', 'Split → CNN (augment on) → Train (60 epochs, rate 0.003) → Evaluate: accuracy 0.985. Without augmentation, 0.92.'],
    solution: pipe(fewDigits.data, 'cnn', { augment: true }, { epochs: 60, lr: 0.003 }),
    wrong: [pipe(fewDigits.data, 'cnn', {}, { epochs: 60, lr: 0.003 }), pipe(fewDigits.data, 'cnn', { augment: true }, { epochs: 15, lr: 0.003 })] }, fewDigits));

  const RF = -0.1, RN = 10, resid = f64(() => { let g = T.tensor([1]); for (let k = 0; k < RN; k++) g = T.mul(g, 1 + RF); return one1(g); });
  ex({ id: 'cn-x-residual', chapter: 'cn-train', title: 'Ten blocks with a shortcut', level: 2, kind: 'number',
    text: ['A gradient of 1 goes back through 10 blocks. In each, the layer’s own derivative is f′ = −0.1. In a plain stack each block multiplies the gradient by f′; in a residual stack, where a block outputs h + f(h), by 1 + f′. How large is the gradient after the 10 residual blocks? Give four decimals.'],
    answer: r4(resid), tolerance: 0.0005, whyNot: { '0': 'That is (nearly) the plain stack: (−0.1)¹⁰ = 10⁻¹⁰. The shortcut adds 1 to each factor.', [String(r4(1.1 ** 10))]: 'The factor is 1 + f′ = 0.9, and f′ is negative here.', '0.9': 'That is one block; the gradient passes ten.' },
    hints: ['Each residual block multiplies the gradient by 1 + f′.', '1 + (−0.1) = 0.9, ten times.', '0.9¹⁰ = ' + r4(resid) + ', against (−0.1)¹⁰ = 10⁻¹⁰ for the plain stack: the identity carries the gradient through.'],
    solution: String(r4(resid)), wrong: ['0', String(r4(1.1 ** 10))] });

  ex({ id: 'cn-x-saliency-choice', chapter: 'cn-train', title: 'What a saliency value says', level: 2, kind: 'choice',
    text: ['A saliency map gives a pixel the value 2.9 for the score of class 3. What does that mean?'],
    options: ['Darkening that pixel a little would change the score of 3 about 2.9 times as much as the change of ink', 'The pixel is 2.9 times darker than the average', 'The network is 2.9% sure the image is a 3', 'That pixel alone gives the score 2.9'], answer: 0,
    whyNot: { 1: 'Saliency is about the network’s score, not the image’s brightness.', 2: 'It is a gradient of the score before softmax, not a probability.', 3: 'A gradient says how the score moves when the pixel moves; it is a local slope, not a share of the score.' },
    hints: ['Saliency is |∂ score / ∂ pixel|.', 'A derivative says how fast one number changes as another changes.', 'A local rate: a small change Δ of that pixel moves the score by about 2.9Δ. That is why the lesson tests the map by blanking pixels.'], solution: 0, wrong: [1, 3] });

  const A1 = [[1, 0], [2, 1]], A2 = [[0, 3], [1, 0]], G1 = [[0.2, 0.4], [0.1, 0.1]], G2 = [[-0.3, -0.1], [-0.2, -0.2]];
  const GC = f64(() => {
    const al = [G1, G2].map((G) => one1(T.mean(T.tensor(G))));
    const s = T.add(T.mul(T.tensor(A1), al[0]), T.mul(T.tensor(A2), al[1]));
    return { al, raw: rows(s.values(), 2, 2), cam: rows(T.relu(s).values(), 2, 2).map((r) => r.map(r3)) };
  });
  ex({ id: 'cn-x-gradcam', chapter: 'cn-train', title: 'Grad-CAM on two maps', level: 3, kind: 'matrix',
    text: ['The last convolution of a network has two 2 × 2 maps (after ReLU), A¹ and A², and backpropagation gives the gradient of the class score with respect to each cell, G¹ and G²:', grid(A1, 'A¹'), grid(A2, 'A²'), grid(G1, 'G¹'), grid(G2, 'G²'), 'Compute the Grad-CAM map before scaling: weight each map by the mean of its gradients, add the weighted maps cell by cell, and set negative cells to zero.'],
    rows: ['row 1', 'row 2'], cols: ['column 1', 'column 2'], corner: 'CAM', answer: GC.cam, tolerance: 0.0015,
    hints: ['α₁ = mean of G¹, α₂ = mean of G².', 'α₁ = ' + r3(GC.al[0]) + ' and α₂ = ' + sign(r3(GC.al[1])) + ': map 2 speaks against the class.', 'α₁A¹ + α₂A² = ' + GC.raw.map((r) => r.map((v) => sign(r3(v))).join(', ')).join(' / ') + '; cut at zero: ' + GC.cam.map((r) => r.join(', ')).join(' / ') + '.'],
    solution: sm(GC.cam), wrong: [sm(GC.raw.map((r) => r.map(r3))), sm([[r3(GC.al[0] * 1), 0], [r3(GC.al[0] * 2), r3(GC.al[0])]])] });
})(globalThis.Nooral = globalThis.Nooral || {});
