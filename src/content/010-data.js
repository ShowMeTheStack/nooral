// Part 1, data and features: the chapters and the lessons. The exercises are in 011-data-ex.js; the model challenge,
// the tour, the course and the pages the diagram checker looks at are in 012-data-ch.js. Each lesson opens with a
// question, runs the topic that shows the idea at the frame that makes the point, asks for a prediction, sums up and
// leads to its exercises. Every value quoted here is one the topics compute from their own inputs with the engines.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('da-table')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'da-tables', part: 'data', n: 1, title: 'Tables and what is in them', blurb: 'Rows, features and labels; one column summed up by its centre, spread and shape; and pairs of columns read off a scatter matrix and a correlation.' });
  C.chapter({ id: 'da-clean', part: 'data', n: 2, title: 'Gaps and wild values', blurb: 'Cells that are missing and what to fill them with, and values far from the rest: how to find them and what to do about them.' });
  C.chapter({ id: 'da-transform', part: 'data', n: 3, title: 'Scales, codes and bins', blurb: 'Putting columns on comparable scales, turning category names into numbers, and cutting a number into ranges.' });
  C.chapter({ id: 'da-features', part: 'data', n: 4, title: 'Making, choosing and leaking features', blurb: 'New columns from old ones, keeping only the columns that help, and the quiet ways information from the test rows leaks into training.' });

  // =========================================================================================================
  // Chapter 1: tables and what is in them
  // =========================================================================================================
  C.lesson({
    id: 'da-table', chapter: 'da-tables', title: 'A table of examples', minutes: 6,
    summary: 'A dataset is a table: one row per example, one column per feature, and a label column holding what we want to predict. A column holds amounts or category names, and the kind of label decides the task: a class means classification, a number means regression.',
    blocks: [
      'A farming co-operative wants to know, before harvest, how much wheat each plot will yield. It has records from past seasons. What exactly does a model learn from, and what does it predict?',
      { def: 'example, feature, label', text: 'an example is one row of the table, a feature is one column measured the same way for every row, and the label is the column we want to predict for new rows.' },
      { topic: 'data-table', frame: 0, caption: 'The valley harvest table: 300 rows and 4 feature columns, plus the label "yield". Every row is one plot in one season.' },
      { topic: 'data-table', frame: 3, caption: '"soil" is a category with 3 values. It is stored as the codes 0 to 2, but sand is not "2 times" clay: the codes are names, not amounts.' },
      { predict: { q: 'The label "yield" is a number of tonnes per hectare, from 3.15 to 7.82. What kind of task is predicting it?', options: ['Classification', 'Regression', 'Clustering'], answer: 1, explain: 'A label that is an amount makes a regression task. If the co-operative only wanted "good" or "poor" plots, the label would be a class and the task classification.' } },
      { topic: 'data-table', frame: 5, caption: 'To a model the table is a matrix X of 300 rows by 4 columns and a vector y of 300 labels. The category column must be turned into numbers first.' },
      { topic: 'data-table', input: { set: 'digits', rows: 8 }, frame: 1, inputs: true, caption: 'Images fit the same mould: read a 28 × 28 digit row by row and it becomes one row of 784 pixel columns, with the digit as the label. Switch the dataset to see loans (two classes) or penguins (with gaps).' },
      { h: 'Recap' },
      { list: ['Rows are examples, columns are features, and one column is the label.', 'A column holds amounts (numbers you can compare and add) or categories (names from a short list).', 'A class label means classification; a numeric label means regression.'] },
      X('da-x-task'), X('da-x-types'),
    ],
  });

  C.lesson({
    id: 'da-describe', chapter: 'da-tables', title: 'One column at a glance', minutes: 7,
    summary: 'Before modelling, look at each column: the median and quartiles (a box), the mean and standard deviation, and a histogram. When a long tail pulls the mean away from the median, the column is skewed, and values beyond the whiskers deserve a closer look.',
    blocks: [
      'A bus company hands over 500 trips. Before any model, what is a quick and honest way to see what one column holds?',
      { topic: 'data-describe', frame: 1, caption: 'Sorted, the middle value is the median, 40 passengers. The box holds the middle half of the values, from Q1 = 19 to Q3 = 59.3.' },
      { topic: 'data-describe', frame: 2, caption: 'The mean is 39.4 and the standard deviation 23.4. The mean nearly agrees with the median, so the column is roughly symmetric.' },
      { topic: 'data-describe', frame: 3, caption: 'The histogram cuts the range into 12 bins of width 6.67; the fullest, 46.7 to 53.3, holds 50 trips. A flat shape, no long tail.' },
      { math: 'position of the quantile q among n sorted values (counting from 0):  (n − 1) · q\nbetween two values, interpolate:  Q = v[i] + (position − i) · (v[i + 1] − v[i])' },
      { predict: { q: 'Years in work of 600 loan applicants: most have worked a few years, a few for decades. Where does the mean sit compared with the median?', options: ['Above the median', 'Below the median', 'On the median'], answer: 0, explain: 'A long tail of large values pulls the mean up. Here the median is 4.05 years and the mean 5.94: the column is skewed right.' } },
      { topic: 'data-describe', input: { set: 'loans', col: 2, bins: 15 }, frame: 2, caption: 'Years employed: the mean, 5.94, sits well above the median, 4.05. A long tail of large values pulls it up.' },
      { topic: 'data-describe', input: { set: 'loans', col: 2, bins: 15 }, frame: 4, caption: '32 values lie beyond the whiskers, which reach 1.5 box widths past the box. They are candidates for a closer look, not automatically errors.' },
      { tip: 'The median and the quartiles barely move when one value goes wild; the mean and the standard deviation can move a long way. Report both kinds when a column is skewed.' },
      { h: 'Recap' },
      { list: ['The box runs from Q1 to Q3; its width is the interquartile range (IQR).', 'The mean is the balance point, the standard deviation the typical distance from it.', 'Mean well above the median: a long tail to the right.'] },
      X('da-x-five'), X('da-x-skew'),
    ],
  });

  C.lesson({
    id: 'da-relations', chapter: 'da-tables', title: 'Columns in pairs', minutes: 8,
    summary: 'A scatter matrix draws every pair of columns, coloured by class, and shows which columns separate the classes and which move together. Pearson’s r sums the products of centred values into one number between −1 and 1; it misses curves and one far point can swing it.',
    blocks: [
      'Four measurements of 150 iris flowers. Which of them tell the species apart, and which two say nearly the same thing?',
      'A scatter matrix draws every pair of columns in a grid, one colour per class, with each column’s own histogram on the diagonal.',
      { topic: 'data-scatter-matrix', frame: 1, caption: 'On iris the diagonal shows that petal length (η² = 0.94 of its variance lies between the species) and petal width (0.93) separate the species far better than sepal width (0.4).' },
      { topic: 'data-scatter-matrix', frame: 8, caption: 'The whole matrix: petal length and petal width move almost in step, so a model gains little from the second once it has the first.' },
      { h: 'Pearson’s r' },
      { math: 'r = Σ (xᵢ − x̄)(yᵢ − ȳ) / √( Σ (xᵢ − x̄)² · Σ (yᵢ − ȳ)² )' },
      { topic: 'data-correlation', frame: 1, caption: 'Centre both columns on their means. 145 points lie in the quarters where both are above or both below the mean (positive products), 5 in the others.' },
      { topic: 'data-correlation', frame: 3, caption: 'Dividing by the spreads removes the units: r = 0.963. On ranks instead of values, Spearman’s r is 0.938.' },
      { topic: 'data-correlation', frame: 4, caption: 'The same r for every pair of the four iris columns. The strongest pair, petal length with petal width, carries overlapping information: a model gains little from the second once it has the first.' },
      { predict: { q: 'Wheat yield rises with rain and then falls again when the roots flood. What is Pearson’s r between rain and yield likely to be?', options: ['Close to 1: a strong pattern', 'Close to 0: little straight-line relation', 'Close to −1'], answer: 1, explain: 'r measures how well a straight line fits. A rise-then-fall pattern is strong but not straight: r is −0.155 on the harvest table.' } },
      { topic: 'data-correlation', input: { pair: 'rain' }, frame: 3, caption: 'Rain against yield: r = −0.155, yet yield clearly depends on rain. A straight-line measure barely sees a curve.' },
      { topic: 'data-correlation', input: { pair: 'outlier' }, frame: 4, caption: 'Twenty points that fall, and one far point: r is 0.741 with it and −0.846 without it. Always look at the scatter, not just the number.' },
      { h: 'Recap' },
      { list: ['A scatter matrix shows separation (the diagonal) and relations (the panels) at once.', 'r runs from −1 to 1 and measures straight-line association only.', 'One far point can flip r; Spearman’s r, on ranks, is steadier.'] },
      X('da-x-corr'), X('da-x-corr-trap'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: gaps and wild values
  // =========================================================================================================
  C.lesson({
    id: 'da-missing', chapter: 'da-clean', title: 'Missing values', minutes: 8,
    summary: 'Most models cannot take a gap. Dropping the rows with a gap can throw away most of the table; filling each gap with the column’s mean, median or most frequent value keeps every row but shrinks the column’s spread. The fill values are learned from the training rows only.',
    blocks: [
      'A sensor drops out, a form field is left blank. The model in the next step will refuse a table with holes in it. Throw the rows away, or guess what was there?',
      { topic: 'data-missing', frame: 0, caption: '83 of the 800 cells are missing (red), spread over 71 of the 200 rows. The column "sun" has 29 gaps; its 171 present values have mean 502 and sd 117.' },
      { topic: 'data-missing', input: { set: 'rents', how: 'drop' }, frame: 1, caption: 'Flat rents with gaps in every column: dropping every row with a gap keeps only 59 of 150 rows, and throws away 308 perfectly good cells with them.' },
      { predict: { q: 'Every gap in "sun" is filled with the mean of the present values. What happens to the column’s standard deviation?', options: ['It shrinks', 'It grows', 'It stays the same'], answer: 0, explain: 'Every filled value sits exactly at the mean, adding no spread while adding rows: the sd falls from 117 to 108. The column looks more certain than it is.' } },
      { topic: 'data-missing', frame: 2, caption: 'The filled values (green) all land in one bin: a spike at the mean.' },
      { topic: 'data-missing', input: { set: 'harvest', how: 'knn' }, frame: 2, caption: 'Filling from the 5 most similar rows instead spreads the filled values over 7 bins, and the sd falls only to 111.' },
      { warn: 'The fill values are learned from the training rows, then applied unchanged to the test rows. A mean worked out over all rows lets the test rows shape the training data: a small leak, but a leak.' },
      { workflow: 'prepare-pumps', live: true, height: 400, caption: 'Pump readings with gaps, prepared on the canvas: the split comes first, and Impute learns its fill values from the training rows only. Open Impute to see them.' },
      { h: 'Recap' },
      { list: ['Dropping rows is simple but can lose most of a table when gaps are spread over many columns.', 'Mean, median or most-frequent filling keeps every row but piles values up at one point.', 'Learn the fill values on the training rows; apply them to every other row.'] },
      X('da-x-drop-count'), X('da-x-impute'), X('da-x-impute-sd'), X('da-x-wf-gaps'),
    ],
  });

  C.lesson({
    id: 'da-outliers', chapter: 'da-clean', title: 'Outliers', minutes: 7,
    summary: 'Two rules flag values far from the rest: beyond 1.5 interquartile ranges outside the box, or more than 3 standard deviations from the mean. The second is fooled by the outliers themselves. A flagged value may be an error to fix, or the very event you are looking for.',
    blocks: [
      'A botany club typed in 150 stem heights. One of them reads 1600 cm. Is it a sixteen-metre wildflower, or a slip of the finger?',
      { topic: 'data-outliers', frame: 0, caption: 'Stem heights from 10.1 to 1600 cm. The box runs from Q1 = 20.7 to Q3 = 29.3, so the IQR is 8.6.' },
      { topic: 'data-outliers', frame: 1, caption: 'The fences sit 1.5 × IQR beyond the box, at 7.78 and 42.2. One value lies outside.' },
      { predict: { q: 'The 1600 cm value pulls the mean to 35.3 and stretches the standard deviation to 128. Where does the "3 standard deviations" fence land?', options: ['Near 42, like the box-plot fence', 'Far out, at about 420', 'Below the mean'], answer: 1, explain: '35.3 + 3 × 128 ≈ 420. The outlier inflates the very spread used to judge it. Here the 1600 still lies beyond, but with a little less extreme a value it would hide behind its own sd.' } },
      { topic: 'data-outliers', input: { data: 'typo', rule: 'z', k: 3, act: 'drop' }, frame: 1, caption: 'The 3 sd rule on the same values: the fences sit at −350 and 420, so wide that only the most extreme value can cross them.' },
      { topic: 'data-outliers', frame: 2, caption: 'Clipping pulls the flagged value in to the fence: the mean drops from 35.3 to 25 and the sd from 128 to 6.29, while the median stays at 25.' },
      { topic: 'data-outliers', input: { data: 'pumps', rule: 'iqr', k: 1.5, act: 'clip' }, frame: 2, caption: 'On pump temperatures the 14 flagged minutes were all real faults. Here the outliers are what we want to find: clipping them away would hide them.' },
      { h: 'Recap' },
      { list: ['Box-plot rule: outside Q1 − 1.5·IQR or Q3 + 1.5·IQR. It is not moved by the outliers themselves.', '3 sd rule: the outliers stretch the sd and can hide behind it.', 'Fix a typing error; keep a real rare event; clip only when an extreme value should not dominate.'] },
      X('da-x-fences'), X('da-x-zfence'), X('da-x-outlier-act'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: scales, codes and bins
  // =========================================================================================================
  C.lesson({
    id: 'da-scaling', chapter: 'da-transform', title: 'Scaling, and why distances need it', minutes: 8,
    summary: 'A distance adds up differences in whatever units the columns have, so a column in the thousands decides every distance alone. Standard scaling uses the mean and standard deviation, min-max scaling maps onto [0, 1], robust scaling uses the median and IQR. All three learn their numbers from the training rows.',
    blocks: [
      'Two wines differ by 300 in proline and by 0.5 in hue. To a distance, the first gap is 600 times larger. Is it 600 times more important?',
      { topic: 'data-scaling', frame: 1, caption: 'On raw columns the cloud is a thin line: the 5 nearest neighbours of the big test point are the rows with the closest proline, and they vote for the wrong cultivar.' },
      { math: 'standard:  z = (x − mean) / sd\nmin-max:   z = (x − min) / (max − min)\nrobust:    z = (x − median) / IQR' },
      { topic: 'data-scaling', frame: 2, caption: 'Standard scaling learns two numbers per column from the 124 training rows only, the mean and the sd (proline 741 and 308, hue 0.95 and 0.233), and transforms every row, training or test, with those same numbers.' },
      { topic: 'data-scaling', frame: 3, caption: 'After standard scaling both columns count. All 5 neighbours change, and they vote for the right cultivar.' },
      { predict: { q: 'One proline value is mistyped as 8880. Which scaler squeezes the other values the most?', options: ['Standard', 'Min-max', 'Robust'], answer: 1, explain: 'Min-max divides by max − min, and the wild value sets the max: every other proline lands in a narrow band near 0. The median and IQR of robust scaling ignore it.' } },
      { topic: 'data-scaling', input: { data: 'outlier', scaler: 'minmax' }, frame: 3, caption: 'Min-max with the wild value: its range is 8602, so the rest of the column is squeezed into a narrow band.' },
      { topic: 'data-scaling', input: { data: 'outlier', scaler: 'robust' }, frame: 3, caption: 'Robust scaling uses the median (674) and the IQR (435), which one wild value cannot move: the ordinary rows keep their spread and the wild one lands far out, at 18.9.' },
      { topic: 'data-scaling', frame: 4, caption: 'On the 54 held-out rows, 5-NN is right 75.9% of the time on raw columns and 88.9% after scaling.' },
      { tip: 'Trees split one column at a time, so scaling does not change them. It matters for anything that measures distances or adds weighted columns: k-NN, SVMs, regularised linear models and networks.' },
      { h: 'Recap' },
      { list: ['Standard: subtract the mean, divide by the sd.', 'Min-max: onto [0, 1] for the training rows; new rows can fall outside.', 'Robust: median and IQR, for columns with outliers. Always fitted on the training rows.'] },
      X('da-x-zscore'), X('da-x-minmax-test'), X('da-x-scaler-outliers'), X('da-x-code-minmax'), X('da-x-code-standard'),
    ],
  });

  C.lesson({
    id: 'da-encoding', chapter: 'da-transform', title: 'Categories as numbers', minutes: 7,
    summary: 'Ordinal encoding gives each category one code: compact, but a linear model then treats the categories as points on a line, in that order and equally spaced. One-hot encoding gives each category its own 0/1 column, so each gets its own effect. Ordinal suits categories with a real order; one-hot suits names.',
    blocks: [
      'Soil is loam, clay or sand. A model needs numbers. Is clay = 0, loam = 1, sand = 2 good enough?',
      { topic: 'data-encoding', frame: 1, caption: 'Ordinal encoding in alphabetical order: clay 0, loam 1, sand 2. One column of numbers.' },
      { predict: { q: 'A linear model on that one code column must change the yield by the same step from clay to loam as from loam to sand. Mean yields are clay 5.46, loam 6.16 and sand 4.74. Can a straight line through the codes fit them?', options: ['Yes', 'No: loam is highest, in the middle of the codes'], answer: 1, explain: 'The yields rise and then fall along the codes, which no straight line follows. The line fitted has a mean squared error of 0.735.' } },
      { topic: 'data-encoding', frame: 2, caption: 'The best straight line through the codes: yield ≈ 5.84 − 0.308 × code, with mean squared error 0.735.' },
      { topic: 'data-encoding', frame: 3, caption: 'One-hot encoding: 3 columns of 0 and 1, a single 1 per row. They always add up to 1, so a linear model with an intercept drops one of them.' },
      { topic: 'data-encoding', frame: 4, caption: 'With one-hot columns each soil gets its own level, its mean: mean squared error 0.466 against 0.735.' },
      { topic: 'data-encoding', input: { data: 'rooms', order: 'given' }, frame: 2, inputs: true, caption: 'When the categories have a real order, such as numbers of rooms, ordinal codes in that order work well. Try the days of the week: is Sunday next to Monday?' },
      { h: 'Recap' },
      { list: ['Ordinal: one column, implies an order and equal steps.', 'One-hot: a 0/1 column per category, no order implied; drop one with an intercept.', 'Use ordinal for sizes and grades, one-hot for names.'] },
      X('da-x-encode-names'), X('da-x-encode-order'), X('da-x-onehot'), X('da-x-onehot-count'),
    ],
  });

  C.lesson({
    id: 'da-binning', chapter: 'da-transform', title: 'Binning a number', minutes: 6,
    summary: 'Binning cuts a number into ranges. Equal-width bins split the axis evenly; quantile bins hold equal numbers of rows. One-hot encode the bins and a linear model fits one level per bin: a staircase that can follow bends a straight line cannot.',
    blocks: [
      'A straight line cannot bend. Can it still follow a wave, if we change what it is given?',
      { topic: 'data-binning', frame: 0, caption: '120 points along a wave. A straight line on the 84 training rows misses the bends: test MSE 0.238.' },
      { topic: 'data-binning', frame: 1, caption: 'Equal-width edges cut the training range into 6 bins of width 0.99. They hold from 10 to 19 rows each.' },
      { predict: { q: 'Each bin becomes a one-hot column and a linear model is fitted on them. What does it predict inside a bin?', options: ['A sloped line through the bin', 'One constant: the mean of the bin’s training rows', 'The value at the bin’s centre on the true curve'], answer: 1, explain: 'With one 0/1 column per bin, the model can only give each bin its own level, and least squares makes that level the bin’s mean.' } },
      { topic: 'data-binning', frame: 2, caption: 'One level per bin: the steps follow the wave, test MSE 0.12 against 0.238 for the line.' },
      { topic: 'data-binning', input: { data: 'sine', how: 'quantile', n: 6 }, frame: 1, inputs: true, caption: 'Quantile bins hold equal numbers of rows, so their widths vary. Raise the number of bins to 24 and watch the steps start to follow the noise.' },
      { h: 'Recap' },
      { list: ['Equal width: even edges, uneven counts. Quantile: even counts, uneven edges.', 'One-hot bins let a linear model fit a staircase.', 'More bins follow the data more closely, noise included.'] },
      X('da-x-bins'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: making, choosing and leaking features
  // =========================================================================================================
  C.lesson({
    id: 'da-new-features', chapter: 'da-features', title: 'New columns from old', minutes: 8,
    summary: 'A linear model is linear in its columns, not in the world. Give it x, x², x³ and it fits a curve; give it x1·x2 and it can separate XOR. The degree is a dial: the training error only falls as it rises, while the test error falls and then climbs.',
    blocks: [
      'Least squares draws straight lines. Is a curved relation out of its reach?',
      { topic: 'data-poly', frame: 1, caption: 'Polynomial features turn the column x into x, x², x³. For x = 1.13 the row becomes 1.13, 1.28, 1.46. The model stays linear in these columns.' },
      { topic: 'data-poly', frame: 2, caption: 'Least squares on the 3 columns fits a cubic: test MSE 0.0361, against 0.226 for the straight line.' },
      { predict: { q: 'Raise the degree from 1 to 12, with 28 training points. What happens to the test error?', options: ['It keeps falling', 'It falls, then climbs', 'It stays flat after degree 3'], answer: 1, explain: 'The training error only falls (0.013 at degree 12), but the test error is lowest at degree 4 (0.0223) and explodes to 294 at degree 12: the curve bends to follow every noisy point.' } },
      { topic: 'data-poly', frame: 3, caption: 'Every degree from 1 to 12: the test error is lowest at degree 4 and climbs steeply after.' },
      { h: 'Interactions' },
      { topic: 'data-interactions', frame: 1, caption: 'XOR: no straight line in (x1, x2) works. Logistic regression gets 62% right, little better than guessing.' },
      { topic: 'data-interactions', frame: 3, caption: 'Add the single column x1·x2, whose sign is the class rule: 97% right, with the largest weight on the new column.' },
      { workflow: 'features-moons', title: 'Polynomial terms on two moons', text: 'Compare a straight boundary with and without squares and cubes of the two columns.' },
      { h: 'Recap' },
      { list: ['Powers of a column let a linear model fit curves.', 'Products of columns (interactions) let it model effects that depend on two columns together.', 'Choose the degree by held-out error: the training error always prefers more.'] },
      X('da-x-poly-row'), X('da-x-poly-count'), X('da-x-poly-curve'), X('da-x-interaction'),
    ],
  });

  C.lesson({
    id: 'da-select', chapter: 'da-features', title: 'Keeping the columns that help', minutes: 7,
    summary: 'A filter scores each column alone, quickly, but cannot see redundancy or teamwork. Forward selection adds the column that most improves the cross-validated score, one at a time. Lasso drives useless weights to exactly zero while it fits. Selection must happen inside each fold, or the score is optimistic.',
    blocks: [
      'Five extra columns of pure noise are added to the rents table. They cannot help; can they hurt?',
      { topic: 'data-select', frame: 0, caption: 'A filter scores each column alone by its correlation with the rent: area 0.895, rooms 0.836, distance 0.344. The noise columns score at most 0.11.' },
      { topic: 'data-select', frame: 2, caption: 'Forward selection, step 2: each remaining column is tried beside area with 5-fold cross-validation. Distance wins, R² 0.94, although its correlation with the rent alone is only 0.344.' },
      { predict: { q: 'The filter ranked rooms above distance. Why does forward selection pick distance second?', options: ['Rooms says much the same as area; distance adds something new', 'Forward selection prefers columns with smaller values', 'It is random'], answer: 0, explain: 'Rooms and area move together, so once area is in, rooms adds little. Distance is nearly unrelated to area and explains what area cannot. A filter, judging each column alone, cannot see this.' } },
      { topic: 'data-select', frame: 4, caption: 'Lasso with α = 8 sets 5 of the 10 weights to exactly 0, including all 5 noise columns.' },
      { topic: 'data-select', frame: 5, caption: 'With 3 columns: filter 0.942, forward 0.949, lasso 0.942, against 0.953 with all 10. The forward score chose the columns, so it is a little optimistic.' },
      { warn: 'If columns are chosen by looking at every row’s label and then cross-validated, the score is biased upwards. The next lesson shows how far.' },
      { h: 'Recap' },
      { list: ['Filter: fast, one column at a time, blind to redundancy.', 'Forward selection: judges columns together, at the price of many fits.', 'Lasso and trees select while they fit.'] },
      X('da-x-select-filter'),
    ],
  });

  C.lesson({
    id: 'da-leakage', chapter: 'da-features', title: 'Leakage, shown happening', minutes: 8,
    summary: 'A test score means something only if nothing about the test rows reached training. Choosing columns on all the labels, a column recorded after the outcome, copies of an example on both sides of the split, and preparation fitted on all rows all leak. On pure noise a leaky pipeline scored 73%; an honest one 54%.',
    blocks: [
      'A model scores 73% in cross-validation on data that holds no signal at all. How is that possible?',
      { topic: 'data-leakage', frame: 0, caption: '100 rows of 400 columns of pure noise, and labels shuffled at random. No model can beat 50% on new rows.' },
      { topic: 'data-leakage', frame: 1, caption: 'The 10 columns most correlated with the label over all 100 rows reach |r| = 0.224 to 0.311, purely by chance.' },
      { predict: { q: 'A pipeline keeps those 10 columns, then cross-validates a model on them. What score do you expect?', options: ['About 50%', 'Well above 50%', 'Below 50%'], answer: 1, explain: 'The columns were chosen because they happen to line up with these very labels, test folds included. The five folds average 73%.' } },
      { topic: 'data-leakage', frame: 3, caption: 'The honest pipeline chooses its 10 columns afresh inside each training fold: a mean of 54%, close to guessing.' },
      { topic: 'data-leakage', frame: 4, caption: 'On 100 brand-new noise rows the leaky model scores 41%: its 73% was an illusion.' },
      { h: 'Four ways to leak' },
      { list: ['**Selection or tuning on all rows:** choose columns or settings inside the training folds only.', '**A column recorded after the outcome:** keep only what is known at the moment of prediction.', '**Copies on both sides:** split by patient, customer or plot, not by row.', '**Preparation fitted on all rows:** fit scalers, imputers and encoders on the training rows.'] },
      { topic: 'data-leakage', input: { leak: 'target' }, frame: 0, inputs: true, caption: 'Choose another leak above to watch it happen: a column that comes after the label, copies of a row, or scaling on all rows.' },
      { workflow: 'leakage-loans', title: 'Leakage on the canvas', text: 'Scaling and feature selection placed before the split. Open Scale to read the warning, then move the split first.' },
      { h: 'Recap' },
      { list: ['Everything that learns from data learns from the training rows only.', 'Ask of every column: would I know this at prediction time?', 'A score too good to be true usually is.'] },
      X('da-x-leak-later'), X('da-x-leak-spot'), X('da-x-leak-score'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
