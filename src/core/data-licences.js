// The classic open datasets bundled with Nooral: their licences, the attribution text for NOTICE, and loaders.
// Each licence was checked on the official source page before the set was bundled. The handwritten-digit set
// that is often shipped with teaching tools is not bundled: its official page states no licence, and the
// licences claimed for it elsewhere disagree, so Nooral's own generated digits (data-img.js) are used instead.
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});
  const CHANGES = 'Changes: stored in a compact numeric form (integers with a per-column number of decimals); category columns are stored as indices.';

  D.licences = {
    iris: { name: 'Iris', licence: 'CC BY 4.0', licenceUrl: 'https://creativecommons.org/licenses/by/4.0/', source: 'https://archive.ics.uci.edu/dataset/53/iris', checked: '2026-10-18',
      attribution: 'Fisher, R. A. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76. Licensed under CC BY 4.0. Values as in the original 1936 paper (two entries differ from the repository file). ' + CHANGES },
    wine: { name: 'Wine', licence: 'CC BY 4.0', licenceUrl: 'https://creativecommons.org/licenses/by/4.0/', source: 'https://archive.ics.uci.edu/dataset/109/wine', checked: '2026-10-18',
      attribution: 'Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J. Licensed under CC BY 4.0. ' + CHANGES },
    breastCancer: { name: 'Breast Cancer Wisconsin (Diagnostic)', licence: 'CC BY 4.0', licenceUrl: 'https://creativecommons.org/licenses/by/4.0/', source: 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic', checked: '2026-10-18',
      attribution: 'Wolberg, W., Mangasarian, O., Street, N. & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Licensed under CC BY 4.0. ' + CHANGES },
    penguins: { name: 'Palmer Penguins', licence: 'CC0 1.0', licenceUrl: 'https://creativecommons.org/publicdomain/zero/1.0/', source: 'https://allisonhorst.github.io/palmerpenguins/', checked: '2026-10-18',
      attribution: 'Horst, A. M., Hill, A. P. & Gorman, K. B. (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. https://doi.org/10.5281/zenodo.3960218. Data collected and made available by Dr. Kristen Gorman and the Palmer Station Antarctica LTER (Palmer Station Antarctica LTER and K. Gorman, 2020; Environmental Data Initiative, https://doi.org/10.6073/pasta/98b16d7d563f265cb52372c8ca99e60f, https://doi.org/10.6073/pasta/7fca67fb28d56ee2ffa3d9370ebda689, https://doi.org/10.6073/pasta/c14dfcfada8ea13a17536e73eb6fbe9e). Available under CC0 in accordance with the Palmer Station LTER Data Policy; credited here as a courtesy.' },
  };
  /** sets considered and left out, with the reason */
  D.excluded = {
    handwrittenDigits: 'The well-known 28×28 handwritten-digit database is not bundled: its official page states no licence, and other listings give CC BY-SA 3.0, CC BY-SA 4.0, CC BY 4.0 or CC0. Nooral generates its own digits instead.',
  };
  /** NOTICE-ready text for the bundled datasets */
  D.notice = () => ['Bundled datasets', '', 'Nooral bundles the following open datasets. All other datasets in Nooral are generated or written for Nooral and are covered by its own licence.', '',
    ...Object.values(D.licences).map((l) => '- ' + l.name + ' (' + l.licence + ', ' + l.licenceUrl + '): ' + l.attribution), ''].join('\n');

  const decode = (raw) => {
    const X = raw.rows.split(/[;\n]/).filter(Boolean).map((r) => r.split(',').map((v, j) => (v === '' ? NaN : +v / 10 ** raw.dec[j])));
    return { X, y: Array.from(raw.y, Number) };
  };
  const num = (name, unit) => ({ name, type: 'number', unit });
  const BC = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'];
  const SETS = {
    iris: { title: 'Iris', story: 'Measurements of 150 iris flowers of three species, 50 of each: the oldest classic set in statistics teaching.', features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].map((n) => num(n, 'cm')), target: 'species', classes: ['setosa', 'versicolor', 'virginica'] },
    wine: { title: 'Wine', story: 'Chemical analysis of 178 wines grown in one region of Italy from three cultivars.', features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280_od315', 'proline'].map((n) => num(n, '')), target: 'cultivar', classes: ['cultivar 1', 'cultivar 2', 'cultivar 3'] },
    breastCancer: { title: 'Breast Cancer Wisconsin (Diagnostic)', story: '569 cell-nucleus images from breast masses, described by 30 measurements (the mean, standard error and worst of ten features); the target is the diagnosis.', features: [...BC.map((n) => num('mean_' + n, '')), ...BC.map((n) => num(n + '_error', '')), ...BC.map((n) => num('worst_' + n, ''))], target: 'diagnosis', classes: ['malignant', 'benign'] },
    penguins: { title: 'Palmer Penguins', story: '344 penguins of three species on three islands near Palmer Station, Antarctica, measured from 2007 to 2009. A few measurements and sexes are missing.', features: [{ name: 'island', type: 'category', categories: ['Biscoe', 'Dream', 'Torgersen'] }, num('bill_length', 'mm'), num('bill_depth', 'mm'), num('flipper_length', 'mm'), num('body_mass', 'g'), { name: 'sex', type: 'category', categories: ['female', 'male'] }, num('year', '')], target: 'species', classes: ['Adelie', 'Chinstrap', 'Gentoo'] },
  };
  for (const [key, s] of Object.entries(SETS)) {
    const lic = D.licences[key], raw = () => (D.classicRaw || {})[key];
    const counts = raw() ? Array.from(raw().y).reduce((m, c) => ((m[c] = (m[c] || 0) + 1), m), {}) : {};
    D.register(key, { title: s.title, kind: 'classic', story: s.story, rule: 'Real measurements, bundled unchanged except for the storage form.', source: lic.source, licence: lic.licence, attribution: lic.attribution,
      size: (raw() ? raw().y.length : '?') + ' rows, ' + s.features.length + ' features', features: s.features.map((f) => f.name), target: s.target + ' (' + s.classes.join(', ') + ')',
      balance: s.classes.map((c, i) => c + ' ' + (counts[i] || 0)).join(', ') }, () => {
      const { X, y } = decode(raw());
      return D.table(X, y, s.features, { name: s.target, type: 'category', categories: s.classes }, { classes: s.classes });
    });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
