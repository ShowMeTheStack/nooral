# Reference outputs for the metrics tests, from scikit-learn on fixed seeds.
# Run with the local test environment: ~/.nooral-venv/bin/python tests/ref/gen_metrics.py
# Writes tests/ref/metrics/*.json. NaN is written as "nan" and infinity as "inf".
import json, math, os, warnings
import numpy as np
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import partial_dependence

warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')
os.makedirs(OUT, exist_ok=True)


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return 'nan'
        if math.isinf(v):
            return 'inf' if v > 0 else '-inf'
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def dump(name, data):
    with open(os.path.join(OUT, name + '.json'), 'w') as fh:
        json.dump(clean(data), fh, separators=(',', ':'))
    print(name, len(data))


def safe(fn):
    try:
        return fn()
    except ValueError:
        return 'error'


# --- classification -------------------------------------------------------------------------------
rng = np.random.default_rng(20261018)
cases = []
for c in range(160):
    K = int(rng.integers(2, 6))
    n = int(rng.integers(1, 60))
    yt = rng.integers(0, K, n)
    yp = np.where(rng.random(n) < rng.random(), yt, rng.integers(0, K, n))
    kind = c % 8
    if kind == 1:
        yt[:] = yt[0]                      # all one class in y_true
    elif kind == 2:
        yp[:] = 0                          # one predicted class
    elif kind == 3:
        yp = np.where(yp == K - 1, K, yp)  # a predicted class that never occurs
    yt, yp = [int(v) for v in yt], [int(v) for v in yp]
    labels = sorted(set(yt) | set(yp))
    cm = skm.confusion_matrix(yt, yp, labels=labels)
    case = dict(yTrue=yt, yPred=yp, labels=labels, counts=cm,
                normTrue=skm.confusion_matrix(yt, yp, labels=labels, normalize='true'),
                normPred=skm.confusion_matrix(yt, yp, labels=labels, normalize='pred'),
                normAll=skm.confusion_matrix(yt, yp, labels=labels, normalize='all'),
                accuracy=skm.accuracy_score(yt, yp),
                balanced=skm.balanced_accuracy_score(yt, yp),
                mcc=skm.matthews_corrcoef(yt, yp),
                kappa=skm.cohen_kappa_score(yt, yp))
    for avg in ['macro', 'micro', 'weighted']:
        p, r, f, _ = skm.precision_recall_fscore_support(yt, yp, average=avg, zero_division=0)
        case[avg] = [p, r, f]
    p, r, f, s = skm.precision_recall_fscore_support(yt, yp, average=None, labels=labels, zero_division=0)
    case['perClass'] = [p, r, f, s]
    if set(labels) <= {0, 1}:
        p, r, f, _ = skm.precision_recall_fscore_support(yt, yp, average='binary', pos_label=1, zero_division=0)
        case['binary'] = [p, r, f]
        tn = sum(1 for a, b in zip(yt, yp) if a != 1 and b != 1)
        fp = sum(1 for a, b in zip(yt, yp) if a != 1 and b == 1)
        case['specificity'] = tn / (tn + fp) if tn + fp else 0.0
    cases.append(case)
# string labels
yt = ['cat', 'dog', 'dog', 'eel', 'cat', 'eel', 'dog', 'cat']
yp = ['cat', 'dog', 'cat', 'eel', 'eel', 'eel', 'dog', 'dog']
cases.append(dict(yTrue=yt, yPred=yp, labels=sorted(set(yt) | set(yp)), counts=skm.confusion_matrix(yt, yp),
                  accuracy=skm.accuracy_score(yt, yp), balanced=skm.balanced_accuracy_score(yt, yp),
                  mcc=skm.matthews_corrcoef(yt, yp), kappa=skm.cohen_kappa_score(yt, yp),
                  macro=list(skm.precision_recall_fscore_support(yt, yp, average='macro')[:3]),
                  micro=list(skm.precision_recall_fscore_support(yt, yp, average='micro')[:3]),
                  weighted=list(skm.precision_recall_fscore_support(yt, yp, average='weighted')[:3])))
dump('classification', cases)

# --- curves and probabilistic scores -----------------------------------------------------------------
cases = []
for c in range(140):
    n = int(rng.integers(2, 80))
    y = [int(v) for v in rng.integers(0, 2, n)]
    if c % 10 == 3:
        y = [1] * n
    if c % 10 == 4:
        y = [0] * n
    dec = [1, 2, 3][c % 3]                 # few decimals → many ties
    s = np.clip(np.round(np.array(y) * rng.random() * 0.6 + rng.random(n) * 0.7, dec), 0, 1)
    if c % 10 == 5:
        s = np.full(n, 0.5)               # every score tied
    s = [float(v) for v in s]
    case = dict(y=y, s=s)
    fpr, tpr, th = skm.roc_curve(y, s)
    case['roc'] = dict(fpr=fpr, tpr=tpr, thresholds=th)
    fpr2, tpr2, th2 = skm.roc_curve(y, s, drop_intermediate=False)
    case['rocAll'] = dict(fpr=fpr2, tpr=tpr2, thresholds=th2)
    case['auc'] = skm.roc_auc_score(y, s) if len(set(y)) == 2 else 'nan'
    pr, rc, th = skm.precision_recall_curve(y, s)
    case['pr'] = dict(precision=pr, recall=rc, thresholds=th)
    case['ap'] = skm.average_precision_score(y, s)
    if len(set(y)) == 2:
        case['logloss'] = skm.log_loss(y, s)
    case['brier'] = skm.brier_score_loss(y, s, pos_label=1)
    nb = int(rng.integers(2, 9))
    case['nBins'] = nb
    pt, pp = calibration_curve(y, s, n_bins=nb, strategy='uniform')
    case['calUniform'] = [pt, pp]
    pt, pp = calibration_curve(y, s, n_bins=nb, strategy='quantile')
    case['calQuantile'] = [pt, pp]
    cases.append(case)
dump('curves', cases)

cases = []
for c in range(60):
    K = int(rng.integers(3, 6))
    n = int(rng.integers(K * 3, 70))
    y = [int(v) for v in np.concatenate([np.arange(K), rng.integers(0, K, n - K)])]
    P = rng.random((n, K)) ** 2
    P[np.arange(n), y] += rng.random(n)
    P = P / P.sum(1, keepdims=True)
    P[:, :-1] = np.floor(P[:, :-1] * 20) / 20   # coarse values → ties between classes
    P[:, -1] = 1 - P[:, :-1].sum(1)
    case = dict(y=y, P=P, logloss=skm.log_loss(y, P, labels=list(range(K))), brier=skm.brier_score_loss(y, P, labels=list(range(K))),
                aucMacro=skm.roc_auc_score(y, P, multi_class='ovr', average='macro'),
                aucWeighted=skm.roc_auc_score(y, P, multi_class='ovr', average='weighted'),
                topk=[skm.top_k_accuracy_score(y, P, k=k, labels=list(range(K))) for k in (1, 2, 3)])
    cases.append(case)
dump('multiclass', cases)

# --- regression -----------------------------------------------------------------------------------
cases = []
for c in range(80):
    n = int(rng.integers(1, 50))
    y = rng.normal(3, 2, n)
    if c % 9 == 2:
        y[:] = 4.0
    if c % 9 == 5:
        y[0] = 0.0
    yh = y + rng.normal(rng.normal(), rng.random() * 2, n)
    if c % 9 == 7:
        yh = y.copy()
    case = dict(y=y, yh=yh, mae=skm.mean_absolute_error(y, yh), mse=skm.mean_squared_error(y, yh),
                rmse=skm.root_mean_squared_error(y, yh), mape=skm.mean_absolute_percentage_error(y, yh))
    if n >= 2:
        case['r2'] = skm.r2_score(y, yh)
        case['ev'] = skm.explained_variance_score(y, yh)
    cases.append(case)
dump('regression', cases)

# --- clustering -----------------------------------------------------------------------------------
cases = []
for c in range(60):
    K = int(rng.integers(2, 5))
    n = int(rng.integers(K + 2, 40))
    lab = np.concatenate([np.arange(K), rng.integers(0, K, n - K)])
    X = rng.normal(0, 1, (n, 2)) + lab[:, None] * rng.random() * 3
    other = np.where(rng.random(n) < 0.3, rng.integers(0, K + 1, n), lab)
    if c % 7 == 3:
        other = lab.copy()
    if c % 7 == 4:
        other[:] = 0
    if c % 7 == 5:
        other = (lab + 1) % K            # the same partition renamed
    X = np.round(X, 3)
    lab = [int(v) for v in lab]
    other = [int(v) for v in other]
    case = dict(X=X, labels=lab, other=other, silhouette=skm.silhouette_score(X, lab), silhouetteSamples=skm.silhouette_samples(X, lab),
                db=skm.davies_bouldin_score(X, lab), ari=skm.adjusted_rand_score(lab, other), nmi=skm.normalized_mutual_info_score(lab, other))
    cases.append(case)
dump('clustering', cases)

# --- fairness (plain definitions) ---------------------------------------------------------------------
cases = []
for c in range(30):
    n = int(rng.integers(10, 80))
    g = [['a', 'b', 'c'][int(v)] for v in rng.integers(0, 2 + c % 2, n)]
    y = [int(v) for v in rng.integers(0, 2, n)]
    yp = [int(v) if rng.random() < 0.7 else 1 - int(v) for v in y]
    groups = sorted(set(g))
    sel, tpr, fpr = [], [], []
    for gr in groups:
        ix = [i for i in range(n) if g[i] == gr]
        sel.append(sum(yp[i] for i in ix) / len(ix))
        P = [i for i in ix if y[i] == 1]
        N = [i for i in ix if y[i] == 0]
        tpr.append(sum(yp[i] for i in P) / len(P) if P else 0.0)
        fpr.append(sum(yp[i] for i in N) / len(N) if N else 0.0)
    cases.append(dict(y=y, yp=yp, g=g, dp=max(sel) - min(sel), dt=max(tpr) - min(tpr), df=max(fpr) - min(fpr)))
dump('fairness', cases)

# --- partial dependence of a small regression tree ----------------------------------------------------
X = rng.normal(0, 1, (120, 3))
X[:, 2] = rng.integers(0, 4, 120)        # few distinct values: the grid is those values
yv = X[:, 0] ** 2 + np.sin(2 * X[:, 1]) + 0.5 * X[:, 2] + rng.normal(0, 0.1, 120)
tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, yv)
t = tree.tree_
nodes = [dict(feature=int(t.feature[i]), threshold=float(t.threshold[i]), left=int(t.children_left[i]), right=int(t.children_right[i]),
              value=float(t.value[i][0][0]), cover=float(t.n_node_samples[i])) for i in range(t.node_count)]
pds = []
for feat in range(3):
    r = partial_dependence(tree, X, [feat], grid_resolution=11, kind='both', method='brute')
    pds.append(dict(feature=feat, grid=r['grid_values'][0], average=r['average'][0], ice=r['individual'][0][:5]))
dump('explain', dict(X=X, nodes=nodes, pred=tree.predict(X), pd=pds))
