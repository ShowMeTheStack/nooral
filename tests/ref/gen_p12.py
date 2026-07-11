# Reference numbers for the topics of parts 1 and 2 (data and features; splitting and evaluation), from
# scikit-learn, numpy and scipy, for the jobs written by tests/ref/p12_jobs.mjs (tests/ref/p12/inputs.json).
# Run with the local test environment: ~/.nooral-venv/bin/python tests/ref/gen_p12.py
# Writes tests/ref/p12/sklearn.json: for each case, one result per job, in the same order.
import json, math, os, warnings
import numpy as np
from scipy import stats
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, KBinsDiscretizer, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p12')


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
        return None if (math.isnan(v) or math.isinf(v)) else v
    return v


def arr(a):
    return np.array([[np.nan if v is None else v for v in r] for r in a], dtype=float) if a and isinstance(a[0], list) else np.array([np.nan if v is None else v for v in a], dtype=float)


def describe(j):
    x = arr(j['x'])
    counts, edges = np.histogram(x, bins=j['bins'], range=(x.min(), x.max()))
    return dict(mean=x.mean(), sd=x.std(), min=x.min(), max=x.max(), q1=np.percentile(x, 25), med=np.percentile(x, 50), q3=np.percentile(x, 75), counts=counts, edges=edges)


def pearson(j):
    x, y = arr(j['x']), arr(j['y'])
    return dict(r=np.corrcoef(x, y)[0, 1], spearman=stats.spearmanr(x, y).statistic)


def eta(j):
    X, y = arr(j['X']), np.array(j['y'])
    out = []
    for c in range(X.shape[1]):
        x = X[:, c]
        m = x.mean()
        between = sum((y == k).sum() * (x[y == k].mean() - m) ** 2 for k in np.unique(y))
        out.append(between / ((x - m) ** 2).sum())
    return dict(eta=out, R=np.corrcoef(X.T))


def impute(j):
    X, cat = arr(j['X']), j['cat']
    res = {}
    for how in ['mean', 'median', 'most_frequent', 'knn']:
        imp = KNNImputer(n_neighbors=5) if how == 'knn' else SimpleImputer(strategy=how)
        Z = imp.fit_transform(X)
        # category columns are always filled with their most frequent value
        mf = SimpleImputer(strategy='most_frequent').fit_transform(X)
        for c, isc in enumerate(cat):
            if isc:
                Z[:, c] = mf[:, c]
        res[how] = Z
    keep = ~np.isnan(X).any(axis=1)
    res['dropKept'] = int(keep.sum())
    return res


def scaling(j):
    Xtr, Xte = arr(j['Xtr']), arr(j['Xte'])
    out = {}
    for name, sc in [('standard', StandardScaler()), ('minmax', MinMaxScaler()), ('robust', RobustScaler())]:
        sc.fit(Xtr)
        centre = sc.mean_ if name == 'standard' else sc.data_min_ if name == 'minmax' else sc.center_
        scale = sc.scale_ if name == 'standard' else sc.data_range_ if name == 'minmax' else sc.scale_
        m = KNeighborsClassifier(n_neighbors=j['k'], algorithm='brute').fit(sc.transform(Xtr), j['ytr'])
        out[name] = dict(centre=centre, scale=scale, acc=m.score(sc.transform(Xte), j['yte']))
    m = KNeighborsClassifier(n_neighbors=j['k'], algorithm='brute').fit(Xtr, j['ytr'])
    out['raw'] = dict(acc=m.score(Xte, j['yte']))
    return out


def encoding(j):
    ci, y, code, k = np.array(j['ci']), arr(j['y']), j['code'], j['k']
    lo = LinearRegression().fit(np.array([[code[c]] for c in ci], dtype=float), y)
    oh = np.eye(k)[ci][:, 1:]
    lh = LinearRegression().fit(oh, y)
    po = lo.predict(np.array([[code[c]] for c in range(k)], dtype=float))
    ph = lh.predict(np.eye(k)[:, 1:])
    mse = lambda p: float(np.mean((y - p[ci]) ** 2))
    return dict(coef=lo.coef_[0], intercept=lo.intercept_, predOrd=po, predOh=ph, mseOrd=mse(po), mseOh=mse(ph))


def binning(j):
    xtr, ytr, xte, yte = arr(j['xtr']), arr(j['ytr']), arr(j['xte']), arr(j['yte'])
    kb = KBinsDiscretizer(n_bins=j['n'], encode='onehot-dense', strategy=j['strategy'], quantile_method='linear') if j['strategy'] == 'quantile' else KBinsDiscretizer(n_bins=j['n'], encode='onehot-dense', strategy=j['strategy'])
    kb.fit(xtr.reshape(-1, 1))
    edges = kb.bin_edges_[0]
    lr = LinearRegression().fit(kb.transform(xtr.reshape(-1, 1)), ytr)
    line = LinearRegression().fit(xtr.reshape(-1, 1), ytr)
    mse = lambda m, A, y: float(np.mean((y - m.predict(A)) ** 2))
    return dict(edges=edges, binTr=mse(lr, kb.transform(xtr.reshape(-1, 1)), ytr), binTe=mse(lr, kb.transform(xte.reshape(-1, 1)), yte),
                lineTr=mse(line, xtr.reshape(-1, 1), ytr), lineTe=mse(line, xte.reshape(-1, 1), yte))


def poly(j):
    xtr, ytr, xte, yte = arr(j['xtr']), arr(j['ytr']), arr(j['xte']), arr(j['yte'])
    tr, te = [], []
    for d in range(1, 13):
        pf = PolynomialFeatures(degree=d, include_bias=False)
        A, B = pf.fit_transform(xtr.reshape(-1, 1)), pf.transform(xte.reshape(-1, 1))
        m = LinearRegression().fit(A, ytr)
        tr.append(np.mean((ytr - m.predict(A)) ** 2)); te.append(np.mean((yte - m.predict(B)) ** 2))
    return dict(tr=tr, te=te)


def logreg(j):
    X, y = arr(j['X']), np.array(j['y'])
    m = LogisticRegression(C=j['C'], max_iter=100000, tol=1e-12).fit(X, y)
    return dict(acc=m.score(X, y), coef=m.coef_[0])


def select(j):
    X, y = arr(j['X']), arr(j['y'])
    filt = [abs(np.corrcoef(X[:, c], y)[0, 1]) for c in range(X.shape[1])]
    out = dict(filt=filt)
    if j['reg']:
        Z = StandardScaler().fit_transform(X)
        out['lasso'] = Lasso(alpha=j['alpha'], tol=1e-12, max_iter=1000000).fit(Z, y).coef_
    return out


def binary(j):
    y, s, t = np.array(j['y']), arr(j['s']), j['t']
    pred = (s >= t).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(tp=tp, fp=fp, fn=fn, tn=tn, auc=skm.roc_auc_score(y, s), ap=skm.average_precision_score(y, s),
               precision=skm.precision_score(y, pred, zero_division=0), recall=skm.recall_score(y, pred), f1=skm.f1_score(y, pred, zero_division=0),
               logloss=skm.log_loss(y, s, labels=[0, 1]), brier=skm.brier_score_loss(y, s), accuracy=skm.accuracy_score(y, pred))
    # the curve over a fine grid of thresholds, as the precision–recall topic draws it
    grid = [0.05 + k * 0.01 for k in range(91)]
    out['f1grid'] = [skm.f1_score(y, (s >= g).astype(int), zero_division=0) for g in grid]
    if 'nBins' in j:
        pt, pp = calibration_curve(y, s, n_bins=j['nBins'], strategy=j['strategy'])
        out['calTrue'], out['calPred'] = pt, pp
    return out


def multi(j):
    yt, yp, labels = j['yt'], j['yp'], j['labels']
    cm = skm.confusion_matrix(yt, yp, labels=labels)
    p, r, f, s = skm.precision_recall_fscore_support(yt, yp, labels=labels, average=None, zero_division=0)
    out = dict(counts=cm, normTrue=skm.confusion_matrix(yt, yp, labels=labels, normalize='true'), normPred=np.nan_to_num(skm.confusion_matrix(yt, yp, labels=labels, normalize='pred')),
               precision=p, recall=r, f1=f, support=s, accuracy=skm.accuracy_score(yt, yp), balanced=skm.balanced_accuracy_score(yt, yp), mcc=skm.matthews_corrcoef(yt, yp))
    for avg in ['macro', 'weighted', 'micro']:
        out[avg] = list(skm.precision_recall_fscore_support(yt, yp, labels=labels, average=avg, zero_division=0)[:3])
    spec = []
    for k in range(len(labels)):
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        spec.append(tn / (tn + fp) if tn + fp else 0.0)
    out['specificity'] = spec
    return out


def regression(j):
    y, p = arr(j['y']), arr(j['p'])
    return dict(mae=skm.mean_absolute_error(y, p), mse=skm.mean_squared_error(y, p), rmse=math.sqrt(skm.mean_squared_error(y, p)), r2=skm.r2_score(y, p))


def model_of(name):
    return dict(knn=lambda: KNeighborsClassifier(n_neighbors=5, algorithm='brute'), nb=lambda: GaussianNB(var_smoothing=1e-9), tree=lambda: DecisionTreeClassifier(max_depth=3, random_state=0))[name]()


def cv(j):
    X, y = arr(j['X']), np.array(j['y'])
    out = []
    for te in j['folds']:
        mask = np.zeros(len(y), bool); mask[te] = True
        m = model_of(j['model']).fit(X[~mask], y[~mask])
        out.append(m.score(X[mask], y[mask]))
    return dict(scores=out)


def svcgrid(j):
    X, y = arr(j['X']), np.array(j['y'])
    res = {}
    for C in j['C']:
        for g in j['gamma']:
            sc = []
            for te in j['folds']:
                mask = np.zeros(len(y), bool); mask[te] = True
                sc.append(SVC(C=C, gamma=g, kernel='rbf', tol=1e-3).fit(X[~mask], y[~mask]).score(X[mask], y[mask]))
            res['%g|%g' % (C, g)] = float(np.mean(sc))
    return dict(grid=res)


def vcknn(j):
    X, y = arr(j['X']), np.array(j['y'])
    tr, te = [], []
    for k in j['values']:
        a, b = [], []
        for f in j['folds']:
            mask = np.zeros(len(y), bool); mask[f] = True
            m = KNeighborsClassifier(n_neighbors=k, algorithm='brute').fit(X[~mask], y[~mask])
            a.append(m.score(X[~mask], y[~mask])); b.append(m.score(X[mask], y[mask]))
        tr.append(np.mean(a)); te.append(np.mean(b))
    return dict(train=tr, test=te)


KINDS = dict(describe=describe, pearson=pearson, corrmat=lambda j: dict(R=np.corrcoef(arr(j['X']).T)), eta=eta, impute=impute, scaling=scaling, encoding=encoding,
             binning=binning, poly=poly, logreg=logreg, select=select, binary=binary, multi=multi, regression=regression, cv=cv, svcgrid=svcgrid, vcknn=vcknn)

if __name__ == '__main__':
    with open(os.path.join(HERE, 'inputs.json')) as fh:
        cases = json.load(fh)
    out = {key: [clean(KINDS[j['kind']](j)) for j in jobs] for key, jobs in cases.items()}
    with open(os.path.join(HERE, 'sklearn.json'), 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print(len(out), 'cases')
