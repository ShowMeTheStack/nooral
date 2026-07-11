"""Reference outputs for Nooral's classical algorithms (tests/classic.mjs).

Run with the local test environment:  ~/.nooral-venv/bin/python tests/ref/gen_classic.py
Every dataset is generated here from fixed seeds (plus iris and wine, which ship with scikit-learn),
and stored with the outputs so the JavaScript side reads exactly the same numbers.
"""
import json
import os
import warnings

import numpy as np
from sklearn import datasets, cluster, decomposition, discriminant_analysis as da, ensemble, manifold
from sklearn import linear_model as lm, mixture, model_selection as ms, naive_bayes as nb, neighbors, svm, tree
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore', message='Some inputs do not have OOB scores')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classic')
os.makedirs(OUT, exist_ok=True)


def tolist(v):
    if isinstance(v, dict):
        return {k: tolist(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [tolist(x) for x in v]
    if isinstance(v, np.ndarray):
        return tolist(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def save(name, obj):
    with open(os.path.join(OUT, name + '.json'), 'w') as fh:
        json.dump(tolist(obj), fh, separators=(',', ':'))


# ---------------- datasets ----------------
rs = np.random.RandomState(7)
D = {}
Xr = rs.normal(size=(50, 4))
D['reg'] = dict(X=Xr, y=Xr @ np.array([1.5, -2.0, 0.0, 0.7]) + 3.0 + 0.5 * rs.normal(size=50))
X2 = np.vstack([rs.normal([0, 0], 1.0, size=(30, 2)), rs.normal([2.2, 1.6], 1.0, size=(30, 2))])
D['cls2'] = dict(X=X2, y=np.array([0] * 30 + [1] * 30))
X3 = np.vstack([rs.normal(c, 0.9, size=(30, 3)) for c in ([0, 0, 0], [2, 2, 0], [0, 2.5, 2])])
D['cls3'] = dict(X=X3, y=np.repeat([0, 1, 2], 30))
iris = datasets.load_iris()
D['iris'] = dict(X=iris.data, y=iris.target)
wine = datasets.load_wine()
W = (wine.data - wine.data.mean(0)) / wine.data.std(0)
D['wine'] = dict(X=W, y=wine.target)
rates = np.array([[5, 1, 1, 3, 0.5, 2, 1, 1], [1, 4, 2, 1, 2, 0.5, 1, 1], [1, 1, 4, 1, 1, 1, 3, 0.3]])
D['counts'] = dict(X=np.vstack([rs.poisson(rates[k], size=(20, 8)) for k in range(3)]).astype(float), y=np.repeat([0, 1, 2], 20))
Xb, yb = datasets.make_blobs(n_samples=90, centers=[[0, 0], [4, 1], [1, 4]], cluster_std=0.8, random_state=3)
D['blobs'] = dict(X=Xb, y=yb)
Xm, ym = datasets.make_moons(n_samples=100, noise=0.1, random_state=2)
D['moons'] = dict(X=Xm, y=ym)
Xw = rs.uniform(-3, 3, size=(80, 1))
D['wave'] = dict(X=Xw, y=np.sin(Xw[:, 0]) * 2 + 0.3 * rs.normal(size=80))
Xt = rs.uniform(0, 10, size=(120, 3))
D['tree3'] = dict(X=Xt, y=((Xt[:, 0] > 5) ^ (Xt[:, 1] + 0.5 * Xt[:, 2] > 6)).astype(int) + (Xt[:, 2] > 8).astype(int))
D['treg'] = dict(X=Xt, y=np.sin(Xt[:, 0]) + 0.3 * Xt[:, 1] + 0.2 * rs.normal(size=120))
save('data', D)
T = lambda k: (D[k]['X'], D[k]['y'])


# ---------------- linear models ----------------
def linear():
    X, y = T('reg')
    r = {}
    m = lm.LinearRegression().fit(X, y)
    r['ols'] = dict(coef=m.coef_, intercept=m.intercept_)
    m = lm.Ridge(alpha=2.5).fit(X, y)
    r['ridge'] = dict(alpha=2.5, coef=m.coef_, intercept=m.intercept_)
    m = lm.Lasso(alpha=0.3, tol=1e-12, max_iter=100000).fit(X, y)
    r['lasso'] = dict(alpha=0.3, coef=m.coef_, intercept=m.intercept_)
    m = lm.ElasticNet(alpha=0.3, l1_ratio=0.4, tol=1e-12, max_iter=100000).fit(X, y)
    r['enet'] = dict(alpha=0.3, l1Ratio=0.4, coef=m.coef_, intercept=m.intercept_)
    alphas, coefs, _ = lm.lasso_path(X - X.mean(0), y - y.mean(), alphas=30, eps=1e-3, tol=1e-12, max_iter=100000)
    r['path'] = dict(alphas=alphas, coefs=coefs.T)
    pf = PolynomialFeatures(degree=3)
    Xp = pf.fit_transform(X[:5, :2])
    r['poly'] = dict(X=X[:5, :2], out=Xp, names=pf.get_feature_names_out().tolist())
    Xw, yw = T('wave')
    Pw = PolynomialFeatures(degree=5, include_bias=False).fit_transform(Xw)
    m = lm.LinearRegression().fit(Pw, yw)
    r['polyreg'] = dict(coef=m.coef_, intercept=m.intercept_)
    X2, y2 = T('cls2')
    m = lm.LogisticRegression(C=0.7, tol=1e-12, max_iter=10000).fit(X2, y2)
    r['logit2'] = dict(C=0.7, coef=m.coef_, intercept=m.intercept_, proba=m.predict_proba(X2))
    Xi, yi = T('iris')
    m = lm.LogisticRegression(C=1.0, tol=1e-12, max_iter=100000).fit(Xi, yi)
    r['logit3'] = dict(C=1.0, coef=m.coef_, intercept=m.intercept_, proba=m.predict_proba(Xi))
    Xs, ys = T('wine')
    m = lm.LogisticRegression(C=0.2, tol=1e-12, max_iter=100000).fit(Xs, ys)
    r['logitWine'] = dict(C=0.2, proba=m.predict_proba(Xs))
    m = lm.Perceptron(shuffle=False, eta0=1.0, max_iter=7, tol=None).fit(X2, y2)
    r['perceptron'] = dict(coef=m.coef_, intercept=m.intercept_, epochs=7, pred=m.predict(X2))
    m = lm.Perceptron(shuffle=False, eta0=0.5, max_iter=4, tol=None).fit(Xi, yi)
    r['perceptron3'] = dict(coef=m.coef_, intercept=m.intercept_, epochs=4, eta=0.5)
    save('linear', r)


def generative():
    r = {}
    for key in ('cls3', 'iris'):
        X, y = T(key)
        r[key] = dict(
            lda_svd=da.LinearDiscriminantAnalysis().fit(X, y).predict_proba(X),
            lda_lsqr=da.LinearDiscriminantAnalysis(solver='lsqr').fit(X, y).predict_proba(X),
            lda_coef=da.LinearDiscriminantAnalysis(solver='lsqr').fit(X, y).coef_,
            lda_evr=da.LinearDiscriminantAnalysis().fit(X, y).explained_variance_ratio_,
            qda=da.QuadraticDiscriminantAnalysis().fit(X, y).predict_proba(X),
            qda_reg=da.QuadraticDiscriminantAnalysis(reg_param=0.1).fit(X, y).predict_proba(X),
            gnb=nb.GaussianNB().fit(X, y).predict_proba(X),
            gnb_var=nb.GaussianNB().fit(X, y).var_,
        )
    X, y = T('counts')
    for a in (1.0, 0.1):
        m = nb.MultinomialNB(alpha=a).fit(X, y)
        r['mnb_%g' % a] = dict(alpha=a, logp=m.feature_log_prob_, proba=m.predict_proba(X))
    save('generative', r)


def knn():
    X, y = T('cls3')
    Q = X[::3] + 0.05
    r = {'Q': Q}
    for name, kw in dict(
            e5u=dict(n_neighbors=5), e7d=dict(n_neighbors=7, weights='distance'),
            m4u=dict(n_neighbors=4, metric='manhattan'), p3d=dict(n_neighbors=6, p=3, weights='distance'),
            cheb=dict(n_neighbors=5, metric='chebyshev')).items():
        m = neighbors.KNeighborsClassifier(algorithm='brute', **kw).fit(X, y)
        dist, idx = m.kneighbors(Q)
        r[name] = dict(params=kw, proba=m.predict_proba(Q), idx=idx, dist=dist)
    Xw, yw = T('wave')
    m = neighbors.KNeighborsRegressor(n_neighbors=6, weights='distance').fit(Xw, yw)
    r['reg'] = dict(Q=Xw[::4] + 0.01, pred=m.predict(Xw[::4] + 0.01))
    save('knn', r)


def export_tree(t):
    tt = t.tree_
    return dict(left=tt.children_left, right=tt.children_right, feature=tt.feature, threshold=tt.threshold,
                impurity=tt.impurity, n=tt.n_node_samples, value=tt.value[:, 0, :] if tt.value.shape[2] > 1 else tt.value[:, 0, 0],
                depth=t.get_depth(), leaves=t.get_n_leaves())


def split_seed(random_state):
    # the state a tree's splitter starts from, for a given integer random_state
    return int(np.random.RandomState(random_state).randint(0, 2**31 - 1))


def trees():
    r = {'seed0': split_seed(0)}
    X, y = T('tree3')
    for crit in ('gini', 'entropy'):
        for depth in (3, None):
            m = tree.DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=0).fit(X, y)
            r['%s_%s' % (crit, depth)] = dict(tree=export_tree(m), imp=m.feature_importances_)
    m = tree.DecisionTreeClassifier(min_samples_leaf=5, random_state=0).fit(X, y)
    r['msl5'] = dict(tree=export_tree(m))
    path = tree.DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
    r['ccp_path'] = dict(alphas=path.ccp_alphas, impurities=path.impurities)
    m = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.01).fit(X, y)
    r['ccp'] = dict(alpha=0.01, tree=export_tree(m))
    Xr, yr = T('treg')
    m = tree.DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xr, yr)
    r['reg4'] = dict(tree=export_tree(m), imp=m.feature_importances_)
    m = tree.DecisionTreeRegressor(min_samples_leaf=4, random_state=0).fit(Xr, yr)
    r['reg_msl4'] = dict(tree=export_tree(m))
    path = tree.DecisionTreeRegressor(random_state=0, max_depth=6).cost_complexity_pruning_path(Xr, yr)
    r['reg_ccp_path'] = dict(alphas=path.ccp_alphas, impurities=path.impurities)
    Xi, yi = T('iris')
    m = tree.DecisionTreeClassifier(random_state=0).fit(Xi, yi)
    r['iris'] = dict(depth=m.get_depth(), leaves=m.get_n_leaves(), acc=m.score(Xi, yi))
    save('trees', r)


def ensembles():
    from sklearn.ensemble._gradient_boosting import _random_sample_mask
    r = {}
    X, y = T('tree3')
    n = len(y)
    rf = ensemble.RandomForestClassifier(n_estimators=15, oob_score=True, random_state=3).fit(X, y)
    seeds = [e.random_state for e in rf.estimators_]
    r['rf'] = dict(boot=[np.random.RandomState(s).randint(0, n, n) for s in seeds], split=[split_seed(s) for s in seeds],
                   proba=rf.predict_proba(X[::5]), oob=rf.oob_score_, oob_dec=rf.oob_decision_function_, imp=rf.feature_importances_)
    Xr, yr = T('treg')
    rr = ensemble.RandomForestRegressor(n_estimators=10, max_depth=5, oob_score=True, random_state=4).fit(Xr, yr)
    seeds = [e.random_state for e in rr.estimators_]
    r['rfr'] = dict(boot=[np.random.RandomState(s).randint(0, n, n) for s in seeds], split=[split_seed(s) for s in seeds],
                    pred=rr.predict(Xr[::5]), oob=rr.oob_score_, imp=rr.feature_importances_)
    for key, data in (('ada2', 'cls2'), ('ada3', 'tree3')):
        Xa, ya = T(data)
        ab = ensemble.AdaBoostClassifier(n_estimators=12, random_state=1).fit(Xa, ya)
        r[key] = dict(split=[split_seed(e.random_state) for e in ab.estimators_], alphas=ab.estimator_weights_[:len(ab.estimators_)],
                      errors=ab.estimator_errors_[:len(ab.estimators_)], dec=ab.decision_function(Xa), pred=ab.predict(Xa))

    def replay(seed, stages, K, sub, n):
        rs = np.random.RandomState(seed)
        masks, split = [], []
        for _ in range(stages):
            if sub < 1:
                masks.append(_random_sample_mask(n, max(1, int(sub * n)), rs).astype(bool))
            for _ in range(K):
                split.append(int(rs.randint(0, 2**31 - 1)))
        return masks, split
    for key, sub in (('gbr', 1.0), ('gbr_sub', 0.7)):
        g = ensemble.GradientBoostingRegressor(n_estimators=25, learning_rate=0.2, max_depth=3, subsample=sub, random_state=5).fit(Xr, yr)
        masks, split = replay(5, 25, 1, sub, n)
        r[key] = dict(sub=sub, masks=masks, split=split, pred=g.predict(Xr), staged=[p for i, p in enumerate(g.staged_predict(Xr)) if i in (0, 9, 24)])
    X2, y2 = T('cls2')
    g = ensemble.GradientBoostingClassifier(n_estimators=20, learning_rate=0.3, max_depth=2, random_state=6).fit(X2, y2)
    r['gbc2'] = dict(split=replay(6, 20, 1, 1.0, len(y2))[1], dec=g.decision_function(X2), proba=g.predict_proba(X2))
    g = ensemble.GradientBoostingClassifier(n_estimators=15, learning_rate=0.2, max_depth=2, subsample=0.8, random_state=8).fit(X, y)
    masks, split = replay(8, 15, 3, 0.8, n)
    r['gbc3'] = dict(masks=masks, split=split, dec=g.decision_function(X), proba=g.predict_proba(X))
    Xm, ym = T('moons')
    out = np.vstack([Xm, [[3, 3], [-2, 2], [2.5, -1.5], [-1.5, -1.5]]])
    iso = ensemble.IsolationForest(n_estimators=200, random_state=0).fit(out)
    r['iso'] = dict(X=out, score=iso.score_samples(out))
    r['rf_stats'] = dict(oob=[ensemble.RandomForestClassifier(n_estimators=60, oob_score=True, random_state=s).fit(X, y).oob_score_ for s in range(5)])
    save('ensemble', r)


def svms():
    r = {}
    X2, y2 = T('cls2')
    Xm, ym = T('moons')
    Xi, yi = T('iris')
    cases = dict(
        lin=('cls2', dict(kernel='linear', C=0.5)), poly=('cls2', dict(kernel='poly', degree=3, gamma=0.5, coef0=1.0, C=1.0)),
        rbf=('moons', dict(kernel='rbf', gamma=2.0, C=3.0)), rbf_scale=('moons', dict(kernel='rbf', C=1.0)),
        sig=('cls2', dict(kernel='sigmoid', gamma=0.1, coef0=0.0, C=1.0)),
        iris_rbf=('iris', dict(kernel='rbf', gamma=0.3, C=2.0)), iris_lin=('iris', dict(kernel='linear', C=0.3)))
    for key, (data, kw) in cases.items():
        X, y = T(data)
        m = svm.SVC(tol=1e-8, **kw).fit(X, y)
        out = dict(params=kw, data=data, support=m.support_, n_support=m.n_support_, pred=m.predict(X), gamma=m._gamma)
        if len(m.classes_) > 2:
            out['ovo'] = svm.SVC(tol=1e-8, decision_function_shape='ovo', **kw).fit(X, y).decision_function(X)
            out['ovr_shape'] = m.decision_function(X)
        else:
            out['dec'] = m.decision_function(X)
        r[key] = out
    from sklearn.multiclass import OneVsRestClassifier
    m = OneVsRestClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=2.0, tol=1e-8)).fit(Xi, yi)
    r['iris_ovr'] = dict(dec=m.decision_function(Xi), pred=m.predict(Xi))
    Xw, yw = T('wave')
    for key, kw in dict(svr_rbf=dict(kernel='rbf', gamma=0.5, C=5.0, epsilon=0.2), svr_lin=dict(kernel='linear', C=1.0, epsilon=0.5)).items():
        m = svm.SVR(tol=1e-8, **kw).fit(Xw, yw)
        r[key] = dict(params=kw, pred=m.predict(Xw), support=m.support_, dual=m.dual_coef_[0], intercept=m.intercept_[0])
    Xb, _ = T('blobs')
    m = svm.OneClassSVM(kernel='rbf', gamma=0.5, nu=0.1, tol=1e-8).fit(Xb)
    r['ocsvm'] = dict(dec=m.decision_function(Xb), support=m.support_, pred=m.predict(Xb))
    save('svm', r)


def unsupervised():
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.manifold import trustworthiness
    from sklearn.manifold._t_sne import _joint_probabilities
    from sklearn.metrics import pairwise_distances
    r = {}
    Xb, yb = T('blobs')
    Xi, yi = T('iris')
    for key, X, init in (('blobs', Xb, Xb[[0, 1, 2]]), ('iris', Xi, Xi[[0, 50, 100]]), ('iris_bad', Xi, Xi[[0, 1, 2]])):
        km = cluster.KMeans(n_clusters=3, init=init, n_init=1, algorithm='lloyd').fit(X)
        r['km_' + key] = dict(init=init, labels=km.labels_, inertia=km.inertia_, centers=km.cluster_centers_, n_iter=km.n_iter_,
                              sil=silhouette_score(X, km.labels_), sil_samples=silhouette_samples(X, km.labels_))
    r['km_best'] = dict(inertia=cluster.KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xi).inertia_)
    Xm, ym = T('moons')
    for key, X, eps, ms in (('db_moons', Xm, 0.2, 5), ('db_moons2', Xm, 0.12, 4), ('db_blobs', Xb, 0.6, 6)):
        db = cluster.DBSCAN(eps=eps, min_samples=ms).fit(X)
        r[key] = dict(eps=eps, min_samples=ms, labels=db.labels_, core=db.core_sample_indices_)
    Xs = Xb[::3]
    r['agg_X'] = Xs
    for link in ('single', 'complete', 'average', 'ward'):
        ag = cluster.AgglomerativeClustering(n_clusters=3, linkage=link, compute_distances=True).fit(Xs)
        r['agg_' + link] = dict(children=ag.children_, distances=ag.distances_, labels=ag.labels_)
    for ct in ('full', 'diag', 'spherical'):
        K = 3
        means0 = Xb[[0, 1, 2]]
        if ct == 'full':
            cov0 = np.array([np.eye(2) * 1.5] * K); prec0 = np.array([np.linalg.inv(c) for c in cov0])
        elif ct == 'diag':
            cov0 = np.ones((K, 2)) * 1.5; prec0 = 1 / cov0
        else:
            cov0 = np.ones(K) * 1.5; prec0 = 1 / cov0
        g = mixture.GaussianMixture(n_components=K, covariance_type=ct, means_init=means0, weights_init=np.ones(K) / K, precisions_init=prec0, tol=1e-6, max_iter=500).fit(Xb)
        r['gmm_' + ct] = dict(means0=means0, cov0=cov0, means=g.means_, weights=g.weights_, cov=g.covariances_, lower=g.lower_bound_, bic=g.bic(Xb), n_iter=g.n_iter_, labels=g.predict(Xb))
    Xw = T('wine')[0]
    for key, X in (('iris', Xi), ('wine', Xw)):
        pca = decomposition.PCA().fit(X)
        comps = pca.components_.copy()
        for c in comps:
            if c[np.argmax(np.abs(c))] < 0:
                c *= -1
        r['pca_' + key] = dict(components=comps, var=pca.explained_variance_, ratio=pca.explained_variance_ratio_, sv=pca.singular_values_, mean=pca.mean_)
    p2 = decomposition.PCA(n_components=2).fit(Xi)
    r['pca_iris']['recon'] = p2.inverse_transform(p2.transform(Xi[:10]))
    r['pca_iris']['n95'] = decomposition.PCA(n_components=0.95).fit(Xw).n_components_
    Xt = Xi[::3]
    rs = np.random.RandomState(0)
    init = 1e-4 * rs.standard_normal((len(Xt), 2))
    D2 = pairwise_distances(Xt, squared=True)
    P = _joint_probabilities(D2, 10.0, 0)
    ts = manifold.TSNE(perplexity=10.0, method='exact', init=init, max_iter=500, learning_rate='auto').fit(Xt)
    r['tsne'] = dict(X=Xt, init=init, P=P, kl=ts.kl_divergence_, Y=ts.embedding_, n_iter=ts.n_iter_, trust=trustworthiness(Xt, ts.embedding_, n_neighbors=5))
    Xbig = np.vstack([Xb, Xi[:, :2] + 8])
    tb = manifold.TSNE(perplexity=20.0, method='barnes_hut', init='pca', random_state=0).fit(Xbig)
    r['tsne_bh'] = dict(X=Xbig, kl=tb.kl_divergence_, trust=trustworthiness(Xbig, tb.embedding_, n_neighbors=5))
    save('unsupervised', r)


def selection():
    r = {}
    Xi, yi = T('iris')
    Xw, yw = T('wine')
    Xr, yr = T('reg')
    r['kfold'] = [dict(train=a, test=b) for a, b in ms.KFold(4).split(Xr)]
    r['kfold7'] = [dict(train=a, test=b) for a, b in ms.KFold(7).split(np.zeros(52))]
    r['skf_wine'] = [dict(train=a, test=b) for a, b in ms.StratifiedKFold(5).split(Xw, yw)]
    yodd = np.array([2, 0, 1, 1, 2, 0, 0, 1, 2, 2, 2, 1, 0, 2, 1, 1, 2, 0, 2, 2, 1, 0, 2])
    r['skf_odd'] = dict(y=yodd, folds=[dict(train=a, test=b) for a, b in ms.StratifiedKFold(3).split(np.zeros(len(yodd)), yodd)])
    tr, te = ms.train_test_split(np.arange(len(yw)), test_size=0.3, stratify=yw, random_state=0)
    r['tts'] = dict(test_counts=np.bincount(yw[te]), train_counts=np.bincount(yw[tr]))
    seed0 = split_seed(0)
    r['seed0'] = seed0
    r['cv_logit'] = ms.cross_val_score(lm.LogisticRegression(C=1.0, tol=1e-12, max_iter=10000), Xi, yi, cv=5)
    r['cv_tree'] = ms.cross_val_score(tree.DecisionTreeClassifier(random_state=0), Xw, yw, cv=5)
    r['cv_ridge'] = ms.cross_val_score(lm.Ridge(alpha=1.0), Xr, yr, cv=5)
    gs = ms.GridSearchCV(neighbors.KNeighborsClassifier(algorithm='brute'), {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}, cv=5).fit(Xw, yw)
    r['grid'] = dict(mean=gs.cv_results_['mean_test_score'], std=gs.cv_results_['std_test_score'], best=gs.best_index_)
    sizes, trs, tes = ms.learning_curve(tree.DecisionTreeClassifier(random_state=0, max_depth=3), Xw, yw, cv=5)
    r['lc'] = dict(sizes=sizes, train=trs, test=tes)
    trs, tes = ms.validation_curve(tree.DecisionTreeClassifier(random_state=0), Xw, yw, param_name='max_depth', param_range=[1, 2, 3, 4, 6], cv=5)
    r['vc'] = dict(train=trs, test=tes)
    save('selection', r)


linear()
generative()
knn()
trees()
ensembles()
svms()
unsupervised()
selection()
