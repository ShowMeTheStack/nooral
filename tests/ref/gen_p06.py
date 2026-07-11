"""Fit every job of tests/ref/p06/jobs.json with scikit-learn and write tests/ref/p06/sk.json (see gen_p06.mjs)."""
import json
import os

import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p06')
jobs = json.load(open(os.path.join(HERE, 'jobs.json')))
out = {}
for j in jobs:
    X = np.array(j['X'], dtype=float)
    p = j['params']
    kw = dict(kernel=p['kernel'], C=p.get('C', 1.0), gamma=p['gamma'], degree=p.get('degree', 3), coef0=p.get('coef0', 0.0), tol=1e-8)
    r = {}
    if j['kind'] == 'svc':
        y = np.array(j['y'])
        k = len(set(j['y']))
        if k == 2:
            m = svm.SVC(**kw).fit(X, y)
            r['dec'] = m.decision_function(X).tolist()
            r['support'] = sorted(m.support_.tolist())
            r['pred'] = m.predict(X).tolist()
            r['alpha'] = [[int(i), float(abs(a))] for i, a in zip(m.support_, m.dual_coef_[0])]
            if 'Xe' in j:
                r['dec_e'] = m.decision_function(np.array(j['Xe'])).tolist()
            if p['kernel'] == 'linear':
                r['w'] = m.coef_[0].tolist()
                r['b'] = float(m.intercept_[0])
        elif p.get('multiClass') == 'ovr':
            m = OneVsRestClassifier(svm.SVC(**kw)).fit(X, y)
            r['dec'] = m.decision_function(X).tolist()
            r['pred'] = m.predict(X).tolist()
            r['nsv'] = [len(e.support_) for e in m.estimators_]
            if 'Xe' in j:
                r['dec_e'] = m.decision_function(np.array(j['Xe'])).tolist()
        else:
            m = svm.SVC(decision_function_shape='ovo', **kw).fit(X, y)
            r['pred'] = m.predict(X).tolist()
            r['support'] = sorted(m.support_.tolist())
            if 'Xe' in j:
                r['dec_e'] = m.decision_function(np.array(j['Xe'])).tolist()
            # one machine per pair, fitted on its own, for the support vectors of each
            cl = sorted(set(j['y']))
            r['pair_nsv'] = []
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    sel = (y == cl[a]) | (y == cl[b])
                    r['pair_nsv'].append(len(svm.SVC(**kw).fit(X[sel], y[sel]).support_))
    elif j['kind'] == 'svr':
        m = svm.SVR(kernel=p['kernel'], C=p['C'], gamma=p['gamma'], epsilon=p['epsilon'], tol=1e-8).fit(X, np.array(j['y']))
        r['pred'] = m.predict(X).tolist()
        r['support'] = sorted(m.support_.tolist())
    else:
        m = svm.OneClassSVM(kernel=p['kernel'], gamma=p['gamma'], nu=p['nu'], tol=1e-8).fit(X)
        r['dec'] = m.decision_function(X).tolist()
        r['support'] = sorted(m.support_.tolist())
    out[j['id']] = r
json.dump(out, open(os.path.join(HERE, 'sk.json'), 'w'), separators=(',', ':'))
