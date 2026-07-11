# Reference outputs for the engine edge cases in tests/classic.mjs: class weights (logistic regression, a tree, an
# SVC) and SVC probabilities (the Platt sigmoid on fixed decision values, and probability=True with two and three
# classes). Run with the test environment: ~/.nooral-venv/bin/python tests/ref/gen_edge.py
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import _sigmoid_calibration
from sklearn.datasets import make_moons, make_blobs

out = os.path.join(os.path.dirname(__file__), 'classic')

rng = np.random.RandomState(3)
X = np.vstack([rng.randn(40, 2), rng.randn(8, 2) + [1.2, 1.0]]).round(3)
y = [0] * 40 + [1] * 8
lr = LogisticRegression(C=1.0, class_weight='balanced', tol=1e-12, max_iter=1000).fit(X, y)
lr2 = LogisticRegression(C=1.0, class_weight={0: 1, 1: 3}, tol=1e-12, max_iter=1000).fit(X, y)
dt = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=0).fit(X, y)
sv = SVC(C=1.0, kernel='linear', class_weight='balanced').fit(X, y)
with open(os.path.join(out, 'classweight.json'), 'w') as f:
    json.dump({'X': X.tolist(), 'y': y, 'lr': [lr.coef_[0].tolist(), float(lr.intercept_[0])], 'lr2': [lr2.coef_[0].tolist(), float(lr2.intercept_[0])],
               'dt': dt.predict_proba(X).tolist(), 'dtl': int(dt.get_n_leaves()), 'sv': sv.decision_function(X).tolist()}, f)

rng = np.random.RandomState(1)
fv = np.round(rng.randn(60) * 1.5, 3)
fy = (fv + rng.randn(60) > 0.3).astype(int)
a, b = _sigmoid_calibration(fv, fy)
Xm, Ym = make_moons(120, noise=0.3, random_state=2)
Xm = Xm.round(3)
m = SVC(C=1.0, kernel='rbf', gamma=0.5, probability=True, random_state=0).fit(Xm, Ym)
X3, Y3 = make_blobs(90, centers=3, cluster_std=1.6, random_state=4)
X3 = X3.round(3)
m3 = SVC(C=1.0, kernel='rbf', gamma=0.3, probability=True, random_state=0).fit(X3, Y3)
with open(os.path.join(out, 'platt.json'), 'w') as f:
    json.dump({'f': fv.tolist(), 'y': fy.tolist(), 'a': float(a), 'b': float(b), 'X': Xm.tolist(), 'Y': Ym.tolist(), 'p': m.predict_proba(Xm).tolist(),
               'X3': X3.tolist(), 'Y3': Y3.tolist(), 'p3': m3.predict_proba(X3).tolist()}, f)
print('wrote classweight.json and platt.json')
