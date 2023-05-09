import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
import numpy as np
from functools import partial
from scipy.optimize import fmin

class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0
    
    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)
        return -1.0 * auc_score
    
    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

X, y = make_classification(n_samples=10000, n_features=25)

xfold, xfold2, yfold, yfold2 = model_selection.train_test_split(
    X, y, test_size=0.5, stratify=y
)

logres = linear_model.LogisticRegression()
rf = ensemble.RandomForestClassifier()
xgbc = xgb.XGBClassifier()

logres.fit(xfold, yfold)
rf.fit(xfold, yfold)
xgbc.fit(xfold, yfold)

preds_logres = logres.predict_proba(xfold2)[:, 1]
preds_rf = rf.predict_proba(xfold2)[:, 1]
preds_xgbc = xgbc.predict_proba(xfold2)[:, 1]

avg_preds = (preds_logres + preds_rf + preds_xgbc) / 3

fold2_preds = np.column_stack((
    preds_logres, preds_rf, preds_xgbc, avg_preds
))

aucs_fold2 = []
for i in range(fold2_preds.shape[1]):
    auc = metrics.roc_auc_score(yfold2, fold2_preds[:, i])
    aucs_fold2.append(auc)

print(f"Fold-2: LR AUC: {aucs_fold2[0]}")
print(f"Fold-2: RF AUC: {aucs_fold2[1]}")
print(f"Fold-2: XGBC AUC: {aucs_fold2[2]}")
print(f"Fold-2: AVG AUC: {aucs_fold2[3]}")

logres = linear_model.LogisticRegression()
rf = ensemble.RandomForestClassifier()
xgbc = xgb.XGBClassifier()

logres.fit(xfold2, yfold2)
rf.fit(xfold2, yfold2)
xgbc.fit(xfold2, yfold2)

preds_logres = logres.predict_proba(xfold)[:, 1]
preds_rf = rf.predict_proba(xfold)[:, 1]
preds_xgbc = xgbc.predict_proba(xfold)[:, 1]

avg_preds = (preds_logres + preds_rf + preds_xgbc) / 3

fold1_preds = np.column_stack((
    preds_logres, preds_rf, preds_xgbc, avg_preds
))

aucs_fold1 = []
for i in range(fold1_preds.shape[1]):
    auc = metrics.roc_auc_score(yfold, fold1_preds[:, i])
    aucs_fold1.append(auc)

print(f"Fold-1: LR AUC: {aucs_fold1[0]}")
print(f"Fold-1: RF AUC: {aucs_fold1[1]}")
print(f"Fold-1: XGBC AUC: {aucs_fold1[2]}")
print(f"Fold-1: AVG AUC: {aucs_fold1[3]}")

opt = OptimizeAUC()
opt.fit(fold1_preds[:, :-1], yfold)
opt_preds_fold2 = opt.predict(fold2_preds[:, :-1])
auc = metrics.roc_auc_score(yfold2, opt_preds_fold2)
print(f"Optimized AUC: {auc}")
print(f"Optimized Coefficients: {opt.coef_}")

opt = OptimizeAUC()
opt.fit(fold2_preds[:, :-1], yfold2)
opt_preds_fold1 = opt.predict(fold1_preds[:, :-1])
auc = metrics.roc_auc_score(yfold, opt_preds_fold1)
print(f"Optimized AUC: {auc}")
print(f"Optimized Coefficients: {opt.coef_}")