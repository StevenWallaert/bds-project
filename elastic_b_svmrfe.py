from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, plot_roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import check_cv
from joblib import dump, load
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV

# njobs = 1 because only 1 classifier, when doing multilabel then it is useful
# intercept = True because of robustscaler, not standardscaler
# elastic net only works with saga solver
elnet_b = LogisticRegression(penalty="elasticnet", n_jobs=1, fit_intercept=True,
                            max_iter=100, solver="saga", verbose=0)

# hyperparameters
paramgrid = {"C": np.logspace(-7, 3, num=25, base=10),
             "l1_ratio": [0, .1, .5, .7, .9, .95, .99, 1],
             "class_weight": [None, "balanced"]}

# randomized search CV
cv_elnet_b = RandomizedSearchCV(estimator=elnet_b, param_distributions=paramgrid,
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                               n_iter=10, scoring="roc_auc", n_jobs=15,
                                random_state=0, verbose=100)

############ SVM-RFE #########

auc = []
params = []
Cs = [0.001, 0.01, 0.1, 1]

start = time()
for C in Cs:
    print(f"Current C: {C}")
    svm = LinearSVC(dual=False, class_weight="balanced", # dual=False is preferred when n>p,
                    C=C)                                 # when dual=False, no random_state should be given

    svmrfe = RFECV(estimator=svm, cv=3, scoring="roc_auc", n_jobs=3, step=1)

    selector = svmrfe.fit(X_train, y_train_b.values.ravel())

    X_train_rfe = selector.transform(X_train)

    cv_elnet_b.fit(X_train_rfe, y_train_b.values.ravel())

    params.append(cv_elnet_b.best_params_)
    auc.append(cv_elnet_b.best_score_)
    print(f"Best params: {cv_elnet_b.best_params_}")
    print(f"AUC: {cv_elnet_b.best_score_}")
    print(f"Variables selected: {selector.n_features_}")
time_spent = time() - start

plt.plot([110, 96, 104, 86], auc)
# we select C=1 as it is results in the most parsimonious model and highest AUC

# retrain with best values
svm = LinearSVC(dual=False, class_weight="balanced", # dual=False is preferred when n>p,
                    C=1)
svmrfe = RFECV(estimator=svm, cv=3, scoring="roc_auc", n_jobs=3, step=1)
selector = svmrfe.fit(X_train, y_train_b.values.ravel())

X_train_rfe = selector.transform(X_train)
X_val_rfe = selector.transform(X_val)

elnet_b = LogisticRegression(penalty="elasticnet", n_jobs=1, fit_intercept=True,
                            max_iter=3000, solver="saga", verbose=0, tol=0.001,
                             C= 8.25404185268019, class_weight=None, l1_ratio=0.95)
elnet_b.fit(X_train_rfe, y_train_b.values.ravel())

# find best threshold value
cv_elnet_b = elnet_b

y_train_probas = cv_elnet_b.predict_proba(X_train_rfe)[:, 1]

thresholds = np.linspace(0.18, 0.20, 11)

balanced_acc_cv = [None for _ in range(10)]

threshold_cv = check_cv(cv=10, y=y_train_b, classifier=True)

for i, (train_ids, test_ids)  in enumerate(threshold_cv.split(y_train_probas, y_train_b)):
    balanced_acc = [None for _ in thresholds]
    for j, threshold in enumerate(thresholds):
        preds = np.where(y_train_probas[test_ids] > threshold, "normal", "attack")
        balanced_acc[j] = balanced_accuracy_score(y_train_b.iloc[test_ids, 0], preds)
    balanced_acc_cv[i] = balanced_acc
    print("end iter", i + 1, ":", balanced_acc_cv[i])

cv_results = np.mean(balanced_acc_cv, axis=0)
plt.plot(thresholds, cv_results)
optimal = thresholds[cv_results.argmax()] # 0.19

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(elnet_b.predict_proba(X_val_rfe) > optimal, "normal", "attack")[:, 1]
y_val_probas = elnet_b.predict_proba(X_val_rfe)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), elnet_b.classes_).plot()
plot_roc_curve(elnet_b, X_val_rfe, y_val_b)

# save model
dump(elnet_b, 'elnet_binary_svmrfe.joblib')
