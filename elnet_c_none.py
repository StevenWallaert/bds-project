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

# njobs = 1 because only 1 classifier, when doing multilabel then it is useful
# intercept = True because of robustscaler, not standardscaler
# elastic net only works with saga solver
elnet_c = LogisticRegression(penalty="elasticnet", n_jobs=5, fit_intercept=True, multi_class="ovr",
                            max_iter=100, tol=0.02, solver="saga", verbose=20,
                             C=0.001, l1_ratio=0.9, class_weight="balanced")

# hyperparameters
paramgrid = {"C": np.logspace(-7, 3, num=25, base=10),
             "l1_ratio": [0, .1, .5, .7, .9, .95, .99, 1],
             "class_weight": [None, "balanced"]}

# randomized search CV
cv_elnet_c = RandomizedSearchCV(estimator=elnet_c, param_distributions=paramgrid,
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                               n_iter=10, scoring="roc_auc", n_jobs=3,
                                random_state=0, verbose=100)

# train
start = time()
cv_elnet_c = elnet_c.fit(X_train, y_train_c.values.ravel())
time_spent = time() - start

# inspect best params and scores
cv_elnet_b.best_params_
cv_elnet_b.best_score_

# retrain with max_iter much higher so it can converge
elnet = LogisticRegression(penalty="elasticnet", n_jobs=1, fit_intercept=True,
                            max_iter=3000, solver="saga", verbose=10, C=0.0005623413251903491,
                                     l1_ratio=0.7, class_weight='balanced', tol=0.001)
elnet.fit(X_train, y_train_b.values.ravel())

# find best threshold value
cv_elnet_c = fit

y_train_probas = cv_elnet_b.predict_proba(X_train)[:, 1]

thresholds = np.linspace(0.49, .51, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.5

# predict probabilities and predictions on validation set using optimal threshold
#y_val_preds = np.where(cv_elnet_b.predict_proba(X_val) > optimal, "normal", "attack")[:, 1]
y_val_preds = cv_elnet_c.predict(X_val)
y_val_probas = cv_elnet_c.predict_proba(X_val)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), cv_elnet_b.classes_).plot()
plot_roc_curve(cv_elnet_b, X_val, y_val_b)

# save model
dump(cv_elnet_b, 'elnet_binary_none.joblib')
cv_elnet_b = load("elnet_binary_none.joblib")




