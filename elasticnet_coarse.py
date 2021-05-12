import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report
from joblib import dump, load

# njobs = 1 because only 1 classifier, when doing multilabel then it is useful
# intercept = True because robustscaler, not standardscaler
# elastic net only works with saga solver
elnet_c = LogisticRegression(penalty="elasticnet", n_jobs=5, fit_intercept=True,
                                  max_iter=1000, solver="saga", random_state=0, multi_class="ovr")

# hyperparameters
paramgrid = {"C": np.logspace(-3, 3, num=25, base=10),
             "l1_ratio": [0, .1, .5, .7, .9, .95, .975, .99, 1],
             "class_weight": [None, "balanced"]}

# randomized search CV
cv_elnet_c = RandomizedSearchCV(estimator=elnet_c, param_distributions=paramgrid,
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                               n_iter=10, scoring="accuracy", n_jobs=-1, random_state=0, verbose=4)

# train
cv_elnet_c.fit(X_train, y_train_c.values.ravel())

# inspect best params and scores
random_cv.best_params_
random_cv.best_score_

# predict probabilities and predictions on validation set
y_val_probas = random_cv.predict_proba(X_val)
y_val_preds = random_cv.predict(X_val)

# validaton scores
accuracy_score(y_val, y_val_preds)
roc_auc_score(y_val, y_val_probas[:, 1])
balanced_accuracy_score(y_val, y_val_preds)

# extra
print(confusion_matrix(y_val, y_val_preds))
print(classification_report(y_val, y_val_preds))

# save model

dump(random_cv, 'random_cv_elnet_full.joblib')