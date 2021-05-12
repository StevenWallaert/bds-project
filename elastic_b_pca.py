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


######### with pca ###########

from sklearn.decomposition import PCA

ncomps = [5, 10, 15, 20]
auc = []
models = []

### num en cat apart
start = time()
for n in ncomps:
    print("n_components:", n)

    # data preprocessing
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("robust_scaler", RobustScaler()),
        ("pca", PCA(n_components=n))
    ])

    full_pipeline_pca = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    full_pipeline_pca.fit(X_train_pca)

    X_train_pca_ = full_pipeline_pca.transform(X_train_pca)

    X_val_pca_ = full_pipeline_pca.transform(X_val_pca)

    # insert classifier here
    cv_elnet_b.fit(X_train_pca_, y_train_b.values.ravel())
    print("best params:", cv_elnet_b.best_params_)
    print("best score:", cv_elnet_b.best_score_)
    auc.append(cv_elnet_b.best_score_)
    models.append(cv_elnet_b.best_estimator_)
time_spent = time() - start


plt.plot(ncomps, auc)

ncomps[auc.index(max(auc))]

# we choose 10 components: diminishing returns
# data preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("robust_scaler", RobustScaler()),
    ("pca", PCA(n_components=10))
])

full_pipeline_pca = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

full_pipeline_pca.fit(X_train_pca)

X_train_pca_ = full_pipeline_pca.transform(X_train_pca)

X_val_pca_ = full_pipeline_pca.transform(X_val_pca)

elnet_b = LogisticRegression(penalty="elasticnet", n_jobs=1, fit_intercept=True,
                            max_iter=3000, solver="saga", verbose=0, tol=0.001,
                             C=0.0005623413251903491, class_weight="balanced", l1_ratio=0.7)
elnet_b.fit(X_train_pca_, y_train_b.values.ravel())

# find best threshold value
cv_elnet_b = elnet_b

y_train_probas = cv_elnet_b.predict_proba(X_train_pca_)[:, 1]

thresholds = np.linspace(.32, .34, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.332

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(elnet_b.predict_proba(X_val_pca_) > optimal, "normal", "attack")[:, 1]
y_val_probas = elnet_b.predict_proba(X_val_pca_)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), elnet_b.classes_).plot()
plot_roc_curve(elnet_b, X_val_pca_, y_val_b)

# save model
dump(elnet_b, 'elnet_binary_pca.joblib')