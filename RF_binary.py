from sklearn.ensemble import RandomForestClassifier
from math import floor
paramgrid = {"n_estimators": [100, 300, 600, 1000],
             "max_leaf_nodes": range(4, 33),
             "max_features": [floor((X_train.shape[1]*i)**0.5) for i in list(range(1, 4))]}


random_cv_rf = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=15), n_iter=10,
                                  verbose=100, param_distributions=paramgrid,
                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                                  scoring="roc_auc")

start = time()
random_cv_rf.fit(X_train, y_train_b.values.ravel())
time_spent = time() - start
time_spent


random_cv_rf.best_params_
random_cv_rf.best_score_

# find best threshold value
y_train_probas = random_cv_rf.predict_proba(X_train)[:,1]

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
plt.plot(thresholds, cv_results)
optimal = thresholds[cv_results.argmax()] # 0.5

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(random_cv_rf.predict_proba(X_val) > optimal, "normal", "attack")[:, 1]
y_val_probas = random_cv_rf.predict_proba(X_val)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), random_cv_rf.classes_).plot()
plot_roc_curve(random_cv_rf, X_val, y_val_b)

# save model
dump(random_cv_rf, 'rf_binary_none.joblib')


######### with pca ###########

from sklearn.decomposition import PCA

ncomps = [5, 10, 15, 20]
auc = []
params = []
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

    paramgrid = {"n_estimators": [100, 300, 600, 1000],
                 "max_leaf_nodes": range(4, 33),
                 "max_features": [floor((X_train_pca_.shape[1] * i) ** 0.5) for i in list(range(1, 4))]}

    random_cv_rf = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=15), n_iter=10,
                                  verbose=100, param_distributions=paramgrid,
                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                                  scoring="roc_auc")

    # insert classifier here
    random_cv_rf.fit(X_train_pca_, y_train_b.values.ravel())
    print("best params:", random_cv_rf.best_params_)
    print("best score:", random_cv_rf.best_score_)
    auc.append(random_cv_rf.best_score_)
    models.append(random_cv_rf.best_estimator_)
    params.append((random_cv_rf.best_params_))
time_spent = time() - start
time_spent


plt.plot(ncomps, auc)

ncomps[auc.index(max(auc))]


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

rf_b = models[1]
rf_b

# find best threshold value
rf_probabilities = rf_b.predict_proba(X_train_pca_)[:,1]

thresholds = np.linspace(0.44, 0.46, 11)

balanced_acc_cv = [None for _ in range(10)]

threshold_cv = check_cv(cv=10, y=y_train_b, classifier=True)

for i, (train_ids, test_ids)  in enumerate(threshold_cv.split(rf_probabilities, y_train_b)):
    balanced_acc = [None for _ in thresholds]
    for j, threshold in enumerate(thresholds):
        preds = np.where(rf_probabilities[test_ids] > threshold, "normal", "attack")
        balanced_acc[j] = balanced_accuracy_score(y_train_b.iloc[test_ids, 0], preds)
    balanced_acc_cv[i] = balanced_acc
    print("end iter", i + 1, ":", balanced_acc_cv[i])

cv_results = np.mean(balanced_acc_cv, axis=0)
plt.plot(thresholds, cv_results)
optimal = thresholds[cv_results.argmax()] # 0.456

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(rf_b.predict_proba(X_val_pca_) > optimal, "normal", "attack")[:, 1]
y_val_probas = rf_b.predict_proba(X_val_pca_)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), rf_b.classes_).plot()
plot_roc_curve(rf_b, X_val_pca_, y_val_b)

# save model
dump(rf_b, 'rf_binary_pca.joblib')


############ SVM-RFE #########
n_selected = []
auc = []
params = []
Cs = [0.001, 0.01, 0.1, 1]

start = time()
for C in Cs:
    print(f"Current C: {C}")
    svm = LinearSVC(dual=False, class_weight="balanced", # dual=False is preferred when n>p,
                    C=C)                                 # when dual=False, no random_state should be given

    svmrfe = RFECV(estimator=svm, cv=3, scoring="roc_auc", n_jobs=3, step=1, verbose=20)

    selector = svmrfe.fit(X_train, y_train_b.values.ravel())

    X_train_rfe = selector.transform(X_train)
    print(f"Variables selected: {selector.n_features_}")
    paramgrid = {"n_estimators": [100, 300, 600, 1000],
                 "max_leaf_nodes": range(4, 33),
                 "max_features": [floor((X_train_rfe.shape[1] * i) ** 0.5) for i in list(range(1, 4))]}

    random_cv_rf = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=15), n_iter=10,
                                  verbose=100, param_distributions=paramgrid,
                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                                  scoring="roc_auc")

    random_cv_rf.fit(X_train_rfe, y_train_b.values.ravel())

    params.append(random_cv_rf.best_params_)
    auc.append(random_cv_rf.best_score_)
    n_selected.append(selector.n_features_)
    print(f"Best params: {random_cv_rf.best_params_}")
    print(f"AUC: {random_cv_rf.best_score_}")

time_spent = time() - start

plt.plot([1,2,3,4], auc)
# we select C=0.001 as it is results in the highest AUC

# retrain with best values
svm = LinearSVC(dual=False, class_weight="balanced", # dual=False is preferred when n>p,
                    C=0.001)
svmrfe = RFECV(estimator=svm, cv=3, scoring="roc_auc", n_jobs=3, step=1, verbose=20)
selector = svmrfe.fit(X_train, y_train_b.values.ravel())

X_train_rfe = selector.transform(X_train)
X_val_rfe = selector.transform(X_val)

rf_b = RandomForestClassifier(n_jobs=15, n_estimators=600, max_features=18, max_leaf_nodes=31)

rf_b.fit(X_train_rfe, y_train_b.values.ravel())

# find best threshold value

y_train_probas = rf_b.predict_proba(X_train_rfe)[:, 1]

thresholds = np.linspace(0.54, 0.56, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.552

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(rf_b.predict_proba(X_val_rfe) > optimal, "normal", "attack")[:, 1]
y_val_probas = rf_b.predict_proba(X_val_rfe)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), elnet_b.classes_).plot()
plot_roc_curve(elnet_b, X_val_rfe, y_val_b)

# save model
dump(rf_b, 'rf_binary_svmrfe.joblib')
