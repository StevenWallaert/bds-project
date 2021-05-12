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
import tensorflow as tf
from tensorflow import keras
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

cv_elnet_b.fit(codings, y_train_b.values.ravel())

cv_elnet_b.best_score_

n_codings = [5, 10, 15, 20]

auc = []
params = []

start = time()
for n in n_codings:
    print(f"Number of codings: {n}")

    stacked_encoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[X_train.shape[1]]),
        keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(n, activation="selu", kernel_initializer="lecun_normal")
    ])

    stacked_decoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[n]),
        keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(X_train.shape[1], activation=None)
    ])

    ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

    ae.compile(loss="mse", optimizer=keras.optimizers.Nadam())

    history = ae.fit(X_train, X_train, epochs=20, validation_split=0.1)
    pd.DataFrame(history.history).plot()

    codings = stacked_encoder.predict(X_train)

    cv_elnet_b.fit(codings, y_train_b.values.ravel())

    auc.append(cv_elnet_b.best_score_)
    print(f"AUC: {cv_elnet_b.best_score_}")
    params.append(cv_elnet_b.best_params_)
    print(f"Params: {cv_elnet_b.best_params_}")
time_spent = time() - start
time_spent

plt.plot(n_codings, auc)

# fit again with best values

stacked_encoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[117]),
    keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(20, activation="selu", kernel_initializer="lecun_normal")
])

stacked_decoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[20]),
    keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(117, activation=None)
])

ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

ae.compile(loss="mse", optimizer=keras.optimizers.Nadam())

history = ae.fit(X_train, X_train, epochs=20, validation_split=0.1)


codings = stacked_encoder.predict(X_train)
codings_val = stacked_encoder.predict(X_val)

elnet_b = LogisticRegression(penalty="elasticnet", n_jobs=1, fit_intercept=True,
                            max_iter=3000, solver="saga", verbose=10, tol=0.001,
                             l1_ratio=0.7, class_weight="balanced", C=0.0005623413251903491)

elnet_b.fit(codings, y_train_b)

# find best threshold value
cv_elnet_b = elnet_b

y_train_probas = cv_elnet_b.predict_proba(codings)[:, 1]

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
optimal = thresholds[cv_results.argmax()] # 0.496

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(cv_elnet_b.predict_proba(codings_val) > optimal, "normal", "attack")[:, 1]
y_val_probas = cv_elnet_b.predict_proba(codings_val)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), cv_elnet_b.classes_).plot()
plot_roc_curve(cv_elnet_b, codings_val, y_val_b)

# save model
dump(codings, "elnet_binary_ae_codings.joblib")
dump(codings_val, "elnet_binary_ae_codings_val.joblib")
dump(cv_elnet_b, "elnet_binary_ae.joblib")

