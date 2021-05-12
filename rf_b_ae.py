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

# hyperparameters
paramgrid = {"n_estimators": [100, 300, 600, 1000],
             "max_leaf_nodes": range(4, 33),
             "max_features": [floor((codings.shape[1] * i) ** 0.5) for i in list(range(1, 4))]}

# randomized search CV
rf_b_ae = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=15), param_distributions=paramgrid,
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                               n_iter=10, scoring="roc_auc",
                               random_state=0, verbose=100)

rf_b_ae.fit(codings, y_train_b.values.ravel())

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

    # hyperparameters
    paramgrid = {"n_estimators": [100, 300, 600, 1000],
                 "max_leaf_nodes": range(4, 33),
                 "max_features": [floor((n * i) ** 0.5) for i in list(range(1, 4))]}

    # randomized search CV
    rf_b_ae = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=15), param_distributions=paramgrid,
                                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                                    n_iter=10, scoring="roc_auc",
                                    random_state=0, verbose=100)

    rf_b_ae.fit(codings, y_train_b.values.ravel())

    auc.append(rf_b_ae.best_score_)
    print(f"AUC: {rf_b_ae.best_score_}")
    params.append(rf_b_ae.best_params_)
    print(f"Params: {rf_b_ae.best_params_}")
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

# find best threshold value
rf_b = rf_b_ae.best_estimator_

y_train_probas = rf_b.predict_proba(codings)[:, 1]

thresholds = np.linspace(0.52, .54, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.53

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(rf_b.predict_proba(codings_val) > optimal, "normal", "attack")[:, 1]
y_val_probas = rf_b.predict_proba(codings_val)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

ConfusionMatrixDisplay(confusion_matrix(y_val_b, y_val_preds), rf_b.classes_).plot()
plot_roc_curve(rf_b, codings_val, y_val_b)

# save model
dump(codings, "rf_binary_ae_codings.joblib")
dump(codings_val, "rf_binary_ae_codings_val.joblib")
dump(rf_b, "rf_binary_ae.joblib")

