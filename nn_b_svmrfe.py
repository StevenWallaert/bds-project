# multi layre perceptron
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time

from sklearn.model_selection import check_cv
from joblib import dump, load
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
#



auc = []
params = []
Cs = [0.001, 0.01, 0.1, 1]

svm = LinearSVC(dual=False, class_weight="balanced",  # dual=False is preferred when n>p,
                C=0.001)  # when dual=False, no random_state should be given

svmrfe = RFECV(estimator=svm, cv=3, scoring="roc_auc", n_jobs=3, step=1)

start = time()
selector = svmrfe.fit(X_train_std, y_train_b.values.ravel())

X_train_rfe = selector.transform(X_train_std)
X_val_rfe = selector.transform(X_val_std)

# first use label encoder to convert from ["normal", "attack"] to [0, 1]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train_le = encoder.fit_transform(y_train_b.values.reshape(-1,))
y_val_le = encoder.transform(y_val_b.values.reshape(-1,))

# then convert with to_categorical function from keras
y_train_le = keras.utils.to_categorical(y_train_le)
y_val_le = keras.utils.to_categorical(y_val_le)



def build_model(n_hidden=1, n_neurons=30, input_shape=[X_train_rfe.shape[1]], rate=0.1):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu", kernel_initializer="lecun_normal"))
        model.add(keras.layers.AlphaDropout(rate))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
    optimizer = keras.optimizers.Nadam()
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", "AUC"])
    return model

keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [1, 2, 3],
    "n_neurons": np.arange(30, 301),
    "rate": [0.1, 0.2, .3]
}

nn = RandomizedSearchCV(keras_clf, param_distributions=param_distribs, n_iter=10, cv=3, n_jobs=1, verbose=10,
                        scoring="roc_auc")

tf.random.set_seed(0)

nn.fit(X_train_rfe, y_train_le, epochs=50,
       callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
       validation_split=0.1, batch_size=500)
time_spent = time() - start


nn.best_params_
nn.best_score_

# fit model again with best params

# specify model architecture
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[X_train_rfe_1.shape[1]]),
    keras.layers.Dense(258, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.1),
    keras.layers.Dense(258, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.1),
    keras.layers.Dense(2, activation="sigmoid")
])

# overview
model.summary()

# specify loss, optimizer, metric etc
model.compile(loss="binary_crossentropy",
              optimizer="nadam",
              metrics=["accuracy", "AUC"])

# fit model
history = model.fit(X_train_rfe_1, y_train_le, epochs=100, validation_split=0.1, batch_size=500,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# plot learning curve
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 0.1)
plt.show()


# find best threshold value

y_train_probas = model.predict(X_train_rfe)[:, 1]

thresholds = np.linspace(0.32, .36, 11)

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

# evaluate model on val set
y_val_probas = model.predict(X_val_rfe)
y_val_preds = np.where(y_val_probas[:, 1] > optimal, "normal", "attack") # get back into original form

accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))




# monte carlo dropout
y_train_probas = np.stack([model(X_train_rfe, training=True) for sample in range(100)])
y_train_probas = y_train_probas.mean(axis=0)
y_train_probas = y_train_probas[:,1]
# find best threshold value
thresholds = np.linspace(0.1, .9, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.234


y_val_probas = np.stack([model(X_val_rfe, training=True) for sample in range(100)])
y_val_probas = y_val_probas.mean(axis=0)
y_val_preds = np.where(y_val_probas[:, 1] > optimal, "normal", "attack") # get back into original form

accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))

# save model
model.save("nn_b_rfe_model.h5")
model = keras.models.load_model("nn_binary_model.h5")