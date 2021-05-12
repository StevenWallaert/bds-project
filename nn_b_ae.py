# multi layre perceptron
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time

from sklearn.model_selection import check_cv
from joblib import dump, load
codings_5 = codings

n_codings = [15, 20]
best_params = []
best_score = []
times = []

for n in n_codings:
    print(f"Number of codings: {n}")
    start = time()
    tf.random.set_seed(0)
    stacked_encoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[X_train_std.shape[1]]),
        keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(n, activation="selu", kernel_initializer="lecun_normal")
    ])

    stacked_decoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[n]),
        keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(X_train_std.shape[1], activation=None)
    ])

    ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

    ae.compile(loss="mse", optimizer=keras.optimizers.Nadam())

    history = ae.fit(X_train_std, X_train_std, epochs=20, validation_split=0.1,
                     callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    # plot learning curve
    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca().set_ylim(0.1, 1)
    plt.show()



    codings = stacked_encoder.predict(X_train_std)



    # first use label encoder to convert from ["normal", "attack"] to [0, 1]
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_train_le = encoder.fit_transform(y_train_b.values.reshape(-1,))
    y_val_le = encoder.transform(y_val_b.values.reshape(-1,))

    # then convert with to_categorical function from keras
    y_train_le = keras.utils.to_categorical(y_train_le)
    y_val_le = keras.utils.to_categorical(y_val_le)



    def build_model(n_hidden=1, n_neurons=30, input_shape=[codings.shape[1]], rate=0.1):
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
    try:
        nn.fit(codings, y_train_le, epochs=50,
               callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
               validation_split=0.1, batch_size=500)
    except RuntimeError:
        pass
    time_spent = time() - start

    best_params.append(nn.best_params_)
    best_score.append(nn.best_score_)
    times.append(time_spent)

# fit model again with best params

#AE
tf.random.set_seed(0)
stacked_encoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[X_train_std.shape[1]]),
    keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")
])

stacked_decoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[10]),
    keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(X_train_std.shape[1], activation=None)
])

ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

ae.compile(loss="mse", optimizer=keras.optimizers.Nadam())

history = ae.fit(X_train_std, X_train_std, epochs=50, validation_split=0.1,
                 callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

codings = stacked_encoder.predict(X_train_std)
# specify model architecture
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[codings.shape[1]]),
    keras.layers.Dense(172, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.1),
    keras.layers.Dense(172, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.1),
    keras.layers.Dense(172, activation="selu", kernel_initializer="lecun_normal"),
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
history = model.fit(codings, y_train_le, epochs=100, validation_split=0.1, batch_size=500,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# plot learning curve
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 0.1)
plt.show()


# find best threshold value

y_train_probas = model.predict(codings)[:, 1]

thresholds = np.linspace(0.05, .075, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.0625

# evaluate model on val set

codings_val = stacked_encoder.predict(X_val_std)

y_val_probas = model.predict(codings_val)
y_val_preds = np.where(y_val_probas[:, 1] > optimal, "normal", "attack") # get back into original for

accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))




# monte carlo dropout
y_train_probas = np.stack([model(codings, training=True) for sample in range(100)])
y_train_probas = y_train_probas.mean(axis=0)
y_train_probas = y_train_probas[:,1]
# find best threshold value
thresholds = np.linspace(0.1, .3, 11)

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
optimal = thresholds[cv_results.argmax()] # 0.22


y_val_probas = np.stack([model(codings_val, training=True) for sample in range(100)])
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