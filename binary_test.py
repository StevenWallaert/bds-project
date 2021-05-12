# multi layre perceptron
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time

from sklearn.model_selection import check_cv
from joblib import dump, load
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
# import full dataset

kdd = pd.read_csv("kddcup.data.corrected", sep=',', names=colnames)

# create coarse classes and binary classes
kdd["ctarget"] = [attacks_dict[attack] for attack in kdd.target] # coarse 5 classification
kdd["btarget"] = ["normal" if attack == "normal." else "attack" for attack in kdd.target] # binary classification

kdd = kdd.drop("is_host_login", axis=1)

X = kdd.drop(["target", "btarget", "ctarget"], axis=1)

X_train = full_pipeline_std.transform(X)
y_train = kdd[["btarget"]]


# SVMRFE
svm = LinearSVC(dual=False, class_weight="balanced",  # dual=False is preferred when n>p,
                C=0.001)  # when dual=False, no random_state should be given

svmrfe = RFECV(estimator=svm, cv=3, scoring="roc_auc", n_jobs=3, step=1, verbose=20)

# fit selector on limited training set: computational reasons
selector = svmrfe.fit(X_train_std, y_train_b.values.ravel())

X_train = selector.transform(X_train)

# first use label encoder to convert from ["normal", "attack"] to [0, 1]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train_le = encoder.fit_transform(y_train.values.reshape(-1,))


# then convert with to_categorical function from keras
y_train_le = keras.utils.to_categorical(y_train_le)


# Test set

# load in test data
test = pd.read_csv("corrected", sep=',', names=colnames)
# get yb
yb_test = ["normal" if attack == "normal." else "attack" for attack in test.target] # binary classification
X_test = test.drop("target", axis=1)

y_test_le = encoder.transform(yb_test)
# preprocess X
X_test = X_test.drop("is_host_login", axis=1)
X_test = full_pipeline_std.transform(X_test)

# rfe
X_test = selector.transform(X_test)

# ready to go
X_test.shape
X_train.shape

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[X_train.shape[1]]),
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
history = model.fit(X_train, y_train_le, epochs=100, validation_split=0.1, batch_size=500,
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# find best threshold value


y_train_probas = model.predict(X_train)[:, 1]

thresholds = np.linspace(0.25, .27, 11)

balanced_acc_cv = [None for _ in range(10)]

threshold_cv = check_cv(cv=10, y=y_train, classifier=True)

for i, (train_ids, test_ids)  in enumerate(threshold_cv.split(y_train_probas, y_train)):
    balanced_acc = [None for _ in thresholds]
    for j, threshold in enumerate(thresholds):
        preds = np.where(y_train_probas[test_ids] > threshold, "normal", "attack")
        balanced_acc[j] = balanced_accuracy_score(y_train.iloc[test_ids, 0], preds)
    balanced_acc_cv[i] = balanced_acc
    print("end iter", i + 1, ":", balanced_acc_cv[i])

cv_results = np.mean(balanced_acc_cv, axis=0)
plt.plot(thresholds, cv_results)
optimal = thresholds[cv_results.argmax()] # 0.26

# evaluate model on val set
y_test_probas = model.predict(X_test)
y_test_preds = np.where(y_test_probas[:, 1] > optimal, "normal", "attack") # get back into original form

accuracy_score(yb_test, y_test_preds)
roc_auc_score(yb_test, y_test_probas[:, 1])
balanced_accuracy_score(yb_test, y_test_preds)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(yb_test, y_test_preds))
print(classification_report(yb_test, y_test_preds))
ConfusionMatrixDisplay(confusion_matrix(yb_test, y_test_preds), svmrfe.classes_).plot()

model.save("final_binary.h5")
dump(selector, "svmrfe-final.joblib")