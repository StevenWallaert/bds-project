
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV

svm = LinearSVC(dual=False)

svmrfe = RFECV(estimator=svm, cv=3, scoring="accuracy", n_jobs=3, step=1)

selector = svmrfe.fit(X_train, y_train.values.ravel())

X_train_rfe = selector.transform(X_train)
X_val_rfe = selector.transform(X_val)