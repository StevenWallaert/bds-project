class FindLogisticThreshold:
    def __init__(self, estimator):
        self.estimator = estimator
        self.predictions = None

    def fit(self, X, y, threshold):
        


        split = splits.send(i)
        preds = np.where(cv_elnet_b.predict_proba(X_train[split[1],:])[:,1] > threshold, 1, 0)
        acc.append(accuracy_score(y_train_b[split[1]], y_pred=preds))
        balanced_acc.append(balanced_accuracy_score(y_train_b[split[1]], y_pred=preds))