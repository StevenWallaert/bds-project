import pandas as pd
import numpy as np

# read in column names
from tensorflow.python.ops.nn_ops import fractional_avg_pool_v2

with open("colnames.txt") as columns:
    colnames = [line[:line.find(":")] for line in columns] # discard everything from ':' on per line

# add class column name
colnames.append("target")

# read in coarse class names
with open("training_attack_types") as attack_types:
    key_values = [line.strip() for line in attack_types][:-1]  # last line is empty

attacks_dict = {key+".": value for (key, value) in [pair.split(" ") for pair in key_values]} # the +"." accounts for an extra dot that seems to be present in the data
attacks_dict["normal."] = "normal"
# read in training data
kdd = pd.read_csv("kddcup.data.corrected",
                  sep=',',
                  names=colnames)

# create coarse classes and binary classes
kdd["ctarget"] = [attacks_dict[attack] for attack in kdd.target] # coarse 5 classification
kdd["btarget"] = ["normal" if attack == "normal." else "attack" for attack in kdd.target] # binary classification

# sample a fraction to speed up work
kdd =kdd.sample(frac=0.1)


# divide features in y and X (and distinguish between num and cat

cat_features = ["protocol_type", "service", "flag", "is_host_login", "is_guest_login", "root_shell", "logged_in"]
yb = ["btarget"]
yc = ["ctarget"]
ym = ["target"]
num_features = list(kdd.drop(cat_features + yb + yc + ym, axis=1))

for catfeat in cat_features:
    print(kdd[catfeat].value_counts())
    #input("Press Enter to continue...")

print(kdd["target"].value_counts())

#is_hostlogin has only 2 instances in positive class

X = kdd.drop(["target", "btarget", "ctarget"], axis=1)
yb = kdd[["btarget"]]
from sklearn.preprocessing import OneHotEncoder





# make validation set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, yb, test_size=0.15, random_state=0, stratify=yb)

# check ok
y_train.value_counts()
y_val.value_counts()

# make preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("robust_scaler", RobustScaler())
])
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features)
])


##### until here preprocessing

from sklearn.model_selection import StratifiedKFold
import random

k = 3
skf = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)
random.seed(1)

from sklearn.linear_model import SGDClassifier
full_pipeline.fit(X_train)
X_train = full_pipeline.transform(X_train)
X_val = full_pipeline.transform(X_val)

clf = SGDClassifier(loss="log", penalty="elasticnet", early_stopping=True, n_jobs=15)

elastic_net = clf.fit(X_train, y_train.values.ravel())
from sklearn.model_selection import RandomizedSearchCV
paramgrid = {"alpha": np.logspace(-3, 3, num=25, base=10),
             "l1_ratio": [.1, .5, .7, .9, .95, .99, 1]}
random_cv = RandomizedSearchCV(estimator=clf, param_distributions=paramgrid, cv=StratifiedKFold(n_splits=3,
                                                                                                  shuffle=True,
                                                                                                  random_state=0))
random_cv.fit(X_train, y_train)

random_cv.best_params_
random_cv.best_score_


from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

y_val_proba = random_cv.predict_proba(X_val)
y_val_pred = random_cv.predict(X_val)

accuracy_score(y_val, y_val_pred)
roc_auc_score(y_val, y_val_proba[:,1])
balanced_accuracy_score(y_val, y_val_pred)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_val, y_val_pred)
print(classification_report(y_val, y_val_pred))
from sklearn.metrics import roc_curve

roc_curve([i=="attack" for i in y_val.values], y_val_proba[:,1])
import scikitplot as skplt
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier

paramgrid = {"n_estimators": [10, 30, 100, 300, 1000],
             "max_leaf_nodes": [4, 8, 16, 32]}


random_cv_rf = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=3), param_distributions=paramgrid, cv=StratifiedKFold(n_splits=3,
                                                                                                  shuffle=True,
                                                                                                  random_state=0))

random_cv_rf.fit(X_train, y_train)

random_cv_rf.best_params_
random_cv_rf.best_score_

y_val_pred = random_cv_rf.predict(X_val)
y_val_proba = random_cv_rf.predict_proba(X_val)

accuracy_score(y_val, y_val_pred)
roc_auc_score(y_val, y_val_proba[:,1])

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = grid_cv.predict_proba(X_val)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve([y == "normal" for y in y_val.values], preds)
roc_auc = metrics.auc(fpr, tpr)

roc_auc
from joblib import dump, load

dump(random_cv_rf, 'random_cv_rf_limited.joblib')
dump(random_cv, 'random_cv_elnet_limited.joblib')

df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))