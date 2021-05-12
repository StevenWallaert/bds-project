import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer

# read in column names
with open("colnames.txt") as columns:
    colnames = [line[:line.find(":")] for line in columns] # discard everything from ':' on per line

# add class column name
colnames.append("target")

# read in coarse class names
with open("training_attack_types") as attack_types:
    key_values = [line.strip() for line in attack_types][:-1]  # last line is empty

# make dict of form {attack_name: "coarse_class"}
attacks_dict = {key+".": value for (key, value) in [pair.split(" ") for pair in key_values]} # the +"." accounts for an extra dot that seems to be present in the data
attacks_dict["normal."] = "normal"

# read in training data
kdd = pd.read_csv("kddcup.data.corrected", sep=',', names=colnames)

# create coarse classes and binary classes
kdd["ctarget"] = [attacks_dict[attack] for attack in kdd.target] # coarse 5 classification
kdd["btarget"] = ["normal" if attack == "normal." else "attack" for attack in kdd.target] # binary classification

# take stratified sample: 10%
kdd_strat_sample = train_test_split(kdd, test_size=0.01, random_state=0, stratify=kdd["target"])[1]
kdd = kdd_strat_sample


# take low prevalence class data
#kdd_low_prevalence = kdd[[target not in ["normal.", "smurf.", "neptune."] for target in kdd.target]]
#kdd_low_prevalence["target"].value_counts()

# and add to sample
# this way low prevalence classes are better represented
#kdd_high_prevalence = kdd_strat_sample[[target in ["normal.", "smurf.", "neptune."] for target in kdd_strat_sample.target]]

#kdd = kdd_high_prevalence.append(kdd_low_prevalence)

# divide features in y and X (and distinguish between num and cat

cat_features = ["protocol_type", "service", "flag", "is_host_login", "is_guest_login", "root_shell", "logged_in"]
yb = ["btarget"]
yc = ["ctarget"]
ym = ["target"]
num_features = list(kdd.drop(cat_features + yb + yc + ym, axis=1))

#for catfeat in cat_features:
#    print(kdd[catfeat].value_counts())
#    input("Press Enter to continue...")

# is_host_login should be deleted, has only 0 values in training dataset
kdd = kdd.drop("is_host_login", axis=1)
cat_features = ["protocol_type", "service", "flag", "is_guest_login", "root_shell", "logged_in"]

#
print(kdd["target"].value_counts())
print(kdd["ctarget"].value_counts())
print(kdd["btarget"].value_counts())


X = kdd.drop(["target", "btarget", "ctarget"], axis=1)
yb = kdd[["btarget"]]
yc = kdd[["ctarget"]]
ym = kdd[["target"]]

# make validation set
# choose one of following lines per corresponding task
# binary task
X_train, X_val, y_train_b, y_val_b = train_test_split(X, yb, test_size=0.15, random_state=0, stratify=yb)

# Coarse task
#X_train, X_val, y_train_c, y_val_c = train_test_split(X, yc, test_size=0.15, random_state=0, stratify=yc)

# Multi class task
#X_train, X_val, y_train_m, y_val_m = train_test_split(X, ym, test_size=0.15, random_state=0, stratify=ym)

# and make a copy that will be useful when doing pca
X_train_pca = X_train[:]
X_val_pca = X_val[:]


# make preprocessing pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
# numerical
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("robust_scaler", RobustScaler())
])

# numerical + categorical
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features)
])

full_pipeline.fit(X_train)

X_train = full_pipeline.transform(X_train)
X_val = full_pipeline.transform(X_val)

# again but with standard scaler for the neural nets
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
])

# numerical + categorical
full_pipeline_std = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_features)
])

full_pipeline_std.fit(X_train_pca)

X_train_std = full_pipeline_std.transform(X_train_pca)
X_val_std = full_pipeline_std.transform(X_val_pca)

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
elnet_b = LogisticRegression(penalty="elasticnet", fit_intercept=True,
                            max_iter=100, solver="saga", verbose=0)

# hyperparameters
paramgrid = {"C": np.logspace(-7, 3, num=25, base=10),
             "l1_ratio": [0, .1, .5, .7, .9, .95, .99, 1],
             "class_weight": [None, "balanced"]}

# randomized search CV
cv_elnet_b = RandomizedSearchCV(estimator=elnet_b, param_distributions=paramgrid,
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                               n_iter=2, scoring="roc_auc",
                                random_state=0, verbose=100)

# train
start = time()
cv_elnet_b.fit(X_train, y_train_b.values.ravel())
time_spent = time() - start

# predict probabilities and predictions on validation set using optimal threshold
y_val_preds = np.where(cv_elnet_b.predict_proba(X_val) > 0.5, "normal", "attack")[:, 1]
y_val_probas = cv_elnet_b.predict_proba(X_val)

# validaton scores
accuracy_score(y_val_b, y_val_preds)
roc_auc_score(y_val_b, y_val_probas[:, 1])
balanced_accuracy_score(y_val_b, y_val_preds)

print(confusion_matrix(y_val_b, y_val_preds))
print(classification_report(y_val_b, y_val_preds))