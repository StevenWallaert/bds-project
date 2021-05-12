import pandas as pd
import matplotlib.pyplot as plt

# read in column names
with open("project/kddcup/colnames.txt") as columns:
    colnames = [line[:line.find(":")] for line in columns] # discard everything from ':' on per line

# add class column name
colnames.append("target")

# read in coarse class names
with open("project/kddcup/training_attack_types") as attack_types:
    key_values = [line.strip() for line in attack_types][:-1]  # last line is empty

attacks_dict = {key+".": value for (key, value) in [pair.split(" ") for pair in key_values]} # the +"." accounts for an extra dot that seems to be present in the data
attacks_dict["normal."] = "normal"
# read in training data
kdd = pd.read_csv("project/kddcup/kddcup.data.corrected",
                  sep=',',
                  names=colnames)

# how does the df look?
kdd.head(10)
kdd.tail(10)
kdd.sample(10)

# the sample is just for speed, just for now
kdd = kdd.sample(frac=0.1)

# create coarse classes and binary classes
kdd["ctarget"] = [attacks_dict[attack] for attack in kdd.target] # coarse 5 classification
kdd["btarget"] = ["normal" if attack == "normal." else "attack" for attack in kdd.target] # binary classification

# check if worked well
kdd.head(5) # looks okay

##############
# actual EDA #
##############

# shape
kdd.shape

# info on NA, datatype
kdd.info()

# we learn there are no missing data
# TODO: check if this is also the case for full data

# 3 categorical variables: protocol_type, service, flag
kdd.protocol_type.value_counts().plot.bar() # 3 levels

kdd.service.value_counts() # 65 levels

kdd.flag.value_counts() # 11 levels

# check ranges
kdd.iloc[:,0:10].hist(bins=50)

kdd.iloc[:,10:20].hist(bins=50)
# logged_in, root_shell, su_attempted might be binary
kdd.logged_in.value_counts() # binary
kdd.root_shell.value_counts() # binary
kdd.su_attempted.value_counts() # 0, 1, 2 might be categorical

kdd.iloc[:, 20:30].hist(bins=50)
# is_host_login, is_guest_login
kdd.is_host_login.value_counts() # not informative # TODO: still with full data?
kdd.is_guest_login.value_counts() # binary


kdd.iloc[:, 30:40].hist(bins=50)

kdd.iloc[:, 40:45].hist(bins=50)

# conclusion: a lot (all?) features have very skewed (or unbalanced) distributions
# idea: probably good to transform some variables

from pandas.plotting import scatter_matrix

scatter_matrix(kdd.iloc[:,0:5])
scatter_matrix(kdd.iloc[:,5:10])
scatter_matrix(kdd.iloc[:,10:15])
scatter_matrix(kdd.iloc[:,15:20])
scatter_matrix(kdd.iloc[:,20:25])
scatter_matrix(kdd.iloc[:,25:30])
scatter_matrix(kdd.iloc[:,30:35])
# conclusion: a lot of non-linear relationships between X's




