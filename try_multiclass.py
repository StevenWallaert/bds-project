import pandas as pd
from sklearn.datasets import make_classification

X_try, y_try = make_classification(n_samples=20)

### minimal reproducible example
import numpy as np

X = np.array([1, 0, 2, 0, 3, 0, 4, 0]).reshape(-1, 1)
y = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y) # this line fails

##
import numpy as np
X = np.array([1, 0, 2, 0, 3, 0, 4, 0]).reshape(-1, 1)
y = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

X = np.concatenate([np.ones(shape=[8, 1]), X], axis=1)
betas = np.invert(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y) # this line fails
