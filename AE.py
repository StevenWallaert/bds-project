# AE
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras

stacked_encoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[X_train_std.shape[1]]),
    keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(20, activation="selu", kernel_initializer="lecun_normal")
])

stacked_decoder = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[20]),
    keras.layers.Dense(50, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(X_train_std.shape[1], activation=None)
])

ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

ae.compile(loss="mse", optimizer=keras.optimizers.Nadam())

history = ae.fit(X_train_std, X_train_std, epochs=20, validation_split=0.1)
# plot learning curve
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0.99, 1)
plt.show()



codings = stacked_encoder.predict(X_train_std)

plt.scatter(codings[:, 0], codings[:, 1], c=[i == "normal" for i in y_train_b.values], marker="o", alpha=0.1)

val_codings = stacked_encoder.predict(X_val_std)
from MulticoreTSNE import MulticoreTSNE as TSNE_MC

tsne = TSNE_MC(n_jobs=12, n_components=2, n_iter=400, verbose=10, random_state=0)
X_train_codings_2d = tsne.fit_transform(codings)

plt.scatter(X_train_codings_2d[:, 0], X_train_codings_2d[:, 1], c=[i == "normal" for i in y_train_b.values], marker="o", alpha=0.1)

np.savetxt("codings_tsne.csv", X_train_codings_2d, delimiter=",")

pd.DataFrame(y_train).to_csv("y_b.csv")
pd.DataFrame(y_train).to_csv("y_c.csv")