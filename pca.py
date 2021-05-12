from sklearn.decomposition import PCA

variances = [0.8, 0.9, 0.95, 0.99]

acc = []
auc = []

for variance in variances:
    print("variance:", variance)
    pca = PCA(n_components=variance)

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    #insert classifier here


