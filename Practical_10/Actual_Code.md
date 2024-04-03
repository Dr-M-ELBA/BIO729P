```Python


def active_learning(X, y, n_queries=5):
    # Remember the original indeices
    original_indices = np.arange(X.shape[0])


    X_train = np.empty((0, X.shape[1]))
    y_train = np.array([])


    model = RandomForestClassifier(random_state=42)

    for i in range(n_queries):
        if i == 0:

            central_point = np.mean(X, axis=0).reshape(1, -1)
            distances = cdist(X, central_point, 'euclidean').flatten()
            query_idx = np.argmin(distances)
        else:

            distances = cdist(X, central_point, 'euclidean').flatten()
            query_idx = np.argmax(distances)


        original_query_idx = original_indices[query_idx]


        X_query, y_query = X[query_idx].reshape(1, -1), y[query_idx].reshape(1,)
        X_train = np.vstack([X_train, X_query])
        y_train = np.append(y_train, y_query)

        X = np.delete(X, query_idx, axis=0)
        y = np.delete(y, query_idx, axis=0)
        original_indices = np.delete(original_indices, query_idx)

        pred = model.fit(X_train, y_train).predict(X)

        print(f"Query {i+1}: Original sample index {original_query_idx} added. The MCC score is {mcc(y, pred)}.")

```
