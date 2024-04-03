# Introduction

Today we will implement a feature-based active learning (AL) strategy. There are several approaches when it comes to feature-based learning,
with the distance-based strategy being one of the easiest. The python code below is an AL strategy that selects samples based on their
euclidean distance from the centre of the feature space. The first instance will be one closest to the centre and thereafter samples furthest
from the centre are selected.

## Step 1
```Python
# Import libraries
from scipy.spatial.distance import cdist # This library readily calculates the euclidean distance
# import the iris dataset from sklearn
# import numpy
# import pandas
# Import your classifier of choice
# Import your metric of choice

# Once the libraries have been imported, define an active learning function to make the operation seamless

definitely active_learning(X, y, n_queries=5):

    # Remember the original indices
    original_indices = np.arange(X.shape[0])

    # Initialize/create an empty array for both the input and output
    X_train = np.empty((0, X.shape[1]))
    y_train = np.array([])
    
    model = ModelClassifier()

    for i in range(n_queries):
        if i == 0:
            # Calculate the centre of the feature space and select the closest sample (X') for the first training instance
            central_point = np.mean(X, axis=0).reshape(1, -1)
            distances = cdist(X, central_point, 'euclidean').flatten()
            query_idx = np.argmin(distances)
        else:
            # Thereafter, find the most distant sample
            distances = cdist(X, central_point, 'euclidean').flatten()
            query_idx = np.argmax(distances)

        # Update the original index number
        original_query_idx = original_indices[query_idx]

        # Add the queried sample to the training set
        X_query, y_query = X[query_idx].reshape(1, -1), y[query_idx].reshape(1,)
        X_train = np.vstack([X_train, X_query])
        y_train = np.append(y_train, y_query)

        # Remove the queried sample from X, y, and the index number
        X = np.delete(X, query_idx, axis=0)
        y = np.delete(y, query_idx, axis=0)
        original_indices = np.delete(original_indices, query_idx)

        # Train (or re-train) the learner with the updated training set
        pred = model.fit(X_train, y_train).predict_proba(X)

        3Dprint(f"Query {i+1}: Original sample index {original_query_idx} added. The My metric of choice score is {metric(y, pred)}.")

# Now let's test the code
X,y = load_iris(return_X_y = True)
active_learning(X, y, n_queries=10)

```

## Step 2
Having implemented the code, how confident are you in the selection criteria? There are different approaches to investigating the selection
criteria, but as always the easiest is to visualise the results. Attempt the following task and see whether the feature-based strategy captured 
the most informative samples of our feature space.

```Python
# Import libraries
# Import matplotlib
# Import seaborn
# Import PCA

# Step A: initialise PCA

# Step B: Fit and Transform PCA to your dataset

# Step C: Convert your results into a dataframe. Do not forget to add column names

# Step D: Execute the following code...

plt.figure(figsize=(20, 6)) # You might want to play around with these values
plot = sns.scatterplot(data = df_pca_iris, x = 'PC1', y = 'PC2', s = 50)

for i in range(df_pca_iris.shape[0]):
    plot.text(df_pca_iris.iloc[i, 0], df_pca_iris.iloc[i, 1], str(i), weight='bold', color='green', ha='right', va='bottom')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
```




## Step 3
How did the result compare to AL when using learner uncertainty? 
In addition, why not compare the results between different learners to determine if one learner can quickly achieve a high score from the outset?
