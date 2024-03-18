# Introduction

This week, we're diving into how to code an Active Learning (AL) model, specifically focusing on the strategy of uncertainty sampling via least confidence. 
Unlike previous exercises, this week's task will be structured differently. Instead of encountering intentional bugs, you'll find comments throughout 
the code prompting you to add your insights. So when you see "# Please insert comment", please replace it with your own comment. This approach encourages 
you to actively think about the code you're executing. For an optimal learning experience, I recommend executing each line of code in individual Colab cells, 
with the exception of the 'for loop', which should be run in a single cell to maintain its iterative context. 

# The Code
As we discussed during the lectures, there are several AL strategies available. Uncertainty is a common and effective strategy that uses a learner's predictive 
confidence to guide us in which samples to query for each cycle. In other words, the samples in which the learner is less confident about in its prediction are 
prioritised and tested. To keep things simple, we will work with the iris dataset and train our learner, random forest, on initially 5 samples and ask it to 
predict the remaining 145 samples. For each iteration, the code records the Matthews Correlation Coefficient (MCC), which will help us track whether the learner 
is improving in performance or not. At the end, you will plot a bar plot that looks at how the MCC score changes with increasing query cycle. There are more tasks
below the code if you wish to further challenge yourself.


``` Python

#import libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier # Please insert comment
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import numpy as np # Please insert comment
import pandas as pd # Please insert comment
import matplotlib.pyplot as plt # Please insert comment

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Please insert comment
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5, random_state=42)

# Please insert comment
clf_rf = RandomForestClassifier(random_state=42)

# List to store MCC scores
mcc_scores = []

for iteration in range(10): #Please insert your own comment here
    # Please insert comment
    clf_rf.fit(X_train, y_train)
    pred_q1 = clf_rf.predict_proba(X_test)
    
    # Calculate uncertainty
    uncertainty = (1 - np.max(pred_q1, axis=1)) * 100
    
    # Find indexes of the 2 most uncertain samples
    q_indexes = np.argsort(uncertainty)[-2:]
    
    # Add these samples to the training set
    X_train = np.vstack((X_train, X_test[q_indexes]))
    y_train = np.concatenate((y_train, y_test[q_indexes]))
    
    # Remove these samples from the test set
    X_test = np.delete(X_test, q_indexes, axis=0)
    y_test = np.delete(y_test, q_indexes)
    
    # Retrain the model and compute MCC score
    y_pred = clf_rf.fit(X_train, y_train).predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    mcc_scores.append(mcc)

# Please insert comment
pd.DataFrame(mcc_scores).plot.bar()
```

# Optional Tasks

1. For a holistic evaluation, repeat the analysis and track other metrics (e.g., F1 score)
2. For a reproducible analysis, repeat the analysis by changing the initial 5 starting samples and record the mean ± std over 10 cycles
3. For a real challenge, create a function that allows you to seamlessly adjust the learner, the starting sample size and the number of samples to test per query
