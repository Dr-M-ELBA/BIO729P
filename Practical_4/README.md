# Practical 4

## Introduction

For this week, we covered common pre-processing techniques applied during the machine learning (ML) pipeline. For the practical, you will put some of these into practice. You will work with the <ins>Pima Indian Diabetes</ins> and the aim will be to compare the effects of pre-processing on two learners: k-nearest neighbor (kNN) and multi-layer perceptron (MLP). The pre-processing that you will apply will include scaling, under-sampling the majority class and over-sampling the minority class. Once you have finished, make sure to document your findings on github.


## Lab Task 1

1. Clone the repo to github

!git clone https://github.com/Dr-M-ELBA/Practical-4.git

2. import the libraries

3. ```Python

   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import sys
  
   # pre-processing libraries
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import make_pipeline

   # ML training libraries
   from sklearn.model_selection import train_test_split
   from sklearn.neural_network import MLPClassifier
   from sklearn.neighbors import KNeighborsClassifier

   # Evaluation libraries
   from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

   # Class imbalance libraries
   from imblearn.under_sampling import RandomUnderSampler
   from imblearn.over_sampling import RandomOverSampler, SMOTE
   ```

4. Make sure the libraries are installed. On colab, this can easily be done by executing a cell with !pip install followed by the library. For example: !pip install imbalanced-learn

5. Once you've imported the relevant libraries, try the following
   a. Upload the dataset and check for any missing values.Feel free to perform any other exploratory data analysis (EDA).
   b. Assign the input features to X and the target to y
   c. Compare the results of MLP when the dataset is unscaled vs scaled. Compare the accuracy, f1 score when the minority class is set as the positive class, and the MCC.
   d. Compare the results of kNN when the dataset is unscaled vs scaled. Compare the accuracy, f1 score when the minority class is set as the positive class, and the MCC.

   Note, make sure to use the make_pipeline function to avoid data leakage. For example:

   ``` Python

   scaled_mlp_pipeline = make_pipeline(StandardScaler(), MLPClassifier(random_state=42))
   scaled_mlp_pipeline.fit(X_train, y_train)
   ```

6. Once you have reported the effects of scaling on the performance of MLP and kNN. Move on to lab task 2

## Lab Task 2

For the second task, you will evaluate the effects of both under- and oversampling on the performance of MLP and kNN, using the same metrics as in Lab Task 1. If you do not finish, then please continue the following week.

1. Start off my randomly undersampling the majority class and record the accuracy, F1 score and MCC.
2. Repeat the analysis but this type randomly oversample the minority class and evaluate the results of both MLP and kNN.
3. Lastly, apply SMOTE to oversample the minority class and evaluate the results of both MLP and kNN.

Once you have finished, upload your findings to github.

   
