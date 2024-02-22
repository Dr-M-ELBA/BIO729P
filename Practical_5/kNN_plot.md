# Plotting the K values
A Code for visually observing the impact of the k value on the accuracy. 


``` Python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split  


# Assuming your X and y have been defined.

#Split arrays or matrices into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 
knn = KNeighborsClassifier(n_neighbors=7)  
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)

#Check the code is working
print("Preliminary model score:")
print(knn.score(X_test,y_test))

no_neighbors = np.arange(1, 11) # select the range of k values to explore
train_accuracy = np.empty(len(no_neighbors)) # setup an empty array for the training accuracy
test_accuracy = np.empty(len(no_neighbors)) # setup an empty array for the testing accuracy

# A for loop to reiterate through the code
for i, k in enumerate(no_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k) # What is happening here?
    knn.fit(X_train,y_train) # And here?
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Visualization of k values vs accuracy

plt.title('The effect of k on Accuracy')
plt.plot(no_neighbors, test_accuracy, label = 'Test Accuracy')
plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show() ```
