import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from random_forest import RandomForest

# create accuracy metric
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Load the iris dataset
data = datasets.load_iris()
X = data.data
y = data.target

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)  # 80% training and 20% test

# create a random forest
clf = RandomForest(n_trees=3, max_depth=10)
clf.fit(X_train, y_train)

# make predictions
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print ("Accuracy:", acc)