import numpy as np
from decision_tree import DecisionTree
from collections import Counter  # Counter is a dict subclass for counting hashable objects

# create a bootstrap sample
def bootstrap_sample(X, y):    # it means we are creating a sample with replacement
    n_samples = X.shape[0]    # number of rows
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)  # create a list of random numbers
    return X[idxs], y[idxs]  # return the bootstrap sample

# create most common label
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


# create a random forest class
class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []       # store the trees created in a list

    # define a fit method
    def fit(self, X, y):
        self.trees = []
        # create n_trees number of trees
        for _ in range(self.n_trees):
            # create a decision tree
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            # create a bootstrap sample
            X_sample, y_sample = bootstrap_sample(X, y)
            # fit the decision tree
            tree.fit(X_sample, y_sample)
            # append the tree to the list
            self.trees.append(tree)    

    # define a predict method
    def predict(self, X):
        # make predictions
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # transpose the predictions, so that we can get the predictions for each row
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # create a list of the predictions
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
                