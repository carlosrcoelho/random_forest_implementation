import numpy as np
from collections import Counter  # Counter is a dict subclass for counting hashable objects

# create entropy function to calculate entropy
def entropy(y):
    hist = np.bincount(y)  # count the number of each class
    ps = hist / len(y)     # calculate the probability of each class
    return -np.sum([p * np.log2(p) for p in ps if p > 0])      # calculate the entropy


# create Node class to store the information of each node
class Node:
    # initialize the node
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):  # * means the following parameters are keyword-only arguments
        self.feature = feature          # feature to split on
        self.threshold = threshold      # threshold value of the feature
        self.left = left                # left child
        self.right = right              # right child
        self.value = value              # value if the node is a leaf node

    # check if the node is a leaf node
    def is_leaf_node(self):
        return self.value is not None   # if the node has a value, it is a leaf node


# create DecisionTree class to build the decision tree
class DecisionTree:
    # initialize the decision tree
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split    # minimum number of samples to split a node
        self.max_depth = max_depth                    # maximum depth of the tree
        self.n_feats = n_feats                        # number of features to consider when splitting

        self.root = None                              # root node of the decision tree

    
    def fit(self, X, y):  
        # grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])   # if n_feats is None, use all features
        self.root = self._grow_tree(X, y)                                                    # grow the tree from the root node

    def _grow_tree(self, X, y, depth=0):   # depth is the current depth of the tree
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria, which means we reach the leaf node
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)   # randomly select n_feats features
        
        # greedy search, which means we find the best feature and threshold to split on
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    def _best_criteria(self, X, y, feat_idxs):  # find the best feature and threshold to split on
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):  # calculate the information gain
        # parent Entropy
        parent_entropy = entropy(y)
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weighted average child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        # return ig
        ig = parent_entropy - child_entropy
        return ig
    
    def _split(self, X_column, split_thresh):  # split the data
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # traverse tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)