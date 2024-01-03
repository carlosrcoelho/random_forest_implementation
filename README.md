# Random Forest

Random Forest is like a group decision-making process where you ask several people (trees) a question (to predict something). Each person has their own experiences and perspectives.

Here's how it works:

Many Decision-Makers (Trees): You gather a bunch of people (trees) and ask them a question.
Each Has a Say: Each person (tree) gives an answer based on their own knowledge (subset of features).
Voting: Then, everyone's answers are counted, and the most popular answer (mode in classification or average in regression) is chosen as the final decision.
This teamwork helps to reduce mistakes made by any single individual and tends to give a more reliable answer, making it great for making predictions in various situations in machine learning.

Here's a step-by-step explanation of the code:


### Step 1: Bootstrap Sample Creation
```python
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]
```
This function `bootstrap_sample` creates a bootstrap sample by randomly selecting rows (samples) from the dataset `X` along with their corresponding labels `y`. Sampling is done with replacement (`replace=True`).

### Step 2: Finding Most Common Label
```python
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common
```
`most_common_label` uses the `Counter` class from the `collections` module to identify the most common label in a given set of labels `y`.

### Step 3: RandomForest Class Initialization
```python
class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []  # List to store the decision trees created
```
The `RandomForest` class initializes a Random Forest classifier with parameters such as the number of trees, minimum samples to split a node, maximum depth of the trees, and number of features to consider when splitting.

### Step 4: Fit Method
```python
def fit(self, X, y):
    self.trees = []
    for _ in range(self.n_trees):
        tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
        X_sample, y_sample = bootstrap_sample(X, y)
        tree.fit(X_sample, y_sample)
        self.trees.append(tree)
```
The `fit` method of `RandomForest` constructs the ensemble by creating `n_trees` Decision Trees, generating a bootstrap sample from the data, fitting each tree to its respective bootstrap sample, and storing the trained trees in a list.

### Step 5: Predict Method
```python
def predict(self, X):
    tree_preds = np.array([tree.predict(X) for tree in self.trees])
    tree_preds = np.swapaxes(tree_preds, 0, 1)
    y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
    return np.array(y_pred)
```
The `predict` method uses the ensemble of trained Decision Trees to make predictions for new data `X`. It collects predictions from each tree, transposes the predictions for easier processing, and finds the most commonly predicted class among the trees for each data point.

This implementation creates a Random Forest classifier by utilizing Decision Trees as base learners, leveraging bootstrap sampling and aggregation of predictions to enhance predictive performance.
