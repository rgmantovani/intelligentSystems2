## Import decision trees
import import_ipynb
import regression_tree as rt;

## Import numpy and visualization packages
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

class Bagger:
    
    def fit(self, X_train, y_train, B, max_depth = 100, min_size = 2, seed = None):
        
        self.X_train = X_train
        self.N, self.D = X_train.shape
        self.y_train = y_train
        self.B = B
        self.seed = seed
        self.trees = []
        
        np.random.seed(seed)
        for b in range(self.B):
            
            sample = np.random.choice(np.arange(self.N), size = self.N, replace = True)
            X_train_b = X_train[sample]
            y_train_b = y_train[sample]
            
            tree = rt.DecisionTreeRegressor()
            tree.fit(X_train_b, y_train_b, max_depth = max_depth, min_size = min_size)
            self.trees.append(tree)
            
        
    def predict(self, X_test):
        
        y_test_hats = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X_test)
        
        return y_test_hats.mean(0)

## Build model
bagger = Bagger()
bagger.fit(X_train, y_train, B = 30, max_depth = 20, min_size = 5, seed = 123)
y_test_hat = bagger.predict(X_test)

## Plot
fig, ax = plt.subplots(figsize = (7, 5))
sns.scatterplot(y_test, y_test_hat)
ax.set(xlabel = r'$y$', ylabel = r'$\hat{y}$', title = r'Observed vs. Fitted Values for Bagging')
sns.despine()