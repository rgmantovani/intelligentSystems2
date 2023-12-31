# https://towardsdatascience.com/adaboost-from-scratch-37a936da3d50

import numpy as np

class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
    
    # Compute error rate, alpha and w
    def compute_error(self, y, y_pred, w_i):
        '''
        Calculate the error rate of a weak classifier m. Arguments:
        y: actual target value
        y_pred: predicted value by weak classifier
        w_i: individual weights for each observation
        
        Note that all arrays should be the same length
        '''
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

    def compute_alpha(self, error):
        '''
        Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
        alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
        error: error rate from weak classifier m
        '''
        return np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        ''' 
        Update individual weights w_i after a boosting iteration. Arguments:
        w_i: individual weights for each observation
        y: actual target value
        y_pred: predicted value by weak classifier  
        alpha: weight of weak classifier used to estimate y_pred
        '''  
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))


    def fit(self, X, y, M = 100):
        '''
        Fit model. Arguments:
        X: independent variables - array-like matrix
        y: target variable - array-like vector
        M: number of boosting rounds. Default is 100 - integer
        '''
        
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in range(0, M):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # (d) Update w_i
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)

    def predict(self, X):
        '''
        Predict using fitted model. Arguments:
        X: independent variables - array-like
        '''

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
