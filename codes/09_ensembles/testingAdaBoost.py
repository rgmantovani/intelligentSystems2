# Dataset
df = pd.read_csv('./spambase/spambase.data', header = None)

# Column names
names = pd.read_csv('./spambase/spambase.names', sep = ':', skiprows=range(0, 33), header = None)
col_names = list(names[0])
col_names.append('Spam')

# Rename df columns
df.columns = col_names

# Convert classes in target variable to {-1, 1}
df['Spam'] = df['Spam'] * 2 - 1

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'Spam').values, 
                                                    df['Spam'].values, 
                                                    train_size = 3065, 
                                                    random_state = 2) 

# Fit model
ab = AdaBoost()
ab.fit(X_train, y_train, M = 400)

# Predict on test set
y_pred = ab.predict(X_test)
