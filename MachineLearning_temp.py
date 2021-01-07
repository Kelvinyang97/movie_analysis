from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn._selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
### Linear Regression
# Importing dataset
data = pd.read_csv('movie.csv')
# Descriptive statistics
df.describe()
# Name of the cols y and X
data.columns
# Visualisation
sns.pairplot(data)
# See if the y is normally distributed for linear regression
sns.distplot(data['Revenue'])
sns.heatmap(data.corr(), annot=True)

# Building model
X = data[['insert X var names']]
y = data['Revenue']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Training the linear model
lm = LinearRegression()
lm.fit(X_train, y_train)
lm_res = pd.DataFrame(lm.coef_, X.columns, columns=['Marginal Effect'])

### Random Forest
# Default Random Forest as the baseline here
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
# Tuning the hyperparameters
# Creating the range of hyperparameters to search from
# Number of trees
n_estimators = [int(i) for i in np.linspace(start = 100, stop = 2000, num = 20)]
# Number of features to consider at each node
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(k) for k in np.linspace(start = 10, stop = 150, num = 15)]
# Minimum number of samples required to split a node
min_samples split = [2, 5, 10]
# Minimuym number of samples retuired at each leaf node
min_samples_leaf = [1, 2, 5]

# Create the random search grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_sample_split': min_samples_split
               'min_samples_leaf': min_samples_leaf
               }
pprint(random_grid)

# Random Search Training for finding the best hyperparameters
rf_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 200, cv = 5, verbose = 2, random_state = 42, scoring = 'neg_mean_squared_error')
rf_search.best_params_

# Grid Search with smaller range: Obtain a more accurate result
# by fitting all combinations in a narrower range

### XG Boost with trees
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from modelfit import modelfit

# Run base line GBM without tuning
gbm_base = XGBClassifier(learning_rate = 0.1, n_estimators = 1000, max_depth = 6, 
                         min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8,
                         seed = 42)

# Concatenating two test data
data_test = np.concatenate((y_train, X_train, axis = 1))
predictors = [x for x in data_test if x not in Revenue]
# Obtaining the best n_estimators
modelfit(gmb_base, data_test, predictors)

# Tuning max_depth and min_child_weight
max_depth = range(4, 10, 2)
min_child_weight = range(1, 8, 2)

Grid_1 = {'max_depth': max_depth,
        'min_child_weight': min_child_weight
}
Grid_search1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimator = "insert best n_estimators from base", 
                            max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, seed = 42),
                            param_grid = Grid_1, scoring = 'neg_mean_squared_error', n_jobs = 4, iid = False, cv = 5)
Grid_search1.fit(X_train, y_train)
# Obtaining the best max_depth and min_child_weight values
Grid_search1.best_params_, Grid_search1.best_score_

# Tuning gamma (the minimum loss reduction required to make a further partition on a leaf node of the tree)
Grid_2 = {
    'gamma': np.arange(0, 0.6, 0.1)
}
Grid_search2 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimator = "insert best n_estimators from base", 
                            max_depth = "insert best max_depth",
                            min_child_weight = "insert best min_child_weight", gamma = 0, subsample = 0.8, colsample_bytree = 0.8, 
                            seed = 42), param_grid = Grid_2, scoring = 'neg_mean_squared_error', n_jobs = 4, iid = False, cv = 5)
Grid_search2.fit(X_train, y_train)
# Obtaining the best gamma value
Grid_search2.best_params_, Grid_search2.best_score_
##! Cloud potentially recalibrate the n_estimates before tuning other hyperparameters
# Tuning subsample and colsample_bytree using random search rather than grid search to be a bit mode efficient
 Grid_3 = {
    'subsample': np.arange(0, 1, 0.05)
    'col_sample_bytree': np.arange(0, 1, 0.05)
}
Grid_search3 = RandomizedSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimator = "insert best n_estimators from base", 
                            max_depth = "insert best max_depth",
                            min_child_weight = "insert best min_child_weight", gamma = "insert best gamma", subsample = 0.8, 
                            colsample_bytree = 0.8,  seed = 42), param_distributions = Grid_3, n_iter = 50, 
                            scoring = 'neg_mean_squared_error', n_jobs = 4, iid = False, cv = 5)
Grid_search3.fit(X_train, y_train)
# Obtaining the best subsample and col_sample_tree values
Grid_search3.best_params_, Grid_search3.best_score_

# Tuning regularization parameters (alpha only here if needed add lambda)
Grid_4 = {
    'reg_alpha': np.linspace(start = 0, stop = 100, num = 40)
}
Grid_search4 = RandomizedSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimator = "insert best n_estimators from base", 
                            max_depth = "insert best max_depth",
                            min_child_weight = "insert best min_child_weight", gamma = "insert best gamma", 
                            subsample = "insert best subsample", 
                            colsample_bytree = "insert best colsample_bytree", 
                            seed = 42), param_distributions = Grid_4, n_iter = 20, scoring = 'neg_mean_squared_error', 
                            n_jobs = 4, iid = False, cv = 5)
Grid_search4.fit(X_train, y_train)
# Obtaining the best subsample and col_sample_tree values
Grid_search4.best_params_, Grid_search4.best_score_

### Support Vector Regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

St = StandardScaler()
St_X = St.fit_transform(X_train)
St_y = St.fit_transform(y_train)

SVR.fit(St_X, St_y, kernel = 'rbf')

plt.scatter(St_X, St_y, color = 'red')
preds = SVR.predict((St_X, St_y, kernel = 'rbf')
plt.plot(St_X, preds, color = 'blue')


