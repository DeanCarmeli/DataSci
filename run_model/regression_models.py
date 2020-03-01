import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

mse_exp = 4


def mse(preds, y):
    n = len(y)
    p = 10 ** mse_exp
    return p * (((preds - y) ** 2).sum() / n)


def predict_linear_reg(model, data, y_col):
    data = sm.add_constant(data, prepend=False)
    X = data.drop(y_col, axis=1)
    Y = data[y_col]
    return model.predict(X)


def run_linear_reg(data, test_data=None, print_summary=False, print_r2=True, y_col="aar_5"):
    """
    Run linear regression
    param: data: DataFrame , print_:Boolean
    return: linear regression model
    """
    data = sm.add_constant(data, prepend=False)

    X = data.drop([y_col], axis = 1).astype(float)
    Y = data[y_col].astype(float)
    
    result = sm.OLS(Y, X).fit()
    if print_summary: print(result.summary())
    if not print_summary and print_r2:
        print('Train data R squared: {}'.format(result.rsquared))
        if test_data is not None: print("Test data R squared: {}".format(r_squared(result, test_data, y_col=y_col)))
    return result


def evaluate_Rsq_CV(data, y_col="aar_5", n_splits=5, print_r2=True):
    """
    Split the data into n_splits folds, at each iteration fit linear regression for n_splits-1 folds and calculate R sqaured for the last fold.
    param: data: DataFrame , print_r2: Boolean
    return: list of the R sqaured values
    """
    data = sm.add_constant(data, prepend=False)
    kf = KFold(n_splits=n_splits)
    X = data.drop([y_col], axis=1)
    Y = data[y_col]
    R_sqs = []
    n = kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index,].reset_index(drop=True), X.loc[test_index,].reset_index(drop=True)
        y_train, y_test = Y.loc[train_index,].reset_index(drop=True), Y.loc[test_index,].reset_index(drop=True)
        model = sm.OLS(y_train, X_train).fit()
        R_sqs.append(r_squared(model, pd.concat([X_test, y_test], axis=1), y_col))
    if print_r2: print("CV mean R squared: {}".format(np.array(R_sqs).mean()))
    return R_sqs



def step7_bl_models(X_train, X_test, y_train, y_test, lr_features = ['delta_%_t-4'], print_ = True):
    """
    Return two prediction models:
        1. Simple linear regression
        2. Constant prediction model
    Params:
    print_: if true print test-MSE of both
    """
    model1 = LinearRegression()
    model1.fit(X_train[lr_features], y_train)
    model2 = DummyRegressor(strategy = 'mean')
    model2.fit(X_train, y_train)
    if print_: 
        print("Linear Regression model mse*e+{}: {}".format(mse_exp, mse(model1.predict(X_test[lr_features]), y_test)))
    model1.fit(X_train[['delta_%_t-4']], y_train)
    model2 = DummyRegressor(strategy='mean')
    model2.fit(X_train, y_train)
    if print_:
        print("Linear Regression model mse*e+{}: {}".format(mse_exp,
                                                            mse(model1.predict(X_test[['delta_%_t-4']]), y_test)))
        print("Constant model mse*e+{}: {}".format(mse_exp, 
                                                   mse(model2.predict(X_test), y_test)))
    return model1, model2


def rfe(data, y_col, model=None, n_features_to_select=1):
    """
    Wrapper for sklearn's RFE
    Params:
        model: if None use linear regression
    Return: list of selcted features
    """
    X, y = data.drop([y_col], axis=1), data[y_col]
    if not model: model = LinearRegression().fit(X, y)
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    return set(X.columns[rfe.support_])




def r_squared(model, data, y_col):
    """
    compute R^2
    """
    y = data[y_col]
    y_hat = predict_linear_reg(model, data, y_col)
    RSS = ((y - y_hat) ** 2).sum().item()
    SST = ((y - y.mean()) ** 2).sum().item()
    R_squared = 1 - (RSS / SST)  # by definition
    return R_squared

####Random Forest
def fit_rf(X, y, n_estimators):
    rf_model = RandomForestRegressor(n_estimators=n_estimators)
    rf_model.fit(X, y)
    return rf_model

def rf_cv(X_train, y_train, ns, ds):
    """
    CV for parameters in random forest.
    return: GridSearchCV
    """
    grid = {'n_estimators': ns,
                   'max_depth': ds}
    rf = RandomForestRegressor()
    gcv = GridSearchCV(estimator=rf, cv=2, param_grid = grid, verbose=100, n_jobs = -1, scoring=mse_score)
    gcv.fit(X_train , y_train)
    return gcv

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))
    plt.figure(figsize=(20,10))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        plt.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    plt.title("Grid Search Scores", fontsize=20, fontweight='bold')
    plt.xlabel(name_param_1, fontsize=16)
    plt.ylabel('CV Average Score', fontsize=16)
    plt.legend(loc="best", fontsize=15)
    plt.grid('on')
    
def total_mse(model, X, y):
    predictions = model.predict(X) + bl_model_lg.predict(X[['delta_%_t-4']])
    result = regression_models.mse(predictions, y)
    print("total mse: {}".format(result))
    return result

def mse_score(model, X, y):
    predictions = model.predict(X)
    result = mse(predictions, y)
    return result