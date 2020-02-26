import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

mse_exp = 4

def mse(preds, y):
    n = len(y)
    p = 10**mse_exp
    return p*(((preds-y)**2).sum()/n)

def predict_linear_reg(model, data, y_col):
    data = sm.add_constant(data, prepend=False)
    X = data.drop(y_col, axis = 1)
    Y = data[y_col]
    return model.predict(X)

def run_linear_reg(data, print_summary = False, print_r2 = True, y_col = "aar_5"):
    """
    Run linear regression
    param: data: DataFrame , print_:Boolean
    return: linear regression model
    """
    data = sm.add_constant(data, prepend=False)
    X = data.drop([y_col], axis = 1)
    Y = data[y_col]
    result = sm.OLS(Y, X).fit()
    if print_summary: print(result.summary())
    if not print_summary and print_r2: print('R^2: {}'.format(result.rsquared))
    return result

def step7_bl_models(X_train, X_test, y_train, y_test, print_ = True):
    """
    Return two prediction models:
        1. Simple linear regression
        2. Constant prediction model
    Params:
    print_: if true print test-MSE of both
    """
    model1 = LinearRegression()
    model1.fit(X_train[['delta_%_t-4']], y_train)
    model2 = DummyRegressor(strategy = 'mean')
    model2.fit(X_train, y_train)
    if print_: 
        print("Linear Regression model mse*e+{}: {}".format(mse_exp, mse(model1.predict(X_test[['delta_%_t-4']]), y_test)))
        print("Constant model mse*e+{}: {}".format(mse_exp, mse(model2.predict(X_test), y_test)))
    return model1, model2
    
def rfe(data, y_col,model = None, n_features_to_select = 1):
    """
    Wrapper for sklearn's RFE
    Params:
        model: if None use linear regression
    Return: list of selcted features
    """
    X, y = data.drop([y_col], axis =1), data[y_col]
    if not model: model = LinearRegression().fit(X, y)
    rfe = RFE(estimator = model, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    return set(X.columns[rfe.support_])

def fit_rf(X, y, n_estimators):
    rf_model = RandomForestRegressor(n_estimators=n_estimators)
    rf_model.fit(X, y)
    return rf_model

def r_squared(model, data, y_col):
    """
    compute R^2
    """
    y = data[y_col]
    y_hat = predict_linear_reg(model, data, y_col)
    RSS = ((y-y_hat)**2).sum().item()
    SST = ((y-y.mean())**2).sum().item()
    R_squared = 1 - (RSS/SST) #by definition
    return R_squared

def rf_cv(X_train, y_train):
    """
    CV for parameters in randaom forest.
    return: RandomizedSearchCV
    """
    rf_model = RandomForestRegressor()
    distributions = dict(n_estimators=randint(10,500), max_depth = [None, 5, 10 ,20])
    clf = RandomizedSearchCV(rf_model, distributions, random_state=0)
    clf.fit(X_train, y_train)
    return clf