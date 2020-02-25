import numpy as np
import pandas as pd
import statsmodels.api as sm




def run_linear_reg(data, print_summary = False, print_r2 = True, y_col = "aar_5"):
    """
    Run linear regression
    param: data: DataFrame , print_:Boolean
    return: linear regression model
    """
    data = sm.add_constant(data, prepend=False)
    X = data.drop(y_col, axis = 1)
    Y = data[y_col]
    result = sm.OLS(Y, X).fit()
    if print_summary: print(result.summary())
    if not print_summary and print_r2: print('R^2: {}'.format(result.rsquared))
    return result
