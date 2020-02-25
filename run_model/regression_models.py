import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def split_test_train(df, y):
    pass

def generate_bl_model_data(df, print_ = True, y_col = "aar_5"):
    """
    Generate data for the baseline models in step 3.
    """
    cat_cols = list(df.columns[df.dtypes == 'object'])
    drop_cols_baseline = ['price_t-5', 'vol_t-5', 'sp_price_t-5', 'sp_vol_t-5',
           'price_t-4', 'vol_t-4', 'sp_price_t-4', 'sp_vol_t-4', 'price_t-3',
           'vol_t-3', 'sp_price_t-3', 'sp_vol_t-3', 'price_t-2', 'vol_t-2',
           'sp_price_t-2', 'sp_vol_t-2', 'price_t-1', 'vol_t-1', 'sp_price_t-1',
           'sp_vol_t-1', 'price_t0', 'vol_t0', 'sp_price_t0', 'sp_vol_t0',
           'price_t1', 'vol_t1', 'sp_price_t1', 'sp_vol_t1', 'price_t2', 'vol_t2',
           'sp_price_t2', 'sp_vol_t2', 'price_t3', 'vol_t3', 'sp_price_t3',
           'sp_vol_t3', 'price_t4', 'vol_t4', 'sp_price_t4', 'sp_vol_t4',
           'price_t5', 'vol_t5', 'sp_price_t5', 'sp_vol_t5','expected_t-5', 'expected_t-4', 'expected_t-3',
           'expected_t-2', 'expected_t-1', 'expected_t0', 'expected_t1',
           'expected_t2', 'expected_t3', 'expected_t4', 'expected_t5', 'ar_t-5',
           'ar_t-4', 'ar_t-3', 'ar_t-2', 'ar_t-1', 'ar_t0', 'ar_t1', 'ar_t2',
           'ar_t3', 'ar_t4', 'ar_t5', 'aar_0', 'aar_1', 'aar_2', 'aar_3', 'aar_4', 'aar_5',
           'aar_0%', 'aar_1%', 'aar_2%', 'aar_3%', 'aar_4%', 'aar_5%'] + cat_cols
    drop_cols_baseline.remove(y_col)
    drop_cols_baseline.remove('sector')
    baseline_models_data = df.drop(drop_cols_baseline, axis = 1)
    baseline_models_data = pd.get_dummies(data = baseline_models_data,\
                                          prefix='sector',
                                          columns = ['sector'],
                                          drop_first = True)
    baseline_models_data.reset_index(inplace = True, drop = True)
    if print_: print("Baseline model features: {}".format(set(baseline_models_data.columns) - set([y_col])))
    return baseline_models_data

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
