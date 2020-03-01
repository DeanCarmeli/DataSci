"""
This module include functions for use in the notebook.
The main purpose here is to encapsulate long code parts from the notebook to make it more readable.
Divided into parts: one general part, and part for each notebook step.
"""


from aggregate import aggregate
from aggregate import capm_params
from feature_handler import feature_handler
from run_model import regression_models
import zipfile
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from visualize import visualize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor






##########################General
fin_ratios = ['MV Debt Ratio',
       'BV Debt Ratio', 'Effective Tax Rate', 'Std Deviation In Prices',
       'EBITDA/Value', 'Fixed Assets/BV of Capital',
       'Capital Spending/BV of Capital']

drop_cols = [ 'sp_vol_t-1', #columns we didn't use in project at all
 'sp_vol_t-2',
 'sp_vol_t-3',
 'sp_vol_t-4',
 'sp_vol_t-5',
 'sp_vol_t0',
 'sp_vol_t1',
 'sp_vol_t2',
 'sp_vol_t3',
 'sp_vol_t4',
 'sp_vol_t5',
 'symbol',
 'vol_t-1',
 'vol_t-2',
 'vol_t-3',
 'vol_t-4',
 'vol_t-5',
 'vol_t0',
 'vol_t1',
 'vol_t2',
 'vol_t3',
 'vol_t4',
 'vol_t5',
]

def print_basic_stats(data, nans = True):
    r, c = data.shape
    nans_count = (data.isna().sum(axis = 1) > 0).sum()
    print("Data Stats:")
    print("\t#Samples: {}\n\t#Features: {}".format(r, c))
    if nans: print("\t#Samples with NaNs: {}".format(nans_count))
    return r, c, nans_count

def my_split_test_train(df, y_col=None, test_size = 0.2):
    """
    split into train/tets set
    Params:
    df: data
    y_col: the y_col name (string), if None - split df into two dataframes without considering y.
    test_size: ...
    """
    if 'div_direction' in df: stratify = df['div_direction']
    
    
    if y_col:
        X_train, X_test, y_train, y_test = train_test_split(df.drop([y_col], axis =1), df[y_col], test_size = test_size, stratify=stratify)
        return X_train.reset_index(drop = True), X_test.reset_index(drop = True),\
                y_train.reset_index(drop = True), y_test.reset_index(drop = True)
    else:
        df_train, df_test, _, _ = train_test_split(df, df, test_size = test_size,  stratify=stratify)
        return df_train.reset_index(drop = True), df_test.reset_index(drop = True)


def generate_model_data(df, 
                           window_size = 5, 
                           y_col = "aar_5", 
                           drop_08_09 = False, 
                           dummies = ['year', 'month'],
                           delta_precentage = False,
                           test_size = 0,
                           return_year = False,
                           print_ = True):
    """
    Generate data for the baseline model.
    Params:
    df: the data to generate from
    window_size: the window_size we wish to predict, either in 1,..,5 or a tuple. if a tuple the window is asymetric and y_col is ignored!
    y_col: the Y column
    drop_08_09: whether to drop 2008,2009 or not
    delta_precentage: wether to add "delta_%_t-i"  or not (i depends on the window size)
    test_size: if >0 split into train and test.
    print_: print result details
    
    Return: if no split_train_test: data
            if test_size>0: data_train , data_test
    """
    cat_cols = list(df.columns[df.dtypes == 'object'])
    if drop_08_09: 
        df = \
        (df[df['year']!= 2008][df['year']!=2009])\
        .reset_index(drop = True)

    drop_cols_baseline = set(['price_t-5', 'vol_t-5', 'sp_price_t-5', 'sp_vol_t-5',
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
           'aar_0%', 'aar_1%', 'aar_2%', 'aar_3%', 'aar_4%', 'aar_5%'] + cat_cols)
    drop_cols_baseline = drop_cols_baseline - set(drop_cols)
    drop_cols_baseline = list(drop_cols_baseline)
    
    if isinstance(window_size, tuple):
        start, end = window_size
        df = feature_handler.create_asymmetric_window(df ,start, end)
        y_col = "aar_asy{}_{}%".format(str(start), str(end))
        start = abs(start)
    else: start = window_size
        
    if start < 3:
        check = lambda c: sum(["t-{}".format(i) in c for i in range(start+3, 6)]) > 0 
        for c in [c for c in drop_cols_baseline if check(c)]: drop_cols_baseline.remove(c)
        if delta_precentage: 
            df = feature_handler.gen_delta_precent_t(df, ts = list(range(start+3,6)),print_ = False)

    if y_col in drop_cols_baseline: drop_cols_baseline.remove(y_col)
    if dummies:
        for c in dummies: 
            if c in drop_cols_baseline: drop_cols_baseline.remove(c)
    if return_year: year_col = df['year']
    baseline_models_data = df.drop(drop_cols_baseline, axis = 1)
    if dummies:
        baseline_models_data = pd.get_dummies(data = baseline_models_data,\
                                          columns = dummies,\
                                          drop_first = True)
    baseline_models_data.reset_index(inplace = True, drop = True)
    if print_: print("Baseline model features: {}".format(set(baseline_models_data.columns) - set([y_col])))
    if test_size>0:
        return split_test_train(baseline_models_data, y_col=None, test_size = 0.33)
    if return_year: return baseline_models_data, year_col
    return baseline_models_data



##########################Step 1
def unzip_data():
    """
    Unzips the merged data that was created by the fetch_data module.
    Name of the unzipped file = "all_prices.json"
    :return: None
    """
    with zipfile.ZipFile("all_prices.zip", "r") as zip_ref:
        zip_ref.extractall()
    pass


def get_quarter(x):
    if x == 1 or x == 2 or x == 3:
        return 1
    if x == 4 or x == 5 or x == 6:
        return 2
    if x == 7 or x == 8 or x == 9:
        return 3
    return 4


def split_date(df):
    # create cols for year month and quarter
    df['dividend_date'] = pd.to_datetime(df['dividend_date'], infer_datetime_format=True)
    df['year'] = df['dividend_date'].apply(lambda x: x.year)
    df['month'] = df['dividend_date'].apply(lambda x: x.month)
    df['quarter'] = df['month'].apply(lambda x: get_quarter(x))
    return df


def do_aggregate_steps(all_prices):
    """
    WARNING - THIS METHOD TAKES ABOUT 5 MINUTES TO RUN
    Adds the data sector, industry and 7 financial ratios.
    :param all_prices: dictionary with dividend data and prices
    as returned by fetch_data module.
    :return: DataFrame
    """

    # Step 1: Calculate alpha and beta parameters based on CAPM model
    data_with_capm = capm_params.get_data_with_capm_params(all_prices)

    # Step 2: Add sector and industry + change format from dict to DataFrame
    df = aggregate.aggregate_meta_data(data_with_capm)
    df = aggregate.fill_missing_meta_data(df)

    # Step 3: Split the date column to have year, month and quarter
    df = split_date(df)

    # Step 4: Add financial ratios.
    # Drop samples from before 1998 (no fin ratio data)
    df.drop(df[df["year"] <= 1997].index, inplace=True)
    df = aggregate.aggregate_fin_ratios(df)
    return df

def step1_wrapper(df, run_speed, print_ = False):
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    print("Dropped {} rows with NaN values".format(rows_before-rows_after))

    # dividend_amount to integer
    df = feature_handler.create_div_amount_num(df, print_ = print_)
    df.drop(df[df["div_amount_num"] == 0].index, inplace=True)
    df.drop("dividend_amount", axis = 1, inplace = True)
    # Add dividend direction and dividend change
    df = feature_handler.gen_div_direction_and_change(df, print_ = print_)
    print("Created: div_direction, div_change")

    # Add abnormal return data
    df = feature_handler.create_abnormal_return(df, print_ = print_)
    print("Created: expected_t, ar_t, aar_t, aar_t%")
    df.reset_index(inplace = True, drop = True)
    print("")
    if run_speed <= 1: df.to_csv("data_with_ar.csv", index=False)
##########################Step 2
def remove_outliers(df, range_min, range_max, y_col = 'aar_5', print_ = True):
    df_no_outliers = (df[df[y_col] < range_max][df[y_col] > range_min]).reset_index(drop = True)
    if print_: print("Removed {} outliers".format(df.shape[0] - df_no_outliers.shape[0]))
    return df_no_outliers

##########################Step 3

##########################Step 4
def remove_2008_2009(df, print_ = True):
    samples_before = df.shape[0]
    df = (df[df['year']!= 2008][df['year']!=2009]).reset_index(drop = True)
    samples_after = df.shape[0]
    if print_: print("Removed {} samples from years 2008 - 2009".format(samples_before-samples_after))
    return df
##########################Step 4.5
def gen_AsyScore(data, drop_fin_ratios = True, add_direction_asy=True):
    data = data.copy(deep = True)
    helper_cols = []
    hightech_sectors = set(["Technology"])
    regulated_sectors = set(["Energy", "Finance"])
    
    #High Tec Increase assymetry
    helper_cols.append(data['sector'].apply(lambda x: 1 if x in hightech_sectors else 0))
    
    #Regulation decrease assymety
    helper_cols.append(data['sector'].apply(lambda x: -1 if x in regulated_sectors else 0))
    
    fin_means = pd.read_csv("fin_ratio_means_by_year.csv", index_col='year')  #get means of financial ratios per year from a file
    
    #ratios we want to compare to the year's mean. 1 means that high value -> high asy, -1 the opposite. 
    relevant_fin_ratios = [('MV Debt Ratio', 1),('BV Debt Ratio',1),  ('Std Deviation In Prices', 1)]
    for ratio, effect in relevant_fin_ratios:
        def tag(i):
            i = int(i)
            year = min(data.loc[i, 'year'], 2018)
            mean = fin_means.loc[year, ratio]
            if data.loc[i, ratio] > mean: result = 1 * effect
            else: result = -1 * effect
            return result #We want it to be non-negative
        series = pd.Series(data.index).apply(tag)
        helper_cols.append(series)
    total = pd.concat(helper_cols, axis = 1)
    AsyScore = total.mean(axis = 1)
    data['AsyScore'] = AsyScore + 1
    if add_direction_asy:
        data['Asy_direction'] = data['AsyScore'] * data['div_direction']
    if drop_fin_ratios:
        data.drop(fin_ratios, axis = 1, inplace = True)
    data.reset_index(drop = True, inplace = True)
    return data
    

##########################Step 5

def plot_ws_by_div_dir(data, ws = ['aar_0%', 'aar_1%', 'aar_2%', 'aar_3%', 'aar_4%', 'aar_5%', 'aar_asy-1_5%']):
    alpha=0.01; line=True; dens=False
    data = feature_handler.create_asymmetric_window(data, start=-1, end=5)
    fig = plt.figure()
    #fig.tight_layout() 
    plt.subplots_adjust(right = 3)
    for i,window_name in enumerate(ws):
        plt.subplot(3, 3, i+1)
        visualize.hisograms_seperated(data, window_name, 'div_direction', alpha=alpha, dens=dens, line=line)
    plt.show()


##########################Step 6

##########################Step 7


def fit_random_forest(X, y, n_estimators, max_depth):
    """
    Train Random forest using X,y with the parameters given.
    if run_speed == 3: load already trained one.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth)
    rf_model.fit(X, y)
    return rf_model


def rf_cv(X_train, y_delta_train, ns, ds, run_speed, idx = 1):
    if run_speed == 1:
        gcv = regression_models.rf_cv(X_train, y_delta_train, ns, ds)
        return gcv.cv_results_
    else:
        with open("notebook_files/step7_gcv_{}".format(idx), 'rb') as file:
            gcv = pickle.load(file)
    return gcv
##########################Step 8

##########################Step 9

