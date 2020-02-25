from aggregate import aggregate
from aggregate import capm_params
from feature_handler import feature_handler
import zipfile
import json
import pandas as pd

from visualize import visualize

<<<<<<< HEAD
#General
=======
##########################General
>>>>>>> elad
def print_basic_stats(data, nans = True):
    r, c = data.shape
    nans_count = (data.isna().sum(axis = 1) > 0).sum()
    print("Data Stats:")
    print("\t#Samples: {}\n\t#Features: {}".format(r, c))
    if nans: print("\t#Samples with NaNs: {}".format(nans_count))
    return r, c, nans_count

def generate_bl_model_data(df, window_size = 5, y_col = "aar_5", drop_08_09 = False, print_ = True):
    """
    Generate data for the baseline model.
    Params:
    df: the data to generate from
    window_size: the window_size we wish to predict, either in 1,..,5 or a tuple. if a tuple the window is asymetric and y_col is ignored!
    y_col: the Y column
    drop_08_09: whether to drop 2008,2009 or not
    print_: print result details
    
    Return: DataFrame
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
    if isinstance(window_size, tuple):
        start, end = window_size
        df = feature_handler.create_asymmetric_window(df ,start, end)
        y_col = "aar_asy{}_{}%".format(str(start), str(end))
        start = abs(start)
    else: 
        start = window_size
    if start < 3:
        check = lambda c: sum(["t-{}".format(i) in c for i in range(start+3, 6)]) > 0 
        for c in [c for c in drop_cols_baseline if check(c)]: drop_cols_baseline.remove(c)

    if y_col in drop_cols_baseline: drop_cols_baseline.remove(y_col)
    drop_cols_baseline.remove('sector')
    baseline_models_data = df.drop(drop_cols_baseline, axis = 1)
    baseline_models_data = pd.get_dummies(data = baseline_models_data,\
                                          prefix='sector',
                                          columns = ['sector'],
                                          drop_first = True)
    baseline_models_data.reset_index(inplace = True, drop = True)
    if drop_08_09: 
        baseline_models_data = \
        (baseline_models_data[baseline_models_data['year']!= 2008][baseline_models_data['year']!=2009])\
        .reset_index(drop = True)
    if print_: print("Baseline model features: {}".format(set(baseline_models_data.columns) - set([y_col])))
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
##########################Step 5

##########################Step 6

##########################Step 7

##########################Step 8

##########################Step 9

##########################

def main():
    # Step 1: data pre-process steps and initial feature extraction
    unzip_data()
    all_prices = json.load(open("all_prices.json", "r"))
    df = do_aggregate_steps(all_prices)
    # df = pd.read_csv("aggregated_data.csv")

    # Scan data - Number of samples, NaN samples
    print(df.shape)
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    print(df.shape)
    rows_after = df.shape[0]
    print("Dropped {} rows with NaN values".format(rows_before-rows_after))

    # dividend_amount to integer
    df = feature_handler.create_div_amount_num(df)
    df.drop(df[df["div_amount_num"] == 0].index, inplace=True)

    # Add dividend direction and dividend change
    df = feature_handler.gen_div_direction_and_change(df)

    # Add abnormal return data
    df = feature_handler.create_abnormal_return(df)
    df.to_csv("all_data.csv", index=False)
    # >>>> Finished data pre-process steps and initial feature extraction

    # Step 2: Data insights and visualization including drop outliers
    # Basic stats, dividends per year, abnormal return hist given div direction

    # Step 3: Base line model

    # Step 4: Analysis of error + conclusions (years 2008-2009)

    # Step 5: Sensitivity analysis (size of window)

    # Step 6: Run new model (Regression) (naive + feature handler)

    # Step 7: Run different Regression models and compare

    # Step 8: Discrete Models

    # Step 9: Cross Validation, results and conclusions


if __name__ == "__main__":
    # main()
    df = pd.read_csv("all_data.csv")
    df.drop(df[df["aar_5"] > 10].index, inplace=True)
    df.drop(df[df["aar_5"] < -10].index, inplace=True)
    df = feature_handler.create_asymmetric_window(df, -1, 5)
    visualize.window_analysis(df, "ar")
    visualize.window_analysis(df, "aar")
    visualize.window_analysis(df, "aar%")
    visualize.window_analysis(df, "asy")


