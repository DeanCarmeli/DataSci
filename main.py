from aggregate import aggregate
from aggregate import capm_params
from feature_handler import feature_handler
import zipfile
import json
import pandas as pd

from visualize import visualize

#General
def print_basic_stats(data, nans = True):
    r, c = data.shape
    nans_count = (data.isna().sum(axis = 1) > 0).sum()
    print("Data Stats:")
    print("\t#Samples: {}\n\t#Features: {}".format(r, c))
    if nans: print("\t#Samples with NaNs: {}".format(nans_count))
    return r, c, nans_count



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
def remove_outliers(df, range_min, range_max, print_ = True):
    df_no_outliers = (df[df['aar_5'] < range_max][df['aar_5'] > range_min]).reset_index(drop = True)
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


