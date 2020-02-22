import os
import pandas as pd
from datetime import datetime, timedelta

META_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), "companylist.csv"))
COLUMNS = ['company_name', 'symbol', 'industry', 'sector',
           'dividend_date', 'dividend_amount', 'alpha', 'beta',
           'price_t-5', 'vol_t-5', 'sp_price_t-5', 'sp_vol_t-5',
           'price_t-4', 'vol_t-4', 'sp_price_t-4', 'sp_vol_t-4',
           'price_t-3', 'vol_t-3', 'sp_price_t-3', 'sp_vol_t-3',
           'price_t-2', 'vol_t-2', 'sp_price_t-2', 'sp_vol_t-2',
           'price_t-1', 'vol_t-1', 'sp_price_t-1', 'sp_vol_t-1',
           'price_t0', 'vol_t0', 'sp_price_t0', 'sp_vol_t0',
           'price_t1', 'vol_t1', 'sp_price_t1', 'sp_vol_t1',
           'price_t2', 'vol_t2', 'sp_price_t2', 'sp_vol_t2',
           'price_t3', 'vol_t3', 'sp_price_t3', 'sp_vol_t3',
           'price_t4', 'vol_t4', 'sp_price_t4', 'sp_vol_t4',
           'price_t5', 'vol_t5', 'sp_price_t5', 'sp_vol_t5']
FIN_RATIOS = pd.read_csv(os.path.join(os.path.dirname(__file__), "fin_ratio_sum.csv"))
RATIOS = ["MV Debt Ratio", "BV Debt Ratio", "Effective Tax Rate",
          "Std Deviation In Prices", "EBITDA/Value", "Fixed Assets/BV of Capital",
          "Capital Spending/BV of Capital"]


def _get_meta(company, col_name):
    df = META_DATA.query("Symbol == '{}'".format(company))
    if df.empty:
        return None
    else:
        return df.get(col_name).to_list()[0]


def _add_meta_data(div_data_with_capm):
    for company in div_data_with_capm:
        company["industry"] = _get_meta(company["symbol"], "Industry")
        company["sector"] = _get_meta(company["symbol"], "Sector")


def _parse_date(string_date):
    parsed_date = string_date.split('/')
    month = int(parsed_date[0])
    day = int(parsed_date[1])
    year = int(parsed_date[2])
    date = datetime(year, month, day)
    return date


def _date_to_str(date):
    return str(date.month) + '/' + str(date.day) + '/' + str(date.year)


def _order_stock_data(div_info):
    string_t0 = div_info["announcement_date"]
    date_t0 = _parse_date(string_t0)
    dict_string_date = {}
    for i in range(-5, 6):
        dict_string_date[i] = _date_to_str(date_t0 + timedelta(days=i))
    res = {}
    not_found = []
    for i in range(-5, 6):
        try:
            res[i] = {"stock_p": div_info["stock_10_days_window"][dict_string_date[i]]["close_price"],
                      "stock_v": div_info["stock_10_days_window"][dict_string_date[i]]["volume"],
                      "sp_p": div_info["sp_500_10_days"][dict_string_date[i]]["close_price"],
                      "sp_v": div_info["sp_500_10_days"][dict_string_date[i]]["volume"]}
        except KeyError:
            not_found.append(i)
    if len(not_found) > 0:
        for i in not_found:
            fixed = 0
            for j in range(i-1, -6, -1):
                try:
                    res[i] = {"stock_p": div_info["stock_10_days_window"][dict_string_date[j]]["close_price"],
                              "stock_v": div_info["stock_10_days_window"][dict_string_date[j]]["volume"],
                              "sp_p": div_info["sp_500_10_days"][dict_string_date[j]]["close_price"],
                              "sp_v": div_info["sp_500_10_days"][dict_string_date[j]]["volume"]}
                    fixed = 1
                    break
                except KeyError:
                    continue
            if fixed == 0:
                for j in range(i+1, 6):
                    try:
                        res[i] = {"stock_p": div_info["stock_10_days_window"][dict_string_date[j]]["close_price"],
                                  "stock_v": div_info["stock_10_days_window"][dict_string_date[j]]["volume"],
                                  "sp_p": div_info["sp_500_10_days"][dict_string_date[j]]["close_price"],
                                  "sp_v": div_info["sp_500_10_days"][dict_string_date[j]]["volume"]}
                        fixed = 1
                        break
                    except KeyError:
                        continue
            if fixed == 0:
                res[i] = {"stock_p": None, "stock_v": None, "sp_p": None, "sp_v": None}
    return res


def _extract_entry(company, div_index):
    ordered_dates = _order_stock_data(company["dividend"][div_index])
    entry = [company["company"], company["symbol"],
             company["industry"], company["sector"],
             company["dividend"][div_index]["announcement_date"],
             company["dividend"][div_index]["dividend_size"],
             company["dividend"][div_index]["alpha"],
             company["dividend"][div_index]["beta"]]
    for i in range(-5, 6):
        entry.append(ordered_dates[i]["stock_p"])
        entry.append(ordered_dates[i]["stock_v"])
        entry.append(ordered_dates[i]["sp_p"])
        entry.append(ordered_dates[i]["sp_v"])
    return entry


def fill_missing_meta_data(df):
    missing_meta_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "missing_meta.csv"))
    for index, row in missing_meta_data.iterrows():
        res = df.query("company_name == '{}'".format(row["company_name"]))
        for i, r in res.iterrows():
            df.loc[i, "industry"] = row["industry"]
            df.loc[i, "sector"] = row["sector"]
    return df


def aggregate_meta_data(div_data_with_capm):
    """
    Adds meta data for each company (sector, industry)
    and changes the hierarchy of the provided dictionary.
    :param div_data_with_capm:
    The output of fetch_data.capm_params.get_data_with_capm_params()
    :return: Returns a DataFrame object with columns=COLUMNS
    """
    _add_meta_data(div_data_with_capm)
    # Iterate over the json to create a list of entries
    entries = []
    for company in div_data_with_capm:
        num_of_div = len(company["dividend"])
        if num_of_div > 0 and company["dividend"][0] is not "":
            for i in range(0, num_of_div):
                entries.append(_extract_entry(company, i))

    # Create a dataFrame object and save it as csv file
    df = pd.DataFrame(data=entries, columns=COLUMNS)
    # df.to_csv("data_with_stock_return_30.12.csv", index=False)
    return df


def aggregate_fin_ratios(df):
    """
    WARNING: THIS METHOD TAKES ABOUT 5 MINUTES TO RUN
    Adds all samples 7 financial ratios based on sector.
    :param df: DataFrame of samples dated from 1998 - 2019
    :return: The DataFrame with the financial ratios
    """
    df["temp"] = df["sector"] + df["year"].apply(lambda x: str(x))
    FIN_RATIOS["temp"] = FIN_RATIOS["sector"] + FIN_RATIOS["year"].apply(lambda x: str(x+1))
    for ratio in RATIOS:
        df[ratio] = df["temp"].apply(lambda x: FIN_RATIOS[FIN_RATIOS["temp"] == x][ratio].values[0])
    df.drop(["temp"], axis=1, inplace=True)
    return df
