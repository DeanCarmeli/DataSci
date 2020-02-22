import json
from _datetime import datetime, timedelta
import pandas_datareader as pd
import os

DIV_DATA = json.load(open(os.path.join(os.path.dirname(__file__), "all_dividends.json"), "r"))
SP500 = '^GSPC'
DATA_MERGED = os.path.join(os.path.dirname(__file__), 'all_prices.json')
FILE_NAME_PRICES_PRE = os.path.join(os.path.dirname(__file__), "prices_pre_div.json")
FILE_NAME_PRICES_POST = os.path.join(os.path.dirname(__file__), "dividend_and_prices.json")


def _parse_date(string_date):
    parsed_date = string_date.split('/')
    month = int(parsed_date[0])
    day = int(parsed_date[1])
    year = int(parsed_date[2])
    date = datetime(year, month, day)
    return date


def _date_to_str(date):
    return str(date.month) + '/' + str(date.day) + '/' + str(date.year)


def _parse_stock_data(stock_data):
    parsed_data = {}
    close_prices = stock_data['Close']
    for date, price in close_prices.items():
        parsed_data[_date_to_str(date)] = {"close_price": price, "volume": stock_data['Volume'][date]}
    return parsed_data


def _add_stock_info_to_div_info(start, end, file_name):
    """
    Creates a new json file named DIV_AND_PRICES_DATA.
    Adds to each dividend event the stock information and S&P
    information in a time window around the announcement date.
    :param start: Days before the announcement date.
    :param end: Days after the announcement date.
    :param file_name: The name of the file to save the results
    :return: None
    """
    errors = 0
    for company in DIV_DATA:
        print("Getting info fo company {}".format(company["company"]))
        if len(company["dividend"]) > 0 and company["dividend"][0] != "":
            adj_div_data = []
            for dividend in company["dividend"]:
                try:
                    div_as_list = dividend.split()
                    div_date = _parse_date(div_as_list[3])
                    div_amount = div_as_list[2]
                except IndexError:
                    print("Index error for company {}. dividend data = {}".format(company['company'], company['dividend']))
                    errors += 1
                    continue
                start_date = div_date + timedelta(days=start)
                end_date = div_date + timedelta(days=end)
                try:
                    stock_data = pd.DataReader(name=company["symbol"],
                                               data_source='yahoo',
                                               start=start_date,
                                               end=end_date).to_dict()
                    sp_500_data = pd.DataReader(name=SP500,
                                                data_source='yahoo',
                                                start=start_date,
                                                end=end_date).to_dict()

                    adj_div_data.append({"announcement_date": _date_to_str(div_date),
                                         "dividend_size": div_amount,
                                         "stock_10_days_window": _parse_stock_data(stock_data),
                                         "sp_500_10_days": _parse_stock_data(sp_500_data)})
                except Exception:
                    print("Exception. check company={} , symbol={}".format(company['company'], company['symbol']))
                    errors += 1
                    continue
            company["dividend"] = adj_div_data
    with open(file_name, 'w') as fp:
        json.dump(DIV_DATA, fp)
    print("Finished. Errors found: {}".format(errors))


def _create_pre_and_post_prices_files():
    _add_stock_info_to_div_info(-5, 5, "dividend_and_prices.json")
    _add_stock_info_to_div_info(-30, -5, "prices_pre_div.json")


def _get_post_prices(dividend):
    stock = []
    sp = []
    for date in dividend["stock_10_days_window"].values():
        stock.append(date["close_price"])
    for date in dividend["sp_500_10_days"].values():
        sp.append(date["close_price"])
    return stock, sp


def _get_pre_prices(index, dividend, data_pre):
    stock = []
    sp = []
    # iterate over the dividend info in pre data to find the equivalent announcement date
    for div_info in data_pre[index]["dividend"]:
        if div_info["announcement_date"] == dividend["announcement_date"]:
            for date in div_info["stock_prior"].values():
                stock.append(date["close_price"])
            for date in div_info["sp_prior"].values():
                sp.append(date["close_price"])
        else:
            continue
        break
    return stock, sp


class FetchPrices:
    """
    ****************************************************************
    WARNING: FETCHING STOCK PRICES TAKES ABOUT A DAY
    ****************************************************************
    By creating an object of this class, 3 files would be created:
    1) Dividend info + prices during 30 to 5 days before the announcement
    2) Dividend info + prices during a window of 5 days before and after the announcement
    3) Merged data - all of the information
    """
    def __init__(self):
        self.DATA_PRE = None
        self.DATA_POST = None
        self.fetch_prices_and_div()

    def _merge_pre_and_post(self):
        res = []
        for i in range(0, len(self.DATA_POST)):
            if len(self.DATA_POST[i]["dividend"]) == 0 or self.DATA_POST[i]["dividend"][0] == "":
                continue
            else:
                company = {"company": self.DATA_POST[i]["company"],
                           "symbol": self.DATA_POST[i]["symbol"],
                           "dividend": []}
                dividends = self.DATA_POST[i]["dividend"]
                for div in dividends:
                    post_prices = _get_post_prices(div)
                    pre_prices = _get_pre_prices(i, div)
                    merged_data = {"announcement_date": div["announcement_date"],
                                   "dividend_size": div["dividend_size"],
                                   "pre_stock_prices": pre_prices[0],
                                   "pre_sp_prices": pre_prices[1],
                                   "post_stock_prices": post_prices[0],
                                   "post_sp_prices": post_prices[1],
                                   "post_stock_dict": div["stock_10_days_window"],
                                   "post_sp_dict": div["sp_500_10_days"]}
                    company["dividend"].append(merged_data)
            res.append(company)
        with open(DATA_MERGED, 'w') as f:
            json.dump(res, f)

    def fetch_prices_and_div(self):
        _create_pre_and_post_prices_files()
        self.DATA_PRE = json.load(open(FILE_NAME_PRICES_PRE, "r"))
        self.DATA_POST = json.load(open(FILE_NAME_PRICES_POST, "r"))
        self._merge_pre_and_post()

