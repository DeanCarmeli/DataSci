import numpy as np
from sklearn.linear_model import LinearRegression


def _estimate_reg_params(stock_pre_div, sp_pre_div, all_data):
    x = np.array(sp_pre_div).reshape(-1, 1)
    y = np.array(stock_pre_div).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    all_data["alpha"] = reg.intercept_[0]
    all_data["beta"] = reg.coef_[0][0]


def get_data_with_capm_params(all_prices):
    """
    Calculates the CAPM parameters based based on data from 30 days
    before the announcement and 5 days before the announcement
    :param all_prices: Dictionary (result of reading the json file of
    the merged data created by FetchPrices class)
    :return: Dictionary
    """
    result = []
    errors = 0
    for entity in all_prices:
        info = {"company": entity["company"],
                "symbol": entity["symbol"],
                "dividend": []}
        for event in entity["dividend"]:
            stock_prediction_info = {"alpha": None,
                                     "beta": None,
                                     "stock_10_days_window": event["post_stock_dict"],
                                     "sp_500_10_days": event["post_sp_dict"]}
            try:
                _estimate_reg_params(event["pre_stock_prices"],
                                     event["pre_sp_prices"],
                                     stock_prediction_info)
            except Exception:
                errors += 1
                # print("Error: check company {}, dividend date {}".format(entity["company"], event["announcement_date"]))
            div_data = {"announcement_date": event["announcement_date"],
                        "dividend_size": event["dividend_size"],
                        "alpha": stock_prediction_info["alpha"],
                        "beta": stock_prediction_info["beta"],
                        "stock_10_days_window": stock_prediction_info["stock_10_days_window"],
                        "sp_500_10_days": stock_prediction_info["sp_500_10_days"]
                        }
            info["dividend"].append(div_data)
        result.append(info)
    # print("Found {} errors".format(errors))
    return result



