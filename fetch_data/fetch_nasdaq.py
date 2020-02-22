import os
from selenium import webdriver
import json

URL = "https://www.nasdaq.com/market-activity/stocks/{}/dividend-history"
SYMBOLS = json.load(open(os.path.join(os.path.dirname(__file__), "symbols"), "r"))
DRIVER = webdriver.Chrome()
DIV_DATA = os.path.join(os.path.dirname(__file__), "all_dividends.json")


def _get_dividend_data(url):
    """
    Given a url of historical dividends of a company form
    Nasdaq's website, returns a list of strings.
    Each string is a row of the table in the website with all dividend info.
    :param url: URL to a web page of historical dividend info of a Nasdaq's traded company.
    :return: A list of strings or an empty list in case of an exception.
    """
    DRIVER.get(url)
    try:
        body = DRIVER.find_element_by_tag_name("tbody")
        return body.text.split('\n')
    except Exception:
        return []


def fetch_nasdaq():
    """
    ****************************************************************
        WARNING: FETCHING DIVIDEND INFORMATION TAKES ABOUT A DAY
    ****************************************************************
    This method iterates over Nasdaq's website and gets all historical
    dividend data for all trading companies.
    It saves the info in a json format file named DIV_DATA
    :return: None.
    """
    all_dividends = []
    for company in SYMBOLS:
        url = URL.format(company["Symbol"].lower())
        dividends = _get_dividend_data(url)
        all_dividends.append({
            "company": company["Company Name"],
            "symbol": company["Symbol"],
            "dividend": dividends
        })

    with open(DIV_DATA, "w") as fp:
        json.dump(all_dividends, fp)
    DRIVER.close()
