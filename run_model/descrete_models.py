import pandas as pd
from sklearn import neural_network
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt

import numpy as np

big     = 0.025
small   = 0.006


def bucket_assignment(x):
    if x < -big:
        return -2
    elif x < -small:
        return -1
    elif x < small:
        return 0
    elif x < big:
        return 1
    else:
        return 2


def prep_data(Data):
    drop_cols = ['symbol', 'company_name', 'dividend_date', 'dividend_amount', 'sector', 'industry', 'month',
                 'price_t-5', 'vol_t-5', 'sp_price_t-5', 'sp_vol_t-5', 'expected_t-5',
                 'price_t-4', 'vol_t-4', 'sp_price_t-4', 'sp_vol_t-4', 'expected_t-4',
                 'price_t-3', 'vol_t-3', 'sp_price_t-3', 'sp_vol_t-3', 'expected_t-3',
                 'price_t-2', 'vol_t-2', 'sp_price_t-2', 'sp_vol_t-2', 'expected_t-2',
                 'price_t-1', 'vol_t-1', 'sp_price_t-1', 'sp_vol_t-1', 'expected_t-1',
                 'price_t0', 'vol_t0', 'sp_price_t0', 'sp_vol_t0', 'expected_t0',
                 'price_t1', 'vol_t1', 'sp_price_t1', 'sp_vol_t1', 'expected_t1',
                 'price_t2', 'vol_t2', 'sp_price_t2', 'sp_vol_t2', 'expected_t2',
                 'price_t3', 'vol_t3', 'sp_price_t3', 'sp_vol_t3', 'expected_t3',
                 'price_t4', 'vol_t4', 'sp_price_t4', 'sp_vol_t4', 'expected_t4',
                 'price_t5', 'vol_t5', 'sp_price_t5', 'sp_vol_t5', 'expected_t5',
                 'ar_t-5', 'ar_t-4', 'ar_t-3', 'ar_t-2', 'ar_t-1', 'ar_t0', 'ar_t1', 'ar_t2', 'ar_t3', 'ar_t4', 'ar_t5',
                 'aar_5', 'aar_4', 'aar_3', 'aar_2', 'aar_1', 'aar_0',
                 'aar_5%', 'aar_4%', 'aar_3%', 'aar_2%', 'aar_0%',
                 'alpha', 'beta', 'year', 'quarter', 'MV Debt Ratio', 'BV Debt Ratio',
                 'Effective Tax Rate', 'Std Deviation In Prices', 'EBITDA/Value',
                 'Fixed Assets/BV of Capital', 'Capital Spending/BV of Capital',
                 'div_amount_num', 'div_direction', 'div_change']
    Data["delta_t-4"] = (Data["price_t-4"] - Data["expected_t-4"]) / Data["expected_t-4"]
    Data.drop(drop_cols, axis=1, inplace=True)
    # print(Data.axes)
    # Data.drop(Data[Data["year"] == 2008].index, inplace=True)
    # Data.drop(Data[Data["year"] == 2009].index, inplace=True)
    # Data            = pd.get_dummies(Data, columns=['year', 'quarter'])
    # col_drop_multi  = ['quarter_1', 'year_1998']
    # Data.drop(col_drop_multi, axis=1, inplace=True)
    return Data


def run_K_Neighbors_Classifier(data):
    data            = prep_data(data)
    data['bucket']  = data['aar_1%'].apply(bucket_assignment)
    data.drop(['aar_1%'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test    = train_test_split(data.drop(['bucket'], axis =1),
                                                           data['bucket'], test_size=0.2)
    score_train = []
    score_test  = []
    r = [i for i in range(49, 50)]
    for i in r:
        resu = KNeighborsClassifier(i)
        resu.fit(X_train, y_train)
        predicted = resu.predict(data.drop(['bucket'], axis=1))
        score_train.append(resu.score(X_train, y_train))
        score_test.append(resu.score(X_test, y_test))
    plt.plot(r, score_train, marker='o', color='r', label='Train score', linestyle='solid')
    plt.plot(r, score_test,  marker='o', color='b', label='Test score',  linestyle='solid')
    plt.xlabel('K')
    plt.legend()
    # plt.show()
    d = {'pred_bucket': resu.predict(X_test), 'bucket': y_test, "Test - Distribution": y_test}
    df = pd.DataFrame(data=d)

    table = pd.pivot_table(df, index=['bucket'],
                           columns=['pred_bucket'], aggfunc=pd.np.ma.count, fill_value=0, margins=True)
    pd.options.display.float_format = '{:.2f}'.format
    table2 = table.div(table.sum(axis=1), axis=0).multiply(200)
    print("Absolute vales:\n", table, "\n\n")
    print("Percentage vales:\n", table2)

    return df