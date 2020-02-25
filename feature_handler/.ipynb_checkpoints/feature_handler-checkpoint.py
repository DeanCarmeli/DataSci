import pandas as pd


def create_div_amount_num(df):
    """
    Detects invalid values of dividend_amount and converts string to number.
    Saves it as a new feature
    :param df: DataFrame
    :return: DataFrame with a column named "div_amount_num"
    """
    drop = 0
    print("################# Creating div_amount_num #################")
    # Find rows were dividend_amount doesn't start with $
    valid = df["dividend_amount"].str.startswith('$').sum()
    print("Found {} samples were dividend_amount did not start with $ - drop".format(df.shape[0]-valid))
    drop += df.shape[0]-valid
    df.drop(df[~df["dividend_amount"].str.startswith('$')].index, inplace=True)

    # Create a new column with the dividend amount as float
    df["div_amount_num"] = df["dividend_amount"].apply(lambda x: float(x.split("$")[1]))
    row_before = df.shape[0]

    # Drop dividens that equal to 0
    df.drop(df[df["div_amount_num"] == 0].index, inplace=True)
    print("Found {} divdends that were eqaul to 0 - drop".format(row_before - df.shape[0]))
    drop += row_before - df.shape[0]
    print(">>> Finished. Total dropped rows due to invalid dividend info: {}".format(drop))
    return df


def gen_div_direction_and_change(df):
    """
    Adds 2 features:
    div_direction = whether the dividend went up (1), down (-1)
    or flat (0) vs. last announcement.
    div_change = By how much the dividend change in %
    :param df: DataFrame
    :return: DataFrame
    """
    print("################# Creating div_direction and div_change ###")
    df.reset_index(inplace=True, drop=True)
    df["temp"] = df["company_name"] + "___" + df["div_amount_num"].apply(lambda x: str(x))
    direction = []
    change = []
    for index, value in df["temp"].iteritems():
        try:
            prev = df["temp"].iat[index + 1]
        except Exception:
            direction.append(1)
            change.append(1)
            continue
        curr_info = value.split("___")
        prev_info = prev.split("___")
        if curr_info[0] == prev_info[0]:
            change.append((float(curr_info[1]) - float(prev_info[1])) / float(prev_info[1]))
            if float(curr_info[1]) == float(prev_info[1]):
                direction.append(0)
                continue
            if float(curr_info[1]) < float(prev_info[1]):
                direction.append(-1)
                continue
        else:
            change.append(1)
        direction.append(1)
    df["div_direction"] = direction
    df["div_change"] = change
    df.drop(["temp"], axis=1, inplace=True)
    print(">>> Finished")
    return df


def create_abnormal_return(df):
    """
    Adds the following features:
    1. expected_t = the calculated expected price at t days from announcement date
                    expected_t = alpha + beta * sp_price_t
    2. ar_t = the abnormal return at day t from announcement date
                    ar_t = price_t - expected_t
    3. aar_x = the average abnormal return over a time window of x days before
               and after the announcement date:
                    aar_x = average(aar_-x, ...., aar_x)
    4. aar_x% = aar_x divided by the average of expected prices during
                the time window
                    aar_x% = aar_x/average(expected_t-x, ..., expected_tx)
    :param df: DataFrame
    :return: DataFrame with the extra features
    """
    days = [x for x in range(-5, 6)]
    expected_cols = ["expected_t{}".format(x) for x in days]
    ar_cols = ["ar_t{}".format(x) for x in days]
    aar_cols = ["aar_{}".format(x) for x in range(0, 6)]
    aar_percen_cols = ["aar_{}%".format(x) for x in range(0, 6)]
    temp_cols = ["temp_{}".format(x) for x in range(0, 6)]
    print("################# Creating abnormal return related features ###")
    print("\tfeature type = expected_t\t\t11 features")
    for i in range(len(expected_cols)):
        df[expected_cols[i]] = df["alpha"].add(df["beta"].mul(df["sp_price_t{}".format(str(days[i]))]))

    print("\tfeature type = ar_t\t\t11 features")
    for i in range(len(ar_cols)):
        df[ar_cols[i]] = df["price_t{}".format(str(days[i]))].sub(df[expected_cols[i]])

    print("\tfeature type = aar_t\t\t6 features")
    df["temp_0"] = df["ar_t0"]
    for i in range(1, len(temp_cols)):
        df[temp_cols[i]] = df["ar_t-{}".format(i)].add(df["temp_{}".format(i - 1)].add(df["ar_t{}".format(i)]))
    for i in range(len(aar_cols)):
        if i == 0:
            df[aar_cols[i]] = df[temp_cols[i]]
        else:
            df[aar_cols[i]] = df[temp_cols[i]].div(i)

    print("\tfeature type = aar_t%\t\t6 features")
    df["temp_0"] = df["expected_t0"]
    for i in range(1, len(temp_cols)):
        df[temp_cols[i]] = df["expected_t-{}".format(i)].add(
            df["temp_{}".format(i - 1)].add(df["expected_t{}".format(i)]))
    for i in range(len(aar_percen_cols)):
        if i == 0:
            df[aar_percen_cols[i]] = df["aar_{}".format(i)].div(df["temp_{}".format(i)])
        else:
            df[aar_percen_cols[i]] = df["aar_{}".format(i)].div(df["temp_{}".format(i)].div(i))
    df.drop(temp_cols, axis=1, inplace=True)
    print(">>> Finished. Created {} features".format(34))
    return df


def create_asymmetric_window(df, start=-1, end=5):
    """
    Calculates the average abnormal return normalized by the
    expected price over a time window of "start" days before the dividend
    announcement and "end" days after
    :param df: DataFrame
    :param start: int in range(-5, 0) - number of days before announcement
    :param end: int in range(1, 6) - number of days after announcement
    :return: DataFrame
    """
    ar_cols = ["ar_t{}".format(x) for x in range(start+1, end+1)]
    ex_cols = ["expected_t{}".format(x) for x in range(start+1, end+1)]

    # create a column of the average ar over the provided time window
    df["temp_ar"] = df["ar_t{}".format(str(start))]
    for ar in ar_cols:
        df["temp_ar"] = df["temp_ar"].add(df[ar])
    df["temp_ar"] = df["temp_ar"].apply(lambda x: x/(end+1-start))

    # create a column of the average expected over the provided time window
    df["temp_ex"] = df["expected_t{}".format(str(start))]
    for ex in ex_cols:
        df["temp_ex"] = df["temp_ex"].add(df[ex])
    df["temp_ex"] = df["temp_ex"].apply(lambda x: x/(end+1-start))

    # divide
    df["aar_asy{}_{}%".format(str(start), str(end))] = df["temp_ar"].div(df["temp_ex"])
    df.drop(["temp_ex", "temp_ar"], axis=1, inplace=True)

def gen_delta_precent_t(df, print_ = True):
    df['delta_%_t-5'] = (df['price_t-5'] - df['expected_t-5']) / df['expected_t-5']
    df['delta_%_t-4'] = (df['price_t-4'] - df['expected_t-4']) / df['expected_t-4']
    if print_: print("Created delta_%_t-4 and delta_%_t-5 features")
    return df
