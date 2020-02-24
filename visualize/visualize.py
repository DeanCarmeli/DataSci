import pandas as pd
import matplotlib.pyplot as plt


def color_neg_pos(val):
    if val < 0:
        color = 'red'
    elif val > 0:
        color = 'green'
    else:
        color = 'black'
    return "color: {}".format(color)


# HOW TO STYLE??? https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
def window_analysis(df, col_type, start=-1, end=5):
    """
    Calculates the average col_type over each column in the data set
    that is of that type given a dividend direction.
    As an example : col type = aar --> relevant columns are:
        [aar_0, ..., aar_5]. for each of those, calculate the average
        over all the samples were di_direction = -1 (0, 1).
    :param df: DataFrame
    :param col_type: ar , aar, aar%, asy
    :param start: Only when col_type == asy. start of asymmetric window
    :param end: Only when col_type == asy. end of asymmetric window
    :return: DataFrame or None in case of wrong col_type
    """
    data = {}
    if col_type == "ar":
        relevant_cols = ["ar_t{}".format(x) for x in range(-5, 6)]
    elif col_type == "aar":
        relevant_cols = ["aar_{}".format(x) for x in range(0, 6)]
    elif col_type == "aar%" or col_type == "asy":
        relevant_cols = ["aar_{}%".format(x) for x in range(0, 6)]
        if col_type == "asy":
            relevant_cols.append("aar_asy{}_{}%".format(str(start), str(end)))
    else:
        print("Wrong type of col_type provided. Can be only ar, aar, aar% or asy")
        return None

    for col in relevant_cols:
        res = []
        for i in range(-1, 2):
            average = float("{0:.2f}".format(df[df["div_direction"] == i][col].mean()*100))
            res.append(average)
        data[col] = res
    win_df = pd.DataFrame(data, index=["down", "flat", "up"])
    df.style.applymap(color_neg_pos, subset=relevant_cols)
    print(win_df)
    return win_df
