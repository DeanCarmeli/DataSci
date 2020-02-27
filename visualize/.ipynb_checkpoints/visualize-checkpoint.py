import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


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

##############################################################################################
def number_of_samples(data, category="year", f7=False, dens=False):
    """
will present the number of samples of each category in a bar plot;
category must be a valid column name; 
defualt: year
for the total number of samples look for the title
    """
    d = dict((key,0) for key in sorted(list(data[category].unique())))
    for key in d.keys():
        s = data.apply(lambda x: True if x[category]==key else False, axis=1)
        d[key] = len(s[s==True].index)
    y_pos = np.arange(len(d.keys()))
    bar_list = plt.bar(y_pos, d.values(), color='deepskyblue', edgecolor='black')
    plt.xticks(y_pos, d.keys(), rotation=70)
    plt.ylabel("# of Samples")
    if f7:
        plt.title("|erros|>0.05 by {}".format(category))
        bar_list[10].set_color('r')
        bar_list[10].set_edgecolor('black')
        bar_list[11].set_color('r')
        bar_list[11].set_edgecolor('black')
    else:
        plt.title("{0} samples / {1}". format(str(sum(d.values())), category))
    #plt.show()


def hisograms_seperated(data,category, filter_cat, alpha=0, dens=False, line=False):
    distinct_cat = sorted(list(data[filter_cat].unique()))
    col = ['red','blue','green']
    avgs = []
    for x in range(len(distinct_cat)):
        filter = data[filter_cat]==distinct_cat[x]
        filtered_data = data.where(filter)
        filtered_data = filtered_data.dropna()
        to_draw = filtered_data[category]
        to_draw = to_draw[to_draw.between(to_draw.quantile(alpha), to_draw.quantile(1-alpha))]
        plt.hist(to_draw, label=distinct_cat[x], bins=200, histtype='step', density=dens, color=col[x])
        if line:
            plt.axvline(to_draw.mean(), linestyle='solid', linewidth=1, color=col[x])
            avgs.append(to_draw.mean())
    if filter_cat=="div_direction":
        leg2 = plt.legend(["mean: "+ str(x*100)[:5] + "%" for x in avgs], loc='upper left')
        plt.legend(['down','flat','up'], loc='upper right')
        plt.gca().add_artist(leg2)
    else:
        plt.legend(loc='upper right')
    plt.xlim(min(to_draw), max(to_draw))
    plt.title("hist of {0} for each {1}".format(category, filter_cat))
    #plt.show()


def histogram(data, category, alpha=0, dens=False):
    to_draw = data[category]
    to_draw = to_draw[to_draw.between(to_draw.quantile(alpha), to_draw.quantile(1-alpha))]
    plt.hist(to_draw, color='deepskyblue', edgecolor='black', alpha=0.8, density=dens)
    plt.title("histogram of {}".format(category))
    plt.xlim(min(to_draw), max(to_draw))
    print (min(to_draw), max(to_draw))
    plt.show()


def histogram_for_outliers(data, category, lower_bound, upper_bound, jump):
    list_jump = np.arange(lower_bound, upper_bound+jump, jump)
    for i in range(len(list_jump)):
        if list_jump[i]> 0-(10e-15) and list_jump[i]<0+(10e-15):
            list_jump[i]=0
    binss = np.append([min(data[category])],list_jump)
    binss = np.append(binss, max(data[category]))
    x = np.histogram(data[category],binss)
    bars_list = plt.bar(np.arange(len(x[0])), x[0], color="deepskyblue", edgecolor='black')
    ticks=[]
    for i in range(1, len(binss)-1, 1):
        c = len(str(jump))+1
        nom= str(binss[i])[:c]
        if nom[-1]=='0':
            nom = nom[:-1]
        if nom[-2]=='9':
            nom = nom[:-2]
            nom = str(nom[:-1]) +str((int(nom[-1])+1)%10)
        if nom[-1]=='9':
            nom = nom[:-1]
            nom = str(nom[:-1]) + str((int(nom[-1])+1)%10)
        ticks.append(nom)
    ticks = ["["+str(ticks[i])+", "+str(ticks[i+1])+"]" for i in range(0, len(ticks)-1, 1)]
    ticks = ["[<"+str(lower_bound)+"]"] + ticks + ["[>"+str(upper_bound)+"]"]
    plt.xticks(np.arange(len(x[0])), ticks, rotation=90)
    bars_list[0].set_color("r")
    bars_list[-1].set_color("r")
    bars_list[0].set_edgecolor("black")
    bars_list[-1].set_edgecolor("black")
    plt.title("distirbution of the errors")
    #plt.show()



## the functions in order as in the list:
def hist_by_col(data, col):
    number_of_samples(data, col)

def func1(data):
    number_of_samples(data)

def func2(data):
    number_of_samples(data, "sector")

def func3(data, alpha=0.01, line=False, dens=False):
    hisograms_seperated(data, "aar_5%", 'div_direction', alpha=alpha, dens=dens, line=line)

def func4(data, alpha=0.001, dens=True):
    histogram(data,"aar_5", alpha=alpha, dens=dens)

def func5(data):
    number_of_samples(data, "div_direction")

def func6(data, lower_bound=-0.05, upper_bound=0.05, jump=0.005):
    data['diff'] = data['aar_5%'] - data['pred']
    histogram_for_outliers(data, 'diff', lower_bound, upper_bound, jump)

def func7(data, lower_bound=-0.05, upper_bound=0.05):
    data['diff'] = data['aar_5%'] - data['pred']
    x  = data[(data['diff']<lower_bound) | (data['diff']>upper_bound)]
    #print(x)
    number_of_samples(x, f7=True)

#
# data = pd.read_csv(r"C:\Users\mcarmon\DataSci\all_data.csv")
# func1(data)
# func2(data)
# func3(data, line=True)
# func4(data)
# func5(data)
# data = pd.read_csv(r"C:\Users\mcarmon\DataSci\data_with_preds.csv")
# func6(data)
# func7(data)

##step2 wrappers
import seaborn as sns
def plot_time_related_hist(df, run_speed = 2):
    fig = plt.figure()
    plt.subplots_adjust(right = 2, wspace=0.2, hspace=0.2)
    plt.subplot(1, 3, 1)
    hist_by_col(df, "year")
    plt.subplot(1, 3, 2)
    hist_by_col(df, "month")
    plt.subplot(1, 3, 3)
    hist_by_col(df, "quarter")
    plt.show()
    
def plot_time_sector_change_hist(df):
    fig = plt.figure()
    plt.subplots_adjust(right = 2.1, wspace=0.2, hspace=0.2)
    plt.subplot(1, 3, 1)
    sns.distplot(df["div_amount_num"], hist=False)
    plt.title("Dividend amount density")
    plt.xlabel("Dividend amount")
    plt.subplot(1, 3, 2)
    hist_by_col(df, "div_direction")
    plt.subplot(1, 3, 3)
    sns.distplot(df["div_change"], hist = False)
    plt.title("Dividend change density")
    plt.xlabel("Dividend change")
    plt.show()
    
def plot_aar5_given_dir(data):
    alpha=0.001; line=True; dens=False
    hisograms_seperated(data, "aar_5", 'div_direction', alpha=alpha, dens=dens, line=line)
    
def plot_aar5p_given_dir(data):
    alpha=0.01; line=True; dens=False
    hisograms_seperated(data, "aar_5%", 'div_direction', alpha=alpha, dens=dens, line=line)
    
def plot_aar5_and_aar5p_given_dir(data):
    fig = plt.figure()
    plt.subplots_adjust(right = 2, wspace=0.2, hspace=0.2)
    plt.subplot(1, 2, 1)
    plot_aar5_given_dir(data)
    plt.subplot(1, 2, 2)
    plot_aar5p_given_dir(data)
    plt.show()

def plot_errors_dist(d , pred):
    d = d.copy(deep = True)
    d['pred'] = pred
    d['diff'] = d['aar_5%'] - pred
    fig = plt.figure()
    plt.subplots_adjust(right = 2, wspace=0.2, hspace=0.2)
    plt.subplot(1, 2, 1)
    sns.distplot(d['aar_5%']-d['pred'], hist = False)
    plt.title("Density estimation of model errors (aar_5% - prediction)")
    plt.subplot(1, 2, 2)
    lower_bound=-0.05; upper_bound=0.05; jump=0.005
    histogram_for_outliers(d, 'diff', lower_bound, upper_bound, jump)    
    plt.show()
    
def plot_error_by_year(data, pred, by = 'year'):
    d = data.copy(deep = True)
    d['pred'] = pred
    lower_bound=-0.05; upper_bound=0.05
    d['diff'] = d['aar_5%'] - d['pred']
    x  = d[(d['diff']<lower_bound) | (d['diff']>upper_bound)]
    number_of_samples(x, f7=True, category = by)
    plt.show()