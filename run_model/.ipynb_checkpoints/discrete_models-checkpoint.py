import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from IPython.display import display_html

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



def knn_choose_K(X, y, ks):
    """
    CV for choose k in kNN classifier.
    return: GridSearchCV
    """
    grid = {'n_neighbors': ks}
    model = KNeighborsClassifier()
    gcv = GridSearchCV(estimator=model, cv=2, param_grid = grid, verbose=0, n_jobs = -1)
    gcv.fit(X , y)
    plt.plot(ks, gcv.cv_results_['mean_test_score'], marker='o', color='b', label='Mean CV score', linestyle='solid')
    plt.xlabel('K', fontsize=16)
    plt.title("Grid Search Scores", fontsize=20, fontweight='bold')
    plt.ylabel('CV Average Score (Accuracy)', fontsize=16)

    plt.show()
    return gcv


def run_knn(X_train, X_test, y_train, y_test, K):
    """
    Run kNN clasiffier.
    Print it's accuracy on the test set and return the model.
    """
    model = KNeighborsClassifier(K)
    model.fit(X_train, y_train)
    print("Test Accuracy: {}".format(model.score(X_test, y_test)))
    return model



def knn_tables(model,X_test, y_test):
    
    d = {'pred_bucket': model.predict(X_test), 'bucket': y_test}
    df = pd.DataFrame(data=d)

    table1 = pd.pivot_table(df, index=['bucket'],
                           columns=['pred_bucket'], aggfunc=pd.np.ma.count, fill_value=0, margins=True)
    pd.options.display.float_format = '{:.2f}'.format
    table2 = table1.div(table1.sum(axis=1), axis=0).multiply(200)
    print("Absolute values:" + "\t"*3 + "Proportion values:")
    print("")
    table1.style.apply(style_diag, axis=None)
    table2.style.apply(style_diag, axis=None)
    display_side_by_side(table1,table2)

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.style.apply(highlight_diags, axis = None).render()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def highlight_diags(data):
    attr1 = 'background-color: lightgreen'
    attr2 = ''

    df_style = data.replace(data, '')
    np.fill_diagonal(df_style.values, attr1)
    np.fill_diagonal(np.flipud(df_style), attr2) 
    return df_style

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
