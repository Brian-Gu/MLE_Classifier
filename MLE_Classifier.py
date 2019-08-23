
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Data Preparation

df_dev = pd.read_csv('propublicaTrain.csv')
df_tst = pd.read_csv('propublicaTest.csv')

# Univariate Analysis

varList = df_dev.dtypes
var_num = varList.index[varList.values == 'int64']


def univariate_num_var(df, vars):
    """
    :param df:   Input DataFrame
    :param vars: Numerical Variable List
    :return:     Summary Statistic Results
    """
    res = pd.DataFrame()
    for var in vars:
        sum_stat = df[var].describe().transpose()
        sum_stat["Variable"] = var
        sum_stat["Miss#"] = len(df) - sum_stat["count"]
        sum_stat["Miss%"] = sum_stat["Miss#"] * 100 / len(df)
        res = res.append(sum_stat, ignore_index=True)
    order = ['Variable', 'count', 'Miss#', 'Miss%', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return res[order]


summary_num = univariate_num_var(df_dev, var_num)
print('Summary Statistics')


# 2. MLE Classifier

def mle_classifier(train_x, train_y):
    """
    :param train_x:  Train X Set
    :param train_y:  Train Y Set
    :return:         MLE Multivariate Gaussian Distribution Parameters
    """
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    params = {}
    classes = np.unique(train_y)
    for i in range(0, len(classes)):
        params[i] = {}
        indx = train_y[train_y == i].index
        params[i]['mu'] = np.mean(train_x.loc[indx, :])
        params[i]['sigma'] = np.cov(train_x.loc[indx, :], rowvar=False)
        params[i]['sigma'] += 0.0001*np.eye(train_x.shape[1])
        params[i]['prior'] = len(indx)/len(train_y)
    return params


def mle_predictor(params, test_x, test_y):
    """
    :param params:  MLE Multivariate Gaussian Distribution Parameters
    :param test_x:  Test X Set
    :param test_y:  Test Y Set
    :return:        Test Set with Prediction
    """
    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    x = test_x.copy()
    [n, d] = x.shape
    classes = []
    for c in params:
        mu = params[c]['mu']
        sigma = params[c]['sigma']
        prior = params[c]['prior']
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)
        for j in range(0, n):
            exp = np.exp(-0.5 * (x.loc[j, :] - mu).dot(inv_sigma).dot((x.loc[j, :] - mu).T))
            test_x.loc[j, str(c)] = prior * exp / np.sqrt(((2 * np.pi) ** d) * det_sigma)
        classes.append(str(c))
    test_x.loc[:, 'y_hat'] = [int(classes[k]) for k in np.argmax(test_x.loc[:, classes].values, axis=1)]
    test_x['y'] = test_y
    return test_x


def model_performance(scored, prd, act):
    """
    :param scored: Scored Set with Prediction
    :param prd:    Predicted Variable Name
    :param act:    Actual Variable Name
    :return:       Accuracy Rate
    """
    correct = scored[(scored[prd] == scored[act])].index
    rt = len(correct)/len(scored)
    return "{:.2%}".format(rt)


X, y = df_dev.drop(['two_year_recid'], axis=1), df_dev['two_year_recid']

X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.30, random_state=567)

pars = mle_classifier(X_dev, y_dev)

scored_dev = mle_predictor(pars, X_dev, y_dev)
scored_val = mle_predictor(pars, X_val, y_val)
scored_tst = mle_predictor(pars, df_tst.drop(['two_year_recid'], axis=1), df_tst['two_year_recid'])

rate_dev = model_performance(scored_dev, 'y_hat', 'y')
rate_val = model_performance(scored_val, 'y_hat', 'y')
rate_tst = model_performance(scored_tst, 'y_hat', 'y')

print('Accuracy Rate on Dev Sample: '  + str(rate_dev))
print('Accuracy Rate on Val Sample: '  + str(rate_val))
print('Accuracy Rate on Test Sample: ' + str(rate_tst))



