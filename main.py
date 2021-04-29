import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVR


def get_mnk_parameters(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def get_distance(y_model, y_regr):
    dist_y = sum([(y_model[i] - y_regr[i]) ** 2 for i in range(len(y_model))])
    return dist_y


def MNK(x, y):
    beta_0, beta_1 = get_mnk_parameters(x, y)
    print('beta_0 = ' + str(beta_0), 'beta_1 = ' + str(beta_1))
    y_new = [beta_0 + beta_1 * x_ for x_ in x]
    return y_new


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # sns.set_color_codes()

    df = pd.read_csv("spb_data.csv", delimiter=',')
    print(len(df))

    print(df['date'])
    x = np.array([x for x in range(1, len(df) + 1)])
    sns.set_theme()
    # x = [01.01, 01.02, 01.03, 01.04]
    axe = sns.scatterplot(data=df, x=x, y='act_case')
    plt.xlabel('Days from the first of January 2021')
    plt.ylabel('Active cases')
    plt.tight_layout()
    sns.set_theme()
    sns.set_color_codes("dark")
    # x = np.array(df['date']).reshape(-1, 1)
    y = np.array(df['act_case'])  # .reshape(-1, 1)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
    # .reshape(-1, 1)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_dist_lin = mean_absolute_error(y_test, lm.predict(X_test))
    print("lin",y_dist_lin)
    plt.plot(X_test, lm.predict(X_test), color='b', label='lin')
    # SVR

    y_train = y_train.reshape(len(y_train), )
    y_test = y_test.reshape(len(y_test), )

    eps = 3
    svr = LinearSVR(epsilon=eps, C=30000, fit_intercept=True)
    svr.fit(X_train, y_train)
    plt.plot(X_test, svr.predict(X_test), color='m', label ='svr')
    plt.legend()
    plt.show()
    y_dist_svr = mean_absolute_error(y_test, svr.predict(X_test))
    print("svr",y_dist_svr)
    test_mae_list = []
    perc_within_eps_list = []



    #OPt of C


    # c_space = np.linspace(20000, 30000)
    #
    # for c in c_space:
    #     varied_svr = LinearSVR(epsilon=eps, C=c, fit_intercept=True, max_iter=10000000)
    #
    #     varied_svr.fit(X_train, y_train)
    #
    #     test_mae = mean_absolute_error(y_test, varied_svr.predict(X_test))
    #     test_mae_list.append(test_mae)
    #
    #     perc_within_eps = 100 * np.sum(abs(y_test - varied_svr.predict(X_test)) <= eps) / len(y_test)
    #     perc_within_eps_list.append(perc_within_eps)
    #
    # fig, ax1 = plt.subplots(figsize=(12, 7))
    #
    # color = 'green'
    # ax1.set_xlabel('C')
    # ax1.set_ylabel('% within Epsilon', color=color)
    # ax1.scatter(c_space, perc_within_eps_list, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # color = 'blue'
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Test MAE', color=color)  # we already handled the x-label with ax1
    # ax2.scatter(c_space, test_mae_list, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # m = max(perc_within_eps_list)
    # inds = [i for i, j in enumerate(perc_within_eps_list) if j == m]
    # C = c_space[inds[0]]
    #
    # print("best C =", C)
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
