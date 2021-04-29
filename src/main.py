import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

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
    mae = mean_absolute_error
    rmse = mean_squared_error
    cor = pearsonr
    r = r2_score
    df = pd.read_csv("spb_data.csv", delimiter=',')
    print(len(df))

    print(df['date'])
    sns.set_theme()
    # x = [01.01, 01.02, 01.03, 01.04]
    stand_x = np.array([x for x in range(1, len(df) + 1)])
    axe = sns.scatterplot(data=df, x=stand_x, y='act_case')
    plt.xlabel('Days from the first of January 2021')
    plt.ylabel('Active cases')
    plt.tight_layout()
    # x = np.array(df['date']).reshape(-1, 1)
    y = np.array(df['act_case'])  # .reshape(-1, 1)
    x = stand_x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.21, random_state=15)
    # .reshape(-1, 1)
    y_train = y_train.reshape(len(y_train), )
    y_test = y_test.reshape(len(y_test), )
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_dist_lin = mean_absolute_error(y_test, lm.predict(X_test))
    print("lin", y_dist_lin)

    # SVR

    eps = 6900  # 5440  # 10368
    svr = LinearSVR(epsilon=eps, C=200000, fit_intercept=True, max_iter=10000000)  # 40000 #69000
    svr.fit(X_train, y_train)
    plt.plot(X_test, svr.predict(X_test) + eps, color='pink', linewidth=1.5, linestyle='dotted')
    plt.plot(X_test, svr.predict(X_test) - eps, color='pink', linewidth=1.5, linestyle='dotted')
    sns.set_theme()
    sns.set_color_codes("dark")
    plt.plot(X_test, svr.predict(X_test), color='m', label='SVR')
    plt.show()

    sns.scatterplot(data=df, x=stand_x, y=np.array(df['act_case']))
    LR_res = lm.predict(X_test)
    SVR_res = svr.predict(X_test)
    print("LR:")
    print("MAE ", mae(y_test, LR_res))
    print("RMSE ", rmse(y_test, LR_res))
    # c = cor(y_test,LR_res)
    print("cor ", cor(y_test, LR_res)[0])
    print("R2 ", r(y_test, LR_res))

    print("SVR")
    print("MAE ", mae(y_test, SVR_res))
    print("RMSE ", rmse(y_test, SVR_res))
    print('cor ', cor(y_test, SVR_res)[0])
    print("R2 ", r(y_test, SVR_res))
    plt.plot(X_test, lm.predict(X_test), color='k', label='LN')
    plt.plot(X_test, svr.predict(X_test), color='m', label='SVR')
    plt.xlabel('Days from the first of January 2021')
    plt.ylabel('Active cases')
    plt.tight_layout()
    plt.legend()
    plt.show()
    y_dist_svr = mean_absolute_error(y_test, svr.predict(X_test))
    print("svr", y_dist_svr)
    test_mae_list = []
    perc_within_eps_list = []

    # grid = {
    #     'C': np.linspace(60000, 90000, num=20),
    #     'epsilon': np.linspace(1000, 21000, num=10)
    # }
    #
    # svr_gridsearch = LinearSVR(fit_intercept=True, max_iter=10000000)
    # # grid_svr = GridSearchCV(svr_gridsearch, grid, scoring='neg_mean_absolute_error', cv=5)
    # # grid_svr.fit(X_train, y_train)
    #
    # # def frac_within_eps(y_true, y_pred):
    # #     return np.sum(abs(y_true - y_pred) <= eps) / len(y_true)
    #
    # # my_scorer = make_scorer(frac_within_eps, greater_is_better=True)
    # grid_svr_eps = GridSearchCV(svr_gridsearch, grid, scoring='neg_mean_absolute_error', cv=5)
    #
    # grid_svr_eps.fit(X_train, y_train)
    #
    # best_grid_svr_eps = grid_svr_eps.best_estimator_
    # best_grid_svr_eps.fit(X_train, y_train)
    # print("C: {}".format(best_grid_svr_eps.C))
    # print("Epsilon: {}".format(best_grid_svr_eps.epsilon))
    # m = mean_absolute_error(y_test, best_grid_svr_eps.predict(X_test))

    epss = np.linspace(1000, 10000, num=30)
    list_mae = []
    list_rmse = []
    list_cor = []
    list_r = []
    all_list = [[list_mae, "MAE"], [list_rmse, "RMSE"], [list_cor, "COR"], [list_r, "R2"]]
    for eps in epss:
        svr = LinearSVR(epsilon=eps, C=200000, fit_intercept=True, max_iter=10000000)
        svr.fit(X_train, y_train)
        pred = svr.predict(X_test)
        list_mae.append(mae(y_test, pred))
        list_rmse.append(rmse(y_test, pred))
        list_cor.append(cor(y_test, pred)[0])
        list_r.append(r(y_test, pred))
    print(list_cor)

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    pred = lm.predict(X_test)
    LR_list = [mae(y_test, pred), rmse(y_test, pred), cor(y_test, pred)[0], r(y_test, pred)]
    for name_lis in zip(all_list, LR_list):
        SVR, LR = name_lis
        lis, name = SVR
        plt.plot(epss, lis, label="SVR")
        plt.ylabel(name)
        plt.xlabel("eps")
        plt.title("SVR")
        plt.plot(epss, [LR] *len(epss) , label="LR")
        plt.legend()
        plt.show()

    # list_mae = []
    # list_rmse = []
    # list_cor = []
    # list_r = []
    # all_list = [[list_mae, "MAE"], [list_rmse, "RMSE"], [list_cor, "COR"], [list_r, "R2"]]
    # for eps in epss:
    #     lm = LinearRegression()
    #     lm.fit(X_train, y_train)
    #     pred = lm.predict(X_test)
    #     list_mae.append(mae(y_test, pred))
    #     list_rmse.append(rmse(y_test, pred))
    #     list_cor.append(cor(y_test, pred)[0])
    #     list_r.append(r(y_test, pred))
    #
    # for name_lis in all_list:
    #     lis, name = name_lis
    #     plt.plot(epss, lis)
    #     plt.ylabel(name)
    #     plt.xlabel("eps")
    #     plt.title("LR")
    #     plt.show()

    # svr_results(y_test, X_test, best_grid_svr_eps)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
