#
# Autor: Ing. Matias Selser
#


##
import numpy as np
import pandas as pd
import sklearn.linear_model as sk_lm
from sklearn.metrics import mean_squared_error
from mlfin import utils

##
class Portfolio():

    def __init__(self, daily_ret=[]):
        self.assets = daily_ret

    def get_balanced_portfolio(self,daily_factors=[]):

        retTuple = ([],[])
        min_MSE = 1

        self.assets = self.assets.iloc[:,1:]

        rf = daily_factors.iloc[:,-1:]
        rf = rf['RF']
        rf = rf.reset_index(drop=True)

        daily_factors = daily_factors.iloc[:,1:-1]
        X = daily_factors
        X = X.reset_index(drop=True)

        alloc = utils.get_allocations(self.assets.shape[1])
        for weights in alloc:
            Y = self.assets.dot(weights)
            Y = Y.reset_index(drop=True)

            Y_P = 100*Y.subtract(rf)
            regression = sk_lm.LassoCV(cv=5).fit(X, Y_P)
            MSE = mean_squared_error(np.ones(X.shape[1])/X.shape[1], regression.coef_)
            if MSE < min_MSE:
                retTuple = (weights,regression.coef_)
                min_MSE = MSE

        return retTuple

if __name__ == '__main__':
    df_etfs = pd.read_csv('selected_etfs.csv')
    df_etfs = df_etfs.dropna()

    date = df_etfs.iloc[:,0]
    df_etfs = df_etfs.pct_change()
    df_etfs.iloc[:,0] = date

    df_FF = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv')

    #   Elijo solo 'DATE','BOND','SUSA','DNL','XLF','XSLV'
    df_etfs = df_etfs[['DATE','BOND','SUSA','DNL','XLF','XSLV']]

    #   Elijo solo año 2017
    df_FF = df_FF[(df_FF['Date'] >= 20170000) & (df_FF['Date'] < 20180000)]
    df_etfs = df_etfs[(df_etfs['DATE'] >= 20170000) & (df_etfs['DATE'] < 20180000)]


    myPort = Portfolio(df_etfs)
    weigths, exposure = myPort.get_balanced_portfolio(df_FF)

    print("--- Portfolio Balanceado ---")
    print("Weights  : ", end = '')
    print(["{:0.2%}".format(x) for x in weigths])
    print("Exposures  : ", end = '')
    print(["{:0.2%}".format(x) for x in exposure])