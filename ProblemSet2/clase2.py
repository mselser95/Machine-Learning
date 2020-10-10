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
        self.assetcount = daily_ret.shape[1]

    def get_balanced_portfolio(self,daily_factors=[]):
        retTuple = ([],[])
        min_MSE = 1


        df = daily_factors.join(self.assets)
        X = df.iloc[:,0:daily_factors.shape[1]-1]

        for weights in utils.get_allocations(self.assets.shape[1]):
            Y = df.iloc[:, daily_factors.shape[1]:] @ weights
            regression = sk_lm.LassoCV(cv=5).fit(X, Y)
            MSE = mean_squared_error(np.ones(X.shape[1])/X.shape[1], regression.coef_)
            if MSE < min_MSE:
                retTuple = (weights,regression.coef_)
                min_MSE = MSE

        return retTuple

if __name__ == '__main__':
    df_etfs = pd.read_csv('selected_etfs.csv',index_col=0,parse_dates=True).pct_change()
    df_FF = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv',index_col=0,parse_dates=True)/100

    myPort = Portfolio(df_etfs.loc['2017',['BOND','SUSA','DNL','XLF','XSLV']])
    weigths, exposure = myPort.get_balanced_portfolio(df_FF.loc['2017'])

    print("--- Portfolio Balanceado ---")
    print("Weights  : ", end = '')
    print(["{:0.2%}".format(x) for x in weigths])
    print("Exposures  : ", end = '')
    print(["{:0.2%}".format(x) for x in exposure])