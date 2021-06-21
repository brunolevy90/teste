
import numpy as np
import pandas as pd
from datetime import datetime


def simulate_rets_factors(k=3, p=10,n=1000):
    # k: number of common factors
    # p: number of assets
    # n: sample size

    factors = pd.DataFrame()
    retornos = pd.DataFrame()

    for i in range(k):
        factors['f_' + str(i)] = np.random.normal(loc=0.05/252, scale=0.005, size=n)

    for i in range(p):
        retornos['Ret' + str(i)] = factors.dot(np.random.uniform(0, 1.5, size=k)) + np.random.normal(loc=0,
                                                                                                     scale=0.01,
                                                                                                     size=n)

    return factors, retornos




def simulate_rets_mom(p,start, end):
    # p: number of assets
    # n: sample size
    dates = pd.date_range(start=start, end=end ,freq='B')
    retornos = pd.DataFrame(index=dates)

    for i in range(p):
        ret = []
        ret.append(0)
        for t in range(len(retornos.index)):
            if t>0:
                temp = np.random.uniform(0, 0.3, size=1)*ret[t-1]+np.random.normal(loc=0,scale=0.01)
                ret.append(float(temp))

        retornos['Ret' + str(i)] = pd.Series(ret, index=retornos.index)

    return retornos


def naive_tsmom(rets, look_back = 252, vol_target=0.4):

    prices = (1+rets).cumprod()
    vols = pd.DataFrame(index=rets.index)
    df_tsmom = prices.pct_change(look_back)
    df_strat = pd.DataFrame(index=rets.index, columns=rets.columns)

    for i in rets.columns:
        vols[str(i)] = rets[str(i)].ewm(ignore_na=False,
                        adjust=True,
                        com=60,
                        min_periods=0).std(bias=False) * np.sqrt(252)

        for t in rets.index:
            if t > look_back:

                if t % 5 ==0:
                    signal = np.sign(df_tsmom[str(i)].iloc[t - 1])

                    if signal>=0:
                        df_strat[str(i)].iloc[t] = (vol_target/vols[str(i)].iloc[t])*signal*rets[str(i)].iloc[t]

                    else:
                        df_strat[str(i)].iloc[t] = (vol_target / vols[str(i)].iloc[t]) *(signal* rets[str(i)].iloc[t])
                else:



    df_final = df_strat.mean(axis=1).dropna()

    return df_final




