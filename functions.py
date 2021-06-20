
import numpy as np
import pandas as pd


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



def simulate_rets_mom(k=3, p=10,n=1000):
    # k: number of common factors
    # p: number of assets
    # n: sample size
    retornos = pd.DataFrame()

    for i in range(p):
        ret = []
        ret.append(0)
        for t in range(n):
            if t>0:
                temp = np.random.uniform(0, 0.3, size=1)*ret[t-1]+np.random.normal(loc=0,scale=0.01)
                ret.append(temp)

        retornos['Ret' + str(i)] = pd.Series(ret, index=range(100))






    for i in range(p):
        retornos['Ret' + str(i)] = factors.dot(np.random.uniform(0, 1.5, size=k)) + np.random.normal(loc=0,
                                                                                                     scale=0.01,
                                                                                                     size=n)

    return factors, retornos



def naive_tsmom(rets, look_back = 252, vol_target=0.4):

    prices = (1+rets).cumprod()
    vols = pd.DataFrame(index=rets.index)
    df_tsmom = prices.pct_change(252)
    df_strat = pd.DataFrame(index=rets.index, columns=rets.columns)

    for i in rets.columns:
        vols[str(i)] = rets[str(i)].ewm(ignore_na=False,
                        adjust=True,
                        com=60,
                        min_periods=0).std(bias=False) * np.sqrt(252)

        for t in rets.index:
            if t > look_back:

                if df_tsmom[str(i)].iloc[t-1]>=0:
                    df_strat[str(i)].iloc[t] = (vol_target/vols[str(i)].iloc[t])*rets[str(i)].iloc[t]

                else:
                    df_strat[str(i)].iloc[t] = (vol_target / vols[str(i)].iloc[t]) *(-1* rets[str(i)].iloc[t])


    df_final = df_strat.mean(axis=1).dropna()

    return df_final




