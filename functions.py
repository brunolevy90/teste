
import numpy as np
import pandas as pd


def simulate_rets(k=3, p=10,n=1000):
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






def naive_tsmom(rets, look_back = 252, vol_target=0.4):

    vols = pd.DataFrame(index=rets.index)
    df_tsmom = pd.DataFrame(index=rets.index)

    for i in rets.columns:
        vols[str(i)] = rets[str(i)].ewm(ignore_na=False,
                        adjust=True,
                        com=60,
                        min_periods=0).std(bias=False) * np.sqrt(252)

        for t in rets.index:
            if t >= look_back:
                df_tsmom[str(i)].iloc[t] =






    return




