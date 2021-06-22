
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
        random.seed(10)
        for t in range(len(retornos.index)):
            if t>0:
                temp = np.random.uniform(0, 0.3, size=1)*ret[t-1]+np.random.normal(loc=0,scale=0.01)
                ret.append(float(temp))

        retornos['Ret' + str(i)] = pd.Series(ret, index=retornos.index)

    return retornos


def naive_tsmom(rets, look_back = 52, vol_target=0.4):

    prices = (1+rets).cumprod().resample('W', loffset=pd.offsets.timedelta(days=-2)).last()
    ret_new = prices.pct_change(1)
    vols = pd.DataFrame(index=ret_new.index)
    df_tsmom = prices.pct_change(look_back)
    df_strat = pd.DataFrame(index=ret_new.index, columns=ret_new.columns)

    for i in ret_new.columns:
        vols[str(i)] = ret_new[str(i)].ewm(ignore_na=False,
                        adjust=True,
                        com=60,
                        min_periods=0).std(bias=False) * np.sqrt(52)


        for t in ret_new.index:
            if t > ret_new.index[look_back-1]:

                signal = np.sign(df_tsmom[str(i)].loc[t - timedelta(7)])

                if signal>=0:
                    df_strat[str(i)].loc[t] = (vol_target/vols[str(i)].loc[t - timedelta(7)])*signal*ret_new[str(i)].loc[t]

                else:
                    df_strat[str(i)].loc[t] = (vol_target / vols[str(i)].loc[t - timedelta(7)]) *(signal* ret_new[str(i)].loc[t])




    df_final = df_strat.mean(axis=1).dropna()

    return df_final



def logistic_tsmom(rets, look_back = 52, vol_target, train_size):

    prices = (1 + rets).cumprod().resample('W', loffset=pd.offsets.timedelta(days=-2)).last()
    ret_new = prices.pct_change(1).iloc[look_back+1:]
    ret_signals = np.sign(ret_new)
    ret_signals[ret_signals==-1] = 0
    vols = pd.DataFrame(index=ret_new.index)
    tsmom_preditores = prices.pct_change(look_back).shift().iloc[look_back+1:]
    df_strat = pd.DataFrame(index=ret_new.index, columns=ret_new.columns)
    accuracy = []

    for i in ret_new.columns:
        vols[str(i)] = ret_new[str(i)].ewm(ignore_na=False,
                        adjust=True,
                        com=60,
                        min_periods=0).std(bias=False) * np.sqrt(52)

        pred = []
        true_values = []
        for t in ret_new.index[train_size - 1:-1]:
            X_train = np.array(tsmom_preditores[str(i)].loc[:t]).reshape((-1,1))
            X_test = tsmom_preditores[str(i)].loc[t + timedelta(7)].reshape((-1,1))
            Y_train = np.array(np.array(ret_signals[str(i)].loc[:t]))

            logit = LogisticRegression(solver='liblinear')
            logit.fit(X_train, Y_train)
            signal_pred = logit.predict(X_test)[0]
            pred.append(signal_pred)
            true_values.append(ret_signals[str(i)].loc[t + timedelta(7)])

            # Portfolio construction
            if signal_pred ==1:
                df_strat[str(i)].loc[t] = (vol_target / vols[str(i)].loc[t - timedelta(7)])*ret_new[str(i)].loc[t]

            else:
                df_strat[str(i)].loc[t] = (vol_target / vols[str(i)].loc[t - timedelta(7)])*(-1*ret_new[str(i)].loc[t])



        accuracy.append(accuracy_score(true_values, pred))

    df_final = df_strat.mean(axis=1).dropna()

    return df_final, accuracy
