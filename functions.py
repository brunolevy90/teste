
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


factors, retornos = simulate_rets(k=3, p=100,n=1000)



sr_factors = factors.mean()*np.sqrt(252)/np.std(factors)
sr_rets = retornos.mean()*np.sqrt(252)/np.std(retornos)







