import numpy as np
import pandas as pd
import functions as fc


factors, retornos = fc.simulate_rets(k=3, p=100,n=1000)

sr_factors = factors.mean()*np.sqrt(252)/np.std(factors)
sr_rets = retornos.mean()*np.sqrt(252)/np.std(retornos)

