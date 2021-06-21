import numpy as np
import pandas as pd
import functions as fc



#factors, retornos = fc.simulate_rets(k=3, p=100,n=1000)
#sr_factors = factors.mean()*np.sqrt(252)/np.std(factors)

p = 100
n=1000
retornos = fc.simulate_rets_mom(p=p,n=n)
sr_rets = retornos.mean()*np.sqrt(252)/np.std(retornos)

ret_tsmom = fc.naive_tsmom(retornos, look_back = 252, vol_target=0.4)

sr_tsmom = ret_tsmom.mean()*np.sqrt(252)/np.std(ret_tsmom)