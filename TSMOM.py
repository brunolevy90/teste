import numpy as np
import pandas as pd
import functions as fc
from datetime import datetime
import matplotlib.pyplot as plt

#factors, retornos = fc.simulate_rets(k=3, p=100,n=1000)
#sr_factors = factors.mean()*np.sqrt(252)/np.std(factors)

start = "1/1/2010"
end = datetime.date(datetime.now())

retornos = fc.simulate_rets_mom(p=100,start=start, end=end)
sr_rets = retornos.mean()*np.sqrt(252)/np.std(retornos)

ret_tsmom = fc.naive_tsmom(retornos, look_back = 52, vol_target=0.4)

sr_tsmom = ret_tsmom.mean()*np.sqrt(252)/np.std(ret_tsmom)

(1+ret_tsmom).cumprod().plot()


ret_tsmom_logit, accuracy = logistic_tsmom(retornos, look_back = 52, vol_target=0.4, train_size=104)

