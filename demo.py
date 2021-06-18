
import numpy as np
import pandas as pd

retornos = pd.DataFrame()

for i in range(10):
    retornos['Ret'+ str(i) ] =  np.random.standard_normal(1000)




