from pathlib import Path

from dolfin import *
import numpy as np
import pandas as pd

dt = 0.001
num_steps = 1 / 0.001

Ts = [0.2]
Dus = [1/10]# [1/10, 1/100, 1/1000]
chis = [10]
vars = ['m', 'f'] 

# vars = ['u', 'v']
nodes = 10201 #40401 #2601 
chi = 10

data_all = pd.read_csv(folder_name + "/" + file_name + ".csv", header=None)#, nrows=1, usecols=range(start_col, end_col))


for T_extract in Ts:
    for chi in chis: #for Du in Dus: 
        for varr in vars:
            idx = int(T_extract/dt)
            start_col = idx * nodes
            end_col = (idx+1) * nodes
            
            # folder_name = f"Schnak_adv_Du{Du}_timedep_vel_coarse"
            # file_name = "Schnak_adv_" + varr
            folder_name = f"chtx_chi{chi}_simplfeathers_dx0.1"
            file_name = "/chtx_" + varr
            
            # Load only the first row and select specific columns
            data = pd.read_csv(folder_name + "/" + file_name + ".csv", header=None, nrows=1, usecols=range(start_col, end_col))
            
            # Convert to a NumPy array
            extracted_array = data.values.flatten()
            
            # Save the extracted array to a new CSV file
            np.savetxt(folder_name + "/" + file_name + f"T{T_extract:03}.csv", extracted_array, delimiter=",")


