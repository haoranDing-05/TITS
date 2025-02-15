import torch
import numpy as np
import pickle
from main import process_data
from NLT_main import likelihood_transformation
import os, sys
import matplotlib.pyplot as plt

file_path = r".\unsw-nb15\versions\1\UNSW-NB15_1.csv"

#记录data运行结果 加速代码运行 类似jupyter
if not os.path.isfile('data.pkl'):
    data=process_data(file_path)
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
else:
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)


#print(data.shape)

data=process_data(file_path)
result=likelihood_transformation(data)
print(result.shape)