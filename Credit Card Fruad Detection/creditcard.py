''' Data Exploration'''

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")
print(df.head())