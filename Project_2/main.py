###
# Driver code that creates and runs different classifiers for the problem of
# classifying an author's age based on their blog posts.
# Written by Edmond Pan (841389)
###

# Import all of the required libraries for data representation and storage as
# well as the classifiers to tune on the data
import pandas as pd
import numpy as np
from function import *

# Read in the data
file_path = 'COMP30027_2018S1_proj2-data/'
file_name = 'train_raw.csv'
data = pd.read_csv(file_path + file_name, header=None, encoding='ISO-8859-1')

# Apply pre_process function to the text column
data[6] = data[6].apply(pre_process)
