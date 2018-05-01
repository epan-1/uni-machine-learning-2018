###
# This Python file is used for taking the raw datasets, pre-processing them
# and then writing them back out as new processed csv files.
###

import pandas as pd
from function import *

# Read in the data taking only the age column and text column
file_path = 'COMP30027_2018S1_proj2-data/'
file_name = 'train_raw.csv'
data = pd.read_csv(file_path + file_name, header=None, usecols=(2, 6),
                   encoding='ISO-8859-1')

# Apply pre_process function to the text column
data[6] = data[6].apply(pre_process)

# Write out the dataframe to a csv file
data.to_csv(file_path + 'train_processed.csv', columns=(6, 2), index=False)

# The file that was written out will have the blog post in column 1 and age
# in column 2.
