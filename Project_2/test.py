###
# This Python file is simply to test indexing and other code bits from the
# different imported libraries
###

import pandas as pd

file_path = 'COMP30027_2018S1_proj2-data/'

top10_train = pd.read_csv(file_path + 'train_top10.csv', header=None,
                          encoding='ISO-8859-1')
X_top10 = top10_train.iloc[:, 1:-1]

