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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB

# Read in the data
file_path = 'COMP30027_2018S1_proj2-data/'
file_name = 'train_processed.csv'
data = pd.read_csv(file_path + file_name, header=None, skiprows=1,
                   skipinitialspace=True, encoding='ISO-8859-1')

# Fill in nan values with a single whitespace
data = data.fillna(value=' ')

# Get X and y out of the data
X_train = data[0].tolist()
y_train = np.array(data[1].tolist())

# Create and fit a CountVectoriser
vectoriser = CountVectorizer()
X_train_cv = vectoriser.fit_transform(X_train)

# Using Naive Bayes
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_cv, y_train)
print("NB Accuracy = " + str(nb.score(X_train_cv, y_train)))
