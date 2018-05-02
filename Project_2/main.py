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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

# Read in the data
file_path = 'COMP30027_2018S1_proj2-data/'
file_name = 'train_processed.csv'
train = pd.read_csv(file_path + file_name, header=None, skiprows=1,
                    skipinitialspace=True, encoding='ISO-8859-1')
dev = pd.read_csv(file_path + 'dev_processed.csv', header=None, skiprows=1,
                  skipinitialspace=True, encoding='ISO-8859-1')

top10_train = pd.read_csv(file_path + 'train_top10.csv', header=None,
                          encoding='ISO-8859-1')
top10_dev = pd.read_csv(file_path + 'dev_top10.csv', header=None,
                        encoding='ISO-8859-1')

# Fill in nan values with a single whitespace
train = train.fillna(value=' ')
dev = dev.fillna(value=' ')

# Get X and y out of the data
X_train = train[0].tolist()
X_dev = dev[0].tolist()
X_top10_train = top10_train.iloc[:, 1:-1]
X_top10_dev = top10_dev.iloc[:, 1:-1]

y_train = np.array(train[1].tolist())
y_dev = np.array(dev[1].tolist())
y_top10_train = top10_train[31]
y_top10_dev = top10_dev[31]

# Create and fit a CountVectoriser
vectoriser = CountVectorizer()
X_train_cv = vectoriser.fit_transform(X_train)
X_dev_cv = vectoriser.transform(X_dev)

# Using Naive Bayes
# nb = MultinomialNB(alpha=1.0)
# nb.fit(X_train_cv, y_train)
# print("NB Accuracy = " + str(nb.score(X_dev_cv, y_dev)))

# Using Decision trees
# dt = DecisionTreeClassifier(max_depth=None)
# dt.fit(X_train_cv, y_train)
# print("DT Accuracy = " + str(dt.score(X_dev_cv, y_dev)))

# Using SVMs
C = 0.001
clf = LinearSVC(C=C)
clf.fit(X_train_cv, y_train)
print(clf.score(X_dev_cv, y_dev))
