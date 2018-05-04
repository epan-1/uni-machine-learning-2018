###
# This Python file will implement the Stacking machine learning ensemble
###

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import datasets


"""
IMPORTANT THIS ENSEMBLE STACKER DOES NOT WORK CORRECTLY IN ITS CURRENT FORM
IN ORDER TO MAKE IT WORK ONE NEEDS TO:
    - CHANGE IT INTO A CLASS WITH
        INIT METHOD THAT CREATES THE META DATASET OF PREDICTIONS WHEN A TRAIN/FIT
        METHOD IS CALLED ON IT. LIKE THOSE IN SKLEARN CLASSIFIERS
        ADD A PREDICT METHOD THAT TAKES IN THE ATTRIBUTES OF THE SAME FORM AS THOSE
        THAT WERE USED TO TRAIN THE ENSEMBLE STACKING MODEL. THIS PREDICT FUNCTION
        WILL BE THE ONE THAT RETURNS THE PREDICTED CLASS LABELS FOR THE GIVEN
        INPUT DATA
"""

def create_meta_dataset(clfs, X, y, n_folds):

    # Output dataset
    data = pd.DataFrame()

    # List to store the different predictions and corresponding true class labels
    predictions = [[] for i in range(len(clfs))]
    y_labels = []

    kf = KFold(n_splits=n_folds)

    for train, test in kf.split(X, y):
        new_X = [X[i] for i in train]
        new_y = [y[i] for i in train]

        for train_2, test_2 in kf.split(new_X, new_y):
            X_train = [new_X[i] for i in train_2]
            y_train = [new_y[i] for i in train_2]
            clf_test = [new_X[i] for i in test_2]
            clf_y = [new_y[i] for i in test_2]

            classifier_num = 0
            for classifier in clfs:
                classifier.fit(X_train, y_train)
                # Also predict the training instance and add it to the column
                # for the new dataset
                predictions[classifier_num] += classifier.predict(clf_test).tolist()
                classifier_num += 1
            y_labels += clf_y

    # Create the new dataset
    i = 0
    while i < len(predictions):
        data[i] = predictions[i]
        i += 1
    data[i] = y_labels

    return data

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# final_clf = LogisticRegression()
# clfs = [MultinomialNB(), LinearSVC(), DecisionTreeClassifier()]
#
# data = create_meta_dataset(clfs, X, y, 10)
#
# final_train = data.iloc[:, 0:-1].as_matrix()
# final_y = data.iloc[:, -1].as_matrix()

