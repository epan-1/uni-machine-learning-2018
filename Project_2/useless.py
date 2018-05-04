###
# This Python module contains commented out classifiers that I will no longer
# be using
###

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Using Decision trees
# dt = DecisionTreeClassifier(max_depth=None)
# dt.fit(X_train_cv, y_train)
# print("DT Accuracy = " + str(dt.score(X_dev_cv, y_dev)))

# Using AdaBoost (takes too long)
# clf = DecisionTreeClassifier()
# ada = AdaBoostClassifier(clf)
# ada.fit(X_train_cv, y_train)
# print("ADA accuracy = " + str(ada.score(X_dev_cv, y_dev)))

# Using Bagging as a classifier with KNN
# clf = KNeighborsClassifier(n_neighbors=10)
# bag = BaggingClassifier(clf, max_features=0.5, max_samples=0.5)
# bag.fit(X_top10_train, y_top10_train)
# print("Bag accuracy = " + str(bag.score(X_top10_dev, y_top10_dev)))

# Using a random forest classifier
# rforest = RandomForestClassifier(max_depth=10000)
# rforest.fit(X_train_cv, y_train)
# print("Random Forest accuracy = " + str(rforest.score(X_dev_cv, y_dev)))
