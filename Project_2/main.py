###
# Driver code that creates and runs different classifiers for the problem of
# classifying an author's age based on their blog posts.
# Written by Edmond Pan (841389)
###

# Import all of the required libraries for data representation and storage as
# well as the classifiers to tune on the data
from function import *
from stacking_ensemble import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

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

test = pd.read_csv(file_path + 'test_processed.csv', header=None, skiprows=1,
                   skipinitialspace=True, encoding='ISO-8859-1')

# # Fill in nan values with a single whitespace
train = train.fillna(value=' ')
dev = dev.fillna(value=' ')
test = test.fillna(value=' ')

# Get X and y out of the data
X_train = train[0].tolist()
X_dev = dev[0].tolist()
X_top10_train = top10_train.iloc[:, 1:-1]
X_top10_dev = top10_dev.iloc[:, 1:-1]
X_test = test[0].tolist()


y_train = np.array(train[1].tolist())
y_dev = np.array(dev[1].tolist())
y_top10_train = top10_train[31]
y_top10_dev = top10_dev[31]

# Create and fit a CountVectoriser
# vectoriser = CountVectorizer()
# X_train_cv = vectoriser.fit_transform(X_train)
# X_dev_cv = vectoriser.transform(X_dev)

# Create and fit a TFIDF Vectoriser
vectoriser = CountVectorizer(ngram_range=(1, 2))
X_train_cv = vectoriser.fit_transform(X_train)
X_dev_cv = vectoriser.transform(X_dev)
X_test_cv = vectoriser.transform(X_test)

# Select k Best features via chi-squared
# x2 = SelectKBest(chi2, k=50000)
# x2.fit(X_train_cv, y_train)
# X_train_x2 = x2.transform(X_train_cv)
# X_dev_x2 = x2.transform(X_dev_cv)

# Using Naive Bayes
nb = MultinomialNB(alpha=1.0)
# nb.fit(X_train_cv, y_train)
# print("NB Accuracy = " + str(nb.score(X_dev_cv, y_dev)))

# Using SVMs (This has the best parameters)
C = 0.1
lsv = LinearSVC(C=C)
# lsv.fit(X_train_cv, y_train)
# print(lsv.score(X_dev_cv, y_dev))

# Using Logistic regression (This is the optimal performance)
lr = LogisticRegression(C=1, solver='saga', max_iter=1000, n_jobs=-1)
lr.fit(X_train_cv, y_train)
print("LR Accuracy = " + str(lr.score(X_dev_cv, y_dev)))

# voter = VotingClassifier(estimators=[('nb', nb), ('lsv', lsv), ('lr', lr)], voting='hard', n_jobs=-1)
# voter.fit(X_train_cv, y_train)
# print("Voter accuracy = " + str(voter.score(X_dev_cv, y_dev)))

# Producing the predictions
with open(file_path + 'predictions.csv', 'w+') as out:
    out.write('Id,Prediction\n')
    # Get the predictions
    predictions = lr.predict(X_test_cv)
    instance_id = 1
    for predict in predictions:
        line = ''
        line += '3' + str(instance_id) + ',' + predict + '\n'
        out.write(line)
        instance_id += 1

