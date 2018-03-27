###
# Driver code that creates and runs both the supervised and unsupervised version
# of the Naive Bayes classifier.
# Written by Edmond Pan (841389)
###

from NB_classifier import *
from NB_Unsupervised import *

# Supervised NB testing

# Unsupervised NB testing
# a = DataSet('flu-test.csv')
# evaluate_unsupervised('flu-test.csv')
# temp_test(a.table)
# a.random_initial()
# b = UnsupervisedModel(a)
# b.iterate(a, 2)
# b = UnsupervisedModel(a)
# c = b.prior_counts
# d = b.posterior_counts
# e = b.prior_prob
# f = b.posterior_prob
# ans = predict_unsupervised(['severe', 'mild', 'high', 'yes'], b)
ans = evaluate_unsupervised('flu-test.csv')
