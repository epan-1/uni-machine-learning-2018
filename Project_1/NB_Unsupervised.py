# Code written by Edmond Pan (Student_Num = 841389)
# Python script that implements an unsupervised version
# of the Naive Bayes classifier

import math as mth
from collections import defaultdict

# Get the DataSet class from the other Python file
from NB_classifier import DataSet


class UnsupervisedModel:

    def __init__(self, dataset, num_iter = 3):
        """
        Constructor for an UnsupervisedModel
        :param dataset: A DataSet object containing the dataset to classify
        :param num_iter: An integer representing the number of iterations to
                         use when attempting to get more accurate probabilities
                         DEFAULTS to 3 iterations
        """
        # Variables to store prior counts and posterior counts
        self.prior_counts, self.posterior_counts, self.missing_counts = self.create_counts(dataset)


    @classmethod
    def create_counts(cls, dataset):
        """
        Function to produce the fractional counts of the different classes and attributes
        in the dataset
        :param dataset: A DataSet object to read and produce counts from
        :return: 2 dictionaries in tuple form. i.e. (dict1, dict2) that represent the
                 counts to be used for prior and posterior probabilities respectively
        """
        prior_counts = defaultdict(int)
        # Triple nested dictionary. Key of first dict contains attribute names, second dict
        # contains attribute values as keys, third dict contains class names as keys
        posterior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Structure to store how many missing values exist for each attribute. A key refers
        # to the attribute name, second key refers to class name
        missing_counts = defaultdict(lambda: defaultdict(int))

        # Add prior counts to default dict
        for row in dataset.table:
            # Add the corresponding fractional count to the correct class
            for class_name, value in row[-1].items():
                prior_counts[class_name] += value

        for row in dataset.table:
            row_index = 0
            for attribute in row[:-1]:
                # If attribute == ?, then add to missing counts dict
                if attribute == '?':
                    for class_name, value in row[-1].items():
                        missing_counts[row_index][class_name] += value
                else:
                    # Initialise all the dictionaries of each possible attribute value
                    # to contain all possible classes. Note missing values will not contribute
                    # to the counts and are treated as a separate count that will not be used
                    for key, value in prior_counts.items():
                        posterior_counts[row_index][attribute][key]
                row_index += 1
            # Now add the counts
            row_index = 0
            for attribute in row[:-1]:
                # Skip adding missing values
                if attribute == '?':
                    row_index += 1
                else:
                    # Add to the counts
                    for class_name, value in row[-1].items():
                        posterior_counts[row_index][attribute][class_name] += value
                    row_index += 1

        return prior_counts, posterior_counts, missing_counts

    def __calc_probabilities__(self, dataset):
        """
        Function that takes the counts and produces prior and posterior probabilities
        :param dataset: A DataSet object to create the probabilities from
        :return: 2 dictionaries in tuple form. i.e. (dict1, dict2) that represent prior
                 and posterior probabilities respectively
        """
        return


a = DataSet('flu-test.csv')
a.random_initial()
b = UnsupervisedModel(a)
c = b.prior_counts
d = b.posterior_counts
