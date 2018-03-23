# Code written by Edmond Pan (Student_Num = 841389)
# Python script that implements the Naive Bayes classifier

import numpy as np
import math as mth
from collections import defaultdict


class DataSet:

    def __init__(self, filename):
        """
        Reads in the csv file into an appropriate data structure
        :param filename: Name of the .csv file
        """
        # Variables to store metadata about the table structure
        self.num_rows = 0
        self.num_cols = 0
        self.table = []
        file = open('2018S1-proj1_data/' + filename, 'r')
        for line in file.readlines():
            # Split based on common to get the values
            row = line.split(',')
            self.num_cols = len(row)
            # Add row to table and increment row count
            self.table.append(row)
            self.num_rows += 1
        file.close()

    def get_num_rows(self):
        """
        Gets the number of rows in the table
        :return: Integer referencing number of rows in the table
        """
        return self.num_rows

    def get_num_cols(self):
        """
        Gets the number of cols in the table
        :return: Integer referring to number of cols in table
        """
        return self.num_cols

    def get_table(self):
        """
        Gets the stored table
        :return: Returns a list of rows
        """
        return self.table

    def random_initial(self):
        """
        Function that replaces the class labels with random (non-uniform)
        class distributions. NOTE ONLY USE WHEN DOING UNSUPERVISED NB
        :return: None
        """
        # Default dict containing all possible classes
        classes = defaultdict(float)
        for row in self.table:
            # Keep them all at 0 since they will b replaced with
            # random fractional counts that sum to 1
            classes[row[-1]] = 0

        # Now remove the class labels and add the classes dictionaries while
        # initialising the values to random fractional counts
        for row in self.table:
            # Add the random values to the dictionary
            values = np.random.dirichlet(np.ones(len(classes)), size=1)
            i = 0
            for key, value in classes.items():
                classes[key] = values[0][i]
                i += 1
            # Replace the old class label with the random class distribution.
            # Make sure to return a copy of classes instead of passing the reference
            row[-1] = classes.copy()
        return


class SupervisedModel:

    def __init__(self, dataset):
        """
        Constructor for a SupervisedModel object
        :param dataset: A DataSet object containing the .csv file to calculate
                        probabilities from
        """
        # Variables to store counts to use in calculating prior and posterior probabilities
        self.prior_counts, self.posterior_counts, self.missing_counts = self.create_counts(dataset)

        self.prior_prob, self.posterior_prob = self.__calc_probabilities__(dataset)

    def get_prior_counts(self):
        return self.prior_counts

    def get_posterior_counts(self):
        return self.posterior_counts

    def get_prior_prob(self):
        return self.prior_prob

    def get_posterior_prob(self):
        return self.posterior_prob

    @classmethod
    def create_counts(cls, dataset):
        """
        Function to count up the frequencies of the different classes and attributes
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
            # Assuming last column is the class
            prior_counts[row[-1]] += 1
        # Add posterior counts to data structure
        for row in dataset.table:
            row_index = 0
            for attribute in row[:-1]:
                # If attribute == ?, then add to missing counts dict
                if attribute == '?':
                    missing_counts[row_index][row[-1]] += 1
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
                    posterior_counts[row_index][attribute][row[-1]] += 1
                    row_index += 1

        return prior_counts, posterior_counts, missing_counts

    def __calc_probabilities__(self, dataset):
        """
        Function that takes the counts and produces prior and posterior probabilities
        :param dataset: A DataSet object to create the probabilities from
        :return: 2 dictionaries in tuple form. i.e. (dict1, dict2) that represent prior
                 and posterior probabilities respectively
        """
        prior_prob = defaultdict(float)
        # Format for this dict is: First dict key refers to attribute name, second dict key refers
        # attribute value and third dict key refers to class name. E.g. P[0]['20-29']['recurrence-events\n']
        posterior_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Values for laplace smoothing
        k = 1

        # Calculating the prior probabilities
        for key, value in self.prior_counts.items():
            prior_prob[key] = value/dataset.get_num_rows()

        # Calculating the posterior probabilities
        for attr_name, val_dict in self.posterior_counts.items():
            # Value to add to denominator when doing Laplace smoothing.
            unique_attr_num = len(self.posterior_counts[attr_name].items())
            for attr_val, class_dict in self.posterior_counts[attr_name].items():
                for class_name, count in self.posterior_counts[attr_name][attr_val].items():
                    # Do Laplace smoothing of the counts
                    numerator = count + k
                    denominator = self.prior_counts[class_name] + unique_attr_num - \
                                  self.missing_counts[attr_name][class_name]
                    posterior_prob[attr_name][attr_val][class_name] = numerator/denominator

        return prior_prob, posterior_prob


def arg_max(dictionary):
    """
    Function that returns the key that has the highest value in a dictionary
    :param dictionary: Dictionary containing the values to check
    :return: A string containing the name of key with the highest value
    """
    return max(dictionary, key=lambda key: dictionary[key])


def predict_supervised(data_instance, model):
    """
    This function takes a single instance and returns a classification
    :param data_instance: A list of attribute values
    :param model: A SupervisedModel object to use for evaluation
    :return: A string corresponding to the most likely class this instance belongs to
    """
    nb_scores = defaultdict(float)
    for class_name, value in model.get_prior_counts().items():
        prior = model.get_prior_prob()
        posterior = model.get_posterior_prob()
        nb_scores[class_name] = prior[class_name]
        attr_index = 0
        for attribute in data_instance:
            # If test instance has missing value, skip it
            if attribute == '?':
                attr_index += 1
            else:
                nb_scores[class_name] *= posterior[attr_index][attribute][class_name]
                attr_index += 1
    return arg_max(nb_scores)


def log_pred_supervised(data_instance, model):
    """
    This function takes a single instance and returns a classification
    :param data_instance: A list of attribute values
    :param model: A SupervisedModel object to use for evaluation
    :return: A string corresponding to the most likely class this instance belongs to
    """
    nb_scores = defaultdict(float)
    for class_name, value in model.get_prior_counts().items():
        prior = model.get_prior_prob()
        posterior = model.get_posterior_prob()
        nb_scores[class_name] = mth.log(prior[class_name])
        attr_index = 0
        for attribute in data_instance:
            if attribute == '?':
                attr_index += 1
            else:
                nb_scores[class_name] += mth.log(posterior[attr_index][attribute][class_name])
                attr_index += 1
    return arg_max(nb_scores)


def evaluate_supervised(filename):
    """
    Function that returns the accuracy rating for a given dataset when using Naive Bayes
    to classify it's instances. NOTE: It uses all instances in the dataset to train and
    will also be testing on the same instances
    :param filename: filename of the dataset to test on
    :return: An accuracy rating as a percentage for the number of instances correctly
             classified
    """
    # Read and build the model from the dataset
    data = DataSet(filename)
    model = SupervisedModel(data)
    # Now for each instance classify it and check if its correct
    num_correct = 0
    total_instances = data.get_num_rows()
    for row in data.table:
        # Check both the normal and log evaluation methods.
        # Also skip last attribute as that is the true class for that instance
        classified = predict_supervised(row[:-1], model)
        classified_log = log_pred_supervised(row[:-1], model)
        if classified == classified_log:
            # Correct class is located at the end of the instance
            if classified == row[-1]:
                num_correct += 1
    return (num_correct/total_instances) * 100


# breast = DataSet('breast-cancer.csv')
data = DataSet('outlook-test.csv')
a = SupervisedModel(data)
b = a.get_prior_counts()
c = a.get_posterior_counts()
prior = a.get_prior_prob()
posterior = a.get_posterior_prob()
test_instance = ['overcast', 'hot', 'humid', 'false']
# test_instance = ['mild', 'severe', 'normal', 'no']
ans = log_pred_supervised(test_instance, a)
# ans = predict_supervised(test_instance, a)
a = evaluate_supervised('car.csv')
