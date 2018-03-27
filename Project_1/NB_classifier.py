# Code written by Edmond Pan (Student_Num = 841389)
# Python script that implements the Naive Bayes classifier

import math as mth
from collections import defaultdict
from dataset import DataSet
from shared_funcs import *


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


def predict_sup_single(data_instance, model):
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


def log_pred_sup_single(data_instance, model):
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


def predict_supervised(filename):
    """
    Function that predicts the class for a set of instances
    :param filename: The filename of the .csv file to create predictions from
    :return: A list of predicted classes with indices corresponding to the row number
    """
    predicted = []
    # Build the model from the dataset.
    data = DataSet(filename)
    model = SupervisedModel(data)
    for row in data.table:
        # Also skip last attribute as that is the class distribution
        classified = predict_sup_single(row[:-1], model)
        if classified == log_pred_sup_single(row[:-1], model):
            predicted.append(classified)
    return predicted


def evaluate_supervised(filename):
    """
    Function that prints the accuracy rating for a given set of predictions and
    constructs a confusion matrix
    :param filename: The name of the .csv files the predictions were created from
    :return: A confusion matrix in the format of a 2D dictionary accessible in the
             format matrix[predicted_class][expected_class]
    """
    # Get the expected classes from the dataset
    dataset = DataSet(filename)
    expected = []
    for row in dataset.table:
        expected.append(row[-1])
    predicted = predict_supervised(filename)
    matrix = print_confusion(predicted, expected)
    accuracy = 0
    total_instances = dataset.get_num_rows()
    for key, value in matrix.items():
        accuracy += max(matrix[key].values())
    print('Accuracy = ' + str((accuracy/total_instances) * 100))
    return matrix

