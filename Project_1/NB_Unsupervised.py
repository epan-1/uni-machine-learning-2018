# Code written by Edmond Pan (Student_Num = 841389)
# Python script that implements an unsupervised version
# of the Naive Bayes classifier

# Get the DataSet class from the other Python file
from NB_classifier import DataSet, print_confusion, arg_max


class UnsupervisedModel:

    def __init__(self, dataset):
        """
        Constructor for an UnsupervisedModel
        :param dataset: A DataSet object containing the dataset to classify
        """
        # Variables to store prior counts and posterior counts
        self.prior_counts, self.posterior_counts, self.missing_counts = self.create_counts(dataset)
        # Variables to store prior and posterior probabilities
        self.prior_prob, self.posterior_prob = self.__calc_probabilities__(dataset)

    def iterate(self, dataset, n=3):
        """
        Function that iteratively assigns new class distributions to dataset calculated
        from the current model
        :param dataset: A DataSet object to iteratively assign new class distributions
        :param n: The number of iterations to perform. Defaults to 3
        :return: None
        """
        for i in range(n):
            self.recalculate(dataset)
        return

    @classmethod
    def recalculate(cls, dataset):
        """
        Function that will recalculate all of the class distributions in
        dataset and assign more reliable distributions
        :param dataset: A DataSet object to be altered
        :return: None
        """
        # Contains the current model of the dataset
        old_model = UnsupervisedModel(dataset)
        # Go through every instance of the dataset and reassign the class distributions
        for row in dataset.table:
            row[-1] = predict_uns_single(row[:-1], old_model)
        return

    @classmethod
    def create_counts(cls, dataset):
        """
        Function to produce the fractional counts of the different classes and attributes
        in the dataset
        :param dataset: A DataSet object to read and produce counts from
        :return: 2 dictionaries in tuple form. i.e. (dict1, dict2) that represent the
                 counts to be used for prior and posterior probabilities respectively
        """
        prior_counts = defaultdict(float)
        # Triple nested dictionary. Key of first dict contains attribute names, second dict
        # contains attribute values as keys, third dict contains class names as keys
        posterior_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Structure to store how many missing values exist for each attribute. A key refers
        # to the attribute name, second key refers to class name
        missing_counts = defaultdict(lambda: defaultdict(float))

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
        prior_prob = defaultdict(float)
        # Format for this dict is: First dict key refers to attribute name, second dict key refers
        # attribute value and third dict key refers to class name. E.g. P[0]['20-29']['recurrence-events\n']
        posterior_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Calculating the prior probabilities by dividing by the number of instances
        for class_name, value in self.prior_counts.items():
            prior_prob[class_name] = value/dataset.get_num_rows()

        # Now calculate the posterior probabilities by dividing the fractional counts
        # by the fractional class counts
        for attr_name, val_dict in self.posterior_counts.items():
            for attr_val, class_dict in self.posterior_counts[attr_name].items():
                for class_name, count in self.posterior_counts[attr_name][attr_val].items():
                    numerator = count
                    # Subtract the fractional counts from the total if this attribute of
                    # the instance is missing
                    denominator = self.prior_counts[class_name] - self.missing_counts[attr_name][class_name]
                    posterior_prob[attr_name][attr_val][class_name] = numerator/denominator

        return prior_prob, posterior_prob


def predict_uns_single(data_instance, model):
    """
    This function uses a trained unsupervised model to predict the class distribution
    for test instance
    :param data_instance: A list of attributes to predict the class distribution for 
    :param model: An UnsupervisedModel object that will be used to make the predictions
    :return: A dictionary containing a new class distribution for the test_instance
    """
    class_dist = defaultdict(float)
    for class_name, value in model.prior_counts.items():
        prior = model.prior_prob
        posterior = model.posterior_prob
        class_dist[class_name] = prior[class_name]
        attr_index = 0
        for attribute in data_instance:
            # If the data instance has a missing value, skip it during
            # calculations
            if attribute == '?':
                attr_index += 1
            else:
                class_dist[class_name] *= posterior[attr_index][attribute][class_name]
                attr_index += 1
    # Now normalise the values to get the new class distribution.
    # Create a copy to allow the original class_dist to be modified
    temp_values = class_dist.copy()
    for class_name, value in temp_values.items():
        class_dist[class_name] = value/sum(temp_values.values())

    return class_dist


def predict_unsupervised(filename, n=3):
    """
    Function that predicts the class for a set of instances
    :param filename: The filename of the .csv file that contains the set of instances
                     to make predictions from
    :param n: The number of iterations to perform. Defaults to 3
    :return: A list of predicted classes with indices corresponding to the row number
    """
    predicted = []
    # Read and build the model from the dataset.
    data = DataSet(filename)
    data.random_initial()
    model = UnsupervisedModel(data)
    # Iterate over the specified amount of times
    model.iterate(data, n)
    model = UnsupervisedModel(data)
    for row in data.table:
        # Also skip last attribute as that is the class distribution
        predicted.append(arg_max(predict_uns_single(row[:-1], model)))
    return predicted


def evaluate_unsupervised(filename, n=3):
    """
    Function that returns the accuracy rating for a given dataset when using Naive Bayes
    to classify it's instances. NOTE: It uses all instances in the dataset to train and
    will also be testing on the same instances.
    :param filename: filename of the dataset to test on
    :param n: number of times to iterate when building the model. Default is 3
    :return: A confusion matrix in the format of a 2D dictionary accessible in the
             format matrix[predicted_class][expected_class]
    """
    predicted = predict_unsupervised(filename, n)
    expected = []
    data = DataSet(filename)
    for row in data.table:
        expected.append(row[-1])
    matrix = print_confusion(predicted, expected)
    accuracy = 0
    total_instances = data.get_num_rows()
    for key, value in matrix.items():
        accuracy += max(matrix[key].values())
    print('Accuracy = ' + str((accuracy/total_instances) * 100))
    return matrix


from test_cases import *

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
