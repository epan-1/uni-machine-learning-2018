# Code written by Edmond Pan (Student_Num = 841389)
# Python script that implements both a supervised and unsupervised of the
# Naive Bayes classifier

import numpy as np
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
        file = open('Project_1/2018S1-proj1_data/' + filename, 'r')
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

    def __str__(self):
        """
        Format: Returns a line by line output of what is stored in the table
        :return: String containing the above format
        """
        result = ''
        for row in self.table:
            result += ' '.join(row)
        return result


class SupervisedModel:

    def __init__(self, dataset):
        """

        :param dataset: A DataSet object containing the .csv file to calculate
                        probabilities from
        """
        # Variables to store both the prior and posterior probabilities
        self.priors = defaultdict(int)
        self.posteriors = {}

        # Add class counts to default dict
        for row in dataset.table:
            self.priors[row[-1]] += 1

    def get_priors(self):
        return self.priors


breast = DataSet('breast-cancer.csv')
a = SupervisedModel(breast)
