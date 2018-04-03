###
# Class that represents a dataset that has been stored as a .csv file
# Written by Edmond Pan (841389)
###

from collections import defaultdict
import numpy as np


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