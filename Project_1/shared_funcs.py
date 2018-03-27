###
# Python file containing functions that are shared and used between both a
# Supervised NB classifier and an unsupervised one
# Written by Edmond Pan (841389)
###

from collections import defaultdict


def arg_max(dictionary):
    """
    Function that returns the key that has the highest value in a dictionary
    :param dictionary: Dictionary containing the values to check
    :return: A string containing the name of key with the highest value
    """
    return max(dictionary, key=lambda key: dictionary[key])


def print_confusion(predicted, expected):
    """
    Function to print the confusion matrix from a list of predicted and
    expected values
    :param predicted: A list of predicted values
    :param expected:  A list of expected values
    :return: None
    """
    if len(predicted) != len(expected):
        print("FATAL ERROR: List lengths do not match")
        return
    # Structure to store the matrix data. Can access the individual values
    # in this format matrix[predicted_class][expected_class]
    matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(predicted)):
        matrix[predicted[i]][expected[i]] += 1
    help_print(matrix)
    return matrix


def help_print(matrix):
    """
    Function that aids in printing the confusion matrix
    :param matrix: A confusion matrix to print. Essentially a double nested
                   dictionary
    :return: None
    """
    header = ''
    line_list = []
    stat = 0
    head_padding = 12
    padding = 8
    for predicted in matrix.keys():
        if stat == 0:
            header += (head_padding - len(predicted)) * ' ' + \
                       predicted.strip('\n') + ' |'
        else:
            header += ' ' + predicted.strip('\n') + ' |'
        line = ''
        stat = 0
        for expected in matrix.keys():
            if stat == 0:
                line += (padding - len(predicted)) * ' ' + \
                         predicted.strip('\n') + ' | '
                line += str(matrix[predicted][expected]) + ' | '
            else:
                line += str(matrix[predicted][expected])
            stat = 1
        line += '\n'
        line_list.append(line)
    header += '\n'

    print(header)
    for x in line_list:
        print(x)

    return
