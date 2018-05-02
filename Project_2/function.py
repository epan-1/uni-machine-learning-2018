###
# This Python module contains helper functions for pre-processing the text data
# given in the datasets
###

import re
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def pre_process(data):
    """
    The function will do the following to the data.
        - Force all characters to be lower case
        - Remove any non-alphanumeric characters from the string.
    :param data: Data is a string representing the blog post
    :return: Another string representing the pre-processed data
    """
    # Uses regex '\W+' to remove any character that is not alphabetical or a
    # number.
    output = ''
    for word in data.split(' '):
        if word != '' and word[0].isalnum():
            output += ' ' + re.sub('[\W_]+', '', word.lower(), flags=re.UNICODE)
    return output


def convert_age(raw):
    """
    This function takes the age class and puts it into the specified range given
    by the
    :param raw: A string containing the self-identified age of the blog author
    :return: Returns the corresponding range that the age is in or ? if it
             does not fit into any of the ranges. Inclusive ranges are as follows:
                - 14-16
                - 24-26
                - 34-36
                - 44-46
    """
    if raw == '?':
        # If it's already a question mark skip it
        return '?'
    else:
        if 14 <= int(raw) <= 16:
            return '14-16'
        elif 24 <= int(raw) <= 26:
            return '24-26'
        elif 34 <= int(raw) <= 36:
            return '34-36'
        elif 44 <= int(raw) <= 46:
            return '44-46'
        else:
            return '?'


def ave_cross_val_score(classifier, x, y, n):
    """
    This function returns the average score from an N-Fold Cross Validation
    strategy
    :param classifier:
    :param x:
    :param y:
    :param n:
    :return:
    """
    sum = 0
    scores = cross_val_score(classifier, x, y, cv=n)
    for i in range(n):
        sum += scores[i]
    return sum/n
