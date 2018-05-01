###
# This Python module contains helper functions for pre-processing the text data
# given in the datasets
###

import re


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
