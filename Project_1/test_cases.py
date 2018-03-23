# Python file that contains some brute-force, copy pasted
# functions that implement different test datasets allowing
# calculations to be done on them

from collections import defaultdict

def temp_test(table):
    a = defaultdict(float)
    a['cold\n'] = 0.4
    a['flu\n'] = 0.6
    table[0][-1] = a

    b = defaultdict(float)
    b['cold\n'] = 0.7
    b['flu\n'] = 0.3
    table[1][-1] = b

    c = defaultdict(float)
    c['cold\n'] = 0.9
    c['flu\n'] = 0.1
    table[2][-1] = c

    d = defaultdict(float)
    d['cold\n'] = 0.2
    d['flu\n'] = 0.8
    table[3][-1] = d

    e = defaultdict(float)
    e['cold\n'] = 0.6
    e['flu\n'] = 0.4
    table[4][-1] = e

