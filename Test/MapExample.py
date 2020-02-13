import numpy as np
from functools import partial

ij_all = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (2, 2),
             (2, 3), (3, 2), (3, 3), (4, 3), (3, 4), (4, 4)]

myother = 5
mylst = np.zeros((5, 5))

def myLoop(ij_inputs, other, lst):
    print(ij_inputs)
    i, j = ij_inputs

    lst[i, j] = other

    # return i + j + other
    # return lst


list(map(partial(myLoop, other=myother, lst=mylst), ij_all))
# map(partial(myLoop, other=myother, lst=mylst), ij_all)


# def func(u, v, w, x):
#     return u*4 + v*3 + w*2 + x
#
# p = partial(func, 5, 6, 7)
# print(p(8))
