import numpy as np

def update(a):
    # Has to be += to update internally, doesn't work with a = a + 10
    a += 10

a = np.arange(5)

print('\nBefore internal update: {}'.format(a))

update(a)

print('\nAfter internal update: {}'.format(a))


print('\nAgain: ')
def update_lst(a):
    a += [10]

b = [1,2,3]

print('\nBefore internal update: {}'.format(b))

update_lst(b)

print('\nAfter internal update: {}'.format(b))

print('\nAgain: ')
def update_str(a):
    a += '99'

c = '123'

print('\nBefore internal update: {}'.format(c))

update_str(c)

print('\nAfter internal update: {}'.format(c))


print('\nAgain: ')

def update_dict(a):
    a['new'] = 10

d = dict(old=2)

print('\nBefore internal update: {}'.format(d))

update_dict(d)

print('\nAfter internal update: {}'.format(d))


print('\nAgain: ')
def update_toarr(a):
    a = np.arange(3)
    print('\nIn function: {}'.format(a))

e = 1

print('\nBefore internal update: {}'.format(e))

update_toarr(e)

print('\nAfter internal update: {}'.format(e))


print('\nAgain: ')
def update_nparr(a):
    a = np.dot(a, a)
    print('\nIn function: {}'.format(a))

f = np.arange(4).reshape((2, 2))

print('\nBefore internal update: {}'.format(f))

update_nparr(f)

print('\nAfter internal update: {}'.format(f))
