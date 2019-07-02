import pickle
import numpy as np
# Refer to https://github.com/sdpython/mlinsights/blob/master/mlinsights/mlmodel/direct_blas_lapack.pyx
# Can't run executable on external drive, thus running direct_blas_lapack from local HDD
import sys
sys.path.append('/home/yluan/Documents/ML')
from direct_blas_lapack import dgelss
from scipy.linalg.lapack import dgelss as scipy_dgelss

"""
Demonstration with A[3 x 4] * x[4 x 1] = b[3 x 1], an Under-determined Linear System
"""
# 3 x 2:
A_0302 = np.array([[10., 1.], [12., 1.], [13., 1]])
# 3 x 1
B_0301 = np.array([[20., 22., 23.]]).T

# Analytical solution of this under-determined system can be
# x = [[1], [2], [3], [4]] or [[1], [3], [3], [3]]
# 5 x 4:
A_0504 = np.array([[10., 1., 1, 1], [12., 1., 1, 1], [13., 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
# 5 x 1:
B_0501 = np.array([[19, 21, 22, 0., 0]]).T
# Least-squares from Scipy's dgelss
print('\nScipy dgelss on simple linear system:')
v, xAB0, s, rank, work, infoAB0 = scipy_dgelss(A_0504, B_0501)
print(xAB0[:-1])

# 4 x 5:
A_0405 = A_0504.T.copy()
# Least-squares from mlinsights' direct_blas_lapack
print('\ndirect_blas_lapack dgelss on simple linear system:')
infoAB1 = dgelss(A_0405, B_0501)
# 0 info means success
assert infoAB1 == 0
# B contains the solution
xAB1 = B_0501[:-1]
print(xAB1)



"""
Demonstration with TB[9 x 10] * g[10 x 1] = bij[9 x 1], an Under-determined Linear System
"""
# Both TB and bij here are C-contiguous
# TB is A and is nPoint x 10 bases x 9 components
tb = pickle.load(open('/media/yluan/ALM_N_H_OneTurb/Fields/Result/24995.0788025/TB_Confined3.p', 'rb'))
# bij is b and is nPoint x 9 components
bij = pickle.load(open('/media/yluan/ALM_N_H_OneTurb/Fields/Result/24995.0788025/bij_Confined3.p', 'rb'))

# sum(TB) is 10 bases x 9 components by summing all points
tb_sum = np.sum(tb, axis = 0)
# sum(bij) is 9 components x 1
bij_sum = np.sum(bij, axis = 0)
bij_sum = np.atleast_2d(bij_sum).T

# 11 components x 1:
bij_sum_1101 = np.vstack((bij_sum, np.zeros((2, 1))))
# 10 bases x 11 components:
tb_sum_1011 = np.hstack((tb_sum, np.zeros((10, 2))))
# 11 components x 10 bases
# copy() to preserve C-contiguous
tb_sum_1110 = tb_sum_1011.T.copy()

# # Verification from Numpy's lstsq
# g, residual, rank, s = np.linalg.lstsq(tb_sum.T, bij_sum, rcond = None)

print("\nScipy dgelss on TB*g = bij:")
v, xTb0, s, rank, work, infoTb0 = scipy_dgelss(tb_sum_1110, bij_sum_1101)
print(xTb0[:-1])
# RMSE between origional b and and reconstructed Ax
residual0 = np.sqrt(np.mean((bij_sum - np.dot(tb_sum.T, xTb0[:-1]))**2))
print('RMSE from Scipy dgelss, {}, should ~0'.format(residual0))

print('\ndirect_blas_lapack dgelss on TB*g = bij:')
infoTb1 = dgelss(tb_sum_1011, bij_sum_1101)
assert infoTb1 == 0
# g is contained in first 10 of bij_sum_1101
xTb1 = bij_sum_1101[:-1]
print(xTb1)
# RMSE between origional b and and reconstructed Ax
residual1 = np.sqrt(np.mean((bij_sum - np.dot(tb_sum.T, xTb1))**2))
print('RMSE from direct_blas_lapack dgelss, {}, should ~0'.format(residual1))


"""
Demonstration with TB^T*TB[10 x 10] * g[10 x 1] = TB^T*bij[10 x 1]
"""
# Recall tb_sum is 10 bases x 9 components,
# and TB^T*TB should be [10 x 9] x [9 x 10] = 10 bases x 10 bases
tb_tb = np.dot(tb_sum, tb_sum.T)
# Recall bij_sum is 9 x 1, TB^T*bij should be 10 x 9 x 9 x 1 = 10 bases x 1
tb_bij = np.dot(tb_sum, bij_sum)

# Like in previous demonstration, add an extra col so that 11 cols yield 10 g
# FIXME: in fact, as long as row >= col in A, dgelss of direct_blas_lapack should be fine
tb_tb_1011 = np.hstack((tb_tb, np.zeros((10, 1))))
tb_bij_1101 = np.vstack((tb_bij, np.zeros((1, 1))))
tb_tb_1110 = np.vstack((tb_tb, np.zeros((1, 10))))

print("\nScipy dgelss on (TB^T*TB)*g = (TB^T*bij):")
v, xTb2, s, rank, work, infoTb2 = scipy_dgelss(tb_tb_1110, tb_bij_1101)
print(xTb2[:-1])
# RMSE between origional b and and reconstructed Ax
residual2 = np.sqrt(np.mean((bij_sum - np.dot(tb_sum.T, xTb2[:-1]))**2))
print('RMSE from Scipy dgelss, {}, should ~0'.format(residual2))

print('\ndirect_blas_lapack dgelss on (TB^T*TB)*g = (TB^T*bij):')
# infoTb3 = dgelss(tb_tb_1011, tb_bij_1101)
infoTb3 = dgelss(tb_tb, tb_bij)
assert infoTb3 == 0
# g is contained in tb_bij
xTb3 = tb_bij
print(xTb3)
# RMSE between origional b and and reconstructed Ax
residual3 = np.sqrt(np.mean((bij_sum - np.dot(tb_sum.T, xTb3))**2))
print('RMSE from direct_blas_lapack dgelss, {}, should ~0'.format(residual3))

# Verification from Numpy's lstsq
g, residual4, rank, s = np.linalg.lstsq(tb_tb, tb_bij, rcond = None)
