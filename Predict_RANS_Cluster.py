from joblib import load
from Preprocess.Tensor import processReynoldsStress, getBarycentricMapData, expandSymmetricTensor, contractSymmetricTensor,makeRealizable
import time as t
import numpy as np
import pickle
import os

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case in both ML and test
ml_casename = 'N_H_OneTurb_LowZ_Rwall2'  # str
test_casename = 'N_H_OneTurb_LowZ_Rwall2'  # str
# Absolute parent directory of ML and test case
casedir = '/home/yluan/TurbML/'  # str
# The case folder name storing the estimator
estimator_folder = "Result"  # str
estimator_name = 'TBDT'  # 'TBDT', 'TBRF', 'TBAB', 'TBGB'
confinezone = '2'  # '', '1', '2'
# Iteration to make predictions realizable
realize_iter = 0  # int
# What to do with prediction too far from realizable range
# If bij_novelty is 'excl', 2 realize_iter is automatically used
bij_novelty = None  # None, 'excl', 'reset'
# Whether filter the prediction field with Gaussian filter
filter = False
# Multiplier to realizable bij limits [-1/2, 1/2] off-diagonally and [-1/3, 2/3] diagonally.
# Whatever is outside bounds is treated as NaN.
# Whatever between bounds and realizable limits are made realizable
bijbnd_multiplier = 2.

estimator_fullpath = casedir + '/' + ml_casename + '/' + estimator_folder + '/'
estimator_name += '_Confined' + str(confinezone)

time = ''

bij_novelty_ext = '' if bij_novelty is None else bij_novelty
result_dir = '/home/yluan/scratch/TurbML/'+ test_casename + '/' + estimator_name + bij_novelty_ext + '/' + time + '/'
os.makedirs(result_dir, exist_ok=True)
print("\nCurrent test case: {}, estimator: {}, bij_novelty: {}".format(test_casename, estimator_name, bij_novelty))


"""
Load Data and Regressor
"""
print('\nLoading regressor and data... ')
regressor = load(estimator_fullpath + estimator_name + '.joblib')
list_data_test = pickle.load(open(casedir + '/' + test_casename + '/list_data_test_Confined' + str(confinezone) + '.p', 'rb'),
                             encoding='ASCII')
# ccx_test = list_data_test[0][:, 0]
# ccy_test = list_data_test[0][:, 1]
# ccz_test = list_data_test[0][:, 2]

x_test = list_data_test[1]
x_test[x_test > 1e10] = 1e10
x_test[x_test < -1e10] = 1e10
y_test = list_data_test[2]
tb_test = list_data_test[3]
mask = list_data_test[4]
del list_data_test


"""
Predict
"""
print('\nPredicting bij...')
t0 = t.time()
y_pred = regressor.predict(x_test, tb=tb_test, bij_novelty=bij_novelty)
# Remove NaN predictions
if bij_novelty == 'excl':
    print("Since bij_novelty is 'excl', removing NaN and making y_pred realizable...")
    nan_mask = np.isnan(y_pred).any(axis=1)
    # ccx_test = ccx_test[~nan_mask]
    # ccy_test = ccy_test[~nan_mask]
    # ccz_test = ccz_test[~nan_mask]
    y_pred = y_pred[~nan_mask]
    for _ in range(2):
        y_pred = makeRealizable(y_pred)

t1 = t.time()
print('\nFinished bij prediction in {:.4f} s'.format(t1 - t0))

print('\nAssigning predicted domain bij back to full domain, with unpredicted region being 0...')
y_pred_all = np.zeros((len(mask), 6))
y_pred_all[mask] = y_pred


"""
Write Predicted bij back to OpenFOAM File
"""
fieldname = 'bij_pred'
print('\nWriting bij prediction to OpenFOAM format...')
header_symmtensor = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.4.0                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    location    "%s";
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<symmTensor> 
%d
(
""" % (time, fieldname, len(mask))

footer_symmtensor = """)
;

boundaryField
{
    lower
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0);
    }
    upper
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0);
    }
    south
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0);
    }
    west
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0);
    }
    east
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0);
    }
    north
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0);
    }
}

// ************************************************************************* //
"""

fh = open(result_dir + fieldname, 'w')
fh.write(header_symmtensor)
np.savetxt(fh, y_pred_all, fmt="(%.10f %.10f %.10f %.10f %.10f %.10f)")
fh.write(footer_symmtensor)
fh.close()
print('\nFinished writing bij prediction to OpenFOAM format')
del y_pred_all


"""
OpenFOAM Header and Footer for Vectors and Full Tensor
"""
def headerVecOF(time, fieldname, n_cells):
    header_vec = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.4.0                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "%s";
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<vector> 
%d
(
"""%(time, fieldname, n_cells)

    return header_vec

footer_vec = """)
;

boundaryField
{
    lower
    {
        type            calculated;
        value           uniform (0 0 0);
    }
    upper
    {
        type            calculated;
        value           uniform (0 0 0);
    }
    south
    {
        type            calculated;
        value           uniform (0 0 0);
    }
    west
    {
        type            calculated;
        value           uniform (0 0 0);
    }
    east
    {
        type            calculated;
        value           uniform (0 0 0);
    }
    north
    {
        type            calculated;
        value           uniform (0 0 0);
    }
}

// ************************************************************************* //
"""

def headerTensorOF(time, fieldname, n_cells):
    header_tensor = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.4.0                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volTensorField;
    location    "%s";
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<tensor> 
%d
(
"""%(time, fieldname, n_cells)

    return header_tensor

footer_tensor = """)
;

boundaryField
{
    lower
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0 0 0 0);
    }
    upper
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0 0 0 0);
    }
    south
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0 0 0 0);
    }
    west
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0 0 0 0);
    }
    east
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0 0 0 0);
    }
    north
    {
        type            calculated;
        value           uniform (0 0 0 0 0 0 0 0 0);
    }
}

// ************************************************************************* //
"""


"""
Calculate Eigenvals and Eigenvecs
"""
t0 = t.time()
_, eigval_test, eigvec_test = processReynoldsStress(y_test, make_anisotropic=False, realization_iter=0, to_old_grid_shape=False)
del y_test
# If filter was True, eigval_pred_test is a mesh grid
_, eigval_pred, eigvec_pred = processReynoldsStress(y_pred, make_anisotropic=False, realization_iter=0, to_old_grid_shape=False)
del y_pred
t1 = t.time()
print('\nFinished calculating eigenvals and eigenvecs in {:.4f} s'.format(t1 - t0))


"""
Write Eigenvecs back to OpenFOAM File
"""
print('\nAssigning true eigenvec of confined domain to the whole domain, with out-of confinement eigenvec being (0, 0, 0, 0, 0, 0, 0, 0, 0)...')
fieldname = 'Eigenvec'
eigvec_all = np.zeros((len(mask), 9))
eigvec_all[mask] = eigvec_test.reshape((eigval_test.shape[0], 9))
print('\nWriting true eigenvec to OpenFOAM format...')
header_tensor = headerTensorOF(time, fieldname, len(mask))
fh = open(result_dir + fieldname, 'w')
fh.write(header_tensor)
np.savetxt(fh, eigvec_all, fmt="(%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f)")
fh.write(footer_tensor)
fh.close()
print('\nFinished writing true eigenvec to OpenFOAM format')
del eigvec_all

print('\nAssigning predicted eigenvec of confined domain to the whole domain, with out-of confinement RGB being (0, 0, 0, 0, 0, 0, 0, 0, 0)...')
fieldname = 'Eigenvec_pred'
eigvec_pred_all = np.zeros((len(mask), 9))
eigvec_pred_all[mask] = eigvec_pred.reshape((eigvec_pred.shape[0], 9))
print('\nWriting predicted eigenvec to OpenFOAM format...')
header_tensor = headerTensorOF(time, fieldname, len(mask))
fh = open(result_dir + fieldname, 'w')
fh.write(header_tensor)
np.savetxt(fh, eigvec_pred_all, fmt="(%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f)")
fh.write(footer_tensor)
fh.close()
print('\nFinished writing predicted eigenvec to OpenFOAM format')
del eigvec_pred_all


"""
Calculate Barycentric Map
"""
t0 = t.time()
xy_bary, rgb_bary = getBarycentricMapData(eigval_test)
del eigval_test
xy_bary_pred, rgb_bary_pred = getBarycentricMapData(eigval_pred)
del eigvec_pred
t1 = t.time()
print('\nFinished getting Barycentric map data in {:.4f} s'.format(t1 - t0))

# Limit RGB values to max of 1
rgb_bary[rgb_bary > 1.] = 1.
rgb_bary_pred[rgb_bary_pred > 1.] = 1.


"""
Write Barycentric RGB back to OpenFOAM File
"""
print('\nAssigning true RGB values of confined domain to the whole domain, with out-of confinement RGB being (0, 0, 0)...')
fieldname = 'RGB'
rgb_bary_all = np.zeros((len(mask), 3))
rgb_bary_all[mask] = rgb_bary
del rgb_bary
print('\nWriting true barycentric RGB to OpenFOAM format...')
header_vec = headerVecOF(time, fieldname, len(mask))
fh = open(result_dir + fieldname, 'w')
fh.write(header_vec)
np.savetxt(fh, rgb_bary_all, fmt="(%.10f %.10f %.10f)")
fh.write(footer_vec)
fh.close()
print('\nFinished writing true barycentric RGB to OpenFOAM format')
del rgb_bary_all

print('\nAssigning predicted RGB values of confined domain to the whole domain, with out-of confinement RGB being (0, 0, 0)...')
fieldname = 'RGB_pred'
rgb_bary_pred_all = np.zeros((len(mask), 3))
rgb_bary_pred_all[mask] = rgb_bary_pred
del rgb_bary_pred
print('\nWriting predicted barycentric RGB to OpenFOAM format...')
header_vec = headerVecOF(time, fieldname, len(mask))
fh = open(result_dir + fieldname, 'w')
fh.write(header_vec)
np.savetxt(fh, rgb_bary_pred_all, fmt="(%.10f %.10f %.10f)")
fh.write(footer_vec)
fh.close()
print('\nFinished writing predicted barycentric RGB to OpenFOAM format')
del rgb_bary_pred_all


"""
Write Barycentric Map Coordinate back to OpenFOAM File
"""
print('\nAssigning true barycentric map coordinate of confined domain to the whole domain, with out-of confinement coordinate being (0, 0, 0)...')
fieldname = 'XYbary'
xy_bary_all = np.zeros((len(mask), 3))
xy_bary_all[:, :2][mask] = xy_bary
del xy_bary
print('\nWriting true barycentric xy-coordinate to OpenFOAM format...')
header_vec = headerVecOF(time, fieldname, len(mask))
fh = open(result_dir + fieldname, 'w')
fh.write(header_vec)
# Last D is dummy
np.savetxt(fh, xy_bary_all, fmt="(%.10f %.10f %d)")
fh.write(footer_vec)
fh.close()
print('\nFinished writing true barycentric map coordinate to OpenFOAM format')
del xy_bary_all

print('\nAssigning predicted barycentric map coordinate of confined domain to the whole domain, with out-of confinement coordinate being (0, 0, 0)...')
fieldname = 'XYbary_pred'
xy_bary_pred_all = np.zeros((len(mask), 3))
xy_bary_pred_all[:, :2][mask] = xy_bary_pred
del xy_bary_pred
print('\nWriting predicted barycentric xy-coordinate to OpenFOAM format...')
header_vec = headerVecOF(time, fieldname, len(mask))
fh = open(result_dir + fieldname, 'w')
fh.write(header_vec)
# Last D is dummy
np.savetxt(fh, xy_bary_pred_all, fmt="(%.10f %.10f %d)")
fh.write(footer_vec)
fh.close()
print('\nFinished writing predicted barycentric map coordinate to OpenFOAM format')
