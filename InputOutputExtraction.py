import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from FieldData import FieldData
from SliceData import SliceProperties
from SetData import SetProperties
from Preprocess.Tensor import processReynoldsStress, expandSymmetricTensor, contractSymmetricTensor, getStrainAndRotationRateTensor, getInvariantBases
from Preprocess.Feature import getInvariantFeatureSet, getSupplementaryInvariantFeatures, getRadialTurbineDistance
from Preprocess.FeatureExtraction import splitTrainTestDataList
from Utility import rotateData

# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from warnings import warn
import time as t

"""v
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case
casename = 'ALM_N_H_OneTurb'  # str
# Absolute directory of this flow case
casedir = '/media/yluan'  # str
# Which time to extract input and output for ML
time = 'latestTime'  # str/float/int or 'latestTime'
# What keyword does the gradient fields contain
grad_kw = 'grad'  # str
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# Flow field counter-clockwise rotation in x-y plane
# so that tensor fields are parallel/perpendicular to flow direction
rotz = np.pi/6  # float [rad]
# Whether confine and visualize the domain of interest, useful if mesh is too large
confine, plot_confinebox = True, False  # bool; bool
# Only when confine is True:
if confine:
    # Subscript of the confined field file name
    confinedfield_namesub = 'Confined'
    # Whether auto generate confine box for each case,
    # the confinement is bordered by
    # 'first': 1st refinement zone
    # 'second': 2nd refined zone
    # 'third': left half of turbine in 2nd refinement zone
    confinezone = 'second'  # 'first', 'second', 'third', None
    # Only when confinezone is False:
    if confinezone is None:
        # Confine box counter-clockwise rotation in x-y plane
        rotbox = np.pi/6  # float [rad]
        # Confine box origin, width, length, height
        boxorig = (0, 0, 0)  # (x, y, z)
        boxl, boxw, boxh = 0, 0, 0  # float

# Absolute cap value for Sij and Rij; TB coefficients' basis;  and TB coefficients
cap_sijrij, cap_tij = 1e9, 1e9  # float/int
# Enforce 0 trace when computing Tij?
tij_0trace = False
# Whether scale Tij like done in Ling et al. (2016), useless for tree models
scale_tb = False
# Kinematic viscosity
nu = 1e-5  # float

# Save anything when possible
save_fields, resultfolder = True, 'Result'  # bool; str


"""
Machine Learning Settings
"""
# Whether to calculate features or directly read from pickle data
calculate_features = False  # bool
# Whether to split train and test data or directly read from pickle data
prep_train_test_data = False  # bool
# Number of samples for Grid Search before training on all data
# Also number of samples to do ML, if None, all samples are used for training
samples_gs, samples_train = 10000, None  # int; int, None
# Fraction of data for testing
test_fraction = 0.  # float [0-1]
# Feature set choice
fs = 'grad(TKE)_grad(p)+'  # 'grad(TKE)', 'grad(p)', 'grad(TKE)_grad(p)', 'grad(TKE)_grad(p)+'
# Use seed for reproducibility, set to None for no seeding
seed = 123  # int, None


"""
Visualization Settings
"""
# Whether to process slices and save them for prediction visualization later
process_slices, process_sets = True, True  # bool
# Slice names for prediction visualization
slicenames = 'auto'
set_types = 'auto'


"""
Process User Inputs, No Need to Change
"""
# Average fields of interest for reading and processing
if 'grad(TKE)_grad(p)' in fs:
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'grad_kResolved', 'grad_kSGSmean', 'UAvg',
              'GAvg', 'divDevR')
elif fs == 'grad(TKE)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_kResolved', 'grad_kSGSmean',
              'GAvg', 'divDevR')
elif fs == 'grad(p)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'UAvg',
              'GAvg', 'divDevR')
else:
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'uuPrime2',
              'grad_UAvg',
              'GAvg', 'divDevR')

# Ensemble name of fields useful for Machine Learning
mlfield_ensemble_name = 'ML_Fields_' + casename
# Case related settings
if 'ParTurb' in casename:
    # Latest time assignment
    if time == 'latestTime':
        if '_H_ParTurb2' in casename:
             time = '25000.0838025'
        elif '_L_ParTurb_Yaw' in casename:
            time = '23000.065'
        elif casename == 'ALM_N_L_ParTurb':
            time = '23000.07'
        elif casename == 'ALM_N_H_ParTurb_HiSpeed':
            # FIXME: update
            time = ''

    if slicenames == 'auto': slicenames = ('alongWindSouthernRotor', 'alongWindNorthernRotor',
                                           'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'oneDupstreamTurbines', 'rotorPlanes', 'oneDdownstreamTurbines',
                                           'threeDdownstreamTurbines', 'fiveDdownstreamTurbines', 'sevenDdownstreamTurbines')
    # TODO: update
    if set_types == 'auto': set_types = ('oneDdownstreamSouthernTurbine_H',
                                         'oneDdownstreamNorthernTurbine_H',
                                         'threeDdownstreamSouthernTurbine_H',
                                         'threeDdownstreamNorthernTurbine_H',
                                         'sevenDdownstreamSouthernTurbine_H',
                                         'sevenDdownstreamNorthernTurbine_H',
                                         'oneDdownstreamSouthernTurbine_V',
                                         'oneDdownstreamNorthernTurbine_V',
                                         'threeDdownstreamSouthernTurbine_V',
                                         'threeDdownstreamNorthernTurbine_V',
                                         'sevenDdownstreamSouthernTurbine_V',
                                         'sevenDdownstreamNorthernTurbine_V')
    # Southern turbine; northern turbine center coordinates, 3D apart from each other
    turblocs = [[1244.083, 1061.262, 90.], [992.083, 1497.738, 90.]]
elif casename == 'ALM_N_H_OneTurb':
    if time == 'latestTime': time = '24995.0438025'
    if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
     'oneDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'fiveDdownstreamTurbine', 'sevenDdownstreamTurbine')
    if set_types == 'auto': set_types = ('oneDdownstreamTurbine_V',
                                         'threeDdownstreamTurbine_V',
                                         'fiveDdownstreamTurbine_V',
                                         'sevenDdownstreamTurbine_V')
    # 1 turbine center coordinate at the upwind turbine location
    turblocs = [1118.083, 1279.5, 90.]
elif 'SeqTurb' in casename:
    if time == 'latestTime':
        if casename == 'ALM_N_H_SeqTurb':
            time = '25000.1638025'
        elif casename == 'ALM_N_L_SeqTurb':
            time = '23000.135'
#             FIXME: merge below
elif casename == 'ALM_N_H_SeqTurb':
    if time == 'latestTime': time = '25000.1638025'
    # FIXME: update
    if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'twoDupstreamTurbineOne', 'rotorPlaneOne', 'rotorPlaneTwo',
                                           'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo',
                                           'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo',
                                           'sixDdownstreamTurbineTwo')
    # Upwind turbine; downwind turbine center coordinates, 7D apart
    turblocs = [[1118.083, 1279.5, 90.], [1881.917, 1720.5, 90.]]
elif casename == 'ALM_N_L_SeqTurb':
    if time == 'latestTime': time = '23000.135'
    # FIXME: update
    if slicenames == 'auto': slicenames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
                                           'twoDupstreamTurbineOne', 'rotorPlaneOne', 'rotorPlaneTwo',
                                           'oneDdownstreamTurbineOne', 'oneDdownstreamTurbineTwo',
                                           'threeDdownstreamTurbineOne', 'threeDdownstreamTurbineTwo',
                                           'sixDdownstreamTurbineTwo')
    turblocs = [[1118.083, 1279.5, 90.], [1881.917, 1720.5, 90.]]

# Automatically define the confined domain region
if confine and confinezone is not None:
    rotbox = np.pi/6
    # For 2 parallel turbines 1 in north 1 in south
    if 'ParTurb' in casename:
        # 1st refinement zone as confinement box around parallel turbines
        if confinezone == 'first':
            boxorig = (1074.225, 599.464, 0)
            boxl, boxw, boxh = 1134, 1134, 405
            confinedfield_namesub += '1'
        # 2nd refinement zone as confinement box around parallel turbines
        elif confinezone == 'second':
            boxorig = (1120.344, 771.583, 0)
            boxl, boxw, boxh = 882, 882, 216
            confinedfield_namesub += '2'

    # For 1 turbine only
    elif casename == 'ALM_N_H_OneTurb':
        if confinezone == 'first':
            boxorig = (948.225, 817.702, 0)
            boxl, boxw, boxh = 1134, 630, 405
            confinedfield_namesub += '1'
        elif confinezone == 'second':
            boxorig = (994.344, 989.821, 0)
            boxl, boxw, boxh = 882, 378, 216
            confinedfield_namesub += '2'

    # For 2 sequential turbines 1 behind the other
    # FIXME: update
    elif 'SeqTurb' in casename:
        if confinezone == 'first':
            boxorig = (948.225, 817.702, 0)
            boxl, boxw, boxh = 1134, 630, 405
            confinedfield_namesub += '1'
        elif confinezone == 'second':
            boxorig = (994.344, 989.821, 0)
            boxl, boxw, boxh = 882, 378, 216
            confinedfield_namesub += '2'

if not confine: confinedfield_namesub = ''

# Subscript for the slice names
slicename_sub = 'Slice'

# Ensemble file name, containing fields related to ML
mlfield_ensemble_namefull = mlfield_ensemble_name + '_' + confinedfield_namesub
# Initialize case object
case = FieldData(casename=casename, casedir=casedir, times=time, fields=fields, save=save_fields, result_folder=resultfolder)


"""
Read and Process Raw Field Data
"""
if process_raw_field:
    # Read raw field data specified in fields
    field_data = case.readFieldData()
    # Initialize gradient of U as nPoint x 9 and U as n_points x 3
    grad_u, u = np.zeros((field_data[fields[0]].shape[0], 9)), np.zeros((field_data[fields[0]].shape[0], 3))
    # Initialize gradient of p_rgh and TKE as nPoint x 3
    grad_p, grad_k = (np.zeros((field_data[fields[0]].shape[0], 3)),)*2
    # Initialize k, SGS epsilon as nPoint x 0
    k, epsilon = (np.zeros(field_data[fields[0]].shape[0]),)*2
    # Go through each read (and rotated) field to assign different field to variable,
    # and also aggregate resolved and SGS fields
    for field in fields:
        # If 'k' string in current field
        # There should be 'grad_k' 'kResolved' and 'kSGSmean' available
        if 'k' in field:
            # If moreover, there's grad_kw in current field, then it's a TKE gradient field
            if grad_kw in field:
                grad_k += field_data[field]
            # Otherwise it's a TKE field
            else:
                k += field_data[field]

        # Same with epsilon, there should be 'epsilonSGSmean',
        # although currently only SGS component available
        elif 'epsilonSGS' in field:
            epsilon = field_data[field]
            
        # Same with U, there should be 'grad_UAvg'
        elif 'U' in field:
            if grad_kw in field:
                grad_u += field_data[field]
            else:
                u += field_data[field]

        # Same with p_rgh, there should be 'grad_p_rgh'
        elif 'p_rgh' in field:
            if grad_kw in field:
                grad_p += field_data[field]

        # Same with u'u', there should be 'uuprime2',
        # although no aggregation is needed since u'u' is SGS only
        # u'u' is symmetric
        elif 'uuPrime2' in field:
            uuprime2 = field_data[field]

    del field_data
    # Convert 1D array to 2D so that I can hstack them to 1 array ensemble, nCell x 1
    k, epsilon = k.reshape((-1, 1)), epsilon.reshape((-1, 1))
    # Assemble all useful fields for Machine Learning
    mlfield_ensemble = np.hstack((grad_k, k, epsilon, grad_u, u, grad_p, uuprime2))
    print('\nField variables identified and resolved and SGS TKE aggregated')
    # Read cell center coordinates of the whole domain, nCell x 0
    ccx, ccy, ccz, cc = case.readCellCenterCoordinates()


    """
    Confine the Whole Field, If confine Is True
    """
    if confine:
        # Confine to domain of interest and save confined cell centers as well as field ensemble if requested
        _, _, _, cc, mlfield_ensemble, box, _ = case.confineFieldDomain_Rotated(ccx, ccy, ccz, mlfield_ensemble,
                                                                                        box_l=boxl, box_w=boxw,
                                                                                        box_h=boxh, box_orig=boxorig,
                                                                                        rot_z=rotbox,
                                                                                        filenames_sub=confinedfield_namesub,
                                                                                        vals_name=mlfield_ensemble_name)
        del ccx, ccy, ccz
        # Update old whole fields to new confined fields
        grad_k, k = mlfield_ensemble[:, :3], mlfield_ensemble[:, 3]
        epsilon = mlfield_ensemble[:, 4]
        grad_u = mlfield_ensemble[:, 5:14]
        u = mlfield_ensemble[:, 14:17]
        grad_p = mlfield_ensemble[:, 17:20]
        uuprime2 = mlfield_ensemble[:, 20:]
        # Visualize the confined box if requested
        if plot_confinebox:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            patch = patches.PathPatch(box, facecolor='orange', lw=0)
            ax.add_patch(patch)
            ax.axis('equal')
            ax.set_xlim(0, 3000)
            ax.set_ylim(0, 3000)
            plt.show()
         
    
    """
    Rotate Fields by Given Rotation Angle
    """
    # FIXME: uuprime2 probably wrong
    # if rotz != 0.:
    #     grad_k = rotateData(grad_k, anglez=rotz)
    #     # Switch to matrix form for grad(U) and rotate
    #     grad_u = grad_u.reshape((-1, 3, 3))
    #     grad_u = rotateData(grad_u, anglez=rotz).reshape((-1, 9))
    #     u = rotateData(u, anglez=rotz)
    #     grad_p = rotateData(grad_p, anglez=rotz)
    #     # Extend symmetric tensor to full matrix form
    #     uuprime2 = expandSymmetricTensor(uuprime2).reshape((-1, 3, 3))
    #     uuprime2 = rotateData(uuprime2, anglez=rotz).reshape((-1, 9))
    #     uuprime2 = contractSymmetricTensor(uuprime2)

    if save_fields:
        k, epsilon = k.reshape((-1, 1)), epsilon.reshape((-1, 1))
        # Reassemble ML field ensemble after possible field rotation
        mlfield_ensemble = np.hstack((grad_k, k, epsilon, grad_u, u, grad_p, uuprime2))
        case.savePickleData(time, mlfield_ensemble, filenames=mlfield_ensemble_namefull)
        case.savePickleData(time, cc, filenames='CC_' + confinedfield_namesub)

    del mlfield_ensemble
# Else if directly read pickle data
else:
    if process_invariants:
        # Load rotated and/or confined field data useful for Machine Learning
        mlfield_ensemble = case.readPickleData(time, mlfield_ensemble_namefull)
        grad_k, k = mlfield_ensemble[:, :3], mlfield_ensemble[:, 3]
        epsilon = mlfield_ensemble[:, 4]
        grad_u = mlfield_ensemble[:, 5:14]
        u = mlfield_ensemble[:, 14:17]
        grad_p = mlfield_ensemble[:, 17:20]
        uuprime2 = mlfield_ensemble[:, 20:]
        # Load confined cell centers too
        cc = case.readPickleData(time, 'CC_' + confinedfield_namesub)

        del mlfield_ensemble


"""
Calculate Field Invariants
"""
if process_invariants:
    # Step 1: non-dimensional strain rate and rotation rate tensor Sij and Rij
    # epsilon is SGS epsilon as it's not necessary to use total epsilon
    # Sij shape (n_samples, 6); Rij shape (n_samples, 9)
    t0 = t.time()
    sij, rij = getStrainAndRotationRateTensor(grad_u, tke=k, eps=epsilon, cap=cap_sijrij)
    t1 = t.time()
    print('\nFinished Sij and Rij calculation in {:.4f} s'.format(t1 - t0))
    # Step 2: 10 invariant bases scaled Tij, shape (n_samples, 6, 10)
    t0 = t.time()
    tb = getInvariantBases(sij, rij, quadratic_only=False, is_scale=scale_tb)
    t1 = t.time()
    print('\nFinished Tij calculation in {:.4f} s'.format(t1 - t0))
    # Step 3: anisotropy tensor bij, shape (n_samples, 6)
    bij = case.getAnisotropyTensorField(uuprime2, use_oldshape=False)
    del uuprime2
    # Save tensor invariants related fields
    if save_fields:
        case.savePickleData(time, sij, filenames=('Sij_' + confinedfield_namesub))
        case.savePickleData(time, rij, filenames=('Rij_' + confinedfield_namesub))
        case.savePickleData(time, tb, filenames=('Tij_' + confinedfield_namesub))
        case.savePickleData(time, bij, filenames=('bij_' + confinedfield_namesub))

# Else if read invariants data from pickle
else:
    if calculate_features:
        invariants = case.readPickleData(time, filenames=('Sij_' + confinedfield_namesub,
                                                            'Rij_' + confinedfield_namesub,
                                                            'Tij_' + confinedfield_namesub,
                                                            'bij_' + confinedfield_namesub))
        sij = invariants['Sij_' + confinedfield_namesub]
        rij = invariants['Rij_' + confinedfield_namesub]
        tb = invariants['Tij_' + confinedfield_namesub]
        bij = invariants['bij_' + confinedfield_namesub]
        del invariants


"""
Calculate Feature Sets
"""
if calculate_features:
    if fs == 'grad(TKE)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
    elif fs == 'grad(p)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
    elif 'grad(TKE)_grad(p)' in fs:
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u,
                                                 grad_u=grad_u)
        # 4 additional invariant features
        if '+' in fs:
            nu *= np.ones_like(k)
            # Radial distance to (closest) turbine center.
            # Don't supply z to get horizontal radial distance
            r = getRadialTurbineDistance(cc[:, 0], cc[:, 1], z=None, turblocs=turblocs)
            fs_data2, labels2 = getSupplementaryInvariantFeatures(k, cc[:, 2], epsilon, nu, sij, r=r)
            fs_data = np.hstack((fs_data, fs_data2))
            del nu, r, fs_data2

    del sij, rij, grad_k, k, epsilon, grad_u, u, grad_p
    # If only feature set 1 used for ML input, then do train test data split here
    if save_fields:
        case.savePickleData(time, fs_data, filenames='FS_' + fs + '_' + confinedfield_namesub)

# Else, directly read feature set data
else:
    if prep_train_test_data: fs_data = case.readPickleData(time, filenames='FS_' + fs + '_' + confinedfield_namesub)


"""
Train, Test Data Preparation
"""
if prep_train_test_data:
    # X is either RANS or LES invariant features shape (n_samples, n_features)
    x = fs_data
    # y is LES bij shape (n_samples, 6)
    y = bij
    del bij
    # Prepare GS samples of specified size
    list_data_gs, _ = splitTrainTestDataList([cc, x, y, tb], test_fraction=0., seed=seed, sample_size=samples_gs)
    # Prepare training samples of specified size for actual training
    list_data_train, _ = splitTrainTestDataList([cc, x, y, tb], test_fraction=0., seed=seed, sample_size=samples_train)
    del cc, x, y, tb
    if save_fields:
        # Extra tuple treatment to list_data_* that's already a tuple since savePickleData thinks tuple means multiple files
        case.savePickleData(time, (list_data_train,), 'list_data_train_' + confinedfield_namesub)
        case.savePickleData(time, (list_data_gs,), 'list_data_GS_' + confinedfield_namesub)

# # Else if directly read GS, train and test data from pickle data
# else:
#     list_data_gs = case.readPickleData(time, 'list_data_GS_' + confinedfield_namesub)
#     list_data_train = case.readPickleData(time, 'list_data_train_' + confinedfield_namesub)


"""
Process Slices for Prediction Visualizations
"""
if process_slices:
    # Initialize case
    slice = SliceProperties(time=time, casedir=casedir, casename=casename, rot_z=rotbox, result_folder=resultfolder)
    # Read slices
    slice.readSlices(properties=fields, slicenames=slicenames, slicenames_sub=slicename_sub)
    slice_data = slice.slices_val
    # Dict storing all slice values with slice type as key, e.g. alongWind, hubHeight, etc.
    list_slicevals, list_sliceproperties, list_slicecoor = {}, {}, {}
    slicenames_iter = iter(slicenames)
    # Go through each slice type, e.g. alongWind, hubHeight, etc.
    for itype in range(len(slicenames)):
        # Get it's type, e.g. alongWind, hubHeight, etc.
        slice_type = next(slicenames_iter)
        list_slicevals[slice_type], list_sliceproperties[slice_type] = [], []
        store_slicescoor = True
        # Go through every slices incl. every type and every flow property
        for i in range(len(slice.slicenames)):
            slicename = slice.slicenames[i]
            # Skip kSGSmean since it will be discovered when kResolved or epsilonSGSmean is discovered
            if 'kSGSmean' in slicename: continue
            # If matching slice type, proceed
            if slice_type in slicename:
                val = slice.slices_val[slicename]
                # If kResolved and kSGSmean in properties, get total kMean
                # Same with grad_kResolved
                if 'kResolved' in slicename:
                    if grad_kw in slicename:
                        for i2 in range(len(slice.slicenames)):
                            slicename2 = slice.slicenames[i2]
                            if slice_type in slicename2 and 'kSGSmean' in slicename2 and grad_kw in slicename2:
                                print(' Calculating total grad(<k>) for {}...'.format(slicenames[itype]))
                                val += slice.slices_val[slicename2]
                                break

                    else:
                        for i2 in range(len(slice.slicenames)):
                            slicename2 = slice.slicenames[i2]
                            if slice_type in slicename2 and 'kSGSmean' in slicename2 and grad_kw not in slicename2:
                                print(' Calculating total <k> for {}...'.format(slicenames[itype]))
                                val += slice.slices_val[slicename2]
                                val = val.reshape((-1, 1))
                                break

                list_slicevals[slice_type].append(val)
                list_sliceproperties[slice_type].append(slicename)
                list_slicecoor[slice_type] = slice.slices_coor[slicename]
                store_slicescoor = False

        # Assign list to individual variables
        for i, name in enumerate(list_sliceproperties[slice_type]):
            if 'k' in name:
                if grad_kw in name:
                    grad_k = list_slicevals[slice_type][i]
                else:
                    k = list_slicevals[slice_type][i]

            elif 'U' in name:
                if grad_kw in name:
                    grad_u = list_slicevals[slice_type][i]
                else:
                    u = list_slicevals[slice_type][i]

            elif 'p_rgh' in name:
                if grad_kw in name:
                    grad_p = list_slicevals[slice_type][i]
                else:
                    p = list_slicevals[slice_type][i]

            # epsilon SGS
            elif 'epsilon' in name:
                epsilon = list_slicevals[slice_type][i]
            elif 'uu' in name:
                uuprime2 = list_slicevals[slice_type][i]
            elif 'G' in name:
                g_tke = list_slicevals[slice_type][i]
            elif 'divDevR' in name:
                div_devr = list_slicevals[slice_type][i]
            else:
                warn("\nError: {} slice not assigned to a variable!".format(name), stacklevel=2)

        # TODO: uuprime2 probably wrong, so not rotation for now
        # # Rotate fields if requested
        # if rotz != 0.:
        #     grad_k = rotateData(grad_k, anglez=rotz)
        #     # Switch to matrix form for grad(U) and rotate
        #     grad_u = grad_u.reshape((-1, 3, 3))
        #     grad_u = rotateData(grad_u, anglez=rotz).reshape((-1, 9))
        #     u = rotateData(u, anglez=rotz)
        #     grad_p = rotateData(grad_p, anglez=rotz)
        #     # Extend symmetric tensor to full matrix form
        #     uuprime2 = expandSymmetricTensor(uuprime2).reshape((-1, 3, 3))
        #     uuprime2 = rotateData(uuprime2, anglez=rotz).reshape((-1, 9))
        #     uuprime2 = contractSymmetricTensor(uuprime2)

        # Process invariants
        # Non-dimensional Sij (n_samples, 6) and Rij (n_samples, 9)
        # Using epsilon SGS as epsilon total is not necessary
        # according to the definition in Eq 5.65, Eq 5.66 in Sagaut (2006)
        sij, rij = getStrainAndRotationRateTensor(grad_u, tke=k, eps=epsilon, cap=cap_sijrij)
        # Step 2: 10 invariant bases scaled Tij, shape (n_samples, 6)
        # possibly with scaling of 1/[1, 10, 10, 10, 100, 100, 1000, 1000, 1000, 1000]
        tb = getInvariantBases(sij, rij, quadratic_only=False, is_scale=scale_tb)
        # Step 3: anisotropy tensor bij, shape (n_samples, 6)
        bij = case.getAnisotropyTensorField(uuprime2, use_oldshape=False)
        # # Step 4: evaluate 10 Tij coefficients g as output y
        # # g, rmse = case.evaluateInvariantBasisCoefficients(tb, bij, cap = capG, onegToRuleThemAll = False)
        # t0 = t.time()
        # g, gRMSE = evaluateInvariantBasisCoefficients(tb, bij, cap = capG)
        # t1 = t.time()
        # print('\nFinished getInvariantBasisCoefficientsField in {:.4f} s'.format(t1 - t0))
        # Save tensor invariants related fields
        if save_fields:
            case.savePickleData(time, sij, filenames=('Sij_' + slice_type))
            case.savePickleData(time, rij, filenames=('Rij_' + slice_type))
            case.savePickleData(time, tb, filenames=('Tij_' + slice_type))
            case.savePickleData(time, bij, filenames=('bij_' + slice_type))
            # Visualization related slice data
            case.savePickleData(time, k, 'TKE_' + slice_type)
            case.savePickleData(time, grad_u, 'grad(U)_' + slice_type)
            case.savePickleData(time, g_tke, 'G_' + slice_type)
            case.savePickleData(time, div_devr, 'div(dev(R))_' + slice_type)
            case.savePickleData(time, list_slicecoor[slice_type], filenames='CC_' + slice_type)

        # Calculate features
        if fs == 'grad(TKE)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
        elif fs == 'grad(p)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
        elif 'grad(TKE)_grad(p)' in fs:
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u,
                                                     grad_u=grad_u)
            # 4 additional invariant features
            if '+' in fs:
                nulist = nu*np.ones_like(k)
                # Radial distance to (closest) turbine center.
                # Don't supply z to get horizontal radial distance
                r = getRadialTurbineDistance(list_slicecoor[slice_type][:, 0], list_slicecoor[slice_type][:, 1], z=None, turblocs=turblocs)
                fs_data2, labels2 = getSupplementaryInvariantFeatures(k, list_slicecoor[slice_type][:, 2], epsilon, nulist, sij, r=r)
                fs_data = np.hstack((fs_data, fs_data2))
                del nulist, r, fs_data2

        # If only feature set 1 used for ML input, then do train test data split here
        if save_fields:
            case.savePickleData(time, fs_data, filenames=('FS_' + fs + '_' + slice_type))

        # Test data preparation
        x = fs_data
        # y is LES bij shape (n_samples, 6)
        y = bij
        # Prepare GS samples of specified size
        # list_data_test, _ = splitTrainTestDataList([list_slicecoor[slice_type], x, y, tb], test_fraction=0., seed=seed,
        #                                          sample_size=None)
        list_data_test = [list_slicecoor[slice_type], x, y, tb]
        if save_fields:
            case.savePickleData(time, (list_data_test,), 'list_data_test_' + slice_type)


"""
Process Sets If Requested for Prediction & Visualization
"""
if process_sets:
    setcase = SetProperties(casename=casename, casedir=casedir, time=time)
    setcase.readSets()
    # Go through each line type
    for set_type in set_types:
        # Go through each line full name
        for set in setcase.sets:
            # If line type in full name
            if set_type in set:
                if '_kSGSmean' in set: continue
                # If velocity name in full name
                elif '_U' in set:
                    if grad_kw in set:
                        grad_u = setcase.data[set]
                    else:
                        u = setcase.data[set]
                        distance = setcase.coor[set]

                elif '_kResolved' in set:
                    if grad_kw in set:
                        grad_k = setcase.data[set]
                        for set2 in setcase.sets:
                            if set_type in set2 and (grad_kw + '_kSGSmean') in set2:
                                print(' Calculating total grad(<k>) for {}...'.format(set_type))
                                grad_k += setcase.data[set2]

                    else:
                        k = setcase.data[set]
                        for set2 in setcase.sets:
                            if set_type in set2 and '_kSGSmean' in set2 and grad_kw not in set2:
                                print(' Calculating total <k> for {}...'.format(set_type))
                                k += setcase.data[set2]

                elif '_p_rgh' in set:
                    if grad_kw in set:
                        grad_p = setcase.data[set]
                    else:
                        g = setcase.data[set]

                elif '_eps' in set:
                    epsilon = setcase.data[set]
                elif '_uu' in set:
                    uuprime2 = setcase.data[set]
                elif '_G' in set:
                    g_tke = setcase.data[set]
                elif '_divDevR' in set:
                    div_devr = setcase.data[set]
                else:
                    warn("Error: {} set not assigned to a variable!".format(set), stacklevel=2)

        sij, rij = getStrainAndRotationRateTensor(grad_u, tke=k, eps=epsilon, cap=cap_sijrij)
        tb = getInvariantBases(sij, rij, quadratic_only=False, is_scale=scale_tb)
        bij = case.getAnisotropyTensorField(uuprime2, use_oldshape=False)
        if save_fields:
            case.savePickleData(time, sij, filenames=('Sij_' + set_type))
            case.savePickleData(time, rij, filenames=('Rij_' + set_type))
            case.savePickleData(time, tb, filenames=('Tij_' + set_type))
            case.savePickleData(time, bij, filenames=('bij_' + set_type))
            # Save (purely) visualization related sets
            case.savePickleData(time, k, 'TKE_' + set_type)
            case.savePickleData(time, grad_u, 'grad(U)_' + set_type)
            case.savePickleData(time, g_tke, 'G_' + set_type)
            case.savePickleData(time, div_devr, 'div(dev(R))_' + set_type)
            case.savePickleData(time, distance, filenames='CC_' + set_type)

        # Calculate features
        if fs == 'grad(TKE)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
        elif fs == 'grad(p)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
        elif 'grad(TKE)_grad(p)' in fs:
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u,
                                                     grad_u=grad_u)
            # 4 additional invariant features
            if '+' in fs:
                nulist = nu*np.ones_like(k)
                # If horizontal line
                if '_H' in set_type:
                    # Get the origion coordinates in x, y
                    if 'oneDdownstreamTurbine' in set_type:
                        # If 2nd turbine in SeqTurb.
                        # Else if ParTurb or OneTurb or 1st turbine in SeqTurb, origin is the same
                        orig = (1288.69, 3000.) if 'Two' in set_type else (270.244, 3000.)
                    elif 'threeDdownstreamTurbine' in set_type:
                        orig = (1579.674, 3000.) if 'Two' in set_type else (561.228, 3000.)
                    # 6D downstream only exists for 2nd turbine in SeqTurb
                    elif 'sixDdownstreamTurbineTwo' in set_type:
                        orig = (2016.151, 3000.)
                    # 7D downstream only exists in ParTurb and OneTurb
                    elif 'sevenDdownstreamTurbine' in set_type:
                        orig = (1143.198, 3000.)

                    xline = orig[0] + distance*np.sin(rotz)
                    yline = orig[1] - distance*np.cos(rotz)
                    zline = np.ones_like(distance)*90.
                # Else if vertical line
                else:
                    if 'oneDdownstreamTurbine' in set_type:
                        if 'Two' in set_type:
                            xline = np.ones_like(distance)*1991.036
                            yline = np.ones_like(distance)*1783.5
                        elif 'Southern' in set_type:
                            xline = np.ones_like(distance)*1353.202
                            yline = np.ones_like(distance)*1124.262
                        elif 'Northern' in set_type:
                            xline = np.ones_like(distance)*1101.202
                            yline = np.ones_like(distance)*1560.738
                        # 1st turbine in SeqTurb is same as OneTurb
                        else:
                            xline = np.ones_like(distance)*1227.702
                            yline = np.ones_like(distance)*1342.5

                    elif 'threeDdownstreamTurbine' in set_type:
                        if 'Two' in set_type:
                            xline = np.ones_like(distance)*2209.275
                            yline = np.ones_like(distance)*1909.5
                        elif 'Southern' in set_type:
                            xline = np.ones_like(distance)*1571.44
                            yline = np.ones_like(distance)*1250.262
                        elif 'Northern' in set_type:
                            xline = np.ones_like(distance)*1319.44
                            yline = np.ones_like(distance)*1686.738
                        else:
                            xline = np.ones_like(distance)*1445.44
                            yline = np.ones_like(distance)*1468.5

                    elif 'fiveDdownstreamTurbine' in set_type:
                        if 'Southern' in set_type:
                            xline = np.ones_like(distance)*1789.679
                            yline = np.ones_like(distance)*1376.262
                        elif 'Northern' in set_type:
                            xline = np.ones_like(distance)*1537.679
                            yline = np.ones_like(distance)*1812.738
                        else:
                            xline = np.ones_like(distance)*1663.679
                            yline = np.ones_like(distance)*1594.5

                    # Only for 2nd turbine in SeqTurb
                    elif 'sixDdownstreamTurbineTwo' in set_type:
                        xline = np.ones_like(distance)*2536.632
                        yline = np.ones_like(distance)*2098.5
                    elif 'sevenDdownstreamTurbine' in set_type:
                        if 'Southern' in set_type:
                            xline = np.ones_like(distance)*2007.917
                            yline = np.ones_like(distance)*1502.262
                        elif 'Northern' in set_type:
                            xline = np.ones_like(distance)*1755.917
                            yline = np.ones_like(distance)*1938.738
                        # Otherwise for OneTurb
                        else:
                            xline = np.ones_like(distance)*1881.917
                            yline = np.ones_like(distance)*1720.5

                    zline = distance

                # Radial distance to (closest) turbine center.
                # Again, don't supply z to get horizontal radial distance
                r = getRadialTurbineDistance(xline, yline, z=None, turblocs=turblocs)
                fs_data2, labels2 = getSupplementaryInvariantFeatures(k, zline, epsilon, nulist, sij, r=r)
                fs_data = np.hstack((fs_data, fs_data2))
                del nulist, r, fs_data2

        if save_fields:
            case.savePickleData(time, fs_data, filenames=('FS_' + fs + '_' + set_type))

        x = fs_data
        y = bij
        # list_data_test, _ = splitTrainTestDataList([distance, x, y, tb], test_fraction=0., seed=seed,
        #                                            sample_size=None)
        list_data_test = [distance, x, y, tb]
        if save_fields:
            case.savePickleData(time, (list_data_test,), 'list_data_test_' + set_type)

















