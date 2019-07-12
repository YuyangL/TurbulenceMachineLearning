import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
from PostProcess_SliceData import SliceProperties
from Preprocess.Tensor import processReynoldsStress
from Preprocess.Feature import getInvariantFeatureSet
from Preprocess.FeatureExtraction import splitTrainTestDataList

# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from warnings import warn

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case
casename = 'ALM_N_H_OneTurb'  # str
# Absolute directory of this flow case
casedir = '/media/yluan'  # str
# Which time to extract input and output for ML
time = 'last'  # str/float/int or 'last'
# Interpolation method when interpolating mesh grids
interp_method = "nearest"  # "nearest", "linear", "cubic"
# What keyword does the gradient fields contain
gradFieldKW = 'grad'  # str
# Slice names for prediction visualization
sliceNames = ('alongWind', 'hubHeight', 'quarterDaboveHub', 'turbineApexHeight',
              'twoDupstreamTurbine', 'rotorPlane', 'oneDdownstreamTurbine', 'threeDdownstreamTurbine', 'sevenDdownstreamTurbine')
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# Flow field counter-clockwise rotation in x-y plane
# so that tensor fields are parallel/perpendicular to flow direction
fieldRot = np.pi/6  # float [rad]
# When fieldRot is not 0, whether to infer spatial correlation fields and rotate them
# Otherwise the name of the fields needs to be specified
# Strain rate / rotation rate tensor fields not supported
if fieldRot != 0.:
    spatialCorrelationFields = ('infer',)  # list/tuple(str) or list/tuple('infer')
    spatial_corr_slices = ('infer',)

# Whether confine and visualize the domain of interest, useful if mesh is too large
confineBox, plotConfinedBox = True, True  # bool; bool
# Only when confineBox is True:
if confineBox:
    # Subscript of the confined field file name
    confinedFieldNameSub = 'Confined'
    # Whether auto generate confine box for each case,
    # the confinement is bordered by
    # 'first': 1st refinement zone
    # 'second': 2nd refined zone
    # 'third': left half of turbine in 2nd refinement zone
    boxAutoDim = 'second'  # 'first', 'second', 'third', None
    # Only when boxAutoDim is False:
    if boxAutoDim is None:
        # Confine box counter-clockwise rotation in x-y plane
        boxRot = np.pi/6  # float [rad]
        # Confine box origin, width, length, height
        boxOrig = (0, 0, 0)  # (x, y, z)
        boxL, boxW, boxH = 0, 0, 0  # float

# Absolute cap value for Sij and Rij; TB coefficients' basis;  and TB coefficients
cap_sijrij, cap_tij = 1e9, 1e9  # float/int
# Enforce 0 trace when computing Tij?
tij_0trace = False

# Save anything when possible
save_fields, resultFolder = True, 'Result'  # bool; str


"""
Machine Learning Settings
"""
# Whether to calculate features or directly read from pickle data
calculate_features = False  # bool
# Whether to split train and test data or directly read from pickle data
prep_train_test_data = False  # bool
# Number of samples for Grid Search before training on all data
# Also number of samples to do ML, if None, all samples are used for training
samples_gs, samples_train = 10000, 2e6  # int; int, None
# Fraction of data for testing
test_fraction = 0.  # float [0-1]
# Feature set choice
fs = 'grad(TKE)_grad(p)'  # 'grad(TKE)', 'grad(p)', 'grad(TKE)_grad(p)'
# Use seed for reproducibility, set to None for no seeding
seed = 123  # int, None
# Fraction of data used for testing
testSize = 0.2  # float


"""
Visualization Settings
"""
# Whether to process slices and save them for prediction visualization later
# TODO: maybe try without rotation
process_slices = True  # bool


"""
Process User Inputs, No Need to Change
"""
# Average fields of interest for reading and processing
if fs == 'grad(TKE)_grad(p)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'grad_kResolved', 'grad_kSGSmean', 'UAvg') 
elif fs == 'grad(TKE)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_kResolved', 'grad_kSGSmean')
elif fs == 'grad(p)':
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg', 'grad_p_rghAvg', 'UAvg')
else:
    fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
              'grad_UAvg')


# Ensemble name of fields useful for Machine Learning
mlFieldEnsembleName = 'ML_Fields_' + casename
# Automatically select time if time is set to 'latest'
if time == 'last':
    if casename == 'ALM_N_H_ParTurb':
        time = '22000.0918025'
    elif casename == 'ALM_N_H_OneTurb':
        time = '24995.0788025'
else:
    time = str(time)

# Automatically define the confined domain region
if confineBox and boxAutoDim is not None:
    boxRot = np.pi/6
    if casename == 'ALM_N_H_ParTurb':
        # 1st refinement zone as confinement box
        if boxAutoDim == 'first':
            boxOrig = (1074.225, 599.464, 0)
            boxL, boxW, boxH = 1134, 1134, 405
            confinedFieldNameSub += '1'
        # 2nd refinement zone as confinement box
        elif boxAutoDim == 'second':
            boxOrig = (1120.344, 771.583, 0)
            boxL, boxW, boxH = 882, 378, 216
            confinedFieldNameSub += '2'
        elif boxAutoDim == 'third':
            boxOrig = (1120.344, 771.583, 0)
            boxL, boxW, boxH = 882, 378/2., 216
            confinedFieldNameSub += '2'

    elif casename == 'ALM_N_H_OneTurb':
        if boxAutoDim == 'first':
            boxOrig = (948.225, 817.702, 0)
            boxL, boxW, boxH = 1134, 630, 405
            confinedFieldNameSub += '1'
        elif boxAutoDim == 'second':
            boxOrig = (994.344, 989.583, 0)
            boxL, boxW, boxH = 882, 378, 216
            confinedFieldNameSub += '2'
        elif boxAutoDim == 'third':
            boxOrig = (994.344, 989.583, 0)
            boxL, boxW, boxH = 882, 378/2., 216
            confinedFieldNameSub += '3'

if not confineBox:
    confinedFieldNameSub = ''

# Subscript for the slice names
sliceNamesSub = 'Slice'

# Ensemble file name, containing fields related to ML
mlFieldEnsembleNameFull = mlFieldEnsembleName + '_' + confinedFieldNameSub
# Initialize case object
case = FieldData(caseName=casename, caseDir=casedir, times=time, fields=fields, save=save_fields, resultFolder=resultFolder)


"""
Read and Process Raw Field Data
"""
if process_raw_field:
    # Read raw field data specified in fields
    fieldData = case.readFieldData()
    # Rotate fields if fieldRot is not 0
    if fieldRot != 0.:
        listData, uuPrime2 = [], []
        # If spatialCorrelationFields is list/tuple('infer'),
        # infer which field is a spatial single/double correlation for rotation
        if spatialCorrelationFields[0] == 'infer':
            # Redefine spatialCorrelationFields
            spatialCorrelationFields = []
            # Go through each item in fieldData dictionary
            for field, data in fieldData.items():
                # Pick up what ever is 2D since it means multiple components (columns) exist.
                # Exclude u'u' and do it manually later
                if len(data.shape) == 2:
                    if 'uu' not in field:
                        listData.append(data)
                        spatialCorrelationFields.append(field)
                    else:
                        uuPrime2 = data
                        uuPrime2_name = field

        # Else if spatialCorrelationFields is provided
        else:
            for field in spatialCorrelationFields:
                listData.append(fieldData[field])

        # The provided tensor field should be single correlation, i.e. x instead of xx
        listData = case.rotateSpatialCorrelationTensors(listData, rotateXY=fieldRot, dependencies='x')
        for i, field in enumerate(spatialCorrelationFields):
            fieldData[field] = listData[i]

        if len(uuPrime2) != 0:
            uuPrime2 = case.rotateSpatialCorrelationTensors(uuPrime2, rotateXY=fieldRot, dependencies='xx')
            uuPrime2flag = True

    # Initialize gradient of U as nPoint x 9 and U as n_points x 3
    grad_u, u = np.zeros((fieldData[fields[0]].shape[0], 9)), np.zeros((fieldData[fields[0]].shape[0], 3))
    # Initialize gradient of p_rgh and TKE as nPoint x 3
    grad_p, grad_k = (np.zeros((fieldData[fields[0]].shape[0], 3)),)*2
    # Initialize k, SGS epsilon, SGS nu, and p_rgh as nPoint x 0
    k, epsilon, nuSGS = (np.zeros(fieldData[fields[0]].shape[0]),)*3
    # Go through each read (and rotated) field to assign different field to variable,
    # and also aggregate resolved and SGS fields
    # Also keep track which field has been provided and aggregated
    grad_kflag, kFlag, epsFlag, nuFlag, grad_uflag, uFlag, grad_pFlag \
        = (False,)*7
    ij9to6 = (0, 1, 2, 4, 5, 8)
    for field in fields:
        # If 'k' string in current field
        # There should be 'grad_k' 'kResolved' and 'kSGSmean' available
        if 'k' in field:
            # If moreover, there's gradFieldKW in current field, then it's a TKE gradient field
            if gradFieldKW in field:
                grad_k += fieldData[field]
                grad_kflag = True
            # Otherwise it's a TKE field
            else:
                k += fieldData[field]
                kFlag = True

        # Same with epsilon, there should be 'epsilonSGSmean',
        # although currently only SGS component available
        elif 'epsilonSGS' in field:
            epsilon = fieldData[field]
            epsSGSflag = True

        # Same with nu, there should be 'nuSGSmean',
        # although only SGS component needed to later evaluate total epsilon
        elif 'nuSGS' in field:
            nuSGS = fieldData[field]
            nuSGSflag = True

        # Same with U, there should be 'grad_UAvg'
        elif 'U' in field:
            if gradFieldKW in field:
                grad_u += fieldData[field]
                grad_uflag = True
            else:
                u += fieldData[field]
                uFlag = True

        # Same with p_rgh, there should be 'grad_p_rgh'
        elif 'p_rgh' in field:
            if gradFieldKW in field:
                grad_p += fieldData[field]
                grad_pFlag = True

        # # Same with u'u', there should be 'uuPrime2',
        # # although no aggregation is needed since u'u' is SGS only
        # elif 'uuPrime2' in field:
        #     uuPrime2 = fieldData[field][:, ij9to6]
        #     # # Expand symmetric tensor to it's full form, nCell x 9
        #     # # From xx, xy, xz
        #     # #          yy, yz
        #     # #              zz
        #     # # to xx, xy, xz
        #     # #    yx, yy, yz
        #     # #    zx, zy, zz
        #     # uuPrime2 = np.vstack((fieldData[field][:, 0], fieldData[field][:, 1], fieldData[field][:, 2],
        #     #                       fieldData['uuPrime2'][:, 1], fieldData[field][:, 3], fieldData[field][:, 4],
        #     #                       fieldData[field][:, 2], fieldData[field][:, 4],
        #     #                       fieldData[field][:, 5])).T
        #     uuPrime2flag = True

    # # Collect all flags
    # fieldFlags = (grad_kflag, kFlag, epsFlag, nuFlag, grad_uflag, uFlag, grad_pFlag, prghFlag, uuPrime2flag)
    # Useful flags for Machine Learning
    mlFieldFlags = (grad_kflag, kFlag, epsFlag, grad_uflag, grad_pFlag, uuPrime2flag)
    # Since epsilon is only SGS temporal mean,
    # calculate total temporal mean using SGS epsilon and SGS nu
    if all((epsFlag, nuFlag)):
        epsilon = case.getMeanDissipationRateField(epsilonSGSmean=epsilon, nuSGSmean=nuSGS)
        epsMax, epsMin = np.amax(epsilon), np.amin(epsilon)
        print(' Max of epsilon is {0}; min of epsilon is {1}'.format(epsMax, epsMin))

    # Convert 1D array to 2D so that I can hstack them to 1 array ensemble, nCell x 1
    k, epsilon = k.reshape((-1, 1)), epsilon.reshape((-1, 1))
    # Assemble all useful fields for Machine Learning
    mlFieldEnsemble = np.hstack((grad_k, k, epsilon, grad_u, u, grad_p, uuPrime2))
    print('\nField variables identified and resolved and SGS properties aggregated')
    # Read cell center coordinates of the whole domain, nCell x 0
    ccx, ccy, ccz, cc = case.readCellCenterCoordinates()


    """
    Confine the Whole Field, If confineBox Is True
    """
    if confineBox:
        # Confine to domain of interest and save confined cell centers as well as field ensemble if requested
        _, _, _, cc, mlFieldEnsemble, box, _ = case.confineFieldDomain_Rotated(ccx, ccy, ccz, mlFieldEnsemble,
                                                                                        boxL=boxL, boxW=boxW,
                                                                                        boxH=boxH, boxO=boxOrig,
                                                                                        boxRot=boxRot,
                                                                                        fileNameSub=confinedFieldNameSub,
                                                                                        valsName=mlFieldEnsembleName)
        del ccx, ccy, ccz
        # Update old whole fields to new confined fields
        grad_k, k = mlFieldEnsemble[:, :3], mlFieldEnsemble[:, 3]
        epsilon = mlFieldEnsemble[:, 4]
        grad_u = mlFieldEnsemble[:, 5:14]
        u = mlFieldEnsemble[:, 14:17]
        grad_p = mlFieldEnsemble[:, 17:20]
        uuPrime2 = mlFieldEnsemble[:, 20:]

        # Visualize the confined box if requested
        if plotConfinedBox:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            patch = patches.PathPatch(box, facecolor = 'orange', lw = 0)
            ax.add_patch(patch)
            ax.axis('equal')
            ax.set_xlim(0, 3000)
            ax.set_ylim(0, 3000)
            plt.show()

    # Else if not confining domain, save all whole fields and cell centers
    if save_fields:
        case.savePickleData(time, mlFieldEnsemble, fileNames = mlFieldEnsembleNameFull)
        case.savePickleData(time, cc, fileNames = 'cc_' + confinedFieldNameSub)

# # Else if directly read pickle data
# else:
#     # Load rotated and/or confined field data useful for Machine Learning
#     mlFieldEnsemble = case.readPickleData(time, mlFieldEnsembleNameFull)
#     grad_k, k = mlFieldEnsemble[:, :3], mlFieldEnsemble[:, 3]
#     epsilon = mlFieldEnsemble[:, 4]
#     grad_u = mlFieldEnsemble[:, 5:14]
#     u = mlFieldEnsemble[:, 14:17]
#     grad_p = mlFieldEnsemble[:, 17:20]
#     uuPrime2 = mlFieldEnsemble[:, 20:]
#     # Load confined cell centers too
#     cc = case.readPickleData(time, 'cc_' + confinedFieldNameSub)
#
# del mlFieldEnsemble


"""
Calculate Field Invariants
"""
if process_invariants:
    # Step 1: strain rate and rotation rate tensor Sij and Rij
    # Sij shape (n_samples, 6); Rij shape (n_samples, 9)
    sij, rij = case.getStrainAndRotationRateTensorField(grad_u, tke=k, eps=epsilon, cap=cap_sijrij)
    # Step 2: 10 invariant bases TB, shape (n_samples, 6)
    tb = case.getInvariantBasesField(sij, rij, quadratic_only=False, is_scale=True)
    # Step 3: anisotropy tensor bij, shape (n_samples, 6)
    bij = case.getAnisotropyTensorField(uuPrime2, use_oldshape=False)
    del uuPrime2
    # # Step 4: evaluate 10 TB coefficients g as output y
    # # g, rmse = case.evaluateInvariantBasisCoefficients(tb, bij, cap = capG, onegToRuleThemAll = False)
    # t0 = t.time()
    # g, gRMSE = evaluateInvariantBasisCoefficients(tb, bij, cap = capG)
    # t1 = t.time()
    # print('\nFinished getInvariantBasisCoefficientsField in {:.4f} s'.format(t1 - t0))
    # Save tensor invariants related fields
    if save_fields:
        case.savePickleData(time, sij, fileNames = ('Sij_' + confinedFieldNameSub))
        case.savePickleData(time, rij, fileNames = ('Rij_' + confinedFieldNameSub))
        case.savePickleData(time, tb, fileNames = ('TB_' + confinedFieldNameSub))
        case.savePickleData(time, bij, fileNames = ('bij_' + confinedFieldNameSub))
        # case.savePickleData(time, g, fileNames = ('g_' + confinedFieldNameSub))
        # case.savePickleData(time, gRMSE, fileNames = ('g_RMSE_True_' + confinedFieldNameSub))

# # Else if read invariants data from pickle
# else:
#     invariants = case.readPickleData(time, fileNames = ('Sij_' + confinedFieldNameSub,
#                                                         'Rij_' + confinedFieldNameSub,
#                                                         'TB_' + confinedFieldNameSub,
#                                                         'bij_' + confinedFieldNameSub))
#     sij = invariants['Sij_' + confinedFieldNameSub]
#     rij = invariants['Rij_' + confinedFieldNameSub]
#     tb = invariants['TB_' + confinedFieldNameSub]
#     bij = invariants['bij_' + confinedFieldNameSub]
#     # g = invariants['g_' + confinedFieldNameSub]
#     del invariants


"""
Calculate Feature Sets
"""
if calculate_features:
    if fs == 'grad(TKE)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
    elif fs == 'grad(p)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
    elif fs == 'grad(TKE)_grad(p)':
        fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u,
                                                 grad_u=grad_u)

    del sij, rij, grad_k, k, epsilon, grad_u, u, grad_p
    # If only feature set 1 used for ML input, then do train test data split here
    if save_fields:
        case.savePickleData(time, fs_data, fileNames=('FS_' + fs + '_' + confinedFieldNameSub))

# # Else, directly read feature set data
# else:
#     fs_data = case.readPickleData(time, fileNames=('FS_' + fs + '_' + confinedFieldNameSub))


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
        case.savePickleData(time, (list_data_train,), 'list_data_train_' + confinedFieldNameSub)
        case.savePickleData(time, (list_data_gs,), 'list_data_GS_' + confinedFieldNameSub)

# # Else if directly read GS, train and test data from pickle data
# else:
#     list_data_gs = case.readPickleData(time, 'list_data_GS_' + confinedFieldNameSub)
#     list_data_train = case.readPickleData(time, 'list_data_train_' + confinedFieldNameSub)


"""
Process Slices for Prediction Visualizations
"""
if process_slices:
    # Initialize case
    slice = SliceProperties(time=time, caseDir=casedir, caseName=casename, xOrientate=boxRot, resultFolder=resultFolder)
    # Read slices
    slice.readSlices(propertyNames=fields, sliceNames=sliceNames, sliceNamesSub=sliceNamesSub)
    slice_data = slice.slicesVal
    # Rotate fields if fieldRot is not 0
    if fieldRot != 0.:
        listData, list_uuPrime2, uuPrime2_names = [], [], []
        # If spatialCorrelationFields is list/tuple('infer'),
        # infer which field is a spatial single/double correlation for rotation
        if spatial_corr_slices[0] == 'infer':
            # Redefine spatialCorrelationFields
            spatial_corr_slices = []
            # Go through each item in fieldData dictionary
            for field, data in slice_data.items():
                # Pick up what ever is 2D since it means multiple components (columns) exist.
                # Exclude u'u' and do it manually later
                if len(data.shape) == 2:
                    if 'uu' not in field:
                        listData.append(data)
                        spatial_corr_slices.append(field)
                    else:
                        list_uuPrime2.append(data)
                        uuPrime2_names.append(field)

        # Else if spatialCorrelationFields is provided
        else:
            for field in spatial_corr_slices:
                listData.append(slice_data[field])

        # The provided tensor field should be single correlation, i.e. x instead of xx
        listData = case.rotateSpatialCorrelationTensors(listData, rotateXY=fieldRot, dependencies='x')
        for i, field in enumerate(spatialCorrelationFields):
            slice_data[field] = listData[i]

        if len(list_uuPrime2) != 0:
            list_uuPrime2 = case.rotateSpatialCorrelationTensors(list_uuPrime2, rotateXY=fieldRot, dependencies='xx')
            uuPrime2flag = True
            for i, name in enumerate(uuPrime2_names):
                slice_data[field] = list_uuPrime2[i]

    # Dict storing all slice values with slice type as key, e.g. alongWind, hubHeight, etc.
    list_slicevals, list_sliceproperties, list_slicecoor = {}, {}, {}
    slicenames_iter = iter(sliceNames)
    # Go through each slice type, e.g. alongWind, hubHeight, etc.
    for itype in range(len(sliceNames)):
        # Get it's type, e.g. alongWind, hubHeight, etc.
        slice_type = next(slicenames_iter)
        list_slicevals[slice_type], list_sliceproperties[slice_type] = [], []
        store_slicescoor = True
        # Go through every slices incl. every type and every flow property
        for i in range(len(slice.sliceNames)):
            sliceName = slice.sliceNames[i]
            # Skip kSGSmean and nuSGSmean since they will be discovered when kResolved or epsilonSGSmean is discovered
            if 'kSGSmean' in sliceName or 'nuSGSmean' in sliceName:
                continue

            # If matching slice type, proceed
            if slice_type in sliceName:
                vals2D = slice.slicesVal[sliceName]
                # If kResolved and kSGSmean in propertyNames, get total kMean
                # Same with grad_kResolved
                if 'kResolved' in sliceName:
                    if gradFieldKW in sliceName:
                        for i2 in range(len(slice.sliceNames)):
                            sliceName2 = slice.sliceNames[i2]
                            if slice_type in sliceName2 and 'kSGSmean' in sliceName2 and gradFieldKW in sliceName2:
                                print(' Calculating total grad(<k>) for {}...'.format(sliceNames[itype]))
                                vals2D += slice.slicesVal[sliceName2]
                                break

                    else:
                        for i2 in range(len(slice.sliceNames)):
                            sliceName2 = slice.sliceNames[i2]
                            if slice_type in sliceName2 and 'kSGSmean' in sliceName2 and gradFieldKW not in sliceName2:
                                print(' Calculating total <k> for {}...'.format(sliceNames[itype]))
                                vals2D += slice.slicesVal[sliceName2]
                                vals2D = vals2D.reshape((-1, 1))
                                break

                # Else if epsilonSGSmean and nuSGSmean in propertyNames then get total epsilonMean
                # By assuming isotropic homogeneous turbulence and
                # <epsilon> = <epsilonSGS>/(1 - 1/(1 + <nuSGS>/nu))
                elif 'epsilonSGSmean' in sliceName:
                    for i2 in range(len(slice.sliceNames)):
                        sliceName2 = slice.sliceNames[i2]
                        if slice_type in sliceName2 and 'nuSGSmean' in sliceName2:
                            print(' Calculating total <epsilon> for {}...'.format(sliceNames[itype]))
                            vals2D_2 = slice.slicesVal[sliceName2]
                            # Calculate epsilonMean
                            vals2D = slice.calcSliceMeanDissipationRate(epsilonSGSmean=vals2D, nuSGSmean=vals2D_2)
                            vals2D = vals2D.reshape((-1, 1))

                list_slicevals[slice_type].append(vals2D)
                list_sliceproperties[slice_type].append(sliceName)
                list_slicecoor[slice_type] = slice.slicesCoor[sliceName]
                store_slicescoor = False

        # # Confine flow properties
        # z = np.zeros((list_slicecoor[slice_type].shape[0]))
        # _, _, _, list_slicecoor[slice_type], list_slicevals[slice_type], box, _ \
        #     = case.confineFieldDomain_Rotated(list_slicecoor[slice_type][0], list_slicecoor[slice_type][1], [0], list_slicevals[slice_type],
        #                                                                            boxL=boxL, boxW=boxW,
        #                                                                            boxH=0., boxO=boxOrig,
        #                                                                            boxRot=boxRot,
        #                                                                            fileNameSub=confinedFieldNameSub,
        #                                                                            valsName=slice_type)

        # Assign list to individual variables
        for i, name in enumerate(list_sliceproperties[slice_type]):
            if 'k' in name:
                if gradFieldKW in name:
                    grad_k = list_slicevals[slice_type][i]
                else:
                    k = list_slicevals[slice_type][i]

            elif 'U' in name:
                if gradFieldKW in name:
                    grad_u = list_slicevals[slice_type][i]
                else:
                    u = list_slicevals[slice_type][i]

            elif 'p_rgh' in name:
                if gradFieldKW in name:
                    grad_p = list_slicevals[slice_type][i]
                else:
                    p = list_slicevals[slice_type][i]

            elif 'epsilon' in name:
                epsilon = list_slicevals[slice_type][i]
            elif 'uu' in name:
                uuprime2 = list_slicevals[slice_type][i]
            else:
                warn("\nError: {} not assigned to a variable!".format(name), stacklevel=2)

        # Process invariants
        sij, rij = case.getStrainAndRotationRateTensorField(grad_u, tke=k, eps=epsilon, cap=cap_sijrij)
        # Step 2: 10 invariant bases TB, shape (n_samples, 6)
        tb = case.getInvariantBasesField(sij, rij, quadratic_only=False, is_scale=True)
        # Step 3: anisotropy tensor bij, shape (n_samples, 6)
        bij = case.getAnisotropyTensorField(uuprime2, use_oldshape=False)
        # # Step 4: evaluate 10 TB coefficients g as output y
        # # g, rmse = case.evaluateInvariantBasisCoefficients(tb, bij, cap = capG, onegToRuleThemAll = False)
        # t0 = t.time()
        # g, gRMSE = evaluateInvariantBasisCoefficients(tb, bij, cap = capG)
        # t1 = t.time()
        # print('\nFinished getInvariantBasisCoefficientsField in {:.4f} s'.format(t1 - t0))
        # Save tensor invariants related fields
        if save_fields:
            case.savePickleData(time, sij, fileNames=('Sij_' + slice_type))
            case.savePickleData(time, rij, fileNames=('Rij_' + slice_type))
            case.savePickleData(time, tb, fileNames=('TB_' + slice_type))
            case.savePickleData(time, bij, fileNames=('bij_' + slice_type))
            case.savePickleData(time, list_slicecoor[slice_type], fileNames='cc_' + slice_type)

        # Calculate features
        if fs == 'grad(TKE)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k, k=k, eps=epsilon)
        elif fs == 'grad(p)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_p=grad_p, u=u, grad_u=grad_u)
        elif fs == 'grad(TKE)_grad(p)':
            fs_data, labels = getInvariantFeatureSet(sij, rij, grad_k=grad_k, grad_p=grad_p, k=k, eps=epsilon, u=u,
                                                     grad_u=grad_u)

        # If only feature set 1 used for ML input, then do train test data split here
        if save_fields:
            case.savePickleData(time, fs_data, fileNames=('FS_' + fs + '_' + slice_type))

        # Test data preparation
        x = fs_data
        # y is LES bij shape (n_samples, 6)
        y = bij
        # Prepare GS samples of specified size
        list_data_test, _ = splitTrainTestDataList([list_slicecoor[slice_type][:2], x, y, tb], test_fraction=0., seed=seed,
                                                 sample_size=None)
        if save_fields:
            case.savePickleData(time, (list_data_test,), 'list_data_test_' + slice_type)


















