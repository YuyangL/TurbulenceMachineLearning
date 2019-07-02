import numpy as np
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
from PostProcess_FieldData import FieldData
from Preprocess.FeatureExtraction import getFeatureSet1

# For Python 2.7, use cpickle
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt

"""
User Inputs, Anything Can Be Changed Here
"""
# Name of the flow case
caseName = 'ALM_N_H_OneTurb'  # str
# Absolute directory of this flow case
caseDir = '/media/yluan'  # str
# Which time to extract input and output for ML
time = 'last'  # str/float/int or 'last'
# Average fields of interest for reading and processing
fields = ('kResolved', 'kSGSmean', 'epsilonSGSmean', 'nuSGSmean', 'uuPrime2',
          'grad_UAvg', 'grad_p_rghAvg', 'grad_kResolved', 'grad_kSGSmean')  # list/tuple(str)
# What keyword does the gradient fields contain
gradFieldKW = 'grad'  # str
# Whether process field data, invariants, features from scratch,
# or use raw field pickle data and process invariants and features
# or use raw field and invariants pickle data and process features
process_raw_field, process_invariants = False, False  # bool
# The following is only relevant is process_raw_field is True
if process_raw_field:
    # Flow field counter-clockwise rotation in x-y plane
    # so that tensor fields are parallel/perpendicular to flow direction
    fieldRot = np.pi/6  # float [rad]
    # When fieldRot is not 0, whether to infer spatial correlation fields and rotate them
    # Otherwise the name of the fields needs to be specified
    # Strain rate / rotation rate tensor fields not supported
    if fieldRot != 0.:
        spatialCorrelationFields = ('infer',)  # list/tuple(str) or list/tuple('infer')

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

# The following is only relevant if process_invariants is True
if process_invariants:
    # Absolute cap value for Sij and Rij; TB coefficients' basis;  and TB coefficients
    capSijRij, capG, capScalarBasis = 1e9, 1e9, 1e9  # float/int

# Save anything when possible
saveFields = True  # bool


"""
Machine Learning Settings
"""
# Use seed for reproducibility, set to None for no seeding
seed = 12345  # int, None
# Fraction of data used for testing
testSize = 0.2  # float
# Feature set to use as ML input
fs = '1'  # '1', '12', '123'


"""
Process User Inputs, No Need to Change
"""
# Ensemble name of fields useful for Machine Learning
mlFieldEnsembleName = 'ML_Fields_' + caseName
# Automatically select time if time is set to 'latest'
if time == 'last':
    if caseName == 'ALM_N_H_ParTurb':
        time = '22000.0918025'
    elif caseName == 'ALM_N_H_OneTurb':
        time = '24995.0788025'
else:
    time = str(time)

# Automatically define the confined domain region
if confineBox and boxAutoDim is not None:
    boxRot = np.pi/6
    if caseName == 'ALM_N_H_ParTurb':
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
    elif caseName == 'ALM_N_H_OneTurb':
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

# Ensemble file name, containing fields related to ML
mlFieldEnsembleNameFull = mlFieldEnsembleName + '_' + confinedFieldNameSub
# Initialize case object
case = FieldData(caseName=caseName, caseDir=caseDir, times=time, fields=fields, save=saveFields)


"""
Read and Process Raw Field Data
"""
if process_raw_field:
    # Read raw field data specified in fields
    fieldData = case.readFieldData()
    # Rotate fields if fieldRot is not 0
    if fieldRot != 0.:
        listData = []
        # If spatialCorrelationFields is list/tuple('infer'),
        # infer which field is a spatial single/double correlation for rotation
        if spatialCorrelationFields[0] == 'infer':
            # Redefine spatialCorrelationFields
            spatialCorrelationFields = []
            # Go through each item in fieldData dictionary
            for field, data in fieldData.items():
                # Pick up what ever is 2D since it means multiple components (columns) exist
                if len(data.shape) == 2:
                    listData.append(data)
                    spatialCorrelationFields.append(field)

        # Else if spatialCorrelationFields is provided
        else:
            for field in spatialCorrelationFields:
                listData.append(fieldData[field])

        # The provided tensor field should be single correlation, i.e. x instead of xx
        listData = case.rotateSpatialCorrelationTensors(listData, rotateXY = fieldRot, dependencies = 'x')
        for i, field in enumerate(spatialCorrelationFields):
            fieldData[field] = listData[i]

    # Initialize gradient of U and U itself as nPoint x 9
    gradU, u = (np.zeros((fieldData[fields[0]].shape[0], 9)),)*2
    # Initialize u'u' symmetric tensor as nPoint x 6
    uuPrime2 = np.zeros((fieldData[fields[0]].shape[0], 6))
    # Initialize gradient of p_rgh and TKE as nPoint x 3
    gradPrgh, gradK = (np.zeros((fieldData[fields[0]].shape[0], 3)),)*2
    # Initialize k, SGS epsilon, SGS nu, and p_rgh as nPoint x 0
    k, epsilon, nuSGS, prgh = (np.zeros(fieldData[fields[0]].shape[0]),)*4
    # Go through each read (and rotated) field to assign different field to variable,
    # and also aggregate resolved and SGS fields
    # Also keep track which field has been provided and aggregated
    gradKflag, kFlag, epsFlag, nuFlag, gradUflag, uFlag, gradPrghFlag, prghFlag, uuPrime2flag \
        = (False,)*9
    for field in fields:
        # If 'k' string in current field
        # There should be 'grad_k' 'kResolved' and 'kSGSmean' available
        if 'k' in field:
            # If moreover, there's gradFieldKW in current field, then it's a TKE gradient field
            if gradFieldKW in field:
                gradK += fieldData[field]
                gradKflag = True
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
                gradU += fieldData[field]
                gradUflag = True
            else:
                u += fieldData[field]
                uFlag = True

        # Same with p_rgh, there should be 'grad_p_rgh'
        elif 'p_rgh' in field:
            if gradFieldKW in field:
                gradPrgh += fieldData[field]
                gradPrghFlag = True
            else:
                prgh += fieldData[field]
                prghFlag = True

        # Same with u'u', there should be 'uuPrime2',
        # although no aggregation is needed since u'u' is SGS only
        elif 'uuPrime2' in field:
            # Expand symmetric tensor to it's full form, nCell x 9
            # From xx, xy, xz
            #          yy, yz
            #              zz
            # to xx, xy, xz
            #    yx, yy, yz
            #    zx, zy, zz
            uuPrime2 = np.vstack((fieldData[field][:, 0], fieldData[field][:, 1], fieldData[field][:, 2],
                                  fieldData['uuPrime2'][:, 1], fieldData[field][:, 3], fieldData[field][:, 4],
                                  fieldData[field][:, 2], fieldData[field][:, 4],
                                  fieldData[field][:, 5])).T
            uuPrime2flag = True

    # # Collect all flags
    # fieldFlags = (gradKflag, kFlag, epsFlag, nuFlag, gradUflag, uFlag, gradPrghFlag, prghFlag, uuPrime2flag)
    # Useful flags for Machine Learning
    mlFieldFlags = (gradKflag, kFlag, epsFlag, gradUflag, gradPrghFlag, uuPrime2flag)
    # Since epsilon is only SGS temporal mean,
    # calculate total temporal mean using SGS epsilon and SGS nu
    if all((epsFlag, nuFlag)):
        epsilon = case.getMeanDissipationRateField(epsilonSGSmean=epsilon, nuSGSmean=nuSGS, saveToTime=time)
        epsMax, epsMin = np.amax(epsilon), np.amin(epsilon)
        print(' Max of epsilon is {0}; min of epsilon is {1}'.format(epsMax, epsMin))

    # Convert 1D array to 2D so that I can hstack them to 1 array ensemble, nCell x 1
    k, epsilon = k.reshape((-1, 1)), epsilon.reshape((-1, 1))
    # Assemble all useful fields for Machine Learning
    mlFieldEnsemble = np.hstack((gradK, k, epsilon, gradU, gradPrgh, uuPrime2))
    print('\nField variables identified and resolved and SGS aggregated')
    # Read cell center coordinates of the whole domain, nCell x 0
    ccx, ccy, ccz, cc = case.readCellCenterCoordinates()


    """
    Confine the Whole Field, If confineBox Is True
    """
    if confineBox:
        # Confine to domain of interest and save confined cell centers as well as field ensember if requested
        ccx, ccy, ccz, cc, mlFieldEnsemble, box, _ = case.confineFieldDomain_Rotated(ccx, ccy, ccz, mlFieldEnsemble,
                                                                                        boxL = boxL, boxW = boxW,
                                                                                        boxH = boxH, boxO = boxOrig,
                                                                                        boxRot = boxRot,
                                                                                        fileNameSub = confinedFieldNameSub,
                                                                                        valsName = mlFieldEnsembleName)
        # Update old whole fields to new confined fields
        gradK, k = mlFieldEnsemble[:, :3], mlFieldEnsemble[:, 3]
        epsilon = mlFieldEnsemble[:, 4]
        gradU = mlFieldEnsemble[:, 5:14]
        gradPrgh = mlFieldEnsemble[:, 14:17]
        uuPrime2 = mlFieldEnsemble[:, 17:]
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
    else:
        case.savePickleData(time, mlFieldEnsemble, fileNames = mlFieldEnsembleNameFull)
        case.savePickleData(time, cc, fileNames = 'cc_' + confinedFieldNameSub)

# Else if directly read pickle data
else:
    # Load rotated and/or confined field data useful for Machine Learning
    mlFieldEnsemble = case.readPickleData(time, mlFieldEnsembleNameFull)
    gradK, k = mlFieldEnsemble[:, :3], mlFieldEnsemble[:, 3]
    epsilon = mlFieldEnsemble[:, 4]
    gradU = mlFieldEnsemble[:, 5:14]
    gradPrgh = mlFieldEnsemble[:, 14:17]
    uuPrime2 = mlFieldEnsemble[:, 17:]
    # Load confined cell centers too
    cc = case.readPickleData(time, 'cc_' + confinedFieldNameSub)


"""
Calculate Field Invariants
"""
if process_invariants:
    # Step 1: strain rate and rotation rate tensor Sij and Rij
    sij, rij = case.getStrainAndRotationRateTensorField(gradU, tke = k, eps = epsilon, cap = capSijRij)
    # Step 2: 10 invariant bases TB
    tb = case.getInvariantBasesField(sij, rij, quadratic_only = False, is_scale = True)
    # Since TB is nPoint x 10 x 3 x 3, reshape it to nPoint x 10 x 9
    tb = tb.reshape((tb.shape[0], tb.shape[1], 9))
    # Step 3: anisotropy tensor bij
    bij = case.getAnisotropyTensorField(uuPrime2)
    # # Step 4: evaluate 10 TB coefficients g as output y
    # # g, rmse = case.evaluateInvariantBasisCoefficients(tb, bij, cap = capG, onegToRuleThemAll = False)
    # t0 = t.time()
    # g, gRMSE = evaluateInvariantBasisCoefficients(tb, bij, cap = capG)
    # t1 = t.time()
    # print('\nFinished getInvariantBasisCoefficientsField in {:.4f} s'.format(t1 - t0))
    # Save tensor invariants related fields
    if saveFields:
        case.savePickleData(time, sij, fileNames = ('Sij_' + confinedFieldNameSub))
        case.savePickleData(time, rij, fileNames = ('Rij_' + confinedFieldNameSub))
        case.savePickleData(time, tb, fileNames = ('TB_' + confinedFieldNameSub))
        case.savePickleData(time, bij, fileNames = ('bij_' + confinedFieldNameSub))
        # case.savePickleData(time, g, fileNames = ('g_' + confinedFieldNameSub))
        # case.savePickleData(time, gRMSE, fileNames = ('g_RMSE_True_' + confinedFieldNameSub))

# Else if read invariants data from pickle
else:
    invariants = case.readPickleData(time, fileNames = ('Sij_' + confinedFieldNameSub,
                                                        'Rij_' + confinedFieldNameSub,
                                                        'TB_' + confinedFieldNameSub,
                                                        'bij_' + confinedFieldNameSub))
    sij = invariants['Sij_' + confinedFieldNameSub]
    rij = invariants['Rij_' + confinedFieldNameSub]
    tb = invariants['TB_' + confinedFieldNameSub]
    bij = invariants['bij_' + confinedFieldNameSub]
    # g = invariants['g_' + confinedFieldNameSub]


"""
Calculate Feature Sets
"""
# Feature set 1
fs1 = getFeatureSet1(sij, rij)
# If only feature set 1 used for ML input, then do train test data split here
if fs == '1':
    # FIXME: TB couldn't be split here
    # xTrain, xTest, yTrain, yTest, _ = splitTrainTestData(x = fs1, y = bij, randState = seed, scalerScheme = None)
    if saveFields:
        case.savePickleData(time, fs1, fileNames = ('FS' + fs + '_' + confinedFieldNameSub))
        # case.savePickleData(time, yTrain, fileNames=('yTrain_' + confinedFieldNameSub))
        # case.savePickleData(time, xTrain, fileNames = ('FS' + fs + '_Train_' + confinedFieldNameSub))
        # case.savePickleData(time, xTest, fileNames = ('FS' + fs + '_Test_' + confinedFieldNameSub))
        # case.savePickleData(time, yTrain, fileNames = ('yTrain_' + confinedFieldNameSub))
        # case.savePickleData(time, yTest, fileNames = ('yTest_' + confinedFieldNameSub))




















