"""
Extract Features from Provided Invariants
"""
import sys
# See https://github.com/YuyangL/SOWFA-PostProcess
sys.path.append('/home/yluan/Documents/SOWFA PostProcessing/SOWFA-Postprocess')
import numpy as np
from PostProcess_Tensor import convertTensorTo2D
from Utilities import nDto2D_TensorField, timer
from numba import jit, njit, prange

@timer
@jit(parallel=True, fastmath=True)
def getFeatureSet1(sij, rij):
    # Ensure tensor field is nPoint x nComponent (9 in this case)
    sij, rij = convertTensorTo2D(sij), convertTensorTo2D(rij)
    # Reshape Sij and Rij to nPoint x 3 x 3 for matrix calculations later
    sij, rij = sij.reshape((sij.shape[0], 3, 3)), rij.reshape((sij.shape[0], 3, 3))
    # Initialize feature set invariants as nPoint x 6 features
    fs1 = np.empty((sij.shape[0], 6))
    # Go through each point
    for i in prange(fs1.shape[0]):
        # Sij^2
        ss_i = np.dot(sij[i], sij[i])
        # Rij^2
        rr_i = np.dot(rij[i], rij[i])
        # RijSij
        rs_i = np.dot(rij[i], sij[i])
        # Feature 1: tr(Sij^2)
        fs1[i][0] = np.trace(ss_i)
        # Feature 2: tr(Sij^3)
        fs1[i][1] = np.trace(np.dot(sij[i], ss_i))
        # Feature 3: tr(Rij^2)
        fs1[i][2] = np.trace(rr_i)
        # Feature 4: tr(Rij^2Sij)
        fs1[i][3] = np.trace(np.dot(rij[i], rs_i))
        # Feature 5: tr(Rij^2Sij^2)
        fs1[i][4] = np.trace(np.dot(rij[i], np.dot(rij[i], ss_i)))
        # Feature 6: tr(Rij^2SijRijSij^2)
        fs1[i][5] = np.trace(np.dot(rij[i], np.dot(rij[i], np.dot(sij[i], np.dot(rij[i], ss_i)))))

    return fs1


@timer
def splitTrainTestDataList(list_data, test_fraction=0.2, sample_size=None, replace=False, seed=None):
    """
    Split a list of data into train and test data based on given test fraction.
    Each data in the list should have row for sample index, don't care about other axes.

    :param list_data: List of data to sample and train-test split.
    If only one data provided, then it doesn't have to be in a list.
    :type list_data: list/tuple(np.ndarray[n_samples, _])
    or np.ndarray[n_samples, _]

    :param test_fraction: Fraction of data to use for test.
    :type test_fraction: float [0-1], optional (default=0.2)

    :param sample_size: Number of samples if doing sampling.
    If None, then no sampling is done and all samples are used.
    :type sample_size: int or None, optional (default=None)

    :param replace: Whether to draw samples with replacement.
    :type replace: bool, optional (default=False)

    :param seed: Whether to use seed for reproducibility.
    If None, then seed is not provided.
    :type seed: int or None, optional (default=None)

    :return: (Sampled) list of train data and list of test data
    :rtype: list(np.ndarray[n_samples_train, _]), list(np.ndarray[n_samples_test, _])
    """

    # Ensure list
    if isinstance(list_data, np.ndarray):
        list_data = [list_data]
    elif isinstance(list_data, tuple):
        list_data = list(list_data)

    # If sample_size is None, use all samples
    # Otherwise use provided sample_size but limit it
    sample_size = len(list_data[0]) if sample_size is None else int(min(sample_size, len(list_data[0])))

    # Indices of the samples randomized
    np.random.seed(seed)
    rand_indices = np.random.choice(np.arange(len(list_data[0])), len(list_data[0]), replace=replace)
    # Train and test data sample size taking into account of sampling
    train_size = int(sample_size*(1. - test_fraction))
    test_size = sample_size - train_size
    # Indices of train and test data after randomization and sampling
    indices_train, indices_test = rand_indices[:train_size], rand_indices[train_size:sample_size]
    # Go through all provided data
    list_data_train, list_data_test = list_data.copy(), list_data.copy()
    for i in range(len(list_data)):
        # Pick up samples for train and test respectively from index lists prepared earlier
        list_data_train[i] = list_data[i][indices_train]
        list_data_test[i] = list_data[i][indices_test]

    return list_data_train, list_data_test


@timer
def splitTrainTestData(x, y, randState = None, testSize = 0.2, scalerScheme = None, scaler = None):
    from warnings import warn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    # Train and test data split
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = testSize, random_state = randState)
    # If scaler already exists and is provided, then don't need to learn anymore and directly transform instead of
    # fit_transform
    if scaler is not None:
        xTrain = scaler.transform(xTrain)
        # Transform to get test data using the train data scaler
        # In case all data was train data
        try:
            xTest = scaler.transform(xTest)
        except ValueError:
            warn('\nAll data are used for training so no scaling for test data!\n', stacklevel = 2)

    # Else if no current scaler exists, then we have to specify if we want to use a scheme and
    # then return both standardized data as well such learned scaler
    else:
        # If scalerScheme is enabled, scale the input train and test data
        if scalerScheme is not None:
            # Call the scaler
            if scalerScheme == 'standard':
                print(' Standardizing the input data...')
                scaler = StandardScaler()
                # Fit to data first and then transform
                xTrain = scaler.fit_transform(xTrain)
                # Transform to get test data using the train data scaler
                # In case all data was train data
                try:
                    xTest = scaler.transform(xTest)
                except ValueError:
                    warn('\nAll data are used for training so no scaling for test data!\n', stacklevel
                    = 2)

            elif scalerScheme == 'quantile':
                print(' Quantile transforming the input data...')
                scaler = QuantileTransformer(random_state = randState)
                xTrain = scaler.fit_transform(xTrain)
                try:
                    xTest = scaler.transform(xTest)
                except ValueError:
                    warn('\nAll data are used for training so no scaling for test data!\n', stacklevel
                    = 2)

            else:
                if scalerScheme != 'robust':
                    warn('\nInvalid scalerScheme! Using default [robust] scaler...\n', stacklevel = 2)
                print(' Robust scaling the input data...')
                scaler = RobustScaler()
                # Fit to data first and then transform
                xTrain = scaler.fit_transform(xTrain)
                try:
                    xTest = scaler.transform(xTest)
                except ValueError:
                    warn('\nAll data are used for training so no scaling for test data!\n',
                         stacklevel = 2)

            # End of if scalerScheme is 'standard', 'robust' or 'quantile'
        # End of if data will be scaled here instead of pipeline in sklearn
    # End of if a current scaler is provided

    return xTrain, xTest, yTrain, yTest, scaler


@timer
def inputOutlierDetection(xTrain, xTest, yTrain, yTest, outlierPercent = 0.2, removal = None, isoForest = None, randState = None, onlyTrain = False):
    from sklearn.ensemble import IsolationForest
    import numpy as np
    # If no current Isolation Forest exists, so to learn the current data to train the Isolation Forest model
    if isoForest is None:
        isoForest = IsolationForest(n_jobs = -1, verbose = 1, contamination = outlierPercent, random_state = randState)
        # Train the isolation forest to define and detect outliers
        isoForest.fit(xTrain)
        # If I just intend to train an Isolation Forest, then return the Isolation Forest and end the function
        if onlyTrain:
            return isoForest

    # Yield score arrays on the training and test data in which -1 means anomaly
    xTrainAnomalyScores = isoForest.predict(xTrain)
    # If testSize = 0., then skip test data prediction
    try:
        xTestAnomalyScores = isoForest.predict(xTest)
    except:
        xTestAnomalyScores = []

    # meanScoreTrain = isoForest.decision_function(xTrain)
    # meanScoreTest = isoForest.decision_function(xTest)
    # print('Train data anomaly score (higher is better): {0}'.format(meanScoreTrain))
    # print('Test data anomaly score (higher is better): {0}'.format(meanScoreTest))

    # Get the index array of all data considered abnormal (-1)
    anomalyIdxTrains = np.where(xTrainAnomalyScores == -1)
    anomalyIdxTests = np.where(xTestAnomalyScores == -1)
    anomalyIdxTrains = anomalyIdxTrains[0]
    anomalyIdxTests = anomalyIdxTests[0]
    # Initialize "empty" array/list for outliers
    xTrainOutliers = np.empty([1, xTrain.shape[1]])
    # yTrainOutliers = np.empty(len(anomalyIdxTrains))
    yTrainOutliers = np.empty([1, yTrain.shape[1]])
    # If xTest is a an empty list then skip this step
    try:
        xTestOutliers = np.empty([1, xTest.shape[1]])
        yTestOutliers = np.empty([1, yTest.shape[1]])
    except AttributeError:
        xTestOutliers = ['dummy']
        yTestOutliers = ['dummy']

    xTrainRaw = xTrain
    yTrainRaw = yTrain
    # Remove the anomaly indices iteratively
    for i, idx in enumerate(anomalyIdxTrains):
        xTrainOutliers = np.vstack((xTrainOutliers, xTrainRaw[idx]))
        # yTrainOutliers[i] = yTrain[idx]
        yTrainOutliers = np.vstack((yTrainOutliers, yTrainRaw[idx]))
        # If removal is 'train' or 'both, remove the outliers in the train input and target data
        if removal in ('train', 'both'):
            yTrain = np.delete(yTrain, idx - i, 0)
            # 0 means the first axis -- row
            xTrain = np.delete(xTrain, idx - i, 0)

    # # Update xTrain and yTrain if removal was done
    # if removal in('train', 'both'):
    #     xTrain = xTrainInliers
    #     yTrain = yTrainInliers

    xTestRaw = xTest
    yTestRaw = yTest
    # When anomalyIdxTests is empty, this loop will not execute
    for i, idx in enumerate(anomalyIdxTests):
        xTestOutliers = np.vstack((xTestOutliers, xTestRaw[idx]))
        # yTestOutliers[i] = yTest[idx]
        yTestOutliers = np.vstack((yTestOutliers, yTestRaw[idx]))
        # If removal is 'test' or 'both', then remove the outliers in test input and target data
        if removal in ('test', 'both'):
            yTest = np.delete(yTest, idx - i, 0)
            xTest = np.delete(xTest, idx - i, 0)

    # Since the arrays were not actually empty when they were initiated, remove the first row, first 0 means "first", second 0 means "row"
    xTrainOutliers = np.delete(xTrainOutliers, 0, 0)
    yTrainOutliers = np.delete(yTrainOutliers, 0, 0)

    # xTestOutliers = xTestOutliers[1:]
    xTestOutliers = np.delete(xTestOutliers, 0, 0)
    yTestOutliers = np.delete(yTestOutliers, 0, 0)

    # Get the outliers for later inspection
    outliers = dict(xTrain = xTrainOutliers,
                    xTest = xTestOutliers,
                    yTrain = yTrainOutliers,
                    yTest = yTestOutliers,
                    anomalyIdxTrains = anomalyIdxTrains,
                    anomalyIdxTests = anomalyIdxTests,
                    xTrainAnomalyScores = xTrainAnomalyScores,
                    xTestAnomalyScores = xTestAnomalyScores)

    return xTrain, xTest, yTrain, yTest, outliers, isoForest


# TODO: Feature set 2
# TODO: Feature set 3
# TODO: Feature selection
