import numpy as np
from warnings import warn

def InputOutlierDetection(xTrain, xTest, yTrain, yTest, outlierPercent=0.2, removal=None, isoForest=None, randState=None, onlyTrain=False, n_estimators=100):
    from sklearn.ensemble import IsolationForest
    print('\nExecuting [InputOutlierDetection] using Isolation Forest...')
    # If no current Isolation Forest exists, so to learn the current data to train the Isolation Forest model
    if isoForest is None:
        isoForest = IsolationForest(n_jobs=-1, verbose=2, contamination=outlierPercent, random_state=randState, n_estimators=n_estimators,
                                    bootstrap=True)
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
