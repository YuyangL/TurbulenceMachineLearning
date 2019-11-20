import numpy as np
from warnings import warn

def InputOutlierDetection(xtrain, xtest, ytrain, ytest, outlier_percent=0.2, removal=None, isoforest=None, randstate=None, onlytrain=False, n_estimators=100):
    from sklearn.ensemble import IsolationForest
    print('\nExecuting [InputOutlierDetection] using Isolation Forest...')
    # If no current Isolation Forest exists, so to learn the current data to train the Isolation Forest model
    if isoforest is None:
        isoforest = IsolationForest(n_jobs=-1, verbose=2, contamination=outlier_percent, random_state=randstate, n_estimators=n_estimators,
                                    bootstrap=True)
        # Train the isolation forest to define and detect outliers
        isoforest.fit(xtrain)
        # If I just intend to train an Isolation Forest, then return the Isolation Forest and end the function
        if onlytrain:
            return isoforest

    # Yield score arrays on the training and test data in which -1 means anomaly
    xtrain_anomalyscore = isoforest.predict(xtrain)
    # If testSize = 0., then skip test data prediction
    try:
        xtest_anomalyscore = isoforest.predict(xtest)
    except:
        xtest_anomalyscore = []

    # meanScoreTrain = isoforest.decision_function(xtrain)
    # meanScoreTest = isoforest.decision_function(xtest)
    # print('Train data anomaly score (higher is better): {0}'.format(meanScoreTrain))
    # print('Test data anomaly score (higher is better): {0}'.format(meanScoreTest))

    # Get the index array of all data considered abnormal (-1)
    anomaly_idx_train = np.where(xtrain_anomalyscore == -1)
    anomaly_idx_test = np.where(xtest_anomalyscore == -1)
    anomaly_idx_train = anomaly_idx_train[0]
    anomaly_idx_test = anomaly_idx_test[0]

    # Initialize "empty" array/list for outliers
    xtrainOutliers = np.empty([1, xtrain.shape[1]])
    # ytrainOutliers = np.empty(len(anomaly_idx_train))
    ytrainOutliers = np.empty([1, ytrain.shape[1]])
    # If xtest is a an empty list then skip this step
    try:
        xtestOutliers = np.empty([1, xtest.shape[1]])
        ytestOutliers = np.empty([1, ytest.shape[1]])
    except AttributeError:
        xtestOutliers = ['dummy']
        ytestOutliers = ['dummy']

    xtrainRaw = xtrain
    ytrainRaw = ytrain
    # Remove the anomaly indices iteratively
    for i, idx in enumerate(anomaly_idx_train):
        xtrainOutliers = np.vstack((xtrainOutliers, xtrainRaw[idx]))
        # ytrainOutliers[i] = ytrain[idx]
        ytrainOutliers = np.vstack((ytrainOutliers, ytrainRaw[idx]))
        # If removal is 'train' or 'both, remove the outliers in the train input and target data
        if removal in ('train', 'both'):
            ytrain = np.delete(ytrain, idx - i, 0)
            # 0 means the first axis -- row
            xtrain = np.delete(xtrain, idx - i, 0)

    # # Update xtrain and ytrain if removal was done
    # if removal in('train', 'both'):
    #     xtrain = xtrainInliers
    #     ytrain = ytrainInliers

    xtestRaw = xtest
    ytestRaw = ytest
    # When anomaly_idx_test is empty, this loop will not execute
    for i, idx in enumerate(anomaly_idx_test):
        xtestOutliers = np.vstack((xtestOutliers, xtestRaw[idx]))
        # ytestOutliers[i] = ytest[idx]
        ytestOutliers = np.vstack((ytestOutliers, ytestRaw[idx]))
        # If removal is 'test' or 'both', then remove the outliers in test input and target data
        if removal in ('test', 'both'):
            ytest = np.delete(ytest, idx - i, 0)
            xtest = np.delete(xtest, idx - i, 0)

    # Since the arrays were not actually empty when they were initiated, remove the first row, first 0 means "first", second 0 means "row"
    xtrainOutliers = np.delete(xtrainOutliers, 0, 0)
    ytrainOutliers = np.delete(ytrainOutliers, 0, 0)

    # xtestOutliers = xtestOutliers[1:]
    xtestOutliers = np.delete(xtestOutliers, 0, 0)
    ytestOutliers = np.delete(ytestOutliers, 0, 0)

    # Get the outliers for later inspection
    outliers = dict(xtrain=xtrainOutliers,
                    xtest=xtestOutliers,
                    ytrain=ytrainOutliers,
                    ytest=ytestOutliers,
                    anomaly_idx_train=anomaly_idx_train,
                    anomaly_idx_test=anomaly_idx_test,
                    xtrain_anomalyscore=xtrain_anomalyscore,
                    xtest_anomalyscore=xtest_anomalyscore)

    return xtrain, xtest, ytrain, ytest, outliers, isoforest
