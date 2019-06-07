from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np
from warnings import warn
from Utilities import timer

@timer
def setupRandomForestGridSearch(nEstimator = 100, maxFeatPercents = (1/3.,), minSampleSplits = (2,), criterion = ('mse',),
                                scalerScheme = 'robust', splitScheme = 'RF', verbose = 1,
                                customOobScore = None, randState = None):
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    if customOobScore in ('errVar', 'monotonic'):
        warn(
            '\n[SetupRandomForestGridSearch] requires addition of custom score functions by [yluan] to the [sklearn] '
            'source '
            'code. '
            'Make sure the corresponding file has been copied from the current directory to [sklearn]\n!',
            stacklevel = 2)

    # # Give it a name for the figure window title
    # if splitScheme == 'RF':
    #     estimatorName = 'RandForest'
    # else:
    #     if splitScheme != 'extremeRF':
    #         warn('\nInvalid [splitScheme] in [SetupRandomForestGridSearch]! Using default [RF] split scheme...\n',
    #              stacklevel = 2)
    #         splitScheme = 'RF'
    #
    #     estimatorName = 'ExtremeRandForest'

    # Pipeline to queue data scaling and estimator together
    if splitScheme == 'RF':
        if scalerScheme == 'standard':
            # If customScore is either 'errVar' or 'monotonic', then sklearn/ensemble/forest.py should've already
            # modified by [
            # yluan] with the
            # inclusion of a new parameter [customOobScore]
            if customOobScore in ('errVar', 'monotonic'):
                # Try to add customOobScore_yluan argument. If fails, then it means the source code hasn't been
                # updated to yluan's version
                try:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                            oob_score =
                                                            True, random_state = randState, customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme,
                         RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose, oob_score =
                         True, random_state = randState))])

            # Else if customScore not 'errVar' nor 'monotonic', don't care whether forest.py has been modified and
            # don't
            # provide a
            #  new
            # parameter arg
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    (splitScheme,
                     RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose, oob_score =
                     True, random_state = randState))])


        else:
            if scalerScheme != 'robust':
                warn('\nInvalid [scalerScheme]! Using default "robust" scaler...\n', stacklevel = 2)

            # The same with robust scaler
            if customOobScore:
                try:
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                            oob_score = True, random_state = randState,
                                                            customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                            oob_score = True, random_state = randState))])

            else:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                        oob_score = True, random_state = randState))])
        # End of if standard or robust scaler

    # Else if splitScheme is extremeRF
    else:
        if scalerScheme == 'standard':
            if customOobScore in ('errVar', 'monotonic'):
                try:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = randState, customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = randState))])

            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                      bootstrap = True, oob_score =
                                                      True, random_state = randState))])


        else:
            if scalerScheme != 'robust':
                warn('\nInvalid [scalerScheme]! Using default [robust] scaler...\n', stacklevel = 2)

            if customOobScore in ('errVar', 'monotonic'):
                try:
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = randState, customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = randState))])

            else:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                      bootstrap = True, oob_score =
                                                      True, random_state = randState))])
        # End of if standard or robust scaler
    # End of if splitScheme

    if splitScheme == 'RF':
        tuneParams = dict(RF__max_features = maxFeatPercents,
                          RF__min_samples_split = minSampleSplits,
                          RF__criterion = criterion)
    else:
        tuneParams = dict(extremeRF__max_features = maxFeatPercents,
                          extremeRF__min_samples_split = minSampleSplits,
                          extremeRF__criterion = criterion)

    RF_GS = pipeline

    return RF_GS, tuneParams


@timer
def processEstimatorGridSearch(estimatorGS, tuneParams, xTrain, yTrain, xTest, yTest,
                               customScore = None, sortCol = 0, staged = False, verbose = 1):
    from sklearn.model_selection import ParameterGrid
    from sklearn.multioutput import RegressorChain
    # from sklearn.metrics import make_scorer
    import copy

    # TODO: at least 2D
    # TODO: ensure tuple input

    print('\nExecuting [processEstimatorGridSearch] using {} custom scorer...'.format(customScore))
    print(' {0}'.format(tuneParams))
    # Initialize best score and will be updated once found better
    bestScore = -10000
    bestGrid = {}
    bestEstimator = copy.deepcopy(estimatorGS)
    for grid in ParameterGrid(tuneParams):
        estimatorGS.set_params(**grid)
        # If multiple outputs, then use regression chain,
        # i.e. X -> y1
        #     {X, y1} -> y2
        #     {X, y1, y2} -> y3
        estimatorGS_Chain = RegressorChain(estimatorGS) if yTrain.shape[1] > 1 else estimatorGS
        # Fit the train data for ML
        estimatorGS_Chain.fit(xTrain, yTrain)
        # Estimator chain has a china of trained estimators equal to number of outputs
        # Thus get the list of trained chain estimators if possible
        estimatorGS = estimatorGS_Chain.estimators_ if yTrain.shape[1] > 1 else (estimatorGS_Chain,)
        # If the estimator uses out-of-bag samples, then the score is oob_score_ that uses the train data
        if hasattr(estimatorGS[0]._final_estimator, 'oob_score_'):
            print(' Calculating OOB score...')
            score = 0
            for estimatorGS_i in estimatorGS:
                score += estimatorGS_i._final_estimator.oob_score_

            # Get the average OOB score if multiple outputs
            score /= yTrain.shape[1]
        # Else, use the default/custom score method on the test data
        else:
            # Setup custom scorer if enabled
            if customScore in ('errVar', 'monotonic'):
                if staged and hasattr(estimatorGS[0]._final_estimator, 'staged_score'):
                    # Since staged_score is based on _final_estimator of the estimatorGS pipeline only, it does not
                    # really
                    #  go
                    # through the scaler. Thus apply scaler manually to transform xTest
                    xTestScaled = estimatorGS[0].named_steps['scaler'].transform(xTest)
                    print('')
                    # At each boost, in staged_score() generator, a yPred is yielded from staged_prediction()
                    # generator,
                    # then a corresponding
                    # score_iBoost is yielded
                    # staged_score(xTest, yTest) is a generator of 1D array dim(nEstimator), nEstimator = nBoost
                    # For val in generator <=> for i in nIter: val = next(generator)
                    for iBoost, scoreStaged in enumerate(estimatorGS[0]._final_estimator.staged_score(xTestScaled, yTest)):
                        # When iBoost reaches the end, that score should be the best for this hyper-parameter grid
                        score = scoreStaged
                        if verbose == 1:
                            print(' Staged score after boost {0}: {1}'.format(iBoost + 1, score))

                    print('')
                # else:
                #     score = customScorer(estimatorGS[0], xTest, yTest)

            else:
                if customScore not in (None, 'r2'):
                    warn('\nInvalid [customScore]! Using default scorer...\n', stacklevel = 2)

                if staged and hasattr(estimatorGS[0]._final_estimator, 'staged_score'):
                    xTestScaled = estimatorGS[0].named_steps['scaler'].transform(xTest)
                    print('')
                    for iBoost, scoreStaged in enumerate(estimatorGS[0]._final_estimator.staged_score(xTestScaled, yTest)):
                        # When iBoost reaches the end, that score should be the best for this hyper-parameter grid
                        score = scoreStaged
                        if verbose == 1:
                            print(' Staged score after boost {0}: {1}'.format(iBoost + 1, score))
                    print('')
                else:
                    score = estimatorGS[0].score(xTest, yTest)

        # Save if best
        if score > bestScore:
            print(' Old best score is {0}'.format(bestScore))
            print(' New best score is {0}'.format(score))
            bestScore = score
            bestGrid = grid
            bestEstimator = copy.deepcopy(estimatorGS_Chain) if yTrain.shape[1] > 1 else copy.deepcopy(estimatorGS[0])

        # Change estimator back to regression chain or normal estimator
        estimatorGS = estimatorGS[0]

    print(' Best [score] is {}'.format(bestScore))
    print(' Best [hyper-parameters] are \n  {}'.format(bestGrid))

    # Reassign estimatorGS to the best trained model so that it's reusable
    estimatorGS = bestEstimator
    # # Set the best hyper-parameters to the estimator
    # estimatorGS.set_params(**bestGrid)
    # # The previous fits were cleared thus need to refit using the best hyper-parameters
    # print(' Re-fitting the model using the best hyper-parameters...')
    # estimatorGS.fit(xTrain, yTrain)

    # yPredStagedDict = {}
    # # Again, if the estimator uses out-of-bag samples, then the training set is the "test" data to predict on
    # if hasattr(estimatorGS._final_estimator, 'oob_prediction_'):
    #     print(' Performing OOB prediction using training set...')
    #     yPred = estimatorGS._final_estimator.oob_prediction_
    #     # sortedIdx = np.argsort(yTrain[:, sortCol])
    # else:
    #     if staged and hasattr(estimatorGS._final_estimator, 'staged_predict'):
    #         # staged_predict(xTest) is a generator of array dim(nEstimator, nSample), nEstimator = nBoost
    #         # Use xTestScaled here since _final_estimator.staged_predict() bypasses the scaler of the
    #         # estimatorGS pipeline
    #         for iBoost, yPredStaged in enumerate(estimatorGS._final_estimator.staged_predict(xTestScaled)):
    #             yPredStagedDict[str(iBoost)] = yPredStaged
    #
    #         # Final yPred is the last boost's result
    #         yPred = yPredStagedDict[str(iBoost)]
    #
    #     else:
    #         yPred = estimatorGS.predict(xTest)
    #
    # #     # Sorted index list based on yTest[:, output_i] from low to high
    # #     sortedIdx = np.argsort(yTest[:, sortCol])
    # #
    # # # Reverse the index list so that it is high to low
    # # sortedIdx = reversed(sortedIdx)
    # # yPred_sorted = np.empty(yPred.shape)
    # # for i, idxVal in enumerate(sortedIdx):
    # #     yPred_sorted[i, :] = yPred[idxVal, :]

    return bestGrid
