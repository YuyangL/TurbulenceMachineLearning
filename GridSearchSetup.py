from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np
from warnings import warn
from Utilities import timer

@timer
def setupDecisionTreeGridSearchCV(max_features=(1.,), min_samples_split=(2,), min_samples_leaf=(1,),
                                  alpha_g_fit=(0.,), alpha_g_split=(0.,),
                                  cv=5, presort=True, split_finder="brute",
                                  tb_verbose=False, split_verbose=False, scaler='robust', rand_state=None, gscv_verbose=1,
                                  max_depth=None,
                                  g_cap=None):
    """
    Setup for (Tensor Basis) Decision Tree Regressor Grid Search Cross Validation.
    If tensor basis tb is provided to self.fit(), then tensor basis criterion is enabled. Otherwise, Decision Tree Regressor is a default model.
    If split_finder is given as "brent" or "1000",
    then Brent optimization or a total split limit of 1000 per node is implemented to find the best split at a node.
    If scaler is "robust" or "standard",
    then RobustScaler or StandardScaler is implemented in a pipeline, before (Tensor Basis) Decision Tree Regressor.
    The hyper-parameters to tune are max_features, min_samples_split, and min_samples_leaf.

    :param max_features: Maximum number of features to consider for the best split.
    If a fraction is given, then the fraction of all features are considered.
    :type max_features: tuple/list(int / float 0-1), or int / float 0-1
    :param min_samples_split: Minimum number of samples to consider for a split.
    If a fraction is given, then the fraction of all samples are considered.
    :type min_samples_split: tuple/list(int / float 0-1) or int / float 0-1
    :param min_samples_leaf: Minimum number of samples for the final leaf node.
    If a fraction is given, then the fraction of all samples are considered.
    :type min_samples_leaf: tuple/list(int / float 0-1) or int / float 0-1
    :param cv: Number of cross-validation folds.
    :type cv: int, optional (default=5)
    :param presort: Whether to presort the training data before start building the tree.
    :type presort: bool, optional (default=True)
    :param split_finder: Best split finding scheme.
    If "brent", then use Brent optimization. Should only be used with tensor basis criterion.
    If "1000", then the maximum number of split per node is limited to 1000. Should only be used with tensor basis criterion.
    If "brute", then all possible splits are considered each node.
    :type split_finder: "brute" or "brent" or "1000", optional (default="brute")
    :param tb_verbose: Whether to verbose tensor basis criterion related executions.
    :type tb_verbose: bool, optional (default=False)
    :param split_verbose: Whether to verbose best split finding scheme.
    :type split_verbose: bool, optional (default=False)
    :param scaler: Training data scaler scheme.
    If "robust", then RobustScaler is used as data pre-processing.
    If "standard", then StandardScaler is used as data pre-processing.
    If None, then no data scaler is used.
    :type scaler: "robust" or "standard" or None, optional (default="robust")
    :param rand_state: Whether use random state to reproduce the same fitting procedure.
    :type rand_state: int or RandomState instance or None, optional (default=None)
    :param gscv_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gscv_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV

    # Ensure hyper-parameters in a sequence
    if isinstance(max_features, (int, float)): max_features = (max_features,)
    if isinstance(min_samples_split, (int, float)): min_samples_split = (min_samples_split,)
    if isinstance(min_samples_leaf, (int, float)): min_samples_leaf = (min_samples_leaf,)
    if isinstance(alpha_g_fit, (int, float)): alpha_g_fit = (alpha_g_fit,)
    if isinstance(alpha_g_split, (int, float)): alpha_g_split = (alpha_g_split,)

    # Pipeline to queue data scaling and estimator together
    if scaler == 'standard':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                           max_depth=max_depth))])
    elif scaler == 'robust':
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('tree', DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                           max_depth=max_depth,
                                           g_cap=g_cap))])
    else:
        pipeline = DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                         max_depth=max_depth,
                                         g_cap=g_cap)

    # Append hyper-parameters to a dict, depending on pipeline or not
    if scaler in ("robust", "standard"):
        tune_params = dict(tree__max_features=max_features,
                           tree__min_samples_split=min_samples_split,
                           tree__min_samples_leaf=min_samples_leaf,
                           tree__alpha_g_fit=alpha_g_fit,
                           tree__alpha_g_split=alpha_g_split)
    else:
        tune_params = dict(max_features=max_features,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf,
                           alpha_g_fit=alpha_g_fit,
                           alpha_g_split=alpha_g_split)

    # Construct GSCV for DT
    tree_gscv = GridSearchCV(pipeline,
                            cv=cv,
                            param_grid=tune_params,
                            n_jobs=-1,
                            error_score='raise', verbose=gscv_verbose, scoring=None)

    print('\nSetup [Decision Tree] Grid Search using {0}-fold Cross-Validation and [{1}] scaler:'.format(cv, scaler))
    return tree_gscv, tune_params


@timer
def setupRandomForestGridSearch(nEstimator = 100, max_features = (1/3.,), min_samples_split = (2,), criterion = ('mse',),
                                scaler = 'robust', splitScheme = 'RF', verbose = 1,
                                customOobScore = None, rand_state = None):
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
        if scaler == 'standard':
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
                                                            True, random_state = rand_state, customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme,
                         RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose, oob_score =
                         True, random_state = rand_state))])

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
                     True, random_state = rand_state))])


        else:
            if scaler != 'robust':
                warn('\nInvalid [scaler]! Using default "robust" scaler...\n', stacklevel = 2)

            # The same with robust scaler
            if customOobScore:
                try:
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                            oob_score = True, random_state = rand_state,
                                                            customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                            oob_score = True, random_state = rand_state))])

            else:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    (splitScheme, RandomForestRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                        oob_score = True, random_state = rand_state))])
        # End of if standard or robust scaler

    # Else if splitScheme is extremeRF
    else:
        if scaler == 'standard':
            if customOobScore in ('errVar', 'monotonic'):
                try:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = rand_state, customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = rand_state))])

            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                      bootstrap = True, oob_score =
                                                      True, random_state = rand_state))])


        else:
            if scaler != 'robust':
                warn('\nInvalid [scaler]! Using default [robust] scaler...\n', stacklevel = 2)

            if customOobScore in ('errVar', 'monotonic'):
                try:
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = rand_state, customOobScore = customOobScore))])
                except TypeError:
                    warn('\n[customOobScore] failed! [sklearn/ensemble/forest.py] has not been updated from ['
                         'Files_CustomOobScore]. '
                         'Using '
                         'default scoring scheme...\n', stacklevel = 2)
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                          bootstrap = True, oob_score =
                                                          True, random_state = rand_state))])

            else:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    (splitScheme, ExtraTreesRegressor(n_estimators = nEstimator, n_jobs = -1, verbose = verbose,
                                                      bootstrap = True, oob_score =
                                                      True, random_state = rand_state))])
        # End of if standard or robust scaler
    # End of if splitScheme

    if splitScheme == 'RF':
        tuneParams = dict(RF__max_features = max_features,
                          RF__min_samples_split = min_samples_split,
                          RF__criterion = criterion)
    else:
        tuneParams = dict(extremeRF__max_features = max_features,
                          extremeRF__min_samples_split = min_samples_split,
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
