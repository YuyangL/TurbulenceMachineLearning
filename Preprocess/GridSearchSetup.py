import numpy as np
from warnings import warn
from Utilities import timer

@timer
def setupDecisionTreeGridSearchCV(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                  min_samples_leaf=2,
                                  cv=5, presort=True, split_finder="brute",
                                  tb_verbose=False, split_verbose=False, rand_state=None, gscv_verbose=1,
                                  max_depth=None,
                                  g_cap=None,
                                  realize_iter=0,
                                  n_jobs=-1,
                                  refit=False,
                                  return_train_score=False,
                                  # [DEPRECATED]
                                  alpha_g_fit=0.,
                                  **kwargs):
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
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)

    tree = DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                   max_depth=max_depth,
                                   g_cap=g_cap,
                                   realize_iter=realize_iter,
                                 min_samples_leaf=min_samples_leaf,
                                 alpha_g_fit=alpha_g_fit)

    # Append hyper-parameters to a dict, depending on pipeline or not
    tuneparams = dict(max_features=gs_max_features,
                      min_samples_split=gs_min_samples_split,
                      alpha_g_split=gs_alpha_g_split)

    # Construct GSCV for DT
    tree_gscv = GridSearchCV(tree,
                             cv=cv,
                             param_grid=tuneparams,
                             n_jobs=n_jobs,
                             error_score='raise',
                             verbose=gscv_verbose,
                             scoring=None,
                             refit=refit,
                             return_train_score=return_train_score,
                             # Mean score of folds is not sample weighted
                             iid=False)

    return tree_gscv, tree, tuneparams


@timer
def setupRandomForestGridSearch(n_estimators=8, gs_max_features=(1/3.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                min_samples_leaf=2,
                                criterion='mse', verbose=1, 
                                split_finder="brent",
                                tb_verbose=False, split_verbose=False, rand_state=None,
                                max_depth=None,
                                g_cap=None,
                                realize_iter=0,
                                bij_novelty='excl', 
                                median_predict=True,
                                oob_score=True,
                                n_jobs=-1,
                                # [DEPRECATED]
                                alpha_g_fit=0., 
                                **kwargs):
    from sklearn.ensemble import RandomForestRegressor
    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)        
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose,
                                 oob_score=oob_score, random_state=rand_state,
                                 max_depth=max_depth, 
                                 min_samples_leaf=min_samples_leaf, 
                                 criterion=criterion, 
                                 realize_iter=realize_iter, 
                                 split_verbose=split_verbose, 
                                 split_finder=split_finder, 
                                 tb_verbose=tb_verbose, 
                                 median_predict=median_predict, 
                                 bij_novelty=bij_novelty, 
                                 g_cap=g_cap,
                                 # [DEPRECATED]
                                 alpha_g_fit=alpha_g_fit)

    tuneparams = dict(max_features=gs_max_features,
                      min_samples_split=gs_min_samples_split,
                      alpha_g_split=gs_alpha_g_split)

    return rf, tuneparams


def setupAdaBoostGridSearch(n_estimators=50, gs_max_features=(1/3.,), gs_max_depth=(3,), gs_alpha_g_split=(0.,),                             gs_learning_rate=(0.1,),
                            cv=5,
                            gscv_verbose=1,
                            min_samples_leaf=2,
                            criterion='mse',
                            split_finder="brent",
                            tb_verbose=False, split_verbose=False, rand_state=None,
                            min_samples_split=4,
                            g_cap=None,
                            realize_iter=0,
                            bij_novelty='lim',
                            loss='linear', 
                            presort=True,
                            n_jobs=-1,
                            refit=False,
                            return_train_score=False,
                            # [DEPRECATED]
                            alpha_g_fit=0.,
                            **kwargs):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV
    
    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, (int, float)): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)
    base = DecisionTreeRegressor(presort=presort, tb_verbose=tb_verbose,
                                           min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 split_finder=split_finder, split_verbose=split_verbose,
                                 alpha_g_fit=alpha_g_fit,
                                 g_cap=g_cap,
                                 realize_iter=realize_iter,
                                 criterion=criterion)
    
    ab = AdaBoostRegressor(base_estimator=base,
                             n_estimators=n_estimators,
                             loss=loss,
                             random_state=rand_state,
                             bij_novelty=bij_novelty)

    tuneparams = dict(base_estimator__max_features=gs_max_features,
                      base_estimator__max_depth=gs_max_depth,
                      base_estimator__alpha_g_split=gs_alpha_g_split,
                      learning_rate=gs_learning_rate)

    # Construct GSCV for DT
    ab_gscv = GridSearchCV(ab,
                           cv=cv,
                           param_grid=tuneparams,
                           n_jobs=n_jobs,
                           error_score='raise',
                           verbose=gscv_verbose,
                           scoring=None,
                           refit=refit,
                           return_train_score=return_train_score,
                           # Mean score of folds is not sample weighted
                           iid=False)

    return ab_gscv, ab, tuneparams


@timer
def performEstimatorGridSearch(estimator_gs, tuneparams, x_train, y_train, tb_train=None, x_test=None, y_test=None, tb_test=None,
                               staged=False, refit=False,
                               **kwargs):
    from sklearn.model_selection import ParameterGrid
    import copy

    print(' {0}'.format(tuneparams))
    # Initialize best score and will be updated once found better
    best_score = -np.inf
    best_grid = {}
    best_estimator = copy.deepcopy(estimator_gs)
    for grid in ParameterGrid(tuneparams):
        estimator_gs.set_params(**grid)
        # Fit the train data for ML
        estimator_gs.fit(x_train, y_train, tb=tb_train)
        # If the estimator uses out-of-bag samples, then the score is oob_score_ that uses the train data
        if hasattr(estimator_gs, 'oob_score_'):
            print(' Using OOB score...')
            score = estimator_gs.oob_score_
        # Else, use the default/custom score method on the test data
        else:
            if staged and hasattr(estimator_gs, 'staged_score'):
                print('')
                for iboost, score_staged in enumerate(estimator_gs.staged_score(x_test, y_test, tb=tb_test, bij_novelty=estimator_gs.bij_novelty)):
                    # When iboost reaches the end, that score should be the best for this hyper-parameter grid
                    score = score_staged
                    print(' Staged score after boost {0}: {1}'.format(iboost + 1, score))
                print('')
            else:
                score = estimator_gs.score(x_test, y_test, tb=tb_test, bij_novelty=estimator_gs.bij_novelty)

        print(' Current score is {} for {}'.format(score, grid))
        # Save if best
        if score > best_score:
            print(' Old best score is {0}'.format(best_score))
            print(' New best score is {0}'.format(score))
            best_score = score
            best_grid = grid
            best_estimator = copy.deepcopy(estimator_gs)

    print(' Best [score] is {}'.format(best_score))
    print(' Best [hyper-parameters] are \n  {}'.format(best_grid))

    # Reassign estimator_gs to the best trained model so that it's reusable
    estimator_gs = best_estimator
    # Set the best hyper-parameters to the estimator
    estimator_gs.set_params(**best_grid)
    # The previous fits were cleared thus need to refit using the best hyper-parameters
    if refit:
        print(' Re-fitting the model using the best hyper-parameters...')
        estimator_gs.fit(x_train, y_train, tb=tb_train)

    return best_grid
