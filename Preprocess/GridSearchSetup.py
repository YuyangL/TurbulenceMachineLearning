import numpy as np
from warnings import warn
# from Utilities import timer
from Utility import timer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from shutil import rmtree
from sklearn.model_selection import ParameterGrid
import copy
from joblib import dump, load
from sklearn.base import clone
import time as t


def setupDecisionTreePipelineGridSearchCV(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                   min_samples_leaf=2,
                                   cv=5, presort=True, split_finder="brute",
                                   tb_verbose=False, split_verbose=False, rand_state=None, gs_verbose=1,
                                   max_depth=None,
                                   g_cap=None,
                                   realize_iter=0,
                                   n_jobs=-1,
                                   refit=False,
                                   return_train_score=False,
                                   var_threshold=0.,
                                   scaler=None,
                                   rf_selector=False,
                                   rf_selector_n_estimators=1000,
                                   rf_selector_threshold='median',
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
    :param gs_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gs_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector, rf_selector_n_estimators, rf_selector_threshold, 
                                                 verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    # Ensure hyper-parameters in a sequence
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    # Initialize decision tree regressor object
    tree = DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                  max_depth=max_depth,
                                  g_cap=g_cap,
                                  realize_iter=realize_iter,
                                  min_samples_leaf=min_samples_leaf,
                                  # [DEPRECATED]
                                  alpha_g_fit=alpha_g_fit)
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        regressor = tree
        # Append hyper-parameters to a dict, key words depend on pipeline or not
        tuneparams = dict(max_features=gs_max_features,
                          min_samples_split=gs_min_samples_split,
                          alpha_g_split=gs_alpha_g_split)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            regressor = Pipeline([('scaler', scaler),
                                 ('feat_selector', feat_selector),
                                 ('tree', tree)])
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                 ('tree', tree)])

        # Kwargs of the regressor parameters will bare a superscript "{regressor name}__"
        tuneparams = dict(tree__max_features=gs_max_features,
                          tree__min_samples_split=gs_min_samples_split,
                          tree__alpha_g_split=gs_alpha_g_split)
        # So is the kwarg to supply Tij to regressor.fit() method
        fit_param_key = 'tree__tb'

    # Construct GSCV for DT
    regressor_gscv = GridSearchCV(regressor,
                                 cv=cv,
                                 param_grid=tuneparams,
                                 n_jobs=n_jobs,
                                 error_score='raise',
                                 verbose=gs_verbose,
                                 scoring=None,
                                 refit=refit,
                                 return_train_score=return_train_score,
                                 # Mean score of folds is not sample weighted
                                 iid=False)

    return regressor_gscv, regressor, tuneparams, fit_param_key

@timer
def setupDecisionTreeGridSearchCV(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                  min_samples_leaf=2,
                                  cv=4, presort=True, split_finder="brute",
                                  tb_verbose=False, split_verbose=False, rand_state=None, gs_verbose=1,
                                  max_depth=None,
                                  g_cap=None,
                                  realize_iter=0,
                                  n_jobs=-1,
                                  return_train_score=False,
                                  var_threshold=0.,
                                  scaler=None,
                                  rf_selector_n_estimators=0,
                                  rf_selector_threshold='median',
                                  pipeline_cachedir=None,
                                  criterion='mse',
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
    :param gs_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gs_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators, rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    pipeline_verbose = True if gs_verbose > 1 else False
    # Ensure hyper-parameters in a sequence
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    # Initialize DT regressor object
    tree = DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                 max_depth=max_depth,
                                 g_cap=g_cap,
                                 realize_iter=realize_iter,
                                 min_samples_leaf=min_samples_leaf,
                                 criterion=criterion,
                                 # [DEPRECATED]
                                 alpha_g_fit=alpha_g_fit)
    # Hyper-parameters for GSCV
    tuneparams = dict(max_features=gs_max_features,
                      min_samples_split=gs_min_samples_split,
                      alpha_g_split=gs_alpha_g_split)
    # Initialize GSCV object for DT.
    # No refit as GSCV is done with GS data not the full train data
    tree_gscv = GridSearchCV(tree,
                             cv=cv,
                             param_grid=tuneparams,
                             n_jobs=n_jobs,
                             error_score='raise',
                             verbose=gs_verbose,
                             scoring=None,
                             refit=False,
                             return_train_score=return_train_score,
                             # Mean score of folds is not sample weighted
                             iid=False)
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        # No pipeline, only GSCV
        regressor, regressor_gscv = tree, tree_gscv
        # Tensor basis kwarg for fit method, will be different if Pipeline object
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            # Pipeline without GSCV object, to be used after GSCV
            regressor = Pipeline([('scaler', scaler),
                                  ('feat_selector', feat_selector),
                                  ('tree', tree)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose)
            # Pipeline with GSCV object as last step, for GSCV
            regressor_gscv = Pipeline([('scaler', scaler),
                                       ('feat_selector', feat_selector),
                                       ('tree', tree_gscv)],
                                      memory=pipeline_cachedir,
                                      verbose=pipeline_verbose)
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                  ('tree', tree)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose)
            regressor_gscv = Pipeline([('feat_selector', feat_selector),
                                       ('tree', tree_gscv)],
                                      memory=pipeline_cachedir,
                                      verbose=pipeline_verbose)

        # So is the kwarg to supply Tij to regressor.fit() method
        fit_param_key = 'tree__tb'

    return regressor_gscv, regressor, tuneparams, fit_param_key


def setupRandomForestGridSearch(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                n_estimators=8,
                                min_samples_leaf=2,
                                criterion='mse', split_finder="brute",
                                tb_verbose=False, split_verbose=False, rand_state=None, gs_verbose=1,
                                max_depth=None,
                                g_cap=None,
                                realize_iter=0,
                                median_predict=True,
                                oob_score=True,
                                n_jobs=-1,
                                var_threshold=0.,
                                scaler=None,
                                rf_selector_n_estimators=0,
                                rf_selector_threshold='median',
                                pipeline_cachedir=None,
                                # [DEPRECATED]
                                alpha_g_fit=0.,
                                bij_novelty=None,
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
    :param gs_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gs_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators, rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    pipeline_verbose = True if gs_verbose > 1 else False
    # Ensure hyper-parameters in a sequence
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    # Initialize RF regressor object
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, verbose=max(gs_verbose - 1, 0),
                               oob_score=oob_score, random_state=rand_state,
                               max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf,
                               criterion=criterion,
                               realize_iter=realize_iter,
                               split_verbose=split_verbose,
                               split_finder=split_finder,
                               tb_verbose=tb_verbose,
                               median_predict=median_predict,
                               g_cap=g_cap,
                               bootstrap=True,
                               # [DEPRECATED]
                               alpha_g_fit=alpha_g_fit,
                               bij_novelty=bij_novelty)
    # Hyper-parameters for GS
    tuneparams = dict(max_features=gs_max_features,
                      min_samples_split=gs_min_samples_split,
                      alpha_g_split=gs_alpha_g_split)

    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        # No pipeline, only GS
        regressor, regressor_gs = (rf,)*2
        # Tensor basis kwarg for fit method, will be different if Pipeline object
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            # Pipeline with and without GS object
            regressor, regressor_gs = (Pipeline([('scaler', scaler),
                                  ('feat_selector', feat_selector),
                                  ('rf', rf)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose),)*2
        # Otherwise, it's a TBRF feature selector
        else:
            regressor, regressor_gs = (Pipeline([('feat_selector', feat_selector),
                                  ('rf', rf)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose),)*2

        # So is the kwarg to supply Tij to regressor.fit() method
        fit_param_key = 'rf__tb'

    return regressor_gs, regressor, tuneparams, fit_param_key


def setupRandomForestPipelineGridSearch(n_estimators=8, gs_max_features=(1/3.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                min_samples_leaf=2,
                                criterion='mse', gs_verbose=1, 
                                split_finder="brute",
                                tb_verbose=False, split_verbose=False, rand_state=None,
                                max_depth=None,
                                g_cap=None,
                                realize_iter=0,
                                bij_novelty=None,
                                median_predict=True,
                                oob_score=True,
                                n_jobs=-1,
                                var_threshold=0.,
                                scaler=None,
                                rf_selector_n_estimators=0,
                                rf_selector_threshold='median',
                                # [DEPRECATED]
                                alpha_g_fit=0., 
                                **kwargs):
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators,
                                                  rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    # Initialize random forest regressor object
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, verbose=gs_verbose,
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
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        regressor = rf
        # Append hyper-parameters to a dict, key words depend on pipeline or not
        tuneparams = dict(max_features=gs_max_features,
                          min_samples_split=gs_min_samples_split,
                          alpha_g_split=gs_alpha_g_split)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            regressor = Pipeline([('scaler', scaler),
                                 ('feat_selector', feat_selector),
                                 ('rf', rf)])
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                 ('rf', rf)])

        # Kwargs of the regressor parameters will bare a superscript "{regressor name}__"
        tuneparams = dict(rf__max_features=gs_max_features,
                          rf__min_samples_split=gs_min_samples_split,
                          rf__alpha_g_split=gs_alpha_g_split)
        # So is the kwarg to supply Tij to regressor.fit() method  
        fit_param_key = 'rf__tb'

    return regressor, tuneparams, fit_param_key


def setupAdaBoostGridSearchCV(gs_max_features=(1.,), gs_max_depth=(5,), gs_alpha_g_split=(0.,), gs_learning_rate=(0.1,),
                              n_estimators=16,
                              min_samples_leaf=2,
                              min_samples_split=4,
                              cv=4, presort=True, split_finder="brute",
                              tb_verbose=False, split_verbose=False, rand_state=None, gs_verbose=1,
                              g_cap=None,
                              realize_iter=0,
                              n_jobs=-1,
                              loss='square',
                              return_train_score=False,
                              var_threshold=0.,
                              scaler=None,
                              rf_selector_n_estimators=0,
                              rf_selector_threshold='median',
                              pipeline_cachedir=None,
                              criterion='mse',
                              # [DEPRECATED]
                              alpha_g_fit=0.,
                              bij_novelty=None,
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
    :param gs_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gs_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators, rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    pipeline_verbose = True if gs_verbose > 1 else False
    # Ensure hyper-parameters in a sequence
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, int): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)
    # Initialize DT regressor object as base estimator
    base = DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                 min_samples_split=min_samples_split,
                                 g_cap=g_cap,
                                 realize_iter=realize_iter,
                                 min_samples_leaf=min_samples_leaf,
                                 criterion=criterion,
                                 # [DEPRECATED]
                                 alpha_g_fit=alpha_g_fit)
    # Initialize AdaBoost object
    ab = AdaBoostRegressor(base_estimator=base,
                           n_estimators=n_estimators,
                           loss=loss,
                           random_state=rand_state,
                           # [DEPRECATED]
                           bij_novelty=bij_novelty)
    # Hyper-parameters for GSCV
    tuneparams = dict(base_estimator__max_features=gs_max_features,
                      base_estimator__max_depth=gs_max_depth,
                      base_estimator__alpha_g_split=gs_alpha_g_split,
                      learning_rate=gs_learning_rate)
    # Initialize GSCV object for AdaBoost.
    # No refit as GSCV is done with GS data not the full train data
    ab_gscv = GridSearchCV(ab,
                             cv=cv,
                             param_grid=tuneparams,
                             n_jobs=n_jobs,
                             error_score='raise',
                             verbose=gs_verbose,
                             scoring=None,
                             refit=False,
                             return_train_score=return_train_score,
                             # Mean score of folds is not sample weighted
                             iid=False)
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        # No pipeline, only GSCV
        regressor, regressor_gscv = ab, ab_gscv
        # Tensor basis kwarg for fit method, will be different if Pipeline object
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            # Pipeline without GSCV object, to be used after GSCV
            regressor = Pipeline([('scaler', scaler),
                                  ('feat_selector', feat_selector),
                                  ('ab', ab)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose)
            # Pipeline with GSCV object as last step, for GSCV
            regressor_gscv = Pipeline([('scaler', scaler),
                                       ('feat_selector', feat_selector),
                                       ('ab', ab_gscv)],
                                      memory=pipeline_cachedir,
                                      verbose=pipeline_verbose)
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                  ('ab', ab)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose)
            regressor_gscv = Pipeline([('feat_selector', feat_selector),
                                       ('ab', ab_gscv)],
                                      memory=pipeline_cachedir,
                                      verbose=pipeline_verbose)

        # So is the kwarg to supply Tij to regressor.fit() method
        fit_param_key = 'ab__tb'

    return regressor_gscv, regressor, tuneparams, fit_param_key


def setupAdaBoostPipelineGridSearchCV(n_estimators=50, gs_max_features=(1/3.,), gs_max_depth=(3,), gs_alpha_g_split=(0.,),                             gs_learning_rate=(0.1,),
                            cv=5,
                            gs_verbose=1,
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
                            var_threshold=0.,
                            scaler=None,
                            rf_selector_n_estimators=1000,
                            rf_selector_threshold='median',
                            # [DEPRECATED]
                            alpha_g_fit=0.,
                            **kwargs):
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators,
                                                  rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, (int, float)): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)

    # Initialize wake tree learner object
    base = DecisionTreeRegressor(presort=presort, tb_verbose=tb_verbose,
                                           min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 split_finder=split_finder, split_verbose=split_verbose,
                                 g_cap=g_cap,
                                 realize_iter=realize_iter,
                                 criterion=criterion,
                                 # [DEPRECATED]
                                 alpha_g_fit=alpha_g_fit)
    # Initialize AdaBoost object
    ab = AdaBoostRegressor(base_estimator=base,
                             n_estimators=n_estimators,
                             loss=loss,
                             random_state=rand_state,
                             bij_novelty=bij_novelty)
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        regressor = ab
        tuneparams = dict(base_estimator__max_features=gs_max_features,
                          base_estimator__max_depth=gs_max_depth,
                          base_estimator__alpha_g_split=gs_alpha_g_split,
                          learning_rate=gs_learning_rate)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            regressor = Pipeline([('scaler', scaler),
                                 ('feat_selector', feat_selector),
                                 ('ab', ab)])
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                 ('ab', ab)])

        # Kwargs of the regressor parameters will bare a superscript "{regressor name}__"
        tuneparams = dict(ab__base_estimator__max_features=gs_max_features,
                          ab__base_estimator__max_depth=gs_max_depth,
                                ab__base_estimator__alpha_g_split=gs_alpha_g_split,
                          ab__learning_rate=gs_learning_rate)
        # So is the kwarg to supply Tij to regressor.fit() method
        fit_param_key = 'ab__tb'

    # Construct GSCV for DT
    ab_gscv = GridSearchCV(regressor,
                           cv=cv,
                           param_grid=tuneparams,
                           n_jobs=n_jobs,
                           error_score='raise',
                           verbose=gs_verbose,
                           scoring=None,
                           refit=refit,
                           return_train_score=return_train_score,
                           # Mean score of folds is not sample weighted
                           iid=False)

    return ab_gscv, regressor, tuneparams, fit_param_key


def setupGradientBoostGridSearchCV(gs_max_features=(1.,), gs_max_depth=(5,), gs_alpha_g_split=(0.,), gs_learning_rate=(0.1,),
                                    n_estimators=16,
                                    subsample=1.,
                                   n_iter_no_change=None,
                                   tol=1e-4,
                                   init='zero',
                                   validation_fraction=0.1,
                                   alpha=0.9,
                                    min_samples_leaf=2,
                                    min_samples_split=0.002,
                                    cv=4, presort=True, split_finder="brute",
                                    tb_verbose=False, split_verbose=False, rand_state=None, gs_verbose=1,
                                    g_cap=None,
                                    realize_iter=0,
                                    n_jobs=-1,
                                    loss='square',
                                    return_train_score=False,
                                    var_threshold=0.,
                                    scaler=None,
                                    rf_selector_n_estimators=0,
                                    rf_selector_threshold='median',
                                    pipeline_cachedir=None,
                                    criterion='mse',
                                    # [DEPRECATED]
                                    alpha_g_fit=0.,
                                    bij_novelty=None,
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
    :param gs_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gs_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators, rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    pipeline_verbose = True if gs_verbose > 1 else False
    # Ensure hyper-parameters in a sequence
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, int): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)
    # Initialize GradientBoosting object
    gb = GradientBoostingRegressor(subsample=subsample,
                                   criterion=criterion,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   verbose=max(gs_verbose - 1, 0),
                                   presort=presort,
                                   n_iter_no_change=n_iter_no_change,
                                   tol=tol,
                                   init=init,
                                   realize_iter=realize_iter,
                                   tb_verbose=tb_verbose,
                                   split_verbose=split_verbose,
                                   split_finder=split_finder,
                                   g_cap=g_cap,
                                   n_estimators=n_estimators,
                                   loss=loss,
                                   random_state=rand_state,
                                   validation_fraction=validation_fraction,
                                   alpha=alpha,
                                   # [DEPRECATEDD]
                                   alpha_g_fit=alpha_g_fit,
                                   bij_novelty=bij_novelty)

    # Hyper-parameters for GSCV
    tuneparams = dict(max_features=gs_max_features,
                      max_depth=gs_max_depth,
                      alpha_g_split=gs_alpha_g_split,
                      learning_rate=gs_learning_rate)
    # Initialize GSCV object for AdaBoost.
    # No refit as GSCV is done with GS data not the full train data
    gb_gscv = GridSearchCV(gb,
                           cv=cv,
                           param_grid=tuneparams,
                           n_jobs=n_jobs,
                           error_score='raise',
                           verbose=gs_verbose,
                           scoring=None,
                           refit=False,
                           return_train_score=return_train_score,
                           # Mean score of folds is not sample weighted
                           iid=False)
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        # No pipeline, only GSCV
        regressor, regressor_gscv = gb, gb_gscv
        # Tensor basis kwarg for fit method, will be different if Pipeline object
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            # Pipeline without GSCV object, to be used after GSCV
            regressor = Pipeline([('scaler', scaler),
                                  ('feat_selector', feat_selector),
                                  ('gb', gb)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose)
            # Pipeline with GSCV object as last step, for GSCV
            regressor_gscv = Pipeline([('scaler', scaler),
                                       ('feat_selector', feat_selector),
                                       ('gb', gb_gscv)],
                                      memory=pipeline_cachedir,
                                      verbose=pipeline_verbose)
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                  ('gb', gb)],
                                 memory=pipeline_cachedir,
                                 verbose=pipeline_verbose)
            regressor_gscv = Pipeline([('feat_selector', feat_selector),
                                       ('gb', gb_gscv)],
                                      memory=pipeline_cachedir,
                                      verbose=pipeline_verbose)

        # So is the kwarg to supply Tij to regressor.fit() method
        fit_param_key = 'gb__tb'

    return regressor_gscv, regressor, tuneparams, fit_param_key


def setupGradientBoostPipelineGridSearchCV(n_estimators=50, gs_max_features=(1/3.,), gs_max_depth=(3,), gs_alpha_g_split=(0.,),                             gs_learning_rate=(0.1,),
                                 subsample=1., n_iter_no_change=None, tol=1e-4,

                                    cv=4,
                                    gs_verbose=1,
                                    min_samples_leaf=2,
                                    criterion='mse',
                                    split_finder="brent",
                                    tb_verbose=False, split_verbose=False, rand_state=None,
                                    min_samples_split=4,
                                    g_cap=None,
                                    realize_iter=0,
                                    bij_novelty='lim',
                                    loss='ls',
                                    presort=True,
                                    n_jobs=-1,
                                    refit=False,
                                    return_train_score=False,
                                    var_threshold=0.,
                                    scaler=None,
                                   rf_selector_n_estimators=1000,
                                   rf_selector_threshold='median',
                                   validation_fraction=0.1,
                                   alpha=0.9,
                                   init='zero',
                            # [DEPRECATED]
                            alpha_g_fit=0.,
                            **kwargs):
    from sklearn.ensemble import GradientBoostingRegressor

    # Setup feature selector and scaler, could both be None
    feat_selector, scaler = _setupFeatureSelector(var_threshold, scaler, rf_selector_n_estimators,
                                                  rf_selector_threshold,
                                                  verbose=max(gs_verbose - 1, 0), n_jobs=n_jobs)
    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, (int, float)): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)
    # Initialize GradientBoosting object
    gb = GradientBoostingRegressor(subsample=subsample,
                                   criterion=criterion,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   verbose=max(gs_verbose - 1, 0),
                                   presort=presort,
                                   n_iter_no_change=n_iter_no_change,
                                   tol=tol,
                                   init=init,
                                   realize_iter=realize_iter,
                                   tb_verbose=tb_verbose,
                                   split_verbose=split_verbose,
                                   split_finder=split_finder,
                                   g_cap=g_cap,
                                    n_estimators=n_estimators,
                                    loss=loss,
                                    random_state=rand_state,
                                    bij_novelty=bij_novelty,
                                   validation_fraction=validation_fraction,
                                   alpha=alpha,
                                   # [DEPRECATEDD]
                                   alpha_g_fit=alpha_g_fit)
    # If feature selection, a pipeline object is necessary
    if feat_selector is None:
        regressor = gb
        # Append hyper-parameters to a dict, key words depend on pipeline or not
        tuneparams = dict(max_features=gs_max_features,
                          max_depth=gs_max_depth,
                          alpha_g_split=gs_alpha_g_split,
                          learning_rate=gs_learning_rate)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    # Else if feature selection requested
    else:
        # If scaler is not None, i.e. the case of variance threshold filter
        if scaler is not None:
            regressor = Pipeline([('scaler', scaler),
                                 ('feat_selector', feat_selector),
                                 ('gb', gb)])
        # Otherwise, it's a TBRF feature selector
        else:
            regressor = Pipeline([('feat_selector', feat_selector),
                                 ('gb', gb)])

        # Kwargs of the regressor parameters will bare a superscript "{regressor name}__"
        tuneparams = dict(gb__max_features=gs_max_features,
                          gb__max_depth=gs_max_depth,
                          gb__alpha_g_split=gs_alpha_g_split,
                          gb__learning_rate=gs_learning_rate)
        fit_param_key = 'gb__tb'

    # Construct GSCV for DT
    gb_gscv = GridSearchCV(regressor,
                           cv=cv,
                           param_grid=tuneparams,
                           n_jobs=n_jobs,
                           error_score='raise',
                           verbose=gs_verbose,
                           scoring=None,
                           refit=refit,
                           return_train_score=return_train_score,
                           # Mean score of folds is not sample weighted
                           iid=False)

    return gb_gscv, regressor, tuneparams, fit_param_key


def performEstimatorGridSearchCV_Dask(estimator_gscv, estimator_final, x_gs, y_gs,
                                 tb_kw='tb', tb_gs=None,
                                 x_train=None, y_train=None, tb_train=None,
                                 gs=True, refit=True,
                                 save=True, savedir='./', gscv_name='GSCV', final_name='final',
                                      cores=32,
                                      walltime='24:00:00',
                                      memory='96GB',
                                 **kwargs):
    from dask_jobqueue import PBSCluster
    from dask.distributed import Client
    from joblib import parallel_backend
    cluster = PBSCluster(cores=cores,
                         interface='eth0',
                         walltime=walltime,
                         memory=memory)
    client = Client(cluster)

    # FIXME: memory=pipeline_cachedir in Pipeline() not working properly here
    # If refit is enabled i.e. train after GSCV,
    # and if any of the train data is not provided, assume data for train is the same as GSCV
    if refit and x_train is None: x_train = x_gs.copy()
    if refit and y_train is None: y_train = y_gs.copy()
    if refit and tb_train is None: tb_train = tb_gs.copy()
    # Tij fit kwarg to be passed to fit() method during GSCV
    fit_param = {tb_kw: tb_gs}
    # GSCV, will also fit feature selector if estimator_gscv is a pipeline
    if gs:
        print('\nPerforming estimator GSCV...')
        with parallel_backend('dask'):
            estimator_gscv.fit(x_gs, y_gs, **fit_param)

        if save:
            # Save the GSCV for further inspection
            dump(estimator_gscv, savedir + '/' + gscv_name + '.joblib')
            print('\nFitted {0} saved at {1}'.format(gscv_name, savedir))

    is_pipeline = True if hasattr(estimator_gscv, 'steps') else False
    # Best hyper-parameters found through GSCV.
    # best_params_ stored in final step of the estimator_gs pipeline or directly in estimator_gs
    best_params = estimator_gscv._final_estimator.best_params_ if is_pipeline else estimator_gscv.best_params_
    # If there's feature selection i.e. estimator_gscv and estimator_final are pipelines
    if is_pipeline:
        # If pipeline, step names excl. the actual regressor name
        selector_steps = [tuple[0] for tuple in estimator_gscv.steps[:-1]]
        # Then assign the fitted selectors to the unfitted final estimator pipeline
        for i, name in enumerate(selector_steps):
            estimator_final.named_steps[name] = copy.deepcopy(estimator_gscv.named_steps[name])
            estimator_final.steps[i] = copy.deepcopy(estimator_gscv.steps[i])
            # If refit, use the fitted feature selector to transform training x
            if refit:
                with parallel_backend('dask'):#, scatter=[x_train]):
                    x_train = estimator_final.named_steps[name].transform(x_train)

        # Lastly, assign the found best hyper-parameters to estimator_final
        estimator_final._final_estimator.set_params(**best_params)
    else:
        estimator_final.set_params(**best_params)

    print('\nBest hyper-parameters assigned to regressor: {}'.format(best_params))
    # Now we can start actual training, if refit is requested
    if refit:
        print('\nRe-fitting estimator with best hyper-parameters and training data...')
        # If pipeline, only fit the regressor since the feature selector has been fitted already during GSCV.
        # Also, x_train has been transformed by feature selector already above
        with parallel_backend('dask'):#, scatter=[x_train, y_train, tb_train]):
            if is_pipeline:
                estimator_final._final_estimator.fit(x_train, y_train, tb=tb_train)
            # Otherwise, estimator_final itself is the regressor object
            else:
                estimator_final.fit(x_train, y_train, tb=tb_train)

        if save:
            # Save the final fitted regressor
            dump(estimator_final, savedir + '/' + final_name + '.joblib')
            print('\nFitted {0} saved at {1}'.format(final_name, savedir))

    # If transformers in pipelines have been cached, remove them after fitting
    if is_pipeline:
        if estimator_gscv.memory is not None: rmtree(estimator_gscv.memory)
        if refit and estimator_final.memory is not None: rmtree(estimator_final.memory)

    # The pipeline or simply regressor after GSCV.
    # If pipeline, the feature selector is already fitted while the actual regressor might not depending on whether train data is supplied
    return estimator_final, best_params


def performEstimatorGridSearchCV(estimator_gscv, estimator_final, x_gs, y_gs,
                                 tb_kw='tb', tb_gs=None,
                                 x_train=None, y_train=None, tb_train=None,
                                 gs=True, refit=True,
                                 save=True, savedir='./', gscv_name='GSCV', final_name='final',
                                 **kwargs):
    # FIXME: memory=pipeline_cachedir in Pipeline() not working properly here
    # If refit is enabled i.e. train after GSCV,
    # and if any of the train data is not provided, assume data for train is the same as GSCV
    if refit and x_train is None: x_train = x_gs.copy()
    if refit and y_train is None: y_train = y_gs.copy()
    if refit and tb_train is None: tb_train = tb_gs.copy()
    # Tij fit kwarg to be passed to fit() method during GSCV
    fit_param = {tb_kw: tb_gs}
    # GSCV, will also fit feature selector if estimator_gscv is a pipeline
    if gs:
        print('\nPerforming estimator GSCV...')
        estimator_gscv.fit(x_gs, y_gs, **fit_param)
        if save:
            # Save the GSCV for further inspection
            dump(estimator_gscv, savedir + '/' + gscv_name + '.joblib')
            print('\nFitted {0} saved at {1}'.format(gscv_name, savedir))

    is_pipeline = True if hasattr(estimator_gscv, 'steps') else False
    # Best hyper-parameters found through GSCV.
    # best_params_ stored in final step of the estimator_gs pipeline or directly in estimator_gs
    best_params = estimator_gscv._final_estimator.best_params_ if is_pipeline else estimator_gscv.best_params_
    # If there's feature selection i.e. estimator_gscv and estimator_final are pipelines
    if is_pipeline:
        # If pipeline, step names excl. the actual regressor name
        selector_steps = [tuple[0] for tuple in estimator_gscv.steps[:-1]]
        # Then assign the fitted selectors to the unfitted final estimator pipeline
        for i, name in enumerate(selector_steps):
            estimator_final.named_steps[name] = copy.deepcopy(estimator_gscv.named_steps[name])
            estimator_final.steps[i] = copy.deepcopy(estimator_gscv.steps[i])
            # If refit, use the fitted feature selector to transform training x
            if refit: x_train = estimator_final.named_steps[name].transform(x_train)

        # Lastly, assign the found best hyper-parameters to estimator_final
        estimator_final._final_estimator.set_params(**best_params)
    else:
        estimator_final.set_params(**best_params)

    print('\nBest hyper-parameters assigned to regressor: {}'.format(best_params))
    # Now we can start actual training, if refit is requested
    if refit:
        print('\nRe-fitting estimator with best hyper-parameters and training data...')
        t0 = t.time()
        # If pipeline, only fit the regressor since the feature selector has been fitted already during GSCV.
        # Also, x_train has been transformed by feature selector already above
        if is_pipeline:
            estimator_final._final_estimator.fit(x_train, y_train, tb=tb_train)
        # Otherwise, estimator_final itself is the regressor object
        else:
            estimator_final.fit(x_train, y_train, tb=tb_train)

        t1 = t.time()
        print('\nFinished {0} in {1:.4f} min'.format(final_name, (t1 - t0)/60.))
        if save:
            # Save the final fitted regressor
            dump(estimator_final, savedir + '/' + final_name + '.joblib')
            print('\nFitted {0} saved at {1}'.format(final_name, savedir))

    # If transformers in pipelines have been cached, remove them after fitting
    if is_pipeline:
        if estimator_gscv.memory is not None: rmtree(estimator_gscv.memory)
        if refit and estimator_final.memory is not None: rmtree(estimator_final.memory)

    # The pipeline or simply regressor after GSCV.
    # If pipeline, the feature selector is already fitted while the actual regressor might not depending on whether train data is supplied
    return estimator_final, best_params


def performEstimatorGridSearch(estimator_gs, estimator_final, tuneparams, x_gs, y_gs,
                               tb_kw='tb', tb_gs=None, x_train=None, y_train=None, tb_train=None,
                               x_test=None, y_test=None, tb_test=None,
                               gs=True, refit=True,
                               save=True, savedir='./', gs_name='GS', final_name='final',
                               **kwargs):
    # FIXME: memory=pipeline_cachedir in Pipeline() not working properly here
    print('\nHyper-parameter grid: {0}'.format(tuneparams))
    # If refit is enabled i.e. train after GS,
    # and if any of the train/test data is not provided, assume data for train/test is the same as GS
    if refit and x_train is None: x_train = x_gs.copy()
    if refit and y_train is None: y_train = y_gs.copy()
    if refit and tb_train is None: tb_train = tb_gs.copy()
    if refit and x_test is None: x_test = x_gs.copy()
    if refit and y_test is None: y_test = y_gs.copy()
    if refit and tb_test is None: tb_test = tb_gs.copy()
    # Tij fit kwarg to be passed to fit() method during GS
    fit_param = {tb_kw: tb_gs}
    is_pipeline = True if hasattr(estimator_gs, 'steps') else False
    if gs:
        # Initialize best score and will be updated once found better
        best_score = -np.inf
        best_grid = {}
        best_estimator = copy.deepcopy(estimator_gs)
        print('\nPerforming estimator GS...')
        for grid in ParameterGrid(tuneparams):
            if is_pipeline:
                estimator_gs._final_estimator.set_params(**grid)
            else:
                estimator_gs.set_params(**grid)

            # Fit the GS data while also fitting feature selector if pipeline
            estimator_gs.fit(x_gs, y_gs, **fit_param)
            # If the estimator uses out-of-bag samples, then the score is oob_score_ that uses the train data
            estimator_gs_final = estimator_gs.steps[-1][1] if is_pipeline else estimator_gs
            if hasattr(estimator_gs_final, 'oob_score_'):
                print(' Using OOB score...')
                score = estimator_gs_final.oob_score_
            # Else, use the default/custom score method on the test data
            else:
                score = estimator_gs.score(x_test, y_test, tb=tb_test)

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
        # Reassign estimator_gs to the best fitted model so that it's reusable
        estimator_gs = best_estimator
        if save:
            # Save the GS for further inspection
            dump(estimator_gs, savedir + '/' + gs_name + '.joblib')
            print('\nFitted {0} saved at {1}'.format(gs_name, savedir))

    # If there's feature selection i.e. estimator_gs and estimator_final are pipelines
    if is_pipeline:
        # If pipeline, step names excl. the actual regressor name
        selector_steps = [tuple[0] for tuple in estimator_gs.steps[:-1]]
        # Then assign the fitted selectors to the unfitted final estimator pipeline
        for i, name in enumerate(selector_steps):
            estimator_final.named_steps[name] = copy.deepcopy(estimator_gs.named_steps[name])
            estimator_final.steps[i] = copy.deepcopy(estimator_gs.steps[i])
            # If refit, use the fitted feature selector to transform training x
            if refit: x_train = estimator_final.named_steps[name].transform(x_train)

        # Set the best hyper-parameters to the estimator
        regressor_name = estimator_gs.steps[-1][0]
        # estimator_final._final_estimator.set_params(**best_grid)
        estimator_final.named_steps[regressor_name] = clone(estimator_gs.named_steps[regressor_name])
    else:
        # estimator_final.set_params(**best_grid)
        estimator_final = clone(estimator_gs)

    print('\nBest hyper-parameters assigned to regressor')
    # The previous fits were cleared thus need to refit using the best hyper-parameters
    if refit:
        print('\nRe-fitting estimator with best hyper-parameters and training data...')
        t0 = t.time()
        # If pipeline, only fit the regressor since the feature selector has been fitted already during GS.
        # Also, x_train has been transformed by feature selector already above
        if is_pipeline:
            estimator_final._final_estimator.fit(x_train, y_train, tb=tb_train)
        # Otherwise, estimator_final itself is the regressor object
        else:
            estimator_final.fit(x_train, y_train, tb=tb_train)

        t1 = t.time()
        print('\nFinished {0} in {1:.4f} min'.format(estimator_final, (t1 - t0)/60.))
        if save:
            # Save the final fitted regressor
            dump(estimator_final, savedir + '/' + final_name + '.joblib')
            print('\nFitted {0} saved at {1}'.format(final_name, savedir))

        # If transformers in pipelines have been cached, remove them after fitting
    if is_pipeline:
        if estimator_gs.memory is not None: rmtree(estimator_gs.memory)
        if refit and estimator_final.memory is not None: rmtree(estimator_final.memory)

    # The pipeline or simply regressor after GS.
    # If pipeline, the feature selector is already fitted while the actual regressor might not depending on whether train data is supplied
    return estimator_final, best_grid


@timer
def performEstimatorPipelineGridSearch(estimator_gs, tuneparams, x_train, y_train,
                               tb_kw='tb', tb_train=None, x_test=None, y_test=None, tb_test=None,
                               staged=False, refit=False,
                               **kwargs):
    from sklearn.model_selection import ParameterGrid
    import copy

    print(' {0}'.format(tuneparams))
    is_pipeline = True if hasattr(estimator_gs, 'steps') else False
    fit_params = {}
    fit_params[tb_kw] = tb_train
    # Initialize best score and will be updated once found better
    best_score = -np.inf
    best_grid = {}
    best_estimator = copy.deepcopy(estimator_gs)
    for grid in ParameterGrid(tuneparams):
        estimator_gs.set_params(**grid)
        # Fit the train data for ML
        estimator_gs.fit(x_train, y_train, **fit_params)
        # If the estimator uses out-of-bag samples, then the score is oob_score_ that uses the train data
        estimator_gs_final = estimator_gs.steps[-1][1] if is_pipeline else estimator_gs
        if hasattr(estimator_gs_final, 'oob_score_'):
            print(' Using OOB score...')
            score = estimator_gs_final.oob_score_
        # Else, use the default/custom score method on the test data
        else:
            if staged and hasattr(estimator_gs_final, 'staged_score'):
                # FIXME: no transform is done here
                print('')
                for iboost, score_staged in enumerate(estimator_gs_final.staged_score(x_test, y_test, tb=tb_test)):
                    # When iboost reaches the end, that score should be the best for this hyper-parameter grid
                    score = score_staged
                    print(' Staged score after boost {0}: {1}'.format(iboost + 1, score))
                print('')
            else:
                score = estimator_gs.score(x_test, y_test, tb=tb_test)

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
        estimator_gs.fit(x_train, y_train, **fit_params)

    return best_grid


def _setupFeatureSelector(var_threshold=0., scaler=None, rf_selector_n_estimators=0, rf_selector_threshold='median',
                          verbose=1, n_jobs=-1):
    if rf_selector_n_estimators > 0:
        feat_selector = SelectFromModel(RandomForestRegressor(n_estimators=rf_selector_n_estimators,
                                                              max_depth=3,
                                                              split_finder='brent',
                                                              median_predict=True,
                                                              oob_score=True,
                                                              n_jobs=n_jobs,
                                                              verbose=verbose,
                                                              max_features=2/3.,
                                                              bootstrap=True,
                                                              min_samples_leaf=2,
                                                              criterion='mse'), threshold=rf_selector_threshold)
        scaler = None
    elif var_threshold > 0.:
        if scaler is None:
            warn('\nVariance threshold > 0 but scaler is not chosen, setting to default Max Absolute Scaler\n',
                 stacklevel=2)
            scaler = 'maxabs'

        # Initialize scaler object
        if scaler in ('maxabs', 'MaxAbs'):
            scaler = MaxAbsScaler()
        elif scaler in ('minmax', 'MinMax'):
            scaler = MinMaxScaler()

        # Initialize variance threshold object for feature selection if requested
        feat_selector = VarianceThreshold(threshold=var_threshold)
    else:
        feat_selector = None
        scaler = None
        
    return feat_selector, scaler
