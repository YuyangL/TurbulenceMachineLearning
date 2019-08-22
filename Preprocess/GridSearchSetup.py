import numpy as np
from warnings import warn
from Utilities import timer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


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
def setupDecisionTreeGridSearchCV2(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                  min_samples_leaf=2,
                                  cv=5, presort=True, split_finder="brute",
                                  tb_verbose=False, split_verbose=False, rand_state=None, gscv_verbose=1,
                                  max_depth=None,
                                  g_cap=None,
                                  realize_iter=0,
                                  n_jobs=-1,
                                  refit=False,
                                  return_train_score=False,
                                   var_threshold=0.,
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
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold

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

    pipeline = Pipeline([('feat_selector', VarianceThreshold(var_threshold)),
                         ('tree', tree)])

    # Append hyper-parameters to a dict, depending on pipeline or not
    tuneparams = dict(tree__max_features=gs_max_features,
                      tree__min_samples_split=gs_min_samples_split,
                      tree__alpha_g_split=gs_alpha_g_split)

    # Construct GSCV for DT
    tree_gscv = GridSearchCV(pipeline,
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

    return tree_gscv, pipeline, tuneparams


@timer
def setupDecisionTreeGridSearchCV3(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                   min_samples_leaf=2,
                                   cv=5, presort=True, split_finder="brute",
                                   tb_verbose=False, split_verbose=False, rand_state=None, gscv_verbose=1,
                                   max_depth=None,
                                   g_cap=None,
                                   realize_iter=0,
                                   n_jobs=-1,
                                   refit=False,
                                   return_train_score=False,
                                   var_threshold=0.,
                                   scaler=None,
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
    if var_threshold > 0. and scaler is None:
        warn('\nVariance threshold > 0 but scaler is not chosen, setting to default Max Absolute Scaler\n',
             stacklevel=2)
        scaler = 'maxabs'

    # Ensure hyper-parameters in a sequence
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    # Initialize scaler object
    if scaler in ('maxabs', 'MaxAbs'):
        scaler_ = MaxAbsScaler()
    elif scaler in ('minmax', 'MinMax'):
        scaler_ = MinMaxScaler()
    else:
        scaler_ = None

    # Initialize variance threshold object for feature selection if requested
    if var_threshold > 0.:
        feat_selector_ = VarianceThreshold(threshold=var_threshold)
    else:
        feat_selector_ = None

    # Initialize decision tree regressor object
    tree_ = DecisionTreeRegressor(random_state=rand_state, tb_verbose=tb_verbose, split_verbose=split_verbose, split_finder=split_finder, presort=presort,
                                 max_depth=max_depth,
                                 g_cap=g_cap,
                                 realize_iter=realize_iter,
                                 min_samples_leaf=min_samples_leaf,
                                 # [DEPRECATED]
                                 alpha_g_fit=alpha_g_fit)
    # If feature selection, a pipeline object is necessary
    if scaler_ is None:
        pipeline = tree_
        # Append hyper-parameters to a dict, depending on pipeline or not
        tuneparams = dict(max_features=gs_max_features,
                          min_samples_split=gs_min_samples_split,
                          alpha_g_split=gs_alpha_g_split)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    else:
        if feat_selector_ is not None:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('feat_selector', feat_selector_),
                                 ('tree', tree_)])
        else:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('tree', tree_)])

        tuneparams = dict(tree__max_features=gs_max_features,
                          tree__min_samples_split=gs_min_samples_split,
                          tree__alpha_g_split=gs_alpha_g_split)
        fit_param_key = 'tree__tb'


    # Construct GSCV for DT
    regressor_gscv = GridSearchCV(pipeline,
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

    return regressor_gscv, pipeline, tuneparams, fit_param_key


@timer
def setupDecisionTreeGridSearchCV4(gs_max_features=(1.,), gs_min_samples_split=(2,), gs_alpha_g_split=(0.,),
                                   min_samples_leaf=2,
                                   cv=5, presort=True, split_finder="brute",
                                   tb_verbose=False, split_verbose=False, rand_state=None, gscv_verbose=1,
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
    :param gscv_verbose: Whether to verbose grid search cross-validation process.
    If 0, then verbose is off.
    If > 1, then verbose is on.
    :type gscv_verbose: int, optional (default=1)

    :return: (Tensor Basis) Decision Tree Regressor (pipeline) ready for grid search cross-validation,
    and hyper-parameters dictionary for tuning.
    :rtype: (GridSearchCV instance, dict)
    """
    if rf_selector:
        feat_selector = SelectFromModel(RandomForestRegressor(n_estimators=rf_selector_n_estimators,
                                                               max_depth=3,
                                                               split_finder='brent',
                                                               median_predict=True,
                                                               oob_score=True,
                                                               n_jobs=n_jobs,
                                                               verbose=max(gscv_verbose - 1, 0),
                                                               max_features=2/3.), threshold=rf_selector_threshold)
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
        # Append hyper-parameters to a dict, depending on pipeline or not
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
        # Otherwise, it's TBRF feature filter
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
                                 verbose=gscv_verbose,
                                 scoring=None,
                                 refit=refit,
                                 return_train_score=return_train_score,
                                 # Mean score of folds is not sample weighted
                                 iid=False)

    return regressor_gscv, regressor, tuneparams, fit_param_key


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
                                var_threshold=0.,
                                scaler=None,
                                # [DEPRECATED]
                                alpha_g_fit=0., 
                                **kwargs):
    from sklearn.ensemble import RandomForestRegressor

    if var_threshold > 0. and scaler is None:
        warn('\nVariance threshold > 0 but scaler is not chosen, using default Max Absolute Scaler\n',
             stacklevel=2)
        scaler = 'maxabs'

    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_min_samples_split, (int, float)): gs_min_samples_split = (gs_min_samples_split,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    # Initialize scaler object
    if scaler in ('maxabs', 'MaxAbs'):
        scaler_ = MaxAbsScaler()
    elif scaler in ('minmax', 'MinMax'):
        scaler_ = MinMaxScaler()
    else:
        scaler_ = None

    # Initialize variance threshold object for feature selection if requested
    if var_threshold > 0.:
        feat_selector_ = VarianceThreshold(threshold=var_threshold)
    else:
        feat_selector_ = None

    # Initialize random forest regressor object
    rf_ = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose,
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
    # If scaler and/or feature selection, a pipeline object is necessary
    if scaler_ is None:
        pipeline = rf_
        tuneparams = dict(max_features=gs_max_features,
                          min_samples_split=gs_min_samples_split,
                          alpha_g_split=gs_alpha_g_split)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    else:
        if feat_selector_ is not None:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('feat_selector', feat_selector_),
                                 ('rf', rf_)])
        else:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('rf', rf_)])

        tuneparams = dict(rf__max_features=gs_max_features,
                          rf__min_samples_split=gs_min_samples_split,
                          rf__alpha_g_split=gs_alpha_g_split)
        fit_param_key = 'rf__tb'

    return pipeline, tuneparams, fit_param_key


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
                            var_threshold=0.,
                            scaler=None,
                            # [DEPRECATED]
                            alpha_g_fit=0.,
                            **kwargs):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    if var_threshold > 0. and scaler is None:
        warn('\nVariance threshold > 0 but scaler is not chosen, using default Max Absolute Scaler\n',
             stacklevel=2)
        scaler = 'maxabs'

    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, (int, float)): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)
    # Initialize scaler object
    if scaler in ('maxabs', 'MaxAbs'):
        scaler_ = MaxAbsScaler()
    elif scaler in ('minmax', 'MinMax'):
        scaler_ = MinMaxScaler()
    else:
        scaler_ = None

    # Initialize variance threshold object for feature selection if requested
    if var_threshold > 0.:
        feat_selector_ = VarianceThreshold(threshold=var_threshold)
    else:
        feat_selector_ = None

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
    ab_ = AdaBoostRegressor(base_estimator=base,
                             n_estimators=n_estimators,
                             loss=loss,
                             random_state=rand_state,
                             bij_novelty=bij_novelty)
    # If scaler and/or feature selection, a pipeline object is necessary
    if scaler_ is None:
        pipeline = ab_
        tuneparams = dict(base_estimator__max_features=gs_max_features,
                          base_estimator__max_depth=gs_max_depth,
                          base_estimator__alpha_g_split=gs_alpha_g_split,
                          learning_rate=gs_learning_rate)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    else:
        if feat_selector_ is not None:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('feat_selector', feat_selector_),
                                 ('ab', ab_)])
        else:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('ab', ab_)])

        tuneparams = dict(ab__base_estimator__max_features=gs_max_features,
                          ab__base_estimator__max_depth=gs_max_depth,
                          ab__base_estimator__alpha_g_split=gs_alpha_g_split,
                          ab__learning_rate=gs_learning_rate)
        fit_param_key = 'ab__tb'

    # Construct GSCV for DT
    ab_gscv = GridSearchCV(pipeline,
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

    return ab_gscv, pipeline, tuneparams, fit_param_key


def setupGradientBoostGridSearch(n_estimators=50, gs_max_features=(1/3.,), gs_max_depth=(3,), gs_alpha_g_split=(0.,),                             gs_learning_rate=(0.1,),
                                 subample=1.,

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
                            var_threshold=0.,
                            scaler=None,
                            # [DEPRECATED]
                            alpha_g_fit=0.,
                            **kwargs):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    if var_threshold > 0. and scaler is None:
        warn('\nVariance threshold > 0 but scaler is not chosen, using default Max Absolute Scaler\n',
             stacklevel=2)
        scaler = 'maxabs'

    # Ensure tuple grid search hyper-parameters
    if isinstance(gs_max_features, (int, float)): gs_max_features = (gs_max_features,)
    if isinstance(gs_max_depth, (int, float)): gs_max_depth = (gs_max_depth,)
    if isinstance(gs_alpha_g_split, (int, float)): gs_alpha_g_split = (gs_alpha_g_split,)
    if isinstance(gs_learning_rate, (int, float)): gs_learning_rate = (gs_learning_rate,)
    # Initialize scaler object
    if scaler in ('maxabs', 'MaxAbs'):
        scaler_ = MaxAbsScaler()
    elif scaler in ('minmax', 'MinMax'):
        scaler_ = MinMaxScaler()
    else:
        scaler_ = None

    # Initialize variance threshold object for feature selection if requested
    if var_threshold > 0.:
        feat_selector_ = VarianceThreshold(threshold=var_threshold)
    else:
        feat_selector_ = None

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
    ab_ = AdaBoostRegressor(base_estimator=base,
                            n_estimators=n_estimators,
                            loss=loss,
                            random_state=rand_state,
                            bij_novelty=bij_novelty)
    # If scaler and/or feature selection, a pipeline object is necessary
    if scaler_ is None:
        pipeline = ab_
        tuneparams = dict(base_estimator__max_features=gs_max_features,
                          base_estimator__max_depth=gs_max_depth,
                          base_estimator__alpha_g_split=gs_alpha_g_split,
                          learning_rate=gs_learning_rate)
        # Tensor basis kwarg for fit method
        fit_param_key = 'tb'
    else:
        if feat_selector_ is not None:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('feat_selector', feat_selector_),
                                 ('ab', ab_)])
        else:
            pipeline = Pipeline([('scaler', scaler_),
                                 ('ab', ab_)])

        tuneparams = dict(ab__base_estimator__max_features=gs_max_features,
                          ab__base_estimator__max_depth=gs_max_depth,
                          ab__base_estimator__alpha_g_split=gs_alpha_g_split,
                          ab__learning_rate=gs_learning_rate)
        fit_param_key = 'ab__tb'

    # Construct GSCV for DT
    ab_gscv = GridSearchCV(pipeline,
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

    return ab_gscv, pipeline, tuneparams, fit_param_key


@timer
def performEstimatorGridSearch(estimator_gs, tuneparams, x_train, y_train,
                               tb_kw='tb', tb_train=None, x_test=None, y_test=None, tb_test=None,
                               staged=False, refit=False,
                               **kwargs):
    from sklearn.model_selection import ParameterGrid
    import copy

    print(' {0}'.format(tuneparams))
    is_pipeline = True if hasattr(estimator_gs, '_final_estimator') else False
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
        estimator_gs_final = estimator_gs._final_estimator if is_pipeline else estimator_gs
        if not is_pipeline and hasattr(estimator_gs_final, 'oob_score_'):
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
