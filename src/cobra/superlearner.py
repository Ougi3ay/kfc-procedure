"""
SuperLearner regressor using ensemble consensus and meta-learning.

The SuperLearner algorithm (van der Laan, Polley & Hubbard, 2007)
combines predictions from heterogeneous base learners by training
a meta-learner on cross-validated predicted features.

Pipeline overview
------------------
Input -> Base Learners -> CV Predictions -> Meta-Learner -> Output

Core design
-----------
1. Train diverse base learners (experts)
2. Generate cross-validated predictions on training data
3. Train a meta-learner to combine base predictions
4. Use fitted base + meta learners to generate final predictions

This approach minimizes the risk of meta-learning overfitting by
using held-out cross-validated predictions.

Key features
------------
- Flexible base learner pool selection
- Hyperparameter tuning via cross-validation
- Support for additional hand-crafted features
- Sklearn-compatible fit/predict API

References
----------
van der Laan, M., Polley, E., & Hubbard, A. (2007).
"Super Learner." Statistical Applications in Genetics and Molecular Biology, 6(1).
https://doi.org/10.2202/1544-6115.1309
"""

# --------------- Version 1.0.5 -------------------
# =================================================

# Import all the libraries 
# ========================
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array
from scipy import spatial
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# Plotting figures
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde as kde
# Table type
import numpy as np
import pandas as pd
# import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
import warnings
from itertools import product

class SuperLearner(BaseEstimator):
    def __init__(self, 
                random_state = None,
                base_learners = None,
                base_params = None,
                meta_learners = None,
                meta_params_cv = None,
                n_fold = int(10),
                cv_folds = None,
                loss_function = None,
                loss_weight = None):
        """
        This is a class of the implementation of `SuperLearner` by van der Laan, M., Polley, E. and Hubbard, A. (2007): https://doi.org/10.2202/1544-6115.1309.

        * Parameters:
        ------------
            - `random_state`: (default is `None`) set the random state of the random generators in the class.
            
            - `base_learners`: (default is None) the list of candidate learners or estimators. 
                If it is None, intial learners including 'linear_regression', 'ridge', 'lasso', 'tree', and 'random_forest' are used with default parameters.
                It should be a sublist of the following list: L = ['linear_regression', 'knn', 'ridge', 'lasso', 'tree', 'random_forest', 'svm', 'sgd', 'bayesian_ridge', 'adaboost', 'gradient_boost'].

            - `base_params`: (default is `None`) a dictionary containing the parameters of the candidate learners given in the `base_learners` argument. 
                It must be a dictionary with:
                - `key`     : the name of the base learners defined in `base_learners`, 
                - `value`   : a dictionary with (key, value) = (parameter, value).

            - `meta_learners`: (default is `None` and linear regression is used) meta learners that are trained on predicted features $(y_i, z_i)$ where $z_i = (r_1(x_i), ..., r_M(x_i))$ of $\mathbb{R}^M$ for $i=1,...,n$.
                It is the model that takes predicted features given by all the candidate learners as inputs. It must be an element of the list L of all the base learners.
                If a list of predictors (subset of L) is given, then the best one will be selected using CV error defined by `cv_folds`.

            - `meta_params_cv`: (default is `None`) a dictionary with "keys" be the name of the candidate meta learners given in `meta_learners` argument, and the "value" is the parameter dictionary.
                For example, if two meta learners are proposed in `meta_learners = ['ridge', 'lasso']`, then this argument should be the following dictionary:
                `meta_params_cv = {
                    'ridge' : {'alpha' : 2 ** np.linspace(-10,10,100)},
                    'lasso' : {'alpha' : 2 ** np.linspace(-10,10,100)}
                }`
                where in this case, the panalization strenght `alpha = 2 ** np.linspace(-10,10,100)` is to be tuned using cross validation technique.

            - `cv_folds`: (default is `None`) a list or an array `I` of size $n$ (observation size) whose elements are in {0,1,...,K-1}. 
                Then, $I[i]=k$ if and only if observation $i$ belongs to fold $k$ in cross-validation procedure. If `None`, then the folds are selected randomly. 
            
             - `kernel`: (default is 'radial') the kernel function used for the aggregation. 
                It should be an element of the list ['exponential', 'gaussian', 'radial', 'cauchy', 'reverse_cosh', 'epanechnikov', 'biweight', 'triweight', 'triangular', 'cobra', 'naive'].
                Some options such as 'gaussian' and 'radial' lead to the same radial kernel function. 
                For 'cobra' or 'naive', they correspond to Biau et al. (2016).

            - `loss_function`: (default is None) a function or string defining the cost function to be optimized for estimating the optimal bandwidth parameter.
                By defalut, the K-Fold cross-validation MSE is used. Otherwise, it must be either:
                - a function of two argumetns (y_true, y_pred) or
                - a string element of the list ['mse', 'mae', 'mape', 'weighted_mse']. If it is `weighted_mse`, one can define the weight for each training point using `loss_weight` argument below.
            
            - `loss_weight`: (default is None) a list of size equals to the size of the training data defining the weight for each individual data point in the loss function. 
                If it is None and the `loss_function = weighted_mse`, then a normalized weight W(i) = 1/PDF(i) is assigned to individual i of the training data.

        * Returns:
        ---------
            self : returns an instance of self. 

        * Methods: 
        ---------
            - fit : fitting the super learner on the design features (original data or predicted features). The argument of this method are described below.
            - train_base_learners : build base learners on CV data. It is also possible to set the values of (hyper) parameters for each base learner in `base_params`.
            - add_extra_learners : to add additional learners to the list of base learner to train meta learner therefore build super learner. This can be class method or estimator, list, array or data frame of the same numer of rows as the training data.
            - train_meta_learner : to train meta learner on (y_i z_i), CV predicted features. This method must be called if you add any axtra-learners to the list of base learner after calling `fit` method.
            - draw_learning_curve : for plotting the graphic of learning algorithm (error vs parameter).
        """
        self.random_state = random_state
        self.base_learners = base_learners
        self.base_params = base_params
        self.meta_learners = meta_learners
        self.meta_params_cv = meta_params_cv
        self.cv_folds = cv_folds
        self.n_fold = n_fold
        self.loss_weight = loss_weight
        self.loss_function = loss_function
    
    # List of loss functions
    def mse(self, y_true, pred):
        return mean_squared_error(y_true, pred)
    def mae(self, y_true, pred):
        return mean_absolute_error(y_true, pred)
    def mape(self, y_true, pred):
        return mean_absolute_percentage_error(y_true, pred)
    def loss_func(self, y_true, pred):
        return self.loss_function(y_true, pred)
    
    def fit(self, X, y, 
            train_meta_learners = True, 
            as_predictions = False, 
            show_warning = True):
        '''
        This method builds base and meta learner of Super learning algorithm.
        
        * Parameters:
        -------------
            - `X, y`: the training input and out put. If the argument `as_predictions = True`, then the input `X` is treated as predicted features `Z`. 
                In this case, the meta learner is trained directly on (X,y) without building any base learners. 
            - `train_meta_learners`: a boolean variable controlling whether to directly train the meta learner or not after training the base learners given in `base_learners` argument.
                This is useful when you want to add extra learners to the list of base learners before training the meta learner. 
            - `as_predictions` : a boolean variable controlling whether `X` should be treated as predicted features `Z` or not. If it is `True`, then meta learners can be trained directly on (X,y).

        * Important note: 
        ----------------
        You can perform CV over a list of meta learner of `meta_learners` argument by providing its corresponding dictionary of parameters in `meta_params` arguement.
        Moreover, you can also add features that were obtained from anonymous models by specifying in the `fit` method using `train_meta_learners = False`. 
        In this case, fit method only train the base learners and provide predicted features (Z_i) for meta learners. After that, you can use `add_extra_learners` method to add extra learners to the list of base learners.
        These extra learners can be any "sklearn" classes, or pandas data frame, numpy arrays or list containing the same observation as the training data. 
        If the data frame are added as additional learners, then it will be concatenated to the predicted features (Z_i) for training meta models.
        It is important to notice that if extra features are added as extra learners, it the corresponding extra features of the testing data must also be provided.
        '''

        X, y = check_X_y(X, y)
        if X.dtype == object:
            X = X.astype(np.float64)
        if y.dtype == object:
            y = y.astype(np.float64)
        self.X_ = X
        self.y_ = y
        
        if self.cv_folds is None:
            self.cv_folds_ = np.concatenate([
                np.random.permutation(np.repeat(list(range(self.n_fold)), len(self.y_) // self.n_fold)), 
                np.random.choice(self.n_fold, size=len(self.y_) % self.n_fold)])
        else:
            self.cv_folds_ = self.cv_folds

        self.cv_X = {}
        self.cv_y = {}
        for i in range(self.n_fold):
            self.cv_X[str(i)] = self.X_[self.cv_folds_ == i, :]
            self.cv_X['~'+str(i)] = self.X_[self.cv_folds_ != i, :]
            self.cv_y[str(i)] = self.y_[self.cv_folds_ == i]
            self.cv_y['~'+str(i)] = self.y_[self.cv_folds_ != i]
            
        # Loss function
        if (self.loss_function is None) or (self.loss_function == 'mse') or (self.loss_function == 'mean_squared_error'):
            self.loss = self.mse
        elif (self.loss_function == 'mae') or (self.loss_function == 'mean_absolute_error'):
            self.loss = self.mae
        elif (self.loss_function == "mape") or (self.loss_function == 'mean_absolute_percentage_error'):
            self.loss = self.mape
        
        if callable(self.loss_function):
            self.loss = self.loss_func

        self.extra_features = {}
        self.extra_learners = {}
        self.n_extra_learners = 0 
        self.n_extra_features = 0 

        self.as_predictions_ = as_predictions
        
        if not as_predictions:
            self.train_base_learners()
        else:
            self.Z_ = self.X_
            if show_warning:
                warnings.warn('Loading inputs (X_i) as predictions (Z_i)! Make sure they are the validated ones!')
        
        if train_meta_learners:
            self.train_meta_learners()

        return self
    
    def train_base_learners(self, final = False):
        all_estimators = {
            'linear_regression' : LinearRegression(),
            'extra_trees' : ExtraTreesRegressor(random_state=self.random_state),
            'knn' : KNeighborsRegressor(),
            'lasso' : Lasso(random_state=self.random_state),
            'ridge' : Ridge(random_state=self.random_state),
            'tree' : DecisionTreeRegressor(random_state=self.random_state),
            'random_forest' : RandomForestRegressor(random_state=self.random_state),
            'svm' : SVR(),
            'bayesian_ridge' : BayesianRidge(),
            'sgd' : SGDRegressor(random_state=self.random_state),
            'adaboost' : AdaBoostRegressor(random_state=self.random_state),
            'gradient_boost' : GradientBoostingRegressor(random_state=self.random_state)
        }
        learner_dict = {}
        if self.base_learners == "all":
            learner_dict = all_estimators
        elif self.base_learners is None:
            learner_dict = {'linear_regression' : LinearRegression(),
                            'lasso' : Lasso(random_state=self.random_state),
                            'ridge' : Ridge(random_state=self.random_state),
                            'svm' : DecisionTreeRegressor(random_state=self.random_state),
                            'random_forest' : RandomForestRegressor(random_state=self.random_state)}
        else:
            for name in self.base_learners:
                learner_dict[name] = all_estimators[name]
        self.learner_names = list(learner_dict.keys())
        param_dict = {
            'linear_regression' : None,
            'knn' : None,
            'lasso' : None,
            'ridge' : None,
            'tree' : None,
            'random_forest' : None,
            'svm' : None,
            'bayesian_ridge' : None,
            'sgd' : None,
            'adaboost' : None,
            'gradient_boost' : None,
            'extra_trees' : None
        }
        if self.base_params is not None:
            for name in list(self.base_params):
                param_dict[name] = self.base_params[name]
        self.final_base_learners = {}
        first = True
        for machine in self.learner_names:
            try:
                mod = learner_dict[machine]
                if param_dict[machine] is not None:
                    if machine == 'adaboost':
                        mod.estimator = DecisionTreeRegressor(random_state=self.random_state)
                        param_ = {}
                        for p_ in mod.estimator.get_params():
                            if p_ in list(param_dict[machine].keys()):
                                param_[p_] = param_dict[machine][p_]
                                param_dict[machine].pop(p_)
                        mod.estimator.set_params(**param_)
                        mod.set_params(**param_dict[machine])
                    else:
                        mod.set_params(**param_dict[machine])
            except ValueError:
                continue
            if not final:
                temp = np.zeros(shape=len(self.y_))
                for i in range(self.n_fold):
                    mod_fit = mod.fit(self.cv_X['~'+str(i)], self.cv_y['~'+str(i)])
                    temp[self.cv_folds_ == i] = mod_fit.predict(self.cv_X[str(i)])
                if first:
                    self.Z_ = temp
                    first = False
                else:
                    self.Z_ = np.column_stack([self.Z_, temp])
            else:
                self.final_base_learners[machine] = mod.fit(self.X_, self.y_)
            
        if final:
            if self.n_extra_learners > 0:
                self.final_extra_learners = {}
                for exl in self.extra_learners:
                    self.final_extra_learners[exl] = self.extra_learners[exl].fit(self.X_, self.y_)
        return self

    def add_extra_learners(self, extra_learner):
        if isinstance(extra_learner, pd.core.frame.DataFrame) or isinstance(extra_learner, np.ndarray) or isinstance(extra_learner, list):
            self.n_extra_features += 1
            temp = check_array(extra_learner)
            self.extra_features[self.n_extra_features] = temp
        else:
            self.n_extra_learners += 1
            self.extra_learners[self.n_extra_learners] = extra_learner
        return self

    def train_meta_learners(self):
        all_estimators = {
            'linear_regression' : LinearRegression(),
            'extra_trees' : ExtraTreesRegressor(random_state=self.random_state),
            'knn' : KNeighborsRegressor(),
            'lasso' : Lasso(random_state=self.random_state),
            'ridge' : Ridge(random_state=self.random_state),
            'tree' : DecisionTreeRegressor(random_state=self.random_state),
            'random_forest' : RandomForestRegressor(random_state=self.random_state),
            'svm' : SVR(),
            'bayesian_ridge' : BayesianRidge(),
            'sgd' : SGDRegressor(random_state=self.random_state),
            'adaboost' : AdaBoostRegressor(random_state=self.random_state),
            'gradient_boost' : GradientBoostingRegressor(random_state=self.random_state)
        }
        
        # Adding extra features or learners to CV predictions
        if not self.as_predictions_:
            if self.n_extra_learners > 0:
                for extra_learner in self.extra_learners:
                    temp = np.zeros(shape=len(self.y_))
                    for i in range(self.n_fold):
                        mod_fit = self.extra_learners[extra_learner].fit(self.cv_X['~'+str(i)], self.cv_y['~'+str(i)])
                        temp[self.cv_folds_ == i] = mod_fit.predict(self.cv_X[str(i)])
                    self.Z_ = np.column_stack([self.Z_, temp])
        else:
            if self.n_extra_learners > 0:
                warnings.warn("Inputs are treated as predictions! Extra learners must be an array or data frame!")
        if self.n_extra_features > 0:
            for extra_feature in self.extra_features:
                self.Z_ = np.column_stack([self.Z_, self.extra_features[extra_feature]])
                
        self.cv_Z = {}
        for i in range(self.n_fold):
            self.cv_Z[str(i)] = self.Z_[self.cv_folds_ == i, :]
            self.cv_Z['~'+str(i)] = self.Z_[self.cv_folds_ != i, :]

        # CV evaluation
        def cv_eval(method):
            s = 0
            for i in range(self.n_fold):
                ft = method.fit(self.cv_Z['~'+str(i)], self.cv_y['~'+str(i)])
                s += self.loss(ft.predict(self.cv_Z[str(i)]), self.cv_y[str(i)])
            return s / self.n_fold
        # Train meta learners
        if self.meta_learners is None:
            self.SuperLearner = LinearRegression().fit(self.Z_, self.y_)
        else:
            score_ = {}
            ml_dict = {}
            for ml in self.meta_learners:
                try:
                    meta = all_estimators[ml]
                    if self.meta_params_cv is not None:
                        if ml in self.meta_params_cv:
                            param_dict = self.meta_params_cv[ml]
                            combinations = list(product(*param_dict.values()))
                            opt = np.inf
                            opt_method = None
                            if ml == 'adaboost':
                                for combination in combinations:
                                    param_ = dict(zip(param_dict.keys(), combination))
                                    est_param_ = {}
                                    meta.estimator = DecisionTreeRegressor(random_state=self.random_state)
                                    for p_ in meta.estimator.get_params():
                                        if p_ in list(param_dict.keys()):
                                            est_param_[p_] = param_[p_]
                                            param_.pop(p_)
                                    meta.estimator.set_params(**est_param_)
                                    meta.set_params(**param_)
                                    val = cv_eval(meta)
                                    if val < opt:
                                        opt = val
                                        opt_method = meta
                                score_[ml] = val
                                ml_dict[ml] = opt_method.fit(self.Z_, self.y_)
                            else:
                                for combination in combinations:
                                    param_ = dict(zip(param_dict.keys(), combination))
                                    meta.set_params(**param_)
                                    val = cv_eval(meta)
                                    if val < opt:
                                        opt = val
                                        opt_method = meta
                                score_[ml] = val
                                ml_dict[ml] = opt_method.fit(self.Z_, self.y_)
                        else:
                            score_[ml] = cv_eval(meta)
                            ml_dict[ml] = meta.fit(self.Z_, self.y_)
                    else:
                        score_[ml] = cv_eval(method=meta)
                        ml_dict[ml] = meta.fit(self.Z_, self.y_)
                except ValueError:
                    continue
            self.meta_score_ = score_
            key = min(score_, key=score_.get)
            self.SuperLearner = ml_dict[key]
        self.train_base_learners(final=True)
        return self

    def predict(self, X, extra_features = None):
        X = check_array(X)
        if self.as_predictions_:
            self.Z_test = X
            if self.n_extra_features > 0:
                if extra_features is None:
                    raise TypeError('Extra-features is requred!')
                else:
                    ex = check_array(extra_features)
                    self.Z_test = np.column_stack([self.Z_test, ex])
        else:
            first = True
            for flearner in self.final_base_learners:
                if first:
                    self.Z_test = self.final_base_learners[flearner].predict(X)
                    first = False
                else:
                    self.Z_test = np.column_stack([
                        self.Z_test, 
                        self.final_base_learners[flearner].predict(X)
                    ])
            if self.n_extra_learners > 0:
                for exl in self.final_extra_learners:
                    self.Z_test = np.column_stack([
                        self.Z_test, 
                        self.final_extra_learners[exl].predict(X)
                    ])
            if self.n_extra_features > 0:
                if  extra_features is None:
                    raise TypeError('Extra features were added to base learners, therefore "extra_features" cannot be "None" in "predict" method!')
                else:
                    ex = check_array(extra_features)
                    self.Z_test = np.column_stack([self.Z_test, ex])
        if self.Z_test.shape[1] != self.Z_.shape[1]:
            raise TypeError('Columns of train and test predicted features are not consistent! Check your extra-learners!')
        else:
            self.test_prediction = self.SuperLearner.predict(self.Z_test)
            return self.test_prediction

    def draw_learning_curve(self, 
                            y_test = None,  
                            fig_type = 'qq', 
                            save_fig = False,
                            fig_path = False,
                            show_fig = True):
        if (y_test is not None) and (fig_type in ['qq', 'qq-plot', 'qqplot', 'QQ-plot', 'QQplot']):
            df = pd.DataFrame({
                'y_test' : y_test,
                'y_pred' : self.test_prediction})
            fig = go.Figure(data = px.line(df, 
                                               x = "y_test", 
                                               y = "y_test", 
                                               color_discrete_sequence=['red']).data + px.scatter(df, x = "y_pred", 
                                                                                                  y = "y_test").data)
            fig = fig.update_layout(width = 700, 
                                        height = 650, 
                                        title_text = "QQplot of predicted and actual target", 
                                        title_x = .5, 
                                        title_y = 0.925)
            fig.update_xaxes(title_text = "Prediction")
            fig.update_yaxes(title_text = "Actual target")
            if show_fig:
                fig.show()
            if save_fig:
                if fig_path is None:
                    fig.write_image("qqplot_superlearner.png")
                else:
                    fig.write_image(fig_path)
        else:
            if (self.meta_learners is not None) and (len(self.meta_learners) > 1):
                fig = go.Figure([go.Scatter(x = list(self.meta_score_.keys()),
                                        y = list(self.meta_score_.values()),
                                        mode = 'lines+markers',
                                        name = "CV error",
                                        line = dict(color = "red"),
                                        marker = dict(color = "red"),
                                        showlegend=False)])
                fig.update_xaxes(title_text="Meta learners")
                fig.update_yaxes(title_text="CV error")
                fig.update_layout(width = 150 * len(self.meta_learners), height = 100 * len(self.meta_learners), title_text = "CV error of meta learners")
                fig.show()
                if save_fig:
                    if fig_path is None:
                        fig.write_image("best_meta_learner.png")
                    else:
                        fig.write_image(fig_path)