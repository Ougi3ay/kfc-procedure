"""
MixCOBRA implementation built on modular core components.

This class implements the MixCOBRA regression framework, which combines:
- multiple base estimators (expert pool)
- distance-based similarity in joint input/output space
- kernel weighting
- aggregation of neighbor targets
- hyperparameter optimization over mixing coefficients (alpha, beta)

Pipeline:
    Input -> Split -> Estimators -> Normalize -> Distance (X, Y)
    -> Kernel Adapter -> Kernel -> Optimization -> Aggregation -> Output
"""
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from cobra.core.adapters.base import BaseKernelAdapter, KernelAdapterFactory
from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel, KernelFactory
from cobra.core.losses.base import BaseLoss, LossFactory
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer, GradientOptimizerFactory
from cobra.core.optimizers.search.base import BaseSearchOptimizer, SearchOptimizerFactory
from cobra.core.spaces.base import SpaceNormalizerFactory
from cobra.core.splitters.base import BaseDataSplitter, SplitterFactory

class MixCOBRARegressor(ABC, SkBaseEstimator, RegressorMixin):
	"""
	MixCOBRA regressor that learns mixing weights across input/output spaces.

	This estimator implements the MixCOBRA pattern: it trains an ensemble of
	base estimators, constructs a prediction-space representation, computes
	distances in input and output (prediction) spaces, learns mixing
	coefficients (alpha, beta) that balance those spaces, and aggregates
	neighbor targets with a kernel-weighted aggregator.

	Parameters
	----------
	estimators : list[str | BaseEstimator] | None, default=None
		List of estimator identifiers or estimator instances used as the expert
		pool. When a string is provided the ``EstimatorFactory`` is used to
		instantiate the implementation.

	estimators_params : dict[str, Any] | None, default=None
		Optional mapping of estimator name -> init parameters passed to the
		corresponding factory when string identifiers are used.

	distance : str, default='euclidean'
		Identifier for the distance metric used to compute pairwise similarity
		in feature/prediction spaces (resolved via ``DistanceFactory``).

	distance_params : dict | None, default=None
		Additional keyword arguments forwarded to the chosen distance class.

	kernel : str, default='rbf'
		Kernel identifier used to convert distances into weights (resolved via
		``KernelFactory``).

	kernel_params : dict | None, default=None
		Keyword arguments forwarded to the kernel implementation.

	aggregator : str, default='weighted_mean'
		Aggregation strategy identifier used to combine neighbor targets.

	aggregator_params : dict | None, default=None
		Keyword arguments forwarded to the aggregator implementation.

	loss : str, default='mse'
		Loss identifier used during hyperparameter optimization.

	loss_params : dict | None, default=None
		Keyword arguments forwarded to the loss implementation.

	optimizer : str, default='grad'
		Optimizer identifier (gradient or search based) used to tune mixing
		parameters; resolved via optimizer factories.

	optimizer_params : dict | None, default=None
		Parameters forwarded to the optimizer implementation.

	alpha_list, beta_list : np.ndarray | None
		Optional candidate grids used by grid/search optimizers. Not used for
		gradient-based optimizers.

	norm_constant_x, norm_constant_y : float | None
		Optional normalization constants for input (X) and output (Y)
		spaces. When None, the configured ``SpaceNormalizer`` determines
		appropriate scaling.

	opt_method : str, default='grad'
		High-level optimization mode: either ``'grad'`` for gradient-based
		tuning or ``'search'`` for grid/search strategies.

	one_parameter : bool, default=False
		If True, only optimize ``alpha`` and keep ``beta`` fixed to zero.

	random_state : int | None
		PRNG seed for reproducible components (splitters, optimizers, etc.).

	Notes
	-----
	- The implementation follows a pipeline of: estimator training,
	  prediction matrix construction, space normalization, distance/kernel
	  computation, optimizer-driven parameter selection, and aggregation.
	- All components are created via factory classes; to extend behaviour,
	  register new implementations with the appropriate ``*Factory``.

	See Also
	--------
	GradientCOBRA, CombineClassifier
	"""
	def __init__(
		self,
		estimators: list[str | BaseEstimator] | None = None,
		estimators_params: dict[str, Any] | None = None,
		distance: str = "euclidean",
		distance_params: dict[str, Any] | None = None,
		kernel: str = "rbf",
		kernel_params: dict[str, Any] | None = None,
		aggregator: str = "weighted_mean",
		aggregator_params: dict[str, Any] | None = None,
		loss: str = "mse",
		loss_params: dict[str, Any] | None = None,
		optimizer: str = "grad",
		optimizer_params: dict[str, Any] | None = None,

		alpha_list: np.ndarray | None = None,
		beta_list: np.ndarray | None = None,
		norm_constant_x = None,
		norm_constant_y = None,
		opt_method: str = "grad",
		one_parameter: bool = False,
		random_state: int | None = None
	):
		"""
		Initialize MixCOBRA model and store configuration.
		"""

		self.estimators = estimators
		self.estimators_params = estimators_params
		self.distance = distance
		self.distance_params = distance_params
		self.kernel = kernel
		self.kernel_params = kernel_params
		self.aggregator = aggregator
		self.aggregator_params = aggregator_params
		self.loss = loss
		self.loss_params = loss_params
		self.optimizer = optimizer
		self.optimizer_params = optimizer_params

		self.alpha_list = alpha_list
		self.beta_list = beta_list
		self.norm_constant_x = norm_constant_x
		self.norm_constant_y = norm_constant_y
		self.opt_method = opt_method
		self.one_parameter = one_parameter
		self.random_state = random_state
        
	
	def _resolve_fit_split_context(self, X, y, X_l, y_l, pred_features=None):
		"""
		Prepare training and calibration split.

		Supports three modes:
		1. External calibration set (X_l, y_l provided)
		2. Prediction-feature mode (pred_features provided)
		3. Automatic split using SplitOverlap strategy

		Returns
		-------
		X_k, y_k : training data
		X_l, y_l : calibration data
		iloc_k, iloc_l : indices in original dataset
		"""
		X, y = check_X_y(X, y)
		if X_l is not None and y_l is not None:
			X_l, y_l = check_X_y(X_l, y_l)
			X_k_, X_l_ = X, X_l
			y_k_, y_l_ = y, y_l
			iloc_l, iloc_k = np.arange(len(y_l_)), np.arange(len(y))
			self.as_predictions_ = True
		elif pred_features is not None:
			X_l_, y_l_ = X, y
			iloc_l, iloc_k = np.arange(len(y_l_)), np.arange(len(y))
			self.as_predictions_ = True
			self.pred_l_ = check_array(pred_features) * self.norm_constant_y_
			if self.pred_l_.shape[0] != self.y_l_.shape[0]:
				raise ValueError("Incompatible shapes between y_l and pred_features")
		else:
			splitter: BaseDataSplitter = SplitterFactory.create(
				'split_overlap',
				split_ratio=0.5,
				overlap=0.0,
				random_state=self.random_state
			)
			iloc_k, iloc_l = splitter.split(X, y)
			X_k_, y_k_ = X[iloc_k], y[iloc_k]
			X_l_, y_l_ = X[iloc_l], y[iloc_l]
			self.as_predictions_ = False

		return X_k_, y_k_, X_l_, y_l_, iloc_k, iloc_l

	
	def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):
		"""
		Fit base estimator pool on training data.

		Returns
		-------
		list[BaseEstimator]
		    Trained models
		"""

		default_estimators = [
			"linear",
            "ridge",
            "lasso",
            "knn",
            "random_forest",
            "svm",
		]

		estimators = self.estimators or default_estimators

		machines = []

		for est in estimators:
			if isinstance(est, str):
				params = (self.estimators_params or {}).get(est, {})
				model = EstimatorFactory.create(est, **params)
			elif isinstance(est, BaseEstimator):
				model = est
			else:
				raise ValueError(
					f"Invalid estimator: {type(est)}. "
					f"Expected str or BaseEstimator. "
					f"Available: {EstimatorFactory.available()}"
				)

			model.fit(X_k, y_k)
			machines.append(model)

		return machines

	def _load_predictions(self, X: np.ndarray):
		"""
		Generate prediction matrix from all base estimators.

		Returns
		-------
		np.ndarray
		    Shape (n_samples, n_estimators)
		"""
		cols = []
		for est in self.estimators_:
			preds = est.predict(X)
			cols.append(preds)
		return np.column_stack(cols)

	def _space_normalize(self, X, model_outputs):
		"""
		Normalize input and output spaces before distance computation.

		Returns
		-------
		X_norm, Y_norm
		"""
		normalizer = SpaceNormalizerFactory.create(
			"mixcobra",
			norm_constant_x=self.norm_constant_x,
			norm_constant_y=self.norm_constant_y
		)
		return normalizer.transform(X, model_outputs)

	def _resolve_component(self):
		"""
		Instantiate all modular components:
		- distance
		- kernel
		- aggregator
		- loss
		- splitter
		- kernel adapter
		"""
		self.distance_ : BaseDistance = DistanceFactory.create(
			self.distance,
			**(self.distance_params or {})
		)
		self.kernel_ : BaseKernel = KernelFactory.create(
			self.kernel,
			**(self.kernel_params or {})
		)
		self.aggregator_ : BaseAggregator = AggregatorFactory.create(
			self.aggregator,
			**(self.aggregator_params or {})
		)
		self.loss_ : BaseLoss = LossFactory.create(
			self.loss,
			**(self.loss_params or {})
		)
		self.splitter_ : BaseDataSplitter = SplitterFactory.create(
			'kfold',
			n_splits=5,
			random_state=self.random_state
		)

		self.adapter_ : BaseKernelAdapter = KernelAdapterFactory.create(
			"mixcobra",
			alpha=1.0,
			beta=1.0
		)
	
	def objective_1d(self, params):
		"""
		Objective function for 1D optimization (alpha only).

		Returns
		-------
		float
		    Loss value
		"""
		alpha = params[0]
		beta = 0.0
		self.adapter_.set_params(alpha=alpha, beta=beta)
		dist_input = self.distance_.matrix(self.X_l_norm_, self.X_l_norm_)
		dist_output = self.distance_.matrix(self.Y_l_norm_, self.Y_l_norm_)
		mix_distance = np.hstack((dist_input, dist_output))
		D = self.adapter_.transform(mix_distance)
		K = self.kernel_(D)

		preds = np.empty(self.y_l_.shape[0], dtype=float)

		folds = self.splitter_.split(self.X_l_, self.y_l_)

		for train_idx, val_idx in folds:
			w = K[val_idx][:, train_idx]
			y_train = self.y_l_[train_idx]
			for i in range(len(val_idx)):
				if np.allclose(w[i].sum(), 0.0):
					preds[val_idx[i]] = np.mean(y_train)
				else:
					preds[val_idx[i]] = self.aggregator_.aggregate(y_train, w[i])
		
		return self.loss_(self.y_l_, preds)

	def objective_2d(self, params):
		"""
		Objective function for 2D optimization (alpha, beta).

		Returns
		-------
		float
		    Loss value
		"""
		alpha, beta = params
		self.adapter_.set_params(alpha=alpha, beta=beta)
		dist_input = self.distance_.matrix(self.X_l_norm_, self.X_l_norm_)
		dist_output = self.distance_.matrix(self.Y_l_norm_, self.Y_l_norm_)
		D = self.adapter_.transform(dist_input, dist_output)
		K = self.kernel_(D)

		preds = np.empty(self.y_l_.shape[0], dtype=float)

		folds = self.splitter_.split(self.X_l_, self.y_l_)

		for train_idx, val_idx in folds:
			w = K[val_idx][:, train_idx]
			y_train = self.y_l_[train_idx]
			for i in range(len(val_idx)):
				if np.allclose(w[i].sum(), 0.0):
					preds[val_idx[i]] = np.mean(y_train)
				else:
					preds[val_idx[i]] = self.aggregator_.aggregate(y_train, w[i])
		
		return self.loss_(self.y_l_, preds)


	def _optimize_hyperparameters(self):
		"""
		Run hyperparameter optimization using:
		- gradient descent OR
		- grid search

		Optimizes alpha/beta mixing between distance spaces.
		"""
		if self.opt_method == "grad":
			self.optimizer_ : BaseGradientOptimizer = GradientOptimizerFactory.create(
				self.optimizer,
				**(self.optimizer_params or {}),
				random_state=self.random_state
			)

			if self.one_parameter:
				params, history = self.optimizer_(self.objective_1d, np.array([1.0]))
			else:
				params, history = self.optimizer_(self.objective_2d, np.array([1.0, 1.0]))
		else:
			self.optimizer_ : BaseSearchOptimizer = SearchOptimizerFactory.create(
				self.optimizer,
				**(self.optimizer_params or {}),
				random_state=self.random_state
			)

			param_grid = {}
			if self.one_parameter:
				param_grid["alpha"] = self.alpha_list if self.alpha_list is not None else np.linspace(0, 2, 10)
				params, history = self.optimizer_(self.objective_1d, param_grid)
			else:
				param_grid["alpha"] = self.alpha_list if self.alpha_list is not None else np.linspace(0, 2, 10)
				param_grid["beta"] = self.beta_list if self.beta_list is not None else np.linspace(0, 2, 10)
				params, history = self.optimizer_(self.objective_2d, param_grid)
		
		# store optimization outputs
		self.optimization_outputs_ = {
			"method": self.opt_method,
			"params": params,
			"history": history
		}



	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		X_l: np.ndarray | None = None,
		y_l: np.ndarray | None = None,
		pred_features: np.ndarray | None = None
	):
		"""
		Fit MixCOBRA model with hyperparameter optimization.

		This method trains base estimators and learns optimal mixing weights
		(alpha, beta) that balance input-space and output-space distances
		for aggregation.

		Parameters
		----------
		X : np.ndarray
			Training features. Shape: (n_samples, n_features).

		y : np.ndarray
			Training targets. Shape: (n_samples,).

		X_l : np.ndarray | None, default=None
			External calibration features. If provided, used for aggregation
			instead of internal split. Shape: (n_cal_samples, n_features).

		y_l : np.ndarray | None, default=None
			External calibration targets. Shape: (n_cal_samples,).

		pred_features : np.ndarray | None, default=None
			Pre-computed model predictions to use instead of internal
			estimator outputs. Useful for integrating external predictions.

		Returns
		-------
		self : MixCOBRARegressor
			Fitted model instance.

		Workflow
		--------
		1. Split data into training (X_k) and calibration (X_l) subsets
		2. Train base estimators on X_k
		3. Generate prediction matrix on X_l
		4. Normalize input and output spaces
		5. Initialize distance, kernel, adapter, and aggregator components
		6. Optimize alpha/beta mixing parameters

		Examples
		--------
		>>> model = MixCOBRARegressor()
		>>> model.fit(X_train, y_train)
		"""
		
		(
			self.X_k_, self.y_k_,
			self.X_l_, self.y_l_,
			self.iloc_k_, self.iloc_l_
		) = self._resolve_fit_split_context(X, y, X_l, y_l, pred_features)

		# fit base estimators on training set
		self.estimators_ = self._fit_estimators(self.X_k_, self.y_k_)

		# load predictions for aggregation set
		if not self.as_predictions_:
			model_outputs = self._load_predictions(self.X_l_)
		else:
			model_outputs = self.X_l_
		
		# normalize space
		self.X_l_norm_, self.Y_l_norm_ = self._space_normalize(self.X_l_, model_outputs)

		# resolve components
		self._resolve_component()

		# optimize hyperparameters
		self._optimize_hyperparameters()
		return self
		
	def predict(
		self,
		X: np.ndarray,
		pred_X: np.ndarray | None = None,
		alpha: float | None = None,
		beta: float | None = None,
		bandwidth: float | None = None
	) -> np.ndarray:
		"""
		Predict target values using fitted MixCOBRA aggregator.

		For each test sample, this method computes distances in both
		input and output (prediction) spaces, combines them using optimized
		alpha/beta weights, and aggregates neighbor training targets.

		Parameters
		----------
		X : np.ndarray
			Test features. Shape: (n_samples, n_features).

		pred_X : np.ndarray | None, default=None
			Pre-computed predictions for test samples (e.g., from external
			models). If None, predictions are generated using internal
			estimators.

		alpha, beta, bandwidth : float | None
			Optional parameter overrides (currently unused; kept for API
			compatibility).

		Returns
		-------
		np.ndarray
			Predicted target values. Shape: (n_samples,).

		Workflow
		--------
		1. Generate or use provided predictions for test samples
		2. Normalize input and prediction spaces
		3. Compute distances in both spaces
		4. Combine distances using learned alpha/beta weights
		5. Transform combined distance via kernel adapter
		6. Apply kernel to generate similarity weights
		7. Aggregate calibration targets using kernel weights

		Examples
		--------
		>>> y_pred = model.predict(X_test)
		"""
		
		check_is_fitted(self)
		X = check_array(X)

		if pred_X is None:
			pred_X = X

		preds = self._load_predictions(X)
		X_norm, Y_norm = self._space_normalize(pred_X, preds)

		dist_input = self.distance_.matrix(X_norm, self.X_l_norm_)
		dist_output = self.distance_.matrix(Y_norm, self.Y_l_norm_)

		if self.one_parameter:
			alpha = self.optimization_outputs_["params"][0]
			beta = 0.0
			mix_distance = np.hstack((dist_input, dist_output))
			self.adapter_.set_params(alpha=alpha, beta=beta)
			D = self.adapter_.transform(mix_distance)
			K = self.kernel_(D)
		else:
			alpha, beta = self.optimization_outputs_["params"]
			self.adapter_.set_params(alpha=alpha, beta=beta)
			D = self.adapter_.transform(dist_input, dist_output)
			K = self.kernel_(D)
		
		W = self.kernel_(D)
		outputs = np.empty(W.shape[0], dtype=float)

		for i in range(W.shape[0]):
			w = W[i]
			if np.allclose(w.sum(), 0.0):
				outputs[i] = np.mean(self.y_l_)
			else:
				outputs[i] = self.aggregator_.aggregate(self.y_l_, w)

		return outputs
