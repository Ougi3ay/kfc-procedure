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
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.cv.base import BaseCrossValidator, CVFactory
from cobra.utils.preprocessing import compute_normalization_constant, history_to_dataframe
from cobra.utils.resolve import fit_estimators, predict_estimators, resolve_training_context

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
		optimizer: str = "grid",
		optimizer_params: dict[str, Any] | None = None,

		norm_constant_x = None,
		norm_constant_y = None,
		alpha_list: np.ndarray | None = None,
		beta_list: np.ndarray | None = None,
		opt_method: str = "grid",
		learning_rate: float = 0.01,
        max_iter: int = 300,
		n_cv: int = 5,
		speed: str = "constant",
		n_jobs: int = 1,
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

		self.norm_constant_x = norm_constant_x
		self.norm_constant_y = norm_constant_y
		self.alpha_list = alpha_list
		self.beta_list = beta_list
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.n_cv = n_cv
		self.speed = speed
		self.n_jobs = n_jobs
		self.opt_method = opt_method
		self.one_parameter = one_parameter
		self.random_state = random_state

	
	def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):
		"""
		Fit base estimator pool on training data.

		Returns
		-------
		list[BaseEstimator]
		    Trained models
		"""
		default_estimators = [
            "linear_regression",
            "ridge",
            "lasso",
            "k_neighbors_regressor",
            "random_forest_regressor",
            "svr",
        ]

		estimators = self.estimators or default_estimators

		return fit_estimators(
            X=X_k,
            y=y_k,
            estimators=estimators,
            estimators_params=self.estimators_params,
            n_jobs=self.n_jobs,
        )

	def _load_predictions(self, X: np.ndarray):
		"""
		Generate prediction matrix from all base estimators.

		Returns
		-------
		np.ndarray
		    Shape (n_samples, n_estimators)
		"""
		return predict_estimators(
            X=X,
            estimators=self.estimators_,
            n_jobs=self.n_jobs
        )

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
		self.distance_ : BaseDistance = (
			DistanceFactory.create(
				self.distance,
				**(self.distance_params or {})
			)
		)
		self.kernel_ : BaseKernel = (
			KernelFactory.create(
				self.kernel,
				**(self.kernel_params or {})
			)
		)
		self.aggregator_ : BaseAggregator = (
			AggregatorFactory.create(
				self.aggregator,
				**(self.aggregator_params or {})
			)
		)
		self.loss_ : BaseLoss = (
			LossFactory.create(
				self.loss,
				**(self.loss_params or {})
			)
		)

		self.cv_ : BaseCrossValidator = (
            CVFactory.create(
                "kfold",
                n_splits=self.n_cv,
                shuffle=True,
                random_state=self.random_state,
            )
        )

		if self.one_parameter:
			self.adapter_ = KernelAdapterFactory.create(
				"one_parameter",
				bandwidth=1.0
			)
		else:
			self.adapter_ = KernelAdapterFactory.create(
				"two_parameter",
				alpha=1.0,
				beta=1.0
			)
	
	def kappa_cross_validation_error_1d(self, params):
		bandwidth = params[0]
		self.adapter_.set_params(bandwidth=bandwidth)

		D = self.adapter_.transform(self.distance_matrix_mix_)
		K = self.kernel_(D)

		errors = []

		for fold in self.cv_folds_:
			train_idx = fold.train_idx
			val_idx = fold.eval_idx

			K_val_train = K[np.ix_(val_idx, train_idx)]
			y_train = self.y_l_[train_idx]
			y_val = self.y_l_[val_idx]

			preds = self.aggregator_.aggregate_matrix(
                values=y_train,
                weights=K_val_train,
                fallback=0.0
            )

			error = self.loss_(y_val, preds)
			errors.append(error)
		
		return float(np.mean(errors))
	
	def kappa_cross_validation_error_2d(self, params):
		alpha, beta = params
		self.adapter_.set_params(alpha=alpha, beta=beta)

		D = self.adapter_.transform(self.distance_matrix_x_, self.distance_matrix_y_)
		K = self.kernel_(D)

		errors = []

		for fold in self.cv_folds_:
			train_idx = fold.train_idx
			val_idx = fold.eval_idx

			K_val_train = K[np.ix_(val_idx, train_idx)]
			y_train = self.y_l_[train_idx]
			y_val = self.y_l_[val_idx]

			preds = self.aggregator_.aggregate_matrix(
                values=y_train,
                weights=K_val_train,
                fallback=0.0
            )

			error = self.loss_(y_val, preds)
			errors.append(error)
		return np.mean(errors)

	def _optimize_hyperparameters(self):
		"""
		Run hyperparameter optimization using:
		- gradient descent OR
		- grid search

		Optimizes alpha/beta mixing between distance spaces.
		"""
		alpha_candidates = (
			np.asarray(self.alpha_list)
			if self.alpha_list is not None
			else np.linspace(
                0.001,
                10.0,
                self.max_iter,
            )
		)

		beta_candidates = (
			np.asarray(self.beta_list)
			if self.beta_list is not None
			else np.linspace(
				0.001,
				10.0,
				self.max_iter,
			)
		)
		method = self.opt_method.lower()
		if method == "grad" and not self.kernel_.requires_grad:
			method = "grid"

		optimizer = self.optimizer.lower()
		params = dict(self.optimizer_params or {})

		if method == "grad":
			if not OptimizerFactory.supports(
                optimizer,
                category="gradient",
            ):
				raise ValueError(
					f"Optimizer '{optimizer}' "
					f"does not support gradient optimization. "
					f"Available: "
					f"{OptimizerFactory.available_by_category('gradient')}"
				)
			params.update({
				"learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
			})
		elif method == "grid":
			if not OptimizerFactory.supports(
				optimizer,
				category="search",
			):
				raise ValueError(
					f"Optimizer '{optimizer}' "
					f"does not support grid search optimization. "
					f"Available: "
					f"{OptimizerFactory.available_by_category('search')}"
				)
			if self.one_parameter:
				params.update({
					"param_grid": {
						"alpha": alpha_candidates
					}
				})
			else:
				params.update({
					"param_grid": {
						"alpha": alpha_candidates,
						"beta": beta_candidates
					}
				})
		
		else:
			raise ValueError(
				f"Unknown optimization method: {self.opt_method}. Supported: 'grad', 'grid'."
			)
		
		self.optimizer_ = OptimizerFactory.create(
			optimizer,
			**params,
			random_state=self.random_state
		)
		
		if method == "grad":
			if self.one_parameter:
				result = self.optimizer_(
					self.kappa_cross_validation_error_1d,
					np.array([1.0])
				)
			else:
				result = self.optimizer_(
					self.kappa_cross_validation_error_2d,
					np.array([1.0, 1.0])
				)
		else:
			if self.one_parameter:
				param_grid = {"bandwidth": alpha_candidates}
				result = self.optimizer_(
					self.kappa_cross_validation_error_1d,
					param_grid
				)
			else:
				param_grid = {
					"alpha": alpha_candidates,
					"beta": beta_candidates
				}
				result = self.optimizer_(
					self.kappa_cross_validation_error_2d,
					param_grid
				)
		history_df = history_to_dataframe(
            result["history"],
            param_names=["alpha", "beta"],
        )
		self.optimization_outputs_ = {
            "method": self.opt_method,
            "score": result["score"],
            "history": history_df,
            "evaluations": len(result["history"]),
			"params" : result['x']
        }



	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		X_l: np.ndarray | None = None,
		y_l: np.ndarray | None = None,
		split_ratio: float = 0.5,
		overlap: float = 0.0,
		pred_features: np.ndarray | None = None,
		as_predictions=False,
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
		
		ctx = resolve_training_context(
            X,
            y,
            X_l=X_l,
            y_l=y_l,
			pred_features=pred_features,
            as_predictions=as_predictions,
            split_ratio=split_ratio,
            overlap=overlap,
            random_state=self.random_state
        )
		self.X_k_ = ctx.X_k
		self.y_k_ = ctx.y_k
		self.X_l_ = ctx.X_l
		self.y_l_ = ctx.y_l
		self.as_predictions_ = ctx.as_predictions
		self.global_mean_ = float(np.mean(self.y_l_))

		if not self.as_predictions_:
			self.estimators_ = self._fit_estimators(self.X_k_, self.y_k_)
			prediction_space = self._load_predictions(self.X_l_)
		else:
			prediction_space = self.X_l_
		
		self.normalize_constant_x_ = (
			compute_normalization_constant(
				X,
				norm_constant=self.norm_constant_x,
				scale_factor=5.0,
				M=prediction_space.shape[1]
			)
		) 
		self.normalize_constant_y_ = (
			compute_normalization_constant(
				y,
				norm_constant=self.norm_constant_y,
				scale_factor=50.0,
				M=prediction_space.shape[1]
			)
		)
		self.X_l_norm_ = self.X_l_ * self.normalize_constant_x_
		self.Y_l_norm_ = (
			prediction_space
			* self.normalize_constant_y_
		)
		

		self._resolve_component()

		if self.one_parameter:
			self.mix_features_ = np.column_stack([
				self.X_l_norm_,
                self.Y_l_norm_
			])
			self.distance_matrix_mix_ = (
				self.distance_.matrix(
					self.mix_features_,
                    self.mix_features_
				)
			)
		else:
			self.distance_matrix_x_ = (
                self.distance_.matrix(
                    self.X_l_norm_,
                    self.X_l_norm_,
                )
            )
			self.distance_matrix_y_ = (
                self.distance_.matrix(
                    self.Y_l_norm_,
                    self.Y_l_norm_,
                )
            )

		self.cv_folds_ = list(self.cv_.split(self.X_l_, self.y_l_))

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
			pred_X = self._load_predictions(X)

		X_norm = X * self.normalize_constant_x_
		Y_norm = pred_X * self.normalize_constant_y_

		if self.one_parameter:
			bandwidth = (
                self.optimization_outputs_["params"][0]
            )
			mix_test = np.column_stack([
                X_norm,
                Y_norm,
            ])
			dist_mix = self.distance_.matrix(
				mix_test,
				self.mix_features_
			)
			self.adapter_.set_params(bandwidth=bandwidth)
			D = self.adapter_.transform(dist_mix)
		else:
			alpha, beta = (
                self.optimization_outputs_["params"]
            )
			dist_x = self.distance_.matrix(
                X_norm,
                self.X_l_norm_,
            )
			dist_y = self.distance_.matrix(
                Y_norm,
                self.Y_l_norm_,
            )
			self.adapter_.set_params(alpha=alpha, beta=beta)
			D = self.adapter_.transform(dist_x, dist_y)
		
		K = self.kernel_(D)

		preds = self.aggregator_.aggregate_matrix(
            values=self.y_l_,
            weights=K,
            fallback=self.global_mean_,
        )

		return preds
