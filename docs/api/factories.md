# Factory Registry Reference

!!! note "Source-grounded documentation"
    This documentation was generated from direct inspection of the provided repository, packaging metadata, notebooks, tests, and thesis files. It documents behavior observed in source code and tests only. Analysis date: **2026-06-12**.

The package uses registry-based factories. The names below were extracted from the running implementation during documentation generation.

## `BregmanDivergenceFactory`

Module: `kfc_procedure.core.clustering.divergences.base`

Registered names (4):

- `euclidean`, `gkl`, `is`, `logistic`

## `LocalModelFactory`

Module: `kfc_procedure.core.ml.base`

Registered names (101):

- `ada_boost_classifier`, `ada_boost_regressor`, `ard_regression`, `bagging_classifier`, `bagging_regressor`
- `bayesian_ridge`, `bernoulli_nb`, `calibrated_classifier_cv`, `categorical_nb`, `cca`
- `classifier_chain`, `complement_nb`, `decision_tree_classifier`, `decision_tree_regressor`, `dummy_classifier`
- `dummy_mean`, `dummy_regressor`, `elastic_net`, `elastic_net_cv`, `extra_tree_classifier`
- `extra_tree_regressor`, `extra_trees_classifier`, `extra_trees_regressor`, `fixed_threshold_classifier`, `gamma_regressor`
- `gaussian_nb`, `gaussian_process_classifier`, `gaussian_process_regressor`, `gradient_boosting_classifier`, `gradient_boosting_regressor`
- `hist_gradient_boosting_classifier`, `hist_gradient_boosting_regressor`, `huber_regressor`, `isotonic_regression`, `k_neighbors_classifier`
- `k_neighbors_regressor`, `kernel_ridge`, `label_propagation`, `label_spreading`, `lars`
- `lars_cv`, `lasso`, `lasso_cv`, `lasso_lars`, `lasso_lars_cv`
- `lasso_lars_ic`, `linear_discriminant_analysis`, `linear_regression`, `linear_svc`, `linear_svr`
- `logistic_regression`, `logistic_regression_cv`, `mean_regressor`, `mlp_classifier`, `mlp_regressor`
- `multi_output_classifier`, `multi_output_regressor`, `multi_task_elastic_net`, `multi_task_elastic_net_cv`, `multi_task_lasso`
- `multi_task_lasso_cv`, `multinomial_nb`, `nearest_centroid`, `nu_svc`, `nu_svr`
- `one_vs_one_classifier`, `one_vs_rest_classifier`, `orthogonal_matching_pursuit`, `orthogonal_matching_pursuit_cv`, `output_code_classifier`
- `passive_aggressive_classifier`, `passive_aggressive_regressor`, `perceptron`, `pls_canonical`, `pls_regression`
- `poisson_regressor`, `quadratic_discriminant_analysis`, `quantile_regressor`, `radius_neighbors_classifier`, `radius_neighbors_regressor`
- `random_forest_classifier`, `random_forest_regressor`, `ransac_regressor`, `regressor_chain`, `ridge`
- `ridge_classifier`, `ridge_classifier_cv`, `ridge_cv`, `self_training_classifier`, `sgd_classifier`
- `sgd_regressor`, `stacking_classifier`, `stacking_regressor`, `svc`, `svr`
- `theil_sen_regressor`, `transformed_target_regressor`, `tuned_threshold_classifier_cv`, `tweedie_regressor`, `voting_classifier`
- `voting_regressor`

## `CombinerFactory`

Module: `kfc_procedure.core.combiner.base`

Registered names (8):

- `combined_classifier`, `gradientcobra`, `majority_vote`, `mean`, `mixcobra`
- `stacking_classifier`, `stacking_regressor`, `weighted_mean`

## `SplitterFactory`

Module: `kfc_procedure.cobra.core.splitters.base`

Registered names (3):

- `holdout`, `random_holdout`, `split_overlap`

## `EstimatorFactory`

Module: `kfc_procedure.cobra.core.estimators.base`

Registered names (101):

- `ada_boost_classifier`, `ada_boost_regressor`, `ard_regression`, `bagging_classifier`, `bagging_regressor`
- `bayesian_ridge`, `bernoulli_nb`, `calibrated_classifier_cv`, `categorical_nb`, `cca`
- `classifier_chain`, `complement_nb`, `decision_tree_classifier`, `decision_tree_regressor`, `dummy_classifier`
- `dummy_regressor`, `elastic_net`, `elastic_net_cv`, `extra_tree_classifier`, `extra_tree_regressor`
- `extra_trees_classifier`, `extra_trees_regressor`, `fixed_threshold_classifier`, `gamma_regressor`, `gaussian_nb`
- `gaussian_process_classifier`, `gaussian_process_regressor`, `gradient_boosting_classifier`, `gradient_boosting_regressor`, `hist_gradient_boosting_classifier`
- `hist_gradient_boosting_regressor`, `huber_regressor`, `isotonic_regression`, `k_neighbors_classifier`, `k_neighbors_regressor`
- `kernel_ridge`, `label_propagation`, `label_spreading`, `lars`, `lars_cv`
- `lasso`, `lasso_cv`, `lasso_lars`, `lasso_lars_cv`, `lasso_lars_ic`
- `linear_discriminant_analysis`, `linear_regression`, `linear_svc`, `linear_svr`, `logistic_regression`
- `logistic_regression_cv`, `mean`, `mean_regressor`, `mlp_classifier`, `mlp_regressor`
- `multi_output_classifier`, `multi_output_regressor`, `multi_task_elastic_net`, `multi_task_elastic_net_cv`, `multi_task_lasso`
- `multi_task_lasso_cv`, `multinomial_nb`, `nearest_centroid`, `nu_svc`, `nu_svr`
- `one_vs_one_classifier`, `one_vs_rest_classifier`, `orthogonal_matching_pursuit`, `orthogonal_matching_pursuit_cv`, `output_code_classifier`
- `passive_aggressive_classifier`, `passive_aggressive_regressor`, `perceptron`, `pls_canonical`, `pls_regression`
- `poisson_regressor`, `quadratic_discriminant_analysis`, `quantile_regressor`, `radius_neighbors_classifier`, `radius_neighbors_regressor`
- `random_forest_classifier`, `random_forest_regressor`, `ransac_regressor`, `regressor_chain`, `ridge`
- `ridge_classifier`, `ridge_classifier_cv`, `ridge_cv`, `self_training_classifier`, `sgd_classifier`
- `sgd_regressor`, `stacking_classifier`, `stacking_regressor`, `svc`, `svr`
- `theil_sen_regressor`, `transformed_target_regressor`, `tuned_threshold_classifier_cv`, `tweedie_regressor`, `voting_classifier`
- `voting_regressor`

## `NormalizerFactory`

Module: `kfc_procedure.cobra.core.normalizers.base`

Registered names (3):

- `minmax`, `standard`, `zscore`

## `DistanceFactory`

Module: `kfc_procedure.cobra.core.distances.base`

Registered names (8):

- `cosine`, `euclidean`, `hamming`, `l1`, `l2`
- `lp`, `manhattan`, `minkowski`

## `KernelFactory`

Module: `kfc_procedure.cobra.core.kernels.base`

Registered names (12):

- `biweight`, `cauchy`, `cobra`, `epanechnikov`, `exponential`
- `gaussian`, `naive`, `radial`, `rbf`, `reverse_cosh`
- `triangular`, `triweight`

## `KernelAdapterFactory`

Module: `kfc_procedure.cobra.core.adapters.base`

Registered names (2):

- `one_parameter`, `two_parameter`

## `LossFactory`

Module: `kfc_procedure.cobra.core.losses.base`

Registered names (10):

- `cross_entropy`, `hinge`, `huber`, `l1`, `l2`
- `log_loss`, `mae`, `mse`, `quantile`, `squared_error`

## `CVFactory`

Module: `kfc_procedure.cobra.core.cv.base`

Registered names (4):

- `kfold`, `stratified_kfold`, `time_series`, `tscv`

## `OptimizerFactory`

Module: `kfc_procedure.cobra.core.optimizers.base`

Registered names (4):

- `adam`, `gd`, `grid`, `momentum`

## `AggregatorFactory`

Module: `kfc_procedure.cobra.core.aggregators.base`

Registered names (4):

- `weighted_mean`, `weighted_vote`, `wm`, `wv`


## Developer usage pattern

```python
from kfc_procedure.core.ml.base import LocalModelFactory

print(LocalModelFactory.available())
model = LocalModelFactory.create("ridge", alpha=1.0)
```

Use `Factory.available()` before documenting examples or writing notebooks, because short aliases in a README or thesis draft may not be registered in the implementation.
