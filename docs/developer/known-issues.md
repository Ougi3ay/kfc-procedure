# Known Issues

These issues were inferred from the inspected codebase.

## `predict_proba` path is incomplete

`KFCProcedure.predict_proba()` calls `self.fstep_.predict_proba(...)`, but `FStep` currently implements `predict()` only. Add `FStep.predict_proba` before relying on probability outputs.

## Random state is passed to stateless combiners

`CStep._build_combiner()` injects `random_state` into `combiner_params` if it is missing. Stateless combiners such as `MeanCombiner` and `MajorityVoteCombiner` do not define `random_state` in their constructors, so this can raise a constructor error depending on usage.

Potential fix: only pass `random_state` if the target constructor accepts it.

## Internal split is fixed at 50/50

`KFCProcedure.fit()` uses `train_test_split(..., test_size=0.5)`. This is simple and helps calibrate the C-step, but it reduces the number of samples available for local model training.

Potential improvement: add a `calibration_size` parameter.

## NaN prediction risk in F-step

`FStep.predict()` initializes predictions with `NaN`. If a predicted cluster has no corresponding trained local model, the NaN can propagate to the C-step.

Potential improvement: add fallback strategies such as nearest available cluster, global fallback model, or cluster-prior fallback.
