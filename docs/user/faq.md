# FAQ

??? question "Why install `kfc-procedure` but import `kfc_procedure`?"
    PyPI package names can contain hyphens, but Python imports cannot. Use `pip install kfc-procedure` and `import kfc_procedure`.

??? question "Which divergence should I start with?"
    Start with `euclidean`, because it supports real-valued data without special domain constraints.

??? question "Why do `gkl`, `is`, or `logistic` fail?"
    Their domains are restricted. `gkl` and `is` require positive data. `logistic` requires values in `(0, 1)`.

??? question "How many clusters should I use?"
    Start with `2` or `3`. Too many clusters can make local models unstable.

??? question "Can I use sklearn models?"
    Yes. The package auto-registers many sklearn estimators using snake_case names such as `linear_regression`, `decision_tree_classifier`, and `random_forest_regressor`.

??? question "What is COBRA?"
    COBRA is an ensemble aggregation method that compares samples in prediction space and aggregates nearby targets using kernel weights.
