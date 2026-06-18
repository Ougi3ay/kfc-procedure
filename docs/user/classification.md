# Classification

Use `KFCClassifier` for categorical labels.

```python
from kfc_procedure import KFCClassifier

clf = KFCClassifier(
    divergences=["euclidean"],
    local_model="decision_tree_classifier",
    combiner="majority_vote",
    n_clusters=2,
    random_state=42,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## Classification combiners

| Combiner | Behavior |
|---|---|
| `majority_vote` | most frequent local prediction |
| `stacking_classifier` | logistic regression meta-classifier |
| `combined_classifier` | COBRA-style weighted vote |

!!! warning "One-class clusters"
    If a cluster contains only one class, some local classifiers may fail. Reduce `n_clusters`, use more data, or choose a classifier that can handle small local samples.

!!! warning "Current `predict_proba` limitation"
    The inspected code defines `KFCProcedure.predict_proba()`, but `FStep` does not currently implement `predict_proba()`. Probability prediction may require a patch.
