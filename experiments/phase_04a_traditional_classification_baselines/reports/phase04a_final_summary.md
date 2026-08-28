# Phase 04A Final Summary

- Frozen Phase 03 SHA-256: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`
- Primary data: 419 runs, 35 subjects, 1,176 features.
- Outer CV: frozen subject-wise five-fold; inner CV: subject-wise GroupKFold(3).
- Primary selection metric: Macro-F1.
- Gradient Boosting used the V2 candidate-level checkpoint workflow; all five outer folds are complete.

## Complete OOF comparison

| model               | model_slug          |   oof_macro_f1 |   oof_balanced_accuracy |   oof_accuracy |   oof_weighted_f1 |   recall_class_0 |   recall_class_1 |   recall_class_2 |   recall_class_3 |   fold_macro_f1_mean |   fold_macro_f1_std |
|:--------------------|:--------------------|---------------:|------------------------:|---------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------:|---------------------:|--------------------:|
| Gradient Boosting   | gradient_boosting   |       0.935608 |                0.935686 |       0.935561 |          0.935637 |         0.942308 |         0.886792 |         0.932692 |         0.980952 |             0.935888 |           0.0189541 |
| Random Forest       | random_forest       |       0.930234 |                0.931128 |       0.930788 |          0.93018  |         0.951923 |         0.830189 |         0.951923 |         0.990476 |             0.930476 |           0.0319005 |
| RBF SVM             | rbf_svm             |       0.887349 |                0.888087 |       0.887828 |          0.887266 |         0.903846 |         0.811321 |         0.903846 |         0.933333 |             0.888231 |           0.043658  |
| Linear SVM          | linear_svm          |       0.85733  |                0.856883 |       0.856802 |          0.857254 |         0.807692 |         0.820755 |         0.903846 |         0.895238 |             0.858126 |           0.0473179 |
| Logistic Regression | logistic_regression |       0.842475 |                0.842641 |       0.842482 |          0.842317 |         0.769231 |         0.783019 |         0.923077 |         0.895238 |             0.842926 |           0.0546106 |
| KNN                 | knn                 |       0.753279 |                0.75651  |       0.756563 |          0.753195 |         0.682692 |         0.669811 |         0.721154 |         0.952381 |             0.752727 |           0.0518022 |

## Best traditional classifier

Gradient Boosting — OOF Macro-F1 0.935608.

XGBoost: OPTIONAL / NOT RUN. No statistical-significance claim is made in Phase 04A.