# Phase 07 statistical appendix

Statistical unit: subject_id (n=35). Shared paired bootstrap: 2,000 repetitions, seed 42, percentile 95% CI.

## Friedman tests
| task           | metric      | status   |   statistic |     p_value |   n_subjects |
|:---------------|:------------|:---------|------------:|------------:|-------------:|
| classification | macro_f1    | PASS     |     91.6901 | 5.75985e-19 |           35 |
| regression     | bounded_mae | PASS     |     78.1486 | 4.29654e-16 |           35 |

## Wilcoxon-Holm tests
| task           | modality                  | metric      | status   |   statistic |       raw_p |   effect_size_rank_biserial | effect_definition         |   n_subjects |   holm_adjusted_p | significant_alpha_0_05   |
|:---------------|:--------------------------|:------------|:---------|------------:|------------:|----------------------------:|:--------------------------|-------------:|------------------:|:-------------------------|
| classification | physiological_features    | macro_f1    | PASS     |         0   | 5.82077e-11 |                   1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| classification | eye_tracking_features     | macro_f1    | PASS     |         0   | 5.82077e-11 |                   1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| classification | head_movement_features    | macro_f1    | PASS     |         0   | 5.82077e-11 |                   1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| classification | flight_parameter_features | macro_f1    | PASS     |        71.5 | 0.812613    |                   0.0653595 | multimodal minus unimodal |           35 |       0.812613    | False                    |
| classification | body_movement             | macro_f1    | PASS     |         0   | 5.82077e-11 |                   1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| regression     | physiological_features    | bounded_mae | PASS     |         0   | 5.82077e-11 |                  -1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| regression     | eye_tracking_features     | bounded_mae | PASS     |         0   | 5.82077e-11 |                  -1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| regression     | head_movement_features    | bounded_mae | PASS     |         0   | 5.82077e-11 |                  -1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |
| regression     | flight_parameter_features | bounded_mae | PASS     |       243   | 0.244903    |                   0.228571  | multimodal minus unimodal |           35 |       0.244903    | False                    |
| regression     | body_movement             | bounded_mae | PASS     |         0   | 5.82077e-11 |                  -1         | multimodal minus unimodal |           35 |       2.91038e-10 | True                     |

## Bootstrap confidence intervals
| task           | modality                  | metric            |   point_estimate |   ci_95_lower |   ci_95_upper |   n_subjects |   repetitions |   bootstrap_seed | resampling_unit   | interval   |
|:---------------|:--------------------------|:------------------|-----------------:|--------------:|--------------:|-------------:|--------------:|-----------------:|:------------------|:-----------|
| classification | physiological_features    | macro_f1          |        0.33219   |    0.29363    |     0.367828  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | physiological_features    | balanced_accuracy |        0.340745  |    0.305225   |     0.377212  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | physiological_features    | severe_error_rate |        0.312649  |    0.269004   |     0.355609  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | eye_tracking_features     | macro_f1          |        0.346833  |    0.299951   |     0.390278  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | eye_tracking_features     | balanced_accuracy |        0.351635  |    0.307476   |     0.395125  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | eye_tracking_features     | severe_error_rate |        0.238663  |    0.190476   |     0.286413  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | head_movement_features    | macro_f1          |        0.373296  |    0.336032   |     0.410682  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | head_movement_features    | balanced_accuracy |        0.375151  |    0.340004   |     0.413042  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | head_movement_features    | severe_error_rate |        0.23389   |    0.190931   |     0.278571  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | flight_parameter_features | macro_f1          |        0.863458  |    0.801834   |     0.918509  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | flight_parameter_features | balanced_accuracy |        0.864093  |    0.803013   |     0.918847  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | flight_parameter_features | severe_error_rate |        0.0119332 |    0.00238649 |     0.0261905 |           35 |          2000 |               42 | subject_id        | percentile |
| classification | body_movement             | macro_f1          |        0.228379  |    0.195438   |     0.255354  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | body_movement             | balanced_accuracy |        0.245077  |    0.220353   |     0.270832  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | body_movement             | severe_error_rate |        0.331742  |    0.293556   |     0.369048  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | multimodal_reference      | macro_f1          |        0.871057  |    0.823087   |     0.914257  |           35 |          2000 |               42 | subject_id        | percentile |
| classification | multimodal_reference      | balanced_accuracy |        0.871236  |    0.823727   |     0.91412   |           35 |          2000 |               42 | subject_id        | percentile |
| classification | multimodal_reference      | severe_error_rate |        0.0190931 |    0.00477327 |     0.0358852 |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | physiological_features    | bounded_mae       |        1.01079   |    0.934576   |     1.08555   |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | physiological_features    | bounded_rmse      |        1.29195   |    1.20096    |     1.37871   |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | eye_tracking_features     | bounded_mae       |        0.847353  |    0.784116   |     0.910695  |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | eye_tracking_features     | bounded_rmse      |        1.057     |    0.978114   |     1.13632   |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | head_movement_features    | bounded_mae       |        0.890777  |    0.82434    |     0.959102  |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | head_movement_features    | bounded_rmse      |        1.09876   |    1.02309    |     1.17493   |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | flight_parameter_features | bounded_mae       |        0.26111   |    0.207858   |     0.322725  |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | flight_parameter_features | bounded_rmse      |        0.389104  |    0.300815   |     0.478717  |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | body_movement             | bounded_mae       |        1.02607   |    0.970641   |     1.08285   |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | body_movement             | bounded_rmse      |        1.26688   |    1.20326    |     1.33191   |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | multimodal_reference      | bounded_mae       |        0.265727  |    0.212184   |     0.326712  |           35 |          2000 |               42 | subject_id        | percentile |
| regression     | multimodal_reference      | bounded_rmse      |        0.394327  |    0.307341   |     0.484085  |           35 |          2000 |               42 | subject_id        | percentile |
