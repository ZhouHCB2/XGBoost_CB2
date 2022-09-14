*All the results were the average value of ten runs. In each run, the CB2_Chembl_C dataset was split randomly as training, validation and test dataset (ratio: 0.8, 0.1, 0.1) using the random seeds (1, 2, 4, 8, 16, 32, 64, 128, 256 and 512).
 Training used to fit model, validation for adjust parameters, and test for valuate the ferformance

*the top5 fingerprint for regression task are AvalonFP, AtomPairFP, RdkitFP, MorganFP, and TorsionFP
                          classification task are AtomPairFP, MorganFP, AvalonFP, MAP4, and RDkitFP

1.Table3 (regression tasks, xgboost combined with different top5 fingerprint combinations)
1). results of ten runs of the xgb regression model
    xgb_reg_comFP_append.csv 
 
2). average result of xgb regression model
    xgb_reg_combFP_grouped.csv

2.Table4 (classification tasks, xgboost combined with different top5 fingerprint combinations)
1). results of ten runs of the xgb classification model
    xgb_class_comFP_append.csv 
 
2). average result of xgb classification model
    xgb_class_combFP_grouped.csv

3.the training example of the xgboost model combined with different fingerprint combinations
1). regression tasks
    regress.ipynb

2). classification tasks
    class.ipynb




