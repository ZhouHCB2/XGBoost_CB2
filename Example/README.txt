1.the example of data preprocess
  preprocessing_desc_fp.ipynb

2.metric calculation
  calc_metric.py

*the model hyperparameters have been optimized
*the best fingerprint combination for regression model is: AARM(AvalonFP+AtomPairFP+RDkitFP+MorganFP)
                                      classification model is: AMA(AtomPairFP+MoeganFP+AvalonFP)

3.training set
1). CB2_Chembl_R6227.csv (for regression task)

2). CB2_Chembl_C7758.csv (for classification task)

4. params file
1). params_regress (for regression task)

2). params_class (for classification task)

5.features (optimum fingerprints combination)
1). AvalonFP_AtomPairFP_RDkitFP_MorganFP.csv (for regression task)

2). AvalonFP_MorganFP_AtomPairFP.csv (for classification task)

6.optimum model training example 
1).xgb regression model
   shap_CB2_reg_xgboost_best_fps.ipynb

2).xgb classification model
   shap_CB2_class_xgboost_best_fps.ipynb