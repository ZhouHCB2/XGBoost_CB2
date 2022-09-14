1.Datasets
1). training set: ChblClass7758.csv, ChblReg6227.csv (random seed split to divide training set into train, validation, and test 8:1:1,the training set is used to build the model, the validation set is applied to adjust hyperparameters, and the test set for final evaluation of the model prediction ability. And the robustness of each model was assessed by ten different random seeds (1, 2, 4, 8, 16, 32, 64, 128, 256 and 512). )

2). external validation set: CB2_BDB_C1698.csv, BdbReg1344.csv

*.optimum fingerprint combinations for regression model is AvalonFP+AtomPairFP+RDkitFP+MorganFP(AARM), for classification task is AtomPairFP+MorganFP+AvalonFP (AMA)
*.in this part: XGBoost regression model is XGBoost+AARM; XGBoost classification model is AMA

2.results of the regression model
1).the performance of XGBoost regression model under different hyperparameters (ten times)
   ml_reg_append.csv 

2).the best parameters of XGBoost regression model and the performance of the model under the best parameters
   ml_reg_grouped.csv

3.results of the classification model
1).the performance of XGBoost classification model under different hyperparameters (ten times)
   ml_class_append.csv

2).the best parameters of XGBoost classification model and the performance of the model under the best parameters
   ml_class_grouped.csv

3.params (records the parameters of the best xgbosst model)
1).regression model
   params_regress 

2).classification model
   params_class

4.features
1).regression task
   AvalonFP_AtomPairFP_RDkitFP_MorganFP.csv

2).classification task
   AvalonFP_MorganFP_AtomPairFP.csv

5.training example
1).training process and results of the XGBoost classification model under the optimal parameters.
   CB2_class_xgboost_best_fps.ipynb

2).training process and results of the XGBoost regression model under the optimal parameters. 
   CB2_reg_xgboost_best_fps.ipynb 

*The following is the optimized result
6.evaluate by external validation dataset
1).xgboost regression model
   ml_reg_pred_external_*.csv (records the performance of the best regression model on the external validation set, the append file records the results of running the model ten times under ten random seeds, and the group records the average results of ten times)

2).xgboost classification model
   ml_class_pred_external_*.csv (records the performance of the best classification model on the external validation set, the append file records the results of running the model ten times under ten random seeds, and the group records the average results of ten times)