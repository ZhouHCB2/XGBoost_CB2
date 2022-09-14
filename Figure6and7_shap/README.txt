*All the results were the average value of ten runs. In each run, the CB2_Chembl_C dataset was split randomly as training, validation and test dataset (ratio: 0.8, 0.1, 0.1) using the random seed chosen from 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512.
*The easy-to-interpret alternative fingerprints are MACCSFP, PubChemFP and MorganFP

1.datasets
1).regression task
   CB2_Chembl_R6227.csv

2).classification task
   CB2_Chembl_C7758.csv

2.features
1).regression task
   Mo_ma_r.csv
   Mo_Pu_r.csv
   Pu_ma_r.csv
   Mo_Pu_ma_r.csv

2).classification task
   Mo_ma_c.csv
   Mo_Pu_c.csv
   Pu_ma_c.csv
   Mo_Pu_ma_c.csv

3.params
1).regression task
   params_regress

2).classification task
   params_calss

4.instance of shap calculation
shap_boston_reg_xgboost_sklearn_api.ipynb

5.the performance of the easy-to-interpret alternative fingerprint combinations
1).class.ipynb 
   XGBoost + hyperparameters optimized, and the result of the performance is calculated by the mean of ten different random seeds 

2).regress.ipynb
   XGBoost + hyperparameters optimized, and the result of the performance is calculated by the mean of ten different random seeds  

*The optimum easy-to-interpret alternative fingerprints combination for classification and regression tasks both are MACCSFP + PubChemFP + MorganFP

6.shap calculation for optimum easy-to-interpret fingerprint combination
1).SHAP_class.ipynb 
   records the fingerprint importance information about the XGBoost classification model, (model: XGBoost+optimized_hyperparameters+optimum easy-to-interpret fingerprint combination), which is calculated by shap.

2).SHAP_regress.ipynb 
   records the fingerprint importance information about the XGBoost regression model, (model: XGBoost+optimized_hyperparameters+optimum easy-to-interpret fingerprint combination), which is calculated by shap.