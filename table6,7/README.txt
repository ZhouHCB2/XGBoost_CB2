1.datasets
1).training set: ChblClass7758.csv, ChblReg6227.csv (random seed split to divide training set into train, validation, and test 8:1:1,the training set is used to build the model, the validation set is applied to adjust hyperparameters, and the test set for final evaluation of the model prediction ability. And the robustness of each model was assessed by ten different random seeds (1, 2, 4, 8, 16, 32, 64, 128, 256 and 512). )

2).external validation set: CB2_BDB_C1698.csv, BdbReg1344.csv

2.results
1).DMPNN
   The previously trained DMPNN model is used to predict the external validation set to evaluate its generalization
   In DMPNN,classification records the result of the classification model
            regression records the result of the regression model

eva_BdbClass1698_filNullwith0_P_ext_*.csv the results of ten runs
 
eva_folds_BdbClass1698_filNullwith0_P_ext_class4.csv records the average result of ten predictions

2).MolMap
   due to the C-path obtained the optimum performance in previous studis, so we use molmap combined with C-path to evaluate the generality of the molmap

   mp_reg_rand_*.csv records the prediction result of MolMap regression model, append file records the result of ten running, and the group file records the mean result

   mp_class_rand_*.csv records the prediction result of MolMap classification model, append file records the result of ten running, and the group file records the mean result

3).the performance of the XGBoost can be truced in Table 5

3.codes
1).molmap regression model
   01b_cmp_Cpath_reg_rand_with_ext.py (records the codes of the molmap regression model)
   run_reg_cmd (records the commands)

2).molmap classification model
   01b_cmp_Cpath_class_rand_with_ext.py (records the codes of the molmap classification model)
   file run_reg_cmd (records the commands)


