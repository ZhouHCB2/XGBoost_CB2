*All the results were the average value of ten runs. In each run, the CB2_Chembl_C dataset was split randomly as training, validation and test dataset (ratio: 0.8, 0.1, 0.1) using the random seed chosen from 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512.


1.training sets

1).randomsplit file
 class for classification tasks, reg for regression tasks.(model include XGBoost,RF,KNN,MolMap)

2).DMPNN file
 class_trainset for classification tasks, reg_trainset for regression tasks. (model: DMPNN)


2.Results of Table2 (records the results of the classification tasks)

1).ml_class_3pth_* records the classification performance of the XGBoost, RF and KNN based on three kinds of feature genarated path
	ml_class_3path_append.csv records the results of the model under ten random seeds
	ml_class_3path_grouped.csv averages the ten results to summarize

2).mp_class_3pth_rand_* records the classification performance of the MolMap based on three kinds of feature genarated path
	mp_class_3path_rand_append.csv records the results of the model under ten random seeds
	mp_class_3path_rand_grouped.csv averages the ten results to summarize

3).inside file DMPNN, eva_calss_test_folds.csv records the results of the DMPNN classification model


3.Results of Figure3 (records the results of the regression tasks)

1).ml_reg_3pth_* records the regression performance of the XGBoost, RF and KNN based on three kinds of feature genarated path
	ml_reg_3path_append.csv records the results of the model under ten random seeds
	ml_reg_3path_grouped.csv averages the ten results to summarize

2).mp_reg_3pth_rand_* records the regression performance of the MolMap based on three kinds of feature genarated path
	mp_class_3path_rand_append.csv records the results of the model under ten random seeds
	mp_class_3path_rand_grouped.csv averages the ten results to summarize

3).inside file DMPNN, eva_reg_test_folds.csv records the results of the DMPNN regression model