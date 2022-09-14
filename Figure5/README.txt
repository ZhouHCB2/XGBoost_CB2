*All the results were the average value of ten runs. In each run, the CB2_Chembl_C dataset was split randomly as training, validation and test dataset (ratio: 0.8, 0.1, 0.1) using the random seed chosen from 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512.

Figure5A regression tasks
1.records the results of ten running
   xgb_reg_compareFP_append.csv
   records the the performance of xgb regression models based on the CB2_Chembl_R training set, combined with different single fingerprints,and these results were repeated ten times with different random seeds

2.records the average reult of ten running
   xgb_reg_compareFP_group.csv 
   summarizes the above results by taking the mean of ten results

3.ranking the result of the xgb combined with different features
   ml_reg_compareFP_grouped_ranked.csv 
   further sort the results in order to see how they perform

Figure5B classification tasks
1.records the results of ten running
   xgb_compareFP_on_class4_append.csv
   records the the performance of xgb classification models based on the CB2_Chembl_R training set, combined with different single fingerprints,and these results were repeated ten times with different random seeds

2.records the average reult of ten running
   xgb_compareFP_on_class4_group.csv 
   summarizes the above results by taking the mean of ten results

3.ranking the result of the xgb combined with different features
   xgb_compareFP_on_class4_grouped_ranked.csv 
   further sort the results in order to see how they perform
