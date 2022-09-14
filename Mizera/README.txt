1.Mizera.csv is the dataset developed by  Mizera et al
title: Virtual Screening of C. Sativa Constituents for the Identification of Selective Ligands for Cannabinoid Receptor 2
DOI£º10.3390/ijms21155308

Mozera' model R2=0.620

*the best XBGoost model were used to compare with the model developed by Mizera

dataset Mizera.csv was split by ten different random (1,2,4,8,16,32,64,128,256,512) seeds to divided into train,validation,test by a proportion of 8:1:1


2.results of the xgboost
1).ml_reg_append.csv 
   records the ten running performances of the best regression model (hyperparameters were optimized, fingerprints combination:AARM)on the Mizera dataset
   
2).ml_reg_grouped.csv records the average performance of the best regression model (hyperparameters were optimized, fingerprints combination:AARM)on the Mizera dataset



