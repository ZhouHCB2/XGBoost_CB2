import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler


### standardize features
fingerprint_list = ['MorganFP', 'RDkitFP', 'AtomPairFP', 'TorsionFP', 'AvalonFP', 'EstateFP', 'MACCSFP', 'PharmacoErGFP', 'PharmacoPFP', 'PubChemFP', 'MHFP6', 'MAP4']
descriptor_list = ['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge', 'Estate', 'MOE', 'Connectivity', 'Topology', 'Kappa', 'Path', 'Matrix', 'InfoContent']
df=pd.DataFrame()
for fp in descriptor_list:
    fp_file=f'ChblReg6227_{fp}.csv';#print(fp_file)
    tdf=pd.read_csv(fp_file)
    df=pd.concat([df,tdf],axis=1)
df.to_csv('all_desc.csv',index=False)

# remove those columns with same values in descriptors
nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
print(f'cols_to_drop = {cols_to_drop}')
pd.DataFrame(cols_to_drop.to_list()).to_csv('desc_cols_to_drop.csv')
df.drop(cols_to_drop, axis=1,inplace=True)
df.to_csv('all_desc_drop_same_value_cols.csv')
T=df.T
T.drop_duplicates(inplace=True)
df=T.T
df.to_csv('all_desc_dropSame_DeDup.csv',index=False)
cols=df.columns
for c in cols:
    if (df[cols[0]].isna()).all():
        print(df[df[cols[0]].isna()])
np.where(df.values==np.inf)
df.iloc[np.where(df.values==np.inf)[0],np.where(df.values==np.inf)[1]]
df.iloc[:,~]
np.where(df.values==np.inf)[1]
inf_cols_num=np.where(df.values==np.inf)[1]
no_inf_cols_num=[n for n in range(len(df.columns)) if n not in inf_cols_num]
no_inf_df=df.iloc[:,no_inf_cols_num]
no_inf_df.to_csv('all_desc_dropSame_DeDup_rmInf.csv')

scaler=MinMaxScaler()
scaler.fit(no_inf_df.values)
trans_val=scaler.transform(no_inf_df.values)
trans_df=pd.DataFrame(trans_val,columns=no_inf_df.columns)
trans_df.to_csv('MinMaxed_desc.csv',index=False)

# remove those columns with same values in fingerprints
df=pd.DataFrame()
for fp in fingerprint_list:
    fp_file=f'ChblReg6227_{fp}.csv';#print(fp_file)
    tdf=pd.read_csv(fp_file)
    df=pd.concat([df,tdf],axis=1)

df.to_csv('all_fp.csv',index=False)
nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index
print(f'cols_to_drop = {cols_to_drop}')
pd.DataFrame(cols_to_drop.to_list()).to_csv('fp_cols_to_drop.csv')
df.drop(cols_to_drop, axis=1,inplace=True)
df.to_csv('all_fp_dropSame.csv')

T=df.T
T.drop_duplicates(inplace=True)
df=T.T
df.to_csv('all_fp_dropSame_DeDup.csv',index=False)
desc_fp_df=pd.concat([no_inf_df,df],axis=1)
desc_fp_df.to_csv('desc_fp_precocessed.csv',index=False)