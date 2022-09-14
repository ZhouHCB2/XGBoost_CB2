# coding: utf-8

import sys
sys.path.append('/home/phzd/AI/bidd-molmap_v1_3')
from molmap import model as molmodel
import molmap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import load, dump
tqdm.pandas(ascii=True)
import tensorflow as tf
import os,glob
np.random.seed(123)
tf.compat.v1.set_random_seed(123)
from argparse import ArgumentParser
from sklearn.utils import shuffle 

base=os.getcwd()

parser = ArgumentParser(description='class has pos_weights compared with reg')
parser.add_argument('-f',dest='csv_file',type=str,help='data_path')
parser.add_argument('-ext',type=str,help='external')
parser.add_argument('-metric',dest='metric',type=str,
	default='ROC',choices=['ROC','PRC'],help='default ROC')
parser.add_argument('-seeds',nargs='*',default=[2**n for n in range(10)],help="??")
parser.add_argument('-epochs',type=int,default=800,help='default 800')
parser.add_argument('-gpu',dest='gpu_id',type=str,default='0',help='default 0')
parser.add_argument('-use_performance',action='store_true',default=False,
	help='default 0 i.e. not use_performance(faster), 1(slower)')


args = parser.parse_args()
csv_file = args.csv_file
ext = args.ext
metric = args.metric
print('note: here not using reverse_V_test')
epochs = args.epochs
#epochs = 800
x_gpuid =args.gpu_id #'0'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]=x_gpuid
use_performance = args.use_performance

random_seeds=[int(x) for x in args.seeds]

#for fn in args.fold_num: ## only can run single task
df = pd.read_csv(csv_file)
csv_base = csv_file.split('.')[0]

mp1 = molmap.loadmap('/home/phzd/AI/bidd-molmap_v1_3/paper/descriptor.mp')
mp2 = molmap.loadmap('/home/phzd/AI/bidd-molmap_v1_3/paper/fingerprint.mp')
dic_DF={'D':'X2_','F':'X1_'}

smiles_col = df.columns[0]
#values_col = df.columns[1:]
values_col = df.columns[1:]
# Y = df[values_col].astype('float').values  
# run class has problems, need fill MASK, see below
MASK = -1
Y = df[values_col].astype('float').fillna(MASK).values

#Y = df[values_col].astype('float')
if Y.shape[1] == 0:
	Y  = Y.reshape(-1, 1)

tmp_feature_dir = './tempignore/'

if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)

X1_name = os.path.join(tmp_feature_dir, 'X1_.data')
X2_name = os.path.join(tmp_feature_dir, 'X2_.data')
if not os.path.exists(X1_name):
	#X1 = mp1.batch_transform(df.smiles, n_jobs = 40)
	X1 = mp1.batch_trans_map_async(df.smiles, n_jobs = 40)	
	dump(X1, X1_name)
else:
	X1 = load(X1_name)
if not os.path.exists(X2_name): 
	#X2 = mp2.batch_transform(df.smiles, n_jobs = 40)
	X2 = mp2.batch_trans_map_async(df.smiles, n_jobs = 40)
	dump(X2, X2_name)
else:
	X2 = load(X2_name)
molmap1_size = X1.shape[1:]
molmap2_size = X2.shape[1:]
print('X1.shape, X2.shape = ', X1.shape, X2.shape)

# read external dataset
edf = pd.read_csv(ext)
ext_base = ext.split('.')[0]
smiles_col = edf.columns[0]
values_col = edf.columns[1:]
Y_ext = edf[values_col].astype('float').values
if Y_ext.shape[1] == 0:
	Y_ext  = Y_ext.reshape(-1, 1)


# # external feat
X1_name_ext = os.path.join(tmp_feature_dir, 'X1_ext.data')
X2_name_ext = os.path.join(tmp_feature_dir, 'X2_ext.data')
if not os.path.exists(X1_name_ext):
	#X1 = mp1.batch_transform(df.smiles, n_jobs = 40)
	X1_ext = mp1.batch_trans_map_async(edf.smiles, n_jobs = 40)	
	dump(X1_ext, X1_name_ext)
else:
	X1_ext = load(X1_name_ext)
if not os.path.exists(X2_name_ext): 
	#X2 = mp2.batch_transform(df.smiles, n_jobs = 40)
	X2_ext = mp2.batch_trans_map_async(edf.smiles, n_jobs = 40)
	dump(X2_ext, X2_name_ext)
else:
	X2_ext = load(X2_name_ext)
molmap1_size_ext = X1_ext.shape[1:]
molmap2_size_ext = X2_ext.shape[1:]
print('X1_ext.shape, X2_ext.shape = ', X1.shape, X2.shape)


epochs = 210
patience = 50 #early stopping
dense_layers = [256] #12 outputs  ## this diff from reg
batch_size = 128
lr = 1e-4
weight_decay = 0
# loss: will be defined in the function  ## this diff from reg
monitor = 'val_auc'
dense_avf = 'relu'
last_avf = None #sigmoid in loss
#metric = 'ROC'  # choice of ROC or RPC, already definded in argparse

best_epoch = 350

#stypes = ['both_path','descriptor_path', 'fingerprint_path' ]  # , 
stypes = ['both_path']
#idx = list(range(len(df)))
#
def random_split(df, random_state, split_size = [0.8, 0.1, 0.1]):	
	base_indices = np.arange(len(df)) 
	base_indices = shuffle(base_indices, random_state = random_state) 
	nb_test = int(len(base_indices) * split_size[2]) 
	nb_val = int(len(base_indices) * split_size[1]) 
	test_idx = base_indices[0:nb_test] 
	valid_idx = base_indices[(nb_test):(nb_test+nb_val)] 
	train_idx = base_indices[(nb_test+nb_val):len(base_indices)] 
	print(len(train_idx), len(valid_idx), len(test_idx)) 
	return train_idx, valid_idx, test_idx 

def get_pos_weights(trainY):
	"""pos_weights: neg_n / pos_n """
	dfY = pd.DataFrame(trainY)
	pos = dfY == 1
	pos_n = pos.sum(axis=0)
	neg = dfY == 0
	neg_n = neg.sum(axis=0)
	pos_weights = (neg_n / pos_n).values
	neg_weights = (pos_n / neg_n).values
	return pos_weights, neg_weights

results = []
for stype in stypes:
	i = 0
	#for fn in args.fold_nums:
	reverse_V_test = 'no'
	# for dlb_i,fn in enumerate(fold_nums*double_cv) :
	#	 print('fn = ',fn)
	#	 ffn = f'f{fn}'
	#	 pklf= f'fold_{fn}/split_indices.pckl'
	#	 train_idx, valid_idx, test_idx = pd.read_pickle(pklf)
	# # for train_idx, valid_idx in kf.split(idx):
	#	 trainY=Y[train_idx];validY=Y[valid_idx];testY=Y[test_idx]

	for dlb_i,seed in enumerate(random_seeds):		
		print('seed = ',seed)
		train_idx, valid_idx, test_idx = random_split(df,random_state=seed)
		train_idx = [i for i in train_idx if i < len(df)]
		valid_idx = [i for i in valid_idx if i < len(df)]	
		test_idx = [i for i in test_idx if i < len(df)]	
		print(len(train_idx), len(valid_idx), len(test_idx))	 
		trainY=Y[train_idx];validY=Y[valid_idx];testY=Y[test_idx]

		pos_weights, neg_weights = get_pos_weights(trainY)
		loss = lambda y_true, y_pred: molmodel.loss.weighted_cross_entropy(y_true,
			y_pred,pos_weights, MASK = -1)

		if stype == 'descriptor_path':
			trainX=X1[train_idx];validX=X1[valid_idx];testX=X1[test_idx]
			extX = X1_ext
			model = molmodel.net.SinglePathNet(molmap1_size,  
					n_outputs=Y.shape[-1],dense_layers=[128, 32], 
					dense_avf = dense_avf,last_avf=last_avf)
		
		if stype == 'fingerprint_path':
			trainX=X2[train_idx];validX=X2[valid_idx];testX=X2[test_idx]
			extX = X2_ext
			model = molmodel.net.SinglePathNet(molmap2_size,  
					n_outputs=Y.shape[-1],dense_layers=[128, 32], 
					dense_avf = dense_avf,last_avf=last_avf)		

		if stype == 'both_path':
			trainX = (X1[train_idx], X2[train_idx])
			validX = (X1[valid_idx], X2[valid_idx])
			testX  = (X1[test_idx], X2[test_idx])
			extX = (X1_ext, X2_ext)
			model = molmodel.net.DoublePathNet(molmap1_size, molmap2_size, 
					n_outputs=Y.shape[-1],dense_layers=[256, 128, 32], 
					dense_avf = dense_avf,last_avf=last_avf)

		# import tensorflow_addons as tfa
		# loss = lambda y_true, y_pred: molmodel.loss.weighted_cross_entropy(y_true,y_pred, pos_weights, MASK = -1)
		# opt = tfa.optimizers.AdamW(weight_decay = 0.0,learning_rate=lr)
		# model.compile(optimizer=opt, loss = loss)

		opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
		#import tensorflow_addons as tfa
		#opt = tfa.optimizers.AdamW(weight_decay = 0.1,learning_rate=0.001,beta1=0.9,beta2=0.999, epsilon=1e-08)
		model.compile(optimizer = opt, loss = loss)
		performance = molmodel.cbks.CLA_EarlyStoppingAndPerformance((trainX, trainY), 
			(validX, validY),patience = patience,criteria = monitor,metric = metric)
		if use_performance and (i==0):
			model.fit(trainX, trainY, batch_size=batch_size, 
				  epochs=epochs, verbose= 0, shuffle = True, 
				  validation_data = (validX, validY), 
				  callbacks=[performance]) 
			best_epoch = performance.best_epoch
		else:
			model.fit(trainX, trainY, batch_size=batch_size, 
				  epochs=2, verbose= 0, shuffle = True, 
				  validation_data = (validX, validY), 
				  callbacks=[performance])			
			model.fit(trainX, trainY, batch_size=batch_size, 
				  epochs = best_epoch,verbose = 1, shuffle = True, 
				  validation_data = (validX, validY)) 			 
			performance.model.set_weights(model.get_weights())
	   
		#best_epoch = performance.best_epoch
		trainable_params = model.count_params()
		
		train_aucs = performance.evaluate(trainX, trainY)			
		valid_aucs = performance.evaluate(validX, validY)			
		test_aucs = performance.evaluate(testX, testY)
		ext_aucs = performance.evaluate(extX, Y_ext)

		final_res = {
						'stype':stype,
					 	'seed':seed,	
						 'task_name':'_'.join(values_col),			
						 'train_auc':np.nanmean(train_aucs), 
						 'valid_auc':np.nanmean(valid_aucs),					  
						 'test_auc':np.nanmean(test_aucs), 
						 'ext_aucs':np.nanmean(ext_aucs),
						 'metric':metric,
						 '# trainable params': trainable_params,
						 'best_epoch': best_epoch,
						 'batch_size':batch_size,
						 'lr': lr,
						 'weight_decay':weight_decay
						}		
		results.append(final_res)
		i += 1
		pd.DataFrame([final_res]).to_csv(f'./mp_class_rand_append.csv',mode='a')
pd.DataFrame(results).to_csv('./mp_class_rand.csv',mode='a')
pd.DataFrame(results).groupby(['stype'])['test_auc'].apply(lambda x:x.mean()).to_csv('mp_class_rand_grouped.csv',mode='a')




