import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp
import adaboost as ab
#from sklearn.datasets import make_gaussian_quantiles


#data_train = hp.load_data('../train.csv')[:,2:]
#y = hp.load_data('../train.csv')[:,1:2]
data,y = hp.load_data_higgs('../train.csv')

N = data.shape[0]

#Replace invalid data by mean over all valid values
#x = hp.imputer(data,-999,'mean')
x_A,x_B,x_C,a_cols,b_cols,c_cols,new_rows = tb.isolate_missing(data,-999)

x_good_cols = np.concatenate((x_A[:,0:x_B.shape[1]],x_B),axis=0)

print("Study of conditional probabilities on missing values")
x_bad_rows = np.concatenate((x_B,x_C),axis=1)
y_bad_rows = y[new_rows[x_A.shape[0]:]]
x_good_rows = x_A
y_good_rows = y[new_rows[0:x_A.shape[0]]]

p_miss_1 = y_bad_rows.shape[0]/y.shape[0]
p_miss_0 = y_good_rows.shape[0]/y.shape[0]

p_y1_miss_1 = (np.where(y_bad_rows == 1)[0].shape[0]/y.shape[0])/(p_miss_1)
p_y0_miss_1 = (np.where(y_bad_rows == -1)[0].shape[0]/y.shape[0])/(p_miss_1)
p_y0_miss_0 = (np.where(y_good_rows == -1)[0].shape[0]/y.shape[0])/(p_miss_0)
p_y1_miss_0 = (np.where(y_good_rows == 1)[0].shape[0]/y.shape[0])/(p_miss_0)

print("P(Y=1|X has missing attr) = ", p_y1_miss_1)
print("P(Y=-1|X has missing attr) = ", p_y0_miss_1)
print("P(Y=1|X has no missing attr) = ", p_y1_miss_0)
print("P(Y=-1|X has no missing attr) = ", p_y0_miss_0)

y_knn = y[new_rows] #Rearrange y
y_A = y_A[0:x_A.shape[0]]
x_knn =  tb.knn_impute(x_A,np.concatenate((x_B,x_C),axis=1),K=10,nb_rand_ratio=0.01)

data = np.load('train_kNN.npz')
x_knn = data['x_knn']
y_A = data['y_A']

x_miss = np.concatenate((np.zeros((y_good_rows.shape[0],1)),np.ones((y_bad_rows.shape[0],1))),axis=0)
x_knn_aug = np.concatenate((x_miss, x_knn),axis=1)
x_good_cols_aug = np.concatenate((x_miss,x_good_cols),axis=1)

x_pca,eig_pairs = pca(x_A,nb_dims)

w_estim = tb.least_squares_inv_ridge(y_train,x_proj,1)
w_estim = tb.least_squares_SGD(y_train,x_proj,0.00001,500,B=100,init_guess=w_estim)

z = np.dot(x_proj,w_estim)
y_tilda = np.sign(z)

tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#PCA
nb_iters = 200
F = ab.run(y_A,x_proj,nb_iters)
y_tilda =  ab.predict(F,x_proj)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#Missing values replaced with K-Nearest-Neighbors
nb_iters = 300
F = ab.run(y_knn,x_knn,nb_iters)
y_tilda =  ab.predict(F,x_knn)
tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#Missing values replaced with K-Nearest-Neighbors, x_miss attribute added
nb_iters = 120
F = ab.run(y_knn,x_knn_aug,nb_iters)
f = [F[i]['stump']['feat'] for i in range(len(F))]
y_tilda =  ab.predict(F,x_knn_aug)
tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#Only "good" samples (without missing values)
nb_iters = 200
y_clean = y_A[0:x_A.shape[0]]
F = ab.run(y_clean,x_A,nb_iters)
y_tilda =  ab.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#Only "good" columns (without missing values)
nb_iters = 120
F = ab.run(y_knn,x_good_cols,nb_iters)
y_tilda =  ab.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#Only "good" columns (without missing values), x_miss attribute added
nb_iters = 120
F = ab.run(y_knn,x_good_cols_aug,nb_iters)
f = [F[i]['stump']['feat'] for i in range(len(F))]
y_tilda =  ab.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)
