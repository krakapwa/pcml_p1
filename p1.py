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
y_A = y[new_rows] #Rearrange y
y_A = y_A[0:x_A.shape[0]]
x_knn =  tb.knn_impute(x_A,np.concatenate((x_B,x_C),axis=1),K=10,nb_rand_ratio=0.01)

#x_imp_prep, mean_x, std_x = hp.standardize(x_imp)

#w_estim = tb.logit_GD_ridge(y_A,x_A_prep,0.0001,lambda_=10,max_iters=100)
#probs = tb.logit(w_estim,x_A_prep)
#w_estim = tb.logit_GD_ridge(y_A,x_A_prep,0.01,lambda_=0.1,max_iters=100)

#xprep, mean_x, std_x = hp.standardize(x[:,1:])
data = np.load('train_kNN.npz')
x_knn = data['x_knn']
y_A = data['y_A']


x_pca,eig_pairs = pca(x_A,nb_dims)

w_estim = tb.least_squares_inv_ridge(y_train,x_proj,1)
w_estim = tb.least_squares_SGD(y_train,x_proj,0.00001,500,B=100,init_guess=w_estim)

z = np.dot(x_proj,w_estim)
y_tilda = np.sign(z)

tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

#PCA
nb_iters = 120
F = ab.run(y_A,x_proj,nb_iters)
y_tilda =  ab.predict(F,x_proj)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

nb_iters = 120
F = ab.run(y_train,x_proj,nb_iters)
y_tilda =  ab.predict(F,x_proj)
tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

nb_iters = 120
F = ab.run(y_A,x_A,nb_iters)
y_tilda =  ab.predict(F,x_A)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

F = ab.run(y,X,200)
