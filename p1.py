import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

#data_train = hp.load_data('../train.csv')[:,2:]
#y = hp.load_data('../train.csv')[:,1:2]
data,y = hp.load_data_higgs('../train.csv')

N = data.shape[0]

#Replace invalid data by mean over all valid values
#x = hp.imputer(data,-999,'mean')
x_A,x_B,x_C,a_cols,b_cols,c_cols,new_rows = tb.isolate_missing(data,-999)
y_A = y[new_rows] #Rearrange y
x_imp =  tb.knn_impute(x_A,np.concatenate((x_B,x_C),axis=1),K=10,nb_rand_ratio=0.005)

#x_imp_prep, mean_x, std_x = hp.standardize(x_imp)

#w_estim = tb.logit_GD_ridge(y_A,x_A_prep,0.0001,lambda_=10,max_iters=100)
#probs = tb.logit(w_estim,x_A_prep)
#w_estim = tb.logit_GD_ridge(y_A,x_A_prep,0.01,lambda_=0.1,max_iters=100)

#xprep, mean_x, std_x = hp.standardize(x[:,1:])

cov_mat = np.cov(x_imp.T)
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
for ev in eig_vec_cov:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#print('Covariance Matrix:\n', cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

nb_reduced_dim = 30
w_list = list()

for i in range(nb_reduced_dim):
    w_list.append(eig_pairs[i][1])

mat_w = np.asarray(w_list)

x_proj = np.dot(mat_w,x_imp.T).T

#x_aug = np.concatenate((np.ones((xprep.shape[0],1)),xprep[:,np.array([2])]),axis=1)
y_train = y_A
#beta_init = np.zeros((x_aug.shape[1]))

#w_estim = tb.logit_GD_ridge(y_train,x_proj,0.01,lambda_=0.1,max_iters=100)
#w_estim = tb.logit_GD(y_train,x_proj,0.005,max_iters=100)
#w_estim = tb.logit_GD(y_train,x_proj,0.0001,max_iters=100)
w_estim = tb.least_squares_inv_ridge(y_train,x_proj,1)
#print(w_estim)
#probs = tb.comp_p_x_beta_logit(w_estim,x_proj)
#y_tilda = tb.thr_probs(probs,0.5)
#z = w_estim[0] + w_estim[1]*np.linspace(0,np.max(x_aug[:,1]),100)
z = np.dot(x_proj,w_estim)
y_tilda = np.sign(z)
#plt.plot(z,'o'); plt.show()
#plt.plot(y_tilda,'o'); plt.show()

tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)
