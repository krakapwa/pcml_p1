import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp
import boosting as bst

#data_train = hp.load_data('../train.csv')[:,2:]
data_train,y_train,id_train = hp.load_data_higgs('../train.csv')
data_test,_,id_test = hp.load_data_higgs('../test.csv')

N = data.shape[0]

data_knn = np.load('../train_knn.npz')
x_knn = data_knn['x_knn']
y_knn = data_knn['y_A']

data_knn_test = np.load('../test_knn.npz')
x_knn_test = data_knn_test['x_knn']
id_knn_test = data_knn_test['id_']

x_mean_imp = tb.imputer(data,-999,'mean')
x_median_imp = tb.imputer(data,-999,'median')

x_A,x_B,x_C,new_cols,new_rows = tb.isolate_missing(data_train,-999)

#Rearrange columns to align with features of training
x_test = data_test[:,new_cols]

x_knn =  tb.knn_impute(x_A,np.concatenate((x_B,x_C),axis=1),K=10,nb_rand_ratio=0.01)

x_A_test,x_B_test,x_C_test,_,new_rows_test = tb.isolate_missing(data_test,-999)
x_knn_test =  tb.knn_impute(x_A_test,np.concatenate((x_B_test,x_C_test),axis=1),K=10,nb_rand_ratio=0.01)
id_knn_test = id_test[new_rows_test]

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

y_A = y_A[0:x_A.shape[0]]


x_miss = np.concatenate((np.zeros((y_good_rows.shape[0],1)),np.ones((y_bad_rows.shape[0],1))),axis=0)
x_knn_aug = np.concatenate((x_miss, x_knn),axis=1)
x_good_cols_aug = np.concatenate((x_miss,x_good_cols),axis=1)

x_pca,eig_pairs = pca(x_A,nb_dims)
    import pdb; pdb.set_trace()

w_estim = tb.least_squares_inv_ridge(y_train,x_proj,1)
w_estim = tb.least_squares_SGD(y_train,x_proj,0.00001,500,B=100,init_guess=w_estim)

z = np.dot(x_proj,w_estim)
y_tilda = np.sign(z)

tpr,fpr = tb.binary_tpr_fpr(y_train,y_tilda)
print("TPR/FPR:", tpr, "/", fpr)

nb_iters = 2000

#PCA
F = bst.train_adaboost(y_A,x_proj,nb_iters)
y_tilda =  bst.predict(F,x_proj)
tpr,fpr = tb.binary_tpr_fpr(y_A,y_tilda)
error_rate = tb.missclass_error_rate(y_A,y_tilda)

print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

#Missing values replaced with mean
F = bst.train_adaboost(y,x_median_imp,nb_iters)
y_tilda =  bst.predict(F,x_median_imp)
tpr,fpr = tb.binary_tpr_fpr(y,y_tilda)
error_rate = tb.missclass_error_rate(y,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)

mod1 = bst.Adaboost(x_knn,y_knn,nb_iters)
K_cv = 10
mod1.set_K(K_cv)

tpr,fpr,miss_rate = mod1.k_fold_cross_validation()

data_out_cv = dict()
data_out_cv['tpr'] = tpr
data_out_cv['fpr'] = fpr
data_out_cv['miss_rate'] = miss_rate
data_out_cv['nb_iters'] = nb_iters
data_out_cv['K'] = K_cv
np.savez('../cv10_adaboost.npz',**data_out_cv)

data_ada_cv10 = np.load('../cv10_adaboost.npz')
tpr = data_ada_cv10['tpr']
fpr = data_ada_cv10['fpr']
miss_rate = data_ada_cv10['miss_rate']
nb_iters = data_ada_cv10['nb_iters']
mean_miss_rate = np.mean(miss_rate,axis=0)
std_miss_rate = np.std(miss_rate,axis=0)

plt.plot(mean_miss_rate);
plt.plot(mean_miss_rate+std_miss_rate,'b--');
plt.plot(mean_miss_rate-std_miss_rate,'b--');
plt.title('K-fold cross-validation.')
plt.xlabel('Num. of iterations')
plt.ylabel('Missclassification rate')
fig = plt.gcf()
fig.savefig('ada_cv10.eps', format='eps', dpi=1200)
plt.show()
plt.close()

F = mod1.train(y_knn,x_knn,nb_iters)
y_tilda =  mod1.predict(F,x_knn)
tpr,fpr = mod1.binary_tpr_fpr(y_knn,y_tilda)
error_rate = mod1.missclass_error_rate(y_knn,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)
y_pred_test = mod1.predict(F,x_knn_test)
hp.write_submission_higgs(y_pred_test,id_knn_test,"../submission7.csv")

#Missing values replaced with K-Nearest-Neighbors Logitboost
F = bst.train_logitboost(y_knn,x_knn,nb_iters)
y_tilda =  bst.predict_logitboost(F,x_knn)
tpr,fpr = tb.binary_tpr_fpr(y_knn,y_tilda)
error_rate = tb.missclass_error_rate(y_knn,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)
y_pred_test = bst.predict_logitboost(F,x_knn_test)
hp.write_submission_higgs(y_pred_test,id_knn_test,"../submission4.csv")

#Missing values replaced with K-Nearest-Neighbors
#x_knn_bal,y_knn_bal = tb.balance_classes(x_knn,y_knn)
#x_knn_bal,_,_ = hp.standardize(x_knn_bal)
#x_knn_pca,_ = tb.pca(x_knn_bal,30)
F = bst.train_adaboost(y_knn,x_knn,nb_iters)
y_tilda =  bst.predict_adaboost(F,x_knn)
tpr,fpr = tb.binary_tpr_fpr(y_knn,y_tilda)
error_rate = tb.missclass_error_rate(y_knn,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)
y_pred_test = bst.predict_adaboost(F,x_knn_test)
hp.write_submission_higgs(y_pred_test,id_knn_test,"../submission6.csv")

y_pred_test = bst.predict(F,x_test)
hp.write_submission_higgs(y_pred_test,id_test,"../submission.csv")

