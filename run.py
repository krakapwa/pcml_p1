import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp
import boosting as bst

#Load original csv files
data_train,y_train,id_train = hp.load_data_higgs('../train.csv')
data_test,_,id_test = hp.load_data_higgs('../test.csv')

N = data.shape[0]

#data_knn = np.load('../train_knn.npz')
#x_knn = data_knn['x_knn']
#y_knn = data_knn['y_A']
#
#data_knn_test = np.load('../test_knn.npz')
#x_knn_test = data_knn_test['x_knn']
#id_knn_test = data_knn_test['id_']

#Isolate columns with missing values to the right (changes order of features)
x_A,x_B,x_C,new_cols,new_rows = tb.isolate_missing(data_train,-999)
x_test = data_test[:,new_cols]

#Impute values of training data with k-nearest-neighbors
x_knn =  tb.knn_impute(x_A,np.concatenate((x_B,x_C),axis=1),K=10,nb_rand_ratio=0.01)

#Impute values of test data with k-nearest-neighbors
x_A_test,x_B_test,x_C_test,_,new_rows_test = tb.isolate_missing(data_test,-999)
x_knn_test =  tb.knn_impute(x_A_test,np.concatenate((x_B_test,x_C_test),axis=1),K=10,nb_rand_ratio=0.01)
id_knn_test = id_test[new_rows_test]

#Build model and perform k-fold cross-validation
nb_iters = 2000
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

#Train model and make submission file
F = mod1.train(y_knn,x_knn,nb_iters)
y_tilda =  mod1.predict(F,x_knn)
tpr,fpr = mod1.binary_tpr_fpr(y_knn,y_tilda)
error_rate = mod1.missclass_error_rate(y_knn,y_tilda)
print("TPR/FPR/error_rate:", tpr, "/", fpr, "/", error_rate)
y_pred_test = mod1.predict(F,x_knn_test)
hp.write_submission_higgs(y_pred_test,id_knn_test,"../submission8.csv")
