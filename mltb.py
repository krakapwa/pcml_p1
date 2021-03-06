import numpy as np
import matplotlib.pyplot as plt
import helpers as hp
import sys

def pca(x,nb_dims):

    cov_mat = np.cov(x.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    for ev in eig_vec_cov:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    #print('Covariance Matrix:\n', cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    nb_reduced_dim = 5
    w_list = list()

    for i in range(nb_reduced_dim):
        w_list.append(eig_pairs[i][1])

    mat_w = np.asarray(w_list)

    x_proj = np.dot(mat_w,x.T).T

    return x_proj, eig_pairs

def mse(y_true,y_estim):
    """
    Computes the mean squared error between two outputs
        y_true (Nx1): Output vector (True values)
        y_estim (Nx1): Output vector (Estimated values)
    Where N is the number of samples 
    Out: MSE value
    """

    N = x.shape[0] #Number of samples
    e = y_true - y_estim
    e_squared = e**2
    mse = (1./(2*N))*np.sum(e_squared)

    return mse

def mse_lin(y,x,w):
    """
    Computes the mean squared error of a linear system.
    In: x (DxN): Input matrix
        y (Nx1): Output vector
        w (Dx1): Weight vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: MSE value
    """

    N = x.shape[0] #Number of samples
    e = y - np.dot(x.transpose(),w)
    e_squared = e**2
    mse = (1./(2*N))*np.sum(e_squared)

    return mse

def comp_ls_gradient(N,x,e): return (-1./N)*np.dot(x.transpose(),e)


def least_squares_GD(y,x,gamma,max_iters,init_guess = None):
    """
    Estimate parameters of linear system using least squares gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        init_guess (Dx1): Initial guess
        gamma: step_size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess is None):
        init_guess = np.zeros((x.shape[1],1))

    N = x.shape[0]
    #w = list()
    w = init_guess
    w.shape = (w.shape[0],1)

    y = y.ravel()
    y.shape = (y.shape[0],1)

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w = w - gamma*comp_ls_gradient(N,x,y-np.dot(x,w))

    return w

def least_squares_SGD(y,x,gamma,max_iters,B=1,init_guess = None):
    """
    Estimate parameters of linear system using stochastic least squares gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        init_guess (Dx1): Initial guess
        gamma: step_size
        B: batch size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess is None):
        init_guess = np.zeros((x.shape[1],1))

    N = x.shape[0]
    w = list()
    w = init_guess

    for minibatch_y, minibatch_x in hp.batch_iter(y, x, B, num_batches=max_iters, shuffle=True):
        w = w - gamma*comp_ls_gradient(N,minibatch_x,minibatch_y-np.dot(minibatch_x,w))

    return w

def least_squares_inv(y,x):
    """
    Estimate parameters of linear system using matrix inversion
    In: x (NxD): Input matrix
        y (Nx1): Output vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters

    Ref: https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Derivation_of_the_normal_equations
    """

    #(tx*x)^(-1)*tx
    factor = np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose())
    w = np.dot(factor,y)

    return w

def least_squares_inv_ridge(y,phi_tilda,lambda_):
    """
    Estimate parameters of regularized (ridge/L2-norm) system using matrix inversion
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        lambda_: regularization parameter
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    shape_phi = phi_tilda.shape
    N = shape_phi[1]
    lambda_p = lambda_*2*N
    left_term = np.linalg.inv(np.dot(phi_tilda.transpose(),phi_tilda) + lambda_p*np.identity(shape_phi[1]))
    left_term = np.dot(left_term,phi_tilda.transpose())
    w = np.dot(left_term,y)

    return w

def logit(beta,x):
    """
    Computes the probability values 1/(1+exp(-(beta[0] + beta[1:]*x)))
    Note: This function is safe for high values of tx*beta, values are replaced.
    In: x (NxD+1): Input matrix
        beta (D+1 x 1): Parameter vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Probability values (Nx1)
    """

    tbeta = beta.transpose()
    tx = x.transpose()
    the_exp = np.exp(-np.dot(tbeta,tx))
    ind_inf = np.isinf(the_exp)
    res = 1/(1 + np.exp(-np.dot(tbeta,tx)))
    res[ind_inf] = 0.0

    return np.ravel(res)

def comp_grad_logit(beta,x,y):
    """
    Computes the gradient of the logistic regression cost function
    In: x (NxD+1): Input matrix
        beta (D+1 x 1): Parameter vector
        y (N x 1): Output vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Gradient (Dx1)
    """

    tx = x.transpose()

    probs = logit(beta,x)
    y.shape = (y.shape[0],1)
    probs.shape = (probs.shape[0],1)
    y_minus_p = y - probs
    grad = np.dot(tx,y_minus_p)

    return grad

def logit_GD(y,x,gamma,max_iters,init_guess = None):
    """
    Estimate parameters of logistic regression using gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        init_guess (Dx1): Initial guess
        gamma: step_size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess is None):
        init_guess = np.zeros((x.shape[1],1))
    else:
        init_guess.shape = (init_guess.shape[0],1)

    N = x.shape[0]
    w = list()
    w = init_guess

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w = w - gamma*comp_grad_logit(w,x,y)

    return w

def logit_GD_ridge(y,x,gamma,lambda_,max_iters,init_guess = None):
    """
    Estimate parameters of logistic regression using gradient descent.
    In: x (NxD): Input matrix
        y (Nx1): Output vector
        lambda_: regularization parameter
        init_guess (Dx1): Initial guess
        gamma: step_size
        max_iters: Max number of iterations
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Estimated parameters
    """

    if(init_guess == None):
        init_guess = np.zeros((x.shape[1],1))
    else:
        init_guess.shape = (init_guess.shape[0],1)

    w = init_guess

    N = x.shape[0]

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w = w - gamma*(comp_grad_logit(w,x,y)+ 2*lambda_*w)

    return w

def thr_probs(probs,thr):
    """
    Thresholds the values of probs s.t. probs>=thr gives 1 and probs<thr gives 0
    In: probs (Nx1): Input matrix
        thr: Threshold value
    Out: Classes
    """

    classes = np.zeros(probs.shape)
    ind_sup = np.where(probs >= thr)
    classes[ind_sup] = 1

    return classes



def knn_impute(A,M,K=10,nb_rand_ratio = 0.1):
    """
    Imputes missing values of arrays B and C using k-nearest-neighbors. Arrays are obtained using isolate_missing
    In: A (NxD): Training data (No missing values. Will be used for kNN)
        M (N'xD): Training data with missing values (nan)
        K : Number of nearest neighbors
        nb_rand_ratio: Ratio of clean samples to choose randomly from.
    Out: Matrix of size (N+N'xD) without missing values
    """

    print("Computing k-NN for K=", K, " taking", np.round(nb_rand_ratio*A.shape[0]), " samples")
    for i in range(M.shape[0]):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("%f percent" % (float(i+1)/float(M.shape[0])*100))
        sys.stdout.flush()
        ok_cols = ~np.isnan(M[i,:])
        nan_cols = np.isnan(M[i,:])
        norms = np.array([])
        rand_ind = np.random.randint(0,A.shape[0],np.round(A.shape[0]*nb_rand_ratio))
        for j in range(len(rand_ind)):
            this_norm = np.linalg.norm(A[rand_ind[j],ok_cols]-M[i,ok_cols])
            norms = np.append(norms,this_norm)
        sorted_ind = np.argsort(norms)
        sorted_norms = norms[sorted_ind[0:K]]
        sorted_norms = (sorted_norms/np.sum(sorted_norms)).reshape((K,1))
        this_kNN = A[sorted_ind[0:K],:]
        #Change to weighted mean!!
        mean_kNN = np.mean(this_kNN*sorted_norms,axis=0)
        M[i,nan_cols] = mean_kNN[nan_cols]

    return np.concatenate((A,M),axis=0)


def findOffending(data,offending):
    """
    Looks for values corresponding to offending and outputs matrix of same size as data
    with 1->offending and 0 otherwise
    """
    out = np.zeros(data.shape)
    out[np.where(data == offending)] = 1
    return out

def isolate_missing(x,offend):
    """
    Divide the training data matrix into 3 parts (see below)
    Input:
        x (NxD) input matrix
        offend: offending value to look for
    Output:
        A: All features of all samples are not offending (OK). No features are removed.
        B: Matrix of reduced feature dimension. Has no missing values (but lower "column" dimension)
        C: Most features of most samples are offending.
        a_cols: Column indices of A
        b_cols: Column indices of B
        c_cols: Column indices of C
    """

    offending_rows = np.array([])
    offending_cols = np.array([])
    ok_rows = np.array([])
    ok_cols = np.array([])

    offend_mat = findOffending(x,offend)

    #Replace offending values by NaN
    i_offend, j_offend = np.where(offend_mat)
    x[i_offend,j_offend] = np.nan

    for i in range(x.shape[0]):
        if(np.where(offend_mat[i,:])[0].size > 0): #Found offending values in this row
            offending_rows = np.append(offending_rows,i)
        else:
            ok_rows = np.append(ok_rows,i)

    for i in range(x.shape[1]):
        if(np.where(offend_mat[:,i])[0].size > 0): #Found offending values in this column
            offending_cols = np.append(offending_cols,i)
        else:
            ok_cols = np.append(ok_cols,i)

    a_grid = np.ix_(ok_rows.astype(int),np.arange(0,x.shape[1]))
    b_grid = np.ix_(offending_rows.astype(int),ok_cols.astype(int))
    c_grid = np.ix_(offending_rows.astype(int),offending_cols.astype(int))
    A = x[a_grid]
    B = x[b_grid]
    C = x[c_grid]

    new_rows = np.concatenate((ok_rows.astype(int),offending_rows.astype(int)))
    new_cols = np.concatenate((ok_cols.astype(int),offending_cols.astype(int)))

    return (A,B,C,new_cols,new_rows)

def imputer(x,offend,mode):
    """
    Deal with offending values using following modes:
    'del_row': Deletes rows
    'mean': Replace with mean value of column
    'median': Replace with median value of column
    ."""

    offend_mat = findOffending(x,offend)

    if(mode == 'del_row'):
        ok_rows = np.where(np.sum(offend_mat,axis=1) == 0)
        ok_rows = ok_rows[0]
        clean_x = np.squeeze(x[ok_rows,:])
        return clean_x

    for i in range(x.shape[1]):
        not_ok_rows = np.where(offend_mat[:,i] == 1)
        if(mode == 'mean'):
            this_val = np.mean(x[offend_mat[:,i] == 0,i])
        elif(mode == 'median'):
            this_val = np.median(x[offend_mat[:,i] == 0,i])

        x[not_ok_rows,i] = this_val

    return x

def balance_classes(x,y):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]

    if(pos_idx.shape[0] < neg_idx.shape[0]):
        neg_idx = neg_idx[0:pos_idx.shape[0]]
    else:
        pos_idx = pos_idx[0:neg_idx.shape[0]]

    all_idx = np.concatenate((neg_idx,pos_idx))

    return x[all_idx,:], y[all_idx]
