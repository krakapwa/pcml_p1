import numpy as np
import helpers as hp

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

    if(init_guess == None):
        init_guess = np.zeros((x.shape[1],1))

    N = x.shape[0]
    w = list()
    w.append(init_guess)

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w.append(w[-1] - gamma*comp_ls_gradient(N,x,y-np.dot(x,w[-1])))

    return w[-1]

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

    if(init_guess == None):
        init_guess = np.zeros((x.shape[1],1))


    N = x.shape[0]
    w = list()
    w.append(init_guess)

    for minibatch_y, minibatch_x in hp.batch_iter(y, x, B, num_batches=max_iters, shuffle=True):
        w.append(w[-1] - gamma*comp_ls_gradient(N,minibatch_x,minibatch_y-np.dot(minibatch_x,w[-1])))

    return w[-1]

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

    return

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

def comp_p_x_beta_logit(beta,x):
    """
    Computes the probability values 1/(1+exp(-(beta[0] + beta[1:]*x)))
    In: x (NxD+1): Input matrix
        beta (D+1 x 1): Parameter vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Probability values (Nx1)
    """

    tbeta = beta.transpose()
    tx = x.transpose()
    denominator = 1 + np.exp(-np.dot(tbeta,tx))

    return np.ravel(1/denominator)

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

    probs = comp_p_x_beta_logit(beta,x)
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

    if(init_guess == None):
        init_guess = np.zeros((x.shape[1],1))
    else:
        init_guess.shape = (init_guess.shape[0],1)

    N = x.shape[0]
    w = list()
    w.append(init_guess)

    nb_iter = 0
    while(nb_iter<max_iters):
        nb_iter+=1
        w.append(w[-1] - gamma*comp_grad_logit(w[-1],x,y))

    return w[-1]

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
