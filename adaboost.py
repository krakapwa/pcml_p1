import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

def thr_classify(x,dim,thr_val,mode='leq'):
    out = -np.ones((x.shape[0],1))
    nan_ind = np.where(np.isnan(x[:,dim]))[0]
    if(mode == 'leq'):
        out[x[:,dim] <= thr_val] = 1.0
    elif(mode == 'gt'):
        out[x[:,dim] > thr_val] = 1.0

    out[nan_ind] = 0

    return out

def make_stump(y,x,w):
    """
    Makes decision stumps (depth-1) using weights w, splitting samples in nb_steps ranges
    y [Nx1]: Ground-truth
    x [NxD]: Training samples
    M: Number of stages
    nb_steps: Number of ranges for data splitting
    """

    N = x.shape[0]
    D = x.shape[1]

    min_error = np.inf
    stump = {}

    for i in range(D): #Loop over features
        this_col = x[:,i]
        sorted_ind = np.argsort(this_col)
        y_sorted = (y[sorted_ind] + 1)/2

        #lower or equal as positive class, the rest as negative
        cumsum_leq = np.cumsum(w[sorted_ind]*np.invert(y_sorted.astype(bool)))

        #greater than as positive class, the rest as negative
        cumsum_gt = np.cumsum(w[sorted_ind[::-1]]*y_sorted[::-1].astype(bool))

        leq_error = cumsum_leq + cumsum_gt[::-1]
        gt_error = 1 - leq_error

        ind_min_leq = np.argmin(leq_error)
        ind_min_gt = np.argmin(gt_error)

        if(leq_error[ind_min_leq] <= gt_error[ind_min_gt]):
            this_thr = this_col[sorted_ind[ind_min_leq]]
            this_mode = 'leq'
        elif(leq_error[ind_min_leq] > gt_error[ind_min_gt]):
            this_thr = this_col[sorted_ind[ind_min_gt]]
            this_mode = 'gt'
        else:
            this_thr = this_col[sorted_ind[ind_min_gt]]
            this_mode = 'gt'

        pred_vals = thr_classify(x,i,this_thr,mode=this_mode)
        missclassified = np.ones((N,1))
        missclassified[np.where(np.ravel(pred_vals) == np.ravel(y))[0]] = 0
        e = np.dot(w.T,missclassified)

        if (e < min_error):
            min_error = e
            best_pred = pred_vals
            stump['feat'] = i
            stump['thr'] = this_thr
            stump['mode'] = this_mode

    return stump,min_error,best_pred

def run(y,x,M):
    """
    Computes M stages of adaboost using decision stumps
    y [Nx1]: Ground-truth
    x [NxD]: Training samples
    M: Number of stages
    nb_steps: Number of ranges for data splitting
    """

    y = np.ravel(y)
    y.shape = (y.shape[0],1)

    w = list()
    F = list()
    init_classifier = {}
    init_weights = np.ones((x.shape[0],1))
    init_weights = init_weights/np.sum(init_weights)
    init_classifier['weights'] = init_weights
    F.append(init_classifier)
    overall_best_pred = np.zeros((x.shape[0],1))
    for i in range(M):

        stump,min_error,best_pred = make_stump(y,x,F[i]['weights'])
        F[i]['stump'] = stump
        F[i]['min_error'] = min_error
        F[i]['best_pred'] = best_pred
        alpha = 0.5*np.log((1-min_error)/min_error)

        loss = np.exp(-y*alpha*best_pred)

        loss.shape = (loss.shape[0],1)
        F[i]['alpha'] = alpha

        overall_best_pred += F[i]['alpha']*best_pred
        missclassified = np.ones((x.shape[0],1))
        missclassified[np.where(np.ravel(np.sign(overall_best_pred)) == np.ravel(y))[0]] = 0
        print("Iter. ", i, "missclass_rate:", np.sum(missclassified)/x.shape[0], "stump:", stump)

        if(i==M-1): break

        updated_w = F[i]['weights']*loss
        updated_w = updated_w/np.sum(updated_w)
        F.append([])
        F[i+1] = dict()
        F[i+1]['weights'] = updated_w

    return F

def predict(F,x):

    pred = np.zeros((x.shape[0],1))
    w = np.ones((x.shape[0],1))/x.shape[0]
    for i in range(len(F)):
<<<<<<< HEAD
        #pred += F[i]['alpha']*F[i]['best_pred']
        pred += F[i]['best_pred']

    pred = np.sign(pred)
=======
        this_stump = F[i]['stump']
        this_alpha = F[i]['alpha']
        this_feat = this_stump['feat']
        this_thr = this_stump['thr']
        this_mode = this_stump['mode']
        this_pred = this_alpha*thr_classify(x,this_feat,this_thr,mode=this_mode)
        pred += this_pred
>>>>>>> 450214fc088bd191c8fe72400bb2f65f2942240e

    return np.sign(pred)
