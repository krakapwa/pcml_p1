import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

def thr_classify(x,dim,thr_val,mode='lt'):
    out = np.ones((x.shape[0],1))
    if(mode == 'leq'):
        out[x[:,dim] <= thr_val] = -1.0
    else:
        out[x[:,dim] > thr_val] = -1.0

    return out

def make_stump(y,x,w,nb_steps):
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
        this_feat_min = np.min(x[:,i])
        this_feat_max = np.max(x[:,i])
        step_size = (this_feat_max - this_feat_min)/nb_steps #Select nb_steps threshold values within range
        for j in np.arange(0,nb_steps+1):
            for mode in ['leq','gt']:
                this_thr = (this_feat_min + float(j)*step_size)
                pred_vals = thr_classify(x,i,this_thr,mode=mode)
                missclassified = np.ones((N,1))
                missclassified[np.where(np.ravel(pred_vals) == np.ravel(y))[0]] = 0
                e = np.dot(w.T,missclassified)
                if (e < min_error):
                    min_error = e
                    best_pred = pred_vals
                    stump['feat'] = i
                    stump['thr'] = this_thr
                    stump['mode'] = mode

    return stump,min_error,best_pred

def run(y,x,M,nb_steps):
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
    init_classifier['weights'] = np.ones((x.shape[0],1))/x.shape[0]
    init_classifier['alpha'] = 1
    F.append(init_classifier)
    for i in range(M):
        print("AdaBoost iter. ", str(i))
        x = x*F[i]['weights']
        stump,min_error,best_pred = make_stump(y,x,F[i]['weights'],nb_steps)
        F[i]['stump'] = stump
        F[i]['min_error'] = min_error
        F[i]['best_pred'] = best_pred
        if(i==M-1): break
        alpha = 0.5*np.log((1-min_error)/min_error)
        loss = np.exp(-y*alpha*best_pred)
        loss.shape = (loss.shape[0],1)

        updated_w = F[i]['weights']*loss
        updated_w = updated_w/np.sum(updated_w)
        F.append([])
        F[i+1] = dict()
        F[i+1]['weights'] = updated_w
        F[i+1]['alpha'] = alpha

    return F

def predict(F,y):

    y = np.ravel(y)
    y.shape = (y.shape[0], 1)

    pred = np.zeros((y.shape[0],1))
    for i in range(len(F)):
        #pred += F[i]['alpha']*F[i]['best_pred']
        pred += F[i]['best_pred']

    pred = np.sign(pred)

    e = np.ones((y.shape[0],1))
    e[np.where(pred == y)[0]] = 0
    error_rate = np.sum(e)/y.shape[0]

    return pred,error_rate
