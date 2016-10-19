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

def make_stump_random(y,x,w,ratio):
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
        thresholds_ind = np.random.choice(np.arange(0,this_col.shape[0]),size=np.round(ratio*this_col.shape[0]),replace=False)
        for j in range(thresholds_ind.shape[0]):
            for mode in ['leq','gt']:
                this_thr = this_col[thresholds_ind[j]]

                pred_vals = np.zeros((this_col.shape[0],1))
                pred_vals[0:j+1] = -1
                pred_vals[j+1:] = 1
                pred_vals = pred_vals[sorted_ind]
                if(mode == 'gt'): pred_vals = -pred_vals

                missclassified = np.ones((N,1))
                missclassified[np.where(np.ravel(pred_vals) == np.ravel(y))[0]] = 0
                e = np.dot(w.T,missclassified)/np.sum(w)
                #e = 0.5 - 0.5*np.dot(w.T,missclassified)
                if (e < min_error):
                    min_error = e
                    best_pred = pred_vals
                    stump['feat'] = i
                    stump['thr'] = this_thr
                    stump['mode'] = mode

    return stump,min_error,best_pred

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
                e = np.dot(w.T,missclassified)/np.sum(w)
                #e = 0.5 - 0.5*np.dot(w.T,missclassified)
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
    F.append(init_classifier)
    import pdb; pdb.set_trace()
    overall_best_pred = np.zeros((x.shape[0],1))
    for i in range(M):
        #x = x*F[i]['weights']
        stump,min_error,best_pred = make_stump(y,x,F[i]['weights'],nb_steps)
        F[i]['stump'] = stump
        F[i]['min_error'] = min_error
        F[i]['best_pred'] = best_pred
        alpha = 0.5*np.log((1-min_error)/min_error)
        #alpha = 0.5 - 0.5*np.log((1-min_error)/min_error)
        loss = np.exp(-y*alpha*best_pred)
        loss.shape = (loss.shape[0],1)
        F[i]['alpha'] = alpha

        overall_best_pred += F[i]['alpha']*best_pred
        missclassified = np.ones((x.shape[0],1))
        missclassified[np.where(np.ravel(np.sign(overall_best_pred)) == np.ravel(y))[0]] = 0
        print("AdaBoost iter. ", i, "current_missclass_rate:", np.sum(missclassified)/x.shape[0], "stump:", stump)

        if(i==M-1): break

        updated_w = F[i]['weights']*loss
        updated_w = updated_w/np.sum(updated_w)
        F.append([])
        F[i+1] = dict()
        F[i+1]['weights'] = updated_w
        #F[i+1]['alpha'] = alpha

    return F

def predict(F,x):

    pred = np.zeros((x.shape[0],1))
    w = np.ones((x.shape[0],1))/x.shape[0]
    for i in range(len(F)):
        this_stump = F[i]['stump']
        this_weights = F[i]['weights']
        this_alpha = F[i]['alpha']
        this_feat = this_stump['feat']
        this_thr = this_stump['thr']
        this_mode = this_stump['mode']
        #this_pred = np.sign(this_alpha*thr_classify(x,this_feat,this_thr,mode=this_mode))
        this_pred = this_alpha*thr_classify(x,this_feat,this_thr,mode=this_mode)
        #this_pred = F[i]['best_pred']
        pred += this_pred

    #pred = np.sign(pred)

    return np.sign(pred)
