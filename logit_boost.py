import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

def make_stump_logit(z,x,w,sorted_idx):
    """
    Makes decision stumps (depth-1) using weights w, splitting samples in nb_steps ranges
    y [Nx1]: Ground-Truth
    z [Nx1]: Working-response
    x [NxD]: Training samples
    M: Number of stages
    nb_steps: Number of ranges for data splitting
    """

    N = x.shape[0]
    D = x.shape[1]

    min_error = np.inf
    stump = {}

    for i in range(D): #Loop over features

        z_sorted = z[sorted_idx[i]]
        w_sorted = w[sorted_idx[i]]
        x_sorted = x[sorted_idx[i],i]

        #Weighted error is w*(z-f)
        #lower or equal as positive class, the rest as negative. (f=1)
        cumsum_pos_below_thr = np.cumsum((w_sorted*(z_sorted - 1)  )**2)

        #greater as positive class, the rest as negative. (f=-1)
        cumsum_neg_above_thr = np.cumsum((w_sorted[::-1]*(z_sorted[::-1] + 1)  )**2)

        #greater as positive class, the rest as negative. (f=-1)
        cumsum_pos_above_thr = np.cumsum((w_sorted[::-1]*(z_sorted[::-1] - 1)  )**2)

        #greater as positive class, the rest as negative. (f=-1)
        cumsum_neg_below_thr = np.cumsum((w_sorted*(z_sorted + 1)  )**2)

        diff_x = np.diff(x_sorted)
        diff_x = np.insert(diff_x,0,1.)
        forbid_split = diff_x == 0

        leq_error = cumsum_pos_below_thr + cumsum_neg_above_thr[::-1]
        gt_error = cumsum_neg_below_thr + cumsum_pos_above_thr[::-1]

        leq_error[forbid_split] = np.inf
        gt_error[forbid_split] = np.inf

        ind_min_leq = np.argmin(leq_error)
        ind_min_gt = np.argmin(gt_error)

        if(leq_error[ind_min_leq] <= gt_error[ind_min_gt]):
            this_thr = x_sorted[sorted_idx[i][ind_min_leq]]
            this_mode = 'leq'
            this_LS = leq_error[ind_min_leq]
        elif(leq_error[ind_min_leq] > gt_error[ind_min_gt]):
            this_thr = x_sorted[sorted_idx[i][ind_min_gt]]
            this_mode = 'gt'
            this_LS = gt_error[ind_min_gt]
        else:
            this_thr = x_sorted[sorted_idx[i][ind_min_leq]]
            this_mode = 'leq'
            this_LS = leq_error[ind_min_leq]
            #this_thr = x_sorted[sorted_idx[i][ind_min_gt]]
            #this_mode = 'gt'
            #this_LS = gt_error[ind_min_gt]


        if (this_LS < min_error):
            min_error = this_LS
            stump['feat'] = i
            stump['thr'] = this_thr
            stump['mode'] = this_mode

    return stump,min_error

def train_logitboost(y,x,M):

    """
    Computes M stages of adaboost using decision stumps
    y [Nx1]: Ground-truth
    x [NxD]: Training samples
    M: Number of stages
    """

    y = np.ravel(y)
    y.shape = (y.shape[0],1)

    N = y.shape[0]
    D = x.shape[1]

    w = list()
    F = list()
    init_classifier = {}
    this_weights = np.ones((x.shape[0],1))
    this_weights = this_weights/np.sum(this_weights)
    this_p = np.ones((x.shape[0],1))
    this_p = this_p/2

    sorted_idx = list()

    for i in range(D):
        sorted_idx.append(np.argsort(x[:,i]))

    F.append([])
    F[0] = dict()
    overall_best = np.zeros((x.shape[0],1))
    z_bound = 4
    r = (y+1)/2
    r0_idx = np.where(r == 0)[0]
    r1_idx = np.where(r == 1)[0]
    import pdb; pdb.set_trace()
    for i in range(M):
        this_z = np.zeros((N,1))
        this_z[r0_idx] = 1/this_p[r0_idx]
        this_z[r1_idx] = -1/(1-this_p[r1_idx])
        this_z = np.clip(this_z,-z_bound,z_bound,out=None)
        #this_z = ((y+1)/2 - this_p)/(this_p*(1-this_p))

        stump,min_error = make_stump_logit(this_z,x,this_weights,sorted_idx)
        F[i]['stump'] = stump
        F[i]['min_error'] = min_error

        best_pred = thr_classify(x,stump['feat'],stump['thr'],stump['mode'])

        #overall_best_prev = overall_best
        overall_best += 0.5*best_pred

        this_p = np.exp(overall_best)/(np.exp(overall_best)+np.exp(-overall_best))

        this_weights = this_p*(1-this_p)
        this_weights = this_weights/np.sum(this_weights)
        #this_weights = np.clip(this_weights, np.finfo(float).eps, np.inf, out=None)

        missclassified = np.ones((N,1))
        missclassified[y==np.sign(overall_best)]= 0

        print("Iter. ", i, "missclass_rate:", np.sum(missclassified)/N, "stump:", stump)

        if(i==M-1): break

        F.append([])
        F[i+1] = dict()

    return F


def predict_logitboost(F,x,default=-1.):

    pred = np.zeros((x.shape[0],1))

    for i in range(len(F)):
        this_stump = F[i]['stump']
        this_feat = this_stump['feat']
        this_thr = this_stump['thr']
        this_mode = this_stump['mode']
        this_pred = thr_classify(x,this_feat,this_thr,mode=this_mode)
        pred += this_pred

    pred[np.where(pred == 0)] = default

    return np.sign(pred)
