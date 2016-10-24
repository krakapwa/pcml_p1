import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

from abc import ABC,abstractmethod

class BinaryClassifier(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def make_stump(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def set_K(self,K):
        self.K = K

    def thr_classify(self,x,dim,thr_val,mode='leq'):
        """
        Thresholding function of column dim of array x
        x [NxD]: Training samples
        dim: Column of x
        thr_val: Value of threshold
        mode = {'leq',gt}
        """
        out = -np.ones((x.shape[0],1))
        nan_ind = np.where(np.isnan(x[:,dim]))[0]
        if(mode == 'leq'):
            out[x[:,dim] <= thr_val] = 1.0
        elif(mode == 'gt'):
            out[x[:,dim] > thr_val] = 1.0

        out[nan_ind] = 0

        return out

    def missclass_error_rate(self,y_true,y_pred):
        """
        Computes the missclassification error rate. y_true must be either +1 or -1
        In: y_true (Nx1): Training values
            y_pred (Nx1): Predicted values
        Out: Missclassification error rate
        """

        missclass = np.ones((y_true.shape[0],1))
        missclass[np.where(np.ravel(y_true) == np.ravel(y_pred))[0]] = 0

        miss_rate = np.sum(missclass)/y_true.shape[0]

        return miss_rate

    def binary_tpr_fpr(self,y_true,y_pred):
        """
        Computes the true/false positive rates. y_true must be either +1 or -1
        In: y_true (Nx1): Training values
            y_pred (Nx1): Predicted values
        Out: True/False positives rates
        """

        y_true.shape = (y_true.shape[0],1)
        y_pred.shape = (y_pred.shape[0],1)

        positives = np.where(y_true == 1)
        negatives = np.where(y_true == -1)
        true_pos = np.where(y_pred[positives[0]] == 1)[0].shape[0]
        false_pos = np.where(y_pred[negatives[0]] == 1)[0].shape[0]
        true_neg = np.where(y_pred[negatives[0]] == -1)[0].shape[0]
        false_neg = np.where(y_pred[positives[0]] == -1)[0].shape[0]

        tpr = float(true_pos)/(float(true_pos) + float(false_neg))
        fpr = float(false_pos)/(float(false_pos) + float(true_neg))

        return tpr,fpr

    @abstractmethod
    def k_fold_cross_validation(self):
        pass

class Adaboost(BinaryClassifier):

    def __init__(self,x,y,nb_iters):
        self.x = x
        self.y = y
        self.nb_iters = nb_iters

    def k_fold_cross_validation(self):

        F = list()
        pred = list()
        tpr = np.array([])
        fpr = np.array([])
        miss_rate = np.array([])

        idx = np.arange(0,self.x.shape[0],dtype=int) #Indices to shuffle

        np.random.shuffle(idx)
        folds_idx = np.split(idx, self.K)
        test_folds_idx = np.zeros((folds_idx[0].shape[0],self.K),dtype=int)
        train_folds_idx = list()

        for i in range(self.K):
            test_folds_idx[:,i] = folds_idx[i].astype(int)
            train_folds_idx.append([])
            train_folds_idx[i] = np.array([],dtype=int)
            for j in range(self.K):
                if(j!=i):
                    train_folds_idx[i] = np.append(train_folds_idx[i],folds_idx[j].astype(int))

        for i in range(len(folds_idx)):
            print("Starting ", self.K, "-fold cross-validation")
            print("K = ", i)
            this_F = self.train(self.y[train_folds_idx[i]],self.x[train_folds_idx[i],:],self.nb_iters,do_print=False)
            F.append(this_F)

        fpr = np.zeros((self.K-1,self.nb_iters))
        tpr = np.zeros((self.K-1,self.nb_iters))
        miss_rate = np.zeros((self.K-1,self.nb_iters))

        import pdb; pdb.set_trace()
        for i in range(self.K):
            for j in range(self.nb_iters):
                pred.append(self.predict(F[i][0:j+1],self.x[test_folds_idx[i],:]))
                this_tpr,this_fpr = self.binary_tpr_fpr(self.y[test_folds_idx[i]],pred[-1])
                this_miss_rate = self.missclass_error_rate(self.y[test_folds_idx[i]],pred[-1])
                fpr[i,j] = this_fpr
                tpr[i,j] = this_tpr
                miss_rate[i,j] = this_miss_rate

        return tpr,fpr,miss_rate

    def make_stump(self,y,x,w,sorted_idx):
        """
        Makes decision stumps (depth-1) using weights w, splitting samples in nb_steps ranges
        y [Nx1]: Ground-truth
        x [NxD]: Training samples
        w [Nx1]: Weights
        sorted_idx list([Nx1]): Sorted indices of columns of x
        """

        N = x.shape[0]
        D = x.shape[1]

        min_error = np.inf
        stump = {}

        for i in range(D): #Loop over features
            this_col = x[:,i]
            #sorted_ind = np.argsort(this_col)
            y_sorted = (y[sorted_idx[i]] + 1)/2

            #lower or equal as positive class, the rest as negative. We only multiply weights of missclassified, hence negative class in this case
            cumsum_leq = np.cumsum(w[sorted_idx[i]]*np.invert(y_sorted.astype(bool)))

            #greater than as positive class, the rest as negative. We only multiply weights of missclassified, hence positive class in this case
            cumsum_gt = np.cumsum(w[sorted_idx[i][::-1]]*y_sorted[::-1].astype(bool))

            leq_error = cumsum_leq + cumsum_gt[::-1]
            gt_error = 1 - leq_error

            #We want to forbid the selection of a thresholds that correspond to repeating values.
            diff_x = np.diff(this_col[sorted_idx[i]])
            diff_x = np.insert(diff_x,0,1.)
            forbid_split = diff_x == 0

            leq_error[forbid_split] = np.inf
            gt_error[forbid_split] = np.inf

            ind_min_leq = np.argmin(leq_error)
            ind_min_gt = np.argmin(gt_error)

            if(leq_error[ind_min_leq] <= gt_error[ind_min_gt]):
                this_thr = this_col[sorted_idx[i][ind_min_leq]]
                this_mode = 'leq'
            elif(leq_error[ind_min_leq] > gt_error[ind_min_gt]):
                this_thr = this_col[sorted_idx[i][ind_min_gt]]
                this_mode = 'gt'
            else:
                this_thr = this_col[sorted_idx[i][ind_min_gt]]
                this_mode = 'gt'

            pred_vals = self.thr_classify(x,i,this_thr,mode=this_mode)
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

    def train(self,y,x,M,lambda_=None,nu=1,do_print=True):
        """
        Computes M stages of adaboost using decision stumps
        y [Nx1]: Ground-truth
        x [NxD]: Training samples
        M: Number of stages
        nb_steps: Number of ranges for data splitting
        lambda_: For non-homogeneous weight initialization.
            Default: w_i = 1/N, for all i in {0,...,N-1}
            Non-default: w_i = 1/(2*N*lambda_) for positive class
                            w_i = 1/(2*N*(1-lambda_)) for positive class
        nu: Learning rate

        """

        y = np.ravel(y)
        y.shape = (y.shape[0],1)
        N = y.shape[0]

        sorted_idx = list()
        for i in range(x.shape[1]):
            sorted_idx.append(np.argsort(x[:,i]))

        w = list()
        F = list()
        init_classifier = {}
        this_weights = np.ones((x.shape[0],1))
        if lambda_ is not None:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == -1)[0]
            this_weights[pos_idx] = 1/(2*N*(lambda_))
            this_weights[neg_idx] = 1/(2*N*(1-lambda_))

        this_weights = this_weights/np.sum(this_weights)

        #init_classifier['weights'] = init_weights
        F.append(init_classifier)
        overall_best = np.zeros((x.shape[0],1))
        for i in range(M):

            stump,min_error,best_pred = self.make_stump(y,x,this_weights,sorted_idx)
            F[i]['stump'] = stump
            F[i]['min_error'] = min_error
            F[i]['best_pred'] = best_pred
            alpha = 0.5*np.log((1-min_error)/min_error)

            loss = np.exp(-y*alpha*best_pred)

            loss.shape = (loss.shape[0],1)
            F[i]['alpha'] = alpha

            overall_best += F[i]['alpha']*best_pred
            missclassified = np.ones((x.shape[0],1))
            missclassified[np.where(np.ravel(np.sign(overall_best)) == np.ravel(y))[0]] = 0
            if(do_print):
                print("Iter. ", i, "missclass_rate:", np.sum(missclassified)/x.shape[0], "stump:", stump)

            if(i==M-1): break

            updated_w = this_weights*loss
            updated_w = updated_w/np.sum(updated_w)
            F.append([])
            F[i+1] = dict()
            this_weights = updated_w

        return F

    def predict(self,F,x):

        pred = np.zeros((x.shape[0],1))

        for i in range(len(F)):
            this_stump = F[i]['stump']
            this_alpha = F[i]['alpha']
            this_feat = this_stump['feat']
            this_thr = this_stump['thr']
            this_mode = this_stump['mode']
            this_pred = this_alpha*self.thr_classify(x,this_feat,this_thr,mode=this_mode)
            pred += this_pred

        return np.sign(pred)

