import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

#data_train = hp.load_data('../train.csv')[:,2:]
#y = hp.load_data('../train.csv')[:,1:2]
data,y = hp.load_data_higgs('../train.csv')

N = data.shape[0]

#Replace invalid data by mean over all valid values
x = hp.imputer(data,-999,'mean')

#x = data[np.where(np.sum(cats,axis=1) == 0),:]

#plt.hist(x[:,1],100); plt.show() #Gaussian
#plt.hist(x[:,2],100); plt.show() #Mixture of 2 gaussians
#plt.hist(x[:,3],100); plt.show()#Gaussian
#plt.hist(x[:,4],100); plt.show() #Exponential distribution?, skewed pos gauss (out)
#plt.hist(x[:,5],100); plt.show()#Mixture of 2 gaussians (out)
#plt.hist(x[:,6],100); plt.show()#skewed positive gaussian (out)
#plt.hist(x[:,7],100); plt.show() #?
#plt.hist(x[:,8],100); plt.show() #?
#plt.hist(x[:,9],100); plt.show() #?
#plt.hist(x[:,10],100); plt.show() #Exponential distribution?, skewed pos gauss
#plt.hist(x[:,11],100); plt.show() #skewed pos gauss
#plt.hist(x[:,12],100); plt.show() #?
#plt.hist(x[:,13],100); plt.show() #?
#plt.hist(x[:,14],100); plt.show() #?
#plt.hist(x[:,15],100); plt.show() #?
#plt.hist(x[:,16],100); plt.show() #Get offset?
#plt.hist(x[:,17],100); plt.show() #?
#plt.hist(x[:,18],100); plt.show() #?
#plt.hist(x[:,19],100); plt.show() #?
#plt.hist(x[:,20],100); plt.show() #?
#plt.hist(x[:,21],100); plt.show() #?
#plt.hist(x[:,22],100); plt.show() #?
#plt.hist(x[:,23],100); plt.show() #categorical
#plt.hist(x[:,24],100); plt.show() #exponential dist
#plt.hist(x[:,25],100); plt.show() #gaussian
#plt.hist(x[:,26],100); plt.show() #Get offset?
#plt.hist(x[:,27],100); plt.show() #exponential dist
#plt.hist(x[:,28],100); plt.show() #Gaussian
#plt.hist(x[:,29],100); plt.show() #Get offset?
#plt.hist(x[:,30],100); plt.show() #?

xprep, mean_x, std_x = hp.standardize(x)

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

    return 1/denominator

def comp_grad_logit(beta,x,y):
    """
    Computes the gradient of the logistic regression cost function
    In: x (NxD+1): Input matrix
        beta (D+1 x 1): Parameter vector
        y (N x 1): Output vector
    Where N and D are respectively the number of samples and dimension of input vectors
    Out: Gradient (Dx1)
    """

    import pdb; pdb.set_trace()
    tx = x.transpose()

    probs = comp_p_x_beta_logit(beta,x)
    y_minus_p = y - probs
    grad = np.dot(tx,probs.transpose())

    return grad

beta = np.ones((x.shape[1]+1,1))
x_aug = np.concatenate((np.ones((x.shape[0],1)),x),axis=1) #Augmented version (for offset)

#test = comp_p_x_beta_logit(beta,x_aug)
grad = comp_grad_logit(beta,x_aug,y)
