import numpy as np
import matplotlib.pyplot as plt
import mltb as tb
import helpers as hp

#data_train = hp.load_data('../train.csv')[:,2:]
#y = hp.load_data('../train.csv')[:,1:2]
import pdb; pdb.set_trace()
data,y = hp.load_data_higgs('../train.csv')


def make999cats(data):
    """
    Looks for -999 values and outputs matrix of same size as data
    with 1->-999 and 0 otherwise
    """
    out = np.zeros(data.shape)
    out[np.where(data == -999)] = 1
    return out

N = data.shape[0]

#Many features take -999 values: make categories
cats = make999cats(data)

#x = data[np.where(np.sum(cats,axis=1) == 0),:]

plt.hist(data[cats[:,1] == 0,1],100); plt.show() #Gaussian
plt.hist(data[cats[:,2] == 0,2],100); plt.show() #Mixture of 2 gaussians
plt.hist(data[cats[:,3] == 0,3],100); plt.show()#Gaussian
plt.hist(data[cats[:,4] == 0,4],100); plt.show() #Exponential distribution?, skewed pos gauss
plt.hist(data[cats[:,5] == 0,5],100); plt.show()#Mixture of 2 gaussians
plt.hist(data[cats[:,6] == 0,6],100); plt.show()#skewed positive gaussian
plt.hist(data[cats[:,7] == 0,7],100); plt.show() #?
plt.hist(data[cats[:,8] == 0,8],100); plt.show() #?
plt.hist(data[cats[:,9] == 0,9],100); plt.show() #?
plt.hist(data[cats[:,10] == 0,10],100); plt.show() #Exponential distribution?, skewed pos gauss
plt.hist(data[cats[:,11] == 0,11],100); plt.show() #skewed pos gauss
plt.hist(data[cats[:,12] == 0,12],100); plt.show() #?
plt.hist(data[cats[:,13] == 0,13],100); plt.show() #?
plt.hist(data[cats[:,14] == 0,14],100); plt.show() #?
plt.hist(data[cats[:,15] == 0,15],100); plt.show() #?
plt.hist(data[cats[:,16] == 0,16],100); plt.show() #Get offset?
plt.hist(data[cats[:,17] == 0,17],100); plt.show() #?
plt.hist(data[cats[:,18] == 0,18],100); plt.show() #?
plt.hist(data[cats[:,19] == 0,19],100); plt.show() #?
plt.hist(data[cats[:,20] == 0,20],100); plt.show() #?
plt.hist(data[cats[:,21] == 0,21],100); plt.show() #?
plt.hist(data[cats[:,22] == 0,22],100); plt.show() #?
plt.hist(data[cats[:,23] == 0,23],100); plt.show() #categorical
plt.hist(data[cats[:,24] == 0,24],100); plt.show() #exponential dist
plt.hist(data[cats[:,25] == 0,25],100); plt.show() #gaussian
plt.hist(data[cats[:,26] == 0,26],100); plt.show() #Get offset?
plt.hist(data[cats[:,27] == 0,27],100); plt.show() #exponential dist
plt.hist(data[cats[:,28] == 0,28],100); plt.show() #Gaussian
plt.hist(data[cats[:,29] == 0,29],100); plt.show() #Get offset?
plt.hist(data[cats[:,30] == 0,30],100); plt.show() #?


import pdb; pdb.set_trace()
xprep, mean_x, std_x = hp.standardize(x)
