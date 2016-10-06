import numpy as np
import matplotlib.pyplot as plt
import mltb as tb

nb_samples = 100
sig_noise = 3
x = np.linspace(0,1,nb_samples)
x.shape = (1,nb_samples)
x = np.append(np.ones((1,100)),x,axis=0)
w = np.array([3,3])
w.shape = (2,1)
y_true = np.dot(x.transpose(),w)
noise = np.random.normal(0,sig_noise,nb_samples)
noise.shape = (nb_samples,1)
y = y_true + noise
y.shape = (nb_samples,1)

mse = tb.mse_lin(y,x,w)
w_estimate = tb.least_squares_GD(y,x.transpose(),0.5,1000)
print "Gradient descent: True vs. estimate"
print w
print w_estimate[-1]

w_estimate = tb.least_squares_SGD(y,x.transpose(),0.5,1000,B=nb_samples/4)
print "Stochastic gradient descent: True vs. estimate"
print w
print w_estimate[-1]

w_estimate = tb.least_squares_inv(y,x.transpose())
print "Closed-form (normal) equations: True vs. estimate"
print w
print w_estimate
