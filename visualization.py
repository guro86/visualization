#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:51:41 2024

@author: robertgc
"""

import matplotlib.pyplot as plt
import numpy as np
# import emcee
from scipy.stats import norm
from corner import corner
from pymc3.gp.util import plot_gp_dist

#%%

x = np.linspace(.1,2.9,100)
xm = np.linspace(.1,2.9,20)


f = lambda x, a=.8, b=2, c = 0: a*x**2 + b*x + c
f2 = lambda x: f(x) + np.sin(x*10)*2

rng = np.random.default_rng(seed=1)

mu = .5

meas = f2(xm)  + rng.normal(0,mu,len(xm))
 

plt.plot(x,f(x),label='Model')
plt.plot(x,f2(x),'--',label='Truth')
plt.errorbar(
             xm,
             meas,
             yerr = mu,
             fmt = 'o',
             label = 'Measurements'
             )


plt.xticks([])
plt.yticks([])

plt.legend()

plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.savefig('inadecaute_example.pdf')


plt.show()

#%%

# l = np.linspace(-1,13,2)

# plt.plot(l,l)

# # plt.plot(f(xm),f2(xm),'o')
# plt.errorbar(f(xm),f2(xm),fmt='o',yerr=mu)

# plt.xticks([])
# plt.yticks([])

# plt.xlabel('Measured [-]')
# plt.ylabel('Predicted [-]')

# plt.show()

# #%%

# nwalkers = 6
# ndim = 3

# like = norm(loc=meas,scale=mu)

# log_prob_fn = lambda x: like.logpdf(f(xm,*x)).sum()

# initial_state = rng.normal(size=(nwalkers,ndim))
# nsteps = 2000

# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)
# sampler.run_mcmc(initial_state, nsteps,progress = True)

# #%%
# chain = sampler.get_chain(discard=50,flat=True)
# corner(chain,labels=('a','b','c'))
# plt.show()

# #%%
# pred = chain[:,0][:,None] * xm **2 + chain[:,1][:,None] * xm + chain[:,2][:,None]

# #%%



# plt.plot(x,f2(x),'--',label='Truth')
# plt.errorbar(
#              xm,
#              meas,
#              yerr = mu,
#              fmt = 'o',
#              label = 'Measurements'
#              )

# plt.fill_between(
#     xm,
#     pred.mean(axis=0) + 2*pred.std(axis=0),
#     pred.mean(axis=0) - 2*pred.std(axis=0),
#     alpha = .2,
#     label = 'Parameter uncertainty'
#     )

# plt.plot(xm,pred.mean(axis=0),label='Model')

# plt.legend()

# plt.xticks([])
# plt.yticks([])

# plt.xlabel('Inadependent variable')
# plt.ylabel('Dependent variable')


# plt.show()

#%%

l = np.linspace(-1,13,2)

plt.plot(l,l)

plt.plot(f(xm),f2(xm),'o')
# plt.errorbar(f(xm),f2(xm),fmt='o',yerr=(pred.std(axis=0)))

plt.xticks([])
plt.yticks([])

plt.xlabel('Measured [-]')
plt.ylabel('Predicted [-]')

plt.savefig('pm.pdf')

plt.show()

#%%

# cov = np.cov(chain,rowvar=False) * 25

J = np.column_stack((xm**2,xm,np.ones(len(xm))))

cov = np.linalg.inv(J.T @ J) * 6

cov_y = np.diag(J @ cov @ J.T) + mu**2

std_y = cov_y**.5


plt.plot(x,f2(x),'--',label='Truth')
plt.errorbar(
             xm,
             meas,
             yerr = mu,
             fmt = 'o',
             label = 'Measurements'
             )

plt.plot(xm,f(xm),label='Model')


plt.xticks([])
plt.yticks([])


ax = plt.gca()

mean = np.array([.8,2,0])

samples = rng.multivariate_normal(mean, cov,size=500)

samples_y = samples[:,0] * x[:,None]**2 + samples[:,1]*x[:,None] + samples[:,2]

plot_gp_dist(ax, samples_y.T, x,palette='Blues')

plt.legend()

plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')

plt.savefig('inflated.pdf')
plt.show()

#%%

l = np.linspace(-1,13,2)

plt.plot(l,l)

# plt.plot(f(xm),f2(xm),'o')
plt.errorbar(f(xm),f2(xm),fmt='o',yerr=2*std_y)

plt.xticks([])
plt.yticks([])

plt.xlabel('Measured [-]')
plt.ylabel('Predicted [-]')

plt.show()

