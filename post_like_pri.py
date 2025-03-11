#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 00:19:28 2025

@author: robertgc
"""

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

mu_like  = 1
sig_like = 1

mu_prior = -1
sig_prior = 2

mu_posterior = (mu_prior / sig_prior **2 + mu_like / sig_like**2) * (
    1/(1/sig_like**2 + 1/sig_prior**2)
    )

sig_posterior =  1/(1/sig_like**2 + 1/sig_prior**2)

def f(mu,sig,n=200,f=4):
    
   x = np.linspace(mu - f*sig,mu + f*sig,n)
   y = norm(loc=mu,scale=sig).pdf(x)
   
   return x,y


x_like, y_like = f(mu_like, sig_like)
x_posterior, y_posterior = f(mu_posterior, sig_posterior)
x_prior, y_prior = f(mu_prior, sig_prior)

plt.plot(x_like,y_like,label='Likelihood')
plt.plot(x_posterior,y_posterior,label='Posterior')
plt.plot(x_prior,y_prior,label='Prior')

plt.legend()

ax = plt.gca()

ax.set_axis_off()

plt.savefig('pri_post_like.pdf')