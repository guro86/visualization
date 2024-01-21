#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:41:29 2024

@author: robertgc
"""

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import pandas as pd

import matplotlib.animation as animation

#Correlation 
rho = .6

#Covariance matrix, bivarate
cov = np.diag(np.ones(2)) * (1-rho) + np.ones((2,2))*rho

mean = np.zeros(2)

rng = np.random.default_rng(1)


x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
positions = np.vstack([x.ravel(), y.ravel()])

pdf_values = multivariate_normal(mean,cov).pdf(positions.T)
pdf_values = pdf_values.reshape(100, 100)


def make_joint_base(x1=None,x2=None,cross=None,cx1=False,cx2=False):
    
    joint_plot = sns.JointGrid()
    
    joint_plot.ax_joint.set(
        xlabel = 'x1',
        ylabel = 'x2',
        xticks = [],
        yticks = [],
        )
    
    # Create a Seaborn contour plot
    joint_plot.ax_joint.contour(x, y, pdf_values, levels=8, cmap="Blues")
    
    
    return joint_plot

x1 = 1
x2 = 2

n = 10

x1s = np.empty(n)
x2s = np.empty(n)

scale = 1-rho**2

artists = []

fig_id = 1

for i in range(n):
    
    #Make jointplot with cross
    joint_plot = make_joint_base()
    joint_plot.ax_joint.axvline(x1)
    joint_plot.ax_joint.axhline(x2)
    
    joint_plot.ax_joint.plot(x1,x2,'ro')
    joint_plot.ax_joint.plot(x1s[:i],x2s[:i],'ro',mfc='none')
    
    plt.savefig('figs/gibbs_{}.pdf'.format(fig_id))
    fig_id += 1
    

    #Make new joint plot with cross    
    joint_plot = make_joint_base()
    joint_plot.ax_joint.axvline(x1)
    joint_plot.ax_joint.axhline(x2)
    
    #Update X1
    loc = rho*x2
    x1 = rng.normal(loc,scale)
    
    
    xpdf = np.linspace(loc-3*scale, loc+3*scale)
    joint_plot.ax_marg_x.plot(xpdf,norm(loc,scale).pdf(xpdf))

    
    joint_plot.ax_joint.plot(x1,x2,'ro')
    joint_plot.ax_joint.plot(x1s[:i],x2s[:i],'ro',mfc='none')
    
    plt.savefig('figs/gibbs_{}.pdf'.format(fig_id))
    fig_id += 1
    
    #Make new joint plot with cross    
    joint_plot = make_joint_base()
    joint_plot.ax_joint.axvline(x1)
    joint_plot.ax_joint.axhline(x2)
    
    joint_plot.ax_joint.plot(x1,x2,'ro')
    joint_plot.ax_joint.plot(x1s[:i],x2s[:i],'ro',mfc='none')
    
    plt.savefig('figs/gibbs_{}.pdf'.format(fig_id))
    fig_id += 1
    
    #Make new joint plot with cross    
    joint_plot = make_joint_base()
    joint_plot.ax_joint.axvline(x1)
    joint_plot.ax_joint.axhline(x2)

    
    #Update X2
    loc = rho*x1
    x2 = rng.normal(loc,scale)
    
    x1s[i] = x1
    x2s[i] = x2
       
    xpdf = np.linspace(loc-3*scale, loc+3*scale)
    joint_plot.ax_marg_y.plot(norm(loc,scale).pdf(xpdf),xpdf)

    
    joint_plot.ax_joint.plot(x1,x2,'ro')
    joint_plot.ax_joint.plot(x1s[:i],x2s[:i],'ro',mfc='none')
    
    plt.savefig('figs/gibbs_{}.pdf'.format(fig_id))
    fig_id += 1
    