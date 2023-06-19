#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:02:43 2023

@author: evgeni
"""

import numpy as np
import matplotlib.pyplot as plt

def analytic_ratio(x):
    s = np.sign(x)
    return pow(np.e, - 2 * x * s)

xs = np.random.uniform(0,1,10000)**0.5 
ys = np.random.uniform(0,1,10000)**0.5

plt.hist(xs, density=True, bins=30, edgecolor='black', alpha=0.5, facecolor='white', lw=5) 
plt.hist(ys, density=True, bins=30, edgecolor='blue', alpha=0.5, facecolor='white', lw=5) 

zs = [x/y for (x,y) in zip(xs, ys)]

plt.figure(2)
plt.hist(np.log(zs),  density=True,bins=50,  edgecolor='black', alpha=0.5, facecolor='white', lw=3)

x = np.logspace(-1.5,1.5,1000)
plt.plot(np.log(x), [analytic_ratio(y) for y in np.log(x)], lw=3, color='red')
plt.xlabel(r'$w$')
plt.ylabel(r'$f_W(w)$')