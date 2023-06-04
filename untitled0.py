#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:22:55 2023

@author: evgeni
"""

import numpy as np

h = 3.2e14
omega = 2.03e-3/6**1.5
rho=4.3e9*1.67e-24
eps_rad = 2.1e7
sb = 5.6e-5
c=3e10
a_rad=4*sb/c
T= (eps_rad/a_rad)**0.25
cs= (eps_rad/3/rho)**0.5
gamma=5/3
kappa=0.3
chi = 16*gamma*(gamma-1)*sb*T**4 / 3/kappa/rho**2/cs**2#h**2/omega**2
chi2 = 4*gamma*(gamma-1) * c /kappa/rho

l_ratio = gamma*c/kappa/rho/chi
l_ratio2 = gamma*c/kappa/rho/chi2
chi_norm2= chi / h/h/omega

print(l_ratio, l_ratio2)
