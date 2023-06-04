#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:39:12 2023

@author: evgeni
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 20})
from scipy.optimize import fsolve

#import

#constants
G=6.674e-8
msun = 2e33
au = 1.5e13
mbh = 1e8*msun
m=10*msun
c=3e10
rs = 2 * G * mbh / c / c
sb=5.6e-5
kb = 1.38e-16
mH = 1.67e-24
kb_over_mu = kb/mH

r=5e3*rs
Omega = (G*mbh/r**3)**0.5
kappa_smbh=0.4
kappa_disc = 0.4
l_E = 1/2
AGN_efficiency = 1
M_dot = 4*np.pi*G*mbh/AGN_efficiency/c/kappa_smbh
alpha=0.01

def equations(vars):
    T_eff, T, tau, Sigma, p_rad, p_gas, rho, h, c_s = vars
    eq1 = T_eff**4 - 3 * M_dot * Omega**2/8/np.pi/sb
    eq2 =  T**4 - T_eff**4 * (3 * tau / 8 + 1/2 + 1 / 4 / tau)
    eq3 = tau - kappa_disc * Sigma / 2
    eq4 = Sigma - M_dot * Omega / 3 / np.pi / alpha / c_s**2
    eq5 = p_rad - tau * sb / 2 / c * T_eff**4
    eq6 = p_gas - rho * kb_over_mu * T / 0.62 
    eq7 = rho - Sigma / 2 / h
    eq8 = h - c_s/Omega
    eq9 = c_s**2 - (p_gas + p_rad) / rho
    return [eq1**0.25, eq2**0.25, eq3, eq4, eq5, eq6, eq7, eq8, eq9**0.5]

T_eff, T, tau, Sigma, p_rad, p_gas, rho, h, c_s =  fsolve(equations, (1e4, 1e5, 1e3, 1e3, 1e5, 1e4, 1e-11, 1e13, 1e7))

print( T_eff/1e3, T/1e3, tau, Sigma, p_rad, p_gas, rho, h/r, c_s/1e5)

#%%
G=6.674e-8
msun = 2e33
au = 1.5e13
vk = (G*msun/au)**0.5
mbh = 1e8*msun
m=10*msun
c=1e4*vk
rs = 2 * G * mbh / c / c
sb=5.6e-5
kb = 1.38e-16
mH = 1.67e-24
kb_over_mu = kb/mH

r=5e3*rs
Omega = (G*mbh/r**3)**0.5
kappa_smbh=0.4
kappa_disc = 0.4
l_E = 1/2
AGN_efficiency = 1
M_dot = 4*np.pi*G*mbh/AGN_efficiency/c/kappa_smbh
alpha=0.01

def disc_eqns(vars):
    T_eff, T, tau, Sigma, p_rad, p_gas, rho, h, c_s = vars
    f = np.zeros(9)
    
    f[0] = T_eff**4 - 3 * M_dot * Omega**2/8/np.pi/sb
    f[1] =  T**4 - T_eff**4 * (3 * tau / 8 + 1/2 + 1 / 4 / tau)
    f[2] = tau - kappa_disc * Sigma / 2
    f[3] = Sigma - M_dot * Omega / 3 / np.pi / alpha / c_s**2
    f[4] = p_rad - tau * sb / 2 / c * T_eff**4
    f[5] = p_gas - rho * kb_over_mu * T / 0.62 
    f[6] = rho - Sigma / 2 / h
    f[7] = h - c_s/Omega
    f[8] = c_s**2 - (p_gas + 0*p_rad) / rho
    return f

sol = fsolve(disc_eqns, [1e4, 1e5, 1e3, 1e3, 1e5, 1e4, 1e-11, 1e13, 1e7])
print(sol)
print(disc_eqns(sol))
    