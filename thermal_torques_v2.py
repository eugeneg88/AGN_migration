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
#import

#constants
G=6.674e-8
msun = 2e33
M8= 1
au = 1.5e13
mbh = 1e8*M8*msun
m=10*msun
c=3e10
sb=5.6e-5
rs = 2 * G * mbh / c / c
#%%
#derived quantities at the break opint r=1000 rs

#radial space
r0 = 1e3 * rs

rho_0 = 1.22e-9 * M8**-2
h_0=8e-3
Sigma_0 = 6.5e5 * M8 **-1
c_s_0 = 5.3e6 
kappa_0=0.34
tau_0 = kappa_0 * Sigma_0 / 2
Q_0 = 1
T_eff_0 = 3.08e3
T_0 = 4.39e4
## model parameters
n = 3 # pressure gradient
#r3s = np.linspace(0.01, 1000, 10000) # in units of r3
r3s = np.logspace(-2,3,10000)
def kappa(x):
    k=0
    if x <= 20:
        k = kappa_0
    elif 20<= x <= 40:
        k = kappa_0 * (x/20)**-10
    elif  x >= 40:
        k=kappa_0 * 2**-10
    return k

#disc quantities in terms of power laws
def dens(x):
    if x<=1:
        return rho_0 * x**0.5
    elif x>1: 
        return rho_0 * x**-3

def h(x):
    if x<=1:
        return h_0 * x**(-2/3)
    else: 
        return h_0 * x**(1/2)

def Sigma(x):
    if x<=1:
        return Sigma_0 * x**(5/6)
    else: 
        return Sigma_0 * x**(-3/2)

def c_s(x):
    if x<=1:
        return c_s_0 * x**(-7/6)
    else: 
        return c_s_0
    
def tau(x):
    if x<=1:
        return tau_0 * x**(5/6)
    else: 
        return tau_0 * x**(-3/2) * kappa(x)/kappa_0

def Q(x):
    if x<=1:
        return Q_0 * x**(-7/2)
    else: 
        return Q_0
    
    
def T_eff(x):
    if x<=1:
        return T_eff_0 * x**(-3/4)
    else: 
#        return T_eff_0 * x**(-3/8) * (0.375*tau(x)/(0.375*tau(x) + 1/2 + 1/4/tau(x)))**0.25
        return T_eff_0 * x**(-3/8) #* (1/(0.375*tau(x) + 1/2 + 1/4/tau(x)))**0.25    
def T(x):
    if x<=1:
        return T_0 * x**(-13/24)
    elif 1<x<=40: 
        return T_0 * x**(-3/4) #* (0.375*tau(x)/(0.375*tau(x) + 1/2 + 1/4/tau(x)))**-0.25
    else:
        return T_0 * 40**-0.75
def T_eff2(x):
    return  T(x) / (3 *tau(x)/8 + 0.5 + 0.25/tau(x))**0.25   
        
plt.subplot(321)
plt.plot(np.log10(r3s), [np.log10(dens(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.ylabel(r'$\log \rho\ [\rm g\ cm^{-3}]$')
plt.subplot(323)
plt.plot(np.log10(r3s), [np.log10(h(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.ylabel(r'$\log h=H/r$')
plt.subplot(322)
plt.plot(np.log10(r3s), [np.log10(Sigma(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.ylabel(r'$\log \Sigma\ [\rm g\ cm^{-2}]$')
plt.subplot(324)
plt.plot(np.log10(r3s), [np.log10(c_s(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.ylabel(r'$\log c_s\ [\rm cm\ s^{-1}]$')
plt.subplot(325)
#plt.plot(r3s, [Q(x) for x in r3s]); plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(tau(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.ylabel(r'$\log \tau$')
plt.xlabel(r'$\log r_3$')
plt.subplot(326)
plt.plot(np.log10(r3s), [np.log10(T_eff2(x)) for x in r3s],  label=r'$T_{\rm eff}$', linewidth=3, color=
         'red'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(T(x)) for x in r3s],label=r'$T$', linewidth=3, color='blue', ); #plt.xscale('log'); plt.yscale('log')
plt.ylabel(r'$\log T\ [\rm K]$')
plt.xlabel(r'$\log r_3$')
plt.legend(fontsize=18)

#%%
#derived quantities
m_bh = 10
q=m_bh/1e8/M8
#%%
# assume M_SMBH = r3 = G = 1-8
Omega_0 = 2.26e-8
def Omega(x):
    return Omega_0*x**-1.5

#length scales
x_c_0 = 1.39e12
def x_c(x):
    return 1.39e12 * (c_s(x)/c_s_0)**2/(Omega(x)/Omega_0)**2/x

def H(x):
    return h(x) * x

def chi2(x):
    return  1e-21*T(x)**4/dens(x)**2 / H(x)**2 / Omega(x)**2/(9*r0*h_0)

chi_0 = 3.7e19

def lambda_c(x):
    return (chi_0/2/Omega_0)**0.5 * (chi2(x)/chi_0)**0.5 * (Omega(x)/Omega_0)**-0.5

def chi(x):
    if x<=1:
        return chi_0 * x **(-5/6)
    elif x>1:
        return chi_0 * x **3

# mass in msun
def mass_c(x):
    return chi2(x) * c_s(x) / G / msun
    
#plt.plot(np.log10(r3s), [np.log10(chi2(x)) for x in r3s], linewidth=3, color='red'); #plt.xscale('log'); plt.yscale('log')    
#plt.plot(np.log10(r3s), [np.log10(chi(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(tau(x)) for x in r3s], linewidth=3, label=r'$\tau$',color='grey'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(x_c(x)/r0/x) for x in r3s], linewidth=3, label = r'x$_c$', color='red'); #plt.xscale('log'); plt.yscale('log')    
plt.plot(np.log10(r3s), [np.log10(lambda_c(x)/r0/x) for x in r3s], linewidth=3,label = r'$\lambda_c$', color='blue'); #plt.xscale('log'); plt.yscale('log')    
plt.plot(np.log10(r3s), [np.log10(h(x)) for x in r3s], linewidth=3, label = 'H', color='green'); #plt.xscale('log'); plt.yscale('log')    
#plt.plot(np.log10(r3s), [np.log10(mass_c(x)) for x in r3s], linewidth=3, color='red'); #plt.xscale('log'); plt.yscale('log')    
plt.xlabel(r'$\log r/r_3$') #plt.ylabel(r'$\Gamma / \Gamma_0$')
plt.legend()
#%%
#luminosities
def L_edd(x):
    m=10
    return 4*np.pi*m*c/kappa(x) * 6.674e-8*2e33
    
def L_c(x):
    m=10
    return 4*np.pi*m*dens(x)*chi2(x)/1.3333 * 6.674e-8*2e33
#plt.plot(np.log10(r3s), [np.log10(L_c(x)) for x in r3s], linewidth=3, color='red'); #plt.xscale('log'); plt.yscale('log')    
#plt.plot(np.log10(r3s), [np.log10(L_edd(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(L_edd(x)/L_c(x)-1) for x in r3s], linewidth=3, color='green'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(-L_edd(x)/L_c(x)+1) for x in r3s], linewidth=3, color='green', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')

#%%
#torques
gamma=5/3
# auxillary damping function
def g(tau):
    return (tau/2)**10 / (1+(tau/2)**10)
#sample = np.logspace(-2,3,1000)
#plt.plot(sample, [g(x) for x in sample]); plt.xscale('log'); plt.yscale('log')

def Gamma_0(x):
    return q**2 * Sigma(x) * (r0 * x)**4 * Omega(x)**2 * h(x)**-3

def Gamma_type_I(x):
    if x>1:
        C=-3
    elif x<=1:
        C=0.9
    return C * x_c(x)/ H(x)/r0 * Gamma_0(x) / Gamma_0(x) 

def Gamma_tot_thermal(x,m):
    return 1.61 *(gamma-1) / gamma * x_c(x) / lambda_c(x)  * Gamma_0(x) / Gamma_0(x) \
    * (L_edd(x) / L_c(x) * (4*mass_c(x)/(10 + 4 * mass_c(x)) ) - ( 2* mass_c(x)/(10 + 2 * mass_c(x)) )) * g(tau(x))

#plt.plot(np.log10(r3s), [np.log10(Gamma_0(x)) for x in r3s], linewidth=3, color='green'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(Gamma_type_I(x)) for x in r3s], linewidth=3, color='green', label='type I'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(-Gamma_type_I(x)) for x in r3s], linewidth=3, color='green', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(1*Gamma_tot_thermal(x, m)) for x in r3s], linewidth=3, color='red', label='thermal'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(-Gamma_tot_thermal(x,m)) for x in r3s], linewidth=3, color='red', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(Gamma_tot_thermal(x,m) + Gamma_type_I(x)) for x in r3s], linewidth=3, color='blue', label='total'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(-Gamma_tot_thermal(x,m) - Gamma_type_I(x)) for x in r3s], linewidth=3, color='blue', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.legend()
plt.xlabel(r'$r/r_3$'); plt.ylabel(r'$\Gamma / \Gamma_0$')
plt.ylim([-4,1])
#%%
# in terms of optical depth
taus = [tau(x) for x in r3s]
plt.plot(np.log10(taus), [np.log10(Gamma_type_I(x)) for x in r3s], linewidth=3, color='green', label='type I'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(taus), [np.log10(-Gamma_type_I(x)) for x in r3s], linewidth=3, color='green', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(taus), [np.log10(1*Gamma_tot_thermal(x)) for x in r3s], linewidth=3, color='red', label='thermal'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(taus), [np.log10(-Gamma_tot_thermal(x)) for x in r3s], linewidth=3, color='red', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(taus), [np.log10(Gamma_tot_thermal(x) + Gamma_type_I(x)) for x in r3s], linewidth=3, color='blue', label='total'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(taus), [np.log10(-Gamma_tot_thermal(x) - Gamma_type_I(x)) for x in r3s], linewidth=3, color='blue', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.legend()
plt.xlabel(r'$\tau$'); plt.ylabel(r'$\Gamma / \Gamma_0$')
plt.ylim([-10,-1])


#%%
def HH(r):
    hh=0
    if r>=r0:
        hh = 8e-3 * r0 * (r / r0)**1.5
    elif r < r0:
        hh = 8e-3 * r * (r/r0)**-0.6
    return hh

def Omega_f(r):
    return (G * mbh / r**3)**0.5

cs = 1e7
gamma = 4/3
A=1.12
alpha=0.01
epsilon = 1
sb=5.6e-5

def get_chi(gamma, alpha, r):
    return 9 * gamma * (gamma - 1) /2/ np.pi * A(gamma)**2 * alpha * HH(r)**2 * Omega_f(r)

def get_chi_q1(gamma, r):
    return 4/3.14159*gamma*(gamma -1) / tau(r,gamma) * 3e10/cs * HH(r)**2 * Omega_f(r) * min(1, (r/r0)**0.9)

def kappa(r):
    k=0
    if r <= 2e4 * rs:
        k = 0.4
    elif 2e4 * rs <= r <= 4e4 * rs:
        k = 0.4 * (r/2e4/rs)**-10
    elif  r >= 4e4 * rs:
        k=0.00039
    return k

def dens(r):
    rr = 0 
    if r>=r0:
        rr = 1.22e-9  * (r/ r0)**(-3)
    elif r<=r0:
        rr = 1.22e-9 * (r/ r0)**0.4
    return rr
        
#rho= [dens(r) for r in radii]#
#kappas = [kappa(r/rs) for r in radii]

def A(gamma):
    if gamma==4/3:
        a=1.12
    elif gamma==5/3:
        a=1.25
    return a
        
def L_edd(r):
    return 4 * np.pi * G * m * c / kappa(r)
def L_c(r,gamma,alpha):
    return 4 * np.pi * G * m * dens(r) * get_chi(gamma, alpha, r)/gamma

def L_c_q1(r,gamma):
    return 4 * np.pi * G * m * dens(r) * get_chi_q1(gamma, r)/gamma
# luminosities
#L_edd_list = [4 * np.pi * G * m * c / kappa for kappa in kappas]
#L_c_list = [4 * np.pi * G * m * rr * cc/gamma for (cc,rr) in zip(chi,rho)]

def Sigma(r, gamma):
    return dens(r) * HH(r) * 2 * 1.12

def tau(r,gamma):
    return kappa(r) * Sigma(r,gamma)/2

#%%
epsilon=1
gamma=4/3
plt.xlim([30,0.8e6])
plt.rcParams["figure.figsize"] = (12,24)
plt.subplot(221)
plt.plot(radii/rs, np.log10([get_chi(gamma, 0.01, x)/HH(x)**2/Omega_f(x) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10}\  (\chi \ / \ H^2\Omega )$')
plt.plot(radii/rs, np.log10([get_chi_q1(gamma, x)/HH(x)**2/Omega_f(x) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log'); #plt.ylabel(r'$\log_{10} \tau$')
plt.xlabel(r'$r/r_s$')
plt.xlim([20,0.9e6])
plt.subplot(222)
plt.plot(radii/rs, np.log10([epsilon*L_edd(x) for x in radii]), '--', linewidth=3, label=r'$L_{\rm Edd}$');  plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10} (L \rm / erg\ s^{-1})$')
plt.plot(radii/rs, np.log10([0.1*L_edd(x) for x in radii]), '--', linewidth=3, label=r'$0.5L_{\rm Edd}$');  plt.yscale('linear'); plt.xscale('log'); 
plt.plot(radii/rs, np.log10([L_c(x,gamma,alpha) for x in radii]), linewidth=3, label=r'$L_c$');  plt.xscale('log');
plt.plot(radii/rs, np.log10([L_c_q1(x,gamma) for x in radii]), linewidth=3, label=r'$L_{c,Q}$');  plt.xscale('log');
plt.ylabel(r'$\log_{10} (L \rm / erg\ s^{-1})$')
plt.xlabel(r'$r/r_s$')
plt.legend(fontsize=16)
plt.xlim([20,0.9e6])
plt.subplot(223)
plt.plot(([tau(x, 4/3) for x in radii]), np.log10([get_chi(gamma, 0.01, x)/HH(x)**2/Omega_f(x) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log'); plt.ylabel(r'$\log_{10} \tau$')
plt.plot(([tau(x, 4/3) for x in radii]), np.log10([get_chi_q1(gamma, x)/HH(x)**2/Omega_f(x) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log'); plt.ylabel(r'$\log_{10} \tau$')
plt.ylabel(r'$\log_{10}\  (\chi \ / \ H^2\Omega )$')
plt.xlabel(r'$\tau$')
plt.xlim([5,1.2e5])
plt.ylim([-2.3,4.1])

plt.subplot(224)
plt.plot(([tau(x, 4/3) for x in radii]), np.log10([epsilon*L_edd(x) for x in radii]), '--', linewidth=3, label=r'$L_{\rm Edd}$');  plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10} (L \rm / erg\ s^{-1})$')
plt.plot(([tau(x, 4/3) for x in radii]), np.log10([1e-3*L_edd(x) for x in radii]), '--', linewidth=3, label=r'$10^{-3}L_{\rm Edd}$');  plt.yscale('linear'); plt.xscale('log'); 
plt.plot(([tau(x, 4/3) for x in radii]), np.log10([L_c(x,gamma,alpha) for x in radii]), linewidth=3, label=r'$L_c$');  plt.xscale('log');
plt.plot(([tau(x, 4/3) for x in radii]), np.log10([L_c_q1(x,gamma) for x in radii]), linewidth=3, label=r'$L_{c,Q}$');  plt.xscale('log');
#plt.ylabel(r'$\log_{10} (\kappa / \rm cm^2\ g^{-1})$')
plt.xlabel(r'$\tau$')
plt.xlim([5,1.2e5])

#%%
#def TT(gamma,alpha,r):
#TT = [(27/64*alpha*tt*(1 + 8/3/tt +1/4/tt/tt/tt)/sb*ss*O**3*h**2)**0.25  for (tt,ss,O,h) in zip(tau,Sigma,Omega,H)]
epsilon=1
plt.xlim([30,0.8e6])
plt.rcParams["figure.figsize"] = (12,24)
plt.subplot(321)
plt.plot(radii/rs, np.log10([tau(x, 4/3) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log'); plt.ylabel(r'$\log_{10} \tau$')
plt.xlim([20,0.9e6])
plt.subplot(324)
plt.plot(radii/rs, np.log10([kappa(x) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10} (\kappa / \rm cm^2\ g^{-1})$')
plt.xlabel(r'$r/r_s$')
plt.xlim([20,0.9e6])
plt.subplot(322)
plt.plot(radii/rs, np.log10([dens(x) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10} (\rho / \rm g\ cm^{-3})$')
plt.xlim([20,0.9e6])
plt.subplot(325)
plt.xlim([20,0.9e6])
plt.plot(radii/rs, np.log10([Sigma(x,gamma) for x in radii]), linewidth=3); plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10} (\Sigma/ \rm g\ cm^{-2})$')
plt.xlabel(r'$r/r_s$')
plt.xlim([20,0.9e6])
plt.subplot(323)
plt.plot(radii/rs, np.log10([epsilon*L_edd(x) for x in radii]), '--', linewidth=3, label=r'$L_{\rm Edd}$');  plt.yscale('linear'); plt.xscale('log');
plt.ylabel(r'$\log_{10} (L \rm / erg\ s^{-1})$')
plt.plot(radii/rs, np.log10([1e-3*L_edd(x) for x in radii]), '--', linewidth=3, label=r'$10^{-3}L_{\rm Edd}$');  plt.yscale('linear'); plt.xscale('log'); 
plt.plot(radii/rs, np.log10([L_c(x,gamma,alpha) for x in radii]), linewidth=3, label=r'$L_c$');  plt.xscale('log');
plt.legend(fontsize=22)
plt.xlim([20,0.9e6])
plt.subplot(326)
plt.plot(radii/rs, np.log10([HH(x)/x for x in radii]), linewidth=3);  plt.yscale('linear'); plt.xscale('log'); 
plt.ylabel(r'$\log_{10} (H/r)$')
plt.xlabel(r'$r/r_s$')
plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.1)
plt.xlim([20,0.9e6])

#%%
def tau_pm(eps, gamma, alpha, cs):
        
    C = 2 * np.pi/9 /(gamma-1)/A(gamma)
    
    return C * eps / alpha * c / cs
    
def tau_d(eps, gamma, alpha, cs):
    C1 = 1.61* 2 * np.pi/9/gamma**2/A(gamma)**2/3.31*(np.pi/3/(gamma-1))**0.5
    print (C1)
    return C1 * eps * c / cs / alpha**1.5

alphas = np.linspace(1e-3, 1e0, 10000)

plt.plot(alphas, tau_pm(1,4/3, alphas, 1e7)); plt.yscale('log'); plt.xscale('log')
plt.plot(alphas, tau_d(1,4/3, alphas, 1e7), '--'); plt.yscale('log'); plt.xscale('log')

plt.plot(alphas, tau_pm(0.01,4/3, alphas, 1e7)); plt.yscale('log'); plt.xscale('log')

#%%
alpha=0.01
def Gamma_grav(r):
    if r >= r0:
        return -3.31*HH(r)/r
    elif r <= r0:
        return 0.9*HH(r)/r

#%%
def x_over_lambda_q(r):
    return 1*HH(r)/r/ gamma * (3*gamma*tau(r,gamma)/2)**0.5 * (4/np.pi*gamma*(gamma-1)*3e10/cs)**-0.5 * min(1, (r/r0)**0.45)

def x_over_lambda(r):
    return  HH(r)/r * (np.pi**0/3/gamma**2/(gamma-1)/A(gamma)**2/alpha)**0.5

plt.plot([y/rs for y in radii], [x_over_lambda(x) for x in radii], 'blue', linewidth=3, label='accretion' );  plt.xscale('log'); #plt.yscale('log')
plt.plot([y/rs for y in radii], [x_over_lambda_q(x) for x in radii], 'red', linewidth=3, label='rad' );  plt.xscale('log'); #plt.yscale('log')
#%%
plt.plot([tau(y,4/3) for y in radii], [x_over_lambda(x) for x in radii], 'blue', linewidth=3, label='accretion' );  plt.xscale('log'); #plt.yscale('log')
plt.plot([tau(y,4/3) for y in radii], [x_over_lambda_q(x) for x in radii], 'red', linewidth=3, label='rad' );  plt.xscale('log'); #plt.yscale('log')

#%%    
def Gamma_thermal(r, eps):
  #  x_over_lambda = HH(r)/r * (np.pi/3/gamma**2/(gamma-1)/A(gamma)**2/alpha)**0.5
    C = 1.61 * (gamma-1)/gamma * x_over_lambda(r) * (L_edd(r) * eps/L_c(r,gamma,alpha) - 1) * (0.5*tau(r,gamma))**10/(1 + (0.5*tau(r,gamma))**10)
    return C

def Gamma_thermal_Q(r, eps):
  #  x_over_lambda = HH(r)/r/ gamma * (3*gamma*tau(r,gamma)/2)**0.5 * (4*gamma*(gamma-1)*3e10/cs)**-0.5 #(np.pi/3/gamma**2/(gamma-1)/A(gamma)**2/alpha)**0.5
    C = 1.61 * (gamma-1)/gamma * x_over_lambda_q(r) * (L_edd(r) * eps/L_c_q1(r,gamma) - 1) * (0.5*tau(r,gamma))**10/(1 + (0.5*tau(r,gamma))**10)
    return C


#%%
eps_eff=1
plt.rcParams["figure.figsize"] = (8,6)
plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)
if True:
    plt.plot([y/rs for y in radii], np.log10([Gamma_grav(x) for x in radii]), 'blue', linewidth=3, label='type I' );  plt.xscale('log'); #plt.yscale('log')
    plt.plot([y/rs for y in radii], np.log10([Gamma_thermal(x, eps_eff) for x in radii]),'green', linewidth=3,  label='thermal');  plt.xscale('log'); #plt.yscale('log')
    plt.plot([y/rs for y in radii], np.log10([-Gamma_grav(x) for x in radii]),'b--', linewidth=3);  plt.xscale('log'); #plt.yscale('log')
    plt.plot([y/rs for y in radii], np.log10([-Gamma_thermal(x, eps_eff) for x in radii]),'g--',  linewidth=3);  plt.xscale('log'); #plt.yscale('log')
    plt.plot([y/rs for y in radii], np.log10([Gamma_thermal_Q(x, eps_eff) for x in radii]),'r', linewidth=3);  plt.xscale('log'); #plt.yscale('log')
    plt.plot([y/rs for y in radii], np.log10([-Gamma_thermal_Q(x, eps_eff) for x in radii]),'r--',  linewidth=3);  plt.xscale('log'); #plt.yscale('log')

#    plt.plot([y/rs for y in radii], np.log10([Gamma_grav(x) + Gamma_thermal(x, eps_eff) for x in radii]),'r', linewidth=3, label='total');  plt.xscale('log'); #plt.yscale('log')
 #   plt.plot([y/rs for y in radii], np.log10([-Gamma_grav(x) - Gamma_thermal(x, eps_eff)  for x in radii]), 'r--', linewidth=3 );  plt.xscale('log'); #plt.yscale('log')

if False:
    plt.plot([tau(x,4/3) for x in radii], np.log10([Gamma_grav(x) for x in radii]), 'blue', linewidth=3, label='type I' );  plt.xscale('log'); #plt.yscale('log')
    plt.plot([tau(x,4/3) for x in radii], np.log10([Gamma_thermal(x, eps_eff) for x in radii]),'green', linewidth=3,  label='thermal');  plt.xscale('log'); #plt.yscale('log')
    plt.plot([tau(y,4/3) for y in radii], np.log10([-Gamma_grav(x) for x in radii]),'b--', linewidth=3);  plt.xscale('log'); #plt.yscale('log')
    plt.plot([tau(y,4/3) for y in radii], np.log10([-Gamma_thermal(x, eps_eff) for x in radii]),'g--',  linewidth=3);  plt.xscale('log'); #plt.yscale('log')
    plt.plot([tau(y,4/3) for y in radii], np.log10([Gamma_thermal_Q(x, eps_eff) for x in radii]),'r', linewidth=3, label='total');  plt.xscale('log'); #plt.yscale('log')
    plt.plot([tau(y,4/3) for y in radii], np.log10([- Gamma_thermal_Q(x, eps_eff)  for x in radii]), 'r--', linewidth=3 );  plt.xscale('log'); #plt.yscale('log')

plt.text(2000,2.5, r'$\epsilon=$'+str(eps_eff))
plt.text(2000,2, r'$\alpha=$'+str(alpha))
plt.ylabel(r'$\log_{10}\ \Gamma/\Gamma_0$')
#plt.plot([y/rs for y in radii], [Gamma_thermal(x, 1) + Gamma_grav(x) for x in radii] );  plt.xscale('log'); plt.yscale('log')
#plt.ylim([-1,5])
#plt.yscale('log')
plt.legend(loc=2)
plt.xlabel(r'$r/r_s$')
plt.ylim([-3,3])
plt.xlim([20,2e5])
#%%
#plt.plot([tau(x,4/3) for x in radii], np.log10([Gamma_grav(x) for x in radii]), 'blue', linewidth=3, label='type I' );  plt.xscale('log'); #plt.yscale('log')
plt.plot([tau(x,4/3) for x in radii[10:]], np.log10([Gamma_thermal(x, eps_eff) for x in radii[10:]]),'green', linewidth=3,  label='thermal');  plt.xscale('log'); #plt.yscale('log')
plt.plot([tau(y,4/3) for y in radii[10:]], np.log10([-Gamma_grav(x) for x in radii[10:]]),'b--', linewidth=3);  plt.xscale('log'); #plt.yscale('log')
plt.plot([tau(y,4/3) for y in radii[10:]], np.log10([-Gamma_thermal(x, eps_eff) for x in radii[10:]]),'g--',  linewidth=3);  plt.xscale('log'); #plt.yscale('log')
plt.plot([tau(y,4/3) for y in radii[10:]], np.log10([Gamma_grav(x) + Gamma_thermal(x, eps_eff) for x in radii[10:]]),'r', linewidth=3, label='total');  plt.xscale('log'); #plt.yscale('log')
plt.plot([tau(y,4/3) for y in radii[10:]], np.log10([-Gamma_grav(x) - Gamma_thermal(x, eps_eff)  for x in radii[10:]]), 'r--', linewidth=3 );  plt.xscale('log'); #plt.yscale('log')
plt.ylabel(r'$\log_{10}\ \Gamma/\Gamma_0$')
plt.xlabel(r'$\tau$')
plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)
plt.rcParams["figure.figsize"] = (8,6)
plt.text(2, -1.4,  r'$\epsilon=$'+str(eps_eff))
plt.text(2,-1.8, r'$\alpha=$'+str(alpha))
plt.axvline(x=tau_pm(eps_eff,4/3,alpha,5.3e6), linewidth=5, color='grey', linestyle='dashdot', alpha=0.5)
plt.axvline(x=tau_pm(eps_eff,4/3,alpha,5.3e6)*1.45*(alpha/0.01)**-0.5/(1+1.45*(alpha/0.01)**-0.5), linewidth=5, color='black', linestyle='dashed', alpha=0.5)
plt.text(200, 1,  r'$\tau_0$', color='black', alpha=0.5, size=28)
plt.text(2000, 1,  r'$\tau_\pm$', color='grey', alpha=0.5, size=28)

plt.ylim([-2.9, 2.2])
plt.xlim([0.5,8e3])

#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pd.read_csv(url, names=names)

# Split the data into features (X) and target (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors classifier on the training data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%%
import numpy as np
import matplotlib.pyplot as plt


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.01
num_steps = 10000

xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs2 = np.empty((num_steps + 1, 3))  # Need one more for the initial values

xyzs[0] = (0., 1., 1.05)  # Set initial values
xyzs2[0] = (0., 1., 1.06)  # Set initial values

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoublePendulum.py
Created on Oct 16 06:03:32 2020
"Introduction to Modern Dynamics" 2nd Edition (Oxford, 2019)
@author: nolte
"""
 
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import time
 
plt.close('all')
 
E = 1.       # Try 0.8 to 1.5
 
def flow_deriv(x_y_z_w,tspan):
    x, y, z, w = x_y_z_w
 
    A = w**2*np.sin(y-x);
    B = -2*np.sin(x);
    C = z**2*np.sin(y-x)*np.cos(y-x);
    D = np.sin(y)*np.cos(y-x);
    EE = 2 - (np.cos(y-x))**2;
     
    FF = w**2*np.sin(y-x)*np.cos(y-x);
    G = -2*np.sin(x)*np.cos(y-x);
    H = 2*z**2*np.sin(y-x);
    I = 2*np.sin(y);
    JJ = (np.cos(y-x))**2 - 2;
 
    a = z
    b = w
    c = (A+B+C+D)/EE
    d = (FF+G+H+I)/JJ
    return[a,b,c,d]
 
repnum = 150
 
np.random.seed(1)
for reploop  in range(repnum):
     
     
    px1 = 2*(np.random.random((1))-0.499)*np.sqrt(E);
    py1 = -px1 + np.sqrt(2*E - px1**2);            
 
    xp1 = 0   # Try 0.1
    yp1 = 0   # Try -0.2
     
    x_y_z_w0 = [xp1, yp1, px1, py1]
     
    tspan = np.linspace(1,1000,10000)
    x_t = integrate.odeint(flow_deriv, x_y_z_w0, tspan)
    siztmp = np.shape(x_t)
    siz = siztmp[0]
 
    if reploop % 500000 == 0:
        plt.figure(2)
        lines = plt.plot(x_t[:,0],x_t[:,1])
        plt.setp(lines, linewidth=0.5)
        plt.show()
        time.sleep(0.1)
        #os.system("pause")
 
    y1 = np.mod(x_t[:,0]+np.pi,2*np.pi) - np.pi
    y2 = np.mod(x_t[:,1]+np.pi,2*np.pi) - np.pi
    y3 = np.mod(x_t[:,2]+np.pi,2*np.pi) - np.pi
    y4 = np.mod(x_t[:,3]+np.pi,2*np.pi) - np.pi
     
    py = np.zeros(shape=(10*repnum,))
    yvar = np.zeros(shape=(10*repnum,))
    cnt = -1
    last = y1[1]
    for loop in range(2,siz):
        if (last < 0)and(y1[loop] > 0):
            cnt = cnt+1
            del1 = -y1[loop-1]/(y1[loop] - y1[loop-1])
            py[cnt] = y4[loop-1] + del1*(y4[loop]-y4[loop-1])
            yvar[cnt] = y2[loop-1] + del1*(y2[loop]-y2[loop-1])
            last = y1[loop]
        else:
            last = y1[loop]
      
    plt.figure(3)
    lines = plt.plot(yvar,py,'o',ms=1)
    plt.show()
     
plt.savefig('DPen')