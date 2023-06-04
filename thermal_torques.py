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
matplotlib.rcParams.update({'font.size': 23})
#import

#constants
G=6.674e-8
msun = 2e33
au = 1.5e13
mbh = 1e8*msun
m=10*msun
c=3e10
rs = 2 * G * mbh / c / c

## model parameters
n = 3 # pressure gradient

r0 = 1e3 * rs
radii = np.linspace(0.01*r0, 1000*r0, 10000)

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
