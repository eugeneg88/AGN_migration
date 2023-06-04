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
matplotlib.rcParams.update({'font.size': 20})
from scipy.optimize import root 

#constants
G=6.674e-8
msun = 1.989e33
au = 1.49e13
c=2.99792458e10
sb=5.6e-5
#gamma = 5/3
## some tests
#m=1
m_dot=0.1
alpha=0.01
rs= np.logspace(0.01,6,1000)

#%%
#load data from shmuel
TSD=np.loadtxt('shmuel_data/Toomre_stable_disc.txt')
TSD6=np.loadtxt('shmuel_data/Toomre_stable_disc_1e6.txt')

#[R,T,P,rho,Sigma,H,cs] 
#conver to cgs units
R_shmuel = TSD[:,0]*100
T_shmuel=TSD[:,1]
P_shmuel=TSD[:,2]*10
rho_shmuel=TSD[:,3]*1e-3
Sigma_shmuel=TSD[:,4]*1e-1
H_shmuel=TSD[:,5]*100
cs_shmuel=TSD[:,6]*100
r_isco = 1 * G * msun * 1e8/c**2
kappa_shmuel = np.loadtxt('shmuel_data/Kappa.txt')*10 
tau_shmuel = np.loadtxt('shmuel_data/Tau.txt') 

R_shmuel6 = TSD6[:,0]*100
T_shmuel6=TSD6[:,1]
P_shmuel6=TSD6[:,2]*10
rho_shmuel6=TSD6[:,3]*1e-3
Sigma_shmuel6=TSD6[:,4]*1e-1
H_shmuel6=TSD6[:,5]*100
cs_shmuel6=TSD6[:,6]*100
r_isco6 = 1 * G * msun * 1e6/c**2
kappa_shmuel6 = np.loadtxt('shmuel_data/Kappa_1e6.txt')*10 
tau_shmuel6 = np.loadtxt('shmuel_data/Tau_1e6.txt') 

#plt.plot(R_shmuel/r_isco, rho_shmuel); plt.xscale('log'); plt.yscale('log')
#%%


def get_disc_params(r,m, m_dot, alpha):
    R12 = 449.842 * alpha**(2/21) * m**(2/21) * m_dot**(16/21)
    R23 = 987.891 * m_dot**(2/3)
#    R24 =  1861.61 * alpha**-0.24 * m**-0.24 * m_dot**0.423
    R24 =  1383.38 * alpha**-0.24 * m**-0.24 * m_dot**0.423
#    R34 = 1908.43 * alpha**(-0.253) * m**(-0.253) * m_dot**0.413
    R34 = 1401.76 * alpha**(-0.253) * m**(-0.253) * m_dot**0.413
    R1Q = 498 * alpha**(2/9) * m**(-2/9) * m_dot **(4/9)
    R2Q = 634.4 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)
    R3Q = 580.65 * alpha**(28/45) * m**(-52/45) * m_dot**(-22/45)
    R4Q = 1737.68 * alpha**(-0.467) * m**(-0.033) * m_dot**(0.63333)

    X=0.7381
    Y=0.2485
    Z=0.0134
    kappa_es=0.2*(1+X)
    m_dot = m_dot * (1-r**-0.5)
    
    if r<=R12 and r<=R1Q:
        zone=1
        gamma=4/3
        rho = 3.29e-14* alpha **-1 * m**-1 * m_dot**-2 * r**1.5 
        H = 1.3e14 * m * m_dot
        cs = 1.8e10 * m_dot * r**-1.5
        P = 1.065e7 * alpha**-1 * m**-1 * r**-1.5
        Sigma = 8.574 * alpha**-1 * m_dot**-1 * r**1.5 
        T = 254920 * alpha**-0.25 * m**-0.25 * r**(-3/8)
        kappa=kappa_es
        
        kappa_R = 4e25*(1+X)*Z * rho * T**-3.5
        kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa_m17 = (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1
        
        P_grad=1.5
        Sigma_grad=-1.5
        T_grad = 3/8

    elif R12 <= r <= R23 and r<=R2Q:
        zone=2
        gamma=5/3
        rho = 7.495e-6 * alpha**-0.7 * m**-0.7 * m_dot**(2/5) * r**(-33/20)
        H = 2.13e11 * m**0.9 * m_dot**0.2 * r**(21/20) * alpha**-0.1
        cs = 2.945e7 * m**-0.1 *alpha**-0.1 * r**(-9/20) * m_dot**0.2
        P = 6.5e9 * alpha**-0.9 * m**-0.9 * r**(-51/20) * m_dot**0.8
        Sigma = 3.196e6 * alpha**-0.8 * m**0.2 * m_dot**(3/5) * r**(-3/5)
        T = 6.3e6 * alpha**-0.2 * m**-0.2 * m_dot**(2/5) * r**(-9/10)
        kappa=kappa_es
        
        kappa_R = 4e25*(1+X)*Z * rho * T**-3.5
        kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa_m17 = (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1

        P_grad=51/20
        Sigma_grad=3/5
        T_grad = 9/10

    elif R23 <=r <=R3Q: and r<=R34:
        zone=3
        gamma=5/3
        rho =  3.536e-5* alpha**-0.7 * m**-0.7 * m_dot**(11/20) * r**(-15/8)
        H = 1.27e11 * alpha**-0.1 * m**0.9 * m_dot**(3/20) * r**(9/8)
        cs = 1.756e7 * alpha**-0.1 * m**-0.1 * m_dot**(3/20) * r**(-3/8)
        P = 1.09e10 * alpha**-0.9 * m**-0.9 * m_dot**(17/20) * r**(-21/8)
        Sigma = 9e6 * alpha**-0.8 * m**0.2 * m_dot**0.7 * r**(-3/4)
        T = 2.239e6 * alpha**-0.2* m**-0.2 * m_dot**0.3 * r**(-3/4)
        kappa_R = 4e25*(1+X)*Z * rho * T**-3.5
        kappa = kappa_R
        kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        
        kappa_m17 = (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1

        P_grad=21/8
        Sigma_grad=3/4
        T_grad = 3/4
        
    elif (R34 <= r) and r>= R4Q and False: 
        zone=4
        gamma=5/3
        rho = 4.09e13* alpha**-2.154*m**-2.154*m_dot**2.923*r**-7.62
        H = 1.21e5 * alpha**0.385 * m**1.385 * m_dot**-0.64 * r**3.03
        cs = 16.73 * alpha**0.385 * m**0.385 * m_dot**-0.64 * r**1.54
        P = 1.14e16 *  alpha**-1.385 * m**-1.385 * m_dot**1.64 * r**-4.54
        Sigma = 9.9e18 * alpha**-1.77 * m**-0.77 * m_dot**2.28 * r**-4.58
        T = 2.03e-6 * alpha**0.77* m**0.77 * m_dot**-1.282 * r**3.08
        kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa=kappa_H_minus

        kappa_R = 4e25*(1+X)*Z * rho * T**-3.5        
        kappa_m17 = (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1
     
        P_grad=4.54
        Sigma_grad=4.58
        T_grad = -3.08

    #else:
    else:#f (r<=R12 and r>=R1Q) or (R12 <= r <= R23 and r>R2Q) or (R23 <=r <=R34 and r>R3Q) or (R34<=r and r>R3Q):
   # if (r<=R12 and r>=R1Q):
        zone=4#5
        rho = 4.54949e-2 * m **-2 * r**-3
        H = 1.169e10 * alpha**(-1/3) * m**(4/3) * m_dot**(1/3) * r**1.5
        cs = 1.6146e6 *alpha**(-1/3) * m**(1/3) * m_dot**(1/3) 
        P = 1.18e11 * alpha**(-2/3) * m**(-4/3) * m_dot**(2/3) * r**-3
        Sigma = 1.064e9 * alpha**(-1/3) * m**(-2/3) * m_dot**(1/3) * r**-1.5
        R_previous = -1
        if r<R12 and r>=R1Q: # coming from zone 1
            R_previous = R1Q
        if  (R12 <= r <= R23 and r>R2Q): #coming from zone 2:
            R_previous = R2Q
        if  (R23 <=r <=R34 and r>R3Q): # coming from zone 3:
            R_previous = R3Q
        if ((max(R34,R24) <= r) and r>min(R4Q,R3Q)): # coming from zone 4
            R_previous = R4Q
        
#        T = 1.893e4 * alpha**(-2/3)* m**(2/3) * m_dot**(2/3) * (r/R2Q)**(-3/4) / min(1,0.1*Z*Sigma/2)**0.5 # artificial tweak
        mu = 0.62; kb=1.38e-16; mp=1.67e-24
        T_0 = (3*c*kb/4/sb/mu/mp*rho)**(1/3)
        T_g = P/rho*mu*mp/kb;
        T_rad = (3*c*P/4/sb)**0.25
        T = T_g
        gamma=5/3
        T_grad = 0
        if T_rad> T_0:#min(T_g, T_rad) > T_0:
            T = T_rad
            gamma=4/3
        T_grad = 3/4
        T = T / min(1,0.1*Z*Sigma/2)**0.5
        kappa_R=4e25*(1+X)*Z * rho * T**-3.5
        kappa_H_minus =  1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa =  max(0.1*Z*1, (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1)

        kappa_m17 = max(0.1*Z, (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1)
  
        P_grad=3
        Sigma_grad=1.5
        if 0.1*Z*Sigma/2 <= 1:
            T_grad=0

    return rho, H, cs, P, Sigma, T, kappa, zone, kappa_m17, P_grad, Sigma_grad, T_grad, gamma

def get_disc_derived_quantities(mm,m_dot,alpha):
    gamma=5/3
    stellar_bh_m=10
    rg = G*msun*1e8*mm/c/c
    R12 = 449.842 * alpha**(2/21) * mm**(2/21) * m_dot**(16/21)
    which_prefactor = 'GS21'
    
    rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,mm,m_dot,alpha)[i] for x in rs] for i in range(0,13)]
    taus = [x*y/2 for (x,y) in zip(kappas, Sigmas)]
    taus_17 = [x*y/2 for (x,y) in zip(kappa_m17s, Sigmas)]

    chis = [ 16 * gamma * (gamma-1) * sb * t**4 / 3 / k/ rho**2 /cs**2 for (t,k,cs,rho, gamma) in zip(Ts, kappas,css, rhos, gammas)]  
    lambdas = [(2 * chi / 3/gamma/cs*h)**0.5 for (chi,cs,h, gamma) in zip(chis, css, Hs, gammas)]
  #  print(mm, lambdas[0]/1.5e13)
    x_cs = [np.fabs(-P_g * h**2/gamma /r/rg) for (P_g,h,r, gamma) in zip(P_grad, Hs, rs, gammas)]
    r_Hills = [r*rg * (stellar_bh_m/1e8/mm/3)**(1/3) for r in rs]
      
    Gamma_0 = [sigma*(R12*rg/h)**5 * (R12*rg)**2 * cs**2 * (stellar_bh_m/1e8/mm)**2 for (sigma,h,cs) in zip(Sigmas,Hs,css)]    
    if which_prefactor=='GS21':
        Gamma_I = [(-0.8 - t_g - 0.9*sigma_g) * np.fabs(x)/h for (t_g, sigma_g, x,h) in zip(T_grad, Sigma_grad, x_cs, Hs)]
    if which_prefactor=='JM_lin_iso':
        Gamma_I = [-(1.36 + 0.54*sigma_g + 0.5*t_g) * np.fabs(x)/h for (t_g, sigma_g, x,h) in zip(T_grad, Sigma_grad, x_cs, Hs)]
    if which_prefactor=='JM_lin_tot':
       Gamma_I = [(- (2.34 - 0.1*sigma_g + 1.5*t_g) / gamma  + (0.46 - 0.96*sigma_g + 1.8*t_g) / gamma) * np.fabs(x)/h for (t_g, sigma_g, x,h, gamma) in zip(T_grad, Sigma_grad, x_cs, Hs, gammas)]      

    l_ratio = [gamma*c/k/rho/chi for (r,k,rho,chi, gamma) in zip(rs,kappas,rhos, chis, gammas)]
    Gamma_thermal = [(1-np.exp(-tt*alpha**0.5))*1.61*(gamma-1)/gamma * x/lambda_c*(l-1) for (x,lambda_c,l,tt, gamma) in zip(x_cs, lambdas, l_ratio, taus, gammas)]
 #   print(l_ratio[10])        
    return chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratio, Gamma_thermal

vv=get_disc_derived_quantities(1, 0.1, 0.01)
#%%
#rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness = [[get_disc_params(x,m,m_dot,alpha)[i] for x in rs] for i in range(0,8)]
# Hs = [get_disc_params(x,m,m_dot,alpha, args)[1] for x in rs]
#%%
def plot_disc_solution(mm,m_dot,alpha, col):
    rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,mm,m_dot,alpha)[i] for x in rs] for i in range(0,13)]
    chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratios, Gamma_thermal = [get_disc_derived_quantities(mm,m_dot,alpha)[i] for i in range (0,7)]

    taus = [x*y/2 for (x,y) in zip(kappas, Sigmas)]
    taus_17 = [x*y/2 for (x,y) in zip(kappa_m17s, Sigmas)]
  
    rgs = [6 * r for r in rs]
    which_prefactor = 'GS21'
#    which_prefactor = 'JM17_lin_tot' 
    #which_prefactor = 'JM17_lin_iso'
    if fig_1_flag:
        R12 = 449.842 * alpha**(2/21) * mm**(2/21) * m_dot**(16/21)
        R23 = 987.891 * m_dot**(2/3)
        R24 =  1861.61 * alpha**-0.24 * mm**-0.24 * m_dot**0.423
        R34 = 1908.43 * alpha**(-0.253) * mm**(-0.253) * m_dot**0.413
        R1Q = 498 * alpha**(2/9) * mm**(-2/9) * m_dot **(4/9)
        R2Q = 634.4 * alpha**(14/27) * mm**(-26/27) * m_dot**(-8/27)
        R3Q = 580.65 * alpha**(28/45) * mm**(-52/45) * m_dot**(-22/45)
 #       R4Q = 2550.56 * alpha**(-0.467) * mm**(-0.033) * m_dot**(0.63333)
        R4Q = 1737.68 * alpha**(-0.467) * mm**(-0.033) * m_dot**(0.63333)

        plt.figure(1, figsize=(12,25))
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.99, top=0.98, wspace=0.21, hspace=0.01)
        
        plt.subplot(421)  
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(rho_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(rho_shmuel6), color='green')
        plt.plot(np.log10(rgs), np.log10(rhos), linewidth=3, color=col, linestyle='solid'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \rho\  \rm [g\ cm^{-3}]$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-17.5,-3.4])
        plt.yticks([-16,-14,-12,-10,-8,-6, -4])
        plt.xlim([0.5,6.2])
               
        plt.subplot(422)  
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(tau_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(tau_shmuel6), color='green')
        plt.plot(np.log10(rgs), np.log10(taus), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
#        plt.plot(np.log10(rgs), np.log10(taus_17), linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.ylabel(r'$\log \tau$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0, 2,4,6])
        plt.ylim([-2.8,6.6])
        plt.xlim([0.5,6.2])
              
        plt.subplot(423)  
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(Sigma_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(Sigma_shmuel6), color='green')
        plt.plot(np.log10(rgs), np.log10(Sigmas), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \Sigma \  \rm [g\ cm^{-2}]$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0,2,4,6])
        plt.ylim([-2.5,7])
        plt.text(1.1, 1+1.5*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xlim([0.5,6.2])
       
        plt.subplot(424) 
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(T_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(T_shmuel6), color='green')
        plt.plot(np.log10(rgs), np.log10(Ts), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log T_c\ \rm[K] $')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([3,4,5,6])
        plt.xlim([0.5,6.2])

        plt.subplot(425)  
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(H_shmuel/R_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(H_shmuel6/R_shmuel6), color='green')
        plt.plot(np.log10(rgs), [np.log10(x/r/(G*msun*1e8*mm/c/c)) for (x,r) in zip(Hs,rgs)], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log H/r$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-2.7,-0.7])
        plt.xlim([0.5,6.2])
        plt.yticks([-2.5, -2, -1.5, -1])

        plt.subplot(426)  
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(kappa_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(kappa_shmuel6), color='green')
        plt.plot(np.log10(rgs), np.log10(kappas), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
 #       plt.plot(np.log10(rgs), np.log10(kappa_m17s), linewidth=3, color=col, linestyle='dotted'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \kappa\ \rm[cm^2 \ g^{-1}] $')
#        plt.text(1.1, -2.5+0.4*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xlim([0.5,6.2])
        plt.yticks([-3, -2, -1, 0, 1])
     
        plt.subplot(427)  
        if shmuel_flag:
            plt.plot(np.log10(R_shmuel/r_isco), np.log10(cs_shmuel), color='red')
            plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(cs_shmuel6), color='green')
        plt.plot(np.log10(rgs), np.log10(css), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.xticks([1,2,3,4,5,6])
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        plt.ylim([5.5,8.8])
        plt.yticks([6,7,8])   
        plt.ylabel(r'$\log c_s\ \rm[cm \ s^{-1}] $')
        plt.xlim([0.5,6.2])
     
        plt.subplot(428)  
        plt.plot(np.log10(rgs), zoness, linewidth=3, color=col)
        if col=='blue' or col=='green':
            print (col)
   #         plt.axvline(np.log10(6*R3Q), linestyle='dashed', color=col)
   #         plt.axvline(np.log10(6*R24), linestyle='dashed', color=col)

        plt.ylabel('zone')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([1,2,3,4,5])
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        if mm==1:
            plt.text(3.4, 2, r'$\dot{m} = $' + str(m_d), color='black', size=26)
        plt.xlim([0.5,6.2])
        plt.ylim([0.8,4.2])
        
    #    plt.figure(5)  
     #   plt.plot(np.log10(rgs), zoness)
        R1Q = 498 * alpha**(2/9) * mm**(-2/9) * m_dot **(4/9)*6
        R2Q = 634.4 * alpha**(14/27) * mm**(-26/27) * m_dot**(-8/27)*6
        R3Q = 580.65 * alpha**(28/45) * mm**(-52/45) * m_dot**(-22/45)*6
        #plt.axvline(np.log10(R1Q), linestyle='dashed', color=col)
        #plt.axvline(np.log10(R2Q), linestyle='dotted', color=col)
        plt.axvline(np.log10(R3Q), linestyle='solid', color=col)
        plt.axvline(np.log10(R2Q), linestyle='dashed', color=col)
        plt.axvline(np.log10(R1Q), linestyle='dotted', color=col)


    if fig_2_flag:
        plt.figure(2, figsize=(10,5))
        plt.subplots_adjust(left=0.05, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        plt.subplot(121)  
        plt.plot(np.log10(rgs), [np.log10(chi/h/cs) for (chi,h,cs) in zip(chis, Hs, css)], linewidth=3, color=col, label=r'$\log\ \chi/ H^2 \Omega$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(l_ratios), linewidth=3, color=col, linestyle='dashed', label=r'$\log \ L_{\rm Edd}/L_c$'); #plt.xscale('log'); plt.yscale('log')
        #plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-3, -2,-1, 0,1, 2])
        plt.ylim([-3.5,2.2])
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')

        plt.subplot(122)
        plt.plot(np.log10(rgs), np.log10(lambdas) - np.log10(Hs), linewidth=3, color=col, label=r'$\log \lambda/H$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(np.fabs(x_cs)) - np.log10(lambdas), linewidth=3, color=col, linestyle='dashed', label=r'$\log x_c/\lambda$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(Hs) - np.log10(r_Hills), linewidth=3, color=col, linestyle='dotted', label=r'$\log H/r_H$'); #plt.xscale('log'); plt.yscale('log')
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')
 #       plt.ylabel(r'$\log \lambda/h$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-5,-4,-3,-2,-1, 0,1, 2,3,])
        plt.ylim([-5.5, 3.5])
     
    if fig_3_flag:

        plt.figure(3, figsize=(10,5))
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        Gamma_I_minus = [-x for x in Gamma_I]
        Gamma_thermal_minus = [-x for x in Gamma_thermal]
        G_tot = [x+y for (x,y) in zip(Gamma_I,Gamma_thermal)]
        G_tot_minus = [x+y for (x,y) in zip(Gamma_I_minus, Gamma_thermal_minus)]
        plt.subplot(121)
        plt.plot(np.log10(rgs), np.log10(G_tot), linewidth=3, color=col, label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot_minus), linewidth=3, color=col, linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.3,1.95])
        plt.xlim([0.5,6.3])
        R12 = 449.842 * alpha**(2/21) * mm**(2/21) * m_dot**(16/21)
        R23 = 987.891 * m_dot**(2/3)
        R24 =  1861.61 * alpha**-0.24 * mm**-0.24 * m_dot**0.423
        R34 = 1908.43 * alpha**(-0.253) * mm**(-0.253) * m_dot**0.413
        R1Q = 498 * alpha**(2/9) * mm**(-2/9) * m_dot **(4/9)
        R2Q = 634.4 * alpha**(14/27) * mm**(-26/27) * m_dot**(-8/27)
        R3Q = 580.65 * alpha**(28/45) * mm**(-52/45) * m_dot**(-22/45)
#       R4Q = 2550.56 * alpha**(-0.467) * mm**(-0.033) * m_dot**(0.63333)
        R4Q = 1737.68 * alpha**(-0.467) * mm**(-0.033) * m_dot**(0.63333)
        plt.axvline(np.log10(6*R23), color=col)
#
    #    if col=='orange':
     #       plt.legend()

        plt.subplot(122)
        if 2 in zoness:
            index=zoness.index(2)
        else:
            index=zoness.index(4)
        plt.plot(np.log10(taus[index:]), np.log10(G_tot[index:]), linewidth=3, color=col, label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(taus[index:]), np.log10(G_tot_minus[index:]), linewidth=3, color=col, linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.xlabel(r'$\log \tau$')
        plt.xticks([-1, 0,1,2,3,4,5,6])
        #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-3.2,1.95])
        plt.xlim([1,5.6])
        print (0.065*3e10/1e7/alpha**1.5)
        plt.axvline(np.log10(0.5*0.065*3e10/0.64e7/alpha**1.5), color=col)
        plt.axvline(np.log10(0.4*0.065*3e10/0.4e7/alpha**1.5), color=col)

    if fig_4_flag:
        Gamma_I_minus = [-x for x in Gamma_I]
        Gamma_thermal_minus = [-x for x in Gamma_thermal]
        Gamma_tot = [x+y for (x,y) in zip(Gamma_I,Gamma_thermal)]
        Gamma_tot_minus = [x+y for (x,y) in zip(Gamma_I_minus, Gamma_thermal_minus)]

        plt.figure(6, figsize=(10,5))
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        plt.subplot(121)
         
        plt.plot(np.log10(rgs), np.log10(Gamma_I), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(Gamma_I_minus), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(Gamma_thermal), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(Gamma_thermal_minus), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(Gamma_tot), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(Gamma_tot_minus), linewidth=3, color='blue', linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0])
        plt.ylim([-4.5,0.4])
        plt.xlim([0.5,6.3])
        R12 = 449.842 * alpha**(2/21) * mm**(2/21) * m_dot**(16/21)
        plt.axvline(np.log10(6*R12))
# 
  #      plt.legend()
  #      plt.text(0.6,-0.3, 'log M=' + str(int(np.log10(1e8*mm))), color='black', size=26)
  
        plt.subplot(122)
        if 2 in zoness:
            index=zoness.index(2)
        else:
            index=zoness.index(4)

        plt.plot(np.log10(taus[index:]), np.log10(Gamma_I[index:]), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(taus[index:]), np.log10(Gamma_I_minus[index:]), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(taus[index:]), np.log10(Gamma_thermal[index:]), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(taus[index:]), np.log10(Gamma_thermal_minus[index:]), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(taus[index:]), np.log10(Gamma_tot[index:]), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(taus[index:]), np.log10(Gamma_tot_minus[index:]), linewidth=3, color='blue', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.xticks([-1, 0,1,2,3,4,5,6])
     #   plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.5,0.4])
        plt.xlim([1,5.6])
        plt.legend(loc=3)
   
     #%% 
shmuel_flag = True
fig_1_flag = 1
fig_2_flag=0
fig_3_flag=0
fig_4_flag=0
m_d=0.1; alp=1e-2
args1 = [0.001,m_d,alp, 'orange']
args2 = [1,m_d,alp, 'red']
args3 = [0.1,m_d,alp, 'blue']
args4 = [0.01,m_d,alp, 'green']
#from IPython import get_ipython

#plot_disc_solution(*args1)
plot_disc_solution(*args2)
plot_disc_solution(*args3)
plot_disc_solution(*args4)

#%%
rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,1,0.1,0.01)[i] for x in rs] for i in range(0,13)]
#%%
X=0.75
Y=0.24
Z=1-X-Y

def kappas(rho,T):
    kappa_es = 0.2*(1+X)
    kappa_R = 4e25*(1+X)*Z * rho * T**-3.5
    kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
    kappa_mol = 0.1*Z
 #   print(X,Y,Z, rho, T, kappa_R)
    return kappa_es, kappa_R, kappa_H_minus, kappa_mol

Ts = [3000, 10000, 30000, 100000, 300000]
colors = ['red', 'green', 'blue', 'black', 'magenta']
rhos = np.logspace(-15,-5,1000)

for T in Ts:
    i = Ts.index(T)
    ks = [kappas(rho, T) for rho in rhos]
    es = [ks[x][0] for x in range(0, len(rhos))]
    R = [ks[x][1] for x in range(0, len(rhos))]
    Hm = [ks[x][2] for x in range(0, len(rhos))]
    mol = [ks[x][3] for x in range(0, len(rhos))]
    kappa_metzger = [m + (1/h + 1/(e+r))**-1 for (m,h,e,r) in zip(mol, Hm, es, R)]
    min_k = [min(x,y) for (x,y) in zip(es,R)]
    if T==4000:
        plt.plot(np.log10(rhos), np.log10(es), 'gray')
        plt.plot(np.log10(rhos), np.log10(mol), 'gray')
    plt.plot(np.log10(rhos), np.log10(R), color = colors[i],   linestyle='--')
    plt.plot(np.log10(rhos), np.log10(kappa_metzger), color = colors[i],   linewidth=4)
    plt.plot(np.log10(rhos), np.log10(Hm),  color = colors[i], label=str(T))
    plt.legend()

#%%%
def plot_kappa_regions():
    rhos = np.logspace(-20,-5, 1000)
    Ts = np.logspace(2.5,8, 1000)
    X=0.75
    Y=0.24
    Z=1-X-Y
    kappa1 = 4e25*(1+X)*Z
    kappa2 = 1.1e-25*Z**0.5
    kappa_es = 0.2*(1+X)
    def kappa_R(rho,T): return  kappa1* rho * T**-3.5
    def kappa_H_minus(rho,T): return kappa2 * rho**0.5 * T**7.7
    kappa_mol = 0.1*Z
    
    es_and_KR = [kappa_es/kappa1 * T**3.5 for T in Ts]
    H_minus_and_es = [(kappa_es/kappa2)**2 * T**-15.4 for T in Ts]
    H_minus_and_KR = [(kappa2/kappa1)**2 * T**22.4 for T in Ts]
    H_minus_and_mol = [(kappa_mol/kappa2)**2 * T**-15.4 for T in Ts]
    line1 = [max(x,y) for (x,y) in zip(H_minus_and_es, H_minus_and_KR)]
  #  plt.plot(np.log10(Ts), np.log10(line1),label=r'$1$', linewidth=5)
    plt.figure(7, figsize=(8,6))
    plt.subplots_adjust(left=0.15, bottom=0.13, right=0.95, top=0.95, wspace=0.15)

    plt.plot(np.log10(Ts), np.log10(es_and_KR),label=r'$\kappa_{\rm es} = \kappa_{\rm K}$')
    plt.plot(np.log10(Ts), np.log10(H_minus_and_es), label=r'$\kappa_{\rm es} = \kappa_{\rm H-}$')
    plt.plot(np.log10(Ts), np.log10(H_minus_and_KR), label=r'$\kappa_{\rm K} = \kappa_{\rm H-}$')
    plt.plot(np.log10(Ts), np.log10(H_minus_and_mol), label=r'$\kappa_{\rm m} = \kappa_{\rm H-}$')
    plt.xlabel(r'$\log\ T$')
    plt.ylabel(r'$\log\ \rho$')
    plt.fill_between(np.log10(Ts), np.log10(H_minus_and_mol), np.log10(H_minus_and_es), color='red',alpha=0.3, label='H-')
    plt.fill_between(np.log10(Ts), np.log10(line1), -5, color='red',alpha=0.3)
    plt.fill_between(np.log10(Ts), np.log10(H_minus_and_KR),np.log10(es_and_KR), where=([x<y for (x,y) in zip(H_minus_and_es,es_and_KR)]) , color='blue',alpha=0.3, label='K')
    plt.fill_between(np.log10(Ts), -15, np.log10(H_minus_and_mol), color='grey',alpha=0.3, label='m')
    plt.fill_between(np.log10(Ts), np.log10(H_minus_and_es), np.log10(es_and_KR), where=([x<y for (x,y) in zip(H_minus_and_es,es_and_KR)]), color='green',alpha=0.3, label='es')
    plt.legend()
    if True:
        for i in range(0,len(R_shmuel)):
            if 8e5*r_isco >= R_shmuel[i] >= 6*r_isco:
                plt.scatter(np.log10(T_shmuel[i]), np.log10(rho_shmuel[i]), color='blue', label='Shmuel')
      #      if 2e4*r_isco >= R_shmuel[i] >= 8e3*r_isco:
        #        plt.scatter(np.log10(T_shmuel[i]), np.log10(rho_shmuel[i]), color='red')
       #     if 2e4*r_isco <= R_shmuel[i]:
         #       plt.scatter(np.log10(T_shmuel[i]), np.log10(rho_shmuel[i]), color='gray')

    plt.xlim([3,6])
    plt.ylim([-15,-5])
    for i in range(0,len(R_shmuel)):
        rho, H, cs, P, Sigma, T, kappa, zone, kappa_m17, P_grad, Sigma_grad, T_grad = get_disc_params(R_shmuel[i]/6/r_isco, 1e0, 0.1, 0.01)
   #     print(T,rho)
        plt.scatter(np.log10(T_shmuel), np.log10(rho_shmuel), color='purple', label=r'$\log M=8$')
        rho, H, cs, P, Sigma, T, kappa, zone, kappa_m17, P_grad, Sigma_grad, T_grad = get_disc_params(R_shmuel6[i]/6/r_isco6, 1e-2, 0.1, 0.01)
    #    print(T,rho)
        plt.scatter(np.log10(T_shmuel6), np.log10(rho_shmuel6), color='green',  label=r'$\log M=6$')
#        plt.legend()


plot_kappa_regions()

#%%
rs= np.logspace(0.01,6,1000)
rgs = [6*r for r in rs]
mms= np.logspace(-3,0,100)
for mm in mms: #def find_when_torqrues_equal(mm, m_dot, alpha):
    #mm = 1e-2; 
    alpha=0.001; m_dot=0.1
    rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,mm,m_dot,alpha)[i] for x in rs] for i in range(0,13)]
    chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratios, Gamma_thermal = [get_disc_derived_quantities(mm,m_dot,alpha)[i] for i in range (0,7)]
    signs = [np.sign(x+y) for (x,y) in zip(Gamma_I,Gamma_thermal)]
    for i in range(0, len(signs)-1):
        if signs[i+1] != signs[i]:
 #           print (mm, np.log10(rgs[i]))
            plt.scatter(np.log10(mm) + 8, np.log10(rgs[i]), color='red')
        plt.xlabel(r'$\log\ M $')    
        plt.ylabel(r'$\log r \ \rm [r_g] $')
        plt.xlim([4.8,8.2])
        plt.subplots_adjust(left=0.12, bottom=0.16, right=0.99, top=0.98)
