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
TSD5=np.loadtxt('shmuel_data/Toomre_stable_disc_1e5.txt')
TSD4=np.loadtxt('shmuel_data/Toomre_stable_disc_1e4.txt')

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

R_shmuel5 = TSD5[:,0]*100
T_shmuel5=TSD5[:,1]
P_shmuel5=TSD5[:,2]*10
rho_shmuel5=TSD5[:,3]*1e-3
Sigma_shmuel5=TSD5[:,4]*1e-1
H_shmuel5=TSD5[:,5]*100
cs_shmuel5=TSD5[:,6]*100
r_isco5 = 1 * G * msun * 1e5/c**2
kappa_shmuel5 = np.loadtxt('shmuel_data/Kappa_1e5.txt')*10 
tau_shmuel5 = np.loadtxt('shmuel_data/Tau_1e5.txt') 

R_shmuel4 = TSD4[:,0]*100
T_shmuel4=TSD4[:,1]
P_shmuel4=TSD4[:,2]*10
rho_shmuel4=TSD4[:,3]*1e-3
Sigma_shmuel4=TSD4[:,4]*1e-1
H_shmuel4=TSD4[:,5]*100
cs_shmuel4=TSD4[:,6]*100
r_isco4 = 1 * G * msun * 1e4/c**2
kappa_shmuel4 = np.loadtxt('shmuel_data/Kappa_1e4.txt')*10 
tau_shmuel4 = np.loadtxt('shmuel_data/Tau_1e4.txt') 
#plt.plot(R_shmuel/r_isco, rho_shmuel); plt.xscale('log'); plt.yscale('log')
#%%


def get_disc_params(r,m, m_dot, alpha):
    R12 = 449.842 * alpha**(2/21) * m**(2/21) * m_dot**(16/21)
    R23 = 987.891 * m_dot**(2/3)
#    R24 =  1861.61 * alpha**-0.24 * m**-0.24 * m_dot**0.423
    R24 =  1383.38 * alpha**-0.24 * m**-0.24 * m_dot**0.423
    R34 = 3333.58 * alpha**(-0.28) * m**(-0.28) * m_dot**0.385
 #   R34 = 1401.76 * alpha**(-0.253) * m**(-0.253) * m_dot**0.413
    R1Q = 498 * alpha**(2/9) * m**(-2/9) * m_dot **(4/9)
    R2Q = 634.4 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)
    R3Q = 580.65 * alpha**(28/45) * m**(-52/45) * m_dot**(-22/45)
#    R4Q = 1737.68 * alpha**(-0.467) * m**(-0.033) * m_dot**(0.63333)
    R4Q = 184.709 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)

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

    elif R23 <=r <= R34 and r<=R3Q:
        zone=3
        gamma=5/3
        rho =  3.536e-5* alpha**-0.7 * m**-0.7 * m_dot**(11/20) * r**(-15/8)
        H = 1.27e11 * alpha**-0.1 * m**0.9 * m_dot**(3/20) * r**(9/8)
        cs = 1.756e7 * alpha**-0.1 * m**-0.1 * m_dot**(3/20) * r**(-3/8)
        P = 1.09e10 * alpha**-0.9 * m**-0.9 * m_dot**(17/20) * r**(-21/8)
        Sigma = 9e6 * alpha**-0.8 * m**0.2 * m_dot**0.7 * r**(-3/4)
        T = 2.239e6 * alpha**-0.2* m**-0.2 * m_dot**0.3 * r**(-3/4)
        kappa_R = 4e25*(1+X)*Z * rho * T**-3.5
        kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        
        kappa_m17 = (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1
        kappa = kappa_R

        P_grad=21/8
        Sigma_grad=3/4
        T_grad = 3/4
        
    elif R34 <=r <=R4Q: 
        zone=4
        gamma=5/3
        rho = 3.96e-5 * alpha**-0.7 * m**-0.7 * m_dot**(2/5) * r**(-33/20)
        H = 1.22e11 * m**0.9 * m_dot**0.2 * r**(21/20) * alpha**-0.1
        cs = 1.69e7 * m**-0.1 *alpha**-0.1 * r**(-9/20) * m_dot**0.2
        P = 1.13e10 * alpha**-0.9 * m**-0.9 * r**(-51/20) * m_dot**0.8
        Sigma = 9.7e6 * alpha**-0.8 * m**0.2 * m_dot**(3/5) * r**(-3/5)
        T = 2.07e6 * alpha**-0.2 * m**-0.2 * m_dot**(2/5) * r**(-9/10)
        kappa= 0.1 * Z
        kappa_m17 = kappa
        
        P_grad=51/20
        Sigma_grad=3/5
        T_grad = 9/10
     
        P_grad=4.54
        Sigma_grad=4.58
        T_grad = -3.08

    else:
        zone=5
        rho = 4.54949e-2 * m **-2 * r**-3
        H = 1.169e10 * alpha**(-1/3) * m**(4/3) * m_dot**(1/3) * r**1.5
        cs = 1.6146e6 *alpha**(-1/3) * m**(1/3) * m_dot**(1/3) 
        P = 1.18e11 * alpha**(-2/3) * m**(-4/3) * m_dot**(2/3) * r**-3
        Sigma = 1.064e9 * alpha**(-1/3) * m**(-2/3) * m_dot**(1/3) * r**-1.5

        # here we need to do some tricks because we don't know if the pressure is rad or gas
        # let's define some more constants in cgs        
        mu = 0.62; # mean molecular weight
        kb=1.38e-16; # Boltzmann constant 
        mp=1.67e-24 # proton mass
        
        # a critical temperature where the gas and rad pressure are equal
        T_0 = (3*c*kb/4/sb/mu/mp*rho)**(1/3)
        # temperature from gas pressure
        T_g = P/rho*mu*mp/kb;
        # temperature from radiation pressure
        T_rad = (3*c*P/4/sb)**0.25
        # assume first we're in the gas pressure regime
        T = T_g
        gamma=5/3
        T_grad = 0
        # check if we're actually in the radiation pressure
        if T_rad> T_0:
            T = T_rad
            gamma=4/3
            T_grad = 3/4
            #furthermore, T is flat in the optically thin limit
        T = T / min(1,0.1*Z*Sigma/2)**0.5
        
        kappa_R=4e25*(1+X)*Z * rho * T**-3.5
        kappa_H_minus =  1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa =  max(0.1*Z*1, (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1)

        kappa_m17 = max(0.1*Z, (kappa_H_minus**-1 + (kappa_es + kappa_R)**-1)**-1)
  
        P_grad=3
        Sigma_grad=1.5
        #optically thin limit
        if 0.1*Z*Sigma/2 <= 1:
            T_grad=0
   
    return rho, H, cs, P, Sigma, T, kappa, zone, kappa_m17, P_grad, Sigma_grad, T_grad, gamma

#which_prefactor = 'GS21'
which_prefactor='GS21'

def get_disc_derived_quantities(mm,m_dot,alpha):
    gamma=5/3
    stellar_bh_m=10
    rg = G*msun*1e8*mm/c/c
    R12 = 449.842 * alpha**(2/21) * mm**(2/21) * m_dot**(16/21)
    
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
        Gamma_I = [(-0.8 - t_g - 0.9*sigma_g) * h/r/rg for (t_g, sigma_g, r,h) in zip(T_grad, Sigma_grad, rs, Hs)]
    if which_prefactor=='JM_lin_iso':
        Gamma_I = [-(1.36 + 0.54*sigma_g + 0.5*t_g) * h/r/rg for (t_g, sigma_g, r,h) in zip(T_grad, Sigma_grad, rs, Hs)]
    if which_prefactor=='JM_lin_tot':
       Gamma_I = [(- (2.34 - 0.1*sigma_g + 1.5*t_g) / gamma  + (0.46 - 0.96*sigma_g + 1.8*t_g) / gamma) * h/r/rg for (t_g, sigma_g, r,h, gamma) in zip(T_grad, Sigma_grad, rs, Hs, gammas)]      

    if GW_flag:
        Gamma_GW = [-32/5 * (c/cs)**3 * (h/r/rg/6)**6 * r**(-4) * mm * 1e8*msun/sg/r/r/rg/rg for (cs,h,r,sg) in zip(css,Hs,rs,Sigmas)]
        for i in range(0,len(Gamma_I)):
            if i<-1:
                print ( i, '{0:.2E}'.format(rs[i]*6), '{0:.2E}'.format(Gamma_GW[i]/Gamma_I[i]), Gamma_GW[i] )
        Gamma_I = [x+y for (x,y) in zip(Gamma_I,Gamma_GW)]
    ## introduce the reduced BHL rate:
    R_bondis = [2 * G * stellar_bh_m * msun / cs**2 for cs in css]
    b = [(x*y)**0.5 for (x,y) in zip(R_bondis, r_Hills)]
    A = [(1/bb + 1/rr)**-1 for (bb,rr) in zip(b,R_bondis)]
    B = [(1/a + 1/h)**-1 for (a,h) in zip(A,Hs)]
    l_BHL_over_L_edd = [a*b*c*cs*k*rho*0.06/4/G/stellar_bh_m/msun for (a,b,cs,k,rho) in zip(A,B,css,kappas, rhos)]
#    print (min(l_BHL_over_L_edd), max(l_BHL_over_L_edd))
  #  print (l_BHL_over_L_edd)
 #   plt.plot(np.log10(rs), np.log10(l_BHL_over_L_edd))
 #   plt.plot(np.log10(rs), np.log10(taus))
 
    l_ratio = [gamma*c/k/rho/chi* min(1,ll) for (r,k,rho,chi, gamma, ll) in zip(rs,kappas,rhos, chis, gammas, l_BHL_over_L_edd)]
    Gamma_thermal = [(1-np.exp(-tt*alpha**0.5))*1.61*(gamma-1)/gamma * x/lambda_c*(l-1) for (x,lambda_c,l,tt, gamma) in zip(x_cs, lambdas, l_ratio, taus, gammas)]
 #   print(l_ratio[10])        
    return chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratio, Gamma_thermal
GW_flag =True
vv=get_disc_derived_quantities(1e-2,0.1, 0.01)

#%%
#rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness = [[get_disc_params(x,m,m_dot,alpha)[i] for x in rs] for i in range(0,8)]
# Hs = [get_disc_params(x,m,m_dot,alpha, args)[1] for x in rs]
#%%
def plot_disc_solution(mm,m_dot,alpha, col):
    r_max=6.9
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
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(rho_shmuel), linestyle='dashed',color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(rho_shmuel6),linestyle='dashed', color=col)
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(rho_shmuel5), linestyle='dashed',color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(rho_shmuel4),linestyle='dashed', color=col)

        plt.plot(np.log10(rgs), np.log10(rhos), linewidth=3, color=col, linestyle='solid'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \rho\  \rm [g\ cm^{-3}]$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-17.5,-3.4])
        plt.yticks([-16,-14,-12,-10,-8,-6, -4])
        plt.xlim([0.5,r_max])
                
        plt.subplot(422)  
        if shmuel_flag:
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(tau_shmuel),linestyle='dashed', color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(tau_shmuel6),linestyle='dashed', color=col)
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(tau_shmuel5),linestyle='dashed', color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(tau_shmuel4),linestyle='dashed', color=col)

        plt.plot(np.log10(rgs), np.log10(taus), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
#        plt.plot(np.log10(rgs), np.log10(taus_17), linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.text(1.2, 1.6+1.3*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=22)

        plt.ylabel(r'$\log \tau$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0, 2,4,6])
        plt.ylim([-3.99,5.99])
        plt.xlim([0.5,r_max])
              
        plt.subplot(423)  
        if shmuel_flag:
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(Sigma_shmuel),linestyle='dashed', color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(Sigma_shmuel6),linestyle='dashed', color=col)
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(Sigma_shmuel5),linestyle='dashed', color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(Sigma_shmuel4),linestyle='dashed', color=col)

        plt.plot(np.log10(rgs), np.log10(Sigmas), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \Sigma \  \rm [g\ cm^{-2}]$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0,2,4,6])
        plt.ylim([0,7])
#        plt.text(1.1, 1+1.5*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xlim([0.5,r_max])
       
        plt.subplot(424) 
        if shmuel_flag:
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(T_shmuel), linestyle='dashed',color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(T_shmuel6), linestyle='dashed',color=col)
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(T_shmuel5), linestyle='dashed',color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(T_shmuel4), linestyle='dashed',color=col)

        plt.plot(np.log10(rgs), np.log10(Ts), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log T_c\ \rm[K] $')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([3,4,5,6])
        plt.xlim([0.5,r_max])

        plt.subplot(425)  
        if shmuel_flag:
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(H_shmuel/R_shmuel),linestyle='dashed', color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(H_shmuel6/R_shmuel6),linestyle='dashed', color=col)
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(H_shmuel5/R_shmuel5),linestyle='dashed', color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(H_shmuel4/R_shmuel4),linestyle='dashed', color=col)

        plt.plot(np.log10(rgs), [np.log10(x/r/(G*msun*1e8*mm/c/c)) for (x,r) in zip(Hs,rgs)], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log H/r$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-2.7,-0.7])
        plt.xlim([0.5,r_max])
        plt.yticks([-2.5, -2, -1.5, -1])

        plt.subplot(426)  
        if shmuel_flag:
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(kappa_shmuel),linestyle='dashed', color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(kappa_shmuel6),linestyle='dashed', color=col)
        if shmuel_flag:
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(kappa_shmuel5),linestyle='dashed', color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(kappa_shmuel4),linestyle='dashed', color=col)

        plt.plot(np.log10(rgs), np.log10(kappas), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
 #       plt.plot(np.log10(rgs), np.log10(kappa_m17s), linewidth=3, color=col, linestyle='dotted'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \kappa\ \rm[cm^2 \ g^{-1}] $')
#        plt.text(1.1, -2.5+0.4*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xlim([0.5,r_max])
        plt.yticks([-3, -2, -1, 0, 1])
     
        plt.subplot(427)  
        if shmuel_flag:
            if mm==1:
                plt.plot(np.log10(R_shmuel/r_isco), np.log10(cs_shmuel),linestyle='dashed', color=col)
            if mm==1e-2:
                plt.plot(np.log10(R_shmuel6/r_isco6), np.log10(cs_shmuel6),linestyle='dashed', color=col)
            if mm==1e-3:
                plt.plot(np.log10(R_shmuel5/r_isco5), np.log10(cs_shmuel5),linestyle='dashed', color=col)
            if mm==1e-4:
                plt.plot(np.log10(R_shmuel4/r_isco4), np.log10(cs_shmuel4),linestyle='dashed', color=col)

        plt.plot(np.log10(rgs), np.log10(css), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.xticks([1,2,3,4,5,6])
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        plt.ylim([5.5,8.8])
        plt.yticks([6,7,8])   
        plt.ylabel(r'$\log c_s\ \rm[cm \ s^{-1}] $')
        plt.xlim([0.5,r_max])
     
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
        plt.xlim([0.5,r_max])
        plt.ylim([0.8,5.2])
        
    #    plt.figure(5)  
     #   plt.plot(np.log10(rgs), zoness)
        R1Q = 498 * alpha**(2/9) * mm**(-2/9) * m_dot **(4/9)*6
        R2Q = 634.4 * alpha**(14/27) * mm**(-26/27) * m_dot**(-8/27)*6
        R3Q = 580.65 * alpha**(28/45) * mm**(-52/45) * m_dot**(-22/45)*6
        R34 = 3333.58 * alpha**(-0.28) * mm**(-0.28) * m_dot**0.385*6
        R4Q = 184.709 * alpha**(14/27) * mm**(-26/27) * m_dot**(-8/27)*6

        #plt.axvline(np.log10(R1Q), linestyle='dashed', color=col)
        #plt.axvline(np.log10(R2Q), linestyle='dotted', color=col)
   #     plt.axvline(np.log10(R34), linestyle='solid', color=col)
   #     plt.axvline(np.log10(R3Q), linestyle='dashed', color=col)
     #   plt.axvline(np.log10(R4Q), linestyle='dotted', color=col)


    if fig_2_flag:
        plt.figure(2, figsize=(10,5))
        plt.subplots_adjust(left=0.05, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        plt.subplot(121)  
        plt.plot(np.log10(rgs), [np.log10(chi/h/cs) for (chi,h,cs) in zip(chis, Hs, css)], linewidth=3, color=col, label=r'$\log\ \chi/ H^2 \Omega$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(l_ratios), linewidth=3, color=col, linestyle='dashed', label=r'$\log \ L/L_c$'); #plt.xscale('log'); plt.yscale('log')
        #plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-3, -2,-1, 0,1, 2])
        plt.ylim([-3.5,2.2])
        if col == 'red':
            plt.legend()
            plt.axhline(0, color='grey')

        plt.subplot(122)
        plt.plot(np.log10(rgs), np.log10(lambdas) - np.log10(Hs), linewidth=3, color=col, label=r'$\log \lambda/H$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(np.fabs(x_cs)) - np.log10(lambdas), linewidth=3, color=col, linestyle='dashed', label=r'$\log x_c/\lambda$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(r_Hills) - np.log10(Hs), linewidth=3, color=col, linestyle='dotted', label=r'$\log r_H/H$'); #plt.xscale('log'); plt.yscale('log')
        if col == 'red':
            plt.legend()
            plt.axhline(0, color='grey')
 #       plt.ylabel(r'$\log \lambda/h$')
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-5,-4,-3,-2,-1, 0,1, 2,3,])
        plt.ylim([-5.5, 3.5]) #
     
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
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
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
            index=zoness.index(5)
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
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
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
            index=zoness.index(5)

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
fig_2_flag=1
fig_3_flag=1
fig_4_flag=1
m_d=0.1; alp=0.01
args1 = [10,m_d,alp, 'orange']
args2 = [1, m_d,alp, 'red']
args3 = [1e-1,m_d,alp, 'purple']
args4 = [1e-2, m_d,alp, 'blue']
args5 = [1e-3, m_d,alp, 'slategrey']
args6 = [1e-4, m_d,alp, 'green']

#args3 = [1,0.1,0.1, 'green']#[0.1,m_d,alp, 'blue']
#args4 = [1,0.1,1, 'blue']#[0.01,m_d,alp, 'green']
#from IPython import get_ipython

#plot_disc_solution(*args1)
plot_disc_solution(*args2)
#plot_disc_solution(*args3)
plot_disc_solution(*args4)
#plot_disc_solution(*args5)
plot_disc_solution(*args6)

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
    plt.figure(8, figsize=(8,6))
    plt.subplots_adjust(left=0.15, bottom=0.13, right=0.95, top=0.95, wspace=0.15)
    if True:
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
    if False:
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
  #      rho, H, cs, P, Sigma, T, kappa, zone, kappa_m17, P_grad, Sigma_grad, T_grad, gammas = get_disc_params(R_shmuel[i]/6/r_isco, 1e0, 0.1, 0.01)
   #     print(T,rho)
     #   plt.scatter(np.log10(T_shmuel), np.log10(rho_shmuel), color='purple', label=r'$\log M=8$')
        rho, H, cs, P, Sigma, T, kappa, zone, kappa_m17, P_grad, Sigma_grad, T_grad, gammas = get_disc_params(R_shmuel6[i]/6/r_isco6, 1e-2, 0.1, 0.01)
        print(T,rho)
        plt.scatter(np.log10(T_shmuel6), np.log10(rho_shmuel6), color='green',  label=r'$\log M=6$')
        plt.scatter(np.log10(T), np.log10(rho), color='blue')

#        plt.legend()


plot_kappa_regions()

#%%
if True:
    N_r=1000
    N_mm=20
    rs= np.logspace(6,0.1,N_r)
    rgs = [6*r for r in rs]
    r1 = []
    r2 = []
    dmms = np.logspace(-3,0.5,N_mm)
    
    dmmm = []
    for m_dot in dmms: #def find_when_torqrues_equal(mm, m_dot, alpha):
        mm = 1e0; 
        alpha =0.01; 
        rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,mm,m_dot,alpha)[i] for x in rs] for i in range(0,13)]
        chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratios, Gamma_thermal = [get_disc_derived_quantities(mm,m_dot,alpha)[i] for i in range (0,7)]
        signs = [np.sign(x+y) for (x,y) in zip(Gamma_I,Gamma_thermal)]
        for i in range(0, len(signs)-1):
        #print (i)
            if signs[i+1] != signs[i]:
   #             print(m_dot, dmmm)
                if len(dmmm)==0 or dmmm[-1] != m_dot:
                    r1.append(rgs[i])
                    dmmm.append(m_dot)
                else:
                    r2.append(rgs[i])
                print (m_dot, np.log10(rgs[i]))
                plt.figure(3)
                plt.scatter(np.log10(m_dot), np.log10(rgs[i]), color='red')
              #  plt.xlabel(r'$\log\ \alpha $')    
             #   plt.ylabel(r'$\log r \ \rm [r_g] $')
          #      plt.xlim([3.8,8.2])
         #       plt.subplots_adjust(left=0.12, bottom=0.16, right=0.99, top=0.98)

#%%
nn=300-len(r2)
nn2 = 300 - len(r1)
dmm2 = [dmms[i] for i in range(nn-nn2, 300-nn2)]
plt.figure(11)
plt.plot(np.log10(dmmm[:len(r1)]), np.log10(r1[:len(r1)]), color='blue', linewidth=3)
plt.plot(np.log10(dmm2), np.log10(r2), color='red', linewidth=3)

r1212 = [r1[i] for i in range(nn-nn2, 300-nn2)]

plt.fill_between(np.log10(dmm2), np.log10(r2),  np.log10(r1212), color='grey', alpha=0.4)
#plt.fill_between(np.log10(dmmm[:(nn2-4)]), 1.6,  np.log10(r1[:(nn2-4)]), color='grey', alpha=0.4)
plt.fill_between(np.log10(dmmm[:(1+nn-nn2)]), 1.7,  np.log10(r1[:(1+nn-nn2)]), color='grey', alpha=0.4)

plt.xlabel(r'$\log\ \dot{m} $')    
plt.ylabel(r'$\log r \ \rm [r_g] $')
plt.subplots_adjust(left=0.15, bottom=0.16, right=0.97, top=0.98)
plt.ylim([1.7,3.9])
plt.xlim([-3,-1.1])

#%%
if True:
    N=100
    rs= np.logspace(6,0.1,2000)
    rgs = [6*r for r in rs]
    r1 = []
    r2 = []
    alpha = 0.01
    ms = np.logspace(-2,1,N)
    m_dot = np.logspace(-3,0,N)
    how_many_traps = np.zeros([N,N])   
    for i in range(0,N):
        for j in range(0,N):
            rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,ms[i],m_dot[j],alpha)[k] for x in rs] for k in range(0,13)]
            chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratios, Gamma_thermal = [get_disc_derived_quantities(ms[i],m_dot[j],alpha)[k] for k in range (0,7)]
            signs = [np.sign(x+y) for (x,y) in zip(Gamma_I,Gamma_thermal)]
            for k in range(0, len(signs)-1):
                if signs[k+1] != signs[k]:
                    how_many_traps[i,j]+=1
            print('m= ', ms[i], '; md= ', m_dot[j], '; #traps= ', how_many_traps[i][j])

                    #%%
#plt.imshow(ms, m_dot, how_many_traps)
CS=plt.imshow( (np.transpose(how_many_traps)), extent=[6,9,-3,0])
#plt.contourf(np.transpose(how_many_traps))
plt.figure(2)
plt.contourf(np.log10(ms)+8, np.log10(m_dot), np.transpose(how_many_traps), cmap='RdBu')
plt.colorbar()
#%%
if True:
    rs= np.logspace(6,1,1000)
    rgs = [6*r for r in rs]
    r1 = []
    r2 = []
    r3=[]
    r4=[]
    alphass = np.logspace(-4.7,0,30)
    alphasss = []
    for alpha in alphass: #def find_when_torqrues_equal(mm, m_dot, alpha):
        mm = 1e-1; 
    #    alpha =0.01; 
        m_dot=0.1;
        rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,mm,m_dot,alpha)[i] for x in rs] for i in range(0,13)]
        chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratios, Gamma_thermal = [get_disc_derived_quantities(mm,m_dot,alpha)[i] for i in range (0,7)]
        signs = [np.sign(x+y) for (x,y) in zip(Gamma_I,Gamma_thermal)]
        for i in range(0, len(signs)-1):
        #print (i)
            if signs[i+1] != signs[i]:
                if  alpha>=0.1 and rgs[i]<=400:
                    r1.append(rgs[i])
                elif len(alphasss)==0 or alphasss[-1] != alpha:
                    r2.append(rgs[i])
                    alphasss.append(alpha)
                elif len(r2)!=len(r3):
                    r3.append(rgs[i])
                print (alpha, np.log10(rgs[i]))
                plt.scatter(np.log10(alpha), np.log10(rgs[i]), color='red')
              #  plt.xlabel(r'$\log\ \alpha $')    
             #   plt.ylabel(r'$\log r \ \rm [r_g] $')
          #      plt.xlim([3.8,8.2])
         #       plt.subplots_adjust(left=0.12, bottom=0.16, right=0.99, top=0.98)
#%%
if True:
    plt.figure(11)
    nn = 300-len(r1)
    plt.plot(np.log10(alphass[nn:]) , np.log10(r1), color='blue', linewidth=3)
    plt.plot(np.log10(alphasss[:len(r2)]) , np.log10(r2), color='blue', linewidth=3)
    plt.plot(np.log10(alphasss[:len(r3)]) , np.log10(r3), color='red', linewidth=3)
 
    plt.fill_between(np.log10(alphasss[-len(r1):]), 2, np.log10(r1), color='grey', alpha=0.4)
    plt.fill_between(np.log10(alphasss[:len(r3)]), np.log10(r2[:len(r3)]), np.log10(r3), color='grey', alpha=0.4)
    plt.xlabel(r'$\log\ \alpha $')    
    plt.ylabel(r'$\log r \ \rm [r_g] $')
    plt.subplots_adjust(left=0.15, bottom=0.16, right=0.97, top=0.98)
    plt.ylim([2.2,4.9])
    plt.xlim([-4.7,0])

    #%%             
which_prefactor='GS21'
#which_prefactor = 'JM_lin_iso'
#which_prefactor = 'JM_lin_tot'

if True:
    N_r=1000
    N_mm=20
    rs= np.logspace(0.1,6,N_r)
    rgs = [6*r for r in rs]
    r1 = []
    r2 = []
    r_t1 = []
    mms= np.logspace(-2,1,N_mm)

    type_i_torque_matrix = np.zeros(shape=(N_mm,N_r))
    tot_torque_matrix = np.zeros(shape=(N_mm,N_r))

    mmm = []
    for i in range(0,N_mm): #def find_when_torqrues_equal(mm, m_dot, alpha):
        Rgs = [G*msun*1e8*mm/c/c for mm in mms]
        #    mm = 1e-2; 
        alpha =0.01; 
        m_dot=1;
        rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness, kappa_m17s, P_grad, Sigma_grad, T_grad, gammas = [[get_disc_params(x,mms[i],m_dot,alpha)[j] for x in rs] for j in range(0,13)]
        chis, lambdas, x_cs, r_Hills, Gamma_I, l_ratios, Gamma_thermal = [get_disc_derived_quantities(mms[i],m_dot,alpha)[j] for j in range (0,7)]
        signs = [np.sign(x+y) for (x,y) in zip(Gamma_I,Gamma_thermal)]
        signs_type1 = [np.sign(x) for x in Gamma_I]
        
        type_i_torque_matrix[i][:] = np.divide(Gamma_I, np.divide(Hs,rs))*Rgs[i]
        tot_torque_matrix[i][:] = [x+y for (x,y) in zip(Gamma_I,Gamma_thermal)]

        for j in range(0, len(signs)-1):  
         #if j==2:
   #             print (Gamma_I[j], type_i_torque_matrix[i][j], rs[j], Hs[j]/rs[j], Hs[j]/rs[j]/Rgs[i])
            #print (i)
            if signs[j+1] != signs[j]:
                plt.scatter(np.log10(mms[i]) + 8, np.log10(rgs[j]), color='red')
                print(i)
                if len(mmm)==0 or mmm[-1] != mms[i]:
                    r1.append(rgs[j])
                    mmm.append(mms[i])
                else:
                    r2.append(rgs[j])
            if signs_type1[j+1] != signs_type1[j]:
                r_t1.append(rgs[j])
        #         print (mms[i], np.log10(rgs[j]))
    #  d          plt.xlabel(r'$\log\ M $')    
     #           plt.ylabel(r'$\log r \ \rm [r_g] $')
      #          plt.xlim([3.8,8.2])
       #         plt.subplots_adjust(left=0.12, bottom=0.16, right=0.99, top=0.98)

#plt.imshow((type_i_torque_matrix), cmap='hot', interpolation='nearest')
#%%

#%%
plt.figure(figsize=(10,7))
tn = -tot_torque_matrix
t10 = tot_torque_matrix + 10
positive_torque = np.zeros([N_mm,N_r])

negative_torque =  np.zeros([N_mm,N_r])
max_p=0; min_p=1000; max_n=0; min_n=1000

for i in range(0,N_mm):
    for j in range(0, N_r):
        if np.sign(tot_torque_matrix[i][j]) == 1:
            positive_torque[i][j] = tot_torque_matrix[i][j]
            if max_p <= positive_torque[i][j]:
                max_p = positive_torque[i][j]
            if min_p >= positive_torque[i][j]:
                min_p = positive_torque[i][j]

        if np.sign(tot_torque_matrix[i][j]) == -1:
            negative_torque[i][j] = -tot_torque_matrix[i][j]
            if max_n <= negative_torque[i][j]:
                max_n = negative_torque[i][j]
            if min_n >= negative_torque[i][j]:
                min_n = negative_torque[i][j]
            
#plt.contourf(np.log10(mms)+8, np.log10(rgs), np.transpose(type_i_torque_matrix), levels=np.linspace(np.amin(type_i_torque_matrix),np.amax(type_i_torque_matrix),44), cmap='RdBu')
#plt.contourf(np.log10(mms)+8, np.log10(rgs), np.transpose(np.log10(tn)), levels=np.linspace(-6,8,55), cmap='RdBu')
CS=plt.contourf(np.log10(mms)+8, np.log10(rgs), np.transpose(np.log10(positive_torque)), levels =np.linspace(-2, 5.35,801), cmap='bone')
CS=plt.colorbar(location='left')   
CS3=plt.contour(np.log10(mms)+8, np.log10(rgs), np.transpose(np.log10(positive_torque)),  levels =[-1,0,1,2,3],cmap='RdPu')
CS2=plt.contourf(np.log10(mms)+8, np.log10(rgs), np.transpose(np.log10(negative_torque)), levels =np.linspace(-2, 0.1,101), cmap='winter')
plt.clabel(CS3, inline=True, fontsize=16)
#plt.clabel(CS2, inline=1, fontsize=10)

plt.xlabel(r'$\log\ M $')    
plt.ylabel(r'$\log r \ \rm [r_g] $')
plt.subplots_adjust(left=0.08, bottom=0.11, right=0.9, top=0.97, wspace=0.1)
CS2=plt.colorbar()          
       
plt.text(8, 6, 'P10', color='white', size=26)
plt.text(1.3, 3, r'$\log\ |\Gamma_{\rm tot} / \Gamma_0|$', color='black', size=26, rotation=90)
plt.text(10.8, 3, r'$\log\ |-\Gamma_{\rm tot} / \Gamma_0|$', color='black', size=26, rotation=90)

 #%%
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.figure(figsize=(6,5))
plt.plot(np.log10(mmm[:len(r2)]) + 8, np.log10(r1[:len(r2)]), color='red', linewidth=3)
plt.plot(np.log10(mmm[:len(r2)]) + 8, np.log10(r2), color='blue', linewidth=3)
plt.plot(np.log10(mms) + 8, np.log10(r_t1), color='purple', linewidth=3)

plt.fill_between(np.log10(mmm[:len(r2)])+8, np.log10(r1[:len(r2)]), np.log10(r2), color='grey', alpha=0.4)
plt.xlabel(r'$\log\ M $')    
plt.ylabel(r'$\log r \ \rm [r_g] $')
plt.subplots_adjust(left=0.12, bottom=0.16, right=0.99, top=0.98)

import scipy.stats
x = [np.log10(mmm[i])+8 for i in range(0,len(r2))]
y1 = [np.log10(r1[i]) for i in range(0,len(r2))]
y2 = np.log10(r2)
plt.ylim([2,6.9])
plt.xlim([4,7.9])
#ax.xaxis.set_minor_locator(AutoMinorLocator(10))
#ax.yaxis.set_minor_locator(AutoMinorLocator(10))

slope, intercept, rr, p, se = scipy.stats.linregress(x, y1)
print (slope, intercept, rr, p, se*len(r2)**0.0, slope - 1.96*se, slope+1.96*se)

slope2, intercept2, rr2, p2, se2 = scipy.stats.linregress(x, y2)
print (slope2, intercept2, rr2, p2, se2*len(r2)**0.0,  slope2 - 1.96*se2, slope2+1.96*se2)
#plt.plot(x, [intercept2 + slope2*xx for xx in x])

x3 = [np.log10(mmm[i])+8 for i in range(0,110)]
y3 = [np.log10(r2[i]) for i in range(0,110)]
x4 = [np.log10(mmm[i])+8 for i in range(138,len(r2))]
y4 = [np.log10(r2[i]) for i in range(138,len(r2))]
slope3, intercept3, rr3, p3, se3 = scipy.stats.linregress(x3, y3)
print (slope3, intercept3, rr3, p3, se3*len(x3)**0.0,  slope3 - 1.96*se3, slope3+1.96*se3)
slope4, intercept4, rr4, p4, se4 = scipy.stats.linregress(x4, y4)
print (slope4, intercept4, rr4, p4, se4*len(x4)**0.0,  slope4 - 1.96*se4, slope4+1.96*se4)
plt.plot(x, [intercept + slope*xx for xx in x], color='black', linestyle='dashed', linewidth=3)
plt.plot(x3, [intercept3 + slope3*xx for xx in x3], color='black', linestyle='dashed', linewidth=3)
plt.plot(x4, [intercept4 + slope4*xx for xx in x4], color='black', linestyle='dashed', linewidth=3)
#plt.text(5,4, 'r='+str("%.3f" % slope))
#%%
def rs(m, m_dot, alpha):
    R12 = 449.842 * alpha**(2/21) * m**(2/21) * m_dot**(16/21)
    R23 = 987.891 * m_dot**(2/3) * m**0
    R24 =  1383.38 * alpha**-0.24 * m**-0.24 * m_dot**0.423
    R34 = 3333.58 * alpha**(-0.28) * m**(-0.28) * m_dot**0.385
    R1Q = 498 * alpha**(2/9) * m**(-2/9) * m_dot **(4/9)
    R2Q = 634.4 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)
    R3Q = 580.65 * alpha**(28/45) * m**(-52/45) * m_dot**(-22/45)
#    R4Q = 1737.68 * alpha**(-0.467) * m**(-0.033) * m_dot**(0.63333)
    R4Q = 184.709 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)

    return R12, R23, R24, R34, R1Q, R2Q, R3Q, R4Q

ms = np.logspace(-3,1,50)
R12s, R23s, R24s, R34s, R1Qs, R2Qs, R3Qs, R4Qs = rs(ms, 0.1, 0.01) 
#%%
plt.plot(np.log10(ms)+8, np.log10(R12s), label='12')
plt.plot(np.log10(ms)+8, np.log10(R23s), label='23')
plt.plot(np.log10(ms)+8, np.log10(R34s), label='34')
plt.plot(np.log10(ms)+8, np.log10(R1Qs), label='1Q')
plt.plot(np.log10(ms)+8, np.log10(R2Qs), label='2Q')
plt.plot(np.log10(ms)+8, np.log10(R3Qs), label='3Q')
plt.plot(np.log10(ms)+8, np.log10(R4Qs), label='4Q')
#plt.plot(np.log10(ms), np.log10(R23s), label='23')
#plt.plot(np.log10(ms), np.log10(R23s), label='23')
plt.legend()
#%%
for i in range(0,len(R12s)):
    if R12s[i]>=R1Qs[i]:
        plt.scatter(np.log10(ms[i])+8, np.log10(R1Qs[i]), label='1Q', color='red')
    else: 
        plt.scatter(np.log10(ms[i])+8, np.log10(R12s[i]), label='12', color='gray')
        
    if R23s[i]>=R2Qs[i]:
        plt.scatter(np.log10(ms[i])+8, np.log10(R2Qs[i]), label='2Q', color='green')
    else: 
        plt.scatter(np.log10(ms[i])+8, np.log10(R23s[i]), label='23', color='teal')

    if R34s[i]>=R3Qs[i]:
        plt.scatter(np.log10(ms[i])+8, np.log10(R3Qs[i]), label='3Q', color='blue')
    else: 
        plt.scatter(np.log10(ms[i])+8, np.log10(R34s[i]), label='34', color='black')

    if True:
        plt.scatter(np.log10(ms[i])+8, np.log10(R4Qs[i]), label='4Q', color='magenta')
 #   plt.legend()
