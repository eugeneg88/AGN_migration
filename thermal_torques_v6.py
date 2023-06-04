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
gamma = 5/3
## some tests
m=1
m_dot=0.1
alpha=0.01
R12 = 449.842 * alpha**(2/21) * m**(2/21) * m_dot**(16/21)
R23 = 987.891 * m_dot**(2/3)
R24 =  1861.61 * alpha**-0.24 * m**-0.24 * m_dot**0.423
R34 = 1908.43 * alpha**(-0.253) * m**(-0.253) * m_dot**0.413
R1Q = 498 * alpha**(2/9) * m**(-2/9) * m_dot **(4/9)
R2Q = 634.4 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)
R3Q = 580.65 * alpha**(28/45) * m**(-52/45) * m_dot**(-22/45)
R4Q = 2550.56 * alpha**(-0.467) * m**(-0.033) * m_dot**(0.63333)

rs= np.logspace(0.01,6,10000)

#%%
#load data from shmuel
TSD=np.loadtxt('shmuel_data/Toomre_stable_disc.txt')
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
#plt.plot(R_shmuel/r_isco, rho_shmuel); plt.xscale('log'); plt.yscale('log')
#%%


def get_disc_params(r,m, m_dot, alpha, args):
    R12, R23, R24, R34, R2Q, R3Q, R4Q = args
    X=0.7381
    Y=0.2485
    Z=0.0134
    kappa_es=0.2*(1+X)
    m_dot = m_dot * (1-r**-0.5)
    if r<=R12 and r<=R1Q:
        zone=1
        rho = 3.29e-14* alpha **-1 * m**-1 * m_dot**-2 * r**1.5 
        H = 1.3e14 * m * m_dot
        cs = 1.8e10 * m_dot * r**-1.5
        P = 1.065e7 * alpha**-1 * m**-1 * r**-1.5
        Sigma = 8.574 * alpha**-1 * m_dot**-1 * r**1.5 
        T = 254920 * alpha**-0.25 * m_dot**-0.25 * r**(-3/8) 
        kappa=kappa_es
    if R12 <= r <= R23 and r<=R2Q:
        zone=2
        rho = 7.495e-6 * alpha**-0.7 * m**-0.7 * m_dot**(2/5) * r**(-33/20)
        H = 2.13e11 * m**0.9 * m_dot**0.2 * r**(21/20) * alpha**-0.1
        cs = 2.945e7 * m**-0.1 *alpha**-0.1 * r**(-9/20) * m_dot**0.2
        P = 6.5e9 * alpha**-0.9 * m**-0.9 * r**(-51/20) * m_dot**0.8
        Sigma = 3.196e6 * alpha**-0.8 * m**0.2 * m_dot**(3/5) * r**(-3/5)
        T = 6.3e6 * alpha**-0.2 * m**-0.2 * m_dot**(2/5) * r**(-9/10)
        kappa=kappa_es
        
    if R23 <=r <=R34 and r<=R3Q:
        zone=3
        rho =  3.5e-5* alpha**-0.7 * m**-0.7 * m_dot**(11/10) * r**(-15/8)
        H = 1.27e11 * alpha**-0.1 * m**0.9 * m_dot**(3/20) * r**(9/8)
        cs = 1.756e7 * alpha**-0.1 * m**-0.1 * m_dot**(3/20) * r**(-3/8)
        P = 1.09e10 * alpha**-0.9 * m**-0.9 * m_dot**(17/20) * r**(-21/8)
        Sigma = 9e6 * alpha**-0.8 * m**0.2 * m_dot**0.7 * r**(-3/4)
        T = 2.239e6 * alpha**-0.2* m**-0.2 * m_dot**0.3 * r**(-3/4)
        kappa_R = 4e25*(1+X)*Z * rho * T**-3.5
        kappa = kappa_R
        
    if (max(R34,R24) <= r) and r<= R4Q: 
        zone=4
        rho = 4.09e13* alpha**-2.15*m**-2.15*m_dot**2.92*r**-7.62
        H = 1.21e5 * alpha**0.385 * m**1.385 * m_dot**-0.64 * r**3.03
        cs = 16.73 * alpha**0.385 * m**0.385 * m_dot**-0.64 * r**1.54
        P = 1.14e16 *  alpha**-1.385 * m**-1.385 * m_dot**1.64 * r**-4.54
        Sigma = 9.9e18 * alpha**-1.77 * m**-0.77 * m_dot**2.28 * r**-4.58
        T = 2.03e-6 * alpha**0.77* m**0.77 * m_dot**-1.28 * r**3.08
        kappa_H_minus = 1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa=kappa_H_minus
    #  else:
    if r<=R12 and r>=R1Q or (R12 <= r <= R23 and r>R2Q) or (R23 <=r <=R34 and r>R3Q) or ((max(R34,R24) <= r) and r>min(R4Q,R3Q)):
        zone=5
        rho = 4.5e-2 * m **-2 * r**-3
        H = 1.16e10 * alpha**(-1/3) * m**(4/3) * m_dot**(1/3) * r**1.5
        cs = 1.61e6 *alpha**(-1/3) * m**(1/3) * m_dot**(1/3) 
        P = 1.18e11 * alpha**(-2/3) * m**(-4/3) * m_dot**(2/3) * r**-3
        Sigma = 1.06e9 * alpha**(-1/3) * m**(-2/3) * m_dot**(1/3) * r**-1.5
        T = 1.89e4 * alpha**(-2/3)* m**(2/3) * m_dot**(2/3) * (r/R2Q)**(-3/4) / min(1,0.1*Z*Sigma/2)**0.5
        
        kappa_R=4e25*(1+X)*Z * rho * T**-3.5
        kappa_H_minus =  1.1e-25*Z**0.5 * rho**0.5 * T**7.7
        kappa = min(kappa_es, max(0.1*Z,  min(kappa_H_minus, kappa_R))  )
    
    return rho, H, cs, P, Sigma, T, kappa, zone   

args = [R12, R23, R24, R34, R2Q, R3Q, R4Q]
rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness = [[get_disc_params(x,m,m_dot,alpha, args)[i] for x in rs] for i in range(0,8)]
# Hs = [get_disc_params(x,m,m_dot,alpha, args)[1] for x in rs]
#%%
def plot_disc_solution(mm,m_dot,alpha, col):
    R12 = 449.842 * alpha**(2/21) * m**(2/21) * m_dot**(16/21)
    R23 = 987.891 * m_dot**(2/3)
    R24 =  1861.61 * alpha**-0.24 * m**-0.24 * m_dot**0.423
    R34 = 1908.43 * alpha**(-0.253) * m**(-0.253) * m_dot**0.413
    R1Q = 498 * alpha**(2/9) * m**(-2/9) * m_dot **(4/9)
    R2Q = 634.4 * alpha**(14/27) * m**(-26/27) * m_dot**(-8/27)
    R3Q = 580.65 * alpha**(28/45) * m**(-52/45) * m_dot**(-22/45)
    R4Q = 2550.56 * alpha**(-0.467) * m**(-0.033) * m_dot**(0.63333)

    args = [R12, R23, R24, R34, R2Q, R3Q, R4Q]
    rhos, Hs, css, Ps, Sigmas, Ts, kappas, zoness = [[get_disc_params(x,m,m_dot,alpha, args)[i] for x in rs] for i in range(0,8)]
    taus = [x*y/2 for (x,y) in zip(kappas, Sigmas)]

    rgs = [6 * r for r in rs]
    r_g = 6.674e-8*1e8*2e33*mm/3e10**2
    which_prefactor = 'GS21'
#    which_prefactor = 'JM17_lin_tot' 
    #which_prefactor = 'JM17_lin_iso'
#    print (r_ab, r_bc, r_Q, r_tau1)
    #print("--- %s seconds ---" % (time.time() - start_time))
    if fig_1_flag:
        plt.figure(1, figsize=(12,25))
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.99, top=0.98, wspace=0.21, hspace=0.01)
        plt.subplot(421)  
        plt.plot(np.log10(R_shmuel/r_isco), np.log10(rho_shmuel))

  #      plt.plot(np.log10(rgs), [np.log10(rho(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(rhos), linewidth=3, color=col, linestyle='solid'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \rho\  \rm [g\ cm^{-3}]$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-17.5,-3.4])
        plt.yticks([-16,-14,-12,-10,-8,-6, -4])
        
        plt.subplot(422)  
        plt.plot(np.log10(R_shmuel/r_isco), np.log10(P_shmuel))

        plt.plot(np.log10(rgs), np.log10(Ps), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
 #  plt.plot(np.log10(rs), [np.log10(tau_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.ylabel(r'$\log Q_{\rm T}$')
     #   plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-4,-2,0,2,4,6,8])
        plt.ylim([-4.5,9.2])
        
        plt.subplot(423)  
        #plt.plot(np.log10(rs), [np.log10(Sigma(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(R_shmuel/r_isco), np.log10(Sigma_shmuel))
        plt.plot(np.log10(rgs), np.log10(Sigmas), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \Sigma \  \rm [g\ cm^{-2}]$')
        if mm==1:
            plt.text(1.1, 1, r'$\dot{m} = $' + str(m_d), color='black', size=26)
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0,2,4,6])
        plt.ylim([-2.5,7])
        
        plt.subplot(424)  
   #     plt.plot(np.log10(R_shmuel/r_isco), np.log10(rho_shmuel))
    #   plt.plot(np.log10(rs), [np.log10(Q(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(taus), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.ylabel(r'$\log \tau$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0, 2,4,6])
        plt.ylim([-2.8,6.6])

        plt.subplot(425)  
        h_over_r = [h/ x / r_g for (x,h) in zip(rs,Hs)] 
 #       plt.plot(np.log10(R_shmuel/r_isco), np.log10(H_shmuel/R_shmuel))
        plt.plot(np.log10(R_shmuel/r_isco), np.log10(H_shmuel/R_shmuel))

        plt.plot(np.log10(rgs), [np.log10(x/r/(G*msun*1e8*mm/c/c)) for (x,r) in zip(Hs,rgs)], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
    #    plt.plot(np.log10(rgs), np.log10(h_over_r), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log H/r$')
        plt.xticks([1,2,3,4,5,6])
      #  plt.ylim([-2.1,-0.3])
    #    plt.yticks([-2, -1.5, -1, -0.5])
                
        plt.subplot(426) 
        plt.plot(np.log10(R_shmuel/r_isco), np.log10(T_shmuel))

        plt.plot(np.log10(rgs), np.log10(Ts), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log T_c\ \rm[K] $')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([3,4,5,6])

        plt.subplot(427)  
        #css = [c_s_Q(x,mm, m_dot, alpha, r_ab, r_bc, r_Q) for x in rs] 
        plt.plot(np.log10(R_shmuel/r_isco), np.log10(cs_shmuel))

        # plt.plot(np.log10(rs), [np.log10(h(x,mm,m_dot,alpha)/x/r_g) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(css), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log c_s\ \rm [cm\ s^{-1}]$')
        plt.xlabel(r'$\log r\ \rm{[r_g]$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([5.5,8.8])
        plt.yticks([6,7,8])
               
        plt.subplot(428)  
        plt.plot(np.log10(rgs), np.log10(kappas), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \kappa\ \rm[cm^2 \ g^{-1}] $')
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        plt.text(1.1, -1.5+0.4*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xticks([1,2,3,4,5,6])


    if fig_2_flag:
        plt.figure(2, figsize=(10,5))
        plt.subplots_adjust(left=0.05, bottom=0.16, right=0.99, top=0.99, wspace=0.16)
        l_ratio = [c / kappa_metzger17_tau1(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) / rho_Q(r, mm, m_dot, alpha, r_ab, r_bc, r_Q) / chi(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * gamma for r in rs]
        print('lll= ', l_ratio[0])

        plt.subplot(121)  
        plt.plot(np.log10(rgs), [np.log10(chi(x,mm,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)/h_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)**2/Omega(x,mm,m_dot,alpha)**1) for x in rs], linewidth=3, color=col, label=r'$\log\ \chi/ H^2 \Omega$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(l_ratio), linewidth=3, color=col, linestyle='dashed', label=r'$\log \ L_{\rm Edd}/L_c$'); #plt.xscale('log'); plt.yscale('log')
        #plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-3, -2,-1, 0,1, 2])
        plt.ylim([-3.5,2.2])
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')

        plt.subplot(122)
        x_cor = [x_c(x,mm,m_dot,alpha, r_ab, r_bc, r_Q) for x in rs]
        l_c = [lambda_c(x,mm,m_dot,alpha, r_ab, r_bc,r_Q, r_tau1) for x in rs]
        h =  [h_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q) for x in rs]
        r_h = [r_hill(x,mm, 10) for x in rs]
        plt.plot(np.log10(rgs), np.log10(l_c) - np.log10(h), linewidth=3, color=col, label=r'$\log \lambda/H$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(x_cor) - np.log10(l_c), linewidth=3, color=col, linestyle='dashed', label=r'$\log x_c/\lambda$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(h) - np.log10(r_h), linewidth=3, color=col, linestyle='dotted', label=r'$\log H/r_H$'); #plt.xscale('log'); plt.yscale('log')
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')
  #      plt.ylabel(r'$\log \lambda/h$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-5,-4,-3,-2,-1, 0,1, 2,3,])
        plt.ylim([-5.5, 3.5])
  #      plt.axvline(np.log10(r_Q*6), color=col)
        
     #   plt.subplot(223)  
      #  plt.plot(np.log10(rgs), [np.log10(c_s_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
       # plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        #plt.xlabel(r'$\log r / r_g$')
        #plt.xticks([0,1,2,3,4,5,6])
 #       plt.yticks([-2,-1, 0,1, 2,3, 4,5, 6])
    #    plt.ylim([-2.5,6.5])

    if fig_3_flag:
        plt.figure(3, figsize=(10,5))
     #   plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.99)
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1,  which_prefactor) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        plt.subplot(121)
        plt.plot(np.log10(rgs), np.log10(G_tot), linewidth=3, color=col, label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot_minus), linewidth=3, color=col, linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.2,0.95])
        plt.xlim([0.5,6.3])
  #      if col=='red':
   #         plt.arrow(6,-3.5, -1, 0., head_width=0.15, head_length=0.15,linewidth=3, linestyle='dashed', fc=col, ec=col)

 #       plt.text(1,1,r'$\log \dot{m}$')

        plt.subplot(122)
        plt.plot(np.log10(tt), np.log10(G_tot), linewidth=3, color=col, label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot_minus), linewidth=3, color=col, linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.xlabel(r'$\log \tau$')
        plt.xticks([-1, 0,1,2,3,4,5,6])
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.2,0.95])
        plt.xlim([-1.5,6.3])
        plt.legend()

    if fig_4_flag:
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [x+y for (x,y) in zip(G_I,G_th)]#[Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-x-y for (x,y) in zip(G_I,G_th)]#[-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        rpms = find_rpm(mm,m_dot,alpha,r_ab,r_bc,r_Q, r_tau1)
       
        plt.figure(6, figsize=(10,5))
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        plt.subplot(121)
        
        [plt.axvline(np.log10(6*r),color='purple', linestyle='dashed') for r in rpms]
        plt.plot(np.log10(rgs), np.log10(G_I), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_I_minus), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_th), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_th_minus), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot_minus), linewidth=3, color='blue', linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0])
        plt.ylim([-4.5,0.4])
        plt.xlim([0.5,6.3])
        plt.text(0.6,-0.3, 'log M=' + str(int(np.log10(1e8*mm))), color='black', size=26)
 
        plt.subplot(122)
        if len(rpms)>0:

            tau_pm2 = c*gamma/chi(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)*h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)
            tau_pm3 = c*gamma/chi(rpms[1], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)*h_Q(rpms[1], mm, m_dot, alpha, r_ab, r_bc, r_Q)

            prefactor = Gamma_I(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) * h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)/x_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)
  #          term = np.fabs(prefactor)*gamma/1.61/(gamma-1)
            KK = gamma * lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * prefactor / h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)/1.61/(gamma-1)
            print ('KK= ', Gamma_I(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor), Gamma_thermal(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) )
            tau_tot = tau_pm2 /(1 - KK)

    #
        plt.plot(np.log10(tt), np.log10(G_I), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_I_minus), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_th), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_th_minus), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot_minus), linewidth=3, color='blue', linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log \tau$')
        if len(rpms)>0:

            plt.axvline(np.log10(tau_pm2), color='purple',linestyle='dashed')
            plt.axvline(np.log10(tau_pm3), color='purple',linestyle='dashed')
            plt.axvline(np.log10(tau_tot), color='blue',linestyle='dashed')

    #    [plt.axvline(np.log10(t),color=col) for t in taus_pm]
  
        plt.xticks([-1, 0,1,2,3,4,5,6])
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.5,0.4])
        plt.xlim([-1.5,6.3])
        plt.legend()
        
fig_1_flag = 1
fig_2_flag = 0
fig_3_flag = 0
fig_4_flag = 0
test_flag = 0
#assin args: [m (1e8 msun), m_dot (m_cr), alpha, color]
m_d=0.1; alp=1e-2
#args1 = [1e1,m_d,alp, 'orange']
args2 = [1e0,m_d,alp, 'red']
#args3 = [1e-1,m_d,alp, 'blue']
#args4 = [1e-2,m_d,alp, 'green']

#plot_disc_solution(*args1)
plot_disc_solution(*args2)
#plot_disc_solution(*args3)
#plot_disc_solution(*args4)



# zones = [get_disc_params(x,m,m_dot,alpha, args)[1] for x in rs]

# plt.plot(np.log10(R_shmuel/r_isco), np.log10(rho_shmuel)); 
# plt.plot(np.log10(6*rs), np.log10(rhos))
# #plt.plot(np.log10(6*rs), zones)

# plt.axvline(np.log10(6*R12), color='red')
# plt.axvline(np.log10(6*R23), color='red', linestyle='dashed')
# plt.axvline(np.log10(6*R24), color='blue')
# plt.axvline(np.log10(6*R34), color='blue', linestyle='dashed')

# plt.axvline(np.log10(6*R1Q), color='grey')
# plt.axvline(np.log10(6*R2Q), color='grey', linestyle='dashed')
# plt.axvline(np.log10(6*R3Q), color='green')
# plt.axvline(np.log10(6*R4Q), color='green', linestyle='dashed')


#%%
# log space in units of r_g

rs = np.logspace(0.01, 6,1000)

# find transition of different zones
def ff(x,A,p):
    return x - A*(1-x**-0.5)**p

def find_r(m, m_dot, alpha,p):
    if p==16/21:
        A = 867 * (alpha * m )**(2/21) * m_dot **(16/21)
    elif p==2/3:
        A = 6300 * m_dot**(2/3)
    sol = root(ff, A, args=(A, p)).x
    return sol

def find_rab(m, m_dot,alpha):
    return 450*alpha**(2/21)*m**(2/21)* m_dot**(16/21)

def find_rbc(m, m_dot, alpha):
    return 988*m_dot**(2/3)


    
#%%

def rho(r, m, m_dot, alpha, r_ab, r_bc):
    f = 1 - r**(-0.5)
    fm = f * m_dot
    rho_00 = 3.33e-14 * alpha**(-1) *m**-1* fm **(-2) * r_ab**(3/2)
    rho_11 = rho_00 * (r_bc/r_ab)**(-33/20)

    if r<=r_ab:
        return 3.33e-14 * alpha**(-1) *m**-1* fm **(-2) * r**(3/2)
    elif r_ab < r <= r_bc:
        return rho_00 *  (r/r_ab)**(-33/20)
    elif r > r_bc:
        return rho_11 * (r/r_bc)**(-15/8)

def h(r,m,m_dot,alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
    fm = f * m_dot
    if r<=r_ab:
        return 1.294e14 * fm**1 * m**1
    elif r_ab < r <= r_bc:
        return 1.294e14  * fm**1 * m**1 * (r/r_ab)**(21/20)
    elif r > r_bc:
        return 1.294e14 * fm**1 * m**1*  (r_bc/r_ab)**(21/20) * (r/r_bc)**(9/8) 
    return value_to_return

def r_g(m):
    return G * m *1e8 * msun / c**2

def Omega(r, mm, m_dot,alpha):
    return 2.03e-3 / mm / r**1.5 * 6**-1.5

def Q(r,m,m_dot,alpha,r_ab, r_bc):
    return h(r,m,m_dot,alpha, r_ab, r_bc) * Omega(r,m,m_dot,alpha) **2 / np.pi / G / Sigma(r,m,m_dot,alpha,r_ab, r_bc)

    x= np.logspace(1,6,1000)
    for i in range(0, len(x)):
        Qi = Q(x[i],m,m_dot,alpha, r_ab, r_bc)
        if Qi>1:
            continue
        else:
            return x[i]#return rs[i]
    return 1e20

def T_c(r,m,m_dot, alpha, r_ab, r_bc):
    f = 1 - r**(-0.5)
    fm = f * m_dot
    T_c_00 =  2.55e5 * alpha**(-1/4) *m**(-1/4)* fm **(0) * r_ab**(-3/8)
    T_c_11 = T_c_00 * (r_bc/r_ab)**(-9/10)
    if r<=r_ab:
        return 2.55e5 * alpha**(-1/4) *m**(-1/4)* fm **(0) * r**(-3/8)
    elif r_ab < r <= r_bc:
        return T_c_00 * (r/r_ab)**(-9/10)
    elif r > r_bc:
        return T_c_11 *  (r/r_bc)**(-3/4)

def c_s(r,m,m_dot,alpha, r_ab, r_bc):
    return h(r,m,m_dot, alpha, r_ab, r_bc) * Omega(r,m,m_dot,alpha) 

def find_rQ(m,m_dot, alpha, r_ab, r_bc):
    x= np.logspace(1,6,1000)
    for i in range(0, len(x)):
        Qi = Q(x[i],m,m_dot,alpha, r_ab, r_bc)
        if Qi>1:
            continue
        else:
            return x[i]#return rs[i]
    return 1e20

def Sigma_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q):
    Sigma0 = Sigma(r_Q, m, m_dot, alpha, r_ab, r_bc)
    if r <= r_Q:
        return Sigma(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_Q:
        return Sigma0 * (r/r_Q)**-1.5
def h_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q):
    h0 = h(r_Q, m, m_dot, alpha, r_ab, r_bc)
    if r <= r_Q:
        return h(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_Q:
        return h0 * (r/r_Q)**1.5

def rho_Q(r,m,m_dot,alpha,  r_ab, r_bc, r_Q):
    rho0 = rho(r_Q, m, m_dot, alpha, r_ab, r_bc)
    if r <= r_Q:
        return rho(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_Q:
        return rho0 * (r/r_Q)**-3

def c_s_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q):
    return h_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q) * Omega(r,m,m_dot,alpha)

def kappa_metzger17(r,m,m_dot, alpha, r_ab, r_bc, r_Q):
    rho = rho_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q)
    T = T_c(r,m,m_dot, alpha, r_ab, r_bc)
   
    X=0.74
    Z=0.02
    
    kappa_m = 0.1 * Z
    kappa_H_minus = 1.1e-25*Z**0.5*rho**0.5*T**7.7
    kappa_ff = 4e25*Z*(1+X)*rho*T**-3.5
    kappa_e = 0.2*(1+X)
    
    return min(kappa_e, kappa_m + (kappa_H_minus**-1 + (kappa_e + kappa_ff)**-1)**-1)

def tau(r,m,m_dot, alpha, r_ab, r_bc, r_Q):
    return kappa_metzger17(r,m,m_dot,alpha, r_ab, r_bc, r_Q) * Sigma_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q)/2

def find_tau_1(rs, m,m_dot, alpha, r_ab, r_bc, r_Q):
    rs=rs
    taus = [tau(r, m, m_dot, alpha, r_ab, r_bc, r_Q) for r in rs]
    for i in range(0, len(taus)):
        if taus[i]>1:
            continue
        elif taus[i]<=1:
            return rs[i]

def T_c_tau1(r,m,m_dot,alpha,  r_ab, r_bc, r_tau1):
  #  r_tau1 = find_tau_1(m, m_dot, alpha)
    T0 = T_c(r_tau1, m, m_dot, alpha,  r_ab, r_bc)
    if r <= r_tau1:
        return T_c(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_tau1:
        return T0
    
def kappa_metzger17_tau1(r,m,m_dot, alpha, r_ab, r_bc, r_Q, r_tau1):
    rho = rho_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q)
    T = T_c_tau1(r,m,m_dot, alpha, r_ab, r_bc, r_tau1)
   
    X=0.74
    Z=0.02
    
    kappa_m = 0.1 * Z
    kappa_H_minus = 1.1e-25*Z**0.5*rho**0.5*T**7.7
    kappa_ff = 4e25*Z*(1+X)*rho*T**-3.5
    kappa_e = 0.2*(1+X)
    
    return min(kappa_e*1, kappa_m + (kappa_H_minus**-1 + (kappa_e + 0*kappa_ff)**-1)**-1)

def tau_updated(r,m,m_dot, alpha, r_ab, r_bc, r_Q, r_tau1):
    return kappa_metzger17_tau1(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1) * Sigma_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q)/2

def p_tot(r,m,m_dot, alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
    fm = f * m_dot
    if r<=r_ab:
        return 1.52e8 * alpha**(-1) *m**(-1)* fm **(0) * r**(-3/2)
    elif r_ab < r <= r_bc:
        return 5.44e11 * alpha**(-9/10) *m**(-17/10)* fm **(4/5) * r**(-51/20)
    elif r > r_bc:
        return 3.87e11 *alpha**(-9/10) * m**(-7/4) * fm **(17/20)* r**(-21/8) 
    return value_to_return

def pressure_gradient(r, r_ab, r_bc, r_Q):
    if r<=r_ab and r< r_Q:
        return -3/2
    elif r_ab < r <= r_bc  and r< r_Q:
        return -51/20
    elif r > r_bc  and r< r_Q:
        return -21/8
    elif r > r_Q:
        return -3

def density_gradient_Q(r, r_ab, r_bc, r_Q):
    if r<=r_ab:
        return 3/2
    elif r_ab < r <= r_bc:
        return -33/20
    elif r > r_bc:
        return -15/8
    if r >= r_Q:
        return -3

def surface_density_gradient_Q(r, r_ab, r_bc, r_Q):
    if r<=r_ab and  r < r_Q:
        return -1.5
    elif r_ab < r <= r_bc and  r < r_Q:
        return 3/5 
    elif r > r_bc and  r < r_Q:
        return 3/4
    if r >= r_Q:
        return 1.5

def temperature_gradient_Q(r, r_ab, r_bc, r_tau1):
    if r<=r_ab and  r < r_tau1:
        return 3/8
    elif r_ab < r <= r_bc and  r < r_tau1:
        return 9/10 
    elif r > r_bc and  r < r_tau1:
        return 3/4
    if r >= r_tau1:
        return 0

def chi(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1):
    hq = h_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)
    Om = Omega(r, m, m_dot, alpha)
    T = T_c_tau1(r,m,m_dot,alpha, r_ab, r_bc, r_tau1) #note different r_tau1!
    rho0 = rho_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q)
    kappa = kappa_metzger17_tau1(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)
    cs = c_s_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)
    SS=Sigma_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)
    term1 = sb * T**4 * 4 /3 / c + 1.38e-16/1.67e-24 * T * rho0
    term2 = rho0 * cs**2
 #   term2 = cs**2 * SS * 3 * np.pi * alpha / Om/ (1-r**-0.5)
 #   term1 = 1.38e-16/1.67e-24 * T / cs**2
    term3 = 2 * kappa * alpha * rho0**2 * Om**3 * hq**4
  #  if r<r_Q:
   #     print ('test:',r, term1, term2, term1/term2)
 #   val_to_return = 9/8* alpha * SS * Om / rho0**2
   # print (gamma)
    val_to_return = 16 * gamma * (gamma-1) * sb * T**4 / 3 / kappa/ rho0**2 /hq**2 / Om**2# cs**2 #%%
    return val_to_return

def lambda_c(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1):
    chi0 = chi(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)
    Om = Omega(r,m,m_dot,alpha)
    return (2 * chi0 / 3/gamma/Om)**0.5

def x_c(r,m, m_dot,alpha, r_ab,r_bc,r_Q):
    return -pressure_gradient(r,r_ab,r_ab, r_Q) * c_s_Q(r,m,m_dot,alpha, r_ab,r_bc, r_Q)**2/3/r/r_g(m) / gamma / Omega(r,m,m_dot,alpha)**2

def r_hill(r, m, stellar_m):
    return r  *r_g(m)* (stellar_m/1e8/m/3)**(1/3)

def Gamma_0(r_0,m_stellar_bh,m, m_dot, alpha, r_ab, r_bc):
    q= m_stellar_bh / 1e8/m
    SS = Sigma_Q(r_0, m, m_dot, alpha, r_ab, r_bc)
    rg = r_g(m)
    Om =  Omega(r_0, m, m_dot, alpha, r_ab, r_bc)
    h_over_r = h_Q(r_0, m, m_dot, alpha, r_ab, r_bc)/r_0/rg
    return  SS * (r_0 * rg) **4 * Om**2 * q**2 * h_over_r**-3

# EQ. 1 from botsch and kley 2011
#def Gamma_I_prefactor(alpha, beta):
 #   xi = beta - (gamma-1)* alpha
  #  return -2.5 - 1.7*beta + 0.1*alpha + 1.1 * (1.5-alpha) + 7.9 * xi / gamma

# EQ. 1 from botsch and kley 2011
#def Gamma_I_prefactor_JM17(alpha, beta):
 #   lin_term = - (2.34 - 0.1*alpha + 1.5*beta) / gamma # Eq. 39 in JM17
  #  lin_general = lin_term  + (0.46 - 0.96*alpha + 1.8*beta)/gamma # EQ. 67 in JM17
   # lin_iso = - (1.36 + 0.54*alpha + 0.5*beta) #Eq 68 in JM17
    #return lin_general

def Gamma_I(r,m,m_dot,alpha,r_ab,r_bc,r_Q,r_tau1, which_prefactor):
    aa = surface_density_gradient_Q(r, r_ab, r_bc, r_Q)
    bb = temperature_gradient_Q(r, r_ab, r_bc, r_tau1)
    prefactor = 0
    if which_prefactor == 'GS21':
        prefactor = -0.8 - bb - 0.9*aa
    elif which_prefactor == 'JM17_lin_tot': # EQ. 39 and 67 in JM17
        prefactor =  - (2.34 - 0.1*aa + 1.5*bb) / gamma  + (0.46 - 0.96*aa + 1.8*bb) / gamma
    elif prefactor == 'JM17_lin_iso':
        prefactor =  -(1.36 + 0.54*aa + 0.5*bb)
#    pre_factor = Gamma_I_prefactor(aa,bb)/gamma
 #   pre_factor_gs=-0.8 - bb - 0.9*aa #eq 15 in gilbaum and stone 2021  
    return prefactor* x_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / h_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)

def Gamma_thermal(r,m,m_dot,alpha,r_ab,r_bc, r_Q, r_tau1):
    kappa = kappa_metzger17_tau1(r, m, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)
    tau_star = tau(r, m, m_dot, alpha, r_ab, r_bc, r_Q)
    l_ratio = c / kappa / rho_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / chi(r, m, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * gamma
    return (1-np.exp(-tau_star*alpha**0.5)) *1.61 * (gamma-1) / gamma * x_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / lambda_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * (l_ratio - 1)
#%%
def find_rpm(m,m_dot,alpha,r_ab,r_bc,r_Q, r_tau1):
    l_ratio = [c / kappa_metzger17(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / rho_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / chi(r, m, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * gamma for r in rs]
  #  print (l_ratio[100])

    signs = [np.sign(l - 1) for l in l_ratio]
    ds = [signs[i+1] - signs[i] for i in range(0,len(rs)-1)]
    r_sol = []
    for i in range(0,len(ds)):
        if ds[i] !=0:
    #        print (rs[i],ds[i])
            r_sol.append(rs[i])
    return r_sol


#%%

def kappa_khatami(r,m,m_dot,alpha, r_ab, r_bc):
    kappa_es = 0.34
    eps=1e-3
    T_ion = 12000
    dT = 2000
    T_term  = (T_c(r,m,m_dot,alpha, r_ab, r_bc) - T_ion)/ dT
    return eps*kappa_es  + (1-eps)*kappa_es / 2 * (1 + np.tanh(T_term) ) 

def get_T(rs, m,md,al,a,b):
    tmp = np.zeros(len(rs))
    for i in range(0, len(rs)):
        tmp[i] = T_c(rs[i],m,md,al,a,b)
    return tmp

def plot_kappa(mm,mdd,al,cc):
    r_ab = find_rab(mm,mdd, al)
    r_bc = find_rbc(mm,mdd, al)
    r_Q = find_rQ( mm, mdd, al, r_ab, r_bc)
    r_tau1 = find_tau_1(rs, mm, mdd, al, r_ab, r_bc, r_Q)

#    Ts = get_T(rs, mm,mdd,al,r_ab,r_bc)
#Ts = [T_c(r,mm,mdd,al, r_ab, r_bc) for r in rs]
#plt.plot(np.log10(rs), [kappa_khatami(r,mm,mdd,al,r_ab,r_bc) for r in rs])
#plt.plot(np.log10(Ts), [kappa_ff(r,mm,mdd,al,r_ab,r_bc, r_Q) for r in rs])
    plt.plot(np.log10(6*rs), [np.log10(kappa_metzger17_tau1(r,mm,mdd,al,r_ab,r_bc, r_Q, r_tau1)) for r in rs], color =cc)
    plt.text(1,-1.3+0.2*np.log(mm), 'log m = ' + str(int(np.log10(1e8*mm))), color=cc)
    plt.ylabel(r'$\log \kappa\  \rm [cm^2\ g^{-1}]$')

    plt.xlabel(r'$\log r / r_g$')

mdd = 0.1; al=0.01
plot_kappa(10,mdd,al,'orange')
plot_kappa(1,mdd,al,'red')
plot_kappa(0.1,mdd,al,'blue')
plot_kappa(0.01,mdd,al,'green')

#%%
def plot_disc_solution(mm,m_dot,alpha, col):
    rgs = [6 * r for r in rs]
    r_ab = find_rab(mm, m_dot, alpha)[0]
    r_bc = find_rbc(mm,m_dot, alpha)[0]
    r_Q = find_rQ( mm, m_dot, alpha, r_ab, r_bc)
    r_tau1 = find_tau_1(rs, mm, m_dot, alpha, r_ab, r_bc, r_Q)
    r_g = 6.674e-8*1e8*2e33*mm/3e10**2
    print(r_ab, r_bc, r_Q, r_tau1)
    which_prefactor = 'GS21'
#    which_prefactor = 'JM17_lin_tot' 
    #which_prefactor = 'JM17_lin_iso'
#    print (r_ab, r_bc, r_Q, r_tau1)
    #print("--- %s seconds ---" % (time.time() - start_time))
    if fig_1_flag:
        plt.figure(1, figsize=(12,25))
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.99, top=0.98, wspace=0.21, hspace=0.01)
        #plt.rcParams['font.family'] = 'Serif'
        plt.subplot(421)  
  #      plt.plot(np.log10(rgs), [np.log10(rho(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), [np.log10(rho_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col, linestyle='solid'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \rho\  \rm [g\ cm^{-3}]$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-17.5,-3.4])
        plt.yticks([-16,-14,-12,-10,-8,-6, -4])
        
        plt.subplot(422)  
        plt.plot(np.log10(rgs), [np.log10(Q(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
 #  plt.plot(np.log10(rs), [np.log10(tau_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.ylabel(r'$\log P\ {\rm dyne\ cm^{-2}}$')
     #   plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-4,-2,0,2,4,6,8])
        plt.ylim([-4.5,9.2])
        
        plt.subplot(423)  
        #plt.plot(np.log10(rs), [np.log10(Sigma(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), [np.log10(Sigma_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \Sigma \  \rm [g\ cm^{-2}]$')
        if mm==1:
            plt.text(1.1, 1, r'$\dot{m} = $' + str(m_d), color='black', size=26)
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0,2,4,6])
        plt.ylim([-2.5,7])
        
        plt.subplot(424)  
 #   plt.plot(np.log10(rs), [np.log10(Q(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), [np.log10(tau_updated(x,mm,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.ylabel(r'$\log \tau$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,0, 2,4,6])
        plt.ylim([-2.8,6.6])

        plt.subplot(425)  
        h_over_r = [h_Q(x,mm, m_dot, alpha, r_ab, r_bc, r_Q) / x / r_g for x in rs] 
        # plt.plot(np.log10(rs), [np.log10(h(x,mm,m_dot,alpha)/x/r_g) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(h_over_r), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log H/r$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([-2.1,-0.3])
        plt.yticks([-2, -1.5, -1, -0.5])
                
        plt.subplot(426)  
        plt.plot(np.log10(rgs), [np.log10(T_c_tau1(x,mm,m_dot,alpha, r_ab, r_bc, r_tau1)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log T_c\ \rm[K] $')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([3,4,5,6])

        plt.subplot(427)  
        css = [c_s_Q(x,mm, m_dot, alpha, r_ab, r_bc, r_Q) for x in rs] 
        # plt.plot(np.log10(rs), [np.log10(h(x,mm,m_dot,alpha)/x/r_g) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(css), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log c_s\ \rm [cm\ s^{-1}]$')
        plt.xlabel(r'$\log r\ \rm{[r_g]$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([5.5,8.8])
        plt.yticks([6,7,8])
               
        plt.subplot(428)  
        plt.plot(np.log10(rgs), [np.log10(kappa_metzger17(x,mm,m_dot,alpha, r_ab, r_bc, r_tau1)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \kappa\ \rm[cm^2 \ g^{-1}] $')
        plt.xlabel(r'$\log r\ \rm{[r_g]}$')
        plt.text(1.1, -1.5+0.4*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xticks([1,2,3,4,5,6])


    if fig_2_flag:
        plt.figure(2, figsize=(10,5))
        plt.subplots_adjust(left=0.05, bottom=0.16, right=0.99, top=0.99, wspace=0.16)
        l_ratio = [c / kappa_metzger17_tau1(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) / rho_Q(r, mm, m_dot, alpha, r_ab, r_bc, r_Q) / chi(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * gamma for r in rs]
        print('lll= ', l_ratio[0])

        plt.subplot(121)  
        plt.plot(np.log10(rgs), [np.log10(chi(x,mm,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)/h_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)**2/Omega(x,mm,m_dot,alpha)**1) for x in rs], linewidth=3, color=col, label=r'$\log\ \chi/ H^2 \Omega$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(l_ratio), linewidth=3, color=col, linestyle='dashed', label=r'$\log \ L_{\rm Edd}/L_c$'); #plt.xscale('log'); plt.yscale('log')
        #plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-3, -2,-1, 0,1, 2])
        plt.ylim([-3.5,2.2])
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')

        plt.subplot(122)
        x_cor = [x_c(x,mm,m_dot,alpha, r_ab, r_bc, r_Q) for x in rs]
        l_c = [lambda_c(x,mm,m_dot,alpha, r_ab, r_bc,r_Q, r_tau1) for x in rs]
        h =  [h_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q) for x in rs]
        r_h = [r_hill(x,mm, 10) for x in rs]
        plt.plot(np.log10(rgs), np.log10(l_c) - np.log10(h), linewidth=3, color=col, label=r'$\log \lambda/H$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(x_cor) - np.log10(l_c), linewidth=3, color=col, linestyle='dashed', label=r'$\log x_c/\lambda$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(h) - np.log10(r_h), linewidth=3, color=col, linestyle='dotted', label=r'$\log H/r_H$'); #plt.xscale('log'); plt.yscale('log')
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')
  #      plt.ylabel(r'$\log \lambda/h$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.yticks([-5,-4,-3,-2,-1, 0,1, 2,3,])
        plt.ylim([-5.5, 3.5])
  #      plt.axvline(np.log10(r_Q*6), color=col)
        
     #   plt.subplot(223)  
      #  plt.plot(np.log10(rgs), [np.log10(c_s_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
       # plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        #plt.xlabel(r'$\log r / r_g$')
        #plt.xticks([0,1,2,3,4,5,6])
 #       plt.yticks([-2,-1, 0,1, 2,3, 4,5, 6])
    #    plt.ylim([-2.5,6.5])

    if fig_3_flag:
        plt.figure(3, figsize=(10,5))
     #   plt.subplots_adjust(left=0.15, bottom=0.2, right=0.99, top=0.99)
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1,  which_prefactor) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        plt.subplot(121)
        plt.plot(np.log10(rgs), np.log10(G_tot), linewidth=3, color=col, label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot_minus), linewidth=3, color=col, linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.2,0.95])
        plt.xlim([0.5,6.3])
  #      if col=='red':
   #         plt.arrow(6,-3.5, -1, 0., head_width=0.15, head_length=0.15,linewidth=3, linestyle='dashed', fc=col, ec=col)

 #       plt.text(1,1,r'$\log \dot{m}$')

        plt.subplot(122)
        plt.plot(np.log10(tt), np.log10(G_tot), linewidth=3, color=col, label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot_minus), linewidth=3, color=col, linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.xlabel(r'$\log \tau$')
        plt.xticks([-1, 0,1,2,3,4,5,6])
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.2,0.95])
        plt.xlim([-1.5,6.3])
        plt.legend()

    if fig_4_flag:
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [x+y for (x,y) in zip(G_I,G_th)]#[Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-x-y for (x,y) in zip(G_I,G_th)]#[-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        rpms = find_rpm(mm,m_dot,alpha,r_ab,r_bc,r_Q, r_tau1)
       
        plt.figure(6, figsize=(10,5))
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.99, top=0.99, wspace=0.15)
        plt.subplot(121)
        
        [plt.axvline(np.log10(6*r),color='purple', linestyle='dashed') for r in rpms]
        plt.plot(np.log10(rgs), np.log10(G_I), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_I_minus), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_th), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_th_minus), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rgs), np.log10(G_tot_minus), linewidth=3, color='blue', linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([0,1,2,3,4,5,6])
        plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0])
        plt.ylim([-4.5,0.4])
        plt.xlim([0.5,6.3])
        plt.text(0.6,-0.3, 'log M=' + str(int(np.log10(1e8*mm))), color='black', size=26)
 
        plt.subplot(122)
        if len(rpms)>0:

            tau_pm2 = c*gamma/chi(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)*h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)
            tau_pm3 = c*gamma/chi(rpms[1], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)*h_Q(rpms[1], mm, m_dot, alpha, r_ab, r_bc, r_Q)

            prefactor = Gamma_I(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor) * h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)/x_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)
  #          term = np.fabs(prefactor)*gamma/1.61/(gamma-1)
            KK = gamma * lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * prefactor / h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)/1.61/(gamma-1)
            print ('KK= ', Gamma_I(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1, which_prefactor), Gamma_thermal(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) )
            tau_tot = tau_pm2 /(1 - KK)

    #
        plt.plot(np.log10(tt), np.log10(G_I), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_I_minus), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_th), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_th_minus), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot_minus), linewidth=3, color='blue', linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log \tau$')
        if len(rpms)>0:

            plt.axvline(np.log10(tau_pm2), color='purple',linestyle='dashed')
            plt.axvline(np.log10(tau_pm3), color='purple',linestyle='dashed')
            plt.axvline(np.log10(tau_tot), color='blue',linestyle='dashed')

    #    [plt.axvline(np.log10(t),color=col) for t in taus_pm]
  
        plt.xticks([-1, 0,1,2,3,4,5,6])
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.5,0.4])
        plt.xlim([-1.5,6.3])
        plt.legend()
        
    if test_flag:
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th_m = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]

        plt.figure(5)
  #      print(c_s_Q(r_Q, mm, m_dot, alpha, r_ab, r_bc, r_Q), c_s_Q(10*r_Q, mm, m_dot, alpha, r_ab, r_bc, r_Q))
 #       plt.plot(np.log10(rgs), np.log10[(c_s_Q(r,mm,m_dot,alpha, r_ab, r_bc, r_Q) for r in rs]))
        plt.plot(np.log10(rgs), [np.log10(c_s_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
      #  plt.plot(np.log10(tt), np.log10(G_th_m))

    #    plt.text(0.5, -3.5-0.6*np.log10(mm), 'log m= ' + str(int(np.log10(1e8*mm))), color=col)
        if mm==0.1:
            plt.text(0.9, 0.6, r'$\dot{m} = $' +str(m_dot), color='black')
      #      plt.text(1.1, -0.1, 'old type-I C') 
    plt.show()
#%%
fig_1_flag = 0
fig_2_flag = 0
fig_3_flag = 1
fig_4_flag = 0
test_flag = 0
#assin args: [m (1e8 msun), m_dot (m_cr), alpha, color]
m_d=0.1; alp=1e-2
args1 = [1e1,m_d,alp, 'orange']
args2 = [1e0,m_d,alp, 'red']
args3 = [1e-1,m_d,alp, 'blue']
args4 = [1e-2,m_d,alp, 'green']

plot_disc_solution(*args1)
plot_disc_solution(*args2)
plot_disc_solution(*args3)
plot_disc_solution(*args4)
