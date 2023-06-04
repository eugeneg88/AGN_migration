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
#plt.rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 20})
#import

#constants
G=6.674e-8
msun = 2e33
au = 1.5e13
#mbh = 1e8*msun
#m=10*msun
c=3e10
sb=5.6e-5
#rs0 = 2 * G * mbh / c / c
gamma = 5/3


rs = np.logspace(1,6,1000)

def p_ratio1(r,m, m_dot, alpha):
    f = 1 - r**(-0.5)
    fm = f * m_dot
    return 1.31e-9*alpha**-0.25 * m**(7/4) *fm**-2 * r**(21/8)
def p_ratio2(r,m, m_dot, alpha):
    f = 1 - r**-0.5
  #  print(r, f)
    fm = f * m_dot
    return 2.8e-4*alpha**-0.1 * m**0.7 * fm**(-4/5) * r**(21/20)
#%%
def find_rab(m, m_dot, alpha):
    x_range = np.logspace(1,5,10000)
    p_diff = [np.sign(p_ratio1(x,m, m_dot, alpha)- p_ratio2(x,m, m_dot, alpha)) for x in x_range]
    for i in range(0,len(x_range)-1):
        dp = p_diff[i+1] - p_diff[0]
        if dp!=0:
  #          print(i, x_range[i])
            break
  #  print (p_diff)
        if i==len(x_range)-2:
            print('not found!')
            return -1
    return x_range[i]
#approximate solution
def find_rbc(m, m_dot, alpha):
    return 2.2e5 * m**-0.66667 * m_dot**0.666667 

def Sigma(r,m,m_dot,alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
 #   r_ab = find_rab(m, m_dot, alpha)
  #  r_bc = find_rbc(m, m_dot, alpha)
    
    fm = f * m_dot
    if r<=r_ab:
        return 0.927 * alpha**-1 *m**0* fm**-1 * r**1.5 * (1-r**-0.5)**-1 #evgeni - changed
    elif r_ab < r <= r_bc:
        return 1.19e7 * alpha**(-4/5) *m**(-2/5)* fm **(3/5) * r**(-3/5) 
    elif r > r_bc:
        return 1.8e7 * alpha**(-4/5) * fm **(7/10)* m**(-1/2) * r**(-3/4) 
    return value_to_return

def h(r,m,m_dot,alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
   # r_ab = find_rab(m, m_dot, alpha)
    #r_bc = find_rbc(m, m_dot, alpha)

    fm = f * m_dot
    if r<=r_ab:
        return 1.59e14 * fm**1 * m**1
    elif r_ab < r <= r_bc:
        return 4.45e10 * alpha**(-1/10) *m**(7/10)* fm **(1/5) * r**(21/20)
    elif r > r_bc:
        return 2.09e10 *alpha**(-1/10) * m**(3/4) * fm **(3/20)* r**(9/8) 
    return value_to_return

def Omega(r, mm, m_dot,alpha):
    return 2.03e-3 / mm / r**1.5

def Q(r,m,m_dot,alpha,r_ab, r_bc):
    return h(r,m,m_dot,alpha, r_ab, r_bc) * Omega(r,m,m_dot,alpha) **2 / np.pi / G / Sigma(r,m,m_dot,alpha,r_ab, r_bc)
def find_rQ(m,m_dot, alpha, r_ab, r_bc):
    x= np.logspace(1,6,1000)
    for i in range(0, len(x)):
        Qi = Q(x[i],m,m_dot,alpha, r_ab, r_bc)
   #     print(i, Qi)
        if Qi>1:
            continue
        else:
            return x[i]#return rs[i]
    return 1e20

def find_tau_1(rs, m,m_dot, alpha, r_ab, r_bc, r_Q):
    rs=rs
    taus = [tau_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q) for r in rs]
 #   print (taus)
    for i in range(0, len(taus)):
        if taus[i]>1:
            continue
        elif taus[i]<=1:
            return rs[i]
    return 1e20


def r_g(m):
    return G * m *1e8 * msun / c**2
#print (find_rQ(rs, 1e-3,0.0006,0.01))
def rho(r, m, m_dot, alpha, r_ab, r_bc):
 #   m_h = 1.67e-24
    value_to_return=0
    f = 1 - r**(-0.5)
    fm = f * m_dot
    if r<=r_ab:
        return 2.91e-15 * alpha**(-1) *m**-1* fm **(-2) * r**(3/2)
    elif r_ab < r <= r_bc:
        rho_00 = 2.91e-15 * alpha**(-1) *m**-1* fm **(-2) * r_ab**(3/2)
#        return 133e-6 * alpha**(-7/10) *m**(-11/10)* fm **(2/5) * r**(-33/20)
        return rho_00 *  (r/r_ab)**(-33/20)
    elif r > r_bc:
        return 432e-6 *alpha**(-7/10) * m**(-5/4) * fm **(11/20)* r**(-15/8)
    return value_to_return

def p_tot(r,m,m_dot, alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
  #  r_ab = find_rab(m, m_dot, alpha)
   # r_bc = find_rbc(m, m_dot, alpha)
    
    fm = f * m_dot
    if r<=r_ab:
        return 1.52e8 * alpha**(-1) *m**(-1)* fm **(0) * r**(-3/2)
    elif r_ab < r <= r_bc:
        return 5.44e11 * alpha**(-9/10) *m**(-17/10)* fm **(4/5) * r**(-51/20)
    elif r > r_bc:
        return 3.87e11 *alpha**(-9/10) * m**(-7/4) * fm **(17/20)* r**(-21/8) 
    return value_to_return

def pressure_gradient_n(r, r_ab, r_bc):
    if r<=r_ab:
        return -3/2
    elif r_ab < r <= r_bc:
        return -51/20
    elif r > r_bc:
        return -21/8
  #  return value_to_return
    

def c_s(r,m,m_dot,alpha, r_ab, r_bc):
#    return (p_tot(r,m,m_dot,alpha) / rho(r,m,m_dot,alpha) )**0.5 #/ min(1, Q(r,m,m_dot, alpha) )**0.333
    return h(r,m,m_dot, alpha, r_ab, r_bc) * Omega(r,m,m_dot,alpha) 

def c_s_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q):
#    return (p_tot(r,m,m_dot,alpha) / rho(r,m,m_dot,alpha) )**0.5 #/ min(1, Q(r,m,m_dot, alpha) )**0.333
    return h_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q) * Omega(r,m,m_dot,alpha)

def T_c(r,m,m_dot, alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
  #  r_ab = find_rab(m, m_dot, alpha)
   # r_bc = find_rbc(m, m_dot, alpha)

    fm = f * m_dot
    if r<=r_ab:
        return 4.96e5 * alpha**(-1/4) *m**(-1/4)* fm **(0) * r**(-3/8)
    elif r_ab < r <= r_bc:
        return 2.96e7 * alpha**(-1/5) *m**(-3/5)* fm **(2/5) * r**(-9/10)
    elif r > r_bc:
        return 6.51e6 *alpha**(-1/5) * m**(-1/2) * fm **(3/10)* r**(-3/4)
    return value_to_return    

def kappa_ff(r,m,m_dot, alpha, r_ab, r_bc):
    return 3.62e22 * rho(r,m,m_dot, alpha, r_ab, r_bc) * T_c(r,m,m_dot, alpha, r_ab, r_bc)**-3.5

def kappa_khatami(r,m,m_dot,alpha, r_ab, r_bc):
    kappa_es = 0.34
    eps=1e-3
    T_ion = 5000
    dT = 500
    T_term  = (T_c(r,m,m_dot,alpha, r_ab, r_bc) - T_ion)/ dT
    return eps*kappa_es  + (1-eps)*kappa_es / 2 * (1 + np.tanh(T_term) ) 
 
def tau(r,m,m_dot, alpha, r_ab, r_bc):
    return kappa_khatami(r,m,m_dot,alpha, r_ab, r_bc) * Sigma(r,m,m_dot, alpha, r_ab, r_bc)

def Sigma_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q):
    Sigma0 = Sigma(r_Q, m, m_dot, alpha, r_ab, r_bc)
    if r <= r_Q:
        return Sigma(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_Q:
        return Sigma0 * (r/r_Q)**-1.5

def h_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q):
 #   r_Q = find_rQ(rs, m, m_dot, alpha)
    h0 = h(r_Q, m, m_dot, alpha, r_ab, r_bc)
    if r <= r_Q:
        return h(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_Q:
        return h0 * (r/r_Q)**1.5

def rho_Q(r,m,m_dot,alpha,  r_ab, r_bc, r_Q):
 #   r_Q = find_rQ(rs, m, m_dot, alpha)
    rho0 = rho(r_Q, m, m_dot, alpha, r_ab, r_bc)
    if r <= r_Q:
        return rho(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_Q:
        return rho0 * (r/r_Q)**-3

def density_gradient_Q(r, r_ab, r_bc, r_Q):
    value_to_return=0
    if r<=r_ab:
        return 3/2
    elif r_ab < r <= r_bc:
        return -33/20
    elif r > r_bc:
        return -15/8
    if r >= r_Q:
        return -3
 #   return value_to_return

def T_c_Q(r,m,m_dot,alpha,  r_ab, r_bc, r_tau1):
  #  r_tau1 = find_tau_1(m, m_dot, alpha)
    T0 = T_c(r_tau1, m, m_dot, alpha,  r_ab, r_bc)
    if r <= r_tau1:
        return T_c(r,m,m_dot,alpha, r_ab, r_bc)
    elif r>=r_tau1:
        return T0

def tau_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q):
    return kappa_khatami(r,m,m_dot,alpha, r_ab, r_bc) * Sigma_Q(r,m,m_dot, alpha, r_ab, r_bc, r_Q)
    
def chi(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1):
    cs = c_s_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q)
    T = T_c_Q(r,m,m_dot,alpha, r_ab, r_bc, r_tau1) #note different r_tau1!
    rho0 = rho_Q(r,m,m_dot,alpha, r_ab, r_bc, r_Q)
    kappa = kappa_khatami(r,m,m_dot,alpha, r_ab, r_bc)
    sigma_sb = 5.6e-5
    return 16 * gamma * (gamma-1) * sigma_sb * T**4 / 3 / kappa / rho0**2 / cs**2  

def lambda_c(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1):
    chi0 = chi(r,m,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)
    Om = Omega(r,m,m_dot,alpha)
    gamma=5/3
    return (2 * chi0 / 3/gamma/Om)**0.5

def Gamma_0(r_0,m_stellar_bh,m, m_dot, alpha, r_ab, r_bc):
    q= m_stellar_bh / 1e8/m
    r_g = 6.674e-8*1e8*2e33*mm/3e10**2
    return q**2 * Sigma(r_0, m, m_dot, alpha, r_ab, r_bc) * (r_0 * r_g)**4 * Omega(r_0, m, m_dot, alpha, r_ab, r_bc)**2 * h(r_0, m, m_dot, alpha, r_ab, r_bc)**-3

def x_c(r,m, m_dot,alpha, r_ab,r_bc,r_Q):
    return -pressure_gradient_n(r,r_ab,r_ab) * c_s(r,m,m_dot,alpha, r_ab,r_bc)**2/3/r/r_g(m) / gamma / Omega(r,m,m_dot,alpha)**2

def r_hill(r, m, stellar_m):
    return r  *r_g(m)* (stellar_m/1e8/m/3)**(1/3)

def Gamma_I(r,m,m_dot,alpha,r_ab,r_bc,r_Q,r_tau1):
    pre_factor = density_gradient_Q(r, r_ab, r_bc, r_Q) #alter than to the correct one
    return pre_factor * x_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / h_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)

def Gamma_thermal(r,m,m_dot,alpha,r_ab,r_bc, r_Q, r_tau1):
    l_ratio = c / kappa_khatami(r, m, m_dot, alpha, r_ab, r_bc) / rho_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / chi(r, m, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * gamma
    return 1.61 * (gamma-1) / gamma * x_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / lambda_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * (l_ratio - 1)
#%%
def plot_disc_solution(mm,m_dot,alpha, col):
    #import time
    #start_time = time.time()
   # mm=1; m_dot = 0.5; alpha=0.01; col = 'green'
    r_ab = find_rab(mm, m_dot, alpha)
    r_bc = find_rbc(mm,m_dot, alpha)
    r_Q = find_rQ( mm, m_dot, alpha, r_ab, r_bc)
    r_tau1 = find_tau_1(rs, mm, m_dot, alpha, r_ab, r_bc, r_Q)
    r_g = 6.674e-8*1e8*2e33*mm/3e10**2
    print (r_ab, r_bc, r_Q, r_tau1)
    #print("--- %s seconds ---" % (time.time() - start_time))
    if True:
        plt.figure(1, figsize=(16,9))
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.99, wspace=0.25, hspace=0.01)
        #plt.rcParams['font.family'] = 'Serif'
        plt.subplot(231)  
        plt.plot(np.log10(rs), [np.log10(rho(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
      #  plt.plot(np.log10(rs), [np.log10(rho_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \rho$')
        plt.text(1, -16-1.5*np.log10(mm), 'log m= ' + str(int(np.log10(1e8*mm))), color=col)
        plt.xticks([1,2,3,4,5,6])
        plt.axvline(np.log10(r_ab), color=col)

        plt.subplot(232)  
        #plt.plot(np.log10(rs), [np.log10(Sigma(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rs), [np.log10(Sigma_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \Sigma$')
        plt.xticks([1,2,3,4,5,6])

        plt.subplot(233)  
 #   plt.plot(np.log10(rs), [np.log10(Q(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rs), [np.log10(tau_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \tau$')
        plt.xticks([1,2,3,4,5,6])

        plt.subplot(234)  
        h_over_r = [h_Q(x,mm, m_dot, alpha, r_ab, r_bc, r_Q) / x / r_g for x in rs] 
        # plt.plot(np.log10(rs), [np.log10(h(x,mm,m_dot,alpha)/x/r_g) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rs), np.log10(h_over_r), linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log h/r$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.subplot(235)  
        plt.plot(np.log10(rs), [np.log10(Q(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
 #  plt.plot(np.log10(rs), [np.log10(tau_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log Q$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.subplot(236)  
    #plt.plot(np.log10(rs), [np.log10(h(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rs), [np.log10(T_c_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_tau1)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel('$\log T_c$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])

    if True:
        plt.figure(2, figsize=(12,4))
        plt.subplots_adjust(left=0.09, bottom=0.2, right=0.99, top=0.99, wspace=0.18)

        plt.subplot(121)  
        plt.plot(np.log10(rs), [np.log10(chi(x,mm,m_dot,alpha, r_ab, r_bc, r_Q, r_tau1)/h_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)**2/Omega(x,mm,m_dot,alpha)**1) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \chi/ h^2 \Omega$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-2,-1, 0,1, 2,3, 4,5, 6])
        plt.ylim([-2.5,6.5])
        plt.subplot(122)
        x_cor = [x_c(x,mm,m_dot,alpha, r_ab, r_bc, r_Q) for x in rs]
        l_c = [lambda_c(x,mm,m_dot,alpha, r_ab, r_bc,r_Q, r_tau1) for x in rs]
        h =  [h_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q) for x in rs]
        r_h = [r_hill(x,mm, 10) for x in rs]
        plt.plot(np.log10(rs), np.log10(l_c) - np.log10(h), linewidth=3, color=col, label=r'$\lambda/h$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rs), np.log10(x_cor) - np.log10(l_c), linewidth=3, color=col, linestyle='dashed', label=r'$x_c/\lambda$'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(rs), np.log10(h) - np.log10(r_h), linewidth=3, color=col, linestyle='dotted', label=r'$h/r_H$'); #plt.xscale('log'); plt.yscale('log')
        if col == 'orange':
            plt.legend()
            plt.axhline(0, color='grey')
        plt.ylabel(r'$\log \lambda/h$')
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.yticks([-5,-4,-3,-2,-1, 0,1, 2,3,])
        plt.ylim([-5.5, 3.5])

    if False:
        plt.figure(3, figsize=(8,6))
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        tt = [tau_Q(r, mm, m_dot, alpha, r_ab, r_bc, r_Q) for r in rs]
        print (min(tt))
        #plt.plot(np.log10(rs), np.log10(G_I), linewidth=3, color='black', alpha=0.6, label='type I'); #plt.xscale('log'); plt.yscale('log')
        #plt.plot(np.log10(rs), np.log10(G_I_minus), linewidth=3, color='black', alpha=0.6, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        #plt.plot(np.log10(rs), np.log10(G_th), linewidth=3, color='purple', alpha=0.3, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        #plt.plot(np.log10(rs), np.log10(G_th_minus), linewidth=3, color='purple', alpha=0.3, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot), linewidth=3, color=col, label='tot'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot_minus), linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.axvline(np.log10(r_Q), color=col)
        plt.legend()

    print('sucess!')
    plt.show()
#%%

#assin args: [m (1e8 msun), m_dot (m_cr), alpha, color]
args1 = [1e1,0.1,1e-2, 'orange']
args2 = [1e0,0.1,1e-2, 'red']
args3 = [1e-1,0.1,1e-2, 'blue']
args4 = [1e-2,0.1,1e-2, 'green']
plot_disc_solution(*args1)
plot_disc_solution(*args2)
plot_disc_solution(*args3)
plot_disc_solution(*args4)

#%%    

#print (r_ab)
  #%%
#%%
plt.subplot(121)  
plt.plot(np.log10(rs), [np.log10(chi(x,mm,m_dot,alpha)/h(x,mm,m_dot,alpha)**2/Omega(x,mm,m_dot,alpha)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
plt.ylabel('chi')
plt.subplot(122)  
plt.plot(np.log10(rs), [np.log10(lambda_c(x,mm,m_dot,alpha)/h(x,mm,m_dot,alpha)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(rs), [np.log10(tau_Q(x,mm,m_dot,alpha)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
plt.ylabel('lambda/h')

#plt.plot(np.log10(rs), [np.log10(Sigma_Q(x,mm,m_dot,alpha)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
#%%
#%%
#derived quantities at the break opint r=1000 rs
plt.plot(np.log10(r3s), [np.log10(L_edd(x)) for x in r3s], linewidth=3, color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(L_edd(x)/L_c(x)-1) for x in r3s], linewidth=3, color='green'); #plt.xscale('log'); plt.yscale('log')
plt.plot(np.log10(r3s), [np.log10(-L_edd(x)/L_c(x)+1) for x in r3s], linewidth=3, color='green', linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')


#%%
from scipy.special import logsumexp
# approximate the saha equation - get the electron fraction    
def get_e_fraction(rho,T):
    param = 1.5*np.log10(T) - np.log10(rho) - 68240/T - 8.394
    if np.isnan(param):
        print(rho, T, param)
    exp_param = 10**logsumexp(param)#2.71828**param
    x = - logsumexp(exp_param)/2 + logsumexp(exp_param)*(1+4/logsumexp(exp_param))**0.5/2
    return logsumexp(x)
print (get_e_fraction(1e-13, 7000))
#%%
def get_kappa2(r,m,mdot,alpha):
    rho0 = rho_Q(r,m,mdot,alpha) / r**(9/8)
    T0 = T_c(r,m,mdot, alpha)
    kappa_es = 0.34 * get_e_fraction(rho0,T0)
    kappa_kramers = 2.4e24 * rho0 * T0**-3.5
    return kappa_es + 1*kappa_kramers

def get_kappa(r,m,mdot,alpha):
    rho0 = rho_Q(r,m,mdot,alpha)
    T0 = T_c(r,m,mdot, alpha)
    kappa_es = 0.34 * get_e_fraction(rho0,T0)
    kappa_kramers = 2.4e24 * rho0 * T0**-3.5
    return kappa_es + 1*kappa_kramers
#%%

mm=1
mdot=0.5
alpha=1e-2

e_fractions = [get_e_fraction(rho(r, mm, mdot, alpha), T_c(r, mm, mdot, alpha)) for r in rs]
kappa = [get_kappa(r,mm,mdot, alpha) for r in rs]
kappa2 = [get_kappa2(r,mm,mdot, alpha) for r in rs]
k_kh = [kappa_khatami(r,mm,mdot,alpha) for r in rs]
taus = [tau(r,mm,mdot,alpha) for r in rs]
#plt.plot(np.log10(rs), np.log10(e_fractions), linewidth=3, color='green'); #plt.xscale('log'); plt.yscale('log')
#plt.plot(np.log10(rs), [np.log10(T_c(r, 1, 1, 1e-2)/1e5) for r in rs], linewidth=3, color='green'); #plt.xscale('log'); plt.yscale('log')
#plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(e_fractions), linewidth=3, color='green'); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(kappa), linewidth=3, color='green', label='log m=' + str(int(np.log10(1e8*mm)))); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(k_kh), linewidth=3, color='grey', label='Khatami'); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(kappa2), linewidth=3,linestyle='--', color='green', label=r'$\rho \propto r^{-3}$'); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, 10*mm, mdot, 1e-2)/1e0) for r in rs], np.log10(kappa), linewidth=3, color='blue', label='log m=' + str(int(np.log10(1e9*mm)))); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(kappa2), linewidth=3,linestyle='--', color='blue'); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, 100*mm, mdot, 1e-2)/1e0) for r in rs], np.log10(kappa), linewidth=3, color='red', label='log m=' + str(int(np.log10(1e10*mm)))); #plt.xscale('log'); plt.yscale('log')
plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(kappa2), linewidth=3,linestyle='--', color='red'); #plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$\log T \ \rm [K]$')
plt.ylabel(r'$\log \kappa \ \rm [cm^2 \ g^{-1}]$')
#plt.plot(np.log10(rs), [np.log10(T_c(r, mm, mdot, alpha)/1e0) for r in rs], linewidth=3, color='green', label='m=1e8'); #plt.xscale('log'); plt.yscale('log')
plt.legend()
#%%
plt.plot([np.log10(T_c(r, mm, mdot, 1e-2)/1e0) for r in rs], np.log10(taus), linewidth=3, color='green', label='log m=' + str(int(np.log10(1e8*mm)))); #plt.xscale('log'); plt.yscale('log')

#%%
N=100
T_grid = np.logspace(2,7,N)
rho_grid = np.logspace(-21,-0,N)

def get_kappa_grid(rho,T):
    kappa_es = 0.4 * get_e_fraction(rho,T)
    kappa_kramers = 2.4e24 * rho * T**-3.5
    return 1*kappa_es + 1*kappa_kramers
    
kappa_grid = np.zeros([N,N])
e_fr = np.zeros([N,N])
for i in range(0,N):
    for j in range(0,N):
        kappa_grid[i,j] = get_kappa_grid(rho_grid[j], T_grid[i])
        e_fr[i,j] = get_e_fraction(rho_grid[j], T_grid[i])
        if e_fr[i,j]<= 1e-10:
            e_fr[i,j] ==1e-10
#kappa_grid = [get_kappa_grid(x,y) for (x,y) in zip(rho_grid, T_grid)]
#%%
fig,ax=plt.subplots(1,1)
cp = ax.contourf(np.log10(rho_grid), np.log10(T_grid), np.log10(kappa_grid), levels=[-4,-3,-2,-1,0,1,2,3])
#cp = ax.contourf(np.log10(rho_grid), np.log10(T_grid), np.log10(e_fr))
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title(r'$\kappa_{\rm tot}\ \  $')
#ax.set_xlabel('x (cm)')
ax.set_ylabel(r'$\log T\  [K]$')
ax.set_xlabel(r'$\log \rho [\rm g\ cm^{-3}]$')
plt.show()

#%%from matplotlib import rcParams
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], label='test')

ax.legend()
plt.show()
