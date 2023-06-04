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
msun = 2e33
au = 1.5e13
c=3e10
sb=5.6e-5
gamma = 5/3

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
    return find_r(m, m_dot, alpha, 16/21)

def find_rbc(m, m_dot, alpha):
    return find_r(m, m_dot, alpha, 2/3)

#%%
def Sigma(r,m,m_dot,alpha, r_ab, r_bc):
    f = 1 - r**(-0.5)
    fm = f * m_dot # change 4.6 to 3.24
    Sigma_00 = 3.24 * alpha**-1 *m**0* fm**-1 * r_ab**1.5
    Sigma_11 = Sigma_00 * (r_bc/r_ab) ** (-3/5) 

    if r<=r_ab:
        return 3.24 * alpha**-1 *m**0* fm**-1 * r**1.5
    elif r_ab < r <= r_bc:
        return Sigma_00 * (r/r_ab)**(-3/5) 
    elif r > r_bc:
        return Sigma_11* (r/r_bc)**(-3/4) 

def rho(r, m, m_dot, alpha, r_ab, r_bc):
    f = 1 - r**(-0.5)
    fm = f * m_dot
    rho_00 = 7.181e-15 * alpha**(-1) *m**-1* fm **(-2) * r_ab**(3/2)
    rho_11 = rho_00 * (r_bc/r_ab)**(-33/20)

    if r<=r_ab:
        return 7.181e-15 * alpha**(-1) *m**-1* fm **(-2) * r**(3/2)
    elif r_ab < r <= r_bc:
        return rho_00 *  (r/r_ab)**(-33/20)
    elif r > r_bc:
        return rho_11 * (r/r_bc)**(-15/8)

def h(r,m,m_dot,alpha, r_ab, r_bc):
    value_to_return=0
    f = 1 - r**(-0.5)
    fm = f * m_dot
    if r<=r_ab:
        return 2.257e14 * fm**1 * m**1
    elif r_ab < r <= r_bc:
        return 2.257e14  * fm**1 * m**1 * (r/r_ab)**(21/20)
    elif r > r_bc:
        return 2.257e14 * fm**1 * m**1*  (r_bc/r_ab)**(21/20) * (r/r_bc)**(9/8) 
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
    T_c_00 =  2.3e5 * alpha**(-1/4) *m**(-1/4)* fm **(0) * r_ab**(-3/8)
    T_c_11 = T_c_00 * (r_bc/r_ab)**(-9/10)
    if r<=r_ab:
        return 2.3e5 * alpha**(-1/4) *m**(-1/4)* fm **(0) * r**(-3/8)
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
def Gamma_I_prefactor(alpha, beta):
    xi = beta - (gamma-1)* alpha
    return -2.5 - 1.7*beta + 0.1*alpha + 1.1 * (1.5-alpha) + 7.9 * xi / gamma

def Gamma_I(r,m,m_dot,alpha,r_ab,r_bc,r_Q,r_tau1):
    aa = surface_density_gradient_Q(r, r_ab, r_bc, r_Q)
    bb = temperature_gradient_Q(r, r_ab, r_bc, r_tau1)
    pre_factor = Gamma_I_prefactor(aa,bb)/gamma
    pre_factor_gs=-0.8 - bb - 0.9*aa #eq 15 in gilbaum and stone 2021
    
    return pre_factor* x_c(r, m, m_dot, alpha, r_ab, r_bc, r_Q) / h_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)

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
        plt.ylim([-18.5,-4.9])
        plt.yticks([-18,-16,-14,-12,-10,-8,-6])
        
        plt.subplot(422)  
        plt.plot(np.log10(rgs), [np.log10(Q(x,mm,m_dot,alpha, r_ab, r_bc)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
 #  plt.plot(np.log10(rs), [np.log10(tau_Q(x,mm,m_dot,alpha, r_ab, r_bc, r_Q)) for x in rs], linewidth=3, color=col, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.axhline(np.log10(1), color='gray')
        plt.ylabel(r'$\log Q_{\rm T}$')
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
        plt.ylim([-1.8,0.1])
        plt.yticks([-1.5, -1, -0.5, 0])
                
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
        plt.xlabel(r'$\log r / r_g$')
        plt.xticks([1,2,3,4,5,6])
        plt.ylim([5.5,8.8])
        plt.yticks([6,7,8])
               
        plt.subplot(428)  
        plt.plot(np.log10(rgs), [np.log10(kappa_metzger17(x,mm,m_dot,alpha, r_ab, r_bc, r_tau1)) for x in rs], linewidth=3, color=col); #plt.xscale('log'); plt.yscale('log')
        plt.ylabel(r'$\log \kappa\ \rm[cm^2 \ g^{-1}] $')
        plt.xlabel(r'$\log r / r_g$')
        plt.text(1.1, -1.5+0.4*np.log10(mm), 'log M= ' + str(int(np.log10(1e8*mm))), color=col, size=26)
        plt.xticks([1,2,3,4,5,6])


    if fig_2_flag:
        plt.figure(2, figsize=(10,5))
        plt.subplots_adjust(left=0.05, bottom=0.16, right=0.99, top=0.99, wspace=0.16)
        l_ratio = [c / kappa_metzger17_tau1(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) / rho_Q(r, mm, m_dot, alpha, r_ab, r_bc, r_Q) / chi(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * gamma for r in rs]
     #   print('lll= ', l_ratio[0])

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
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
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
        G_I = [Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_I_minus = [-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th_minus = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot = [x+y for (x,y) in zip(G_I,G_th)]#[Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)+ Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_tot_minus = [-x-y for (x,y) in zip(G_I,G_th)]#[-Gamma_I(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)- Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        rpms = find_rpm(mm,m_dot,alpha,r_ab,r_bc,r_Q, r_tau1)
        taus_pm = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rpms] 
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
        tau_pm = 2/9/(gamma-1) /alpha * c /c_s_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)
        tau_pm2 = c*gamma/chi(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)*h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)
        tau_pm3 = c*gamma/chi(rpms[1], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)*h_Q(rpms[1], mm, m_dot, alpha, r_ab, r_bc, r_Q)

        aa = surface_density_gradient_Q(rpms[0], r_ab, r_bc, r_Q)
        bb = temperature_gradient_Q(rpms[0], r_ab, r_bc, r_tau1)
        pre_factor = Gamma_I_prefactor(aa,bb)/gamma
        pre_factor_gs=-0.8 - bb - 0.9*aa #eq 15 in gilbaum and stone 2021
    #    cal_B = 1.61 * (gamma-1)/gamma * x_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q) / lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)
        CI = pre_factor_gs
        term = np.fabs(CI)*gamma/1.61/(gamma-1)
        KK = gamma * lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) * CI / h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q)/1.61/(gamma-1)
     #   print ('KK= ', Gamma_I(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1), Gamma_thermal(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) )
    #     print ('ttttt = ', term, lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)/h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q))
      #  tau_tot = tau_pm2 /(1+term*lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)/h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q))
        tau_tot = tau_pm2 /(1 - KK)
        #print (term*lambda_c(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1)/h_Q(rpms[0], mm, m_dot, alpha, r_ab, r_bc, r_Q))
    
        plt.plot(np.log10(tt), np.log10(G_I), linewidth=3, color='black', alpha=1, label='type I'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_I_minus), linewidth=3, color='black', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_th), linewidth=3, color='purple', alpha=1, label='thermal'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_th_minus), linewidth=3, color='purple', alpha=1, linestyle='dashed'); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot), linewidth=3, color='blue', label='total', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.plot(np.log10(tt), np.log10(G_tot_minus), linewidth=3, color='blue', linestyle='dashed', alpha=1); #plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\log \tau$')
        plt.axvline(np.log10(tau_pm2), color='purple',linestyle='dashed')
        plt.axvline(np.log10(tau_pm3), color='purple',linestyle='dashed')
        plt.axvline(np.log10(tau_tot), color='blue',linestyle='dashed')

    #    [plt.axvline(np.log10(t),color=col) for t in taus_pm]
  
        plt.xticks([-1, 0,1,2,3,4,5,6])
    #    plt.ylabel(r'$\log \Gamma / \Gamma_0$')
        plt.yticks([-4,-3,-2,-1,0,1])
        plt.ylim([-4.5,0.4])
        plt.xlim([-1.5,6.3])
     #   plt.legend()
        
    if test_flag:
        tt = [tau_updated(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th = [Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]
        G_th_m = [-Gamma_thermal(r, mm, m_dot, alpha, r_ab, r_bc, r_Q, r_tau1) for r in rs]

        plt.figure(5)
        print(c_s_Q(r_Q, mm, m_dot, alpha, r_ab, r_bc, r_Q), c_s_Q(10*r_Q, mm, m_dot, alpha, r_ab, r_bc, r_Q))
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
fig_2_flag = 1
fig_3_flag = 0
fig_4_flag = 0
test_flag = 0
#assin args: [m (1e8 msun), m_dot (m_cr), alpha, color]
m_d=0.1
args1 = [1e1,m_d,1e-2, 'orange']
args2 = [1e0,m_d,1e-2, 'red']
args3 = [1e-1,m_d,1e-2, 'blue']
args4 = [1e-2,m_d,1e-2, 'green']

plot_disc_solution(*args1)
plot_disc_solution(*args2)
plot_disc_solution(*args3)
plot_disc_solution(*args4)
#%%
def t_wave(r,m,m_dot,alpha,r_ab, r_bc, r_Q):
    Om = Omega(r, m, m_dot, alpha)
    hh = h_Q(r, m, m_dot, alpha, r_ab, r_bc, r_Q)
    h_over_r = hh/r/r_g(m)
    q=10/1e8/m
    SS = Sigma(r, m, m_dot, alpha, r_ab, r_bc)
    m_disc = SS * r**2 * G * r_g(m)/c**2 # in units of M_smbh
    print (h_over_r, q, m_disc)    
    return  Om**-1 * q**-1 * m_disc**-1 *  h_over_r**2

print (t_wave(1e3,0.1,0.5,0.01,300,1000,200)/3.155e7/1e6)
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
print (get_e_fraction(1e-7, 12000))

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
#%%
from scipy.optimize import root 
def ff(x,A,p):
 #   A=args[0]
  #  p=args[1]
    return x - A*(1-x**-0.5)**p

A=1000; p=2/3
print (root(ff, 1000, args=(A, p) ).x)
#%%
def kappa(rho,T):
    X=0.74
    Z=0.02
    
    kappa_m = 0.1 * Z
    kappa_H_minus = 1.1e-25*Z**0.5*rho**0.5*T**7.7
    kappa_ff = 4e25*Z*(1+X)*rho*T**-3.5
    kappa_e = 0.2*(1+X)
    
    return min(kappa_e, kappa_m + (kappa_H_minus**-1 + (kappa_e + kappa_ff)**-1)**-1)

TT = np.logspace(2,6,1000)
kappas = [kappa(1e-15,T) for T in TT]
plt.plot(TT,kappas)
plt.xscale('log')
plt.yscale('log')


