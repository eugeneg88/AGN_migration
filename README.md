# AGN migration

A script that calculates a simple AGN disc model a-la Shakura-Sunyaev (1973) + conditions for Toomre $Q=1$ zone a-la Sirko and Goodman.

Input: SMBH mass, accretion rate, $\alpha$ viscosity

Output:
1. gas density $\rho$, surface density $\Sigma$, sound speed $c_s$, opacity $\kappa$, optical depth $\tau$, scale-height $H$, pressure $P$, central temperature $T_c$
2. Derived quantities: Luminosities ratio for stellar mass BH, thermal conductivity $\chi$
3. Torques: Typical torque $\Gamma_0$, type I torque $\Gamma_I$, thermal torque $\Gamma_{\rm th}$

Plots:
1. Disc structure
2. different prescriptions for type I migraion
3. Torques for a choice of disc parameters

Other shells run a grid of disc models and calculate the location/existence of migration traps.
