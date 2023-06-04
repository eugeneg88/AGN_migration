#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:11:26 2023

@author: evgeni
"""

import matplotlib.pyplot as plt
import numpy as np

def lane_emden(n, x):
  """Solve the Lane-Emden equation for n=1 polytrope.

  Args:
    n: Polytropic index.
    x: Dimensionless radius.

  Returns:
    Density profile.
  """

  def f(x):
    return x**2 * theta(x)

  def g(x):
    return 0.5 * theta(x)**n

  theta = np.zeros(1000)
  theta[0] = 1.0

  for i in range(1, 1000):
    theta[i] = f(theta[i-1]) / g(theta[i-1])

  return theta

def main():
  # Set the polytropic index.
  n = 1

  # Define the range of dimensionless radii.
  x_min = 0
  x_max = 1

  # Create a grid of dimensionless radii.
  x = np.linspace(x_min, x_max, 1000)

  # Solve the Lane-Emden equation.
  density = lane_emden(n, x)

  # Plot the density profile.
  plt.plot(x, density)
  plt.xlabel("Dimensionless radius")
  plt.ylabel("Density")
  plt.show()

if __name__ == "__main__":
  main()
