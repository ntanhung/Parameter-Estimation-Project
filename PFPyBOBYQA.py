# Py-BOBYQA example: minimize the Rosenbrock function
from __future__ import print_function
import numpy as np
import pybobyqa
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

def deriv(y, t, N, param):
    S, I, R = y
    dSdt = -param[0] * S * I / N
    dIdt = param[0] * S * I / N - param[1] * I
    dRdt = param[1] * I
    return dSdt, dIdt, dRdt


def exact(exparam):
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, 160, 160)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N,exparam))
    Sm, Im, Rm = ret.T
    return Sm, Im, Rm


# Define the objective function
def pf(param, *exparam):    
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, 160, 160)
    # Integrate the SIR equations over the time grid, t.
    #exparam = np.array([0.21, 0.2])
    exret = odeint(deriv, y0, t, args=(N,exparam))
    Sm, Im, Rm = exret.T + 0.2*random.uniform(1.,3.)# ADD SOME NOISE
    ret = odeint(deriv, y0, t, args=(N,param))
    S, I, R = ret.T
    J = 0.5*(np.linalg.norm(I-Im,ord=2)+np.linalg.norm(S-Sm,ord=2)+np.linalg.norm(R-Rm,ord=2)) # + Sm, Rm
    return J


# Define the starting point
param0 = np.array([0.1, 0.12])
exparam = np.array([0.25, 0.2])
# Define the bounds on the independent variables
lower = np.array([0., 0.])
upper = np.array([0.5, 0.3])

# LET's GO

Sm, Im, Rm = exact(exparam)


# Call Py-BOBYQA
soln = pybobyqa.solve(pf, param0, args=exparam,bounds=(lower,upper),print_progress=False, objfun_has_noise= False,seek_global_minimum= True)

# Display output
print(soln.x)
print(soln.x[0])