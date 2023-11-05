# Py-BOBYQA example: minimize the Rosenbrock function
from __future__ import print_function
import numpy as np
import pybobyqa
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import pandas as pd
#SIRE model
#def sire(y, t, N, param):
#   S, I, R = y
#   dSdt = param[2]*N - param[2]*S -param[0] * S * I / N
#   dIdt = param[0] * S * I / N - param[1] * I - param[2]*I
#   dRdt = param[1] * I - param[2]*R 
#   return dSdt, dIdt, dRdt
#SIRW model with 5 parameters 
def sirw(y, t, N, param):
    S, I, R, W = y
    dSdt = -param[0]*S*I/N + param[1]*W
    dIdt = param[0] * S * I/N - param[2]*I
    dRdt = param[2] * I - param[3]*R + param[4]*I*W/N
    dWdt = param[3]*R - param[4]*I*W/N - param[1]*W
    return dSdt, dIdt, dRdt, dWdt

def exact(exparam):
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0, W0 = 1, 0, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - W0
    y0 = S0, I0, R0, W0
    t = np.linspace(0, 160, 160)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(sirw, y0, t, args=(N,exparam))
    Sm, Im, Rm, Wm = ret.T
    return Sm, Im, Rm, Wm



# Define the objective function
def pf(param, *exparam):    
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0, W0 = 1, 0, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - W0
    y0 = S0, I0, R0, W0
    t = np.linspace(0, 160, 160)
    # Integrate the SIR equations over the time grid, t.
    #exparam = np.array([0.21, 0.2])
    exret = odeint(sirw, y0, t, args=(N,exparam))
    Sm, Im, Rm, Wm = exret.T  #+ 0.2*random.uniform(1.,3.)# ADD SOME NOISE
    ret = odeint(sirw, y0, t, args=(N,param))
    S, I, R, W = ret.T
    J = 0.5*(np.linalg.norm(I-Im,ord=2) +np.linalg.norm(S-Sm,ord=2)+np.linalg.norm(R-Rm,ord=2)+np.linalg.norm(W-Wm,ord=2)) # + Sm, Rm
    return J


# Define the starting point
param0 = np.array([0.1, 2.1, 0.1, 10.0, 20.1])
exparam = np.array([0.5, 4, 0.2, 10, 50])
# Define the bounds on the independent variables
lower = np.array([0., 0., 0., 0., 0.])
upper = np.array([0.5, 10, 0.5, 10, 100])

# LET's GO

Sm, Im, Rm, Wm = exact(exparam)


# Call Py-BOBYQA
soln = pybobyqa.solve(pf, param0, args=exparam,rhobeg=0.1,rhoend=0.000001,bounds=(lower,upper),print_progress=False,objfun_has_noise= False,seek_global_minimum= True)

# Display output
print(soln.x)
#print(soln.x[0])
#draw plot

#declare inputs 
total_pop = 1000
recovered = 0
infected = 1
waning = 0
susceptible = total_pop - infected - recovered -waning
waning = total_pop - susceptible - infected - recovered
param00 = exparam
# A list of days, 0-160
days = range(0, 160)

# Use differential equations magic with our population
ret = odeint(sirw,
             [susceptible, infected, recovered, waning],
             days,
             args=(total_pop, param00))
S, I, R, W = ret.T

# Build a dataframe because why not
df = pd.DataFrame({
    'suseptible': S,
    'infected': I,
    'recovered': R,
    'waning': W,
    'day': days
})

plt.style.use('ggplot')
df.plot(x='day',
        y=['infected', 'suseptible', 'recovered', 'waning'],
        color=['#bb6424', '#aac6ca', '#cc8ac0', '#8acc90'],
        kind='area',
        stacked=False)
plt.show()