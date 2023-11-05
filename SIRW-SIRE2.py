# Py-BOBYQA example: minimize the Rosenbrock function
from __future__ import print_function
from pickle import TRUE
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
    dSdt = param[5]-param[0]*S*I/N + param[1]*W - param[6]*S
    dIdt = param[0] * S * I/N - param[2]*I - param[8]*I
    dRdt = param[2] * I - param[3]*R + param[4]*I*W/N + param[7]*W
    dWdt = param[3]*R - param[4]*I*W/N - param[3]*W - param[7]*W 
    return dSdt, dIdt, dRdt, dWdt

exparam = [500,2,40,2.0005,2000,0.000,0.0005,0.000,0.0005]
# 500,2,40.0005,2.0005,2000,0.,0.0005,0.
#draw plot

#declare inputs 
total_pop = 1
recovered = 0
infected = 0.001
waning = 0
susceptible = total_pop - infected - recovered -waning
param00 = exparam
# A list of days, 0-160
days = range(0, 1600)

# Use differential equations magic with our population
ret = odeint(sirw,
             [susceptible, infected, recovered, waning],
             days,
             args=(total_pop, param00))

S, I, R, W = ret.T

# Build a dataframe because why not
df = pd.DataFrame({
    'susceptible': S,
    'infected': I,
    'recovered': R,
    'waning': W,
    'day': days
})
# #8acc90
plt.style.use('ggplot')
df.plot(x='day',
        y=['infected', 'susceptible', 'recovered', 'waning'],
        color=['#bb6424', '#aac6ca', '#cc8ac0', '#ff0000'],
        kind='line',
        stacked=False)
plt.ylabel('population')
plt.show()