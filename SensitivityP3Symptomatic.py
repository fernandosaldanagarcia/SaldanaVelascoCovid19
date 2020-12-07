# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:49:10 2020

@author: Fer
"""




from __future__ import division
import pylab as pl
import numpy as np
from random import gauss
from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.integrate import odeint, simps

import matplotlib.pyplot as plt




######################################################################
####### Code chunk number 1: Fixed parameters ########################
######################################################################



# birth and natura death rates patch i (b_i, d_i)
b = np.array([1./(70*365),1./(70*365), 1./(70*365)])
d = np.array([1./(70*365),1./(70*365), 1./(70*365)])


# disease transmission coefficient (beta_i)
beta = np.array([0.31, 0.40, 0.53]) # (LR-P1 0.99, MR-P2 1.32, HR-P3 2.01)
alpha = np.array([.45, .45, .45]) # asymptomatic reduction in beta

# transition from exposed to infectious
k = np.array([1/5.99, 1/5.99, 1/5.99]) # G1,G2,G3

# fraction who develop symptoms
rho = np.array([.8, .8, .8]) # G1,G2,G3


# rate for reporting
nu = np.array([1./3, 1/3, 1/3])

# recovery rates
# asymtomatic
ga = np.array([1/14., 1/14., 1/14.]) # G1,G2,G3

# symptomatic
g = np.array([1/10.81, 1/10.81, 1/10.81]) # G1,G2,G3

# reported
gr = np.array([1/5, 1/5, 1/5])

# RESIDENCE TIME MATRIX

P = np.array([[1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3]])



#####################################################
###### code block 2: the model      #################
#####################################################

def deriv(y, t, u, psi, w, theta, rhoV):
    S1, V1, E1, Ev1, A1, I1, Ir1, R1, S2, V2, E2, Ev2, A2, I2, Ir2, R2, S3, V3, E3, Ev3, A3, I3, Ir3, R3 = y
    #
    N1 = S1 + V1 + E1 + Ev1 + A1 + I1 + Ir1 +R1
    N2 = S2 + V2 + E2 + Ev2 + A2 + I2 + Ir2 +R2
    N3 = S3 + V3 + E3 + Ev3 + A3 + I3 + Ir3 +R3
    N = np.array([N1, N2, N3])
    A = np.array([A1, A2, A3])
    I = np.array([I1, I2, I3])
    # effective populations
    # 1
    N1e = np.dot(P[:, 0],N)
    A1e = np.dot(P[:, 0],A)
    I1e = np.dot(P[:, 0],I)
    # 2
    N2e = np.dot(P[:, 1],N)
    A2e = np.dot(P[:, 1],A)
    I2e = np.dot(P[:, 1],I)
    # 3
    N3e = np.dot(P[:, 2],N)
    A3e = np.dot(P[:, 2],A)
    I3e = np.dot(P[:, 2],I)
    #
    # Force of infection beta_{j}(I_{j}^{e}+alphaA_{j}^{e})/N_{j}^{e}
    F = np.array([beta[0]*(I1e + alpha[0]*A1e)/N1e,
                  beta[1]*(I2e + alpha[1]*A2e)/N2e,
                  beta[2]*(I3e + alpha[2]*A3e)/N3e])
    # incidence
    # susceptibles
    incS1 = 0.
    incS2 = 0.
    incS3 = 0.
    for j in [0, 1, 2]:
        incS1 += P[0,j]*S1*F[j]
        incS2 += P[1,j]*S2*F[j]
        incS3 += P[2,j]*S3*F[j]
    # vaccinated
    incV1 = 0.
    incV2 = 0.
    incV3 = 0.
    for j in [0, 1, 2]:
        incS1 += P[0,j]*V1*(1-psi[0])*F[j]
        incS2 += P[1,j]*V2*(1-psi[1])*F[j]
        incS3 += P[2,j]*V3*(1-psi[2])*F[j]
    # model equations
    dS1dt = b[0]*N[0] - incS1 - u[0]*S1 + w[0]*R1 +theta[0]*V1- d[0]*S1
    dV1dt = u[0]*S1 - incV1 - theta[0]*V1 - d[0]*V1
    dE1dt = incS1 - (k[0]+ d[0])*E1
    dEv1dt= incV1 - (k[0]+ d[0])*Ev1
    dA1dt = (1-rho[0])*k[0]*E1 + (1-rhoV[0])*k[0]*Ev1 - (ga[0]+d[0])*A1
    dI1dt = rho[0]*k[0]*E1 + rhoV[0]*k[0]*Ev1 - (g[0]+nu[0]+d[0])*I1
    dIr1dt= nu[0]*I1 - (gr[0]+d[0])*Ir1 
    dR1dt = ga[0]*A1 + g[0]*I1 + gr[0]*Ir1 - (w[0]+d[0])*R1
    dS2dt = b[1]*N[1] - incS2 - u[1]*S2 + w[1]*R2 +theta[1]*V2 - d[1]*S2
    dV2dt = u[1]*S2 - incV2  - theta[1]*V2- d[1]*V2
    dE2dt = incS2 - (k[1]+ d[1])*E2
    dEv2dt= incV2 - (k[0]+ d[0])*Ev1
    dA2dt = (1-rho[1])*k[1]*E2 +(1-rhoV[1])*k[1]*Ev2 - (ga[1]+d[1])*A2
    dI2dt = rho[1]*k[1]*E2 + rhoV[1]*k[1]*Ev2 - (g[1]+nu[1]+d[1])*I2
    dIr2dt= nu[1]*I2 - (gr[1]+d[1])*Ir2 
    dR2dt = ga[1]*A2 + g[1]*I2 + gr[1]*Ir2 - (w[1]+d[1])*R2
    dS3dt = b[2]*N[2] - incS3 - u[2]*S3 + w[2]*R3 + theta[2]*V3 - d[2]*S3
    dV3dt = u[2]*S3 - incV3  - theta[2]*V3 - d[2]*V3
    dE3dt = incS3 - (k[2]+ d[2])*E3
    dEv3dt= incV3 - (k[2]+ d[2])*Ev3
    dA3dt = (1-rho[2])*k[2]*E3 + (1-rhoV[2])*k[2]*Ev3 - (ga[2]+d[2])*A3
    dI3dt = rho[2]*k[2]*E3 + rho[2]*k[2]*Ev3 - (g[2]+nu[2]+d[2])*I3
    dIr3dt= nu[2]*I3 - (gr[2]+d[2])*Ir3 
    dR3dt = ga[2]*A3 + g[2]*I3 + gr[2]*Ir3- (w[2]+d[2])*R3
    return dS1dt, dV1dt, dE1dt, dEv1dt, dA1dt, dI1dt, dIr1dt, dR1dt, dS2dt, dV2dt, dE2dt, dEv2dt, dA2dt, dI2dt, dIr2dt, dR2dt, dS3dt, dV3dt, dE3dt, dEv3dt, dA3dt, dI3dt, dIr3dt, dR3dt 




######################################################################
####### Code chunk number 3: initial conditions #######################
######################################################################

t = np.linspace(0, 425, 425)
t2021 = np.linspace(0, 365, 365)

# initial conditions (S, V, E, Ev, A, I, Ir, R)
# CDMX 22 octubre
CdMxN = 150000
S = .99*CdMxN
I = 300   # activos 5115
A = (1-rho[0])*I # asintomaticos
Ir =  0 # confirmados 157000
E =   I/k[0] # sospechosos 53604
R = .01*CdMxN # recuperadas 127000
CdMx = np.array([S, 0., E, 0., A, I, Ir, R])
# P1 
P10 = CdMx*0.63
# P2 
P20 = CdMx*0.3
# P3 
P30 = CdMx*0.07

# complete initial condition
y0 = np.concatenate([P10, P20, P30])




######################################################################
####### Code chunk number 4: Case without control  ###################
######################################################################

# --------- Vaccine parameters
# vaccination rates
uS0 = np.array([.0, .0, .0]) # Scenario 0 (sin control)

# vaccine efficacy
psiS0 = np.array([0, 0, 0]) # Scenario 0 (sin control)

# loss of natural immunity
wS0 = np.array([1./360, 1/360, 1/360]) # Scenario 0 (sin control)

# loss of vaccine immunity
thetaS0 = np.array([1./180, 1/180, 1/180]) # Scenario 0 (sin control)

# vaccinated fraction who develop symptoms
rhoVS0 = np.array([.55, .55, .55]) # Scenario 0 (sin control)

# Solution without control
solution0 = odeint(deriv, y0, t, args=(uS0, psiS0, wS0, thetaS0, rhoVS0))
S1, V1, E1, Ev1, A1, I1, Ir1, R1, S2, V2, E2, Ev2, A2, I2, Ir2, R2, S3, V3, E3, Ev3, A3, I3, Ir3, R3 = solution0.T


# Contando el numero de casos a partir de 2021
#asymptomaticWC = simps(A1[59:-1]+A2[59:-1]+A3[59:-1], t2021)
#symptomaticWC = simps(I1[59:-1]+I2[59:-1]+I3[59:-1], t2021)
symptomaticWC_P3 = simps(I3[59:-1], t2021)


# Initial condition for 2021...for the reported function below
y2021 = solution0[60]


######################################################################
####### Code chunk number 5: reported function #######################
######################################################################

# --------- define reported as a function of a vector of parameters "x"
# ------------ "x" are the parameters for the sensitivity exploration 




def reported(x):
    theta = x[0]
    T = x[1]
    psi = x[2]
    Coverage = x[3]
    rhoV = x[4]
    u = -np.log(1-Coverage)/T
    n1 = gauss(0.9, 0.05) # random number (mean, sd)
    n2 = gauss(0.9, 0.05) # random number (mean, sd)
    thetaVector = np.array([n1*theta, n2*theta, theta])
    psiVector = np.array([n2*psi, n1*psi, psi])
    rhoV_Vector = np.array([rhoV, n1*rhoV, n2*rhoV])
    uVector = np.array([n2*u, n1*u, u])
    # solution with control
    solution1 = odeint(deriv, y2021, t2021, args=(uVector, psiVector, wS0, thetaVector, rhoV_Vector))
    S11, V11, E11, Ev11, A11, I11, Ir11, R11, S21, V21, E21, Ev21, A21, I21, Ir21, R21, S31, V31, E31, Ev31, A31, I31, Ir31, R31 = solution1.T
    # Contando los casos para el escenario con control
    #reported = simps(Ir11+Ir21+Ir31, t2021)
    symptomaticP3 = simps(I31, t2021)
    PERCENTAGE_REDUCTION = 1-(symptomaticP3/symptomaticWC_P3)
    return PERCENTAGE_REDUCTION




########################################################################
#######   Code chunk number 3: Defining sensitivity ranges     #########
########################################################################



#-define a function to evaluate the values of the parameters in R0
     
def evaluate(values):
    Y = np.empty([values.shape[0]])
    for i, X in enumerate(values):
        Y[i] = reported(X)
    return Y



#-define the ranges for the parameters in x
    
problem = {
'num_vars': 5,  # number of parameters 
'names': ['Duration', 'Time', 'Efficacy','Coverage','Symptomatic'], 
'bounds': [[1/365, 1/180],[30, 150],[0.5, 0.95],[.1, .5],[0.01,0.5]] #ranges
}




########################################################################
#######   Code chunk number 6: Performing the analysis         #########
########################################################################


# ------------Generate samples

number_of_samples = 5000
param_values = saltelli.sample(problem, number_of_samples, calc_second_order=True)
# problem represents our parameters ranges defined above
# calc_second_order=True is to compute second order indices


# ------------Run model (example)

Y = evaluate(param_values)
print(Y)


# ------------Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=False)
#Si = sobol.analyze(problem, Y, print_to_console=False)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)


# -------------Printing some results
#print("First Order Indices", Si['S1'])
#print("Total Order Indices", Si['ST'])




########################################################################
#######   Code chunk number 8: Plotting the results         ############
########################################################################




# -------------------------the histogram of the data
#pl.figure()

#pl.hist(Y,bins=20, normed=1)
#pl.title('Re')
#pl.show()
fig1 = plt.figure(facecolor='w')
ax = fig1.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.hist(Y, bins=30, color='g', edgecolor='k')
ax.set_xlabel('Percentage reduction in the symptomatic cases')
ax.set_ylabel('Frequency')
#ax.set_title(r'Histogram for $\mathcal{R}_{e}$')
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
#legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False) 


#---------------------- Bar plot for the Sobol indices

N = len(Si['S1'])
FirstOrder = Si['S1']
TotalOrder = Si['ST']
FOconf = Si['S1_conf']
TOconf = Si['ST_conf']
ind = np.arange(N)   # the x locations for the groups
width = 0.5     # the width of the bars: can also be len(x) sequence


figIndices = plt.figure(facecolor='w')
ax = figIndices.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.bar(ind, FirstOrder, width, color='#d62728', yerr=FOconf, label='First Order')
ax.bar(ind+width, TotalOrder, width, yerr=TOconf, label='Total Order')
ax.set_xticks(ind+width/2)
ax.set_xticklabels((r'$\theta$', r'$T$', r'$\psi$', r'$C$', r'$\tilde{\rho}$'))
ax.set_ylabel('Sobol sensitivity indices')
#ax.set_title(r'$Sobol\'s$ indices for $\mathcal{R}_{e}$')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
#legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False) 



plt.show()

    