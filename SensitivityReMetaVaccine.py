
"""
Created on Fri May 08 08:48:53 2019

@author: Fernando Saldaña
"""

from __future__ import division
import pylab as pl
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

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
rho = np.array([.55, .55, .55]) # G1,G2,G3


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

P = np.array([[.5, .2, .3],
              [.2, .5, .3],
              [.2, .3, .5]])



######################################################################
####### Code chunk number 2: R0 definition ########################
######################################################################

# --------------- define R0 as a function of a vector of parameters "x"
# --------------- "x" are the parameters for the sensitivity exploration 

############ vaccine parameters for Re
# vaccination rates
#u = np.array([.01, .01, .01]) # P1,P2,P3

# vaccine efficacy
#psi = np.array([.6, .6, .6]) # P1,P2,P3

# loss of natural immunity
#w = np.array([1./360, 1/360, 1/360]) # P1,P2,P3

# vaccinated fraction who develop symptoms
#rhoV = np.array([.3, .3, .3])

# loss of vaccine immunity
#theta = np.array([1./200, 1/200, 1/200]) # P1,P2,P3

def R0(x):
    theta = x[0]
    T = x[1]
    psi = x[2]
    Coverage = x[3]
    rhoV = x[4]
    u = -np.log(1-Coverage)/T
    #------------------ DFE
    # constant initial populations
    CdMx=150000
    N1 = 0.63*CdMx #patch 1
    N2 = 0.3*CdMx # patch 2
    N3 = 0.7*CdMx # patch 3
    #----------
    Sdfe1 = b[0]*(theta+d[0])*N1/((u+theta+d[0])*d[0])
    Sdfe2 = b[1]*(theta+d[1])*N2/((u+theta+d[1])*d[1])
    Sdfe3 = b[2]*(theta+d[2])*N1/((u+theta+d[2])*d[2])
    Sdfe = np.array([Sdfe1, Sdfe2, Sdfe3])
    #----------
    Vdfe1 = N1 - Sdfe1
    Vdfe2 = N2 - Sdfe2
    Vdfe3 = N3 - Sdfe3
    Vdfe = np.array([Vdfe1, Vdfe2, Vdfe3])
    #----------
    # F matrix
    # nota: effective total population N^{e}_{k} (tp) at DFE is the sum of tp N_{j}*p_{jk}
    N = np.array([N1, N2, N3]) # population at DFE
    Ne_dfe = np.array([np.dot(P[:, 0],N), np.dot(P[:, 1],N), np.dot(P[:, 2],N)])
    # lamnda and lamndaA components in F
    #------------- Re total movement
    # components for controled F
    phi = np.zeros((3,3), float)
    phiA = np.zeros((3,3), float)
    chi = np.zeros((3,3), float)
    chiA = np.zeros((3,3), float)
    for i in [0,1,2]:
        for j in [0,1,2]:
            for z in [0,1,2]:
                 phi[i][j] += beta[z]*P[i][z]*P[j][z]*Sdfe[z]/Ne_dfe[z]
                 phiA[i][j]+= alpha[z]*beta[z]*P[i][z]*P[j][z]*Sdfe[z]/Ne_dfe[z]
                 chi[i][j] += beta[z]*(1-psi)*P[i][z]*P[j][z]*Vdfe[z]/Ne_dfe[z]
                 chiA[i][j] += alpha[z]*beta[z]*(1-psi)*P[i][z]*P[j][z]*Vdfe[z]/Ne_dfe[z]
    #----
    #print(Falt[0,0])          
    # real 9x9 F matrix
    # 12x12 F matrix
    Fc= np.array([[0, 0, phiA[0][0], phi[0][0], 0, 0, phiA[0][1], phi[0][1], 0, 0, phiA[0][2], phi[0][2]],
                  [0, 0, chiA[0][0], chi[0][0], 0, 0, chiA[0][1], chi[0][1], 0, 0, chiA[0][2], chi[0][2]],
                  [0, 0, 0, 0,   0, 0, 0, 0,     0, 0, 0, 0],
                  [0, 0, 0, 0,   0, 0, 0, 0,     0, 0, 0, 0],
                  [0, 0, phiA[1][0], phi[1][0], 0, 0, phiA[1][1], phi[1][1], 0, 0, phiA[1][2], phi[1][2]],
                  [0, 0, chiA[1][0], chi[1][0], 0, 0, chiA[1][1], chi[1][1], 0, 0, chiA[1][2], chi[1][2]],
                  [0, 0, 0, 0,   0, 0, 0, 0,     0, 0, 0, 0],
                  [0, 0, 0, 0,   0, 0, 0, 0,     0, 0, 0, 0],
                  [0, 0, phiA[2][0], phi[2][0], 0, 0, phiA[2][1], phi[2][1], 0, 0, phiA[2][2], phi[2][2]],
                  [0, 0, chiA[2][0], chi[2][0], 0, 0, chiA[2][1], chi[2][1], 0, 0, chiA[2][2], chi[2][2]],
                  [0, 0, 0, 0,   0, 0, 0, 0,     0, 0, 0, 0],
                  [0, 0, 0, 0,   0, 0, 0, 0,     0, 0, 0, 0]])
    #-----
    # V 9x9 matrix
    Vc= np.array([[k[0]+d[0], 0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, k[0]+d[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-(1-rho[0])*k[0], -(1-rhoV)*k[0], ga[0]+d[0], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-rho[0]*k[0], -rhoV*k[0], 0, g[0]+nu[0]+d[0], 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, k[1]+d[1], 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, k[1]+d[1], 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -(1-rho[1])*k[1], -(1-rhoV)*k[1], ga[1]+d[1], 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -rho[1]*k[1], -rhoV*k[1], 0, g[1]+nu[1]+d[1],  0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0,  k[2]+d[2],  0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0,  0, k[2]+d[2],  0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -(1-rho[2])*k[2], -(1-rhoV)*k[2], ga[2]+d[2], 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -rho[2]*k[2], -rhoV*k[2], 0, g[2]+nu[2]+d[2]] ])
    # controlled NGM
    Kc = np.dot(Fc, np.linalg.inv(Vc))
    Re = np.amax(np.linalg.eigvals(Kc))
    return Re




########################################################################
#######   Code chunk number 3: Defining sensitivity ranges     #########
########################################################################



#-define a function to evaluate the values of the parameters in R0
     
def evaluate(values):
    Y = np.empty([values.shape[0]])
    for i, X in enumerate(values):
        Y[i] = R0(X)
    return Y



#-define the ranges for the parameters in x
    
problem = {
'num_vars': 5,  # number of parameters 
'names': ['Duration', 'Time', 'Efficacy','Coverage','Symptomatic'], 
'bounds': [[1/365, 1/180],[30, 150],[0.5, 0.95],[.1, .6],[0.1,0.5]] #ranges
}




########################################################################
#######   Code chunk number 3: Performing the analysis         #########
########################################################################


# ------------Generate samples

number_of_samples = 10000
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
print("First Order Indices", Si['S1'])
print("Total Order Indices", Si['ST'])





########################################################################
#######   Code chunk number 3: Plotting the results         ############
########################################################################




# -------------------------the histogram of the data
#pl.figure()

#pl.hist(Y,bins=20, normed=1)
#pl.title('Re')
#pl.show()
fig1 = plt.figure(facecolor='w')
ax = fig1.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.hist(Y, bins=30, color='steelblue', edgecolor='k')
ax.set_xlabel(r'$\mathcal{R}_{e}$ value')
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

#-----------------------------------------------------------------------

