# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:21:07 2020

@author: Fer
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



#######################################################
## Code block 1: Model parameters         #############
#######################################################

# birth and natura death rates patch i (b_i, d_i)
b = np.array([1./(70*365),1./(70*365), 1./(70*365)])
d = np.array([1./(70*365),1./(70*365), 1./(70*365)])


# disease transmission coefficient (beta_i)
beta = np.array([0.31, 0.40, 0.53]) # (LR-P1 0.31, MR-P2 .40, HR-P3 0.53)
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

PI = np.array([[.15, .15, .7],
              [.15, .15, .7],
              [.15, .15, .7]])

PII = np.array([[.8, .1, .1],
                [.1, .8, .1],
                [.1, .1, .8]])




#####################################################
###### code block 2: the model      #################
#####################################################

def deriv(y, t, u, psi, w, theta, rhoV, ResidenceTime):
    S1, V1, E1, Ev1, A1, I1, Ir1, R1, S2, V2, E2, Ev2, A2, I2, Ir2, R2, S3, V3, E3, Ev3, A3, I3, Ir3, R3 = y
    #
    N1 = S1 + V1 + E1 + Ev1 + A1 + I1 + Ir1 +R1
    N2 = S2 + V2 + E2 + Ev2 + A2 + I2 + Ir2 +R2
    N3 = S3 + V3 + E3 + Ev3 + A3 + I3 + Ir3 +R3
    N = np.array([N1, N2, N3])
    A = np.array([A1, A2, A3])
    I = np.array([I1, I2, I3])
    #
    P = ResidenceTime
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




#####################################################
###### code block 3: integration and plots  #########
#####################################################
    
# A grid of time points (in days)
t = np.linspace(0, 365, 365)
t2021 = np.linspace(0, 250, 250)

# initial conditions (S, V, E, Ev, A, I, Ir, R)

CdMxN = 160000 # effective total population
S = .97*CdMxN
I = .005*CdMxN   # percentage active cases
A = (1-rho[0])*I # asintomatic
Ir =  0 # 
E =   I/k[0] # exposed fraction
R = .03*CdMxN # 1% inmunity in the population
CdMx = np.array([S, 0., E, 0., A, I, Ir, R])
# P1 
P10 = CdMx*0.5
# P2 
P20 = CdMx*0.3
# P3 
P30 = CdMx*0.2

# complete initial condition
y0 = np.concatenate([P10, P20, P30])


##############Parameters for the scenarios

# Formula 1-exp(-uT)=x, u=vac rate, T=tiempo, x=coverage
u1 = -np.log(1-0.3)/60  # 30% coverage, 1 months
u2 = -np.log(1-0.3)/120 # 40% coverage, 1 months
u3 = -np.log(1-0.3)/180 # 50% coverage, 1 months

# vaccination rates
uS0 = np.array([.0, .0, .0]) # Scenario 0 (sin control)
uS1 = np.array([u1, u1, u1]) # Scenario 1 
uS2 = np.array([u2, u2, u2]) # Scenario 2 
uS3 = np.array([u3, u3, u3]) # Scenario 3 

# vaccine efficacy
psiS0 = np.array([0, 0, 0]) # Scenario 0 (sin control)
psiS1 = np.array([.7, .7, .7]) # Scenario 1 
psiS2 = np.array([.8, .8, .8]) # Scenario 2 
psiS3 = np.array([.9, .9, .9]) # Scenario 3 

# loss of natural immunity
w180 = np.array([1./180, 1/180, 1/180]) # Medio año
w360 = np.array([1./360, 1/360, 1/360]) # 1 año

# loss of vaccine immunity
thetaS0 = np.array([1./180, 1/180, 1/180]) # Scenario 0 (sin control)
thetaS1 = np.array([1./180, 1/180, 1/180]) # Scenario 1
thetaS2 = np.array([1./250, 1/250, 1/250]) # Scenario 2
thetaS3 = np.array([1./365, 1/365, 1/365]) # Scenario 3

# vaccinated fraction who develop symptoms
rhoVS0 = np.array([.8, .8, .8]) # Scenario 0 (sin control)
rhoVS1 = np.array([.5, .5, .5]) # Scenario 1
rhoVS2 = np.array([.3, .3, .3]) # Scenario 2
rhoVS3 = np.array([.1, .1, .1]) # Scenario 3




# Integrate the equation over the grid time t
# Solution low mobility
solutionLM = odeint(deriv, y0, t, args=(uS0, psiS0, w360, thetaS0, rhoVS0, PII))
S1, V1, E1, Ev1, A1, I1, Ir1, R1, S2, V2, E2, Ev2, A2, I2, Ir2, R2, S3, V3, E3, Ev3, A3, I3, Ir3, R3 = solutionLM.T


solutionHM = odeint(deriv, y0, t, args=(uS0, psiS0, w360, thetaS0, rhoVS0, PI))
S1h, V1h, E1h, Ev1h, A1h, I1h, Ir1h, R1h, S2h, V2h, E2h, Ev2h, A2h, I2h, Ir2h, R2h, S3h, V3h, E3h, Ev3h, A3h, I3h, Ir3h, R3h = solutionHM.T

# initial condition 2021
y2021 = solutionLM[39]

# Scenario 1
solution1 = odeint(deriv, y0, t, args=(uS1, psiS2, w360, thetaS1, rhoVS2, PI))
S11, V11, E11, Ev11, A11, I11, Ir11, R11, S21, V21, E21, Ev21, A21, I21, Ir21, R21, S31, V31, E31, Ev31, A31, I31, Ir31, R31 = solution1.T

# Scenario 2
solution2 = odeint(deriv, y0, t, args=(uS2, psiS2, w360, thetaS2, rhoVS2, PI))
S12, V12, E12, Ev12, A12, I12, Ir12, R12, S22, V22, E22, Ev22, A22, I22, Ir22, R22, S32, V32, E32, Ev32, A32, I32, Ir32, R32 = solution2.T

# Scenario 3
solution3 = odeint(deriv, y0, t, args=(uS3, psiS2, w360, thetaS3, rhoVS2, PI))
S13, V13, E13, Ev13, A13, I13, Ir13, R13, S23, V23, E23, Ev23, A23, I23, Ir23, R23, S33, V33, E33, Ev33, A33, I33, Ir33, R33 = solution3.T


cumReportedLM = np.cumsum(Ir1+Ir2+Ir3) + 190000
cumReportedHM = np.cumsum(Ir1h+Ir2h+Ir3h) + 190000
cumReported1 = np.cumsum(Ir11+Ir21+Ir31) + 190000
cumReported2 = np.cumsum(Ir12+Ir22+Ir32) + 190000
cumReported3 = np.cumsum(Ir13+Ir23+Ir33) + 190000


# Plot the data
#--PLOT Acumulated reported cases
fig5 = plt.figure(facecolor='w')
ax = fig5.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, cumReportedLM, 'r--', alpha=1, lw=3, label='Restricted mobility')
ax.plot(t, cumReportedHM, 'r', alpha=1, lw=3, label='High mobility')
ax.plot(t, cumReported1, 'b', alpha=1, lw=2, label='2 Months')
ax.plot(t, cumReported2, 'g', alpha=1, lw=2, label='4 Months')
ax.plot(t, cumReported3, 'gold', alpha=1, lw=2, label='6 Months')
ax.set_xlabel('Days since November 20, 2020')
ax.set_ylabel('Cumulative reported cases')
#ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False) 
    
plt.show()





#####################################################
###### code block 4: R0 values              #########
#####################################################

############ vaccine parameters for Re
# vaccination rates
u = uS2  # Scenario 1

# vaccine efficacy
psi = psiS2

# loss of natural immunity
w = w360

# vaccinated fraction who develop symptoms
rhoV = rhoVS2

# loss of vaccine immunity
theta = thetaS2

#------------------ DFE
# constant initial populations
N1 = np.sum(P10) #patch 1
N2 = np.sum(P20) # patch 2
N3 = np.sum(P30) # patch 3

Sdfe1 = b[0]*(theta[0]+d[0])*N1/((u[0]+theta[0]+d[0])*d[0])
Sdfe2 = b[1]*(theta[1]+d[1])*N2/((u[1]+theta[1]+d[1])*d[1])
Sdfe3 = b[2]*(theta[2]+d[2])*N3/((u[2]+theta[2]+d[2])*d[2])
Sdfe = np.array([Sdfe1, Sdfe2, Sdfe3])
print(Sdfe)

Vdfe1 = N1 - Sdfe1
Vdfe2 = N2 - Sdfe2
Vdfe3 = N3 - Sdfe3
Vdfe = np.array([Vdfe1, Vdfe2, Vdfe3])
#-----------------------------------------

# R0 in the absence of movement
R01 = ((alpha[0]*(1-rho[0])/(ga[0]+d[0])) + rho[0]/(g[0]+d[0]))*(k[0]*beta[0]/(k[0]+nu[0]+d[0]))
R02 = ((alpha[1]*(1-rho[1])/(ga[1]+d[1])) + rho[1]/(g[1]+d[1]))*(k[1]*beta[1]/(k[1]+nu[0]+d[1]))
R03 = ((alpha[2]*(1-rho[2])/(ga[2]+d[2])) + rho[2]/(g[2]+d[2]))*(k[2]*beta[2]/(k[2]+nu[0]+d[2]))

print("R0 in the absence of movement")
print("R01:", R01)
print("R02:", R02)
print("R03:", R03)



#---------------------
# R0 total (movement)
#
# F matrix
# nota: effective total population N^{e}_{k} (tp) at DFE is the sum of tp N_{j}*p_{jk}
N = np.array([N1, N2, N3]) # population at DFE
Ne_dfe = np.array([np.dot(N,P[0]), np.dot(N,P[1]), np.dot(N,P[2])])
#print(Ne_dfe)

# lamnda and lamndaA components in F
L = np.zeros((3,3), float)
La = np.zeros((3,3), float)

for i in [0,1,2]:
    for j in [0,1,2]:
        for z in [0,1,2]:
            L[i][j] += beta[z]*P[i][z]*P[j][z]*N[z]/Ne_dfe[z]
            La[i][j] += alpha[z]*beta[z]*P[i][z]*P[j][z]*N[z]/Ne_dfe[z]
  
#print(Falt[0,0])          
# real 9x9 F matrix
F = np.array([[0, La[0][0], L[0][0], 0, La[0][1], L[0][1], 0, La[0][2], L[0][2]],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, La[1][0], L[1][0], 0, La[1][1], L[1][1], 0, La[1][2], L[1][2]],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, La[2][0], L[2][0], 0, La[2][1], L[2][1], 0, La[2][2], L[2][2]],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]])

# V 9x9 matrix

V = np.array([[k[0]+d[0], 0., 0, 0, 0, 0, 0, 0, 0],
              [-(1-rho[0])*k[0], ga[0]+d[0], 0, 0, 0, 0, 0, 0, 0],
              [-rho[0]*k[0], 0, g[0]+nu[0]+d[0], 0, 0, 0, 0, 0, 0],
              [0, 0, 0, k[1]+d[1], 0, 0, 0, 0, 0],
              [0, 0, 0, -(1-rho[1])*k[1], ga[1]+d[1], 0, 0, 0, 0],
              [0, 0, 0, -rho[1]*k[1], 0, g[1]+nu[1]+d[1],  0, 0, 0],
              [0, 0, 0, 0, 0, 0,  k[2]+d[2],  0, 0],
              [0, 0, 0, 0, 0, 0, -(1-rho[2])*k[2], ga[2]+d[2], 0],
              [0, 0, 0, 0, 0, 0, -rho[2]*k[2], 0, g[2]+nu[2]+d[2]] ])

# NGM
K = np.dot(F, np.linalg.inv(V))
R0 = np.amax(np.linalg.eigvals(K))
print("Total R0:", R0)            


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
            chi[i][j] += beta[z]*(1-psi[z])*P[i][z]*P[j][z]*Vdfe[z]/Ne_dfe[z]
            chiA[i][j] += alpha[z]*beta[z]*(1-psi[z])*P[i][z]*P[j][z]*Vdfe[z]/Ne_dfe[z]
  
         
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

   
# V 9x9 matrix
Vc= np.array([[k[0]+d[0], 0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, k[0]+d[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [-(1-rho[0])*k[0], -(1-rhoV[0])*k[0], ga[0]+d[0], 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [-rho[0]*k[0], -rhoV[0]*k[0], 0, g[0]+nu[0]+d[0], 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, k[1]+d[1], 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, k[1]+d[1], 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -(1-rho[1])*k[1], -(1-rhoV[1])*k[1], ga[1]+d[1], 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -rho[1]*k[1], -rhoV[1]*k[1], 0, g[1]+nu[1]+d[1],  0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0,  k[2]+d[2],  0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0,  0, k[2]+d[2],  0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -(1-rho[2])*k[2], -(1-rhoV[2])*k[2], ga[2]+d[2], 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -rho[2]*k[2], -rhoV[2]*k[2], 0, g[2]+nu[2]+d[2]] ])

# controlled NGM
Kc = np.dot(Fc, np.linalg.inv(Vc))
Re = np.amax(np.linalg.eigvals(Kc))
print("Total Re:", Re)        





