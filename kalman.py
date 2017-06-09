import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab

import os.path
import math
import cv2.cv as cv
import sys

RADIUS = 0.02
MASS = 0.00275
GRAVITY = 9.82
RHO_A = 1.29
C_D = 0.405
C_M = 0.62
PI = 3.141592653
dT = 0.005
kD = 0.5*MASS*C_D*RHO_A*PI*RADIUS*RADIUS*dT
kM = 0.5*MASS*C_M*RHO_A*RADIUS*PI*RADIUS*RADIUS*dT

kalman = cv.CreateKalman(6, 3, 1)
process_noise = cv.CreateMat(6, 1, cv.CV_32FC1)
measurement = cv.CreateMat(3, 1, cv.CV_32FC1)
control = cv.CreateMat(1, 1, cv.CV_32FC1)
rng = cv.RNG(-1)

cv.Zero(measurement)

def init_kalman(kalman,x,y,z):
    
    kalman.transition_matrix[:,:] = 0
    for i in range(6):
        kalman.transition_matrix[i,i] = 1
    kalman.transition_matrix[0,3] = dT
    kalman.transition_matrix[1,4] = dT
    kalman.transition_matrix[2,5] = dT

    kalman.state_post[:,:] = 0

    kalman.control_matrix[:,:] = 0
    kalman.control_matrix[5,0] = 1

    cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
    cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))
    cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-3))
    cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(1))
    
    kalman.state_post[0,0]=x
    kalman.state_post[1,0]=y
    kalman.state_post[2,0]=z

def track(kalman,x,y,z):
    vx = kalman.state_post[3,0]
    vy = kalman.state_post[4,0]
    vz = kalman.state_post[5,0]
    v = math.sqrt(vx**2+vy**2+vz**2)
    kalman.transition_matrix[3,3] = 1-kD*v
    kalman.transition_matrix[4,4] = 1-kD*v
    kalman.transition_matrix[5,5] = 1-kD*v
    control[0,0] = -GRAVITY*dT
    cv.KalmanPredict(kalman, control)
    pred = [kalman.state_pre[0,0],kalman.state_pre[1,0],kalman.state_pre[2,0]]
    
    cv.RandArr(rng, measurement, cv.CV_RAND_NORMAL, cv.RealScalar(0),cv.RealScalar(0.005))
    measurement[0,0] += x
    measurement[1,0] += y
    measurement[2,0] += z
    cv.KalmanCorrect(kalman, measurement)
    
    meas = [measurement[0,0],measurement[1,0],measurement[2,0]]
    
    return pred, meas, np.array([x,y,z])

init = False
preds = []
meass = []
reals = []
coords = open('in.txt','r')
for line in coords.readlines():
    coord = map(float, line.strip().split(' '))
    if not init:
        init_kalman(kalman, coord[0]/1000, coord[1]/1000, coord[2]/1000)
        init = True
    else:
        pred, meas, real = track(kalman, coord[0]/1000, coord[1]/1000, coord[2]/1000)
        preds.append(pred)
        meass.append(meas)
        reals.append(real)
preds = np.array(preds)
meass = np.array(meass)
reals = np.array(reals)
print preds.shape

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(preds[:,0],preds[:,1],preds[:,2],label='pred')
ax.plot(meass[:,0],meass[:,1],meass[:,2],label='meas')
ax.plot(reals[:,0],reals[:,1],reals[:,2],label='real')
ax.legend(loc='best')
plt.show()