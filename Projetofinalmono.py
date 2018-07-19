#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:48:31 2018

@author: joao
"""

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(6))):
    """
    Parameters:
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q, F = np.matrix('1. 0. 1. 0. 1. 0. ;0. 1. 0. 1. 0. 1.; 0. 0. 1. 0. 1. 0.; 0. 0. 0. 1. 0. 1.;0. 0. 0. 0. 1. 0.;0. 0. 0. 0. 0. 1. '),H = np.matrix('1. 0. 0. 0. 0. 0.;0. 1. 0. 0. 0. 0.'))

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

def demo_kalman_xy():
    x = np.matrix('0. 0. 0. 0. 0. 0.').T
    P = np.matrix(np.eye(6)) # initial uncertainty

    N = 100
    observed_x = np.array([-1.59437,-1.87947,-1.72512,-1.60379,-1.3593,-1.0575,-1.30977,-1.34612,-1.11792,-1.39664,-1.87249,-1.23697,-1.31609,-1.54325,-1.23686,-1.71725,-1.16704,-1.22378,-1.29663,-2.39797,-1.31392,-1.17237,-1.49627,-1.26555,-1.29173,-1.12026,-1.31826,-1.26097,-1.32012,-1.1887,-2.02008,-1.00949,-1.85118,-1.40806,-1.76551,-1.44175,-1.03302,-1.76139,-1.0732,-1.66517,-2.21737,-1.25678,-1.52407,-1.18808,-1.21814,-1.70015,-1.79082,-1.19157,-1.40122,-1.48575,-2.18398,-1.77245,-1.03511,-1.30378,-1.45762,-1.24202,-1.23486,-1.83706,-1.03940,-2.17866,-2.09592,-1.17154,-1.51345,-1.93562,-1.23572,-1.23646,-2.25133,-1.13227,-1.18358,-1.18966,-1.00604,-1.43863,-1.25798,-2.02785,-1.01773,-1.9816,-1.38151,-1.06228,-1.71225,-1.05100,-1.41959,-1.45756,-1.6765,-1.60452,-1.27405,-1.46487,-2.37245,-1.57026,-1.38306,-1.43944,-1.43944,-1.48806,-1.63603,-1.45015,-1.71049,-1.41397,-1.81444,-1.45738,-1.15321,-1.62084])
    #observ50x = np.array(observed_x[0:50])
    observ100x = np.array(observed_x[0:100])
    print(len(observ100x))
    observed_y = np.array([8.82227,8.36676,7.97687,7.71154,8.27116,8.73127,8.01568,7.01314,8.10843,7.30258,8.56615,7.26131,7.45834,7.12874,7.04516,7.30349,7.08852,7.41388,7.05812,8.72983,8.16634,7.87218,7.20293,7.11152,6.94413,7.37599,7.95289,7.70964,7.12028,7.02381,7.33975,7.76411,8.23051,8.40861,7.16763,7.31521,7.60372,7.82719,7.18596,8.07546,7.97091,7.32569,8.57975,6.95079,8.09493,7.56951,6.86954,7.50958,6.88742,7.24858,8.21985,7.87717,7.20845,7.77664,7.77836,7.16535,6.80819,7.75594,8.23207,8.40341,7.95993,8.30638,7.91383,8.84908,6.93945,7.17991,8.29019,8.24557,8.33986,8.36712,7.64475,7.83051,7.12862,8.30996,7.41852,8.34404,8.3645,7.94068,7.06371,7.73622,7.80673,6.86016,7.41835,8.1064,6.96466,7.1078,7.66305,8.1075,7.13038,7.19575,7.19575,7.67893,7.0767,7.11074,7.14677,7.72757,7.93441,7.4001,7.89855,7.22054])
    #observ50y = np.array(observed_y[0:50])
    observ100y = np.array(observed_y[0:100])
    print(len(observ100y))
    #plt.plot(observed_x, observed_y, 'rx',label = "Observações")
    #plt.plot(observ50x, observ50y, 'rx',label = "Observações")
    plt.plot(observ100x, observ100y, 'rx',label = "Observações")
    result = []
    R = 0.01**2
    #for meas in zip(observed_x, observed_y):
    #for meas in zip(observ50x, observ50y):
    for meas in zip(observ100x, observ100y):
        x, P = kalman_xy(x, P, meas, R)
        #print(x)
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    #plt.plot(kalman_x, kalman_y, 'go')

    xap = []
    yap = []

    for i in result:
        xap.append(i[0])
        yap.append(i[1])
        #print(str(i)+"\n")
    posicao_real_x = [-2]*N
    posicao_real_y = [8]*N
    print(np.mean(xap))
    print(np.mean(yap))
    print(mean_absolute_error([-2,8],[np.mean(xap),np.mean(yap)]))
    print(mean_squared_error([-2,8],[np.mean(xap),np.mean(yap)]))

    # plt.plot(kalman_x, kalman_y, 'gx')
    plt.plot(np.mean(xap),np.mean(yap), 'og', label = "Predição Kalman - Bola", ms = 10)
    plt.plot(-2,8,'*b', label = "Posição Jogador 2", ms = 10)
    plt.plot(0,0,'bo', label = "Posição Inicial - Bola", ms = 10)
    plt.xlabel("Posição no Eixo x")
    plt.ylabel("Posição no Eixo y")
    plt.axis([-2.5,0.5,-0.5,10])
    plt.legend()
    plt.savefig('obs100.eps')
    plt.show()
    #x, P = kalman_xy(x, P, meas, R)

    #print(x)

demo_kalman_xy()