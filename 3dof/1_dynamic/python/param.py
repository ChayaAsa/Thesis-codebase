import numpy as np

def robot_params():
    p = {}
    p['g']       = 9.81
    p['gravity'] = np.array([0, 0, -p['g']])

    p['L1'] = 120e-3   # base column height (shoulder offset) [m]
    p['L2'] = 150e-3   # upper-arm length [m]
    p['L3'] = 120e-3   # forearm length [m]

    p['r1'] = 0.0
    p['r2'] = 75e-3
    p['r3'] = 48e-3

    p['m1'] = 0.0     # [kg]
    p['m2'] = 100e-3
    p['m3'] = 780e-3

    p['c1'] = np.array([0, -p['L1'] + p['r1'], 0])   # COM of link i in its own DH frame
    p['c2'] = np.array([-p['L2'] + p['r2'], 0, 0])
    p['c3'] = np.array([-p['L3'] + p['r3'], 0, 0])

    # p['q0'] = np.array([0, np.deg2rad(45), np.deg2rad(-45)])
    p['q0'] = np.array([0, 0, 0])

    return p

def dh_params(robot_params:dict):
    l1 = robot_params['L1']
    l2 = robot_params['L2']
    l3 = robot_params['L3']
    dh = {}

    # dh[i] = ['a', 'alpha', 'd', 'theta']
    dh[1] = np.array([0, np.pi/2, l1, 0])
    dh[2] = np.array([l2,      0,  0, 0])
    dh[3] = np.array([l3,      0,  0, 0])

    return dh
