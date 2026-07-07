clc;
clear;

l1 = 120e-3;
l2 = 150e-3;
l3 = 120e-3;
r1 = 0;
r2 = 75e-3;
r3 = 48e-3;
m1 = 0;
m2 = 100e-3;
m3 = 780e-3;

syms q [1 3] real


% Define DH parameters in the order: [a, alpha, d, theta]
dhparams = [0, pi/2, l1, 0;
            l2,   0,  0, 0;
            l3,   0,  0, 0]; 

% Create robot tree and joint objects
robot = rigidBodyTree;
body1 = rigidBody('link1');
joint1 = rigidBodyJoint('joint1', 'revolute');
body2 = rigidBody('link2');
joint2 = rigidBodyJoint('joint2', 'revolute');
body3 = rigidBody('link3');
joint3 = rigidBodyJoint('joint3', 'revolute');

% Apply DH parameters
setFixedTransform(joint1, dhparams(1,:), 'dh');
setFixedTransform(joint2, dhparams(2,:), 'dh');
setFixedTransform(joint3, dhparams(3,:), 'dh');
body1.Joint = joint1;
body2.Joint = joint2;
body3.Joint = joint3;
addBody(robot, body1, 'base');
addBody(robot, body2, 'link1');
addBody(robot, body3, 'link2');

config = homeConfiguration(robot);
config(1).JointPosition = q1;
T0  = getTransform(robot, config, 'base');
T01 = getTransform(robot, config, 'link1');
T12 = getTransform(robot, homeConfiguration(robot), 'link2');
T23 = getTransform(robot, homeConfiguration(robot), 'link3');


figure(Name="Interactive GUI")
gui = interactiveRigidBodyTree(robot,MarkerScaleFactor=0.5,Configuration=[0,pi/4,-pi/4]);
