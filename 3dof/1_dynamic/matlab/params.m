function p = params()
%PARAMS  Physical parameters for the 3-DOF RRR robot arm.
%
%  Returns struct p with fields:
%    g        gravity magnitude [m/s^2]
%    L1/2/3   link lengths: base column / upper-arm / forearm [m]
%    m1/2/3   link masses [kg]
%    c1/2/3   link COM positions in each link's own DH frame [m] (column vectors)

p.g  = 9.81;

p.L1 = 120e-3;   % base column height (shoulder offset)
p.L2 = 150e-3;   % upper-arm length
p.L3 = 120e-3;   % forearm length

p.m1 = 0.0;      % link 1 mass (massless base link)
p.m2 = 100e-3;   % link 2 mass
p.m3 = 780e-3;   % link 3 mass

r1 = 0.0;        % COM radial distances from distal joint
r2 = 75e-3;
r3 = 48e-3;

% COM of link i expressed in its own DH frame (frame i)
p.c1 = [0;             -(p.L1 - r1); 0];
p.c2 = [-(p.L2 - r2);  0;            0];
p.c3 = [-(p.L3 - r3);  0;            0];
end
