%% lagrange_euler.m
% Symbolic Lagrange-Euler dynamics of a PLANAR 2-DOF (RR) arm on a
% VERTICAL plane.  Produces the manipulator equation
%
%       M(q)*qdd + C(q,qd)*qd + G(q) = tau
%
% and returns/prints the mass (M), Coriolis-centrifugal (C) and gravity (G)
% matrices.
%
% Link indices 2,3 match the shoulder/elbow PITCH links of the RRR arm
% (see rrr_params.m), so this is the planar pitch sub-chain of that robot.
%
% Geometry / symbols (as requested):
%   l2 , l3   link lengths              [m]
%   c2 , c3   joint -> COM distance     [m]
%   m2 , m3   link masses (at the COM)  [kg]
%   o2 , o3   joint angles (zeta)       [rad]   (o2 absolute, o3 relative)
%   g         gravity magnitude         [m/s^2]
%   Iz2 ,Iz3  link inertia about its COM, z-axis [kg m^2]
%             -> set to 0 for the "point mass at the COM" model.
%
% Frame: planar x-y plane standing VERTICALLY. x is horizontal, y is "up",
% gravity acts along -y.  Angles measured from the +x axis.

clear; clc;

%% ---- Symbols -------------------------------------------------------
syms l2 l3   real        % link lengths
syms c2 c3   real        % joint -> COM distances
syms m2 m3   real        % link masses (concentrated at each COM)
syms Iz2 Iz3 real        % link inertia about its own COM (z); 0 = point mass
syms g       real        % gravity magnitude (vertical plane)

syms o2 o3   real        % joint angles      (q)
syms do2 do3 real        % joint rates       (qd)
syms ddo2 ddo3 real      % joint accelerations (qdd)

q   = [o2;   o3];
dq  = [do2;  do3];
ddq = [ddo2; ddo3];
n   = numel(q);

%% ---- Forward kinematics via homogeneous transforms -----------------
% Instead of writing cos/sin by hand we chain planar homogeneous
% transforms (the standard DH way, scales to any number of links):
%
%   Hom(th,a) = Rz(th)*Tx(a)   -> rotate th about z, then slide a along
%                                 the rotated local x-axis  (see end of file)
%
% Frame 0 sits at joint 2.  o2 is the absolute angle of link 2; o3 is the
% elbow angle of link 3 RELATIVE to link 2.
T1  = Hom(o2, l2);              % frame 0 (joint 2) -> joint 3 frame
Tee = T1 * Hom(o3, l3);         % -> tip (end-effector) pose; handy for Jacobians

% Each COM lies a distance c_i along its own link's x-axis, so transform
% to the COM and read off the translation part (rows 1:2 of column 3):
p2 = pos2d( Hom(o2, c2) );      % COM of link 2 = Hom(o2,c2)
p3 = pos2d( T1 * Hom(o3, c3) ); % COM of link 3 = T1 * Hom(o3,c3)

%% ---- Velocity Jacobians (linear + angular) -------------------------
% Linear-velocity Jacobians of each COM: v_i = Jv_i * qd
Jv2 = jacobian(p2, q);                  % 2x2
Jv3 = jacobian(p3, q);                  % 2x2

% Angular velocity of each link about z:  w2 = do2 , w3 = do2+do3
Jw2 = [1 0];                            % w2 = Jw2 * qd
Jw3 = [1 1];                            % w3 = Jw3 * qd

%% ---- Mass / inertia matrix  M(q) -----------------------------------
% From the total kinetic energy  T = 1/2 qd' M qd , with
% T = sum_i [ 1/2 m_i v_i'v_i + 1/2 Iz_i w_i^2 ].
M = m2*(Jv2.'*Jv2) + Iz2*(Jw2.'*Jw2) + ...
    m3*(Jv3.'*Jv3) + Iz3*(Jw3.'*Jw3);
M = simplify(M);

%% ---- Gravity vector  G(q) ------------------------------------------
% Potential energy V = sum m_i g y_i  (y = height = 2nd coordinate).
V = m2*g*p2(2) + m3*g*p3(2);
G = simplify( jacobian(V, q).' );       % 2x1

%% ---- Coriolis / centrifugal matrix  C(q,qd) ------------------------
% Christoffel symbols of the first kind (this IS the Lagrange-Euler
% Coriolis term):  C(i,j) = sum_k 1/2( dMij/dqk + dMik/dqj - dMjk/dqi ) qk_dot
C = sym(zeros(n));
for i = 1:n
    for j = 1:n
        cij = sym(0);
        for k = 1:n
            cij = cij + 0.5*( diff(M(i,j), q(k)) ...
                            + diff(M(i,k), q(j)) ...
                            - diff(M(j,k), q(i)) ) * dq(k);
        end
        C(i,j) = cij;
    end
end
C = simplify(C);

%% ---- Full equation of motion ---------------------------------------
tau = simplify( M*ddq + C*dq + G );     % joint torques [tau2; tau3]

%% ---- Correctness check: (Mdot - 2C) must be skew-symmetric ---------
Mdot = sym(zeros(n));
for k = 1:n
    Mdot = Mdot + diff(M, q(k))*dq(k);
end
skewErr = simplify( (Mdot - 2*C) + (Mdot - 2*C).' );
assert( isequal(skewErr, sym(zeros(n))), ...
        'Skew-symmetry of (Mdot-2C) failed: derivation error.');

%% ---- Display -------------------------------------------------------
disp('=== Planar 2-DOF (RR) Lagrange-Euler dynamics =================');
disp('M(q) ='),      pretty(M)
disp('C(q,qd) ='),   pretty(C)
disp('G(q) ='),      pretty(G)
disp('tau = M*qdd + C*qd + G ='), pretty(tau)
disp('(Mdot - 2C) is skew-symmetric: check PASSED.');

%% ---- Optional: export numeric functions ----------------------------
% Uncomment to generate evaluatable functions M(q), C(q,qd), G(q).
% params = {l2,l3,c2,c3,m2,m3,Iz2,Iz3,g};
% matlabFunction(M, 'File','M_fun', 'Vars',[{q},        params]);
% matlabFunction(C, 'File','C_fun', 'Vars',[{q},{dq},   params]);
% matlabFunction(G, 'File','G_fun', 'Vars',[{q},        params]);

%% ---- Local helper functions ----------------------------------------
function T = Hom(th, a)
% Planar homogeneous transform  Rz(th)*Tx(a):
% rotate by th about z, then translate a along the rotated local x-axis.
T = [ cos(th), -sin(th), a*cos(th);
      sin(th),  cos(th), a*sin(th);
      0,        0,        1       ];
end

function r = pos2d(T)
% Extract the planar position [x; y] from a 3x3 homogeneous transform.
r = T(1:2, 3);
end
