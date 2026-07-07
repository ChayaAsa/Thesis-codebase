function dynamic()
%DYNAMIC  Symbolic derivation and Simulink function generation for 3-DOF RRR arm.
%
%  Combines kinematics (DH transforms, COM positions, Jacobian) and kinetics
%  (Lagrangian M, C, G) in one unified symbolic derivation.  Numerical
%  parameters are loaded from params() and substituted directly, so no extra
%  symbolic parameter variables are needed.
%
%  Run once from MATLAB (requires Symbolic Math Toolbox, ~30-60 s):
%
%    >> dynamic()
%
%  Generates three Simulink-ready MATLAB function files in the same folder:
%
%    dyn_MCG.m  —  [M,C,G] = dyn_MCG(q1,q2,q3,qd1,qd2,qd3)
%    dyn_FK.m   —  [T01,T02,T03] = dyn_FK(q1,q2,q3)
%    dyn_J.m    —  J = dyn_J(q1,q2,q3)       % 6×3 geometric Jacobian
%
%  Equation of motion:  tau = M(q)*qdd + C(q,qd)*qd + G(q)
%
%  In Simulink: use MATLAB Function blocks and call dyn_MCG / dyn_FK / dyn_J.

out_dir = fileparts(mfilename('fullpath'));

p = params();
fprintf('[dynamic] Parameters loaded.\n');

%% ── Symbolic joint variables ─────────────────────────────────────────────────
syms q1 q2 q3   real
syms qd1 qd2 qd3 real
q  = [q1; q2; q3];
qd = [qd1; qd2; qd3];

% Numerical params substituted at build time (no extra syms needed)
L1 = p.L1;   L2 = p.L2;   L3 = p.L3;
m1 = p.m1;   m2 = p.m2;   m3 = p.m3;
g  = p.g;
c1 = p.c1;   c2 = p.c2;   c3 = p.c3;

%% ── KINEMATICS: DH homogeneous transforms ───────────────────────────────────
% Standard DH:  T = Rz(θ) * Tz(d) * Tx(a) * Rx(α)
%
% Frame  a      alpha    d    theta
%   1    0      π/2     L1    q1     base yaw
%   2    L2     0        0    q2     shoulder pitch
%   3    L3     0        0    q3     elbow pitch
fprintf('[dynamic] Deriving kinematics (transforms)...\n');

T01 = dh_mat(0,    sym(pi)/2, L1, q1);
T12 = dh_mat(L2,   0,          0, q2);
T23 = dh_mat(L3,   0,          0, q3);

T02 = simplify(T01 * T12);
T03 = simplify(T02 * T23);

FK = sym(zeros(4,4,5));
FK(:,:,1) = T01;
FK(:,:,2) = T02;
FK(:,:,3) = T03;
FK(:,:,4) = T12;
FK(:,:,5) = T23;

%% ── KINEMATICS: COM positions in base frame ─────────────────────────────────
% c_i = COM of link i in its own DH frame (frame i).
% T0i transforms from frame i to base: p_ci_base = T0i * [c_i; 1].
fprintf('[dynamic] Deriving COM positions...\n');

pc1 = simplify(T01 * [c1; 1]);  pc1 = pc1(1:3);
pc2 = simplify(T02 * [c2; 1]);  pc2 = pc2(1:3);
pc3 = simplify(T03 * [c3; 1]);  pc3 = pc3(1:3);
pc  = {pc1, pc2, pc3};

%% ── KINEMATICS: Geometric Jacobian (6×3) ────────────────────────────────────
% Revolute joint i:  Jv_i = z_{i-1} × (p_ee − o_{i-1}),  Jw_i = z_{i-1}
fprintf('[dynamic] Deriving Jacobian...\n');

p_ee    = T03(1:3, 4);
origins = {sym(zeros(3,1)), T01(1:3,4), T02(1:3,4)};
axes    = {sym([0;0;1]),    T01(1:3,3), T02(1:3,3)};

Jv = sym(zeros(3,3));
Jw = sym(zeros(3,3));
for i = 1:3
    Jv(:,i) = cross(axes{i}, p_ee - origins{i});
    Jw(:,i) = axes{i};
end
J_sym = simplify([Jv; Jw]);   % 6×3

%% ── KINETICS: Mass matrix  M = Σ mᵢ Jvᵢᵀ Jvᵢ ───────────────────────────────
% Jvi = ∂p_ci/∂q  (3×3 velocity Jacobian of COM i w.r.t. joint angles)
fprintf('[dynamic] Deriving mass matrix M...\n');

masses = [m1, m2, m3];
M_sym  = sym(zeros(3,3));
for i = 1:3
    Jvi   = jacobian(pc{i}, q);          % 3×3
    M_sym = M_sym + masses(i) * (Jvi.' * Jvi);
end
M_sym = simplify(M_sym);

%% ── KINETICS: Coriolis matrix C via Christoffel symbols ─────────────────────
% C_{kj} = Σ_i  c_{ijk} · qd_i
% c_{ijk} = ½ ( ∂M_{kj}/∂q_i + ∂M_{ki}/∂q_j − ∂M_{ij}/∂q_k )
fprintf('[dynamic] Deriving Coriolis matrix C...\n');

dMdq = cell(1,3);
for l = 1:3
    dMdq{l} = diff(M_sym, q(l));   % 3×3
end

C_sym = sym(zeros(3,3));
for k = 1:3
    for j = 1:3
        s = sym(0);
        for i = 1:3
            c_ijk = (dMdq{i}(k,j) + dMdq{j}(k,i) - dMdq{k}(i,j)) / 2;
            s = s + c_ijk * qd(i);
        end
        C_sym(k,j) = s;
    end
end
C_sym = simplify(C_sym);

%% ── KINETICS: Gravity vector  G = ∂V/∂q,  V = Σ mᵢ g z_ci ─────────────────
fprintf('[dynamic] Deriving gravity vector G...\n');

V = sym(0);
for i = 1:3
    V = V + masses(i) * g * pc{i}(3);   % z-component of COM in base frame
end

G_sym = sym(zeros(3,1));
for i = 1:3
    G_sym(i) = diff(V, q(i));
end
G_sym = simplify(G_sym);

%% ── Generate Simulink-ready MATLAB function files ───────────────────────────
fprintf('[dynamic] Generating dyn_MCG.m...\n');
matlabFunction(M_sym, ...
    'File',     fullfile(out_dir, 'dyn_M'), ...
    'Vars',     {q1, q2, q3}, ...
    'Outputs',  {'M'}, ...
    'Optimize', true);

fprintf('[dynamic] Generating dyn_C.m...\n');
matlabFunction(C_sym, ...
    'File',     fullfile(out_dir, 'dyn_C'), ...
    'Vars',     {q1, q2, q3, qd1, qd2, qd3}, ...
    'Outputs',  {'C'}, ...
    'Optimize', true);

fprintf('[dynamic] Generating dyn_G.m...\n');
matlabFunction(G_sym, ...
    'File',     fullfile(out_dir, 'dyn_G'), ...
    'Vars',     {q1, q2, q3}, ...
    'Outputs',  {'G'}, ...
    'Optimize', true);

fprintf('[dynamic] Generating dyn_FK.m...\n');
matlabFunction(FK, ...
    'File',     fullfile(out_dir, 'dyn_FK'), ...
    'Vars',     {q1, q2, q3}, ...
    'Outputs',  {'FK'}, ...
    'Optimize', true);

fprintf('[dynamic] Generating dyn_J.m...\n');
matlabFunction(J_sym, ...
    'File',     fullfile(out_dir, 'dyn_J'), ...
    'Vars',     {q1, q2, q3}, ...
    'Outputs',  {'J'}, ...
    'Optimize', true);

fprintf('[dynamic] Done.\n');
fprintf('[dynamic]   dyn_M.m    —  M = dyn_M(q1,q2,q3)\n');
fprintf('[dynamic]   dyn_C.m    —  C = dyn_C(q1,q2,q3,qd1,qd2,qd3)\n');
fprintf('[dynamic]   dyn_G.m    —  G = dyn_G(q1,q2,q3)\n');
fprintf('[dynamic]   dyn_FK.m   —  FK = dyn_FK(q1,q2,q3)\n');
fprintf('[dynamic]   dyn_J.m    —  J = dyn_J(q1,q2,q3)   (6x3)\n');
end

%% ── DH homogeneous transform (local helper) ─────────────────────────────────
function T = dh_mat(a, alpha, d, theta)
%DH_MAT  Standard DH matrix: T = Rz(theta)*Tz(d)*Tx(a)*Rx(alpha)
ct = cos(theta);  st = sin(theta);
ca = cos(alpha);  sa = sin(alpha);
T  = [ct, -st*ca,  st*sa, a*ct;
      st,  ct*ca, -ct*sa, a*st;
       0,     sa,     ca,    d;
       0,      0,      0,    1];
end
