function tau = sysid_track_control(q, qdot, qref, qdref)
%SYSID_TRACK_CONTROL  Impedance tracking + gravity comp for the sys-id plant.
%
%   tau = SYSID_TRACK_CONTROL(q, qdot, qref, qdref) returns the 3x1 joint torque
%   that makes the Simscape arm FOLLOW the multisine reference (qref, qdref)
%   while staying bounded under gravity:
%
%       tau = g(q)  +  Kp.*(qref - q)  +  Kd.*(qdref - qdot)
%
%   This is the Simscape analogue of the hardware "Method A" in generate_sysid.py
%   (impedance-tracked multisine).  The STIFFNESS DOES NOT BIAS the identification:
%   we identify on the MEASURED q, qd, qdd and the MEASURED (applied) torque,
%   never on the reference.  Gravity comp just keeps tracking tight so the joints
%   sweep their intended range instead of sagging.
%
%   Stateless map (gains + rigidBodyTree cached in persistents) -> safe to call
%   as coder.extrinsic from a MATLAB Function block, exactly like rrr_control.m.
%
%   See also BUILD_RRR_SYSID_PLANT, RUN_SYSID_SIM, RRR_CONTROL.

persistent robot Kp Kd
if isempty(robot)
    robot = rrr_robot(rrr_params());
    Kp = [120; 120; 90];      % tracking stiffness [N*m/rad]
    Kd = [6;   6;   5];       % tracking damping   [N*m*s/rad]
end

q     = q(:);   qdot  = qdot(:);
qref  = qref(:); qdref = qdref(:);

g   = gravityTorque(robot, q);                 % 3x1 gravity torque [N*m]
tau = g + Kp.*(qref - q) + Kd.*(qdref - qdot); % 3x1 applied joint torque
end
