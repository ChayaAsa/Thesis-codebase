function build_rrr_sysid_plant()
%BUILD_RRR_SYSID_PLANT  Programmatically build rrr_sysid_plant.slx
%
%   A Simscape Multibody model of the bare 3-DOF (RRR) arm used as the PLANT for
%   system identification.  It is build_rrr_force_control.m stripped of the wall,
%   the contact force and the force-PI loop, and given instead:
%     * three torque-driven revolute joints (sensing position + velocity),
%     * an impedance controller that makes the arm FOLLOW a multisine position
%       reference fed from the workspace (qref_ts, qdref_ts),
%     * To-Workspace logs of q, qd, the applied torque tau, and the reference.
%
%   Run:   build_rrr_sysid_plant            % creates/overwrites the .slx
%   Then:  use run_sysid_sim(...) to excite it and collect data, or
%          sys_id('Plant','simscape') for the whole pipeline.
%
%   Inertias/geometry come from rrr_params.m, so the plant matches the
%   rigidBodyTree the identifier and the controller use.
%
%   See also RUN_SYSID_SIM, SYS_ID, BUILD_RRR_FORCE_CONTROL, SYSID_TRACK_CONTROL.

mdl = 'rrr_sysid_plant';
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'claude force control'));
p = rrr_params();
assignin('base','p',p);

if bdIsLoaded(mdl), close_system(mdl,0); end
if exist([mdl '.slx'],'file'), delete([mdl '.slx']); end
new_system(mdl);
load_system(mdl);
load_system('sm_lib');

addb = @(src,name,pos) add_block(src,[mdl '/' name],'Position',pos);
SM  = 'sm_lib/';
NES = 'nesl_utility/';
SL  = 'simulink/';

%% ===================== Simscape plant blocks ========================
addb([SM 'Frames and Transforms/World Frame'], 'WF', [40 400  80 440]);
addb([NES 'Solver Configuration'],             'SC', [40 480 100 510]);
addb([SM 'Utilities/Mechanism Configuration'], 'MC', [40 560 120 600]);
set_param([mdl '/MC'],'UniformGravity','Constant','GravityVector','p.gravity');

% Revolute joints: torque in, q/w sensed, start at home q0.
jpos = {[260 380 320 440],[460 380 520 440],[660 380 720 440]};
for i = 1:3
    J = sprintf('%s/J%d', mdl, i);
    addb([SM 'Joints/Revolute Joint'], sprintf('J%d',i), jpos{i});
    set_param(J,'TorqueActuationMode','InputTorque', ...
                'SensePosition','on','SenseVelocity','on', ...
                'PositionTargetSpecify','on', ...
                'PositionTargetValue', sprintf('p.q0(%d)',i), ...
                'PositionTargetValueUnits','rad', ...
                'PositionTargetPriority','High');
end

% Rigid transforms placing joints / link-COMs / EE.
rt = @(name,pos) addb([SM 'Frames and Transforms/Rigid Transform'],name,pos);
rt('RT_j2',[360 300 420 340]);  rt('RT_j3',[560 300 620 340]);
rt('RT_l1',[300 480 360 520]);  rt('RT_l2',[500 480 560 520]);
rt('RT_l3',[700 480 760 520]);  rt('RT_ee',[760 360 820 400]);
set_param([mdl '/RT_j2'],'TranslationMethod','Cartesian', ...
    'TranslationCartesianOffset','[0 0 p.L1]', ...
    'RotationMethod','StandardAxis','RotationStandardAxis','+X', ...
    'RotationAngle','-90','RotationAngleUnits','deg');
set_param([mdl '/RT_j3'],'TranslationMethod','Cartesian','TranslationCartesianOffset','[p.L2 0 0]');
set_param([mdl '/RT_l1'],'TranslationMethod','Cartesian','TranslationCartesianOffset','[0 0 p.L1/2]');
set_param([mdl '/RT_l2'],'TranslationMethod','Cartesian','TranslationCartesianOffset','[p.L2/2 0 0]');
set_param([mdl '/RT_l3'],'TranslationMethod','Cartesian','TranslationCartesianOffset','[p.L3/2 0 0]');
set_param([mdl '/RT_ee'],'TranslationMethod','Cartesian','TranslationCartesianOffset','[p.L3 0 0]');

% Link solids + EE sphere (same inertia as the force-control plant).
brick = @(name,pos) addb([SM 'Body Elements/Brick Solid'],name,pos);
brick('Br1',[300 540 360 580]);  brick('Br2',[500 540 560 580]);  brick('Br3',[700 540 760 580]);
set_brick([mdl '/Br1'],'[2*p.r1 2*p.r1 p.L1]','p.m1','p.I1');
set_brick([mdl '/Br2'],'[p.L2 2*p.r2 2*p.r2]','p.m2','p.I2');
set_brick([mdl '/Br3'],'[p.L3 2*p.r3 2*p.r3]','p.m3','p.I3');

addb([SM 'Body Elements/Spherical Solid'],'Ball',[860 360 920 400]);
set_param([mdl '/Ball'],'SphereRadius','p.r_ee','InertiaType','Custom', ...
    'Mass','p.m_ee','CenterOfMass','[0 0 0]', ...
    'MomentsOfInertia','p.I_ee','ProductsOfInertia','[0 0 0]');

%% ===================== Converters ===================================
for i = 1:3
    addb([NES 'Simulink-PS Converter'], sprintf('S2PS_%d',i), [140 280-60*i 200 250-60*i+30]);
    set_param([mdl sprintf('/S2PS_%d',i)],'Unit','N*m');
end
addb([NES 'PS-Simulink Converter'],'PS_q1',[820 200 880 230]); set_param([mdl '/PS_q1'],'Unit','rad');
addb([NES 'PS-Simulink Converter'],'PS_q2',[820 250 880 280]); set_param([mdl '/PS_q2'],'Unit','rad');
addb([NES 'PS-Simulink Converter'],'PS_q3',[820 300 880 330]); set_param([mdl '/PS_q3'],'Unit','rad');
addb([NES 'PS-Simulink Converter'],'PS_w1',[820 130 880 160]); set_param([mdl '/PS_w1'],'Unit','rad/s');
addb([NES 'PS-Simulink Converter'],'PS_w2',[820  90 880 120]); set_param([mdl '/PS_w2'],'Unit','rad/s');
addb([NES 'PS-Simulink Converter'],'PS_w3',[820  50 880  80]); set_param([mdl '/PS_w3'],'Unit','rad/s');

%% ===================== Reference + controller ======================
% Multisine reference (built each run by run_sysid_sim) arrives as workspace
% timeseries.  From Workspace blocks replay them as 3-vectors.
addb([SL 'Sources/From Workspace'],'FW_qref', [1180 260 1260 290]);
set_param([mdl '/FW_qref'],'VariableName','qref_ts','Interpolate','on','OutputAfterFinalValue','Holding final value');
addb([SL 'Sources/From Workspace'],'FW_qdref',[1180 320 1260 350]);
set_param([mdl '/FW_qdref'],'VariableName','qdref_ts','Interpolate','on','OutputAfterFinalValue','Holding final value');

addb([SL 'Signal Routing/Mux'],'Mux_q',[1000 280 1010 340]); set_param([mdl '/Mux_q'],'Inputs','3');
addb([SL 'Signal Routing/Mux'],'Mux_w',[1000 100 1010 160]); set_param([mdl '/Mux_w'],'Inputs','3');

addb([SL 'User-Defined Functions/MATLAB Function'],'Controller',[1360 240 1460 360]);
set_mf_code([mdl '/Controller']);

addb([SL 'Signal Routing/Demux'],'Demux_tau',[1520 280 1530 340]);
set_param([mdl '/Demux_tau'],'Outputs','3');

%% ===================== Logging =====================================
add_tw(mdl,'TW_q',  'q_log',  [1100 360 1160 390]);
add_tw(mdl,'TW_qd', 'qd_log', [1100 160 1160 190]);
add_tw(mdl,'TW_tau','tau_log',[1600 300 1660 330]);
add_tw(mdl,'TW_ref','ref_log',[1300 200 1360 230]);

%% ===================== Wiring : physical (frame) ===================
P = {
  'WF/RConn1','SC/RConn1';
  'WF/RConn1','MC/RConn1';
  'WF/RConn1','J1/LConn1';
  'J1/RConn1','RT_l1/LConn1';   'RT_l1/RConn1','Br1/RConn1';
  'J1/RConn1','RT_j2/LConn1';   'RT_j2/RConn1','J2/LConn1';
  'J2/RConn1','RT_l2/LConn1';   'RT_l2/RConn1','Br2/RConn1';
  'J2/RConn1','RT_j3/LConn1';   'RT_j3/RConn1','J3/LConn1';
  'J3/RConn1','RT_l3/LConn1';   'RT_l3/RConn1','Br3/RConn1';
  'J3/RConn1','RT_ee/LConn1';   'RT_ee/RConn1','Ball/RConn1';
  'S2PS_1/RConn1','J1/LConn2';  'S2PS_2/RConn1','J2/LConn2';  'S2PS_3/RConn1','J3/LConn2';
  'J1/RConn2','PS_q1/LConn1';   'J1/RConn3','PS_w1/LConn1';
  'J2/RConn2','PS_q2/LConn1';   'J2/RConn3','PS_w2/LConn1';
  'J3/RConn2','PS_q3/LConn1';   'J3/RConn3','PS_w3/LConn1';
};
connect(mdl, P);

%% ===================== Wiring : Simulink signals ===================
S = {
  'PS_q1/1','Mux_q/1'; 'PS_q2/1','Mux_q/2'; 'PS_q3/1','Mux_q/3';
  'PS_w1/1','Mux_w/1'; 'PS_w2/1','Mux_w/2'; 'PS_w3/1','Mux_w/3';
  'Mux_q/1','Controller/1';     % q
  'Mux_w/1','Controller/2';     % qdot
  'FW_qref/1','Controller/3';   % qref
  'FW_qdref/1','Controller/4';  % qdref
  'Controller/1','Demux_tau/1';
  'Demux_tau/1','S2PS_1/1'; 'Demux_tau/2','S2PS_2/1'; 'Demux_tau/3','S2PS_3/1';
  % logging
  'Mux_q/1','TW_q/1';
  'Mux_w/1','TW_qd/1';
  'Controller/1','TW_tau/1';
  'FW_qref/1','TW_ref/1';
};
connect(mdl, S);

%% ===================== Model configuration =========================
set_param(mdl,'InitFcn','p = rrr_params;');
set_param(mdl,'SolverType','Variable-step','Solver',p.solver, ...
    'StopTime','30','MaxStep',num2str(p.max_step),'AbsTol','1e-6','RelTol','1e-4');

Simulink.BlockDiagram.arrangeSystem(mdl);
save_system(mdl);
fprintf('Built and saved %s.slx\n', mdl);
end % build_rrr_sysid_plant

function add_tw(mdl, name, var, pos)
add_block('simulink/Sinks/To Workspace',[mdl '/' name],'Position',pos);
set_param([mdl '/' name],'VariableName',var,'SaveFormat','Timeseries','SampleTime','0.01');
end

function set_brick(blk, dims, mass, moi)
set_param(blk,'BrickDimensions',dims,'InertiaType','Custom', ...
    'Mass',mass,'CenterOfMass','[0 0 0]', ...
    'MomentsOfInertia',moi,'ProductsOfInertia','[0 0 0]');
end

function set_mf_code(blk)
code = sprintf([ ...
 'function tau = controller(q, qdot, qref, qdref)\n' ...
 '%%#codegen\n' ...
 'coder.extrinsic(''sysid_track_control'');\n' ...
 'tau = zeros(3,1);\n' ...
 'tau = sysid_track_control(q, qdot, qref, qdref);\n' ...
 'end\n']);
rt = sfroot;
chart = rt.find('-isa','Stateflow.EMChart','Path',blk);
chart.Script = code;
end

function connect(mdl, pairs)
for k = 1:size(pairs,1)
    try
        add_line(mdl, pairs{k,1}, pairs{k,2}, 'autorouting','on');
    catch ME
        error('connect:fail','add_line %s -> %s : %s', pairs{k,1}, pairs{k,2}, ME.message);
    end
end
end
