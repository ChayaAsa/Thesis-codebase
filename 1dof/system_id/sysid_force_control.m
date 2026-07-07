%% Open-Loop Force Control System Identification
% Plant:  tau_cmd_Nm (input) --> f_raw_N (output)
% System: motor torque command -> actual end-effector force
%
% Experiment structure detected in data:
%   Phase 1 (t~10-41s):   staircase ramp up   0 -> +0.2 Nm
%   Phase 2 (t~41-67s):   staircase ramp down  0 -> -0.2 Nm
%   Phase 3 (t~67-end):   step pulses / pseudo-random steps
%
% Outputs:
%   - Static gain curve (nonlinearity check)
%   - Non-parametric frequency response (SPA / ETFE)
%   - Parametric transfer function model (tfest)
%   - Model validation plots

clear; clc; close all;

%% 1. Load data
if ~isdeployed
    cd(fileparts(matlab.desktop.editor.getActiveFilename));
end
csv_file = fullfile(fileparts(matlab.desktop.editor.getActiveFilename), './open loop_r/sysid_20260602_164620.csv');
T = readtable(csv_file);

t       = T.time_s;
tau_cmd = T.tau_cmd_Nm;   % INPUT:  commanded torque [Nm]
f_raw   = T.f_raw_N;      % OUTPUT: measured force   [N]
f_filt  = T.f_filt_N;     % filtered force (for comparison)
f_des   = T.f_des_N;      % desired force command
tau_meas= T.tau_meas_Nm;  % actual motor torque

fprintf('Data loaded: %d samples, t = %.2f to %.2f s\n', ...
    height(T), t(1), t(end));

%% 2. Overview plot
figure('Name','Raw Data Overview','NumberTitle','off');
subplot(3,1,1);
plot(t, tau_cmd, 'b'); ylabel('tau\_cmd [Nm]'); grid on; title('Open-Loop SysID Data');
subplot(3,1,2);
plot(t, f_raw, 'r', t, f_filt, 'k--');
ylabel('Force [N]'); legend('f\_raw','f\_filt'); grid on;
subplot(3,1,3);
plot(t, f_des, 'g');
ylabel('Force_des [N]'); xlabel('Time [s]'); grid on;

%% 3. Static characterization (staircase region: Phase 1 & 2)
% Use only the staircase ramps (t = 10 -> ~67 s)
static_mask = (t >= 10.9) & (t <= 67.0);

% For each constant tau_cmd level, compute steady-state mean force
tau_levels = unique(round(tau_cmd(static_mask), 4));
f_ss_mean  = zeros(size(tau_levels));
f_ss_std   = zeros(size(tau_levels));

for k = 1:numel(tau_levels)
    idx = static_mask & (abs(tau_cmd - tau_levels(k)) < 1e-4);
    % Use only the last 60% of each step to avoid transients
    sub_f = f_raw(idx);
    n_trim = round(0.4 * numel(sub_f));
    steady = sub_f(n_trim+1:end);
    f_ss_mean(k) = mean(steady);
    f_ss_std(k)  = std(steady);
end

figure('Name','Static Gain Curve','NumberTitle','off');
errorbar(tau_levels, f_ss_mean, f_ss_std, 'o-b', 'LineWidth', 1.5);
hold on;
% Fit a linear gain to the static data (skip zero command)
nz = tau_levels ~= 0;
p = polyfit(tau_levels(nz), f_ss_mean(nz), 1);
tau_fit = linspace(min(tau_levels), max(tau_levels), 100);
plot(tau_fit, polyval(p, tau_fit), 'r--', 'LineWidth', 1.5);
yline(0, 'k:'); xline(0, 'k:');
xlabel('tau\_cmd [Nm]'); ylabel('Steady-State f\_raw [N]');
title(sprintf('Static Gain: K_{static} = %.3f N/Nm', p(1)));
legend('Measured (mean±std)', sprintf('Linear fit  K=%.3f', p(1)));
grid on;

fprintf('\n--- Static Analysis ---\n');
fprintf('Static gain K = %.4f N/Nm  (offset = %.4f N)\n', p(1), p(2));
fprintf('Equivalent: 1 N desired force requires ~%.4f Nm torque\n', 1/p(1));
fprintf('NOTE: Check for dead-zone / stiction near zero command\n\n');

%% 4. Resample to uniform time grid (required for System ID Toolbox)
% Original data has near-uniform ~10 ms sampling; resample to exact grid
dt_approx = median(diff(t));
dt = round(dt_approx * 1000) / 1000;   % round to nearest ms
fprintf('Detected sample interval: %.4f s  (%.1f Hz)\n', dt, 1/dt);

t_uniform = (t(1) : dt : t(end))';
tau_uniform = interp1(t, tau_cmd, t_uniform, 'linear');
f_uniform   = interp1(t, f_raw,   t_uniform, 'linear');

%% 5. Select active excitation window for dynamic sysid
% Use Phase 3 (step pulses, more dynamic content): t = ~67 s onward
% You can also try the full dataset — uncomment to switch
t_start_dyn = 67.0;
t_end_dyn   = t(end);

dyn_mask = (t_uniform >= t_start_dyn) & (t_uniform <= t_end_dyn);
tau_dyn  = tau_uniform(dyn_mask);
f_dyn    = f_uniform(dyn_mask);
t_dyn    = t_uniform(dyn_mask);

fprintf('Dynamic window: %.1f to %.1f s  (%d samples)\n', ...
    t_dyn(1), t_dyn(end), numel(t_dyn));

%% 6. Remove mean / detrend (operate on deviations from equilibrium)
tau_dyn_d = detrend(tau_dyn, 0);   % remove mean
f_dyn_d   = detrend(f_dyn, 0);

%% 7. Create iddata object
data_dyn = iddata(f_dyn_d, tau_dyn_d, dt, ...
    'Name', 'Force Control Open Loop', ...
    'InputName',  'tau\_cmd', ...
    'InputUnit',  'Nm', ...
    'OutputName', 'f\_raw', ...
    'OutputUnit', 'N', ...
    'TimeUnit',   's');

% Split into estimation (70%) and validation (30%) sets
n_total = size(data_dyn, 1);
n_est   = round(0.7 * n_total);
data_est = data_dyn(1:n_est);
data_val = data_dyn(n_est+1:end);

fprintf('Estimation set: %d samples | Validation set: %d samples\n', n_est, n_total-n_est);

%% 8. Non-parametric frequency response estimate
% SPA: spectral analysis (smoothed periodogram)
% ETFE: empirical transfer function estimate

% Adjust window for data length (longer window = finer freq resolution)
spa_win = min(512, floor(n_est/4));

fprintf('\nComputing non-parametric frequency response...\n');
G_spa  = spa(data_est, spa_win);
G_etfe = etfe(data_est, spa_win);

figure('Name','Non-Parametric Frequency Response','NumberTitle','off');
subplot(2,1,1);
h1 = bodeplot(G_spa, G_etfe);
setoptions(h1, 'FreqUnits', 'Hz', 'MagUnits', 'dB', 'PhaseUnits', 'deg');
legend('SPA (smoothed)', 'ETFE (raw)');
title('Bode Plot — Non-Parametric Estimate');
grid on;

% Check coherence (> 0.8 = reliable frequency range)
subplot(2,1,2);
mscohere(tau_dyn_d, f_dyn_d, spa_win, [], [], 1/dt);
title('Coherence (tau\_cmd -> f\_raw)');
ylabel('Coherence'); xlabel('Frequency [Hz]');
yline(0.8, 'r--', 'threshold 0.8'); grid on;

%% 9. Parametric model fitting — Transfer Function

% Try 1st order + delay: K / (tau_1*s + 1)
% Try 2nd order:         K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

opt_tf = tfestOptions('EnforceStability', true, 'InitializeMethod', 'all');
opt_tf.Display = 'off';

fprintf('\nFitting transfer function models...\n');

% First order (np poles, nz zeros)
sys_tf1  = tfest(data_est, 1, 0, opt_tf);   % 1p 0z
sys_tf1z = tfest(data_est, 1, 1, opt_tf);   % 1p 1z (with zero)
sys_tf2  = tfest(data_est, 2, 0, opt_tf);   % 2p 0z
sys_tf2z = tfest(data_est, 2, 1, opt_tf);   % 2p 1z

models_tf = {sys_tf1, sys_tf1z, sys_tf2, sys_tf2z};
names_tf  = {'TF 1p0z', 'TF 1p1z', 'TF 2p0z', 'TF 2p1z'};

fprintf('\n%-12s  %8s  %8s  %8s\n', 'Model', 'FitEst%', 'FitVal%', 'AIC');
fprintf('%s\n', repmat('-',1,44));
for k = 1:numel(models_tf)
    c_est = compare(data_est, models_tf{k}); fit_e = c_est.Report.Fit.FitPercent;
    c_val = compare(data_val, models_tf{k}); fit_v = c_val.Report.Fit.FitPercent;
    aic_v = aic(models_tf{k});
    fprintf('%-12s  %8.1f  %8.1f  %8.1f\n', names_tf{k}, fit_e, fit_v, aic_v);
end

%% 10. Parametric model fitting — State Space (ARX fallback also shown)

fprintf('\nFitting state space models...\n');
opt_ss = ssestOptions('EnforceStability', true);
opt_ss.Display = 'off';

sys_ss1 = ssest(data_est, 1, opt_ss);
sys_ss2 = ssest(data_est, 2, opt_ss);
sys_ss3 = ssest(data_est, 3, opt_ss);

models_ss = {sys_ss1, sys_ss2, sys_ss3};
names_ss  = {'SS order 1', 'SS order 2', 'SS order 3'};

for k = 1:numel(models_ss)
    c_est = compare(data_est, models_ss{k}); fit_e = c_est.Report.Fit.FitPercent;
    c_val = compare(data_val, models_ss{k}); fit_v = c_val.Report.Fit.FitPercent;
    aic_v = aic(models_ss{k});
    fprintf('%-12s  %8.1f  %8.1f  %8.1f\n', names_ss{k}, fit_e, fit_v, aic_v);
end

%% 11. ARX model (fast, no iterations — good sanity check)
fprintf('\nFitting ARX models...\n');
arx_orders = [1 1 1; 2 1 1; 2 2 1; 3 2 1];  % [na nb nk]
for k = 1:size(arx_orders, 1)
    ord = arx_orders(k,:);
    m   = arx(data_est, ord);
    c_v = compare(data_val, m); fit_v = c_v.Report.Fit.FitPercent;
    fprintf('ARX [%d %d %d]   FitVal = %.1f%%\n', ord(1), ord(2), ord(3), fit_v);
end

%% 12. Select best model and validate
% -- CHANGE THIS to the model with best validation fit --
sys_best = sys_tf2;   % <-- update after reviewing table above
name_best = 'TF 2p0z';

fprintf('\n=== Best model: %s ===\n', name_best);
sys_best

% Discrete -> continuous if needed
if ~isempty(sys_best.Ts) && sys_best.Ts > 0
    sys_best_c = d2c(sys_best, 'tustin');
    fprintf('Continuous-time equivalent:\n');
    sys_best_c
end

%% 13. Validation: compare output
figure('Name','Model Validation — Output Fit','NumberTitle','off');
compare(data_val, sys_tf1, sys_tf2, sys_ss2);
title('Validation Set: Measured vs Simulated');
legend('Measured', names_tf{1}, names_tf{3}, names_ss{2});

%% 14. Residual analysis (white noise residuals = model is adequate)
figure('Name','Residual Analysis','NumberTitle','off');
resid(data_val, sys_best);
title(sprintf('Residuals — %s', name_best));

%% 15. Bode plot of best parametric model vs non-parametric
figure('Name','Final Bode: Parametric vs Non-Parametric','NumberTitle','off');
bode_opt = bodeoptions;
bode_opt.FreqUnits = 'Hz';
bode_opt.MagUnits  = 'dB';
bode(G_spa, sys_tf1, sys_tf2, sys_ss2, bode_opt);
legend('SPA (non-param)', names_tf{1}, names_tf{3}, names_ss{2});
title('Bode Comparison: Parametric vs Non-Parametric');
grid on;

%% 16. Step response of best model
figure('Name','Step Response','NumberTitle','off');
step(sys_best);
title(sprintf('Step Response — %s', name_best));
xlabel('Time [s]'); ylabel('Force [N] per Nm input');
grid on;

%% 17. Summary
fprintf('\n======= IDENTIFICATION SUMMARY =======\n');
fprintf('Plant:         tau_cmd_Nm  -->  f_raw_N\n');
fprintf('Static gain K: %.4f N/Nm\n', p(1));
fprintf('Best model:    %s\n', name_best);
fprintf('DC gain check: %.4f N/Nm\n', dcgain(sys_best));
fprintf('\nNOTES:\n');
fprintf('  - Check static gain curve for dead-zone / stiction near zero\n');
fprintf('  - Coherence plot shows reliable freq range for model fitting\n');
fprintf('  - If fit %% < 60, try: larger excitation, more order, or ARX\n');
fprintf('  - tau_cmd = J * f_des => J = 1/K_static = %.4f Nm/N\n', 1/p(1));
fprintf('=======================================\n');
