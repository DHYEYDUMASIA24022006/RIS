%% RIS-Assisted Channel Generation
% Generates Rician fading channels for BS-RIS-User links
% and exports to .mat files for Python consumption.
%
% Parameters:
%   M - Number of BS antennas
%   N - Number of RIS elements  
%   K - Number of users
%   K_factor - Rician K-factor (dB)
%   num_realizations - Number of channel realizations
%   output_path - Path to save the generated channels

function channel_gen(M, N, K, K_factor, num_realizations, output_path)
    % Default parameters
    if nargin < 1, M = 4; end
    if nargin < 2, N = 64; end
    if nargin < 3, K = 20; end
    if nargin < 4, K_factor = 10; end  % 10 dB
    if nargin < 5, num_realizations = 1000; end
    if nargin < 6, output_path = 'channel_data.mat'; end
    
    % System setup
    f_c = 2.4e9; % Carrier frequency (Hz)
    c = 3e8; % Speed of light
    lambda = c / f_c; % Wavelength
    d = lambda / 2; % Antenna spacing
    
    % Position of BS, RIS, and Users (3D coordinates in meters)
    % Assuming BS is at origin, RIS is at (d_BR, 0, H_RIS)
    pos_BS = [0; 0; 10]; 
    pos_RIS = [50; 20; 10];
    
    % Users distributed randomly
    % x in [40, 60], y in [-10, 10], z = 1.5
    pos_Users = zeros(3, K);
    pos_Users(1, :) = 40 + 20 * rand(1, K);
    pos_Users(2, :) = -10 + 20 * rand(1, K);
    pos_Users(3, :) = 1.5 * ones(1, K);
    
    % Distances
    d_BR = norm(pos_BS - pos_RIS);
    d_RU = zeros(K, 1);
    d_BU = zeros(K, 1);
    for k = 1:K
        d_RU(k) = norm(pos_RIS - pos_Users(:, k));
        d_BU(k) = norm(pos_BS - pos_Users(:, k));
    end
    
    % Path loss models (simplified, e.g. 30 + 20*log10(d) type)
    % Let's use simple distance-dependent path loss
    % PL(d) = PL_0 + 10 * n * log10(d/d_0)
    % For simplicity, just set standard path loss linear gains
    beta_BR = 10^(-(30 + 22.0 * log10(d_BR)) / 10);
    beta_RU = zeros(K, 1);
    beta_BU = zeros(K, 1);
    for k = 1:K
        beta_RU(k) = 10^(-(30 + 28.0 * log10(d_RU(k))) / 10);
        beta_BU(k) = 10^(-(30 + 35.0 * log10(d_BU(k))) / 10);
    end
    
    % Convert K factor to linear
    K_lin = 10^(K_factor/10); 
    
    % Preallocate output variables
    H_BR = zeros(M, N, num_realizations);
    H_RU = zeros(N, K, num_realizations);
    H_d  = zeros(M, K, num_realizations);
    
    % Arrays for spatial responses
    % BS is ULA
    theta_BS_RIS = pi/4; % AoD from BS to RIS
    a_BS = exp(1j * 2 * pi * (0:M-1)' * d/lambda * sin(theta_BS_RIS));
    
    % RIS is UPA (N = N_x * N_y)
    N_x = sqrt(N);
    N_y = sqrt(N);
    if mod(N_x, 1) ~= 0
        N_x = N; N_y = 1; % Fallback to ULA if not a perfect square
    end
    
    phi_RIS_BS = pi/6; % AoA at RIS from BS azimuth
    theta_RIS_BS = pi/3; % AoA at RIS from BS elevation
    a_RIS_BS = zeros(N, 1);
    for nx = 0:N_x-1
        for ny = 0:N_y-1
            idx = nx * N_y + ny + 1;
            a_RIS_BS(idx) = exp(1j * 2 * pi * d/lambda * (nx * sin(theta_RIS_BS)*cos(phi_RIS_BS) + ny * cos(theta_RIS_BS)));
        end
    end
    
    % LoS component for BS-RIS
    H_BR_LoS = a_BS * a_RIS_BS';
    
    fprintf('Generating %d channel realizations...\n', num_realizations);
    
    for r = 1:num_realizations
        % BS-RIS channel (Rician)
        % H_BR = sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS
        H_BR_NLoS = (randn(M, N) + 1j * randn(M, N)) / sqrt(2);
        H_BR_curr = sqrt(beta_BR) * (sqrt(K_lin/(K_lin+1)) * H_BR_LoS + sqrt(1/(K_lin+1)) * H_BR_NLoS);
        H_BR(:, :, r) = H_BR_curr;
        
        % RIS-User and BS-User channels (Assume Rayleigh fading for simplicity or Rician)
        % Let's use Rayleigh for RIS-User and BS-User
        for k = 1:K
            h_RU_NLoS = (randn(N, 1) + 1j * randn(N, 1)) / sqrt(2);
            H_RU(:, k, r) = sqrt(beta_RU(k)) * h_RU_NLoS;
            
            h_d_NLoS = (randn(M, 1) + 1j * randn(M, 1)) / sqrt(2);
            H_d(:, k, r) = sqrt(beta_BU(k)) * h_d_NLoS;
        end
        
        if mod(r, 100) == 0
            fprintf('  Done %d / %d\n', r, num_realizations);
        end
    end
    
    % Parameters struct for metadata
    params.M = M;
    params.N = N;
    params.K = K;
    params.K_factor = K_factor;
    params.num_realizations = num_realizations;
    params.beta_BR = beta_BR;
    params.beta_RU = beta_RU;
    params.beta_BU = beta_BU;
    
    % Save to output path
    save(output_path, 'H_BR', 'H_RU', 'H_d', 'params');
    fprintf('Successfully saved channels to %s\n', output_path);
end
