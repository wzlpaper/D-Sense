% This code is provided by Zhelun Wang.
% Email: wzlpaper@126.com

function generate_save_ADP(DFS_list, ADP_path, Time_sampling_rate, D_x, D_y, ADP_Dim, scaling)
% generate_save_ADP Generate and save ADP from DFS data
% Inputs:
%   DFS_list            - Cell array of file paths to DFS MAT-files.
%   ADP_path            - Output folder path where ADP files will be saved.
%   Time_sampling_rate  - Sampling interval in the time dimension.
%   D_x, D_y            - 3D Distance-to-Frequency Translation_Tensor.
%   ADP_Dim             - 2-element vector [height, width], specifying the output ADP size.
%   scaling             - Interpolation method for resizing (e.g., 'nearest', 'bilinear', 'bicubic').
% Outputs:
%   This function does not return variables to the workspace.
%   For each input DFS_list row, it generates and saves a MAT-file containing:
%       ADP   - 4D matrix of size [ADP_Dim(1), ADP_Dim(2), time, 2],
%               where the last dimension corresponds to [ADPx; ADPy].
if ~exist(ADP_path, 'dir')
    mkdir(ADP_path);
end
    for i = 1:size(DFS_list,1)
        tic
        try
            DFS_x_ = load(DFS_list{i, 1}).doppler_spectrum;         % f_min--->f_max (from top to bottom).
            DFS_y_ = flipud(load(DFS_list{i, 2}).doppler_spectrum); % f_max--->f_min (from top to bottom).
            [DFS_x, DFS_y] = padding_DFS_zreos(DFS_x_, DFS_y_);     % time-domain alignment
            % Time sampling.
            [F, T] = size(DFS_x);
            indices = round(1: Time_sampling_rate: T);
            num_indices = length(indices) - 1;
            DFS_profile_x = zeros(F - 1, num_indices); DFS_profile_y = zeros(F - 1, num_indices); 
            % Removing absolute zero frequency.
            zero_frequency = floor(F / 2);
            ADP_xPerspective = zeros(ADP_Dim(1), ADP_Dim(2), size(DFS_profile_x, 2));
            ADP_yPerspective = zeros(ADP_Dim(1), ADP_Dim(2), size(DFS_profile_y, 2));
            for ii = 1:num_indices
                T_0 = (ii - 1) * Time_sampling_rate + 1;
                T_1 = ii * Time_sampling_rate;
                DFS_x_profiles = DFS_x(:, T_0:T_1);
                DFS_y_profiles = DFS_y(:, T_0:T_1);
                DFS_x_profile_ = mean(DFS_x_profiles, 2);
                DFS_y_profile_ = mean(DFS_y_profiles, 2);
                DFS_x_profile_(zero_frequency) = [];
                DFS_y_profile_(zero_frequency) = [];
                DFS_profile_x(:, ii) = DFS_x_profile_;
                DFS_profile_y(:, ii) = DFS_y_profile_;
            end
            for k = 1: size(DFS_profile_x, 2)
                SPP = DFS_profile_x(:, k) * (DFS_profile_y(:, k)).';
                SPP_ = repmat(SPP, 1, 1, size(D_x,3));
                %{
                  y 
               Rx2|                
                  |    ^
                  |    | 
                  |    |      
                  |    | Rx1(f_max--->f_min)     
                  |    |      
                  |    |      
                  |
                  |____________________________x
                 Tx                           Rx1
                %}
                % Eq.(16) (17)
                ADP_x = SPP_ .* D_x;
                ADP_x_ = squeeze(sum(ADP_x, 1)).'; % [columns, sigmas].' =  [sigmas, columns] f_min--->f_max (top to bottom).
                ADP_x_ = imresize(ADP_x_, [ADP_Dim(1), ADP_Dim(2)], scaling);
                ADP_xPerspective(:, :, k) = (ADP_x_ - min(ADP_x_(:))) / (max(ADP_x_(:)) - min(ADP_x_(:)) + eps);
                %{
                  y 
               Rx2|                
                  |
                  |  
                  |          
                  |          
                  |          
                  |          
                  |
                  |       Rx2(f_max--->f_min)
                  |     ——————————————————————>
                  |____________________________x
                 Tx                           Rx1
                %}
                % Eq.(16) (17)
                ADP_y = SPP_ .* D_y;
                ADP_y_ = fliplr(squeeze(sum(ADP_y, 2))); % [rows, sigmas] f_max--->f_min (left to right).
                ADP_y_ = imresize(ADP_y_, [ADP_Dim(1), ADP_Dim(2)], scaling);
                ADP_yPerspective(:, :, k) = (ADP_y_ - min(ADP_y_(:))) / (max(ADP_y_(:)) - min(ADP_y_(:)) + eps);
            end
            if ~isempty(ADP_xPerspective)
                Nan_x = squeeze(any(any(isnan(ADP_xPerspective), 1), 2));
                indices_Nan_x = find(Nan_x);
                ADPx = ADP_xPerspective(:, :, ~ismember(1:size(ADP_xPerspective, 3), indices_Nan_x));
            end
            if ~isempty(ADP_yPerspective)
                Nan_y = squeeze(any(any(isnan(ADP_yPerspective), 1), 2));
                indices_Nan_y = find(Nan_y);
                ADPy = ADP_yPerspective(:, :, ~ismember(1:size(ADP_yPerspective, 3), indices_Nan_y));
            end
            ADP = cat(4, ADPx, ADPy);
            [~, name, ~] = fileparts(DFS_list{i,1});
            Aname = regexprep(name, '-[^-]*$', '');
            ADP_path_ = fullfile(ADP_path, Aname);
            save(ADP_path_, 'ADP');
            time = toc;
            fprintf('ADP generation and saving completed for file "%s", time: %.2f seconds.\n', Aname, time);
        catch ME
            time = toc; 
            warning('ADP generation failed, time: %.2f seconds. Error message: %s', time, ME.message);
        end
    end
end

%% Viewable (ADPx; ADPy)
%
%  implay Parameter (Recommended): 
%
%  Colormap: parula(256)
%  Zoom factor: 800% (Applicable for size [120;120])
%  Frame rate: 100 frames/sec


