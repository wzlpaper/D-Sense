% This code is provided by Zhelun Wang.
% Email: wzlpaper@126.com

clear;
clc;
close all;
%{
 Set all parameters, and add the "CSI_to_DFS" and "generate_ADP" subfolders 
 under the same directory to the path.
 When running this code, you will obtain the ADP in the folder specified by params{8}. 

 Each ADP will have a size of t*params{4}*2, representing all time steps with 
 the two-link perspectives scaled to params{4}.
 
 At the same time, the frequency-distance translation tensor D generated 
 during the execution will be automatically saved. This tensor D can 
 be directly loaded for future use.

 We have uploaded all the usage data related to D-Sense on IEEE DataPort, 
 and you can also download it from IEEE DataPort. For more details, 
 please refer to the D-Sense paper.
%}

% Parameter Settings.
params = {1;                    % Doppler frequency resolution.
          100;                  % DFS time dimension sampling. (to reduce computational power consumption)
          'N=sigam' or 'delta'; % Spatial representation rules for distance. ('N=sigam' or 'delta', where 'N=sigam' and the equal sign has no spaces)
          [20, 20];             % Dimensions of the ADP.
          '1~9';                % Area index "1~9". (See the paper)
          'CSI\';               % CSI save path.
          'DFS\';               % DFS save path. 
          'ADP\';               % ADP save path.
          'D\';                 % D save path.
          -60;                  % Minimum Doppler frequency. (-DFS_generation.uppe_stop)
          60;                   % Maximum Doppler frequency. (DFS_generation.uppe_stop)
          'Gesture' or 'Gait'   % Task
          }; 

% From CSI to DFS.
save_DFS(params{6}, params{7})

% Extract the DFS filenames corresponding to the sensing area.
[DFS_list, Length, Width] = area_link(params{7}, params{5}, params{12});
if ~exist(params{9}, 'dir')
    mkdir(params{9});
end

% Load or calculate D. (The naming rules for D are as follows: length-width-resolution-perspective-sigma/'delta')
if startsWith(params{3}, 'N=')
    sigma = str2double(strrep(params{3},'N=',''));
else
    sigma = params{3};
end
D_path = fullfile(params{9}, [num2str(Length), '-', num2str(Width), '-', num2str(params{1}), '-', num2str(sigma) '.mat']);
if exist(D_path, 'file') && any(strcmp({whos('-file', D_path).name}, 'D'))
    load(D_path, 'D');
    fprintf('Found "D" in file: %s\n', D_path);
else 
    [distance_frequency_x, D_x] = calculate_D(params{10}, ...
                                              params{11}, ...
                                              params{1}, ...
                                              params{3}, ...                
                                              Length, Width, ...
                                              'x' ...  % The perspective of interest. ('x', 'y')
                                              );
    [distance_frequency_y, D_y] = calculate_D(params{10}, ...
                                              params{11}, ...
                                              params{1}, ...
                                              params{3}, ...                
                                              Length, Width, ...
                                              'y' ...  % The perspective of interest. ('x', 'y')
                                              );
    D = cat(4, D_x, D_y);
    save(D_path, 'D');
end
D_x = squeeze(D(:,:,:,1)); D_y = squeeze(D(:,:,:,2));

ADP_save_path = fullfile(params{8}, ['Area' num2str(params{5})], num2str(sigma), filesep);
generate_save_ADP(DFS_list, ADP_save_path, params{2}, D_x, D_y, params{4}, 'bilinear');
