% This code is provided by Zhelun Wang.
% Email: wzlpaper@126.com

function save_DFS(csi_path, DFS_path)
% save_DFS  Generate DFS from CSI data files and save as .mat files
%
% Inputs:
%   csi_path - Folder path containing CSI data files (.dat)
%   DFS_path - Folder path to save generated DFS files

    if nargin < 2
        error('Please provide both CSI folder path and DFS save path');
    end

    % Create DFS folder if it does not exist
    if ~exist(DFS_path, 'dir')
        mkdir(DFS_path);
    end

    % Get all .dat files in the CSI folder
    list = dir(fullfile(csi_path, '*.dat'));
    num_files = length(list);

    for o = 1:num_files
        file_path = fullfile(csi_path, list(o).name);

        [doppler_spectrum, freq_bin] = DFS_generation(file_path);

        [~, idx] = max(freq_bin);
        circ_len = length(freq_bin) - idx;
        doppler_spectrum = circshift(doppler_spectrum, [circ_len, 0]);

        [~, file_name, ~] = fileparts(list(o).name);
        output_file_path = fullfile(DFS_path, file_name);
        save(output_file_path, 'doppler_spectrum');

        clearvars -except csi_path DFS_path list num_files;
    end
end