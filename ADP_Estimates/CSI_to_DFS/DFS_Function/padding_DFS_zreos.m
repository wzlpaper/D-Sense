% This code is provided by Zhelun Wang.
% Email: wzlpaper@126.com

function [DFS1_, DFS2_] = padding_DFS_zreos(data1, data2)
% padding_DFS_zreos Align the column size (time dimension) of two DFS
% Inputs:
%   data1 - DFS1
%   data2 - DFS2
% Outputs:
%   DFS1_ - zero-padded version of DFS1 (if necessary)
%   DFS2_ - zero-padded version of DFS2 (if necessary)
% The function will zero-pad the DFS with fewer columns so that both
% DFSs have the same number of columns.
    if size(data1, 1) ~= size(data2, 1)
        error('Row count mismatch (frequency dimension) between the two DFSs.');
    end
    if size(data1, 2) > size(data2, 2)
        larger = data1; smaller = data2;
    else
        larger = data2; smaller = data1;
    end
    diff_cols = size(larger, 2) - size(smaller, 2);
    if diff_cols > 0
        smaller = [smaller, zeros(size(smaller, 1), diff_cols)];
    end
    if size(data1, 2) >= size(data2, 2)
        DFS1_ = larger; DFS2_ = smaller;
    else
        DFS1_ = smaller; DFS2_ = larger;
    end
end
