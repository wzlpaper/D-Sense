% This code is provided by Zhelun Wang.
% Email: wzlpaper@126.com

function [DFS_list__, Length, Width]  = area_link(path, area, task)
% area_link  Return receiver file lists and scene dimensions for a given area index
% Inputs:
%     path   - Folder path containing the DFS files
%     area   - Area index, valid range 1–9
%
% Outputs:
%     DFS_list - File names of the receivers corresponding to the specified area
%     Length   - Scene length for the area (x-direction is defined as length)
%     Width    - Scene width for the area (y-direction is defined as width)

    names = {dir(fullfile(path, '*.mat')).name};
    
    RxID_ = regexp(names, '-(r\d)\.mat$', 'tokens', 'once'); RxID = [RxID_{:}];
    Rxs = {'r1','r2','r3','r4','r5','r6'};
    names_class = cellfun(@(r) names(strcmp(RxID,r))', Rxs, 'UniformOutput', false);
    if strcmp(task, 'Gesture')
        Rx_Idx = [1 4 0.5 0.5;  % Area1
                  2 4 1.4 0.5;  % Area2
                  1 5 0.5 1.4;  % Area3
                  3 4 2.0 0.5;  % Area4
                  1 6 0.5 2.0;  % Area5
                  2 5 1.4 1.4;  % Area6
                  3 5 2.0 1.4;  % Area7
                  2 6 1.4 2.0;  % Area8
                  3 6 2.0 2.0]; % Area9

    elseif strcmp(task, 'Gait')
        Rx_Idx = [1 4 1.7 1.65;  % Area1
                  2 4 3.4 1.65;  % Area2
                  1 5 1.7 3.30;  % Area3
                  3 4 4.6 1.65;  % Area4
                  1 6 1.7 4.40;  % Area5
                  2 5 3.4 3.30;  % Area6
                  3 5 4.6 3.30;  % Area7
                  2 6 3.4 4.40;  % Area8
                  3 6 4.6 4.40]; % Area9
    end
    
    assert(isscalar(area) && area>=1 && area<=size(Rx_Idx,1), 'area must be integer between 1 and 9');
    DFS_list = names_class(Rx_Idx(area, 1:2));
    n_max = max(cellfun(@numel, DFS_list)); 
    DFS_list = cellfun(@(c) [c; repmat({''}, n_max-numel(c), 1)], ...
                       DFS_list, 'UniformOutput', false);
    DFS_list = [DFS_list{:}]; 
    [DFS_list_, idx_] = match_DFS(DFS_list);
    % Avoid mismatches, and check again...
    disp(idx_)
    DFS_list__ = fullfile(path, DFS_list_);
    [Length, Width] = deal(Rx_Idx(area, 3), Rx_Idx(area, 4));
end

function [DFS_list_, idx_] = match_DFS(DFS_list)

    DFS_list_ = {};
    
    n = size(DFS_list,1);
    DFS_left = cell(n,1);
    for i = 1:n
        left = DFS_list{i,1};
    
        if isempty(left)
            DFS_left{i} = '';
            continue;
        end
    
        idx = find(left=='-', 1, 'last');
        if isempty(idx)
            DFS_left{i} = left;
        else
            DFS_left{i} = left(1:idx-1);
        end
    end
    
    DFS_right = cell(n,1);
    for i = 1:n
        right = DFS_list{i,2};
    
        if isempty(right)
            DFS_right{i} = '';
            continue;
        end
    
        idx = find(right=='-', 1, 'last');
        if isempty(idx)
            DFS_right{i} = right;
        else
            DFS_right{i} = right(1:idx-1);
        end
    end
    
    for i = 1:n
        left_key = DFS_left{i};

        if isempty(left_key)
            continue;
        end
        
        match_idx = find(strcmp(DFS_right, left_key));

        if ~isempty(match_idx)
            temp = [repmat({DFS_list{i,1}}, numel(match_idx), 1), DFS_list(match_idx,2)];
            DFS_list_ = [DFS_list_; temp]; %#ok<AGROW>
        end
    end
    idx_ = check(DFS_list_);
end

function idx = check(DFS_list_)
    m = size(DFS_list_,1);
    idx = zeros(m,1);
    count = 0; 
    for i = 1:m
        left_ = DFS_list_{i,1};
        right_ = DFS_list_{i,2};

        idx_left_ = find(left_=='-',1,'last');
        key_left = left_; 
        if ~isempty(idx_left_), key_left = left_(1:idx_left_-1); end

        idx_right_ = find(right_=='-',1,'last');
        key_right = right_;
        if ~isempty(idx_right_), key_right = right_(1:idx_right_-1); end

        if ~strcmp(key_left, key_right)
            count = count + 1;
            idx(count) = i;
        end
    end
    idx = idx(1:count);
end
