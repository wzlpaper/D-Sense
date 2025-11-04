% This code is provided by Zhelun Wang.
% Email: wzlpaper@126.com

function [distance_frequency, D] = calculate_D(f_min, f_max, R, Rules, L, W, P)
% calculate_D generates the Distance-to-Frequency Translation_Tensor.
%
% Inputs:
%   f_min   - Minimum Doppler frequency (Hz)
%   f_max   - Maximum Doppler frequency (Hz)
%   R       - Doppler frequency resolution
%   Rules   - Expression rule for filling D:
%             'delta'         -> Impulse
%             'N=sigma'       -> Gaussian-PDF
%   L       - Length of the sensing area
%   W       - Width of the sensing area
%   P       - Perspective axis: 'x' or 'y'
%
% Outputs:
%   distance_frequency - S^i'_j,k. Eq.(13) and (14)
%   D                  - Distance-to-Frequency Translation_Tensor

    tic
    N = (f_max - f_min) / R; % Number of Doppler frequency points.
    Intersection = intersection(L, W, N);
    
    %{ 
      Corresponding to the sensing plane.
      y 
    Rx|                
      |
      |
      |
      |
      |________________x
     Tx               Rx
    %}
    X = flipud(squeeze(Intersection(:,:,1))); Y = flipud(squeeze(Intersection(:,:,2)));

    [m, n] = size(X);
    static_point_x = ceil(m / 2); static_point_y = ceil(n / 2);
    distance = sqrt((X - X(static_point_x,static_point_y)).^2+ ...
               (Y - Y(static_point_x,static_point_y)).^2);
    
    % D related to the link perspective. Eq.(13) and (14)
    switch P
        case 'x'
            distance(1:floor(m/2), :) = -distance(1:floor(m/2), :);
            distance = max(min(distance, W/2), -W/2);
        case 'y'
            distance(:, floor(n/2):end) = -distance(:, floor(n/2):end);
            distance = max(min(distance, L/2), -L/2);
    end

    % From distance to frequency.
    distance_min = min(distance(:)); distance_max = max(distance(:));
    % Linear mapping.
    distance_frequency = f_min + (distance - distance_min) * (f_max - f_min) / (distance_max - distance_min);
    mapped_indices = round((distance_frequency - min(distance_frequency(:))) / ...
                          (max(distance_frequency(:)) - min(distance_frequency(:))) * (N-1) + 1);

    % Initialize D.
    D = zeros(N, N, N);

    % Expression rules.
    if exist('Rules', 'var') && ischar(Rules) || isstring(Rules)
        Rules = char(Rules);

        switch Rules
            case 'delta'  % Impulse

                impulse = eye(N);
                for i = 1:N
                    for j = 1:N
                        D(i,j,:) = impulse(mapped_indices(i,j), :);
                    end
                end
                
            otherwise  % Gaussian PDF, format "N=sigma"
                sigma_ = regexp(Rules, '^N=(\d+(\.\d+)?)$', 'tokens');

                if ~isempty(sigma_)
                    sigma = str2double(sigma_{1}{1});

                    if P == 'x'
                        Expect = linspace(0, L, N);

                    else
                        Expect = linspace(0, W, N);
                    end

                    Gaussian_PDF = arrayfun(@(idx) normpdf(Expect, Expect(idx), sigma), 1:N, 'UniformOutput', false);
                    Gaussian_PDF = cell2mat(Gaussian_PDF') ./ max(cell2mat(Gaussian_PDF'), [], 2); % Normalize

                    for i = 1:N
                        for j = 1:N
                            D(i,j,:) = Gaussian_PDF(mapped_indices(i,j), :);
                        end
                    end
                end
        end
    else
        disp('Expression rules invalid');
    end
    time = toc;
    fprintf('D successfully generated, time: %.2f seconds.\n', time);
end

function Intersection = intersection(L, W, N) 
% Inputs: L and W denote the length and width of the sensing space, respectively.
%         Number of Doppler frequency points.
%
% Outputs: Intersection points of ZFCs on different links.

    [X, Y] = meshgrid(0:L/N:L, 0:W/N:W);
    E1_c = L / 2; E2_c = W / 2;

    % Given relations:
    %   1) Semi-latus rectum formula: p = y = b^2 / a
    %   2) Ellipse focal length relation: b^2 = a^2 - c^2
    %
    % Derive an expression for a:
    % Step 1: y = (a^2 - c^2) / a
    % Step 2: a^2 - a*y - c^2 = 0
    % Step 3: a = sqrt(y^2/4 + c^2) + y/2
    
    a1 = sqrt((Y(:, 1).^2)./4 + (E1_c)^2) + Y(:, 1)./2;
    b1 = sqrt(a1.^2 - E1_c^2);
    a2 = sqrt((X(1, :).^2)./4 + (E2_c)^2) + X(1, :)./2;
    b2 = sqrt(a2.^2 - E2_c^2);

    % Eliminate ambiguity on the axis
    a1_ = a1(2:end); b1_ = b1(2:end);
    a2_ = a2(2:end); b2_ = b2(2:end);

    Intersection = zeros(N, N, 2);
    fprintf('Computing the D from different perspective...\n');

    [n, m] = ndgrid(1:N, 1:N);
    a1_n = a1_(n); b1_n = b1_(n);
    a2_m = a2_(m); b2_m = b2_(m);
    [x, y] = arrayfun(@(a1,b1,a2,b2) ...
                      calculate_intersection(a1, b1, a2, b2, E1_c, E2_c), ...
                      a1_n, b1_n, a2_m, b2_m);
    label = (x > 0) & (y > 0);
    Intersection(:,:,1) = x .* label; Intersection(:,:,2) = y .* label;
end

function [x, y] = calculate_intersection(a1, b1, a2, b2, E1_c, E2_c)
% Compute the intersection point of two ellipses (ZFCs)
% Inputs: a1   - Semi-major axis of the first ellipse (x-axis link)
%         b1   - Semi-minor axis of the first ellipse (x-axis link)
%         a2   - Semi-major axis of the second ellipse (y-axis link)
%         b2   - Semi-minor axis of the second ellipse (y-axis link)
%         E1_c - Center offset of the first ellipse along x-axis
%         E2_c - Center offset of the second ellipse along y-axis
%
% Outputs: x - x-coordinate of the intersection point. Returns NaN if
%          no valid intersection is found.
%          y - y-coordinate of the intersection point. Returns NaN if
%          no valid intersection is found.

    function E = ellipse_eqs(xy)

        x = xy(1); y = xy(2);
        E(1) = ((x - E1_c)^2) / a1^2 + y^2 / b1^2 - 1;
        E(2) = x^2 / b2^2 + ((y - E2_c)^2) / a2^2 - 1;
    end

    solve_ = optimoptions('fsolve', 'Display', 'off', ...
        'FunctionTolerance', 1e-10);
    init = [E1_c, E2_c];

    try
        [solve, ~, exitflag] = fsolve(@ellipse_eqs, init, solve_);
        if exitflag > 0
            x = solve(1);
            y = solve(2);
        else
            x = nan;
            y = nan;
        end
    catch
        x = nan;
        y = nan;
    end
end

