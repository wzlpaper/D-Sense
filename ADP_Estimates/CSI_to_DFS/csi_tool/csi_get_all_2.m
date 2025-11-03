% This code was modified by Zhelun Wang based on CSI-Tools.
% Email: wzlpaper@126.com

function [cfr_array_2, timestamp] = csi_get_all_2(filename)
csi_trace = read_bf_file(filename);
isEmptyStruct = cellfun(@(x) isempty(x) || (isstruct(x) && isempty(fieldnames(x))), csi_trace);
csi_trace(isEmptyStruct) = [];
timestamp = zeros(length(csi_trace), 1);
cfr_array = zeros(length(csi_trace), 90);
for k = 1:length(csi_trace)
    csi_entry = csi_trace{k};
    csi_all = get_scaled_csi(csi_entry); 
    csi_all_tx_2=csi_all(2,:,:);
    csi_all_tx_2=squeeze(csi_all_tx_2).';
    csi_tx_2 = [csi_all_tx_2(:,1); csi_all_tx_2(:,2); csi_all_tx_2(:, 3)].';
    timestamp(k) = csi_entry.timestamp_low;
    cfr_array_2(k,:) = csi_tx_2;
end
end