% This code was modified by Zhelun Wang based on CSI-Tools.
% Email: wzlpaper@126.com
% If you are using the Widar3.0 dataset, you should run csi_get_all_1() because the CSI of this
% dataset only has one transmitting antenna. If your dataset specification is a 2x3 antenna array, 
% you can use csi_get_all_2() to extract the CSI of the second transmitting antenna.

function [cfr_array_1, timestamp] = csi_get_all_1(filename)
csi_trace = read_bf_file(filename);
isEmptyStruct = cellfun(@(x) isempty(x) || (isstruct(x) && isempty(fieldnames(x))), csi_trace);
csi_trace(isEmptyStruct) = [];
timestamp = zeros(length(csi_trace), 1);
cfr_array = zeros(length(csi_trace), 90);
for k = 1:length(csi_trace)
    csi_entry = csi_trace{k};
    csi_all = get_scaled_csi(csi_entry); 
    csi_all_tx_1=csi_all(1,:,:);
    csi_all_tx_1=squeeze(csi_all_tx_1).';
    csi_tx_1 = [csi_all_tx_1(:,1); csi_all_tx_1(:,2); csi_all_tx_1(:, 3)].';
    timestamp(k) = csi_entry.timestamp_low;
    cfr_array_1(k,:) = csi_tx_1;
end

end
