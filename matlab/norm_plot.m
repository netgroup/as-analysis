function [new_lcc_cov] = norm_plot(max_len,lcc_cov)
%NORM_PLOT Summary of this function goes here
%   Detailed explanation goes here
len = length(lcc_cov);
xq = 1:len/max_len:len;
new_lcc_cov = interp1(lcc_cov, xq);
end

