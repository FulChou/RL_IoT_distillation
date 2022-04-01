function [sol,info] = POL21row(x, lambda, param)
%PROX_L21 Proximal operator with row group L21 norm
%   .. sol = argmin_{z} 0.5*||x - z||_2^2 + gamma * ||x||_21
t1=tic;
if  param.evaltrigger
    info.inputl21norm=l21normrow(x);
end


% soft thresholding
S = lambda ./ sqrt(sum(abs(x).^2,2));
sol = sign(x) .* max(abs(x) - S .* abs(x), 0);

if  param.evaltrigger
    info.outputl21norm = l21normrow(sol);
end
info.time = toc(t1);

