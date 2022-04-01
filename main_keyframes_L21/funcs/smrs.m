%--------------------------------------------------------------------------
% This function takes the data matrix and the value of the regularization
% parameter to compute the row-sparse matrix indicating the representatives
% Y: DxN data matrix of N data points in D-dimensional space
% alpha: regularization parameter, typically in [2,50]
% r: project data into r-dim space if needed, enter 0 to use original data
% verbose: enter true if want to see the iterations information
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [repInd,C,error] = smrs(Y,alpha,r,verbose)

if (nargin < 2)
    alpha = 5;
end
if (nargin < 3)
    r = 0;
end
if (nargin < 4)
    verbose = true;
end

q = 2;
regParam = [alpha alpha];
affine = true;
thr = 1 * 10^-7;
maxIter = 5000;
thrS = 0.99; thrP = 0.95;
N = size(Y,2);

Y = Y - repmat(mean(Y,2),1,N);
if (r >= 1)
    [~,S,V] = svd(Y,0);
    r = min(r,size(V,1));
    Y = S(1:r,1:r) * V(:,1:r)';
end

C = almLasso_mat_func(Y,affine,regParam,q,thr,maxIter,verbose);
  error=norm(Y-Y*C);
sInd = findRep(C,thrS,q);
repInd = rmRep(sInd,Y,thrP);