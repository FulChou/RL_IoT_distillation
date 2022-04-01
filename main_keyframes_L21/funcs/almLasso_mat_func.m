%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points
% Y: DxN data matrix
% affine: true if enforcing the affine constraint, false otherwise
% thr1: stopping threshold for the coefficient error ||Z-C||
% thr2: stopping threshold for the linear system error ||Y-YZ||
% maxIter: maximum number of iterations of ALM
% C2: NxN sparse coefficient matrix
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [C2,Err] = almLasso_mat_func(Y,affine,alpha,q,thr,maxIter,verbose)


if (nargin < 2)
    % default subspaces are linear
    affine = false; 
end
if (nargin < 3)
    % default regularizarion parameters
    alpha = 5;
end
if (nargin < 4)
    % default norm in L1/Lq optimization program
    q = 2;
end
if (nargin < 5)
    % default coefficient error threshold to stop ALM
    % default linear system error threshold to stop ALM
    thr = 1*10^-7; 
end
if (nargin < 6)
    % default maximum number of iterations of ALM
    maxIter = 5000; 
end
if (nargin < 7)
    % reporting iterations and errors
    verbose = true; 
end


if (length(alpha) == 1)
    alpha1 = alpha(1);
    alpha2 = alpha(1);
elseif (length(alpha) == 2)
    alpha1 = alpha(1);
    alpha2 = alpha(2);
end

if (length(thr) == 1)
    thr1 = thr(1);
    thr2 = thr(1);
elseif (length(thr) == 2)
    thr1 = thr(1);
    thr2 = thr(2);
end

[D,N] = size(Y);

% setting penalty parameters for the ALM
mu1p = alpha1 * 1/computeLambda_mat(Y,affine);
mu2p = alpha2 * 1;

if (~affine)
    % initialization
    mu1 = mu1p;
    mu2 = mu2p;
    P = Y'*Y;
    A = inv(mu1.*P+mu2.*eye(N));
    C1 = zeros(N,N);
    Lambda2 = zeros(N,N);
    err1 = 10*thr1; 
    i = 1;
    % ALM iterations
    while ( err1 > thr1 && i < maxIter )
        % updating Z
        Z = A * (mu1.*P+mu2.*C1-Lambda2);
        % updating C
        C2 = shrinkL1Lq(Z+Lambda2./mu2,1/mu2,q);
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 .* (Z - C2);
        % computing errors
        err1 = errorCoef(Z,C2);
        %
        %mu1 = min(mu1*(1+10^-5),10^2*mu1p);
        %mu2 = min(mu2*(1+10^-5),10^2*mu2p);
        %
        C1 = C2;
        i = i + 1;
        % reporting errors
        if (verbose && mod(i,100)==0)
            fprintf('Iteration %5.0f, ||Z - C|| = %2.5e, \n',i,err1);
        end
    end
    Err = err1;
    if (verbose)
        fprintf('Terminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, \n',i,err1);
    end
else
    % initialization
    mu1 = mu1p;
    mu2 = mu2p;
    P = Y'*Y;
    A = inv(mu1.*P+mu2.*eye(N)+mu2.*ones(N,N));
    C1 = zeros(N,N);
    Lambda2 = zeros(N,N);
    lambda3 = zeros(1,N);
    err1 = 10*thr1; err2 = 10*thr2;
    i = 1;
    % ALM iterations
    while ( (err1 > thr1 || err2 > thr1) && i < maxIter )
        % updating Z
        Z = A * (mu1.*P+mu2.*(C1-Lambda2./mu2)+mu2.*ones(N,N)+repmat(lambda3,N,1));
        % updating C
        C2 = shrinkL1Lq(Z+Lambda2./mu2,1/mu2,q);  
        % updating Lagrange multipliers
        Lambda2 = Lambda2 + mu2 .* (Z - C2);
        lambda3 = lambda3 + mu2 .* (ones(1,N) - sum(Z,1));
        % computing errors
        err1 = errorCoef(Z,C2);
        err2 = errorCoef(sum(Z,1),ones(1,N));
        %
        %mu1 = min(mu1*(1+10^-5),10^2*mu1p);
        %mu2 = min(mu2*(1+10^-5),10^2*mu2p);
        %
        C1 = C2;
        i = i + 1;
        % reporting errors
        if (verbose && mod(i,100)==0)
            fprintf('Iteration %5.0f, ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e, \n',i,err1,err2);
        end
    end
    Err = [err1;err2];
    if (verbose)
        fprintf('Terminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e. \n',i,err1,err2);
    end
end
