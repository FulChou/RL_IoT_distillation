%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns the regularization constant of the L1/Lq minimization
% Y: DxN data matrix
% lambda: regularization parameter of lambda*||C||_{1,q} + 0.5 ||Y-YC||_F^2
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function lambda = computeLambda_mat(Y,affine)

[~,N] = size(Y);

if (~affine)
    T = zeros(N,1);
    for i = 1:N
        yi = Y(:,i);
        T(i) = norm(yi' * Y);
    end
    lambda = max(T);
else
    T = zeros(N,1);
    for i = 1:N
        yi = Y(:,i);
        ymean = mean(Y,2);
        T(i) = norm(yi'*(ymean*ones(1,N)-Y));
    end
    lambda = max(T);
end
