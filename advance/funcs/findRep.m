%--------------------------------------------------------------------------
% This function takes the coefficient matrix with few nonzero rows and
% computes the indices of the nonzero rows
% C: NxN coefficient matrix
% thr: threshold for selecting the nonzero rows of C, typically in [0.9,0.99]
% q: value of q in the L1/Lq minimization program in {1,2,inf}
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function cssInd = findRep(C,thr,q)

if (nargin < 4)
    q = 2;
end
if (nargin < 3)
    thr = 0.9;
end

N = size(C,1);

r = zeros(1,N);
for i = 1:N
    r(i) = norm(C(i,:),q);
end
[nrm,nrmInd] = sort(r,'descend');
nrmSum = 0;
for j = 1:N
    nrmSum = nrmSum + nrm(j);
    if (nrmSum / sum(nrm) > thr)
        break;
    end
end
cssInd = nrmInd(1:j);