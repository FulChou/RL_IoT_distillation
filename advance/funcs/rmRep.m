%--------------------------------------------------------------------------
% This function takes the data matrix and the indices of the
% representatives and removes the representatives that are too close to
% each other
% Y: DxN data matrix of N data points in D-dimensional space
% sInd: indices of the representatives
% thr: threshold for pruning the representatives, typically in [0.9,0.99]
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function ind = rmRep(sInd,Y,thr)

if (nargin < 3)
    thr = 0.95;
end

Ys = Y(:,sInd);
Ns = size(Ys,2);
d = zeros(Ns,Ns);
for i = 1:Ns-1
    for j = i+1:Ns
        d(i,j) = norm(Ys(:,i) - Ys(:,j));
    end
end
d = d + d';
[dsort,dsorti] = sort(d,'descend');

pind = 1:Ns;
for i = 1:Ns
    if (~isempty(find(pind==i,1)));
        cum = 0;
        t = 0;
        while (cum <= thr * sum(dsort(:,i)))
            t = t + 1;
            cum = cum + dsort(t,i);
        end
        pind = setdiff(pind,setdiff(dsorti(t:end,i),1:i));
    end
end
ind = sInd(pind);