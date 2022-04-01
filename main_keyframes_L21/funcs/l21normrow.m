function [sol] = l21normrow(X)
%L21 norm row group
%   .. sol =  ||X||_21

sol=sum(sqrt(sum(abs(X).^2, 2)));


