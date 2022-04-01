function [ C ] = multcols(A,b)

C=A*spdiags(b',0,length(b),length(b));

end