function [ C ] = multrows(A,b)

C=spdiags(b,0,length(b),length(b))*A; % C = C=spdiags(b,0,length(b),length(b)) creates an length(b)-by-length(b) sparse matrix from the columns of b and places them along the diagonals specified by 0

end