function y = normcols(x)
%NORMCOLS Normalize matrix columns.
%  Y = NORMCOLS(X) normalizes the columns of X to unit length, returning
%  the result as Y.
%
%  See also NORMROWS, MULTROWS, MULTCOLS.

%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  April 2005

y = multcols(x, 1./sqrt(sum(x.^2,1)));

end