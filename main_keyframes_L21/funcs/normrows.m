function y = normrows(x)
%NORMROWS Normalize matrix rows.
%  Y = NORMROWS(X) normalizes the rows of X to unit length, returning the
%  result as Y.
%
%  See also NORMCOLS, MULTROWS, MULTCOLS.

%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  April 2005

y = multrows(x, 1./sqrt(sum(x.^2,2)));

end