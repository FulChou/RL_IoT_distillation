function sparsity = Sparsity_Hoyer(CoefMatrix)

% CoefMatrix_full = full(CoefMatrix);
[I,J]=size(CoefMatrix);
sqrtn = sqrt(I);
Sparsity=zeros(1,J);
for i=1:J
   norm1 = norm(CoefMatrix(:,i),1);
   norm2 = norm(CoefMatrix(:,i),2);
   if norm1==0
       Sparsity(i)=0;
   else
       Sparsity(i)=(sqrtn-norm1/norm2)/(sqrtn-1);
   end
end
sparsity = mean(Sparsity);
% CoefMatrix_full = CoefMatrix;
% norm1 = norm(abs(CoefMatrix_full(:)),1);
% norm2 = norm(CoefMatrix_full(:),2);
% num = numel(CoefMatrix_full);
% Sparsity_y1_y2=(sqrt(num)-(norm1/norm2))/(sqrt(num)-1);