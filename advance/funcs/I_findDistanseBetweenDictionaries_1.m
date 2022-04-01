%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new)
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end

for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [minValue,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
%------------²ÉÓÃdicdist of Alain---------
% for i = 1:size(original,2)
%     original(:,i) = sign(original(1,i))*original(:,i);
% end
% for i = 1:size(original,2)
%     errorOfElement = 1-abs(new(:,i)'*original);
%     [errorOfElement,ind(i)] = min( errorOfElement);
%     totalDistances = totalDistances+errorOfElement;
%     catchCounter = catchCounter+(errorOfElement<0.01);
% end

ratio = 100*catchCounter/size(original,2);