function cohrecord = draw_coherence(D, name)

D = D*diag(1./sqrt(sum(D.*D)));
%similarity of data
DTD=abs(D'*D);
Diag1=diag(ones(size(DTD,1),1));
DTD=DTD.*(1-Diag1)+Diag1;
Sim=DTD(logical(triu(ones(size(DTD)))));
Sim(Sim==1)=[];
cohermean=sum(Sim)/size(Sim,1);
coherstd=std(Sim);
cohrecord=[cohermean,coherstd];
barinternal=0.05;barSimD=zeros(1/barinternal,2);
for i=1:(1/barinternal)
barSimD(i,1)=(i-1)*barinternal;
sim1=Sim(Sim<(i*barinternal));
sim1=sim1(sim1>=((i-1)*barinternal));
barSimD(i,2)=size(sim1,1);
end
barSimDall{1}=barSimD;
figure();clf
bar(barSimD(:,1)+barinternal/2, barSimD(:,2), 1);
xlabel('Coherence')
ylabel('Number')
title(sprintf('Average coherence: %.2f %.2f', cohermean, coherstd))
saveas(gcf, name)

end