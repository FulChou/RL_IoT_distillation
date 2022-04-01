clear
currentpath=cd;
addpath(genpath(currentpath),'-begin');
%% initialization
% Experimental setup
% Data generation paramters
m = 250;       % Size of D
n = 500; N=1; SNRdB=20;
divp=0.05; minx=0;
info.maxiter = 5000;
caselist=[' HT ';' ST ';'Prox';'DCA '];
casenumber=size(caselist,1);
lambda=[5e-5, 6e-5, 5e-5, 5e-5, 5e-5];%ISTA 0.9 0.7 0.5 IHTA
info.convergetol = 1e-6; info.rzero=1e-3; 
%% Generate Dictionary
% normal Gaussian Dic
D = randn(m,n);
D = D*diag(1./sqrt(sum(D.*D)*n));
[X, Zorig, Xorig] = gererateDataBnD(D, N, divp, minx, SNRdB);
Noiseb=D*Zorig-X;
SNR=sum(Xorig.^2)/sum(Noiseb.^2);%norm(btrain)/norm(Noiseb);
SNRdB=10*log10(SNR);

eigv=eig(D'*D);
info.alpha=max(eigv(:))*1.01;
B=D'*X/info.alpha;t=(lambda'/info.alpha*ones(1,n))';
H=eye(n)-D'*D/info.alpha; W=D'/info.alpha;
Z= zeros(n,casenumber);
nonzero=nnz(Zorig);

[Z(:,1)]=Hard(X,W,H,t(:,1),10,info,nonzero);%l0 norm
[~,Z(:,2),~,~]=ISTA(X,W,H,t(:,2),10,info);%l1 norm
[Z(:,3)]=Proximallog(X,W,H,t(:,1),lambda(:,1),10,info);%proximal log
[~,Z(:,4),~,~]=DCDLlog(X,W,H,t(:,1),lambda(:,1),10,info);%DClog





