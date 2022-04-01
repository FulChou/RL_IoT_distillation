function [ind, W,H,error, obj,parm] = DCA_func_fastnmf_4_k(Y,param)
% FASTNMF Fast least-squares non-negative matrix factorization 
%  Approximates V by W*H s.t. W,H>=0 with l1 constraint on H
%
% Usage
%   [W,H] = fastnmf(V,N,[options])
%
% Input
%   Y           Data matrix (I x J)
%   N           Number of components 
%   options
%     maxiter   Number of iterations (default 100)
%     iiter     Number of inner iterations (default 10)
%     W         Initial value for W (I x N)
%     H         Initial value for H (N x J)
%
% Output
%   W           Samples of W (I x N x M)
%   H           Samples of H (N x J x M)
%   obj         Error of V-W*H 


 %U0=rand(size(param.TrueDictionary));
 %U0=normrows(U0);
 Y=normrows(Y);
 U0=param.TrueDictionary;
 V0=rand(size(param.Coefs));
% V0=param.Coefs;
% V0=normrows(V0);
param.TrueDictionary=normrows(param.TrueDictionary);
 
W0=param.TrueDictionary+param.pp*U0;
 %H0=param.Coefs+param.pp*V0;
%creat the initial value 

A_temp=U0*V0;
B_temp=param.Data'*A_temp;
C_temp=trace(B_temp);
D_temp=(norm(A_temp,'fro'))^2;
E_temp=C_temp/D_temp;
%W0=E_temp^(0.5)*U0;
H0=E_temp^(0.5)*V0;

% W0=param.Data(:,1:size(param.TrueDictionary,2));

[I,J] = size(Y);

N=(size(W0,2));

obj = zeros(1,param.maxiter);
sparsityIter=[];
costtimeIter = [];
ratioIter = [];
iterNums = [];
SNR_1=[];
erro=[];
ER1=[];
SNR1=[];
Det=[];

tStart=tic;

% calcuate the initial H_k
 if det(H0*H0')==0;
      H_k=zeros(size(H0*H0',1),size(H0,2));
      H_k(:)=0;    
    elseif det(H0*H0')>0;
        temp=H0*H0';
        temp_det=det(temp);
        temp_inv=inv(temp);
        temp_inv_H=temp_inv*H0;
        H_k=2*param.alpha* temp_det*temp_inv_H;
 end
 %%
 
 
 H=H0;
 W=W0;
 

%%-----------main loop---------------------------------
tic
for iter = 1:param.maxiter
    
%---------- update W ---------------------------------

%[V_1]=Copy_of_nnls_DCA(Y',H',W,param.mu,param.maxiter);
 
%%
 
A=Y';
U=H';
V_0=W;
 
MM=U'*U;
NN=A'*U;
[n,r]=size(V_0);

% 
%  NablaF=V_0*MM-NN;
%  
% % DI=diag(MM);
% % Delta_V=[];
% % k=0;
% % flag=0;
% % for i=1:n
% %     for j=1:r
% %         
% %        V_1(i,j)=min(V_0(i,j),NablaF(i,j)/max(DI(j,1),param.mu)); %The updated value.
% % 
% %        Delta_V(i,j)=V_0(i,j)-V_1(i,j); %displacement.
% %        
% %        delta_F(i,j)= NablaF(i,j)*Delta_V(i,j)-0.5*MM(j,j)*(Delta_V(i,j))^2;
% %     end
% % end
% %        %F=norm(A-U*V_0','fro')-param.alpha*det(U'*U); %the value of object function. 
% % 
% % for i=1:n
% %       
% %        
% %        deltaF_max=max(delta_F(i,:));
% %       
% %        k=1;
% %        
% %        V_1max=max(V_1(i,:));
% %        
% %        V_temp=0.5*(V_1max+V_1(i,:));% the updated t(v) and
% %        
% %        deltaJ =(norm(A(:,i)-U*V_temp',2))^2-(norm(A(:,i)-U*V_0(i,:)',2))^2;
% %        
% %        if deltaJ>deltaF_max
% %            
% %        flag=1;
% %        
% %        end
% %        if flag==1, k=1;       
% %        
% %        D(i,j)=delta_F(i,j);
% %         
% %         else 
% %      
% %        D(i,j)=0;
% %         end
% % 
% % end
% D=V_0;
% D_1=max(D*MM,param.mu);
%     
% for i=1:n
%     for j=1:r
%         if D(i,j)==0
%            V_l(i,j)=V_0(i,j);
%         end
%         if D(i,j)>0
%            V_1(i,j)=V_0(i,j)-D(i,j).*NablaF(i,j)/D_1(i,j);
%         end
%        
%     end
% end
%  
%  %W=abs(V_1); 
%  W=V_1; 
%  W=normcols(W);
W=param.TrueDictionary;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------- improve H -------using w_(k) to updata H-------

%%======algorithm 1 by using quadratic function properties====%%

% M=Y';V=W';U=H';
% MV=Y'*W;
% H_T=H_k';
% 
% for jj=1:N   
%      temp=MV(jj,:)+0.5*H_T(jj,:);
%      if W'*W==0;
%         YTY=M(jj,:)*Y(:,jj);
%         U(jj,:)= temp'*(1/YTY);
%      else
%         temp_1=inv(W'*W);
%         U(jj,:) =abs(temp*temp_1);         
%      end
% end

               %%================END===============%%
               %%================END===============%%    
               
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
%%=========algorithm 2 by using HALS=======%%

M=Y';V=W';U=H';
R_1=V*V';
MV=M*V';
VV=R_1;




for jj=1:N
    Eab_1=R_1;
    Eab_1(jj,jj) = 0;
    U(:,jj) = max(  (   (   MV(:,jj)-U*Eab_1(:,jj) + 0.5* (H_k(jj,:))'  )/VV(jj,jj)  ), param.tol  );

end

H=U';
%H=normrows(H);

%=========Calcuating H_k=========

if det(H*H')==0;
   H_k=zeros(size(H*H',1),size(H,2));
   H_k(:)=0;    
elseif det(H*H')>0;
   temp=H*H';
   temp_det=det(temp);
   temp_inv=inv(temp);
   temp_inv_H=temp_inv*H;
   H_k=2*param.alpha* temp_det*temp_inv_H;
end

  
%% Evaluate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    det2=det(H*H');
    Det=[Det det2];

    obj(1,iter)=0.5*sum(sum((Y-W*H).^2));
    
%     Erro = norm(normrows(Y)-normrows(W*H),'fro')^2-param.alpha*det(H*H');

    
    ER=norm(param.Coefs-H,'fro')^2/(norm(param.Coefs,'fro')^2);
%     SNR0=10*log10(1/ER);
    ER1=[ER1 ER];
%     SNR1=[SNR1 SNR0];
    
    
    
    
    Erro = norm(Y-W*H,'fro')^2/(norm(Y,'fro')^2); 
    SNR=10*log10(1/Erro);
    SNR_1=[SNR_1 SNR];
    erro=[erro Erro];
    
    [ratio,ErrorBetweenDictionaries] = I_findDistanseBetweenDictionaries_1(normcols(param.TrueDictionary),normcols(W));
%     if ratio==100
%         break;
%     end
%     if rem(iter,100)==0
%         disp(['The FASTNMF-L1 algorithm retrived ',num2str(ratio),' atoms from the original dictionary']);
%     end
    sparsity = Sparsity_Hoyer(H);

    sparsityIter=[sparsityIter sparsity];

    costtime=toc(tStart);
    costtimeIter = [costtimeIter costtime];
    
    ratioIter = [ratioIter ratio];
    
    iterNums = [iterNums iter];
    

end
toc
t1=0;
t1=toc-tic
parm.erro=erro;
parm.SNR=SNR_1;
parm.sparsityIter=sparsityIter;
parm.costtimeIter = costtimeIter;
parm.ratioIter = ratioIter;
parm.iterNums = iterNums;
parm.Det=Det;

parm.ER1=ER1; %Error of 
% parm.SNR1=SNR1;% relative error 
  error=norm(Y-Y*H);
sInd = findRep(H,0.99);
ind = rmRep(sInd,Y);
% H=zeros(J,1);
% for ii=1:size(ind,2)
%     t=ind(ii);
%     H(t)=1;
% end


%% BP ALGORITHM FOR SPARSE CODING
% 
% A=param.TrueDictionary;
% H_l1=[];
% parm.sparsity_l1=[];
% tic
% 
% for t=1:1000
% H_l1(:,t)= SolveBP(A, param.Data(:,t), 50,100);
% sparsity_l1 = Sparsity_Hoyer(H_l1);
% parm.sparsity_l1=[parm.sparsity_l1 sparsity_l1];
% end
% figure(1)
% plot(parm.sparsity_l1)
% toc
