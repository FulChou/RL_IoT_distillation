function res = test(B)
B = permute(B, [4 3 2 1]);
% save('input.mat','B')
% load('input.mat', 'B');

shape = size(B);

% reshape erorr
% data = reshape(B, [shape(1),  shape(2), shape(3) * shape(4)]);
% add path
% clear; 清除变量了
list = {'funcs'};
% HOMEDATA = './GT/';
for i = 1:length(list)
    addpath(genpath(list{i}),'-begin');
end
tic % tic表示计时的开始，toc表示计时的结束。

num_frames =  shape(4);
fprintf('num_frames: %.2f',num_frames)
H = shape(1);
W = shape(2);
k = shape(3);
X = zeros(H, W * k,  num_frames, 'uint8');

vec = @(x) x(:); % vec is function
% 若信号是矩阵，则把x矩阵按列拆分后纵向排列成一个大的列向量；
% 若信号是行向量，则相当于转置；若信号是列向量则不变。
scale = 1 / 2;
h = H * scale;
w = W * scale;

Y = zeros(h * w * k, num_frames);
f = 1;
while f <= num_frames
    for i = 1: k
        X(:, (1:W) + (i - 1) * W, f) = B(:, :, i, f);
    end
    Y(:, f) = vec(imresize(im2double(X(:, :, f)), scale));
    f = f + 1;
end

l1 = B(:, :, 1, 1);
l2 = X(:, :, 1);
figure(99)
image(X(:, :, 1))
saveas(gcf,'0.jpg')

% read X(视频帧, Y（视频灰度化，缩放后，纵向读变成列向量
%param.Data=normrows(Data);
%Y=Ys;

% n=Y;
YY = Y; % why？

% divp=0.05;
% minx=0;
% nonzero = 10;
lambda=[1e3, 1e4];
info.maxiter = 5000;
info.regcase='L21';
info.convergetol = 1e-4;
info.rzero=1e-2; %-6,-3
info.evaltrigger=0;
D = YY; % 32400 x 948
eigv = eig(D' * D);
[mm,nn] = size(D);
info.alpha = max(eigv(:)) * 1.01; %1.01

XX = YY;
B = D' * XX / info.alpha;
t = lambda / info.alpha;
H = eye(nn) - D' * D / info.alpha;
WW = D' / info.alpha;


for idx = 1:length(lambda)
    theta = t(idx); % 0.000135221491568400， 0.00135221491568400
    lambda_idx = lambda(idx);
    switch info.regcase
        case {'L21'}
            % initialization
            Zk = zeros(nn, num_frames);  % 948 * 948
            % Main iteration
            zchange = zeros(1, info.maxiter);
            for iter = 1:info.maxiter % iter 5000
                C = B + H * Zk;
                Z = POL21row(C, theta, info);
                zchange(iter) = norm(Zk - Z, 'fro') / norm(Zk,'fro');  % 返回矩阵 X 的 Frobenius 范数
                if zchange(iter) < info.convergetol  % convergence
                    break
                end
                Zk = Z;
            end
    end
    t = size(Z);
    error = norm(XX - XX * Z) / norm(XX);
    sInd = findRep(Z, 0.99);
    ind = rmRep(sInd, XX);
    % [f_measure] = summe_evaluateSummary(Z, 'Jumps', HOMEDATA);
    f_measure = 0.1;
    fprintf('lambda: %.2e, f_measure: %.3f, error: %.2e, cvgiter: %.f\n',lambda_idx, f_measure, error, iter)
    record.f_measure(idx) = idx;
    record.error(idx) = error;
    record.ind{idx} = ind;
    record.Z{idx} = Z;
    record.zchange{idx} = zchange(1:iter);
    record.convergeiter(idx) = iter;
    figure(idx + 1);
    clf  % clf 函数用于清除当前图像窗口
    ht = heatmap(Z(1:100,1:100));
    colormap(gca, 'gray') % 使用 gca 指代当前坐标区。
    saveas(gcf,[int2str(idx+1), '.jpg'])
    toc
end

% drow target image
[~,pos] = max(record.f_measure);
valid_ind = sort(record.ind{pos}); %(sum(Y(:, ind)) > 0);

I = length(valid_ind);
H = h/scale;
W = w/scale;
title = zeros(H, W * k * I, class(X));
for i = 1: I
    title(:, (1: W*k) + (i - 1) * W*k) = X(:, :, valid_ind(i));
end
% TT = title(:,:,1);
figure(1);
clf
image(title);
saveas(gcf,'1.jpg')
res = valid_ind;
end