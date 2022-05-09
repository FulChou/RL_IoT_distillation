function res = test(input, log_path)

% tic表示计时的开始，toc表示计时的结束。
tic

tStart = tic;
input = permute(input, [3 2 1]);
% save('input.mat', 'input');  %存输入
shape = size(input);
% data = reshape(input, [shape(1),  shape(2), shape(3) * shape(4)]);
% add path
% clear; 清除变量了
list = {'funcs'};
% HOMEDATA = './GT/';

for i = 1:length(list)
    addpath(genpath(list{i}),'-begin');
end

num_frames = shape(3);
% fprintf('num_frames: %.2f',num_frames)
H = shape(1);
W = shape(2);

% k = shape(3);
% X = zeros(H, W * k,  num_frames, 'uint8');

X = zeros(H, W,  num_frames, 'uint8');
vec = @(x) x(:); % vec is function 若信号是矩阵，则把x矩阵按列拆分后纵向排列成一个大的列向量；若信号是行向量，则相当于转置；若信号是列向量则不变。
scale = 1 / 2;
h = H * scale;
w = W * scale;
Y = zeros(h * w, num_frames);
double_state = zeros(h * w * 2, num_frames);
f = 1;
while f <= num_frames
    %for i = 1: k
    %    X(:, (1:W) + (i - 1) * W, f) = input(:, :, i, f);
    %end
    X(:, :, f) = input(:, :, f);
    Y(:, f) = vec(imresize(im2double(X(:, :, f)), scale));
    if f >= 2
            double_state(:, f-1) = [Y(:, f-1); Y(:, f)];
        end
    f = f + 1;
end
double_state(1:(h*w), f-1) = Y(:,f-1);


% figure(99);
% image(X(:, :, 1));
% saveas(gcf,'0.jpg');
% read X(视频帧, Y（视频灰度化，缩放后，纵向读变成列向量

YY = Y;
lambda=[1e3];  % , 5000, 1e4
% info.maxiter = 5000;
info.maxiter = 200;
info.regcase='L21';
info.convergetol = 4e-4;
info.rzero=1e-2; %-6,-3
info.evaltrigger=0;
D = YY; %32400 x 948

draw_coherence(D, [log_path, '/pre_coherence.jpg']);
eigv = eig(D' * D);
[mm,nn] = size(D);
info.alpha = max(eigv(:)) * 1.01; %1.01
XX = YY;
B = D' * XX / info.alpha;
t = lambda / info.alpha;
H = eye(nn) - D' * D / info.alpha;
WW = D' / info.alpha;
tEnd = toc(tStart);
% fprintf('before loop: %.2f \n', tEnd)

t_loop_start = tic;
for idx = 1: length(lambda)
    theta = t(idx);  % 0.000135221491568400， 0.00135221491568400
    lambda_idx = lambda(idx);
    switch info.regcase
        case {'L21'}
            % initialization
            Zk = zeros(nn, num_frames);  % 948 * 948
            % Main iteration
            zchange = zeros(1, info.maxiter);
            loss_iter = zeros(1, info.maxiter);
            recon_loss_iter = zeros(1, info.maxiter);
            constraint_loss_iter = zeros(1, info.maxiter);
            time_iter = zeros(1, info.maxiter);
            record_idx = 500;

            key_loop_start = tic;
            for iter = 1: info.maxiter
                tStart = tic;
                C = B + H * Zk;
                Z = POL21row(C, theta, info); % Z is h_k
                tEnd = toc(tStart);

                time_iter(iter) = tEnd;  %record time
                zchange(iter) = norm(Zk - Z, 'fro') / norm(Zk,'fro');  % 返回矩阵 X 的 Frobenius 范数

%                 recon_loss = norm((Y - Y * Z), 'fro') .^ 2;
%                 recon_loss_iter(iter) = recon_loss;
%                 constraint_loss = l21normrow(Z) * lambda_idx;
%                 constraint_loss_iter(iter) = constraint_loss;
%                 loss_iter(iter) = recon_loss + constraint_loss; % record loss

                if zchange(iter) < info.convergetol  % convergence
                    record_idx = iter;
                    break
                end
                Zk = Z;
            end

            key_loop_end = toc(key_loop_start);
            %fprintf('key loop time: %.2f \n', key_loop_end)

%             if record_idx >= 3
%                 x = 1: 1: record_idx; % x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
%                 clf
%                 figure(idx + 200);
%                 plot(x, loss_iter(1: record_idx), '-*'); % 线性，颜色，标记
%                 xlabel('iter')  % x轴坐标描述
%                 ylabel('loss') % y轴坐标描述
%                 ylim tight
%                 saveas(gcf,[int2str(idx), 'loss.jpg'])
%                 file_loss = fopen([int2str(idx), 'Aloss_iter.txt'], 'w');
%                 fprintf(file_loss,'%6s %12s\n','iter','loss_iter');
%                 fprintf(file_loss,'%6.2f %12.8f\n',[x; loss_iter(1: record_idx)]);
%                 fclose(file_loss);
% 
%                 clf
%                 figure(idx + 400);
%                 plot(x, recon_loss_iter(1: record_idx), '-*'); % 线性，颜色，标记
%                 xlabel('iter')  % x轴坐标描述
%                 ylabel('recon_loss') % y轴坐标描述
%                 ylim auto
%                 saveas(gcf, [int2str(idx), 'recon_loss.jpg'])
%                 file_recon_loss = fopen([int2str(idx), 'Arecon_loss.txt'], 'w');
%                 fprintf(file_recon_loss,'%6s %12s\n','iter','recon_loss(iter)');
%                 fprintf(file_recon_loss,'%6.2f %12.8f\n',[x; recon_loss_iter(1: record_idx)]);
%                 fclose(file_recon_loss);
% 
%                 clf
%                 figure(idx + 500);
%                 plot(x, constraint_loss_iter(1: record_idx), '-*'); % 线性，颜色，标记
%                 xlabel('iter')  % x轴坐标描述
%                 ylabel('constraint_loss') % y轴坐标描述
%                 ylim auto
%                 saveas(gcf, [int2str(idx), 'constraint_loss.jpg'])
%                 file_recon_loss = fopen([int2str(idx), 'Aconstraint_loss.txt'], 'w');
%                 fprintf(file_recon_loss,'%6s %12s\n','iter','constraint_loss(iter)');
%                 fprintf(file_recon_loss,'%6.2f %12.8f\n',[x; constraint_loss_iter(1: record_idx)]);
%                 fclose(file_recon_loss);
%             end

    end
    t_heatmap_start = tic;

    % error = norm(XX - XX * Z) / norm(XX);
    sInd = findRep(Z, 0.99);
    ind = rmRep(sInd, XX);

    fprintf('lambda: %.2e, cvg iter: %.f\n',lambda_idx, iter)
    record.f_measure(idx) = idx;
    % record.error(idx) = error;
    record.ind{idx} = ind;
    record.Z{idx} = Z;
    record.zchange{idx} = zchange(1:iter);
    record.convergeiter(idx) = iter;

    %figure(idx + 1);
    %clf  % clf 函数用于清除当前图像窗口
    %ht = heatmap(Z(1:100,1:100));
    %colormap(gca, 'gray') % 使用 gca 指代当前坐标区。
    %saveas(gcf,[int2str(idx), 'heatmap.jpg'])

    t_heatmap_end = toc(t_heatmap_start);
    %fprintf('heatmap time: %.2f \n', t_heatmap_end)
end

t_loop_end = toc(t_loop_start);
%fprintf('loop time: %.2f \n', t_loop_end)

    t_draw_target_start = tic;

    % drow target image
    for idx_key = 1: length(lambda)
        % [~,pos] = record.f_measure(idx_key);

        valid_ind = sort(record.ind{idx_key}); % (sum(Y(:, ind)) > 0);
        draw_coherence(valid_ind, [log_path, '/coherence.jpg']);
        res = valid_ind;
        %fprintf('res len %.2f \n', length(valid_ind));
        %I = length(valid_ind);
        %H = h / scale;
        %W = w / scale;
        %title = zeros(H, W * I, class(X));
        %for i = 1: I
        %    % title(:, (1: W*k) + (i - 1) * W*k) = X(:, :, valid_ind(i));
        %    title(:, (1: W) + (i - 1) * W) = X(:, :, valid_ind(i));
        %end
        %figure(1);
        %clf
        %image(title);
        %saveas(gcf,[int2str(idx), '-', int2str(i), 'output.jpg'])

    end
    t_draw_target_end = toc(t_draw_target_start);
    %fprintf('drow target time: %.2f \n', t_draw_target_end)

toc
end