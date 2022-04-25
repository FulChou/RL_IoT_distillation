function res = test(B)
% tic表示计时的开始，toc表示计时的结束。

tStart = tic;
B = permute(B, [3 2 1]);
shape = size(B);

% data = reshape(B, [shape(1),  shape(2), shape(3) * shape(4)]);
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
vec = @(x) x(:); % vec is function≈
% 若信号是矩阵，则把x矩阵按列拆分后纵向排列成一个大的列向量；
% 若信号是行向量，则相当于转置；若信号是列向量则不变。
scale = 1 / 2;
h = H * scale;
w = W * scale;

% Y = zeros(h * w * k, num_frames);
Y = zeros(h * w, num_frames);
f = 1;
while f <= num_frames
    %for i = 1: k
    %    X(:, (1:W) + (i - 1) * W, f) = B(:, :, i, f);
    %end
    X(:, :, f) = B(:, :, f);
    Y(:, f) = vec(imresize(im2double(X(:, :, f)), scale));
    f = f + 1;
end

% figure(99);
% image(X(:, :, 1));
% saveas(gcf,'0.jpg');
% read X(视频帧, Y（视频灰度化，缩放后，纵向读变成列向量

YY = Y;
lambda=[1e3];  % , 5000, 1e4
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

fprintf('before loop: %.2f',tEnd = toc(tStart))

for idx = 1: length(lambda)
    theta = t(idx); % 0.000135221491568400， 0.00135221491568400
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
            record_idx = info.maxiter;
            for iter = 1: info.maxiter % iter 5000
                tStart = tic;
                C = B + H * Zk;
                Z = POL21row(C, theta, info); % Z is h_k
                tEnd = toc(tStart);

                time_iter(iter) = tEnd; % record time
                zchange(iter) = norm(Zk - Z, 'fro') / norm(Zk,'fro');  % 返回矩阵 X 的 Frobenius 范数
                
                group_log_regular = 0;
                for i = 1: num_frames
                    group_log_regular = group_log_regular + log(1 + norm(Z(:, i), 2) .^ 2 / theta);
                end
                recon_loss = norm((Y - Y * Z), 'fro') .^ 2;
                recon_loss_iter(iter) = recon_loss;
                constraint_loss = group_log_regular * lambda_idx;
                constraint_loss_iter(iter) = constraint_loss;

                loss_iter(iter) = recon_loss + constraint_loss; % record loss
                if zchange(iter) < info.convergetol  % convergence
                    record_idx = iter;
                    break
                end
                Zk = Z;
            end

            x = 1: 1: record_idx; % x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
            clf
            figure(idx + 100);
            plot(x, time_iter(1: record_idx), '-*'); % 线性，颜色，标记
            xlabel('iter')  % x轴坐标描述
            ylabel('时间（s）') % y轴坐标描述
            ylim tight
            saveas(gcf,[int2str(idx), 'time.jpg'])
            file_time = fopen([int2str(idx), 'Atime.txt'], 'w');
            fprintf(file_time,'%6s %12s\n','iter','time_iter');
            fprintf(file_time,'%6.2f %12.8f\n',[x; time_iter(1: record_idx)]);
            fclose(file_time);

            clf
            figure(idx + 200);
            plot(x, loss_iter(1: record_idx), '-*'); % 线性，颜色，标记
            xlabel('iter')  % x轴坐标描述
            ylabel('loss') % y轴坐标描述
            ylim tight
            saveas(gcf,[int2str(idx), 'loss.jpg'])
            file_loss = fopen([int2str(idx), 'Aloss_iter.txt'], 'w');
            fprintf(file_loss,'%6s %12s\n','iter','loss_iter');
            fprintf(file_loss,'%6.2f %12.8f\n',[x; loss_iter(1: record_idx)]);
            fclose(file_loss);

            clf
            figure(idx + 300);
            plot(x, zchange(1: record_idx), '-*'); % 线性，颜色，标记
            xlabel('iter')  % x轴坐标描述
            ylabel('zchange') % y轴坐标描述
            ylim auto
            % set(gca,'YTick',[0:100:700]) % y轴范围0-700，间隔100
            saveas(gcf, [int2str(idx), 'zchange.jpg'])
            file_z_change = fopen([int2str(idx), 'Azchange.txt'], 'w');
            fprintf(file_z_change,'%6s %12s\n','iter','zchange(iter)');
            fprintf(file_z_change,'%6.2f %12.8f\n',[x; zchange(1: record_idx)]);
            fclose(file_z_change);

            clf
            figure(idx + 400);
            plot(x, recon_loss_iter(1: record_idx), '-*'); % 线性，颜色，标记
            xlabel('iter')  % x轴坐标描述
            ylabel('recon_loss') % y轴坐标描述
            ylim auto
            saveas(gcf, [int2str(idx), 'recon_loss.jpg'])
            file_recon_loss = fopen([int2str(idx), 'Arecon_loss.txt'], 'w');
            fprintf(file_recon_loss,'%6s %12s\n','iter','recon_loss(iter)');
            fprintf(file_recon_loss,'%6.2f %12.8f\n',[x; recon_loss_iter(1: record_idx)]);
            fclose(file_recon_loss);

            clf
            figure(idx + 500);
            plot(x, constraint_loss_iter(1: record_idx), '-*'); % 线性，颜色，标记
            xlabel('iter')  % x轴坐标描述
            ylabel('constraint_loss') % y轴坐标描述
            ylim auto
            saveas(gcf, [int2str(idx), 'constraint_loss.jpg'])
            file_recon_loss = fopen([int2str(idx), 'Aconstraint_loss.txt'], 'w');
            fprintf(file_recon_loss,'%6s %12s\n','iter','constraint_loss(iter)');
            fprintf(file_recon_loss,'%6.2f %12.8f\n',[x; constraint_loss_iter(1: record_idx)]);
            fclose(file_recon_loss);
    end

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
    saveas(gcf,[int2str(idx), 'heatmap.jpg'])

end

    % drow target image
    for idx_key = 1: length(lambda)
        % [~,pos] = record.f_measure(idx_key);
        valid_ind = sort(record.ind{idx_key}); % (sum(Y(:, ind)) > 0);
        I = length(valid_ind);
        H = h/scale;
        W = w/scale;
        % title = zeros(H, W * k * I, class(X));
        title = zeros(H, W * I, class(X));
        for i = 1: I
            % title(:, (1: W*k) + (i - 1) * W*k) = X(:, :, valid_ind(i));
            title(:, (1: W) + (i - 1) * W) = X(:, :, valid_ind(i));
        end
        % TT = title(:,:,1);
        figure(1);
        clf
        image(title);
        saveas(gcf,[int2str(idx), '-', int2str(i), 'output.jpg'])
        res = valid_ind;
    end
toc
end