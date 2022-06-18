ip = '127.0.0.1';
port = 5005;
% 构造服务器端tcpip对象
server = tcpserver(ip, port,"Timeout",40);
global  log_path lambda threshold maxiter


server.ConnectionChangedFcn = @requestDataCommand;
configureCallback(server,"terminator", @readArduinoData);
configureTerminator(server,"LF");

function requestDataCommand(src,~)
    global  log_path lambda threshold maxiter
    if src.Connected
        % 设置lambda，xxx等信息
        % Display the server object to see that Arduino client has connected to it.
        disp("The Connected and ClientAddress properties of the tcpserver object show that the Arduino is connected.")
        disp(src.BytesAvailableFcnMode)
        lambda = 500;
        threshold = 0.92;
        maxiter = 200;
        % Request the Arduino to send data.
       
        
    end
end

function readArduinoData(src,~)
% Read the sine wave data sent to the tcpserver object.

global log_path lambda threshold maxiter
recive_tmie = tic;
src.UserData = readline(src);
input = jsondecode(src.UserData);
input = permute(input, [3 2 1]);
toc(recive_tmie)

% matlab_start = tic;
tStart = tic;

shape = size(input);

% add path
list = {'funcs'};
for i = 1:length(list)
    addpath(genpath(list{i}),'-begin');
end
num_frames = shape(3);
% fprintf('num_frames: %.2f',num_frames)
H = shape(1);
W = shape(2);
X = zeros(H, W,  num_frames, 'uint8');
vec = @(x) x(:); % vec is function 若信号是矩阵，则把x矩阵按列拆分后纵向排列成一个大的列向量；若信号是行向量，则相当于转置；若信号是列向量则不变。
scale = 1 / 2;
h = H * scale;
w = W * scale;
Y = zeros(h * w, num_frames);
% double_state = zeros(h * w * 2, num_frames);
f = 1;
while f <= num_frames
    %for i = 1: k
    %    X(:, (1:W) + (i - 1) * W, f) = input(:, :, i, f);
    %end
    X(:, :, f) = input(:, :, f);
    Y(:, f) = vec(imresize(im2double(X(:, :, f)), scale));
%     if f >= 2
%             double_state(:, f-1) = [Y(:, f-1); Y(:, f)];
%         end
    f = f + 1;
end
% double_state(1:(h*w), f-1) = Y(:,f-1);


YY = Y;
lambda=[lambda];  % 5000, 1e4
% info.maxiter = 5000;
info.maxiter = maxiter;
info.regcase='L21';
info.convergetol = 4e-4;
% info.rzero=1e-2; %-6,-3
info.evaltrigger=0;
D = YY; %32400 x 948

% draw_coherence(D, [log_path, '/pre_coherence.jpg']);
eigv = eig(D' * D);
[mm,nn] = size(D);
info.alpha = max(eigv(:)) * 1.01; %1.01
XX = YY;
B = D' * XX / info.alpha;
t = lambda / info.alpha;
H = eye(nn) - D' * D / info.alpha;
% WW = D' / info.alpha;
% tEnd = toc(tStart);
% fprintf('before loop: %.2f \n', tEnd)

t_loop_start = tic;
for idx = 1: length(lambda)
    theta = t(idx);  
    lambda_idx = lambda(idx);
    switch info.regcase
        case {'L21'}
            % initialization
            Zk = zeros(nn, num_frames);  
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
    end
%     t_heatmap_start = tic;

    % error = norm(XX - XX * Z) / norm(XX);
    sInd = findRep(Z, threshold);
%     ind = rmRep(sInd, XX);

    fprintf('lambda: %.2e, cvg iter: %.f\n',lambda_idx, iter)
    record.f_measure(idx) = idx;
    % record.error(idx) = error;
    record.ind{idx} = sInd;
    record.Z{idx} = Z;
    record.zchange{idx} = zchange(1:iter);
    record.convergeiter(idx) = iter;

    %figure(idx + 1);
    %clf  % clf 函数用于清除当前图像窗口
    %ht = heatmap(Z(1:100,1:100));
    %colormap(gca, 'gray') % 使用 gca 指代当前坐标区。
    %saveas(gcf,[int2str(idx), 'heatmap.jpg'])

    % t_heatmap_end = toc(t_heatmap_start);
    %fprintf('heatmap time: %.2f \n', t_heatmap_end)
end

% t_loop_end = toc(t_loop_start);
%fprintf('loop time: %.2f \n', t_loop_end)

    % drow target image
    t_draw_target_start = tic;    
    for idx_key = 1: length(lambda)
        % [~,pos] = record.f_measure(idx_key);
        valid_ind = sort(record.ind{idx_key}); 
        % draw_coherence(valid_ind, [log_path, '/coherence.jpg']);
        res = valid_ind;
        % fprintf('res len %.2f \n', length(valid_ind));
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
    % fprintf('drow target time: %.2f \n', t_draw_target_end)

% matlab_end = toc(matlab_start);
% file_time = fopen(['a-3matlab_time.txt'], 'a');
% % fprintf(file_time,'%6s \n','time');
% fprintf(file_time,'%12.8f\n',[matlab_end]);
% fclose(file_time);

toc(recive_tmie)
write(src, num2str(res), "string");
end