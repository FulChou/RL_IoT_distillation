# 作者：vincent
# code time：2021/12/6 15:33
import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 使得命令行直接调用时，能够访问到我们自定义的tianshou

from tianshou.trainer import test_episode
from tianshou.trainer.utils import test_student_episode
from utils import get_kl
from tianshou.utils import BasicLogger, tqdm_config
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, VectorReplayBuffer
from atari_network import DQN, student_DQN_net1
from atari_wrapper import wrap_deepmind




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=30)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--net-num', type=str, default='net0')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--save-buffer-name', type=str, default=None)
    return parser.parse_args()


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


def get_teacher_policy(args):
    """
    define model and policy return xx_policy
    :param args:
    :return: policy
    """
    teacher_net = DQN(*args.state_shape,
                      args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(teacher_net.parameters(), lr=args.lr)
    teacher_policy = DQNPolicy(teacher_net, optim, args.gamma, args.n_step,
                               target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        teacher_policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    return teacher_policy


def get_student_policy(args):
    """
    define model and policy return xx_policy
    :param args:
    :return: policy
    """
    student_net = student_DQN_net1(*args.state_shape, args.action_shape, args.device).to(args.device)
    student_optim = torch.optim.Adam(student_net.parameters(), lr=args.lr)
    policy_student = DQNPolicy(student_net, student_optim, args.gamma, args.n_step,
                               target_update_freq=args.target_update_freq)  # test  target_update_freq = 0
    return policy_student


def distill_dqn(args=get_args()):
    env = make_atari_env(args)
    print('env reward best', env.spec.reward_threshold)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: make_atari_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # define model and policy
    teacher_policy = get_teacher_policy(args=args)
    student_policy = get_student_policy(args=args)

    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM

    buffer = VectorReplayBuffer(
        args.buffer_size, buffer_num=len(train_envs), ignore_obs_next=True,
        save_only_last_obs=True, stack_num=args.frames_stack)

    # collector
    train_collector = Collector(teacher_policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(teacher_policy, test_envs, exploration_noise=True)
    test_student_collector = Collector(student_policy, test_envs, exploration_noise=True)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}'
    log_path = os.path.join(args.logdir, 'distll', args.task, 'dqn', log_file)
    print('log_path', log_path)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)
    #
    # def save_fn(policy):
    #     print('sava model at: ', os.path.join(log_path, 'policy.pth'))
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    #
    def save_student_policy_fn(policy):
        print('sava model at: ', os.path.join(log_path, 'policy_student.pth'))
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_student.pth'))

    # def stop_fn(mean_rewards):
    #     if env.spec.reward_threshold:
    #         return mean_rewards >= env.spec.reward_threshold
    #     elif 'Pong' in args.task:
    #         return mean_rewards >= 20
    #     else:
    #         return False
    #
    # def train_fn(epoch, env_step):
    #     # nature DQN setting, linear decay in the first 1M steps
    #     if env_step <= 1e6:
    #         eps = args.eps_train - env_step / 1e6 * \
    #               (args.eps_train - args.eps_train_final)
    #     else:
    #         eps = args.eps_train_final
    #     policy.set_eps(eps)
    #     policy_student.set_eps(eps)
    #     logger.write('train/eps', env_step, eps)
    #
    def test_fn(epoch, env_step):
        teacher_policy.set_eps(args.eps_test)
        student_policy.set_eps(args.eps_test)




    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        student_policy.eval()
        student_policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size, buffer_num=len(test_envs),
                ignore_obs_next=True, save_only_last_obs=True,
                stack_num=args.frames_stack)
            collector = Collector(student_policy, test_envs, buffer,
                                  exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num,
                                            render=args.render)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)

    # really distill
    train_collector.collect(n_step=args.batch_size * args.training_num)
    gradient_step = 0
    teacher_policy.eval()
    teacher_policy.set_eps(args.eps_test)
    episode_per_test = args.test_num
    reward_metric = None
    env_step = 0
    best_student_epoch, best_student_reward, best_student_reward_std = -1, 0, 0
    for epoch in range(args.epoch):
        # 是否需要先去 collect 然后再 forward，直接与环境交互即，collect时候记录结果不好吗？ 去看看tianshou源码
        with tqdm.tqdm(
                total=args.step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                result = train_collector.collect(n_step=args.step_per_collect)  # collect
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                for i in range(round(args.update_per_step * result["n/st"])):
                    gradient_step += 1
                    # compute loss
                    batch, indice = train_collector.buffer.sample(args.batch_size)

                    student_res = student_policy.forward(batch)
                    teacher_res = teacher_policy.forward(batch)
                    student_mean = student_res.logits
                    teacher_mean = teacher_res.logits
                    stds = torch.tensor([1e-6] * len(teacher_mean[0]), device=args.device, dtype=torch.float)
                    stds = torch.stack([stds for _ in range(len(teacher_mean))])
                    loss = get_kl([teacher_mean, stds], [student_mean, stds])
                    logger.log_update_data({'kl_loss:': loss}, gradient_step)
                    # print('kl loss：', loss)
                    # TODO: add loss log：
                    student_policy.optim.zero_grad()
                    loss.backward()
                    student_policy.optim.step()
                if t.n <= t.total:
                    t.update()

            # test student policy
            test_time = time.perf_counter()
            test_result = test_episode(teacher_policy, test_collector, test_fn, epoch,
                                       episode_per_test, logger, env_step, reward_metric)
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            print('test time:', time.perf_counter() - test_time,'rew:', rew)
            #  最优teacher 处理？ 不需要

            # test student_policy
            test_time = time.perf_counter()
            test_student_result = test_student_episode(student_policy, test_student_collector, test_fn, epoch,
                                                       episode_per_test, logger, env_step, reward_metric)
            rew_student, rew_std_student = test_student_result["rew"], test_student_result["rew_std"]
            print('student test time:', time.perf_counter() - test_time, 'student-rew: ', rew_student)
            if best_student_epoch < 0 or best_student_reward < rew_student:
                best_student_epoch, best_student_reward, best_student_reward_std = epoch, rew_student, rew_std_student
                if save_student_policy_fn:
                    save_student_policy_fn(student_policy)

        # TODO: add student_policy test log
    watch()


if __name__ == '__main__':
    distill_dqn(get_args())
