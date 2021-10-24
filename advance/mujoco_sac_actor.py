#!/usr/bin/env python3

import os
import gym
import torch
import pprint
import datetime
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) # 使得命令行直接调用时，能够访问到我们自定义的tianshou
# print(sys.path)
# sys.path.append('..')

from utils import get_kl

from tianshou.policy import SACPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu') # change for cpu
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    return parser.parse_args()


def test_sac(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    if args.training_num > 1:
        train_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.training_num)])
    else:
        train_envs = gym.make(args.task)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a, args.action_shape, max_action=args.max_action,
        device=args.device, unbounded=True, conditioned_sigma=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(args.state_shape, args.action_shape,
                 hidden_sizes=args.hidden_sizes,
                 concat=True, device=args.device)
    net_c2 = Net(args.state_shape, args.action_shape,
                 hidden_sizes=args.hidden_sizes,
                 concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # student model:
    net_a_student = Net(args.state_shape, hidden_sizes=[64, 64], device=args.device) # 256 -> 32
    actor_student = ActorProb(
        net_a_student, args.action_shape, max_action=args.max_action,
        device=args.device, unbounded=True, conditioned_sigma=True
    ).to(args.device)
    actor_optim_student = torch.optim.Adam(actor_student.parameters(), lr=args.actor_lr)

    net_c1_student = Net(args.state_shape, args.action_shape,
                 hidden_sizes=args.hidden_sizes,
                 concat=True, device=args.device)
    net_c2_student = Net(args.state_shape, args.action_shape,
                 hidden_sizes=args.hidden_sizes,
                 concat=True, device=args.device)
    critic1_student = Critic(net_c1_student, device=args.device).to(args.device)
    critic1_optim_student = torch.optim.Adam(critic1_student.parameters(), lr=args.critic_lr)
    critic2_student = Critic(net_c2_student, device=args.device).to(args.device)
    critic2_optim_student = torch.optim.Adam(critic2_student.parameters(), lr=args.critic_lr)



    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=args.tau, gamma=args.gamma, alpha=args.alpha,
        estimation_step=args.n_step, action_space=env.action_space)
    # student:
    policy_student = SACPolicy(
        actor_student, actor_optim_student, critic1_student, critic1_optim_student, critic2_student, critic2_optim_student,
        tau=args.tau, gamma=args.gamma, alpha=args.alpha,
        estimation_step=args.n_step, action_space=env.action_space)


    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)

    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    train_collector = Collector(policy_student, train_envs, buffer, exploration_noise=True)  # for student

    test_collector = Collector(policy_student, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_sac_pd_64'
    log_path = os.path.join(args.logdir, args.task, 'sac', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)
    print(log_path, 'have logger!!!')

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def update_student(best_teacher=None):
        sample_size = 1
        batch, indice = train_collector.buffer.sample(sample_size)
        # only need to update the student policy

        teacher = policy.forward(batch)
        student = policy_student.forward(batch)
        teacher_mus, teacher_sigmas = teacher.logits[0], teacher.logits[1]
        student_mus, student_sigmas = student.logits[0], student.logits[1]

        # stds = torch.tensor([1e-6] * len(teacher.logits[0]), device=args.device, dtype=torch.float)
        # stds = torch.stack([stds for _ in range(len(teacher.logits))])  # 自己伪造的 stds
        loss = sum([get_kl([teacher_mu, teacher_sigma], [student_mu, student_sigma])
                    for teacher_mu in teacher_mus
                    for teacher_sigma in teacher_sigmas
                    for student_mu in student_mus
                    for student_sigma in student_sigmas])

        # loss = get_kl([teacher.logits, stds], [student.logits, stds])
        policy_student.actor_optim.zero_grad()
        loss.backward()
        policy_student.actor_optim.step()

        # policy_student.actor.load_state_dict(policy.actor.state_dict())

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, args.test_num,
            args.batch_size, save_fn=save_fn, logger=logger,
            update_student_fn=update_student,
            update_per_step=args.update_per_step, test_in_train=False)
        pprint.pprint(result)

    # Let's watch its performance!
    policy_student.eval() # watch the student policy
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_sac()
