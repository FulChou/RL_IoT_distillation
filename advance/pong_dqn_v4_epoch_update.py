import os
import datetime
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys



sys.path.append(os.path.join(os.path.dirname(__file__),'..')) # 使得命令行直接调用时，能够访问到我们自定义的tianshou
# print(sys.path)
from tianshou.trainer.offpolicy_v2 import offpolicy_trainer_v2
from tianshou.trainer.offpolicy_v3_epoch_update_student import offpolicy_v3_epoch_update_student
from utils import get_kl, get_mykl
from tianshou.policy import DQNPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from atari_network import DQN, student_DQN_net1
from atari_wrapper import wrap_deepmind
from to_matlab import call_matlab


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
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
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


def test_dqn(args=get_args()):
    env = make_atari_env(args)
    print('reward best', env.spec.reward_threshold)

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

    # define model
    net = DQN(*args.state_shape,
              args.action_shape, args.device).to(args.device)

    student_net = student_DQN_net1(*args.state_shape,
              args.action_shape, args.device,).to(args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    student_optim = torch.optim.Adam(student_net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(net, optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    policy_student = DQNPolicy(student_net, student_optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)  # test  target_update_freq = 0

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size, buffer_num=len(train_envs), ignore_obs_next=True,
        save_only_last_obs=True, stack_num=args.frames_stack)
    # collector
    train_collector = Collector(policy_student, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    test_student_collector = Collector(policy_student, test_envs, exploration_noise=True)

    # log
    t0 = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}'+'debug4-16-32-32-256'
    log_path = os.path.join(args.logdir, args.task, 'dqn', log_file)
    print('log_path', log_path)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        print('sava model at: ', os.path.join(log_path, 'policy.pth'))
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def save_student_policy_fn(policy):
        print('sava model at: ', os.path.join(log_path, 'policy_student.pth'))
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_student.pth'))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif 'Pong' in args.task:
            return mean_rewards >= 20
        elif 'MsPacman' in args.task:
            return mean_rewards >= 2000
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        policy_student.set_eps(eps)
        logger.write('train/eps', env_step, eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)
        policy_student.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size, buffer_num=len(test_envs),
                ignore_obs_next=True, save_only_last_obs=True,
                stack_num=args.frames_stack)
            collector = Collector(policy, test_envs, buffer,
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

    def update_student(best_teacher_policy=None, sample_size=1, logger=None, step=0, is_update_student=True):
        loss_bound = 1
        pre_loss = 0
        while loss_bound >= 0.0001:
            # policy_student.load_state_dict(policy.state_dict())
            if len(train_collector.buffer) >= 1e3:
                batch_all, indice_all = train_collector.buffer.sample(0)
                temp_buffer = VectorReplayBuffer(
                    args.buffer_size, buffer_num=len(train_envs), ignore_obs_next=True,
                    save_only_last_obs=True, stack_num=args.frames_stack)
                # TODO: 减少matlab关于每个状态的代码，
                idxs = call_matlab(batch_all)  # 只需要取四个中的一个就行，反正都是一样的！ 重要知道到了
                # print(idxs)
                idxs = list(map(int, idxs))
                idxs = [i-1 for i in idxs]
                print('keys idxs', idxs)
                key_batch = batch_all[idxs]
                key_indice = indice_all[idxs]

                buffer_id = [int(i // 1e4) for i in key_indice]
                temp_buffer.add(key_batch, buffer_id)  # this add (2,2)
                # old_buffer = train_collector.buffer
                train_collector.buffer = temp_buffer
            # print()
            batch_all, indice_all = train_collector.buffer.sample(0)  # 复制了多个多余的。。。
            # sample 32 from [1e4, 2e4]
            batch, indice = train_collector.buffer.sample(args.batch_size)
            # print(batch[[1,2]])
            if best_teacher_policy:
                teacher = best_teacher_policy.forward(batch)
            else:
                teacher = policy.forward(batch)
            student = policy_student.forward(batch)
            stds = torch.tensor([1e-6] * len(teacher.logits[0]), device=args.device, dtype=torch.float)
            stds = torch.stack([stds for _ in range(len(teacher.logits))])
            loss = get_mykl([teacher.logits, stds], [student.logits, stds])
            loss_bound, pre_loss = loss - pre_loss, loss
            logger.log_update_data({'kl_loss:': loss}, step)
            policy_student.optim.zero_grad()
            loss.backward()
            policy_student.optim.step()


    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_v3_epoch_update_student(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, args.test_num,
        args.batch_size, train_fn=train_fn,
        update_student_fn=update_student, test_student_collector=test_student_collector, policy_student=policy_student,
        save_student_policy_fn=save_student_policy_fn,
        test_fn=test_fn, stop_fn=stop_fn, save_fn=save_fn, logger=logger,
        update_per_step=args.update_per_step, test_in_train=False)

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())
