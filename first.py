# 作者：vincent
# code time：2021/7/26 下午8:05
import gym
import tianshou as ts
import torch, numpy as np
from tianshou.data import Batch
from torch import nn
from utils import get_kl

env_name = 'Breakout-v0'
# env = gym.make('CartPole-v0')
env = gym.make(env_name)
# train_envs = gym.make('CartPole-v0')
# test_envs = gym.make('CartPole-v0')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # using Gpu print(device) device = 'cpu'

train_envs = ts.env.SubprocVectorEnv([lambda: gym.make(env_name) for _ in range(10)])
test_envs = ts.env.SubprocVectorEnv([lambda: gym.make(env_name) for _ in range(100)])


class TeacherNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs,device=device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class StudentNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs,device=device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


state_shape = env.observation_space.shape or env.observation_space.n  # (4,)
action_shape = env.action_space.shape or env.action_space.n  # 2
net = TeacherNet(state_shape, action_shape).to(device)
net_student = StudentNet(state_shape, action_shape).to(device)

optim = torch.optim.Adam(net.parameters(), lr=1e-3)
optim_student = torch.optim.Adam(net_student.parameters(), lr=1e-3)

teacher_policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

student_policy = ts.policy.DQNPolicy(net_student, optim_student, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_collector = ts.data.Collector(student_policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(teacher_policy, test_envs, exploration_noise=True)


def train_fn(epoch, env_step):
    teacher_policy.set_eps(0.1)
    student_policy.set_eps(0.1)
    # print(epoch,env_step)
    # if epoch % 2 == 0:
    # teacher_params = teacher_policy.model.state_dict()  # 直接复制的方法改成蒸馏：
    # student_policy.model.load_state_dict(teacher_params)
    # print(student_policy.model.state_dict()['model.6.bias'])


def update_student():
    # teacher_params = teacher_policy.model.state_dict()  # 直接复制的方法改成蒸馏：
    # student_policy.model.load_state_dict(teacher_params)
    # 学习蒸馏算法去！
    sample_size = 10
    if len(train_collector.buffer) > sample_size:
        batch, indice = train_collector.buffer.sample(sample_size)


        # input = Batch(obs=Batch(obs=obs,mask=mask))
        teacher = teacher_policy.forward(batch)
        student = student_policy.forward(batch)

        stds = torch.tensor([1e-6] * len(teacher.logits[0]), device=device, dtype=torch.float)
        stds = torch.stack([stds for _ in range(len(teacher.logits))])
        loss = get_kl([teacher.logits, stds], [student.logits, stds])
        student_policy.optim.zero_grad()
        loss.backward()
        student_policy.optim.step()
        # print(batch)

    # print('update')
    # print('new', student_policy.model.state_dict()['model.6.bias'] == teacher_policy.model.state_dict()['model.6.bias'])


result = ts.trainer.offpolicy_trainer(
    teacher_policy, train_collector, test_collector,
    max_epoch=100, step_per_epoch=10000, step_per_collect=100,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=train_fn,
    update_student_fn=update_student,
    test_fn=lambda epoch, env_step: teacher_policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
print(f'Finished training! Use {result["duration"]}')
print(result)


from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger
writer = SummaryWriter('log/dqn')
logger = BasicLogger(writer)
