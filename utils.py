# original : https://github.com/Mee321/policy-distillation

import torch
import math
from torch.nn.functional import softmax
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import time

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def get_kl(teacher_dist_info, student_dist_info):
    pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])  # means， std
    pi_new = Normal(student_dist_info[0], scale=student_dist_info[1])
    kl = torch.mean(kl_divergence(pi, pi_new))
    return kl

def get_mykl(teacher_dist_info, student_dist_info):
    tau = 0.01
    eps = 0.00001
    p = softmax(teacher_dist_info[0]/ tau, dim=1) + eps
    q = softmax(student_dist_info[0], dim=1) + eps
    return torch.sum(p * torch.log(p / q))
    # Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])  # means， std
    # pi_new = Normal(student_dist_info[0], scale=student_dist_info[1])
    # kl = torch.mean(kl_divergence(pi, pi_new))
    # return kl

def get_wasserstein(teacher_dist_info, student_dist_info):
    means_t, stds_t = teacher_dist_info
    means_s, stds_s = student_dist_info
    return torch.sum((means_s - means_t) ** 2) + torch.sum((torch.sqrt(stds_s) - torch.sqrt(stds_t)) ** 2)


class CountTime:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_ty, exc_val, exc_tb):
        end = time.time()
        self.res = '{}: {}'.format(self.label, end - self.start)
