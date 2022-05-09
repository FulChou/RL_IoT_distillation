# 作者：vincent
# code time：2021/9/28 8:34 下午
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.random import randn

print(torch.cuda.is_available())

def draw_loss_distribution(name, pre_loss, key_loss):
    data = []
    data.append(pre_loss)
    data.append(key_loss)
    # data = np.array(data, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, stacked=True, label=('pre_loss', 'key_loss'))
    # my_x_ticks = np.arange(-1, 3, 0.1)
    # plt.xticks(my_x_ticks)
    plt.legend()
    plt.savefig(name + '.jpg')


pre_key_loss = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10]
# print(pre_key_loss)
key_loss = [5, 6, 6, 6, 9, 6, 10, 7, 6, 4]
# print(key_loss)

draw_loss_distribution('3', pre_key_loss, key_loss)