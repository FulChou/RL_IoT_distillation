import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os

def get_mean_last10(tfevent_file, teacher=True):
    ea = event_accumulator.EventAccumulator(tfevent_file)
    ea.Reload()
    if teacher:
        test_rew = ea.scalars.Items('test/rew')
    else:
        test_rew = ea.scalars.Items('test_student/rew')
    test_rew_value = [i.value for i in test_rew]
    return np.mean(test_rew_value[-10:]), max(test_rew_value)

dir_path = '/home/zhoufu/drl_iot/RL_IoT_distillation/advance/log'
logs = {}

for path, dir_list, file_list in os.walk(dir_path):
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        if 'events' in file_path and 'net1' in file_path:
            print(file_path)
            env_name = file_path.split('/')[-4]
            logs[env_name] = file_path

print('Teacher: ')
for name, path in logs.items():
    print('env_name:{}, mean reward:{}, max:{}'.format(name, *(get_mean_last10(path, teacher=True))))

print('Student: ')
for name, path in logs.items():
    print('env_name:{}, mean reward:{}, max:{}'.format(name, *(get_mean_last10(path, teacher=False))))

