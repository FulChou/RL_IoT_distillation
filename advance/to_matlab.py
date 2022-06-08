# 作者：vincent
# code time：2022/3/31 19:52
import matlab.engine
import time

import cv2
import numpy as np
from scipy import io, integrate, linalg, signal

# # scipy.linalg.eigh
# cap = cv2.VideoCapture('./Jumps.avi')
# num_frames = int(cap.get(7)) - 2 # why - 2 ?
# H, W = int(cap.get(3)), int(cap.get(4))
# print(num_frames, H, W)
# X = np.zeros((num_frames, W, H, 3), np.uint8)
from tianshou.data import Batch
eng = matlab.engine.start_matlab()
eng.cd('/home/zhoufu/drl_iot/RL_IoT_distillation/advance', nargout=0)

def call_matlab_state_by(data:Batch=None, log_path='', lamb=1e3, threshold=0.9, maxiter=200):
    t = time.time()
    if hasattr(data, 'obs'):
        input = data.obs[:, -1]
    else:
        raise Exception('please input a Batch obj')
    list_input = input.tolist()
    print('input list time:', time.time() - t)
    input = matlab.double(list_input)
    print('convert time:', time.time() - t)
    t = time.time()
    idxs = eng.key_state_by(input, log_path, lamb, threshold, maxiter)
    print('matlab time:', time.time() - t)
    idxs = np.asarray(idxs).tolist()[0]
    return idxs


# def call_matlab_state(data:Batch=None, log_path=''):
#     if hasattr(data, 'obs'):
#         input = data.obs[:, -1]
#     else:
#         raise Exception('please input a Batch obj')
#     input = matlab.double(input.tolist())
#     eng = matlab.engine.start_matlab()
#     idxs = eng.key_state(input, log_path)
#     idxs = np.asarray(idxs).tolist()[0]
#     return idxs


# def call_matlab(data:Batch=None, log_path=''):
#     if hasattr(data, 'obs'):
#         input = data.obs[:, -1]
#     else:
#         raise Exception('please input a Batch obj')
#     input = matlab.double(input.tolist())
#     eng = matlab.engine.start_matlab()
#     idxs = eng.test(input, log_path)
#     idxs = np.asarray(idxs).tolist()[0]
#     return idxs

if __name__ == '__main__':
    # call_matlab()
    pass