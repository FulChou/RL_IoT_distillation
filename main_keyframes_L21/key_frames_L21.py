# 作者：vincent
# code time：2022/3/31 14:49
import time

import cv2
import numpy as np
from scipy import io, integrate, linalg, signal
# scipy.linalg.eigh

cap = cv2.VideoCapture('./Jumps.mp4')
num_frames = int(cap.get(7)) - 2 # why - 2 ?
H, W = int(cap.get(3)), int(cap.get(4))
print(num_frames, H, W)
X = np.zeros((num_frames, W, H, 3), np.uint8)
# y = np.array([[1,2],[3,4],[5,6]])
# y_ = y.flatten('F') mean x(:)
# print(y, y_)
scale = 1 / 2
h, w = int(H * scale), int(W * scale)
Y = np.zeros((num_frames, h * w))

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

if cap.isOpened():
    i = 0
    while True and i < num_frames:
        ret, img = cap.read()
        # cv2.imwrite('./frame.jpg', img)
        # print(type(img))
        # np.swapaxes()
        # cv2.waitKey(0)
        # print(img.shap)
        if not ret:
            break
        X[i] = img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # to_double = cv2.normalize(gray_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        Y[i] = cv2.resize(im2double(gray_img), (h, w)).flatten('F')
        i += 1
else:
    print('video can not open')

print('read done')
start = time.perf_counter()
lamb = np.array([1e3, 1e4])
info = {'max_iter': 5000, 'regcase': 'L21','conver_bound': 1e-4, 'evaltrigger': 0}
D = Y
# eigv = [V,D]=eig(a)  D,V = linalg.eig(a)
# eigv = eig(D' * D);
trans = D.conj().T
# print('11')
print('trans during time: ', time.perf_counter() - start)
t = np.matmul(trans, D)
print('matmul during time: ', time.perf_counter() - start)
print(t.shape)

v, d = linalg.eigh(t)  # matrix multiply
print(v, d, type(v))
print('during time: ', time.perf_counter() - start)




