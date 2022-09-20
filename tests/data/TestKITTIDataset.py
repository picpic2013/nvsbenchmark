import torch
import cv2
import numpy as np
from nvsbenchmark import KITTIDataset

DataLoader = KITTIDataset(data_path = 'F:\database\KITTI', mode = 'training', is_mvs = True)

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

cv2.imshow('left first', np.asarray(ret['img_2s'][10]).astype(np.uint8))
cv2.imshow('disparoty', np.asarray(ret['flow_noc']).astype(np.uint8))
cv2.waitKey(0)

print('R_left_first: ', ret['R_mat_2s'][0])
print('P_left_first: ', ret['P_mat_2s'][0])

print('R_left_second: ', ret['R_mat_2s'][1])
print('P_left_second: ', ret['P_mat_2s'][1])

print('R_right_first: ', ret['R_mat_3s'][0])
print('P_right_first: ', ret['P_mat_3s'][0])

print('R_right_second: ', ret['R_mat_3s'][1])
print('P_right_second: ', ret['P_mat_3s'][1])