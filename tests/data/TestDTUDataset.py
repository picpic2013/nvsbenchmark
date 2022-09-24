import torch
import cv2
import numpy as np
from nvsbenchmark import DTUDataset
from nvsbenchmark.data.DTUDataset import DTUDataset_scenes

DataLoader = DTUDataset('F:\database\DTU_SampleSet\MVS Data', ['2_r5000', '3_r5000', 'max'], [1, 4, 6, 10, 11])
ret = DataLoader.__getitem__(1)

print('__len__(): ', DataLoader.__len__())

cv2.imshow('rectified_imgs ', np.asarray(ret['rectified_imgs'][4, 2]).astype(np.uint8))
cv2.imshow('cleaned_imgs ', np.asarray(ret['cleaned_imgs'][4, 2]).astype(np.uint8))
print('pointcloud_points: ',ret['pointcloud_points'])
print('pointcloud_colors: ', ret['pointcloud_colors'])
print('pointcloud_normals: ', ret['pointcloud_normals'])
print('surface_points', ret['surface_points'])
print('surface_triangles', ret['surface_triangles'])
print('ObsMask: ', ret['ObsMask'])
print('Res: ', ret['Res'])
print('Margin: ', ret['Margin'])
print('BB: ', ret['BB'])
print('Plane: ', ret['Plane'])
print('Extrinsics: ', ret['Extrinsics'])
print('Intrinsics: ', ret['Intrinsics'])

cv2.waitKey()