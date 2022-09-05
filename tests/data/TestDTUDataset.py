import torch
from nvsbenchmark import DTUDataset

DataLoader = DTUDataset('F:\database\DTU_SampleSet\MVS Data', ['2_r5000', '3_r5000', 'max'], [0, 4, 6, 10, 11])

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

print('rectified_imgs: ', ret['rectified_imgs'][6, 7])
print('pointcloud_points: ',ret['pointcloud_points'])
print('pointcloud_colors: ', ret['pointcloud_colors'])
print('pointcloud_normals: ', ret['pointcloud_normals'])
print('ObsMask: ', ret['ObsMask'])
print('Res: ', ret['Res'])
print('Margin: ', ret['Margin'])
print('BB: ', ret['BB'])
print('Plane: ', ret['Plane'])
print('Cam_Mats: ', ret['Cam_Mats'][6])