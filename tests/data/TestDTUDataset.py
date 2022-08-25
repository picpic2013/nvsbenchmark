import torch
from nvsbenchmark import DTUDataset

DataLoader = DTUDataset('F:\database\DTU_SampleSet\MVS Data')

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

print('rectified_imgs: ', ret['rectified_imgs'].size())
print('pointcloud_points: ',ret['pointcloud_points'].size())
print('pointcloud_colors: ', ret['pointcloud_colors'].size())
print('pointcloud_normals: ', ret['pointcloud_normals'].size())
print('ObsMask: ', ret['ObsMask'].size())
print('Res: ', ret['Res'])
print('Margin: ', ret['Margin'])
print('BB: ', ret['BB'].size())
print('Plane: ', ret['Plane'].size())
print('Cam_Mats: ', ret['Cam_Mats'].size())