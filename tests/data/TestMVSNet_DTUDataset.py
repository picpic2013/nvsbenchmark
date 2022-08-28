import torch
from nvsbenchmark import MVSNet_DTUDataset

DataLoader = MVSNet_DTUDataset('F:\database\dtu_example\dtu')

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

'''
print('rectified_imgs: ', ret['rectified_imgs'].size())
print('Intrinsics'    , ret['Intrinsics']) # V x 3 x 3
print('Extrinsics'    , ret['Extrinsics']) # V x 4 x 4
print('near_depths'   , ret['near_depths']) # V
print('intervals'     , ret['intervals']) # V
print('depth_maps'    , ret['depth_maps']) # V x H x W
'''