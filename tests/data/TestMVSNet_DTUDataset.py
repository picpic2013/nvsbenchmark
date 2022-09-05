import torch
from nvsbenchmark import MVSNet_DTUDataset

DataLoader = MVSNet_DTUDataset('F:\database\dtu_example\dtu', 1.0, ['2_r5000', '3_r5000', '5_r5000'], [0, 4, 6, 10, 11])

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

print('rectified_imgs: ', ret['rectified_imgs'][6, 5])
print('Intrinsics'    , ret['Intrinsics'][6]) # V x 3 x 3
print('Extrinsics'    , ret['Extrinsics'][6]) # V x 4 x 4
print('near_depths'   , ret['near_depths'][6]) # V
print('intervals'     , ret['intervals'][6]) # V
print('depth_maps'    , ret['depth_maps'][6]) # V x H x W