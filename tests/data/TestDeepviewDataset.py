import torch
import json
import numpy as np
from nvsbenchmark import DeepviewDataset

DataLoader = DeepviewDataset('F:/database/deepview_spaces', '2k', [1,4,5,8], [2, 6])

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

print('extrinsics', ret['extrinsics'][8, 2]) # Rig x Cam x 4 x 4
print('intrinsics', ret['intrinsics'][8, 2]) # Rig x Cam x 3 x 3
print('imgs', ret['imgs'][8, 2]) # Rig x Cam x 3 x H x W

'''

tmp = []

with open('F:\database\deepview_spaces\\2k\scene_030\models.json', 'r') as f:
    data = json.load(f)

Rig = len(data)
Cam = len(data[0])

for rig in range(Rig):
    for cam in range(Cam):
        #print('{}   {}'.format(data[rig][cam]['height'], data[rig][cam]['width']))
        tmp.append(data[rig][cam]['height'])

with open('F:\database\deepview_spaces\\2k\scene_031\models.json', 'r') as f:
    data = json.load(f)

Rig = len(data)
Cam = len(data[0])

for rig in range(Rig):
    for cam in range(Cam):
        #print('{}   {}'.format(data[rig][cam]['height'], data[rig][cam]['width']))
        tmp.append(data[rig][cam]['height'])

tmp = np.asarray(tmp).min()

print(tmp)
'''