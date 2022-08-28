import torch
import json
import numpy as np
from nvsbenchmark import DeepviewDataset

DataLoader = DeepviewDataset('F:/database/deepview_spaces', '2k')

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

print('ret[\'imgs\'].size(): ', ret['imgs'].size())
print('ret[\'extrinsics\']: ', ret['extrinsics'])

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