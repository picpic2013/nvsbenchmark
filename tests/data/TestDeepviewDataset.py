import torch
from nvsbenchmark import DeepviewDataset

DataLoader = DeepviewDataset('F:/database/deepview_spaces', '2k')

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(0)

print(ret['imgs'][0][1].size())