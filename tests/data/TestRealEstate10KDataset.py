import torch
import numpy as np
import cv2
from nvsbenchmark import RealEstate10KDataset


DataLoader = RealEstate10KDataset(data_path = 'F:\database\RealEstate10K', mode = 'train', res = '720p')

print('__len__(): ', DataLoader.__len__())

ret = DataLoader.__getitem__(100)

cv2.imshow('img', np.asarray(ret['imgs'][0]).astype(np.uint8))
cv2.waitKey(0)
cv2.imshow('img', np.asarray(ret['imgs'][-1]).astype(np.uint8))
cv2.waitKey(0)
