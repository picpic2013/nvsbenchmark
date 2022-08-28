import os
import json
import torch
import scipy
import cv2
import re
import open3d as o3d
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def read_cam_param(path, interval_scalar):
    with open(path, 'r') as f:
        data = f.read().split()

    Extrinsic = torch.empty(4,4)
    Intrinsic = torch.empty(3,3)

    for i in range(4):
        for j in range(4):
            Extrinsic[i, j] = float(data[i*4 + j + 1])
    
    for i in range(3):
        for j in range(3):
            Intrinsic[i, j] = float(data[i*3 + j + 1 + 16 + 1])

    near_depth = float(data[1 + 16 + 1 + 9])
    Intrinsic = Intrinsic * 4.0 # Unclear: whether or not multiply 4.0
    interval = float(data[1 + 16 + 1 + 9 + 1]) * interval_scalar

    return Extrinsic, Intrinsic, near_depth, interval

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

class MVSNet_DTUDataset(Dataset):
    def __init__(self, data_path = None, interval_scalar = 1.0) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        '''
        super(MVSNet_DTUDataset, self).__init__()
        self.data_path = os.path.normpath(data_path)
        print('This dataset loader will load data from \'', self.data_path, '\'')
        
        self.scenes_names = os.listdir(os.path.join(self.data_path, 'Rectified'))

        print('This subset contains: ', self.scenes_names)

        self.interval_scalar = interval_scalar

    def __len__(self):
        return len(self.scenes_names)

    def __getitem__(self, index):

        

        if index > self.__len__():
            print('Index out of range.')
            return {
                'rectified_imgs': None, # V x L x H x W x 3       V represents view, L represents light condition
                'pointcloud_points'    : None, # N x 3
                'pointcloud_colors'    : None, # N x 3
                'pointcloud_normals'   : None, # N x 3
                'ObsMask'       : None, # MH x MW x MD
                'Res'           : None, # float
                'Margin'        : None, # int
                'BB'            : None, # 2 x 3
                'Plane'         : None, # 4
                'Cam_Mats'      : None, # V x 3 x 4
                'Distortion'    : None, # 5
                'Intrinsics'    : None, # V x 3 x 3
                'Extrinsics'    : None, # V x 4 x 4
                'near_depths'   : None, # V
                'intervals'     : None, # V
                'depth_maps'    : None, # V x H x W
                'R': None,
                't': None, 
                'dep': None
            }

        scan_name = self.scenes_names[index]
        scan_num = 0
        for i in range(len(scan_name)):
            if ('0' <= scan_name[i]) & (scan_name[i] <= '9'):
                scan_num = scan_num * 10 + int(scan_name[i]) - int('0')

        L, H, W = 7, 512, 640
        V = len(os.listdir(os.path.join(self.data_path, 'Rectified', scan_name))) // L

        rectified_imgs = torch.empty(V, L, 3, H, W)
        Extrinsics = torch.zeros(V, 4, 4)
        Intrinsics = torch.zeros(V, 3, 3)
        near_depths = torch.empty(V)
        intervals = torch.empty(V)
        depth_maps = torch.empty(V, H, W)

        for i in range(V):
            Cam_Mat_Path = os.path.join(self.data_path, 'Cameras\\train', '%08d_cam.txt' % i)
            Extrinsic, Intrinsic, near_depth, interval = read_cam_param(Cam_Mat_Path, self.interval_scalar)

            Intrinsics[i, :, :] = Intrinsic
            Extrinsics[i, :, :] = Extrinsic
            near_depths[i] = near_depth
            intervals[i] = interval

            for j,id in enumerate(['0_r5000', '1_r5000', '2_r5000', '3_r5000', '4_r5000', '5_r5000', '6_r5000']):
                rectified_img_Path = os.path.join(self.data_path, 'Rectified', scan_name, 'rect_%03d_%s.png' % (i+1, id))
                rectified_imgs[i, j, :, :, :] = transforms.ToTensor()(Image.open(rectified_img_Path))
            
            depth_map_Path = os.path.join(self.data_path, 'Depths', ('scan%d' % scan_num), ('depth_map_%04d.pfm' % i))
            depth_map = np.array(read_pfm(depth_map_Path)[0], dtype=np.float32)
            depth_map = cv2.resize(depth_map, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            depth_map = depth_map[44:556, 80:720]
            depth_maps[i, :, :] = torch.tensor(depth_map)
            print('depth_map.size: ', depth_map.shape)

        return {
            'rectified_imgs': rectified_imgs, # V x L x H x W x 3       V represents view, L represents light condition
            'pointcloud_points'    : None, # N x 3
            'pointcloud_colors'    : None, # N x 3
            'pointcloud_normals'   : None, # N x 3
            'ObsMask'       : None, # MH x MW x MD
            'Res'           : None, # float
            'Margin'        : None, # int
            'BB'            : None, # 2 x 3
            'Plane'         : None, # 4
            'Cam_Mats'      : None, # V x 3 x 4
            'Distortion'    : None, # 5
            'Intrinsics'    : Intrinsics, # V x 3 x 3
            'Extrinsics'    : Extrinsics, # V x 4 x 4
            'near_depths'   : near_depths, # V
            'intervals'     : intervals, # V
            'depth_maps'    : depth_maps, # V x H x W
            'R': None,
            't': None, 
            'dep': None
        }