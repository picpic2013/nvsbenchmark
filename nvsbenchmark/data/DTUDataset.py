import os
import json
import torch
import scipy
import open3d as o3d
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DTUDataset(Dataset):
    def __init__(self, data_path = None) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        '''
        super(DTUDataset, self).__init__()
        self.data_path = os.path.normpath(data_path)
        print('This dataset loader will load data from \'', self.data_path, '\'')
        
        self.scenes_names = os.listdir(os.path.join(self.data_path, 'Rectified'))
        print('This subset contains: ', self.scenes_names)

    def __len__(self):
        return len(self.scenes_names)

    def __getitem__(self, index):

        V, L, H, W = 49, 8, 1200, 1600

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
                'Intrinsic'     : None, # 3 x 3
                'Distortion'    : None, # 5
                'Extrinsics'    : None, # V x 4 x 4
                'R': None,
                't': None, 
                'dep': None
            }

        rectified_imgs = torch.empty(V, L, 3, H, W)
        Cam_Mats = torch.empty(V, 3, 4)
        Extrinsics = torch.zeros(V, 4, 4)
        
        scan_name = self.scenes_names[index]
        scan_num = 0
        for i in range(len(scan_name)):
            if ('0' <= scan_name[i]) & (scan_name[i] <= '9'):
                scan_num = scan_num * 10 + int(scan_name[i]) - int('0')

        for i in range(49):
            Cam_Mat_Path = os.path.join(self.data_path, 'Calibration\cal18', 'pos_%03d.txt' % (i+1))
            with open(Cam_Mat_Path, 'r') as f:
                lines = f.readlines()
                for j, line in enumerate(lines):
                    data = line.split(sep = ' ')
                    for k in range(4):
                        Cam_Mats[i, j, k] = float(data[k])
            
            camera_Path = os.path.join(self.data_pth, 'Calibration\cal18', 'Calib_Results_stereo.mat')
            data = scipy.io.loadmat(camera_Path)
            Extrinsics[i, 0:3, 0:3]


            for j,id in enumerate(['0_r5000', '1_r5000', '2_r5000', '3_r5000', '4_r5000', '5_r5000', '6_r5000', 'max']):
                rectified_img_Path = os.path.join(self.data_path, 'Rectified', scan_name, 'rect_%03d_%s.png' % (i+1, id))
                rectified_imgs[i, j, :, :, :] = transforms.ToTensor()(Image.open(rectified_img_Path))

        pointcloud_Path = os.path.join(self.data_path, 'Points', 'stl', 'stl%03d_total.ply' % scan_num)
        pointcloud = o3d.io.read_point_cloud(pointcloud_Path, format = 'ply')
        pointcloud_points = torch.tensor(np.asarray(pointcloud.points))
        pointcloud_colors = torch.tensor(np.asarray(pointcloud.colors))
        pointcloud_normals = torch.tensor(np.asarray(pointcloud.normals))

        ObsMask_Path = os.path.join(self.data_path, 'ObsMask', 'ObsMask%d_10.mat' % scan_num)
        data = scipy.io.loadmat(ObsMask_Path)
        BB = torch.tensor(data['BB'])
        ObsMask = torch.tensor(data['ObsMask'])
        Res = data['Res'][0][0]
        Margin = data['Margin'][0][0]

        Plane_Path = os.path.join(self.data_path, 'ObsMask', 'Plane%d.mat' % scan_num)
        data = scipy.io.loadmat(Plane_Path)
        Plane = torch.tensor(data['P']).squeeze(1)

        return {
            'rectified_imgs': rectified_imgs, # V x L x H x W x 3       V represents view, L represents light condition
            'pointcloud_points'    : pointcloud_points, # N x 3
            'pointcloud_colors'    : pointcloud_colors, # N x 3
            'pointcloud_normals'   : pointcloud_normals, # N x 3
            'ObsMask'       : ObsMask, # MH x MW x MD
            'Res'           : Res, # float
            'Margin'        : Margin, # int
            'BB'            : BB, # 2 x 3
            'Plane'         : Plane, # 4
            'Cam_Mats'      : Cam_Mats, # V x 3 x 4
            'Intrinsic'     : None, # 3 x 3
            'Distortion'    : None, # 5
            'Extrinsics'    : None, # V x 4 x 4
            'R': None,
            't': None, 
            'dep': None
        }