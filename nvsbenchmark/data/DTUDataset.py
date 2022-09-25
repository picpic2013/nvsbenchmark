import os
import json
import torch
import scipy
import cv2
import open3d as o3d
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from typing import List

class DTUDataset(Dataset):
    def __init__(self, data_path = None, light_subset: List[str] = None, view_subset: List[int] = None,
                ret_rectified = True,
                ret_cleaned = True,
                ret_pcd = 'stl',
                ret_surface = 'camp',
                ret_obsmask = True,
                ret_res = True,
                ret_margin = True,
                ret_bb = True,
                ret_plane = True,
                ret_intrinsics = True,
                ret_extrinsics = True) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        @param light_subset: 
        '''
        super(DTUDataset, self).__init__()
        self.data_path = os.path.normpath(data_path)
        
        self.scenes_names = os.listdir(os.path.join(self.data_path, 'Rectified'))

        self.light_subset = light_subset
        self.view_subset = view_subset

        self.ret_rectified = ret_rectified
        self.ret_cleaned = ret_cleaned
        self.ret_pcd = ret_pcd
        self.ret_surface = ret_surface
        self.ret_obsmask = ret_obsmask
        self.ret_res = ret_res
        self.ret_margin = ret_margin
        self.ret_bb = ret_bb
        self.ret_plane = ret_plane
        self.ret_intrinsics = ret_intrinsics
        self.ret_extrinsics = ret_extrinsics

    def __len__(self):
        return len(self.scenes_names)

    def __getitem__(self, index):

        RET = dict.fromkeys(['rectified_imgs', # V x L x H x W x 3       V represents view, L represents light condition
                            'cleaned_imgs', # V x L x H x W x 3
                            'pointcloud_points', # N x 3
                            'pointcloud_colors', # N x 3
                            'pointcloud_normals', # N x 3
                            'surface_points', # N x 3
                            'surface_triangles', # F(faces) x 3
                            'ObsMask', # MH x MW x MD
                            'Res', # float
                            'Margin', # int
                            'BB', # 2 x 3
                            'Plane', # 4
                            'Intrinsics', # V x 3 x 3
                            'Extrinsics'] # V x 4 x 4
                            )

        if index > self.__len__():
            print('Index out of range.')
            return RET

        L, H, W = 8, 1200, 1600
        scene_name = self.scenes_names[index]
        scene_num = 0
        for i in range(len(scene_name)):
            if ('0' <= scene_name[i]) & (scene_name[i] <= '9'):
                scene_num = scene_num * 10 + int(scene_name[i]) - int('0')
        V = len(os.listdir(os.path.join(self.data_path, 'Rectified', 'scan{}'.format(scene_num)))) // 8
        if self.view_subset:
            actual_V = len(self.view_subset)
        else:
            actual_V = V
        if self.light_subset:
            actual_L = len(self.light_subset)
        else:
            actual_L = L


        if self.ret_rectified:
            rectified_imgs = torch.empty(actual_V, actual_L, H, W, 3)
        if self.ret_cleaned:
            cleaned_imgs = torch.empty(actual_V, actual_L, H, W, 3)
        if self.ret_extrinsics:
            Extrinsics = torch.eye(4, 4).unsqueeze(0).repeat(actual_V, 1, 1)
        if self.ret_intrinsics:
            Intrinsics = torch.eye(3, 3).unsqueeze(0).repeat(actual_V, 1, 1)

        for i in range(actual_V):
            view_id = self.view_subset[i]
            
            if self.ret_extrinsics:
                Cam_Mat_Path = os.path.join(self.data_path, 'Calibration', 'cal18', 'pos_%03d.txt' % view_id)
                with open(Cam_Mat_Path, 'r') as f:
                    lines = f.readlines()
                    for j, line in enumerate(lines):
                        data = line.split(sep = ' ')
                        for k in range(4):
                            Extrinsics[i, j, k] = float(data[k])

            if self.ret_rectified:
                for j in range(actual_L):
                    light_id = self.light_subset[j]

                    rectified_img_Path = os.path.join(self.data_path, 'Rectified', 'scan{}'.format(scene_num), 'rect_%03d_%s.png' % (view_id, light_id))
                    rectified_imgs[i, j, :, :, :] = torch.tensor(cv2.imread(rectified_img_Path))
            
            if self.ret_cleaned:
                for j in range(actual_L):
                    light_id = self.light_subset[j]

                    cleaned_img_Path = os.path.join(self.data_path, 'Cleaned', 'scan{}'.format(scene_num), 'clean_%03d_%s.png' % (view_id, light_id))
                    cleaned_imgs[i, j, :, :, :] = torch.tensor(cv2.imread(cleaned_img_Path))

        if self.ret_pcd:
            if self.ret_pcd == 'stl':
                pointcloud_Path = os.path.join(self.data_path, 'Points', 'stl', 'stl%03d_total.ply' % scene_num)
            else:
                pointcloud_Path = os.path.join(self.data_path, 'Points', self.ret_pcd, '%s%03d_l3.ply' % (self.ret_pcd, scene_num))
            pointcloud = o3d.io.read_point_cloud(pointcloud_Path, format = 'ply')
            pointcloud_points = torch.tensor(np.asarray(pointcloud.points))
            pointcloud_colors = torch.tensor(np.asarray(pointcloud.colors))
            pointcloud_normals = torch.tensor(np.asarray(pointcloud.normals))
        
        if self.ret_surface:
            surface_Path = os.path.join(self.data_path, 'Surfaces', self.ret_surface, '%s%03d_l3_surf_11_trim_8.ply' % (self.ret_surface, scene_num))
            surface = o3d.io.read_triangle_mesh(surface_Path)
            surface_points = torch.tensor(np.asarray(surface.vertices))
            surface_triangles = torch.tensor(np.asarray(surface.triangles))

        if self.ret_bb | self.ret_obsmask | self.ret_res | self.ret_margin:
            ObsMask_Path = os.path.join(self.data_path, 'ObsMask', 'ObsMask%d_10.mat' % scene_num)
            data = scipy.io.loadmat(ObsMask_Path)
            BB = torch.tensor(data['BB'])
            if self.ret_obsmask:
                ObsMask = torch.tensor(data['ObsMask'])
            Res = data['Res'][0][0]
            Margin = data['Margin'][0][0]

        if self.ret_plane:
            Plane_Path = os.path.join(self.data_path, 'ObsMask', 'Plane%d.mat' % scene_num)
            data = scipy.io.loadmat(Plane_Path)
            Plane = torch.tensor(data['P']).squeeze(1)

        if self.ret_rectified:
            RET['rectified_imgs'] = rectified_imgs
        if self.ret_cleaned:
            RET['cleaned_imgs'] = cleaned_imgs
        if self.ret_pcd:
            RET['pointcloud_points'] = pointcloud_points
            RET['pointcloud_colors'] = pointcloud_colors
            RET['pointcloud_normals'] = pointcloud_normals
        if self.ret_surface:
            RET['surface_points'] = surface_points
            RET['surface_triangles'] = surface_triangles
        if self.ret_obsmask:
            RET['ObsMask'] = ObsMask
        if self.ret_res:
            RET['Res'] = Res
        if self.ret_margin:
            RET['Margin'] = Margin
        if self.ret_bb:
            RET['BB'] = BB
        if self.ret_plane:
            RET['Plane'] = Plane
        if self.ret_intrinsics:
            RET['Intrinsics'] = Intrinsics
        if self.ret_extrinsics:
            RET['Extrinsics'] = Extrinsics

        return RET
    
# ==================================================================================================

class DTUDataset_per_Scene(Dataset):
    def __init__(self, data_path = None, scene_id = 0, light_subset: List[str] = None, view_subset: List[int] = None,
                ret_rectified = True,
                ret_cleaned = True,
                ret_pcd = 'stl',
                ret_surface = 'camp',
                ret_obsmask = True,
                ret_res = True,
                ret_margin = True,
                ret_bb = True,
                ret_plane = True,
                ret_intrinsics = True,
                ret_extrinsics = True) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        @param light_subset: 
        '''
        super(DTUDataset_per_Scene, self).__init__()
        self.data_path = os.path.normpath(data_path)

        self.scene_id = scene_id
        
        if view_subset == None:
            self.V = len(os.listdir(os.path.join(self.data_path, 'Rectified', 'scan{}'.format(self.scene_id)))) // 8
        else:
            self.V = len(view_subset)

        self.light_subset = light_subset
        self.view_subset = view_subset

        self.ret_rectified = ret_rectified
        self.ret_cleaned = ret_cleaned
        self.ret_pcd = ret_pcd
        self.ret_surface = ret_surface
        self.ret_obsmask = ret_obsmask
        self.ret_res = ret_res
        self.ret_margin = ret_margin
        self.ret_bb = ret_bb
        self.ret_plane = ret_plane
        self.ret_intrinsics = ret_intrinsics
        self.ret_extrinsics = ret_extrinsics

    
    def __len__(self):
        return self.V
    

    def __getitem__(self, index):

        RET = dict.fromkeys(['rectified_imgs', # L x H x W x 3       V represents view, L represents light condition
                            'cleaned_imgs', # L x H x W x 3
                            'pointcloud_points', # N x 3
                            'pointcloud_colors', # N x 3
                            'pointcloud_normals', # N x 3
                            'surface_points', # N x 3
                            'surface_triangles', # F(faces) x 3
                            'ObsMask', # MH x MW x MD
                            'Res', # float
                            'Margin', # int
                            'BB', # 2 x 3
                            'Plane', # 4
                            'Intrinsics', # 3 x 3
                            'Extrinsics'] # 4 x 4
                            )

        if index > self.__len__():
            print('Index out of range.')
            return RET

        L, H, W = 8, 1200, 1600
        scene_num = self.scene_id
        view_id = self.view_subset[index]
        if self.light_subset:
            actual_L = len(self.light_subset)
        else:
            actual_L = L

        rectified_imgs = torch.empty(actual_L, H, W, 3)
        cleaned_imgs = torch.empty(actual_L, H, W, 3)
        Extrinsics = torch.eye(4, 4)
        Intrinsics = torch.eye(3, 3)

        if self.ret_extrinsics:
            Cam_Mat_Path = os.path.join(self.data_path, 'Calibration', 'cal18', 'pos_%03d.txt' % view_id)
            with open(Cam_Mat_Path, 'r') as f:
                lines = f.readlines()
                for j, line in enumerate(lines):
                    data = line.split(sep = ' ')
                    for k in range(4):
                        Extrinsics[j, k] = float(data[k])

        if self.ret_rectified:
            for j in range(actual_L):
                light_id = self.light_subset[j]

                rectified_img_Path = os.path.join(self.data_path, 'Rectified', 'scan{}'.format(scene_num), 'rect_%03d_%s.png' % (view_id, light_id))
                rectified_imgs[j, :, :, :] = torch.tensor(cv2.imread(rectified_img_Path))
        
        if self.ret_cleaned:
            for j in range(actual_L):
                light_id = self.light_subset[j]

                cleaned_img_Path = os.path.join(self.data_path, 'Cleaned', 'scan{}'.format(scene_num), 'clean_%03d_%s.png' % (view_id, light_id))
                cleaned_imgs[j, :, :, :] = torch.tensor(cv2.imread(cleaned_img_Path))

        if self.ret_pcd:
            if self.ret_pcd == 'stl':
                pointcloud_Path = os.path.join(self.data_path, 'Points', 'stl', 'stl%03d_total.ply' % scene_num)
            else:
                pointcloud_Path = os.path.join(self.data_path, 'Points', self.ret_pcd, '%s%03d_l3.ply' % (self.ret_pcd, scene_num))
            pointcloud = o3d.io.read_point_cloud(pointcloud_Path, format = 'ply')
            pointcloud_points = torch.tensor(np.asarray(pointcloud.points))
            pointcloud_colors = torch.tensor(np.asarray(pointcloud.colors))
            pointcloud_normals = torch.tensor(np.asarray(pointcloud.normals))
        
        if self.ret_surface:
            surface_Path = os.path.join(self.data_path, 'Surfaces', self.ret_surface, '%s%03d_l3_surf_11_trim_8.ply' % (self.ret_surface, scene_num))
            surface = o3d.io.read_triangle_mesh(surface_Path)
            surface_points = torch.tensor(np.asarray(surface.vertices))
            surface_triangles = torch.tensor(np.asarray(surface.triangles))

        if self.ret_bb | self.ret_obsmask | self.ret_res | self.ret_margin:
            ObsMask_Path = os.path.join(self.data_path, 'ObsMask', 'ObsMask%d_10.mat' % scene_num)
            data = scipy.io.loadmat(ObsMask_Path)
            BB = torch.tensor(data['BB'])
            if self.ret_obsmask:
                ObsMask = torch.tensor(data['ObsMask'])
            Res = data['Res'][0][0]
            Margin = data['Margin'][0][0]

        if self.ret_plane:
            Plane_Path = os.path.join(self.data_path, 'ObsMask', 'Plane%d.mat' % scene_num)
            data = scipy.io.loadmat(Plane_Path)
            Plane = torch.tensor(data['P']).squeeze(1)

        if self.ret_rectified:
            RET['rectified_imgs'] = rectified_imgs
        if self.ret_cleaned:
            RET['cleaned_imgs'] = cleaned_imgs
        if self.ret_pcd:
            RET['pointcloud_points'] = pointcloud_points
            RET['pointcloud_colors'] = pointcloud_colors
            RET['pointcloud_normals'] = pointcloud_normals
        if self.ret_surface:
            RET['surface_points'] = surface_points
            RET['surface_triangles'] = surface_triangles
        if self.ret_obsmask:
            RET['ObsMask'] = ObsMask
        if self.ret_res:
            RET['Res'] = Res
        if self.ret_margin:
            RET['Margin'] = Margin
        if self.ret_bb:
            RET['BB'] = BB
        if self.ret_plane:
            RET['Plane'] = Plane
        if self.ret_intrinsics:
            RET['Intrinsics'] = Intrinsics
        if self.ret_extrinsics:
            RET['Extrinsics'] = Extrinsics

        return RET

# ====================================================================================================

class DTUDataset_scenes(Dataset):
    def __init__(self, data_path = None, light_subset: List[str] = None, view_subset: List[int] = None,
                ret_rectified = True,
                ret_cleaned = True,
                ret_pcd = 'stl',
                ret_surface = 'camp',
                ret_obsmask = True,
                ret_res = True,
                ret_margin = True,
                ret_bb = True,
                ret_plane = True,
                ret_intrinsics = True,
                ret_extrinsics = True) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        @param light_subset: 
        '''
        super(DTUDataset_scenes, self).__init__()
        self.data_path = os.path.normpath(data_path)
        
        self.scenes_names = os.listdir(os.path.join(self.data_path, 'Rectified'))
 
        self.light_subset = light_subset
        self.view_subset = view_subset

        self.ret_rectified = ret_rectified
        self.ret_cleaned = ret_cleaned
        self.ret_pcd = ret_pcd
        self.ret_surface = ret_surface
        self.ret_obsmask = ret_obsmask
        self.ret_res = ret_res
        self.ret_margin = ret_margin
        self.ret_bb = ret_bb
        self.ret_plane = ret_plane
        self.ret_intrinsics = ret_intrinsics
        self.ret_extrinsics = ret_extrinsics

    def __len__(self):
        return len(self.scenes_names)

    def __getitem__(self, index):

        if index > self.__len__():
            print('Index out of range.')
            return None
        
        scene_name = self.scenes_names[index]
        scene_num = 0
        for i in range(len(scene_name)):
            if ('0' <= scene_name[i]) & (scene_name[i] <= '9'):
                scene_num = scene_num * 10 + int(scene_name[i]) - int('0')

        Dataset_of_Scene = DTUDataset_per_Scene(self.data_path, scene_num, self.light_subset, self.view_subset,
                                                self.ret_rectified,
                                                self.ret_cleaned,
                                                self.ret_pcd,
                                                self.ret_surface,
                                                self.ret_obsmask,
                                                self.ret_res,
                                                self.ret_margin,
                                                self.ret_bb,
                                                self.ret_plane,
                                                self.ret_intrinsics,
                                                self.ret_extrinsics)

        return Dataset_of_Scene