from tkinter import W
import numpy as np
import torch
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset

class KITTIDataset(Dataset):
    def __init__(self, data_path, mode = 'training', is_mvs = True) -> None:
        super(KITTIDataset, self).__init__()
        self.data_path = os.path.normpath(data_path)
        self.mode = 'training'
        self.is_mvs = is_mvs

        self.scene_count = len(os.listdir(os.path.join(self.data_path, 'data_scene_flow_calib', mode, 'calib_cam_to_cam')))

        self.H = 375
        self.W = 1242
    
    def __len__(self):
        return self.scene_count
    
    def __getitem__(self, index):
        '''
        return:
            img_2s: is_mvs = False:   img_2s[0]: left image of the first pair
                                      img_2s[1]: left image of the second pair
                    is_mvs = True:    img_2s[10]: same as img_2s[0] above
                                      img_2s[11]: same as img_2s[1] above
            img_3s: is_mvs = False:   img_3s[0]: right image of the first pair
                                      img_3s[1]: right image of the second pair
                    is_mvs = True:    img_3s[10]: same as img_3s[0] above
                                      img_3s[11]: same as img_3s[1] above
            disp_xxx_0s: first left img's disparity map
            disp_xxx_1s: second left img's disparity map mapped onto the first left img using optical flow
            flow_xxxs: optical flow from first left to second left. specified in the first left image
                       first & second channel: u & v of optical flow
                       third channel: valid or not
            P_mat: intrinsics plus the  position with respect to first left (rectified)
            R_mat: Rotation matrices with respect to the first left (rectified)
            (xxx: noc or occ. Please refer to the readme.)
            (RGB ranges from 0 to 255)
        '''
        if index >= self.scene_count:
            print('Index out of range.')
            return {
                'img_2s': None, # is_mvs: 21 x H x W x 3  else: 2 x H x W x 3
                'img_3s': None, # is_mvs: 21 x H x W x 3  else: 2 x H x W x 3
                'disp_noc_0': None, # H x W
                'disp_noc_1': None, # H x W
                'disp_occ_0': None, # H x W
                'disp_occ_1': None, # H x W
                'flow_noc': None, # H x W x 3
                'flow_occ': None, # H x W x 3
                'P_mat_2s': None, # 2 x 3 x 4
                'P_mat_3s': None, # 2 x 3 x 4
                'R_mat_2s': None, # 2 x 3 x 3
                'R_mat_3s': None  # 2 x 3 x 3
            }
        
        if self.is_mvs == True:
            img_2s = torch.empty(21, self.H, self.W, 3)
            img_3s = torch.empty(21, self.H, self.W, 3)
        else:
            img_2s = torch.empty(2, self.H, self.W, 3)
            img_3s = torch.empty(2, self.H, self.W, 3)
        disp_noc_0 = torch.empty(self.H, self.W)
        disp_noc_1 = torch.empty(self.H, self.W)
        disp_occ_0 = torch.empty(self.H, self.W)
        disp_occ_1 = torch.empty(self.H, self.W)
        flow_noc   = torch.empty(self.H, self.W, 3)
        flow_occ   = torch.empty(self.H, self.W, 3)
        R_mat_2s   = torch.empty(2, 3, 3)
        R_mat_3s   = torch.empty(2, 3, 3)
        P_mat_2s   = torch.empty(2, 3, 4)
        P_mat_3s   = torch.empty(2, 3, 4)

        if self.is_mvs == True:
            IMG_PATH = os.path.join(self.data_path, 'data_scene_flow_multiview', self.mode)
            for i in range(21):
                data = cv2.imread(os.path.join(IMG_PATH, 'image_2', '%06d_%02d.png' % (index, i)))
                img_2s[i, :, :, :] = torch.tensor(data)

                data = cv2.imread(os.path.join(IMG_PATH, 'image_3', '%06d_%02d.png' % (index, i)))
                img_3s[i, :, :, :] = torch.tensor(data)
        
        else:
            IMG_PATH = os.path.join(self.data_path, 'data_scene_flow', self.mode)
            for i in range(2):
                data = cv2.imread(os.path.join(IMG_PATH, 'image_2', '%06d_1%d.png' % (index, i)))
                img_2s[i, :, :, :] = torch.tensor(data)

                data = cv2.imread(os.path.join(IMG_PATH, 'image_3', '%06d_1%d.png' % (index, i)))
                img_3s[i, :, :, :] = torch.tensor(data)
        
        if self.mode == 'training':
            DISP_PATH = os.path.join(self.data_path, 'data_scene_flow', self.mode)
            data = cv2.imread(os.path.join(DISP_PATH, 'disp_noc_0', '%06d_10.png' % index), cv2.IMREAD_UNCHANGED)
            data = torch.tensor( data.astype(float) )/256.0
            disp_noc_0 = data

            data = cv2.imread(os.path.join(DISP_PATH, 'disp_noc_1', '%06d_10.png' % index), cv2.IMREAD_UNCHANGED)
            data = torch.tensor( data.astype(float) )/256.0
            disp_noc_1 = data

            data = cv2.imread(os.path.join(DISP_PATH, 'disp_occ_0', '%06d_10.png' % index), cv2.IMREAD_UNCHANGED)
            data = torch.tensor( data.astype(float) )/256.0
            disp_occ_0 = data

            data = cv2.imread(os.path.join(DISP_PATH, 'disp_occ_1', '%06d_10.png' % index), cv2.IMREAD_UNCHANGED)
            data = torch.tensor( data.astype(float) )/256.0
            disp_occ_1 = data

            FLOW_PATH = os.path.join(self.data_path, 'data_scene_flow', self.mode)
            data = cv2.imread(os.path.join(FLOW_PATH, 'flow_noc', '%06d_10.png' % index), cv2.IMREAD_UNCHANGED)
            data = torch.tensor( data.astype(float) ).flip(2)
            data[:, :, :2] = (data[:, :, :2] - (1<<15)) / 64.0
            flow_noc = data

            data = cv2.imread(os.path.join(FLOW_PATH, 'flow_occ', '%06d_10.png' % index), cv2.IMREAD_UNCHANGED)
            data = torch.tensor( data.astype(float) ).flip(2)
            data[:, :, :2] = (data[:, :, :2] - (1<<15)) / 64.0
            flow_occ = data

        else:
            disp_noc_0 = disp_noc_1 = disp_occ_0 = disp_occ_1 = flow_noc = flow_occ = None
        
        CAM_PATH = os.path.join(self.data_path, 'data_scene_flow_calib', self.mode, 'calib_cam_to_cam', '%06d.txt' % index)
        with open(CAM_PATH, 'r') as f:
            lines = f.readlines()
        R = torch.tensor([float(x) for x in lines[8].split(' ')[1:]]).view(3,3)
        P = torch.tensor([float(x) for x in lines[9].split(' ')[1:]]).view(3,4)
        R_mat_2s[0, :, :] = R
        P_mat_2s[0 ,:, :] = P

        CAM_PATH = os.path.join(self.data_path, 'data_scene_flow_calib', self.mode, 'calib_cam_to_cam', '%06d.txt' % index)
        with open(CAM_PATH, 'r') as f:
            lines = f.readlines()
        R = torch.tensor([float(x) for x in lines[16].split(' ')[1:]]).view(3,3)
        P = torch.tensor([float(x) for x in lines[17].split(' ')[1:]]).view(3,4)
        R_mat_3s[0, :, :] = R
        P_mat_3s[0 ,:, :] = P

        CAM_PATH = os.path.join(self.data_path, 'data_scene_flow_calib', self.mode, 'calib_cam_to_cam', '%06d.txt' % index)
        with open(CAM_PATH, 'r') as f:
            lines = f.readlines()
        R = torch.tensor([float(x) for x in lines[24].split(' ')[1:]]).view(3,3)
        P = torch.tensor([float(x) for x in lines[25].split(' ')[1:]]).view(3,4)
        R_mat_2s[1, :, :] = R
        P_mat_2s[1 ,:, :] = P

        CAM_PATH = os.path.join(self.data_path, 'data_scene_flow_calib', self.mode, 'calib_cam_to_cam', '%06d.txt' % index)
        with open(CAM_PATH, 'r') as f:
            lines = f.readlines()
        R = torch.tensor([float(x) for x in lines[32].split(' ')[1:]]).view(3,3)
        P = torch.tensor([float(x) for x in lines[33].split(' ')[1:]]).view(3,4)
        R_mat_3s[1, :, :] = R
        P_mat_3s[1 ,:, :] = P
        
        return {
            'img_2s': img_2s, # is_mvs: 21 x H x W x 3  else: 2 x H x W x 3
            'img_3s': img_3s, # is_mvs: 21 x H x W x 3  else: 2 x H x W x 3
            'disp_noc_0': disp_noc_0, # H x W
            'disp_noc_1': disp_noc_1, # H x W
            'disp_occ_0': disp_occ_0, # H x W
            'disp_occ_1': disp_occ_1, # H x W
            'flow_noc': flow_noc, # H x W x 3
            'flow_occ': flow_occ, # H x W x 3
            'P_mat_2s': P_mat_2s, # 2 x 3 x 4
            'P_mat_3s': P_mat_3s, # 2 x 3 x 4
            'R_mat_2s': R_mat_2s, # 2 x 3 x 3
            'R_mat_3s': R_mat_3s  # 2 x 3 x 3
        } 
