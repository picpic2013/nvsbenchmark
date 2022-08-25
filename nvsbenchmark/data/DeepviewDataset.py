import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DeepviewDataset(Dataset):
    def __init__(self, data_path = None, subset_name = None) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        '''
        super(DeepviewDataset, self).__init__()
        self.data_path = os.path.join(os.path.normpath(data_path), subset_name)
        print('This dataset loader will load data from \'', self.data_path, '\'')
        
        self.scenes_names = os.listdir(self.data_path)
        print('This subset contains: ', self.scenes_names)

    def __len__(self):
        return len(self.scenes_names)

    def __getitem__(self, index):
        '''
        return: 
            for each rig position:
                for each of the 16 cameras in the rig, a dictionary containing:
                    principal_point: Principal point of the camera in pixels.
                    pixel_aspect_ratio: Aspect ratio of the camera.
                    position: 3 element position of the camera, in meters.
                    focal_length: Effective focal length of the camera.
                    width: Width of the image.
                    height: Height of the image.
                    relative_path: Location of the image for this camera relative to the scene directory, e.g. 'cam_00/image_000.JPG'
                    orientation: 3 element axis-angle representation of the rotation ofthe camera. The length of the axis gives the rotation angle, in radians. 
        '''
        if index > self.__len__():
            print('Index out of range.')
            return {
                'imgs': None, # V x H x W x 3
                'R': None,    # V x 3 x 3
                't': None, 
                'pc': None, 
                'dep': None
            }
        scene_name = self.scenes_names[index]
        P = os.path.join(self.data_path, scene_name, 'models.json')

        with open(P, 'r') as f:
            raw_data = json.load(f)
        
        Rig = len(raw_data)
        Cam = len(raw_data[0])

        principal_points = torch.empty(Rig, Cam ,2)
        pixel_aspect_ratios = torch.empty(Rig , Cam)
        positions = torch.empty(Rig, Cam ,3)
        focal_lengths = torch.empty(Rig, Cam)
        orientations = torch.empty(Rig, Cam ,3)
        imgs = []

        for rig in range(Rig):
            imgs_tmp = []
            for cam in range(Cam):
                principal_points[rig, cam] = torch.tensor(data = raw_data[rig][cam]['principal_point'])
                pixel_aspect_ratios[rig, cam] = torch.tensor(data = raw_data[rig][cam]['pixel_aspect_ratio'])
                positions[rig, cam] = torch.tensor(data = raw_data[rig][cam]['position'])
                focal_lengths[rig, cam] = torch.tensor(data = raw_data[rig][cam]['focal_length'])
                orientations[rig, cam] = torch.tensor(data = raw_data[rig][cam]['orientation'])

                path_img = os.path.join(self.data_path, scene_name, os.path.normpath(raw_data[rig][cam]['relative_path']))
                img = Image.open(path_img)
                img_Tensor = transforms.ToTensor()(img)
                imgs_tmp.append(img_Tensor)
            imgs.append(imgs_tmp)

        return {
            # Rig represents view, Cam represents camera
            'principal_points': principal_points, # Rig x Cam x 2      
            'pixel_aspect_ratios': pixel_aspect_ratios, # Rig x Cam
            'positions': positions, # Rig x Cam x 3
            'focal_lengths': focal_lengths, # Rig x Cam
            'orientations': orientations, # Rig x Cam x 3
            'imgs': imgs, # Rig x Cam x 3 x H x W
            'depths': None,
            'pointclouds': None
        }