import os
import json
from typing import List
import torch
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# =========================================Code from Spaces Dataset=================================================

_EPS = np.finfo(float).eps * 4.0

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.

    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3), dtype=numpy.float64)
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1.0])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, (1, 0, 0))
    >>> numpy.allclose(q, [0.06146124, 0, 0, 0.99810947])
    True

    """
    quaternion = np.zeros((4, ), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion


def _WorldFromCameraFromViewDict(position, orientation):
  """Fills the world from camera transform from the view_json.
  Args:
    view_json: A dictionary of view parameters.
  Returns:
     A 4x4 transform matrix representing the world from camera transform.
  """

  # The camera model transforms the 3d point X into a ray u in the local
  # coordinate system:
  #
  #  u = R * (X[0:2] - X[3] * c)
  #
  # Meaning the world from camera transform is [inv(R), c]

  transform = np.identity(4)
  transform[0:3, 3] = (position[0], position[1], position[2])
  angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
  angle = np.linalg.norm(angle_axis)
  epsilon = 1e-7
  if abs(angle) < epsilon:
    # No rotation
    return transform

  axis = angle_axis / angle
  rot_mat = quaternion_matrix(quaternion_about_axis(-angle, axis))
  transform[0:3, 0:3] = rot_mat[0:3, 0:3]
  return torch.tensor(data = transform)

def _IntrinsicsFromViewDict(focal_length, pixel_aspect_ratio, principal_point):
  """Fills the intrinsics matrix from view_params.
  Args:
    view_params: Dict view parameters.
  Returns:
     A 3x3 matrix representing the camera intrinsics.
  """
  intrinsics = np.identity(3)
  intrinsics[0, 0] = focal_length
  intrinsics[1, 1] = ( focal_length * pixel_aspect_ratio )
  intrinsics[0, 2] = principal_point[0]
  intrinsics[1, 2] = principal_point[1]
  return torch.tensor(data = intrinsics)

# =========================================Code from Spaces Dataset=================================================


def Clamp(pos, img, tarH, tarW):
    H, W = img.size(1), img.size(2)
    U = int(math.floor((H - tarH) / 2.0))
    D = int(math.ceil((H - tarH) / 2.0))
    L = int(math.floor((W - tarW) / 2.0))
    R = int(math.ceil((W - tarW) / 2.0))
    new_img = img[:, U : H - D, L : W - R]
    new_pos = pos - torch.tensor(data = [float(L), float(U)])

    return new_pos, new_img

def Calc_Intrinsic(pixel_aspect_ratio, fx_dx, principal_point):
    ret = torch.zeros(3, 3)
    ret[0, 0] = fx_dx
    ret[1, 1] = fx_dx * pixel_aspect_ratio
    ret[0, 2] = 1.0 

class DeepviewDataset(Dataset):
    def __init__(self, data_path = None, subset_name = None, rig_subset: List[int] = None, cam_subset: List[int] = None) -> None:
        '''
        @param data_path: the home path of the data set
        @param subset_name: the name of the subset of the dataset
        '''
        super(DeepviewDataset, self).__init__()
        self.data_path = os.path.join(os.path.normpath(data_path), subset_name)
        #print('This dataset loader will load data from \'', self.data_path, '\'')
        
        self.scenes_names = os.listdir(self.data_path)

        #print('This subset contains: ', self.scenes_names)

        self.H, self.W = 1190, 2048

        if subset_name == '2k':
            self.H, self.W = 1190, 2048
        elif subset_name == '800':
            self.H, self.W = 460, 800

        self.rig_subset = rig_subset
        self.cam_subset = cam_subset
        

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
                # 'principal_points': None, # Rig x Cam x 2      
                # 'pixel_aspect_ratios': None, # Rig x Cam
                # 'positions': None, # Rig x Cam x 3
                # 'focal_lengths': None, # Rig x Cam
                # 'orientations': None, # Rig x Cam x 3
                'extrinsics': None, # Rig x Cam x 4 x 4
                'intrinsics': None, # Rig x Cam x 3 x 3
                'imgs': None, # Rig x Cam x 3 x H x W
                'Height': None,
                'Width': None,
                'depths': None,
                'pointclouds': None
            }
        scene_name = self.scenes_names[index]
        P = os.path.join(self.data_path, scene_name, 'models.json')

        with open(P, 'r') as f:
            raw_data = json.load(f)
        
        Rig = len(raw_data)
        Cam = len(raw_data[0])

        '''
        principal_points = torch.empty(Rig, Cam ,2)
        pixel_aspect_ratios = torch.empty(Rig , Cam)
        positions = torch.empty(Rig, Cam ,3)
        focal_lengths = torch.empty(Rig, Cam)
        orientations = torch.empty(Rig, Cam ,3)
        '''
        extrinsics = torch.empty(Rig, Cam, 4, 4)
        intrinsics = torch.empty(Rig, Cam, 3, 3)
        imgs = torch.empty(Rig, Cam, 3, self.H, self.W)

        for rig in range(Rig):
            if (rig not in self.rig_subset) & (self.rig_subset != None):
                continue
            for cam in range(Cam):
                if (cam not in self.cam_subset) & (self.cam_subset != None):
                    continue
                principal_point = torch.tensor(data = raw_data[rig][cam]['principal_point'])
                pixel_aspect_ratio = torch.tensor(data = raw_data[rig][cam]['pixel_aspect_ratio'])
                position = torch.tensor(data = raw_data[rig][cam]['position'])
                focal_length = torch.tensor(data = raw_data[rig][cam]['focal_length'])
                orientation = torch.tensor(data = raw_data[rig][cam]['orientation'])

                path_img = os.path.join(self.data_path, scene_name, os.path.normpath(raw_data[rig][cam]['relative_path']))
                img = Image.open(path_img)
                img_Tensor = transforms.ToTensor()(img)
                principal_point, img_Tensor = Clamp(principal_point, img_Tensor, self.H, self.W)

                imgs[rig, cam, :, :, :] = img_Tensor

                extrinsics[rig ,cam, :, :] = _WorldFromCameraFromViewDict(position, orientation)
                intrinsics[rig, cam, :, :] = _IntrinsicsFromViewDict(focal_length, pixel_aspect_ratio, principal_point)

        return {
            # Rig represents view, Cam represents camera
            # 'principal_points': principal_points, # Rig x Cam x 2      
            # 'pixel_aspect_ratios': pixel_aspect_ratios, # Rig x Cam
            # 'positions': positions, # Rig x Cam x 3
            # 'focal_lengths': focal_lengths, # Rig x Cam
            # 'orientations': orientations, # Rig x Cam x 3
            'extrinsics': extrinsics, # Rig x Cam x 4 x 4
            'intrinsics': intrinsics, # Rig x Cam x 3 x 3
            'imgs': imgs, # Rig x Cam x 3 x H x W
            'Height': self.H,
            'Width': self.W,
            'depths': None,
            'pointclouds': None
        }