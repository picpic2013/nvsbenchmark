import enum
import os
from typing import List
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from pytube import YouTube 

def getBestMatchingFrame(frameTimeStamp, times, maxFrameMatchingDistanceInNS=8000):
    for caseIdx, c in enumerate(times):
        distance = abs(c - frameTimeStamp)
        if distance < maxFrameMatchingDistanceInNS:
            return caseIdx
    return None

class RealEstate10KDataset(Dataset):
    def __init__(self, data_path = None, mode = 'train', res = '720p') -> None :
        '''
        @param mode: 'train' or 'test'
        @param res:  resolution. '720p' or '480p' or '360p' or '144p'
        '''
        super(RealEstate10KDataset, self).__init__()
        self.data_path = os.path.join(os.path.normpath(data_path), mode)
        self.res = res

        self.scene_names = os.listdir(self.data_path)

        if res == '720p':
            self.H, self.W = 720, 1280
        elif res == '480p':
            self.H, self.W = 480, 854
        elif res == '360p':
            self.H, self.W = 360, 640
        elif res == '144p':
            self.H, self.W = 144, 256

        if os.path.exists('RealEstate10K_Videos') == False:
            os.mkdir('RealEstate10K_Videos')

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, index):
        if index > self.__len__():
            print('Index out of range.')
            return {
                'extrinsics': None, # V x 4 x 4
                'intrinsics': None, # V x 3 x 3
                'imgs': None, # V x H x W x 3
                'Height': None,
                'Width': None
            }
        scene_name = self.scene_names[index]
        txt_path = os.path.join(self.data_path, scene_name)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        url = lines[0]

        V = len(lines) - 1

        imgs = torch.empty(V, self.H, self.W, 3)
        extrinsics = torch.empty(V, 3, 4)
        intrinsics = torch.zeros(V, 3, 3)
        times = torch.empty(V)

        for i,line in enumerate(lines[1:]):
            nums = line.split(' ')
            times[i] = float(nums[0])

            intrinsics[i, 0, 0] = float(nums[1]) * self.W
            intrinsics[i, 0, 2] = float(nums[2]) * self.W
            intrinsics[i, 1, 1] = float(nums[3]) * self.H
            intrinsics[i, 1, 2] = float(nums[4]) * self.H
            intrinsics[i, 2, 2] = 1.0
            
            for p in range(3):
                for q in range(4):
                    extrinsics[i, p, q] = float(nums[7 + p*4 + q])

        maxFrameMatchingDistanceInNS=8000

        video_path = os.path.join('RealEstate10K_Videos', scene_name[:-4]+'.mp4')
        if os.path.exists(video_path) == False:
            proxy_handler = {
            "http": " http://127.0.0.1:1080",
            'https': ' http://127.0.0.1:1080'
            }
            yt = YouTube(url = url, proxies = proxy_handler)
            stream = yt.streams.get_by_resolution(self.res)
            stream.download(output_path = 'RealEstate10K_Videos', filename = scene_name[:-4]+'.mp4')
        

        
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, int(max(0, times[0]/1000-1000)))
        caseIdx = 0
        while cap.isOpened():
            OK, frame = cap.read()
            if OK == False:
                break

            frameTimeStamp = (int)(round(cap.get(cv2.CAP_PROP_POS_MSEC)*1000))

            while (caseIdx < V) and (times[caseIdx] <= frameTimeStamp):
                if abs(times[caseIdx] - frameTimeStamp) < maxFrameMatchingDistanceInNS:
                    break
                caseIdx += 1
            
            if caseIdx >= V:
                break

            if abs(times[caseIdx] - frameTimeStamp) < maxFrameMatchingDistanceInNS:
                imgs[caseIdx, :, :, :] = torch.tensor(frame)
            

        return {
            'extrinsics': extrinsics, # V x 4 x 4
            'intrinsics': intrinsics, # V x 3 x 3
            'imgs': imgs, # V x H x W x 3
            'Height': self.H,
            'Width': self.W
        }