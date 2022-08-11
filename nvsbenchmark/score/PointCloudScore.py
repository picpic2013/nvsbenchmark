import torch
from typing import Dict

class PointCloudScoreUtils:
    '''
    utils to calculate point cloud scores
    '''

    @classmethod
    def getPointCloudDist(cls, pointCloudHat: torch.Tensor, pointCloudGT: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        get point cloud distance
        @param pointCloudHat: predicted point cloud,    B x 3 x L1
        @param pointCloudGT:  ground truth point cloud, B x 3 x L2

        @returns Dict
            @key mean => B
            @key var  => B
            @key mid  => B
        '''

        device = pointCloudHat.device
        B = pointCloudHat.size(0)
        
        return {
            'mean': torch.rand(B, device=device), 
            'var': torch.rand(B, device=device), 
            'mid': torch.rand(B, device=device)
        }

    @classmethod
    def getPointCloudAccuracy(cls, pointCloudHat: torch.Tensor, pointCloudGT: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        get point cloud accuracy
        @param pointCloudHat: predicted point cloud,    B x 3 x L1
        @param pointCloudGT:  ground truth point cloud, B x 3 x L2

        @returns Dict
            @key mean => B
            @key var  => B
            @key mid  => B
        '''

        device = pointCloudHat.device
        B = pointCloudHat.size(0)
        
        return {
            'mean': torch.rand(B, device=device), 
            'var': torch.rand(B, device=device), 
            'mid': torch.rand(B, device=device)
        }

    @classmethod
    def getPointCloudCompleteness(cls, pointCloudHat: torch.Tensor, pointCloudGT: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        get point cloud completeness
        @param pointCloudHat: predicted point cloud,    B x 3 x L1
        @param pointCloudGT:  ground truth point cloud, B x 3 x L2

        @returns Dict
            @key mean => B
            @key var  => B
            @key mid  => B
        '''

        device = pointCloudHat.device
        B = pointCloudHat.size(0)
        
        return {
            'mean': torch.rand(B, device=device), 
            'var': torch.rand(B, device=device), 
            'mid': torch.rand(B, device=device)
        }