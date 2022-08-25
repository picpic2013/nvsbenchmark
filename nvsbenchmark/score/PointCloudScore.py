import torch
import numpy as np

from sklearn.neighbors import KDTree
from typing import Dict

def reducePts(PC: torch.Tensor, dst) -> torch.Tensor:
    ret = PC.to('cpu')
    points = np.asarray(ret.t())
    n = PC.size(1)
    rand_order = np.random.permutation(n)
    batch_size = int(4e6)
    chosen_points_bool = np.ones(n)
    tree = KDTree(points)

    batch_size = min(batch_size, n)

    for i in range(0, n, batch_size):
        batch_idx = rand_order[i:i+batch_size]
        batch_pts = np.asarray(points[batch_idx])
        near_pts_idx = tree.query_radius(batch_pts, r=dst)

        num = len(near_pts_idx)

        for j in range(num):
            idx = batch_idx[j]
            if chosen_points_bool[idx]:
                chosen_points_bool[near_pts_idx[j]] = 0
                chosen_points_bool[idx] = 1

    chosen_points_idx = np.asarray(np.where(chosen_points_bool == 1)).squeeze(0)
    ret = ret[:, chosen_points_idx].to(PC.device)
    return ret

def build_Hat_Filt(PC: torch.Tensor, BB, res, mask) -> torch.Tensor:
    n = PC.size(1)
    mask_shape = mask.size()
    filt = torch.zeros(n, device = PC.device)
    
    min_bound = BB[:, 0].unsqueeze(1).expand(3, n)
    qv = PC.clone()

    qv = ((qv - min_bound) / res +0.5)
    qv = qv.round().int()

    in_bounds_idx = torch.where((0 <= qv[0, :]) & (qv[0, :] < mask_shape[0]) & \
                                (0 <= qv[1, :]) & (qv[1, :] < mask_shape[1]) & \
                                (0 <= qv[2, :]) & (qv[2, :] < mask_shape[2]), 1, 0)
    in_bounds_idx = torch.nonzero(in_bounds_idx).squeeze(1)
    in_bounds_pts = qv[:, in_bounds_idx]

    mask_flattened = mask.view(mask_shape[0] * mask_shape[1] * mask_shape[2])
    in_bounds_pts = qv[0, in_bounds_idx] * mask_shape[1] * mask_shape[2] + \
                    qv[1, in_bounds_idx] * mask_shape[2] + \
                    qv[2, in_bounds_idx]
    in_bounds_pts = in_bounds_pts.long()

    is_mask_idx = torch.where(mask_flattened[in_bounds_pts] == True, 1, 0)

    is_mask_idx = torch.nonzero(is_mask_idx).squeeze(1)

    filt[in_bounds_idx[is_mask_idx]] = 1

    return filt

def build_GT_Filt(PC: torch.Tensor, P) -> torch.Tensor:
    n = PC.size(1)

    P_t = P.unsqueeze(0)

    tmp_ones = torch.ones(1, n, device = PC.device)
    PC_ex = torch.cat((PC, tmp_ones), dim = 0)

    dist = torch.mm(P_t, PC_ex).squeeze(0)
    filt = torch.where(dist >= 0, 1, 0)

    return filt


def PC_Dist_CD(PC_from, PC_to, BB, Max_Dist, filt):
    num_from = PC_from.size(1)
    Dist = torch.zeros(num_from, dtype = torch.float64).to(PC_from.device)
    Range = torch.div(BB[:, 1] - BB[:, 0], Max_Dist, rounding_mode='floor')


    for x in range( int( Range[0] ) + 1 ):
        for y in range( int( Range[1] ) + 1 ):
            for z in range( int( Range[2] ) + 1 ):
                Low = BB[:, 0] + torch.tensor(data = [x, y, z], device = PC_from.device)*Max_Dist
                High = Low + Max_Dist

                idxF = torch.where((Low[0] <= PC_from[0, :]) & (Low[1] <= PC_from[1, :]) & (Low[2] <= PC_from[2, :]) & \
                                   (PC_from[0, :] < High[0]) & (PC_from[1, :] < High[1]) & (PC_from[2, :] < High[2]), 1, 0)
                idxF = (torch.nonzero(idxF)).squeeze(1)
                if idxF.size(0) == 0:
                    continue

                Low = Low - Max_Dist
                High = High + Max_Dist

                idxT = torch.where((Low[0] <= PC_to[0, :]) & (Low[1] <= PC_to[1, :]) & (Low[2] <= PC_to[2, :]) & \
                                   (PC_to[0, :] < High[0]) & (PC_to[1, :] < High[1]) & (PC_to[2, :] < High[2]), 1, 0)
                idxT = (torch.nonzero(idxT)).squeeze(1)
                
                if idxT.size(0) == 0:
                    Dist[idxF] = Max_Dist

                else:
                    ptsF = torch.index_select(PC_from.t(), dim=0, index=idxF).cpu().numpy()
                    ptsT = torch.index_select(PC_to.t(), dim=0, index=idxT).cpu().numpy()
                    tree = KDTree(ptsT)
                    Dist_tmp, __ = tree.query(X=ptsF)
                    src=torch.tensor(data = Dist_tmp, dtype = torch.float64).to(PC_from.device).squeeze(1)
                    Dist.scatter_(dim=0, index=idxF, src=src)

    Dist_idx = torch.nonzero(filt).squeeze(1)
    Dist = Dist[Dist_idx]

    return {
        'mean': Dist.mean(),
        'var': Dist.var(),
        'mid': Dist.median()
    }


class PointCloudScoreUtils:
    '''
    utils to calculate point cloud scores
    '''

    @classmethod
    def getDist(cls, pointCloudHat: torch.Tensor, pointCloudGT: torch.Tensor, BBs: torch.Tensor = None, masks: list = None, planes: torch.Tensor = None, \
                         ress: torch.Tensor = None, dst: float = 0.2, Max_Dist: float = 60) -> Dict[str, torch.Tensor]:
        '''
        get point cloud distance
        @param pointCloudHat: predicted point cloud,    B x 3 x L1
        @param pointCloudGT:  ground truth point cloud, B x 3 x L2
        @param BBs         :  bounding boxes,          B x 3 x 2
        @param masks       :  masks for pointCloudHat,   [N1, N2, ..., N_B]
        @param planes      :  planes to split pointCloudGT into two parts,    B x 4
        @param ress         :  point cloud resolutions,   B
        @param dst         :  minimum distance between points
        @param Max_dist    :  maximum distance considered into caculation

        @returns Dict
            @key dist=> B
        '''

        device = pointCloudHat.device
        B = pointCloudHat.size(0)
        
        dist_ret = torch.Tensor(B)

        for scan_num in range(B):
            PC_Hat = reducePts(pointCloudHat[scan_num], dst)
            PC_GT  = reducePts(pointCloudGT[scan_num],  dst) # 如果GT已经确保了没有相距0.2的点的话，可以去掉

            BB = torch.tensor(data = [[0,300],[0,300],[0,300]], device = device)
            if BBs != None:
                BB = BBs[scan_num]

            Hat_Filt = torch.ones(PC_Hat.size(1), device = device)
            if masks != None:
                Hat_Filt = build_Hat_Filt(PC_Hat, BB, ress[scan_num], masks[scan_num])
            
            GT_Filt = torch.ones(PC_GT.size(1), device = device)
            if planes != None:
                GT_Filt = build_GT_Filt(PC_GT, planes[scan_num])

            Acc = PC_Dist_CD(PC_Hat, PC_GT, BB, Max_Dist, Hat_Filt)
            Complt = PC_Dist_CD(PC_GT, PC_Hat, BB, Max_Dist, GT_Filt)

            dist_ret[scan_num] = Acc['mean'] * PC_Hat.size(1) + Complt['mean'] * PC_GT.size(1)

        dist_ret = dist_ret.to(device)

        return {
            'dist': torch.rand(B, device=device)
        }

    @classmethod
    def getAccuracy(cls, pointCloudHat: torch.Tensor, pointCloudGT: torch.Tensor, BBs: torch.Tensor = None, masks: list = None, planes: torch.Tensor = None, \
                         ress: torch.Tensor = None, dst: float = 0.2, Max_Dist: float = 60) -> Dict[str, torch.Tensor]:
        '''
        get point cloud accuracy
        @param pointCloudHat: predicted point cloud,    B x 3 x L1
        @param pointCloudGT:  ground truth point cloud, B x 3 x L2
        @param BBs         :  bounding boxes,          B x 3 x 2
        @param masks       :  masks for pointCloudHat,   [N1, N2, ..., N_B]
        @param planes      :  planes to split pointCloudGT into two parts,    B x 4
        @param ress         :  point cloud resolutions,   B
        @param dst         :  minimum distance between points
        @param Max_dist    :  maximum distance considered into caculation

        @returns Dict
            @key mean => B
            @key var  => B
            @key mid  => B
        '''

        device = pointCloudHat.device
        B = pointCloudHat.size(0)
        
        mean_ret = torch.Tensor(B)
        var_ret = torch.Tensor(B)
        mid_ret = torch.Tensor(B)

        for scan_num in range(B):
            PC_Hat = reducePts(pointCloudHat[scan_num], dst)
            PC_GT  = reducePts(pointCloudGT[scan_num],  dst) # 如果GT已经确保了没有相距0.2的点的话，可以去掉

            BB = torch.tensor(data = [[0,300],[0,300],[0,300]], device = device)
            if BBs != None:
                BB = BBs[scan_num]

            Hat_Filt = torch.ones(PC_Hat.size(1), device = device)
            if masks != None:
                Hat_Filt = build_Hat_Filt(PC_Hat, BB, ress[scan_num], masks[scan_num])

            Acc = PC_Dist_CD(PC_Hat, PC_GT, BB, Max_Dist, Hat_Filt)

            mean_ret[scan_num] = Acc['mean']
            var_ret[scan_num] = Acc['var']
            mid_ret[scan_num] = Acc['mid']
        
        mean_ret = mean_ret.to(device)
        var_ret = var_ret.to(device)
        mid_ret = mid_ret.to(device)

        return {
            'mean': mean_ret, 
            'var': var_ret, 
            'mid': mid_ret
        }

    @classmethod
    def getCompleteness(cls, pointCloudHat: torch.Tensor, pointCloudGT: torch.Tensor, BBs: torch.Tensor = None, masks: list = None, planes: torch.Tensor = None, \
                         ress: torch.Tensor = None, dst: float = 0.2, Max_Dist: float = 60) -> Dict[str, torch.Tensor]:
        '''
        get point cloud completeness
        @param pointCloudHat: predicted point cloud,    B x 3 x L1
        @param pointCloudGT:  ground truth point cloud, B x 3 x L2
        @param BBs         :  bounding boxes,          B x 3 x 2
        @param masks       :  masks for pointCloudHat,   [N1, N2, ..., N_B]
        @param planes      :  planes to split pointCloudGT into two parts,    B x 4 x 1
        @param ress         :  point cloud resolutions,   B
        @param dst         :  minimum distance between points
        @param Max_dist    :  maximum distance considered into caculation

        @returns Dict
            @key mean => B
            @key var  => B
            @key mid  => B
        '''

        device = pointCloudHat.device
        B = pointCloudHat.size(0)
        
        mean_ret = torch.Tensor(B)
        var_ret = torch.Tensor(B)
        mid_ret = torch.Tensor(B)

        for scan_num in range(B):
            PC_Hat = reducePts(pointCloudHat[scan_num], dst)
            PC_GT  = reducePts(pointCloudGT[scan_num],  dst) # 如果GT已经确保了没有相距0.2的点的话，可以去掉

            BB = torch.tensor(data = [[0,300],[0,300],[0,300]], device = device)
            if BBs != None:
                BB = BBs[scan_num]

            GT_Filt = torch.ones(PC_GT.size(1), device = device)
            if planes != None:
                GT_Filt = build_GT_Filt(PC_GT, planes[scan_num])

            Complt = PC_Dist_CD(PC_GT, PC_Hat, BB, Max_Dist, GT_Filt)

            mean_ret[scan_num] = Complt['mean']
            var_ret[scan_num] = Complt['var']
            mid_ret[scan_num] = Complt['mid']
        
        mean_ret = mean_ret.to(device)
        var_ret = var_ret.to(device)
        mid_ret = mid_ret.to(device)

        return {
            'mean': mean_ret, 
            'var': var_ret, 
            'mid': mid_ret
        }