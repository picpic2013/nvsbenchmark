import torch

from nvsbenchmark import PointCloudScoreUtils

if __name__ == '__main__':
    device = 'cuda:0'
    B = 5
    L1 = 1000
    L2 = 1009

    pointCloudHat = torch.rand(B, 3, L1, device=device)
    pointCloudGT = torch.rand(B, 3, L2, device=device)

    distanceRes = PointCloudScoreUtils.getPointCloudDist(pointCloudHat, pointCloudGT)
    accRes = PointCloudScoreUtils.getPointCloudAccuracy(pointCloudHat, pointCloudGT)
    compRes = PointCloudScoreUtils.getPointCloudCompleteness(pointCloudHat, pointCloudGT)

    print(distanceRes)
    print(accRes)
    print(compRes)