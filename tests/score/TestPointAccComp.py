import torch

from nvsbenchmark import PointCloudScoreUtils

if __name__ == '__main__':
    device = 'cuda:0'
    B = 5
    L1 = 1000
    L2 = 1009

    pointCloudHat = torch.rand(B, 3, L1, device=device)
    pointCloudGT = torch.rand(B, 3, L2, device=device)

    distanceRes = PointCloudScoreUtils.getDist(pointCloudHat, pointCloudGT)
    accRes = PointCloudScoreUtils.getAccuracy(pointCloudHat, pointCloudGT)
    compRes = PointCloudScoreUtils.getCompleteness(pointCloudHat, pointCloudGT)

    print(distanceRes)
    print(accRes)
    print(compRes)