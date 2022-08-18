import torch

from nvsbenchmark import PointCloudScoreUtils

if __name__ == '__main__':
    device = 'cuda:0'
    B = 5
    L1 = 10000
    L2 = 10000

    pointCloudHat = torch.rand(B, 3, L1, device=device)*100.0
    pointCloudGT = torch.rand(B, 3, L2, device=device)*100.0
    torch.save(pointCloudHat.cpu(), "pointCloudHat.pth")
    torch.save(pointCloudGT.cpu(), "pointCloudGT.pth")

    #pointCloudHat = torch.load("PointCloudHat.pth").to(device)
    #pointCloudGT = torch.load("PointCloudGT.pth").to(device)

    # distanceRes = PointCloudScoreUtils.getDist(pointCloudHat, pointCloudGT)
    accRes = PointCloudScoreUtils.getAccuracy(pointCloudHat, pointCloudGT)
    # compRes = PointCloudScoreUtils.getCompleteness(pointCloudHat, pointCloudGT)

    # print(distanceRes)
    print(accRes)
    # print(compRes)