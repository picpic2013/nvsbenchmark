from os import device_encoding
import torch

from nvsbenchmark import PointCloudScoreUtils

if __name__ == '__main__':
    device = 'cuda:0'
    B = 10
    L1 = 10000
    L2 = 10000

    pointCloudHat = torch.rand(B, 3, L1, device=device)*100.0
    pointCloudGT = torch.rand(B, 3, L2, device=device)*100.0
    #pointCloudHat = torch.load('pointCloudHat.pth')
    #pointCloudGT = torch.load('pointCloudGT.pth')

    ress = torch.tensor(data = [4.6416], device = device).expand(B)

    BBs = torch.tensor(data = [[[0,300], [0,300], [0,300]]], device = device).expand(B, 3, 2)

    mask = torch.randn(22, 22, 22, device = device)
    mask = torch.where(mask >= 0.5, 1 , 0)
    #mask = torch.load('mask.pth')

    torch.save(mask, 'mask.pth')

    masks = []
    for i in range(B):
        masks.append(mask.clone())


    plane = torch.tensor(data = [1.0, 1.0, 1.0, -50.0], device = device)
    torch.save(plane, 'plane.pth')
    planes = plane.expand(B, 4)

    torch.save(pointCloudHat, 'pointCloudHat.pth')
    torch.save(pointCloudGT,  'pointCloudGT.pth')

    distanceRes = PointCloudScoreUtils.getDist(pointCloudHat, pointCloudGT)
    accRes = PointCloudScoreUtils.getAccuracy(pointCloudHat, pointCloudGT)
    compRes = PointCloudScoreUtils.getCompleteness(pointCloudHat, pointCloudGT)

    print(distanceRes)
    print(accRes)
    print(compRes)