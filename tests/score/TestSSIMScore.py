import torch

from nvsbenchmark import SSIMScore

if __name__ == '__main__':
    device = 'cuda:0'
    B, C, H, W = 5, 3, 100, 200

    img1 = torch.rand(B, C, H, W, device=device)*255.0

    img2 = torch.rand(B, C, H, W, device=device)*255.0

    #img2 = img1

    #img1 = torch.load("img1.pth")
    #img2 = torch.load("img2.pth")

    #img1 = img1.unsqueeze(0).to(device)
    #img2 = img2.unsqueeze(0).to(device)

    res = SSIMScore.getScore(img1, img2)

    print(res)