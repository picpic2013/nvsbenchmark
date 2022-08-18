import torch

from nvsbenchmark import PSIMScore

if __name__ == '__main__':
    device = 'cuda:0'
    B, C, H, W = 5, 3, 800, 600

    img1 = torch.rand(B, C, H, W, device=device)*255.0

    img2 = torch.rand(B, C, H, W, device=device)*255.0

    #img2 = img1

    res = PSIMScore.getScore(img1, img2)

    print(res)