import torch

from nvsbenchmark import SSIMScore

if __name__ == '__main__':
    device = 'cuda:0'
    B, C, H, W = 5, 3, 100, 200

    img1 = torch.rand(B, C, H, W, device=device)*255.0

    img2 = torch.rand(B, C, H, W, device=device)*255.0

    #img2 = img1

    res = SSIMScore.getScore(img1, img2)

    print(res)