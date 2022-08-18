import torch

from nvsbenchmark import PSNRScore

if __name__ == '__main__':
    device = 'cuda:0'
    B, C, H, W = 5, 3, 100, 200

    img1 = torch.rand(B, C, H, W, device=device)

    img2 = torch.rand(B, C, H, W, device=device)

    # img2 = img1

    res = PSNRScore.getScore(img1, img2)

    print(res)