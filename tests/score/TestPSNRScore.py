import torch

from nvsbenchmark import PSNRScore

if __name__ == '__main__':
    B, C, H, W = 5, 3, 100, 200

    img = torch.rand(B, C, H, W)

    res = PSNRScore.getScore(img, img)

    print(res)