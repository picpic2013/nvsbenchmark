from numpy import double
import torch
import torch.nn.functional as F
import math

class SSIMScore:

    @classmethod
    def getScore(cls, imageHat: torch.Tensor, imageGT: torch.Tensor) -> torch.Tensor:
        '''
        @param imageHat: B x C x H x W
        @param imageGT:  B x C x H x W
        @returns SSIM:   B

        employing gaussian kernel to convolve
        '''
        
        def gaussian(window_size, sigma) -> torch.Tensor:
            '''
            @returns a vector sampled from gaussian distribution with sigma
            '''
            ret = torch.Tensor([math.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            ret = ret/ret.sum()
            return ret

        def create_window(window_size, channel=1) -> torch.Tensor:
            gauss_vec = gaussian(window_size, 1.5).unsqueeze(1)

            WD = gauss_vec.mm(gauss_vec.t()).unsqueeze(0).unsqueeze(0)
            WD = WD.expand(channel, 1, window_size, window_size).contiguous()

            return WD

        window_size, max_val, min_val = 11, 255, 0
        K1, K2 = 0.01, 0.03
        L = max_val-min_val
        C1, C2 = (K1*L)**2, (K2*L)**2

        device = imageHat.device
        B = imageHat.size(0)
        C = imageHat.size(1)

        WD = create_window(window_size, C).to(device)

        mu1 = F.conv2d(imageHat, WD, groups=C)
        mu2 = F.conv2d(imageGT,  WD, groups=C)

        mu1_pow2 = mu1**2
        mu2_pow2 = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_pow2 = F.conv2d(imageHat**2, WD, groups=C) - mu1_pow2
        sigma2_pow2 = F.conv2d(imageGT**2,  WD, groups=C) - mu2_pow2
        sigma12 = F.conv2d(imageHat*imageGT, WD, groups=C) - mu1_mu2

        SSIM_patch = ((2.0*mu1_mu2 + C1) * (2.0*sigma12 + C2)) / ((mu1_pow2 + mu2_pow2 + C1) * (sigma1_pow2 + sigma2_pow2 + C2))

        ret = SSIM_patch.mean(1)
        ret = torch.Tensor([ret[x,...].mean() for x in range(B)])

        ret=ret.to(device)

        print("ret.size(): ", ret.size())

        return ret