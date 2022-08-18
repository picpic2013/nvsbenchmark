import torch
import math
import numpy as np

class PSNRScore:
    @classmethod
    def getScore(cls, imageHat: torch.Tensor, imageGT: torch.Tensor) -> torch.Tensor:
        '''
        @param imageHat: B x C x H x W
        @param imageGT:  B x C x H x W
        @returns PSNR:   B

        averaging three color channels together

        color ranging from 0 to 255

        '''

        device = imageHat.device
        B = imageHat.size(0)

        ret = torch.ones(B, device=device)
        
        for i in range(B):
            sqrt_mse = math.sqrt(((imageHat[i,:,:,:] - imageGT[i,:,:,:])**2).mean())
            if sqrt_mse == 0:
                ret[i] = 100
                continue
            PIXEL_MAX = 255.0
            ret[i] = 20.0 * math.log10(PIXEL_MAX / sqrt_mse)
        
        return ret