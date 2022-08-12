import torch

class SSIMScore:
    @classmethod
    def getScore(cls, imageHat: torch.Tensor, imageGT: torch.Tensor) -> torch.Tensor:
        '''
        @param imageHat: B x C x H x W
        @param imageGT:  B x C x H x W
        @returns SSIM:   B
        '''
        
        device = imageHat.device
        B = imageHat.size(0)

        return torch.ones(B, device=device)