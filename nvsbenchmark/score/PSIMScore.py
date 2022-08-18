from unicodedata import decimal
import torch
import numpy as np
import math
import torch.nn.functional as F

class PSIMScore:
    @classmethod
    def getScore(cls, imageHat: torch.Tensor, imageGT: torch.Tensor) -> torch.Tensor:
        '''
        @param imageHat: B x C x H x W
        @param imageGT:  B x C x H x W
        @returns PSIM:   B
        '''
        
        def RGB2HSV(img) -> torch.Tensor:

            hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

            hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + 1e-8) ) [ img[:,2]==img.max(1)[0] ]
            hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + 1e-8) ) [ img[:,1]==img.max(1)[0] ]
            hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + 1e-8) ) [ img[:,0]==img.max(1)[0] ]) % 6

            hue[img.min(1)[0]==img.max(1)[0]] = 0.0
            hue = hue/6

            saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + 1e-8 )
            saturation[ img.max(1)[0]==0 ] = 0

            value = img.max(1)[0]
            
            hue = hue.unsqueeze(1)
            saturation = saturation.unsqueeze(1)
            value = value.unsqueeze(1)
            hsv = torch.cat([hue, saturation, value],dim=1)
            return hsv

        def Get_Dominant_color(img) -> torch.Tensor:
            S_t, B_t = 0.0, 0.0

            shp = img.size()
            ret = img.view(shp[0],shp[1]*shp[2]).contiguous().cpu()
            idx = np.asarray(np.where( (ret[1,:] > S_t) & (ret[2,:] > B_t) )).squeeze(0)
            ret = ret[:, idx]

            return ret
        
        def Histogram_Analysis(data) -> torch.Tensor:
            s_data = data.sort().values
            n = data.size(0)
            idx = [n//6, n*2//6, n*3//6, n*4//6, n*5//6, int(n*5.97//6)]
            return s_data[idx]

        def dist(x, y) -> float:
            return min(abs(x-y), 2-abs(x-y))


        def SSIM(imageHat: torch.Tensor, imageGT: torch.Tensor) -> torch.Tensor:
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

            return ret

        device = imageHat.device
        B = imageHat.size(0)
        gama = 1.0/4.0

        HSBHat = RGB2HSV(imageHat)
        HSBGT  = RGB2HSV(imageGT)

        S_c = torch.ones(B, device=device)

        for i in range(B):
            Domi_Color1 = Get_Dominant_color(HSBHat[i]).to(device)
            Domi_Color2 = Get_Dominant_color(HSBGT[i]).to(device)

            H1 = Histogram_Analysis(Domi_Color1[0])
            S1 = Histogram_Analysis(Domi_Color1[1])
            B1 = Histogram_Analysis(Domi_Color1[2])

            H2 = Histogram_Analysis(Domi_Color2[0])
            S2 = Histogram_Analysis(Domi_Color2[1])
            B2 = Histogram_Analysis(Domi_Color2[2])

            for j in range(6):
                A_H = 1.0 - dist(H1[j], H2[j])
                A_S = 1.0 - dist(S1[j], S2[j])
                A_B = 1.0 - dist(B1[j], B2[j])

                S_c[i] = S_c[i] * A_H * A_S * A_B
            
        S_c = S_c.pow(1.0/18.0)

        S_s = SSIM(imageHat, imageGT)

        print("S_c: ", S_c)
        print("S_s: ", S_s)

        S_p = (1.0 - gama) * S_s + gama * S_c

        return S_p