# Implementations of loss functions and metrics that are useful for both estimation and evaluation
import torch
import torch.nn as nn

class ROILoss():
    def __init__(self, w_overlap=1.0, w_class=1.0, batch_wise=True) -> None:
        self.w_overlap = w_overlap
        self.w_class = w_class
        self.batch_wise = batch_wise

    def __call__(self, roi0, roi1):
        '''
        Implements Dice as the overlap loss cross all masks
        roi0: torch.tensor of shape (C,D1,H1,W1) for 3d where C is the number of masks
                                    (D1,H1,W1) for 2d
        roi1: torch.tensor of shape (C,D1,H1,W1) for 3d where C is the number of masks
                                    (D1,H1,W1) for 2d
        '''
        if self.batch_wise:
            roi0 = roi0.flatten()
            roi1 = roi1.flatten()
        else:
            roi0 = roi0.flatten(start_dim=1)
            roi1 = roi1.flatten(start_dim=1)
        
        loss = 0
        if self.w_overlap != 0:
            loss += self.w_overlap * self.overlap_loss(roi0, roi1)
        if self.w_class != 0:
            loss += self.w_class * self.class_loss(roi0, roi1)
        return loss

    def overlap_loss(self, roi0, roi1, eps=1e-8):
        '''
        Implements Dice as the overlap loss
        '''
        intersection = (roi0 * roi1).sum(dim=-1)
        union = roi0.sum(dim=-1) + roi1.sum(dim=-1)
        overlap = 2*intersection / (union+eps)
        return 1 - overlap.mean()
    
    def class_loss(self, roi0, roi1):
        '''
        Implements mean-square-error as the classification loss
        '''
        mse = ((roi0 - roi1)**2).mean(dim=-1)
        return mse.mean()
    
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true,y_pred):
        # 假设输入维度为 [B, C, D, H, W]，C 可以是 1（单类）或多类
        numerator = torch.sum(y_true * y_pred, dim=[1, 2, 3, 4]) * 2
        denominator = torch.sum(y_true, dim=[1, 2, 3, 4]) + torch.sum(y_pred, dim=[1, 2, 3, 4]) + self.eps
        dice_score = numerator / denominator
        return 1 - dice_score.mean()

class DDFLoss():
    def __init__(self, type='l2grad') -> None:
        self.type = type

    def __call__(self, ddf):
        '''
        ddf: torch.tensor of shape (H1,W1,D1,3) for 3d, or (H1,W1,2) for 2d
        '''
        if len(ddf.shape) == 4:
            match self.type.lower():
                case "l2grad":
                    loss = self.gradient_norm(ddf, l1_flag=False)
                case "l1grad":
                    loss = self.gradient_norm(ddf, l1_flag=True)
                case "bending":
                    loss = self.bending_energy(ddf)
                case _:
                    raise ValueError(f"Unknown DDFLoss type: {self.type}")
        elif len(ddf.shape) == 3:
            match self.type.lower():
                case "l2grad":
                    loss = self.gradient_norm_2d(ddf, l1_flag=False)
                case "l1grad":
                    loss = self.gradient_norm_2d(ddf, l1_flag=True)
                case "bending":
                    loss = self.bending_energy_2d(ddf)
                case _:
                    raise ValueError(f"Unknown DDFLoss type: {self.type}")

        return loss
    
    ## 3d versions
    def gradient_norm(self, ddf, l1_flag=False):
        '''
        implements L2-norm over 3d ddf gradients
        '''
        dFdx, dFdy, dFdz = self.ddf_gradients(ddf)
        if l1_flag:
            grad_norms = torch.abs(dFdx) + torch.abs(dFdy) + torch.abs(dFdz)
        else:
            grad_norms = dFdx**2 + dFdy**2 + dFdz**2
        
        return grad_norms.mean()
    
    def bending_energy(self, ddf):
        '''
        implements bending energy estimated over 3d ddf
        '''
        dFdx, dFdy, dFdz = self.ddf_gradients(ddf)
        d2Fdxx, d2Fdxy, d2Fdxz = self.ddf_gradients(dFdx)
        d2Fdyx, d2Fdyy, d2Fdyz = self.ddf_gradients(dFdy)
        d2Fdzx, d2Fdzy, d2Fdzz = self.ddf_gradients(dFdz)

        bending_energy = d2Fdxx**2 + d2Fdyy**2 + d2Fdzz**2 + \
            2*d2Fdxy*d2Fdyx + 2*d2Fdxz*d2Fdzx + 2*d2Fdyz*d2Fdzy
        
        return bending_energy.mean()
    
    @staticmethod
    def ddf_gradients(ddf):
        '''
        computes 3d ddf gradients
        '''
        dXdx, dXdy, dXdz = torch.gradient(ddf[...,0])
        dYdx, dYdy, dYdz = torch.gradient(ddf[...,1])
        dZdx, dZdy, dZdz = torch.gradient(ddf[...,2])
        dFdx = torch.stack([dXdx, dYdx, dZdx], dim=3)
        dFdy = torch.stack([dXdy, dYdy, dZdy], dim=3)
        dFdz = torch.stack([dXdz, dYdz, dZdz], dim=3)

        return dFdx, dFdy, dFdz

    ## 2d versions
    def gradient_norm_2d(self, ddf, l1_flag=False):
        '''
        implements L2-norm over 2d ddf gradients
        '''
        dFdx, dFdy = self.ddf_gradients_2d(ddf)
        if l1_flag:
            grad_norms = torch.abs(dFdx) + torch.abs(dFdy)
        else:
            grad_norms = dFdx**2 + dFdy**2
        
        return grad_norms.mean()

    def bending_energy_2d(self, ddf):
        '''
        implements bending energy estimated over 2d ddf
        '''
        dFdx, dFdy = self.ddf_gradients_2d(ddf)
        d2Fdxx, d2Fdxy = self.ddf_gradients_2d(dFdx)
        d2Fdyx, d2Fdyy = self.ddf_gradients_2d(dFdy)

        bending_energy = d2Fdxx**2 + d2Fdyy**2 + 2*d2Fdxy*d2Fdyx
        
        return bending_energy.mean()
    
    @staticmethod
    def ddf_gradients_2d(ddf):
        '''
        computes 2d ddf gradients
        '''
        dXdx, dXdy = torch.gradient(ddf[...,0])
        dYdx, dYdy = torch.gradient(ddf[...,1])
        dFdx = torch.stack([dXdx, dYdx], dim=2)
        dFdy = torch.stack([dXdy, dYdy], dim=2)

        return dFdx, dFdy
