# @inproceedings{zhao2021point,
#   title={Point transformer},
#   author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
#   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
#   pages={16259--16268},
#   year={2021}
# }
# @inproceedings{
#     ma2022rethinking,
#     title={Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual {MLP} Framework},
#     author={Xu Ma and Can Qin and Haoxuan You and Haoxi Ran and Yun Fu},
#     booktitle={International Conference on Learning Representations},
#     year={2022},
#     url={https://openreview.net/forum?id=3Pbra-_u76D}
# }
# @inproceedings{xu2021paconv,
#   title={PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
#   author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
#   booktitle={CVPR},
#   year={2021}
# }
import torch
import torch.nn as nn
import torch.nn.functional as F
from point_transformer_modules import PointTransformerBlock, TransitionDown
from utility_modules import MLP, ResidualPointBlock

class ResidualTransformer(nn.Module):
    def __init__(self, k=40):
        super().__init__()

        self.first_mlp = MLP(3, 32)
        self.transformer1 = PointTransformerBlock(32)
        self.lin32 = nn.Linear(32, 32)
        self.trans_down1 = TransitionDown(32, 64, stride=8)
        self.lin64 = nn.Linear(64, 64)
        self.transformer2 = PointTransformerBlock(64)
        self.resp1_1 = ResidualPointBlock(64)
        self.resp1_2 = ResidualPointBlock(64)
        self.trans_down2 = TransitionDown(64, 128, stride=8)
        self.resp2_1 = ResidualPointBlock(128)
        self.resp2_2 = ResidualPointBlock(128)
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, k)
        )
    
    def forward(self, p):
        """
        p = [B, D, N]
        B = Batch Size (16)
        N = Num Points (1024)
        D = Feature Length (3)
        """
        # converts points to features using MLP
        x = self.first_mlp(p)                       # (1042,   3,  32)

        # apply transformer (doesn't change output size)
        p, x = self.transformer1((p,x))             # (1024,  32,  32) 
        x = self.lin32(x)                           # (1024,  32,  32)

        # apply transform down block (reduces N -> N/8 and increases D)
        p, x = self.trans_down1((p,x))              # ( 128,  32,  64)
        x = self.lin64(x)                           # ( 128,  64,  64)
        
        # apply transformer (doesn't change output size)
        p, x = self.transformer2((p,x))             # ( 128,  64,  64) 

        # apply resblocks
        x = self.resp1_1(x)                         # ( 128,  64,  64) 
        x = self.resp1_2(x)                         # ( 128,  64,  64) 

        # apply transform down block (reduces N -> N/8 and increases D)
        p, x = self.trans_down2((p,x))              # ( 16,  64,  128)

        # apply resblocks
        x = self.resp2_1(x)                         # ( 16,  128,  128)
        x = self.resp2_2(x)                         # ( 16,  128,  128)

        x = torch.max(x, 2, keepdim=True)[0]        # (   1, 128,  128)
        x = x.view(-1, 128)

        out = self.final_mlp(x)                     # out = [N, 40]

        return F.log_softmax(out, dim=1)

if __name__ == '__main__':
    device = torch.device('cuda')
    data = torch.rand(2, 3, 1024).to(device)
    model = ResidualTransformer().to(device)

    out = model.forward(data)
    print(out.shape)


