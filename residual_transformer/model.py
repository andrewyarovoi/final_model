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
from point_transformer_modules import PointTransformerBlock, TransitionDown


class ResidualPoint(nn.Module):
    def __init__(self, dim):
        self.mlp = nn.Conv1d(dim, dim, 1)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.mlp(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.mlp(out)
        out = self.bn(out)
        out = x + out
        out = self.relu(out)
        return out

class ResidualTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_mlp = nn.Sequential(
            nn.Conv1d(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.lin32 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.lin64 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.lin128 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.transformer1 = PointTransformerBlock(32)
        self.transformer2 = PointTransformerBlock(64)
        self.transformer3 = PointTransformerBlock(128)

        self.transitionDown1 = TransitionDown(32, 64)
        self.transitionDown2 = TransitionDown(64, 128)

        self.max_pool = nn.MaxPool2d((1, 8), stride=8)

        self.residual = ResidualPoint(128)

        self.final_mlp = nn.Sequential(
            nn.Conv1d(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 40),
            nn.BatchNorm1d(40),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        N = 1024
        D = 3
        x: (N, D)
        """
        out = self.first_mlp(x)                     # (1042,   3,  32)

        """??? unsure how to get in_features as input for transformer1 ???"""
        out = self.lin32(out)                       # (1024,  32,  32)
        out = self.transformer1((x,out))            # (1024,  32,  32) 

        out = self.lin32(out)                       # (1024,  32,  32)
        out = self.transitionDown1(out)             # ( 256,  32,  64) -- N/=4

        out = self.lin64(out)                       # ( 256,  64,  64)
        _, out = self.transformer2(out)             # ( 256,  64,  64)

        out = self.lin64(out)                       # ( 256,  64,  64)
        out = self.transitionDown2(out)             # (  64,  64, 128) -- N/=4

        out = self.lin128(out)                      # (  64, 128, 128)
        out = self.transformer3(out)                # (  64, 128, 128)

        out = self.residual(out)                    # (  64, 128, 128)
        out = self.max_pool(out)                    # (   8, 128, 128) -- N/=8

        out = self.residual(out)                    # (   8, 128, 128)
        out = self.max_pool(out)                    # (   1, 128, 128) -- N/=8

        out = self.final_mlp(out)                   # (   1, 128,  40)

        return out

if __name__=="main":
    data = torch.rand(2, 3, 2048)
    model = ResidualTransformer()

    out = model.forward(data)
    print(out.shape)


