import torch
import torch.nn as nn

class Net(nn.Module):
    """
    3-layer Convolutional Encoder + Pose Regression Head.
    Preserves 100% exact architecture from Untitled13.ipynb:
    - Conv2d(3, 32, (5,3), stride=(2,1), padding=(2,1)) + BN + ReLU
    - Conv2d(32, 64, (5,3), stride=(2,1), padding=(2,1)) + BN + ReLU
    - Conv2d(64, 128, (3,3), stride=(2,2), padding=1) + BN + ReLU
    - AdaptiveAvgPool2d((4,2)) + Flatten
    - Linear(128*4*2, m=256)
    - Head: ReLU + Linear(256, 128) + ReLU + Linear(128, 17*3)
    """
    def __init__(self, m=256, num_joints=17):
        super().__init__()
        self.m = m
        self.num_joints = num_joints
        
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, m),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(m, 128),
            nn.ReLU(),
            nn.Linear(128, num_joints * 3)
        )

    def forward(self, x):
        """
        Args:
            x: CSI input tensor (B, 3, 114, 10)
        Returns:
            pose: (B, 17, 3)
            z: Latent representation (B, m)
        """
        z = self.enc(x)
        p = self.head(z).view(-1, self.num_joints, 3)
        return p, z
