import torch

from .cnn import CNNModel
from utils import d2_e, d2_r, d2_mh, d2_mv

class SmoothCNNModel(CNNModel):
    def forward(self, x):
        """
            Expand batch dimension with transformed versions of the input.
            Average predictions over all four symmetries.
        """
        x = torch.cat([d2_e(x), d2_r(x), d2_mh(x), d2_mv(x)], dim=0)
        x = self.convolutions(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(4, -1)
        x = torch.mean(x, dim=0)
        return x
