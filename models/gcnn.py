import torch
import torch.nn as nn
import torch.nn.functional as F

class Z2ConvG(nn.Module):
    """
        Lifting convolution. Takes an image (signal on Omega) and returns a signal on the (affine) group G = Aff(H).
    """
    def __init__(self, group, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.zeros(size=(out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, pad=(self.padding,) * 4, mode="circular")
        x = torch.stack([F.conv2d(x, weight=g(self.weight), padding=0, stride=self.stride) for g in self.group.inverses], dim=1)
        return x

def merge_dims(x):
    """
        Merge channel and group dimensions.
    """
    return x.view(x.shape[0], -1, x.shape[3], x.shape[4])

def unmerge_dims(x, n):
    """
        Unmerge channel and group dimensions. Need group order argument n.
    """
    return x.view(x.shape[0], n, -1, x.shape[2], x.shape[3])

class GConv(nn.Module):
    """
        G-convolution layer.
    """
    def __init__(self, group, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.zeros(size=(out_channels, in_channels, kernel_size, kernel_size)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        # Merge dimensions needed because of padding
        x = merge_dims(x) 
        if self.padding:
            x = F.pad(x, pad=(self.padding,) * 4, mode="circular")
        x = unmerge_dims(x, self.group.order)
        # Convolve over all group elements (the channels are "multiplied" with g as well, i.e., we are permuting the group dimension)
        feature_maps = [F.conv2d(merge_dims(x[:,I,:,:]), g(self.weight), padding=0, stride=self.stride, groups=self.group.order) for I, g in zip(self.group.cayley_table, self.group.inverses)]
        x = torch.stack(feature_maps, dim=1)
        return x

class GPool(nn.Module):
    """
        Pool over group dimension. Takes a signal on the group G and returns a signal on Omega, i.e., an image.
        If the argument 'group' is given, pool in addition over all transformed versions of input making the output G-invariant.
    """
    def __init__(self, group=None, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "max", "min"], f"Invalid reduction type \"{reduction}\" specified."
        self.reduction = reduction
        self.group = group

    def forward(self, x):
        if self.group:
            x = torch.cat([g(x) for g in self.group.functions], dim=1)
        match self.reduction:
            case "mean":
                return torch.mean(x, dim=1)
            case "sum":
                return torch.sum(x, dim=1)
            case "max":
                return torch.max(x, dim=1).values
            case "min":
                return torch.min(x, dim=1).values

class GMaxPool2d(nn.Module):
    """
        Wrapper for max pooling layer to support an extra group dimension.
    """
    def __init__(self, group, kernel_size=2, stride=2):
        super().__init__()
        self.n = group.order
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        y = merge_dims(x)
        y = F.max_pool2d(y, self.kernel_size, self.stride)
        y = unmerge_dims(y, self.n)
        return y

class GBatchNorm2d(nn.Module):
    def __init__(self, group, num_features):
        super().__init__()
        self.n = group.order
        self.batchnorm = nn.BatchNorm2d(num_features * self.n, affine=False)

    def forward(self, x):
        y = merge_dims(x)
        y = self.batchnorm(y)
        y = unmerge_dims(y, self.n)
        return y

class GNNModel(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.lifting_conv = nn.Sequential(
                                Z2ConvG(group, in_channels=3, out_channels=32, kernel_size=5, padding=2),
                                GBatchNorm2d(group, 32),
                                nn.ReLU(),
                                GMaxPool2d(group, 5, 5),
                                )

        self.g_conv_layers = nn.Sequential(
                                GConv(group, in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 32),
                                nn.ReLU(),

                                GConv(group, in_channels=32, out_channels=16, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 16),
                                nn.ReLU(),
                                
                                GMaxPool2d(group, 2, 2),
                                
                                GConv(group, in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 16),
                                nn.ReLU(),

                                GConv(group, in_channels=16, out_channels=8, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 8),
                                nn.ReLU(),
                                
                                GMaxPool2d(group, 2, 2),
                                
                                GConv(group, in_channels=8, out_channels=8, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 8),
                                nn.ReLU(),

                                GConv(group, in_channels=8, out_channels=4, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 4),
                                nn.ReLU(),

                                GConv(group, in_channels=4, out_channels=4, kernel_size=3, padding=1),
                                GBatchNorm2d(group, 4),
                                nn.ReLU(),
                                
                                GPool(group, reduction="mean"),
                                )
        self.fc = nn.Sequential(
                        nn.Linear(400, 1),
                        nn.Sigmoid()
                    )

    def forward(self, x):
        x = self.lifting_conv(x)
        x = self.g_conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(-1)
        return x
