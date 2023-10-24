import torch
import torch.nn as nn

import torch
import torch.nn as nn

# Efficient Spatial Pyramid块
class ESP3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 5, 9]):
        super(ESP3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.ModuleList([nn.AdaptiveAvgPool3d((1, None, None)) for _ in dilations])
        self.conv_dilations = nn.ModuleList([nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, dilation=(1, d, d)) for d in dilations])
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        x = torch.cat([conv_dilation(pool(x)) for conv_dilation, pool in zip(self.conv_dilations, self.pool)], dim=1)
        return x

# 3D-ESPNet模型
class ESP3DNet(nn.Module):
    def __init__(self, num_classes):
        super(ESP3DNet, self).__init__()
        self.initial_conv = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ESP3DBlock(16, 28, kernel_size=(1,3,3), dilations=[1, 2, 5, 9])
        self.layer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ESP3DBlock(28, 36, kernel_size=3, dilations=[1, 2, 5, 9])
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ESP3DBlock(36, 48, kernel_size=3, dilations=[1, 2, 5, 9])
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ESP3DBlock(48, 64, kernel_size=3, dilations=[1, 2, 5, 9])
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ESP3DBlock(64, 80, kernel_size=3, dilations=[1, 2, 5, 9])
        )
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.relu(self.bn(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.up(x)
        x = self.final_conv(x)
        return x

# 定义ESP3DNet模型
def esp3dnet(num_classes):
    model = ESP3DNet(num_classes)
    return model


if __name__=='__main__':
    import time
    import numpy as np
    model = ESP3DNet(3).cuda()
    input = np.random.random((1, 1, 16, 160, 160)).astype(np.float32)
    img = torch.tensor(input).cuda()
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        for i in range(1000):
            model(img)
            torch.cuda.synchronize()
    print((time.time() - t))

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    from torchprofile import profile_macs
    macs = profile_macs(model, img)
    print('model flops (G):', macs / 1.e9, 'input_size:', img.shape)
    