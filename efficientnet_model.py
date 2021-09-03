import torch
import torch.nn as nn
from math import ceil

class conv_block (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              )

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.silu = nn.SiLU()

        # efficientNet uses SiLU

    def forward(self, X):
        out = self.silu(self.bn(self.conv(X)))

        return out

class SElayer (nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SElayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)

        # excitation
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=in_channels,
                                out_features=in_channels//reduction)
        self.fc2 = nn.Linear(in_features=in_channels//reduction,
                             out_features=in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward (self, X):

        out = self.squeeze(X)
        # batch_size, channel, _, _ = x.size()
        batch_size, channel, _, _ = X.shape

        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        # out = out.view(batch_size, channel, 1, 1)
        out = out.reshape(batch_size, channel, 1, 1)

        # element-wise multiplied with X ???
        # out = out.expand(X.size())
        out = out.expand_as(X)

        return X * out

class depthwise_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 kernel_size, padding, groups=1,
                 act_type1=False):
        super(depthwise_block, self).__init__()

        assert act_type1 == True or act_type1 == False
        self.act_type = act_type1

        # dw

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=groups
                               )

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)

        # mobilenet_v3 uses both Hardswish and ReLU

        if act_type1:
            self.relu = nn.ReLU6()

        self.hardswish = nn.Hardswish()

        # pw

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=(0, 0),
                               )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):

        if self.act_type:
            out = self.relu(self.bn1(self.conv1(X)))
        else:
            out = self.hardswish(self.bn1(self.conv1(X)))

        out = self.bn2(self.conv2(out))

        # mobilev2 doesnt have relu in pw

        return out

class InvertedResidual_block (nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, block_type1=False,
                 expand_ratio=6, act_type_=False, use_se=False,
                 survival_prob=0.8):

        super(InvertedResidual_block, self).__init__()

        assert block_type1 == True or block_type1 == False
        assert use_se == True or use_se == False
        self.block_type = block_type1
        self.se_layer = use_se
        self.survival_prob = 0.8

        # expansion

        self.conv1 = conv_block(in_channels=in_channels,
                               out_channels=in_channels*expand_ratio,
                                kernel_size=(1,1),
                               stride=(1,1),
                                padding=(0,0))

        # depthwise_block = depthwise + pointwise

        self.convdw = depthwise_block(in_channels=in_channels*expand_ratio,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      groups=in_channels*expand_ratio,
                                      act_type1=act_type_)

        if block_type1:
            self.padd_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))

        if use_se:
            self.se = SElayer(in_channels=out_channels, reduction=16)


    def stochastic_depth (self, X):
        if not self.training:
            return X

        binary_tensor = torch.randn(X.shape[0], 1, 1, 1,
                                    device=X.device) < self.survival_prob

        return torch.div(X, self.survival_prob) * binary_tensor

    def forward(self, X):

        out = self.conv1(X)     # expand
        out = self.convdw(out)  # depthwise_block

        if self.se_layer:
            out = self.se(out)

        if self.block_type:     # if use residual
            out = self.stochastic_depth(out)
            X = self.padd_conv(X)
            out += X
            return out
        else:
            return out

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class efficientNet (nn.Module):

    def __init__(self, version, num_classes):
        super(efficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.cal_factors(version)

        self.depth_factor = depth_factor

        self.first_layer = conv_block(in_channels=3,
                                      out_channels=int(32*width_factor),
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=(1, 1))

        self.first_layer_n = conv_block(in_channels=int(32*width_factor),
                                      out_channels=int(32 * width_factor),
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=(1, 1))

        self.mbconv1 = InvertedResidual_block(in_channels=int(32*width_factor),
                                              out_channels=int(16*width_factor),
                                              kernel_size=(3, 3),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              block_type1=False,
                                              expand_ratio=1,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv1_n = InvertedResidual_block(in_channels=int(16 * width_factor),
                                              out_channels=int(16 * width_factor),
                                              kernel_size=(3, 3),
                                              stride=(1, 1),
                                              padding=(1, 1),
                                              block_type1=True,
                                              expand_ratio=1,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv2 = InvertedResidual_block(in_channels=int(16 * width_factor),
                                              out_channels=int(24 * width_factor),
                                              kernel_size=(3, 3),
                                              stride=(1, 1),
                                              padding=(1, 1),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv2_n = InvertedResidual_block(in_channels=int(24 * width_factor),
                                              out_channels=int(24 * width_factor),
                                              kernel_size=(3, 3),
                                              stride=(1, 1),
                                              padding=(1, 1),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv3 = InvertedResidual_block(in_channels=int(24 * width_factor),
                                              out_channels=int(40 * width_factor),
                                              kernel_size=(5, 5),
                                              stride=(2, 2),
                                              padding=(2, 2),
                                              block_type1=False,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv3_n = InvertedResidual_block(in_channels=int(40 * width_factor),
                                              out_channels=int(40 * width_factor),
                                              kernel_size=(5, 5),
                                              stride=(1, 1),
                                              padding=(2, 2),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv4 = InvertedResidual_block(in_channels=int(40 * width_factor),
                                              out_channels=int(80 * width_factor),
                                              kernel_size=(3, 3),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              block_type1=False,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv4_n = InvertedResidual_block(in_channels=int(80 * width_factor),
                                              out_channels=int(80 * width_factor),
                                              kernel_size=(3, 3),
                                              stride=(1, 1),
                                              padding=(1, 1),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv5 = InvertedResidual_block(in_channels=int(80 * width_factor),
                                              out_channels=int(112 * width_factor),
                                              kernel_size=(5, 5),
                                              stride=(2, 2),
                                              padding=(2, 2),
                                              block_type1=False,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv5_n = InvertedResidual_block(in_channels=int(112 * width_factor),
                                              out_channels=int(112 * width_factor),
                                              kernel_size=(5, 5),
                                              stride=(1, 1),
                                              padding=(2, 2),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv6 = InvertedResidual_block(in_channels=int(112 * width_factor),
                                              out_channels=int(192 * width_factor),
                                              kernel_size=(5, 5),
                                              stride=(1, 1),
                                              padding=(2, 2),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv6_n = InvertedResidual_block(in_channels=int(192 * width_factor),
                                              out_channels=int(192 * width_factor),
                                              kernel_size=(5, 5),
                                              stride=(1, 1),
                                              padding=(2, 2),
                                              block_type1=True,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.mbconv7 = InvertedResidual_block(in_channels=int(192 * width_factor),
                                              out_channels=int(320 * width_factor),
                                              kernel_size=(3, 3),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              block_type1=False,
                                              expand_ratio=6,
                                              act_type_=False,
                                              use_se=True
                                              )

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7),
                                    stride=(1, 1),
                                    padding=(0, 0))

        self.silu = nn.SiLU()

        self.fc1 = nn.Linear(in_features=int(320 * width_factor),
                             out_features=int(1280 * width_factor))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=int(1280 * width_factor),
                             out_features=num_classes)

    def cal_factors(self, version, alpha=1.2, beta=1.1):

        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi

        return width_factor, depth_factor, drop_rate

    def forward(self, X):

        out = self.first_layer(X) # 3 --> 32
        num_repeat = ceil(1 * self.depth_factor)

        for layer in range(num_repeat - 1):
            out = self.first_layer_n(out)

        out = self.mbconv1(out) # 32 --> 16

        for layer in range(num_repeat-1):
            out = self.mbconv1_n(out)

        out = self.mbconv2(out) # 16 --> 24

        for layer in range(num_repeat-1):
            out = self.mbconv2_n(out)

        out = self.mbconv3(out) # 24 --> 40
        num_repeat = ceil(2 * self.depth_factor)

        for layer in range(num_repeat-1):
            out = self.mbconv3_n(out)

        out = self.mbconv4(out)  # 40 --> 80
        num_repeat = ceil(3 * self.depth_factor)

        for layer in range(num_repeat-1):
            out = self.mbconv4_n(out)

        out = self.mbconv5(out) # 80 --> 112
        for layer in range(num_repeat-1):
            out = self.mbconv5_n(out)

        out = self.mbconv6(out)  # 112 --> 192
        num_repeat = ceil(4 * self.depth_factor)

        for layer in range(num_repeat-1):
            out = self.mbconv6_n(out)

        out = self.mbconv7(out)

        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        out = self.silu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


x= torch.randn([1, 3, 224, 224])
model = efficientNet(version="b0", num_classes=10)
print(model(x).shape)
