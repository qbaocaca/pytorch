import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models

# model = models.mobilenet_v3_small()
# print(model)

# summary(model, (3, 224, 224))

# conv_block in mobilenet_v3 uses Hardswish rather than ReLU6

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
        self.hs = nn.Hardswish()

    def forward(self, X):
        out = self.hs(self.bn(self.conv(X)))

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
                 expand_size=72, act_type_=False, use_se=False):
        super(InvertedResidual_block, self).__init__()

        assert block_type1 == True or block_type1 == False
        assert use_se == True or use_se == False
        self.block_type = block_type1
        self.se_layer = use_se

        # expansion

        self.conv1 = conv_block(in_channels=in_channels,
                               out_channels=expand_size,
                                kernel_size=(1,1),
                               stride=(1,1),
                                padding=(0,0))

        # depthwise_block = depthwise + pointwise

        self.convdw = depthwise_block(in_channels=expand_size,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      groups=expand_size,
                                      act_type1=act_type_)

        if block_type1:
            self.padd_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))

        if use_se:
            self.se = SElayer(in_channels=out_channels, reduction=16)

    def forward(self, X):

        out = self.conv1(X)
        out = self.convdw(out)

        if self.se_layer:
            out = self.se(out)

        if self.block_type:
            X = self.padd_conv(X)
            out += X
            return out
        else:
            return out

class mobilenet_v3_small (nn.Module):
    def __init__(self, in_channels, num_classes):
        super(mobilenet_v3_small, self).__init__()

        self.first_layer = conv_block(in_channels=3,
                                      out_channels=16,
                                      kernel_size=(3,3),
                                      stride=(2,2),
                                      padding=(1,1))

        self.bneck1 = InvertedResidual_block(in_channels=16,
                                             out_channels=16,
                                             kernel_size=(3, 3),
                                             stride=(2,2),
                                             padding=(1,1),
                                             block_type1=False,
                                             expand_size=16,
                                             act_type_=True,
                                             use_se=True)

        self.bneck2 = InvertedResidual_block(16, 24, (3,3), (2,2), (1,1),
                                             False, 72, True, False)

        self.bneck3 = InvertedResidual_block(24, 24, (3, 3), (1, 1), (1, 1),
                                             True, 88, True, False)

        self.bneck4 = InvertedResidual_block(24, 40, (5, 5), (2, 2), (2, 2),
                                             False, 96, False, True)

        self.bneck5 = InvertedResidual_block(40, 40, (5, 5), (1, 1), (2, 2),
                                             True, 240, False, True)

        self.bneck6 = InvertedResidual_block(40, 48, (5, 5), (1, 1), (2, 2),
                                             True, 120, False, True)

        self.bneck7 = InvertedResidual_block(48, 48, (5, 5), (1, 1), (2, 2),
                                             True, 144, False, True)

        self.bneck8 = InvertedResidual_block(48, 96, (5, 5), (2, 2), (2, 2),
                                             False, 288, False, True)

        self.bneck9 = InvertedResidual_block(96, 96, (5, 5), (1, 1), (2, 2),
                                             True, 576, False, True)

        self.conv1 = conv_block(in_channels=96,
                                out_channels=576,
                                kernel_size=(1,1),
                                stride=(1, 1),
                                padding=(0, 0))

        self.se = SElayer(in_channels=576, reduction=16)

        self.avgpool = nn.AvgPool2d(kernel_size=(7,7),
                                      stride=(1, 1),
                                      padding=(0, 0))
        self.hardswish = nn.Hardswish()

        self.fc1 = nn.Linear(in_features=576, out_features=1024)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, X):
        out = self.first_layer(X)
        out = self.bneck1(out)
        out = self.bneck2(out)
        out = self.bneck3(out)
        out = self.bneck4(out)
        out = self.bneck5(out)
        out = self.bneck5(out)
        out = self.bneck6(out)
        out = self.bneck7(out)
        out = self.bneck8(out)
        out = self.bneck9(out)
        out = self.bneck9(out)
        out = self.conv1(out)
        out = self.se(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        out = self.hardswish(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out



# x=  torch.randn([1,32,224,224])
# model = SElayer(in_channels=32, reduction=16)
# print(model(x).shape)

# x=torch.randn([1,3,224,224])
model = mobilenet_v3_small(in_channels=3, num_classes=10)
# print(model(x).shape)

print(model)


