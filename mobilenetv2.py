import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

# model = models.mobilenet_v2()
# summary(model, (3, 224, 224))
# print(model)

class conv_block (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):

        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU6()

    def forward(self, X):

        out = self.relu(self.bn(self.conv(X)))

        return out

class depthwise_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1):
        super(depthwise_block, self).__init__()

        # dw

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=(3, 3),
                               stride=stride,
                               padding=(1, 1),
                               groups=groups
                               )

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()  # should have used ReLU6!

        # pw

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=(0, 0),
                               )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):
        out = self.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))

        # mobilev2 doesnt have relu in pw

        return out

class InvertedResidual_block (nn.Module):
    def __init__(self, in_channels, out_channels, stride, block_type1=False,
                 expand_ratio=6):
        super(InvertedResidual_block, self).__init__()

        assert block_type1 == True or block_type1 == False
        self.block_type = block_type1

        # expansion

        self.conv1 = conv_block(in_channels=in_channels,
                               out_channels=in_channels*expand_ratio,
                                kernel_size=(1,1),
                               stride=(1,1),
                                padding=(0,0))

        # depthwise_block = depthwise + pointwise

        self.convdw = depthwise_block(in_channels=in_channels*expand_ratio,
                                      out_channels=out_channels,
                                      stride=stride,
                                      groups=in_channels*expand_ratio)

        if block_type1:
            self.padd_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))


    def forward(self, X):

        out = self.conv1(X)
        out = self.convdw(out)

        if self.block_type:
            X = self.padd_conv(X)
            out += X
            return out
        else:
            return out


class MobilenetV2 (nn.Module):
    def __init__(self, in_channels, num_classes, width_multiplier=1.0):
        super(MobilenetV2, self).__init__()

        self.conv1 = conv_block(in_channels=in_channels,
                                out_channels=int(32*width_multiplier),
                                kernel_size=(3,3),
                                stride=(2,2),
                                padding=(1,1))

        self.bottleneck1 = InvertedResidual_block(in_channels=int(32*width_multiplier),
                                                  out_channels=int(16*width_multiplier),
                                                  stride=(1,1),
                                                  block_type1=True,
                                                  expand_ratio=1)

        self.bottleneck2 = InvertedResidual_block(in_channels=int(16*width_multiplier),
                                                  out_channels=int(24*width_multiplier),
                                                  stride=(2,2),
                                                  block_type1=False,
                                                  expand_ratio=6)

        self.bottleneck2_2nd = InvertedResidual_block(in_channels=int(24*width_multiplier),
                                                  out_channels=int(24*width_multiplier),
                                                  stride=(1,1),
                                                  block_type1=False,
                                                  expand_ratio=6)

        self.bottleneck3 = InvertedResidual_block(int(24*width_multiplier),
                                                  int(32*width_multiplier),
                                                  (2, 2), False, 6)

        self.bottleneck3_2nd = InvertedResidual_block(int(32*width_multiplier),
                                                      int(32*width_multiplier),
                                                      (1, 1), False, 6)

        self.bottleneck4 = InvertedResidual_block(int(32*width_multiplier),
                                                  int(64*width_multiplier),
                                                  (2, 2), False, 6)

        self.bottleneck4_2nd = InvertedResidual_block(int(64*width_multiplier),
                                                      int(64*width_multiplier),
                                                      (1, 1), False, 6)

        self.bottleneck5 = InvertedResidual_block(int(64*width_multiplier),
                                                  int(96*width_multiplier),
                                                  (1, 1), True, 6)

        self.bottleneck5_2nd = InvertedResidual_block(int(96*width_multiplier),
                                                      int(96*width_multiplier),
                                                      (1, 1), True, 6)

        self.bottleneck6 = InvertedResidual_block(int(96*width_multiplier),
                                                  int(160*width_multiplier),
                                                  (2, 2), False, 6)

        self.bottleneck6_2nd = InvertedResidual_block(int(160*width_multiplier),
                                                      int(160*width_multiplier),
                                                      (1, 1), False, 6)

        self.bottleneck7 = InvertedResidual_block(int(160*width_multiplier),
                                                  int(320*width_multiplier),
                                                  (1, 1), True, 1)

        self.conv2 = conv_block(in_channels=int(320*width_multiplier),
                                out_channels=int(1280*width_multiplier),
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 0))

        self.conv3 = conv_block(in_channels=int(1280*width_multiplier),
                                out_channels=int(1280*width_multiplier),
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 0))

        self.avgpool = nn.AvgPool2d(kernel_size=(7,7),
                                    stride=(1, 1),
                                    padding=(0, 0))

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=int(1280*width_multiplier),
                            out_features=num_classes)


    def forward(self, X):

        out = self.conv1(X)
        # torch.Size([1, 32, 112, 112])
        out = self.bottleneck1(out)
        # torch.Size([1, 16, 112, 112])

        # out = self.bottleneck2(out)
        # out = self.bottleneck2_2nd(out)
        #
        # out = self.bottleneck3(out)
        # out = self.bottleneck3_2nd(out)
        # out = self.bottleneck3_2nd(out)
        #
        # out = self.bottleneck4(out)
        # out = self.bottleneck4_2nd(out)
        # out = self.bottleneck4_2nd(out)
        # out = self.bottleneck4_2nd(out)
        #
        # out = self.bottleneck5(out)
        # out = self.bottleneck5_2nd(out)
        # out = self.bottleneck5_2nd(out)
        #
        # out = self.bottleneck6(out)
        # out = self.bottleneck6_2nd(out)
        # out = self.bottleneck6_2nd(out)
        #
        # out = self.bottleneck7(out)
        # out = self.conv2(out)
        #
        # out = self.conv3(out)
        # out = self.avgpool(out)
        #
        # out = out.reshape(out.shape[0], -1)
        # out = self.dropout(out)
        # out = self.fc(out)

        return out

x= torch.randn([1,3,224,224])

model = MobilenetV2(in_channels=3, num_classes=10, width_multiplier=1.0)
print(model(x).shape)



# testing mismatch
# x=torch.randn([1,24,56,56])
# y=torch.randn([1,24,112,112])
# z=x+y
# print(z)
