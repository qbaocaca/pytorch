import torch
import torch.nn as nn
import torchvision.models as models
import torchsummary as summary


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
        out = self.relu(self.bn2(self.conv2(out)))

        return out


class depthwise_block_sub(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1):
        super(depthwise_block_sub, self).__init__()

        # dw

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=(3, 3),
                               stride=stride,
                               padding=(1, 1),
                               groups=groups
                               )

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()

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
        # out = self.relu(self.bn2(self.conv2(out)))

        return out


class MobilenetV1(nn.Module):
    def __init__(self, in_channels, num_classes, width_multiplier=1.0):
        super(MobilenetV1, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=int(32 * width_multiplier),
                              kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=(1, 1),
                              )
        self.bn = nn.BatchNorm2d(num_features=int(32 * width_multiplier))
        self.relu = nn.ReLU()

        self.dconv1 = depthwise_block(in_channels=int(32 * width_multiplier),
                                      out_channels=int(64 * width_multiplier),
                                      stride=(1, 1),
                                      groups=int(32 * width_multiplier))

        self.dconv2 = depthwise_block(in_channels=int(64 * width_multiplier),
                                      out_channels=int(128 * width_multiplier),
                                      stride=(2, 2),
                                      groups=int(64 * width_multiplier))

        self.dconv3 = depthwise_block(in_channels=int(128 * width_multiplier),
                                      out_channels=int(128 * width_multiplier),
                                      stride=(1, 1),
                                      groups=int(128 * width_multiplier))

        self.dconv4 = depthwise_block(in_channels=int(128 * width_multiplier),
                                      out_channels=int(256 * width_multiplier),
                                      stride=(2, 2),
                                      groups=int(128 * width_multiplier))

        self.dconv5 = depthwise_block(in_channels=int(256 * width_multiplier),
                                      out_channels=int(256 * width_multiplier),
                                      stride=(1, 1),
                                      groups=int(256 * width_multiplier))

        self.dconv6 = depthwise_block(in_channels=int(256 * width_multiplier),
                                      out_channels=int(512 * width_multiplier),
                                      stride=(2, 2),
                                      groups=int(256 * width_multiplier))

        self.dconv7 = depthwise_block(in_channels=int(512 * width_multiplier),  # 5 times
                                      out_channels=int(512 * width_multiplier),
                                      stride=(1, 1),
                                      groups=int(512 * width_multiplier))

        self.dconv8 = depthwise_block(in_channels=int(512 * width_multiplier),
                                      out_channels=int(1024 * width_multiplier),
                                      stride=(2, 2),
                                      groups=int(512 * width_multiplier))

        self.dconv9 = depthwise_block(in_channels=int(1024 * width_multiplier),
                                      out_channels=int(1024 * width_multiplier),
                                      stride=(1, 1),
                                      groups=int(1024 * width_multiplier))

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7),
                                    stride=(1, 1),
                                    padding=(0, 0))

        self.fc = nn.Linear(in_features=int(1024 * width_multiplier),
                            out_features=num_classes)

    def forward(self, X):
        # shape of X is torch.Size([1, 3, 224, 224])
        out = self.relu(self.bn(self.conv(X)))
        # torch.Size([1, 32, 112, 112]), stride=2
        out = self.dconv1(out)
        # torch.Size([1, 32, 112, 112]), stride=1
        # torch.Size([1, 64, 112, 112]), stride=1
        out = self.dconv2(out)
        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 56, 56]) 1st
        out = self.dconv3(out)
        # torch.Size([1, 128, 56, 56]) 2nd
        # torch.Size([1, 128, 56, 56]) 3rd
        out = self.dconv4(out)
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 28, 28])
        out = self.dconv5(out)
        out = self.dconv6(out)

        out = self.dconv7(out)
        out = self.dconv7(out)
        out = self.dconv7(out)
        out = self.dconv7(out)
        out = self.dconv7(out)

        out = self.dconv8(out)
        out = self.dconv9(out)

        out = self.avgpool(out)
        # out = out.view(out.size(0), 1024 * 1 * 1)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


x = torch.randn([1, 3, 224, 224])

model = MobilenetV1(in_channels=3, num_classes=10, width_multiplier=0.5)
print(model(x).shape)
