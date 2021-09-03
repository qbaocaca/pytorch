import torch
import torch.nn as nn

class conv_block (nn.Module):
    def __init__ (self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.relu(self.bn(self.conv(X)))

        return out

class inception_block (nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5,
                 out_5x5, pool_1x1):
        super(inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1,
                                  kernel_size=(1,1),
                                  stride=(1,1),
                                  padding=(0,0))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1,1),
                       stride=(1,1), padding=(0,0)),
            conv_block(red_3x3, out_3x3, kernel_size=(3,3),
                       stride=(1,1), padding=(1,1))
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1),
                       stride=(1, 1), padding=(0, 0)),
            conv_block(red_5x5, out_5x5, kernel_size=(3, 3),
                       stride=(1, 1), padding=(1, 1))
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            conv_block(in_channels, pool_1x1, kernel_size=(1, 1),
                       stride=(1, 1), padding=(0, 0))
        )

    def forward(self, X):

        return torch.cat([self.branch1(X),
                          self.branch2(X),
                          self.branch3(X),
                          self.branch4(X)], 1)


class InceptionAux (nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=(5,5), stride=(3,3),
                                 padding=(0,0))

        self.conv = conv_block(in_channels=in_channels,
                               out_channels=128,
                               kernel_size=(1,1),
                               stride=(1,1),
                               padding=(0,0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(in_features=2048,  out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, X):

        out = self.pool(X)
        out = self.conv(out)

        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class GoogleNet (nn.Module):
    def __init__(self, in_channels, num_classes, use_aux=False):
        super(GoogleNet, self).__init__()
        assert use_aux == True or use_aux == False
        self.use_aux = use_aux
        self.conv1 = conv_block(in_channels=in_channels,
                              out_channels=64,
                              kernel_size=(7,7),
                              stride=(2,2),
                              padding=(3,3))

        self.pool = nn.MaxPool2d(kernel_size=(3,3),
                                 stride=(2,2),
                                 padding=(1,1))

        self.conv2 = conv_block(in_channels=64,
                                out_channels=64,
                                kernel_size=(1,1),
                                stride=(1,1),
                                padding=(0,0))

        self.conv3 = conv_block(in_channels=64,
                                out_channels=192,
                                kernel_size=(3,3),
                                stride=(1,1),
                                padding=(1,1))

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)

        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)

        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1),
                                    padding=(0,0))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        if self.use_aux:
            self.aux4a = InceptionAux(in_channels=512, num_classes=num_classes)
            self.aux4d = InceptionAux(in_channels=528, num_classes=num_classes)
        else:
            self.aux4a = self.aux4d = None


    def forward(self, X):

        out = self.conv1(X)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.pool(out)

        out = self.inception4a(out)

        if self.use_aux and self.training:
            aux1 = self.aux4a(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)

        if self.use_aux and self.training:
            aux2 = self.aux4d(out)

        out = self.inception4e(out)
        out = self.pool(out)

        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc(out)

        if self.use_aux and self.training:
            return out, aux1, aux2
        else:
            return out

x= torch.randn([1,3,224,224])

model = GoogleNet(in_channels=3, num_classes=10, use_aux=True)
model.train()
# model.eval()
# z_score, z_score_1, z_score_2 = model(x)
# print(z_score.shape)
# print(z_score_1.shape)
# print(z_score_2.shape)
#
# print(model)
z_score = model(x)
print(z_score.shape)