import torch
import torch.nn as nn
import torchvision.models as models

class VGG_16 (nn.Module):

    def __init__(self, num_classes):
        super(VGG_16, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1))

        self.conv1_2nd = nn.Conv2d(in_channels=64,
                              out_channels=64,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=(2,2),
                                 stride=(2,2))

        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1,1))

        self.conv2_2nd = nn.Conv2d(in_channels=128,
                              out_channels=128,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1)
                               )

        self.conv3_2nd = nn.Conv2d(in_channels=256,
                              out_channels=256,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1)
                              )

        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1)
                               )

        self.conv4_2nd = nn.Conv2d(in_channels=512,
                              out_channels=512,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1)
                              )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=512*7*7,
                             out_features=4096)

        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # self.softmax = nn.Softmax()

    def forward(self, X):

        input_layer = self.relu(self.conv1(X))
        out = self.relu(self.conv1_2nd(input_layer))
        out = self.pool(out)

        out = self.relu(self.conv2(out))
        out = self.relu(self.conv2_2nd(out))
        out = self.pool(out)

        out = self.relu(self.conv3(out))
        out = self.relu(self.conv3_2nd(out))
        out = self.relu(self.conv3_2nd(out))
        out = self.pool(out)

        out = self.relu(self.conv4(out))
        out = self.relu(self.conv4_2nd(out))
        out = self.relu(self.conv4_2nd(out))
        out = self.pool(out)

        out = self.relu(self.conv4_2nd(out))
        out = self.relu(self.conv4_2nd(out))
        out = self.relu(self.conv4_2nd(out))
        out = self.pool(out)

        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        # out = self.softmax(self.fc3(out))
        out = self.fc3(out)

        return out

print(VGG_16(num_classes=10))

# model = models.vgg16()
# print(model)

# NOTE: if use CrossEntropyLoss no Softmax, input_image_size = 224