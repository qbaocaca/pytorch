import torch
import torch.nn as nn

class LeNet5 (nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=(5,5),
                               stride=(1,1),
                               padding=(0,0))

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=(5,5),
                               stride=(1, 1),
                               padding=(0, 0))

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=(5,5),
                               stride=(1, 1),
                               padding=(0, 0))

        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 184)
        self.fc2 = nn.Linear(184, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, X):
        input_layer = self.relu(self.conv1(X))
        out = self.pool(input_layer)
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        out = self.relu(self.conv3(out))
        # out = out.reshape(out.shape[0], -1)
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out

# model =LeNet5(10)
# print(model)

