import torch
import torch.nn as nn

class residual_block (nn.Module):

	def __init__(self, in_channels, out_channels, stride, use_padd=False):
		super(residual_block, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=in_channels,
								out_channels=out_channels,
								kernel_size=(3,3),
								stride=stride,
								padding=(1,1))

		self.conv2 = nn.Conv2d(in_channels=out_channels,
							   out_channels=out_channels,
							   kernel_size=(3,3),
							   stride=(1,1),
							   padding=(1,1))

		if use_padd:
			self.padd_conv = nn.Conv2d(in_channels=in_channels,
									   out_channels=out_channels,
									   kernel_size=(1, 1),
									   stride=(2, 2),
									   padding=(0, 0))
		else:
			self.padd_conv = None

		self.bn = nn.BatchNorm2d(num_features=out_channels)
		self.relu = nn.ReLU()


	def forward(self, X):

		out = self.relu(self.bn(self.conv1(X)))
		out = self.relu(self.bn(self.conv2(out)))

		out = self.relu(self.bn(self.conv2(out)))
		out = self.relu(self.bn(self.conv2(out)))

		if self.padd_conv:
			X = self.padd_conv(X)

		out += X
		out = self.relu(out)

		return out

class Resnet18 (nn.Module):
	def __init__(self, num_classes):
		super(Resnet18, self).__init__()

		self.conv = nn.Conv2d(in_channels=3,
							   out_channels=64,
							   kernel_size=(7,7),
							   stride=(2,2),
							   padding=(3,3))

		self.bn = nn.BatchNorm2d(num_features=64)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(kernel_size=(3,3),
								 stride=(2,2),
								 padding=(1,1))

		self.layer1 = residual_block(in_channels=64,
									 out_channels=64,
									 stride=(1,1),
									 use_padd=False)

		self.layer2 = residual_block(in_channels=64,
									 out_channels=128,
									 stride=(2,2),
									 use_padd=True)

		self.layer3 = residual_block(in_channels=128,
									 out_channels=256,
									 stride=(2,2),
									 use_padd=True)

		self.layer4 = residual_block(in_channels=256,
									 out_channels=512,
									 stride=(2,2),
									 use_padd=True)

		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
		self.fc = nn.Linear(in_features=512, out_features=num_classes)

	def forward(self, X):

		x_in = self.relu(self.bn(self.conv(X)))
		out = self.pool(x_in)

		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)

		out = self.avgpool(out)
		out = out.reshape(out.shape[0], -1)
		out = self.fc(out)

		return out

print(Resnet18(10))




