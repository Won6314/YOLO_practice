import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
	block = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
	)
	return block

def Conv_block(in_ch, out_ch):

	block = nn.Sequential(
		conv(in_ch, int(out_ch / 2), kernel_size=1, padding=0),
		conv(int(out_ch / 2), out_ch, kernel_size=3, padding=1)
	)
	return block


class YOLO(nn.Module):
	def __init__(self):
		super().__init__()
		self.ConvLayer = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool2d(2, 2),

			conv(64, 192, kernel_size=3, padding=1),
			nn.MaxPool2d(2, 2),

			conv(192, 128, kernel_size=3, padding=1),
			conv(128, 256, kernel_size=3, padding=1),
			conv(256, 256, kernel_size=3, padding=1),
			conv(256, 512, kernel_size=3, padding=1),
			nn.InstanceNorm2d(512),
			nn.LeakyReLU(negative_slope=0.1),
			nn.MaxPool2d(2, 2),

			Conv_block(512, 512),
			Conv_block(512, 512),
			Conv_block(512, 512),
			Conv_block(512, 512),
			conv(512, 512, kernel_size=1, padding=0),
			conv(512, 1024, kernel_size=3, padding=1),
			nn.InstanceNorm2d(1024),
			nn.LeakyReLU(negative_slope=0.1),
			nn.MaxPool2d(2, 2),

			Conv_block(1024, 1024),
			Conv_block(1024, 1024),
			conv(1024, 1024, kernel_size=3, padding=1),
			conv(1024, 1024, kernel_size=3, stride=2, padding=1),
			nn.InstanceNorm2d(1024),
			nn.LeakyReLU(negative_slope=0.1),

			conv(1024, 1024, kernel_size=3, padding=1),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.InstanceNorm2d(1024),
			nn.LeakyReLU(negative_slope=0.1)
		)
		self.FCLayer = nn.Sequential(
			nn.Linear(7 * 7 * 1024, 4096),
			nn.Linear(4096, 7 * 7 * 17)
		)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = x
		out = self.ConvLayer(out)
		out = out.reshape(out.size(0), -1)
		out = self.FCLayer(out)
		out = out.reshape([-1, 17, 7, 7])
		out[:, 3:5, :, :] = self.sigmoid(out[:, 3:5, :, :])
		return out
