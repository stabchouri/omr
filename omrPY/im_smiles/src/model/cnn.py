import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

	'''
	Builds a convolutional neural net with the following attributes:
	Args: batch size
	Inputs: tensor of images in shape [batch x channel, height, width]

    (conv_1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (relu_1): ReLU()
    (maxpool_1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    --Size: batch x 64 x h/2 x w/2

	(conv_2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (relu_2): ReLU()
    (maxpool_2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    --Size: batch x 128, h/2/2, w/2/2

    (conv_3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (batch_norm_3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   	(relu_3): ReLU()
	--Size: batch x 256 x h/2/2, w/2/2

    (conv_4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (relu_4): ReLU()
    (maxpool_4): MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0, dilation=1, ceil_mode=False)
	--Size: batch x 256 x h/2/2, w/2/2/2

    (conv_5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
    (batch_norm_5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu_5): ReLU()
    (maxpool_5): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
    --Size: batch x 512 x h/2/2/2, w/2/2/2

    (conv_6): Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))
    (batch_norm_6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu_6): ReLU()
    --Size: batch x 512 x h/2/2/2, w/2/2/2

	'''

	def __init__(self, num_channels=1):
		super(CNNModel, self).__init__()

		self.num_channels = num_channels

		self.conv = nn.Sequential()
		self.conv.add_module('conv_1', nn.Conv2d(self.num_channels, 64, (3, 3), (1, 1), 1))
		self.conv.add_module('relu_1', nn.ReLU())
		self.conv.add_module('maxpool_1', nn.MaxPool2d((2, 2), (2, 2))) #(kernel), (stride)
		
		self.conv.add_module('conv_2', nn.Conv2d(64, 128, (3, 3), (1, 1), 1))
		self.conv.add_module('relu_2', nn.ReLU())
		self.conv.add_module('maxpool_2', nn.MaxPool2d((2, 2), (2, 2)))

		
		self.conv.add_module('conv_3', nn.Conv2d(128, 256, (3, 3), (1, 1), 1))
		self.conv.add_module('batch_norm_3', nn.BatchNorm2d(256))
		self.conv.add_module('relu_3', nn.ReLU())
		
		self.conv.add_module('conv_4', nn.Conv2d(256, 256, (3, 3), (1, 1), 1))
		self.conv.add_module('relu_4', nn.ReLU())
		self.conv.add_module('maxpool_4', nn.MaxPool2d((1,2), (1,2)))
		
		self.conv.add_module('conv_5', nn.Conv2d(256, 512, (3, 3), (1, 1), 1))
		self.conv.add_module('batch_norm_5', nn.BatchNorm2d(512))
		self.conv.add_module('relu_5', nn.ReLU())
		self.conv.add_module('maxpool_5', nn.MaxPool2d((2,1), (2,1)))
		
		self.conv.add_module('conv_6', nn.Conv2d(512, 512, (2, 2), (1, 1), 1))
		self.conv.add_module('batch_norm_6', nn.BatchNorm2d(512))
		self.conv.add_module('relu_6', nn.ReLU())
		
	def forward(self, im):
		#uncomment the regulizer if not using torchvision to load tensor -- tv.totensor will convert PIL to tensor in range(0,1)
		#reg = torch.add(im, -128.0)
		#reg = torch.mul(reg, (1.0/128))

		#shape here is batch x 512 x h/2/2/2, w/2/2/2
		conv = self.conv.forward(im)
		#transpose to batch x w x h x 512
		conv_out = conv.transpose(1,2)
		conv_out = conv_out.transpose(2,3)
		return conv_out #shape batch x h x w x D(convs)

		

if __name__ == "__main__":
	imgs = [[100,360], [160,400], [200,500],[800,800]]
	for i in imgs:
		img = torch.randn(1, 1, i[0], i[1])
		new = CNNModel()
		convs = new.forward(img)
		print((i, convs.shape[1:3]))


