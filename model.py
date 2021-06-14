'''
Dogs vs cats binary classification.
Lets try using VGG19(with bn).
VGG:         https://arxiv.org/pdf/1409.1556.pdf
'''
import torch
import torch.nn as nn

col_E = [64, 64, 'max', #vgg19 architecture's depth per layer, table 1
		128, 128, 'max', 
		256, 256, 256, 'max', 
		512, 512, 512, 'max', 
		512, 512, 512, 'max']

col_A =[64, "max",
		128, "max",
		256, 256, "max",
		512, 512, "max",
		512, 512, "max"]

class VGG(nn.Module):
	def __init__(self, in_channels=3, num_classes=1):
		super(VGG, self).__init__()
		self.in_channels = in_channels
		self.convs = self.Convs(col_A)
		''' the fully connected sequence that goes at the end of the model,
		    using BCEWITHLOGITSLOSS for better stability, sigmoid is not needed.'''
		self.fc_layers = nn.Sequential(
			nn.Linear(in_features=8192, out_features=4096), # 7x7 receptive field, 7*7*512 for 244^2
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(in_features=4096, out_features=4096),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(in_features=4096, out_features=num_classes)
			)

	def Convs(self, table):
		''' the bulk of table 1 in the paper. order is selfconv, bn, relu. 
		    if integer, conv layer. if str(max), its a maxpool.'''
		layers = []
		in_channels = self.in_channels

		for depth in table:
			out_channels = depth
			if type(depth) == int: 
				layers += [nn.Conv2d(in_channels=in_channels,
									out_channels=out_channels, 
									kernel_size=3, 
									stride=1, 
									padding=1, 
									bias=False),
							nn.BatchNorm2d(out_channels),
							nn.ReLU()
							]
				in_channels = depth
			elif type(depth) == str:
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				print('hey fix this now.')
				break # a tensor comes from out of nowhere and haunts you when Col_E finishes.
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.convs(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc_layers(x)
		return x

def test():
	x = torch.randn(20, 3, 144, 144) #[batch_size, channels, height, width]
	model = VGG(in_channels=3, num_classes=1) # 2 is identical to 1 but 1 is faster. see https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons
	preds = model(x)
	print(preds.shape)
	print(x.shape)
# test()
