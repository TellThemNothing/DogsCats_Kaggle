
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from model import VGG
from dataset import CatsDogsData
from sklearn.model_selection import KFold

# If model is not learning. Lets try a new model to make sure its not that.
# take vgg, patch on fina linear layer to make binary.
# vgg = models.vgg11_bn(pretrained=True)
# for param in vgg.parameters():
#             param.requires_grad = False
# # print(model)
# model = nn.Sequential(vgg, 
# 					nn.Linear(1000, 100),
# 					nn.Linear(100, 1)).to(device)

#hyperparam
image_dir = r'C:\Users\F\Desktop\Snik\Kaggle\DogsVsCats\train'
test_dir = r'C:\Users\F\Desktop\Snik\Kaggle\DogsVsCats\test1'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGG(in_channels=3, num_classes=1).to(device)
learning_rate = 6e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
folds = 32
num_epochs = 1
batch_size = 64
dataset = CatsDogsData() # returns (image, label)
test_set = CatsDogsData(train=False, image_dir=test_dir, transforms=False)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print('done saving.')
def load_checkpoint(checkpoint):
	print('Loading checkpoint.')
	model.load_state_dict(checkpoint['state_dict']) #load whatever you want from the dictionary.

def train(model, epochs, save_model=False, load_model=True):
	kfold = KFold(n_splits=folds, shuffle=True, random_state=137)
	if load_model:
		load_checkpoint(torch.load("my_checkpoint.pth.tar"))

	#load the data into folds using dataloader, sklearn
	for fold, (train_ids, test_ids) in enumerate(tqdm(kfold.split(dataset))):
		print(f'Welcome to fold {fold}')
		# indeces to define folds
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
		test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

		trainloader = torch.utils.data.DataLoader(
					dataset, batch_size=batch_size, sampler=train_subsampler, pin_memory=True)
		testloader = torch.utils.data.DataLoader(
					dataset, batch_size=batch_size, sampler=test_subsampler, pin_memory=True)

		
		#train on test set
		# image, label = next(iter(trainloader)) # if overfitting single batch
		for epoch in range(num_epochs):
			# losses = []
			if save_model:
				if epoch % 32 == 0:
					if save_model==True:
						checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
						save_checkpoint(checkpoint)

			for batch_idx, (image, label) in enumerate(tqdm(trainloader, position=0, leave=True)): #keywords are for tqdm
				image = image.to(device=device)
				label = label.unsqueeze(-1).to(device=device) # had unsqueeze -1

				# forward
				scores = model(image) #added squeeze .squeeze(-1)
				loss = criterion(scores, label)
				# losses.append(loss.item())

				# backward, optimizer
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				# print(loss)
			print(f'loss for epoch {epoch+1} in fold {fold+1} is: {loss}\n')

		# define accuracy by testloader, this includes augmentations.
		for batch_idx, (image, label) in enumerate(testloader):
			num_correct = 0
			num_samples = 0
			with torch.no_grad():
				for x, y in testloader:
					x = x.to(device=device)
					y = y.to(device=device)
					
					scores = model(x)
					scores = 1/(1 + torch.exp(-1*scores)) #used bcelosswlogits, need to add sigmoid if not using it
					scores = torch.round(scores)
					predictions = scores.max(1)[0]
					num_correct += (predictions == y).sum()
					num_samples += predictions.size(0)

		print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 


def Inference(model, data):
	pred_list = []
	load_checkpoint(torch.load("my_checkpoint.pth.tar"))
	model.eval()
	load_data = torch.utils.data.DataLoader(data)
	with torch.no_grad():
		for image in load_data:
			image = image.to(device=device)
			scores = model(image)
			scores = 1/(1 + torch.exp(-1*scores)) #used bcelosswlogits, need to add sigmoid if not using it
			scores = torch.round(scores)
			predictions = scores.max(1)[0]
			pred_list.append(int(predictions.item())) #tuple with 0 to fit form in samplesubmission.csv
	pred_list = np.array(pred_list).astype(int)
	np.savetxt("test_results.csv", pred_list, delimiter=",")
	print('File saved. \nLength', len(pred_list))

	  



# train(model, num_epochs)
Inference(model=model, data=test_set)

