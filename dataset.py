
import os
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import cv2

train_dir = r'C:\Users\F\Desktop\Snik\Kaggle\DogsVsCats\train'
test_dir = r'C:\Users\F\Desktop\Snik\Kaggle\DogsVsCats\test1'
class CatsDogsData(Dataset):
	'''kaggle catsdogs dataset.'''
	def __init__(self, image_dir=train_dir, transforms=True, train=True):
		self.image_dir = image_dir
		self.images = os.listdir(image_dir)
		self.transforms = transforms
		self.train = train

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		image = cv2.imread(img_path, 1) # 1 for color
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (128, 128)) #default 244 for vgg
		image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) #(src, dst, min, max, norm)
		
		if self.train==True:
			if self.transforms==True:
				transform = A.Compose([ #copied and pasted from tutorial.
					A.CLAHE(),
					A.RandomRotate90(),
					A.Transpose(),
					A.Blur(blur_limit=3),
					A.OpticalDistortion(),
					A.GridDistortion(),
					A.HueSaturationValue(),
				])

				image = transform(image=image)['image']
				image = np.array(image, dtype='float32')
				image = np.moveaxis(image, -1, 0) #move channels first
			

			if self.images[index][0:3] == str('cat'):
				label = 0.
				# label = 'cat'
			elif self.images[index][0:3] == str('dog'):
				label = 1.
				# label = 'dog'
			else:
				raise Exception("label is neither cat nor dog.")

			return (image, label)
		else: #train is false
			image = np.array(image, dtype='float32')
			image = np.moveaxis(image, -1, 0) #move channels first
			return image



# dataset = CatsDogsData(image_dir=image_dir)
# first_data = dataset[0]
# for image, label in dataset:
# 	image = np.array(image, dtype='uint8')
# 	cv2.imshow(label, image)
# 	cv2.waitKey(0) 
# 	cv2.destroyAllWindows()