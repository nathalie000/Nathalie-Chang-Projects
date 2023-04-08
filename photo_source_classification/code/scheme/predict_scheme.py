# There is three steps in our scheme:
# 1. Cropping the image
# 2. feed the images to the model
# 3. make majority vote

import numpy as np 
import pandas as pd 
from PIL import Image
import os
import  sys
import os
import numpy as np
from cv2 import imread, IMREAD_GRAYSCALE # IMREAD_GRAYSCALE allow you to load the image as gray scale image
#from classifier import Classifier
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
# from img_dataset import ImgDataset
from pandas import read_csv
from sklearn.model_selection import train_test_split

class Scheme:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
        self.model.to(device)

    def crop_image(self, image):
        """ Crop the image into several small images
        
        Args:
            image: (np.ndarray), the image that to be cropped

        Return:
            The cropped image
        """
        preprocessed_image = []
        for i in range(0, image.shape[0]-224, 224):
            for j in range(0, image.shape[1]-224, 224):
                preprocessed_image.append(image[i:i+224, j:j+224])
        preprocessed_image = np.array(preprocessed_image)
        return preprocessed_image
    
    def model_forward(self, image):
        """ Feed the images into the model
        
        Args:
            image: list/np.ndarray, a list of small images or a single images
        
        Return:
            Result made by the model
        """
        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        
        if type(image) == np.ndarray and len(image.shape) == 4:
            images_cnt = image.shape[0]
            img_datasets = torch.zeros((image.shape[0], image.shape[3], image.shape[1], image.shape[2]))
            for i in range(images_cnt):
                img_datasets[i, :, :, :] = img_transforms(image[i])
            image = img_datasets
        
        # results = []
        # test_loader = DataLoader(img_transforms(image), batch_size=64, shuffle=False)
        # self.model.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(test_loader):
        #         result = self.model(data[0].cuda())
        #         results.append(result)
        img_cnt = image.shape[0]
        results = []
        
        print("Number of small images: {}".format(img_cnt))
        
        for batch in range(0, int(np.ceil(img_cnt / 64))):
            img_batch = image[64*batch : min(64*batch+64, img_cnt)]
            batch_result = self.model(img_batch.to(self.device)).cpu().detach().numpy().squeeze().tolist()
        results.extend(batch_result)
        
        results = (np.round(np.array(results) + 1) - 1).tolist()
        
        return results

    def majority_vote(self, result):
        """ Make majority vote
        
        Args:
            result: list, a list of results
        
        Return:
            Result of majority vote
        """
        if(result.count(0) > result.count(1)):
            brand = 'apple'
        else:
            brand = 'samsung'

        return brand
    
    def make_prediction(self, image):
        """ Make prediction by those 3 steps
        
        Args:
            image: (np.ndarray), the image that is going to be predicted
        
        Return:
            brand: the brand of phone that the image was taken by
        """
        
        small_images = self.crop_image(image)
        results = self.model_forward(small_images)
        brand = self.majority_vote(results)
        
        return brand