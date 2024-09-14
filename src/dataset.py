
""" usage:
from dataset import GroupImageDataset
import torch
from matplotlib import pyplot as plt
from random import randint
from torchvision import transforms
import torchvision.transforms.functional as F
from helper_functions import get_transform

DATASET_DIR = '../dataset/'
sample_test_dataset = GroupImageDataset(DATASET_DIR + 'sample_test.csv', DATASET_DIR + 'sample_test_images/', transform=get_transform(target_size=576), is_test=True)
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import requests
from io import BytesIO
import re

class GroupImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test 
        self.image_cache = {}
        
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        
        self.data = self.data.map(lambda x: x.strip() if isinstance(x, str) else x)  
        self.data['entity_name'] = self.data['entity_name'].str.replace(r'\s+', '', regex=True)
        
    def download_image(self, url, img_path):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(img_path)
        return img

    def load_image(self, image_link):
        img_name = os.path.basename(image_link)
        img_path = os.path.join(self.img_dir, img_name)
        
        if img_name in self.image_cache:
            return self.image_cache[img_name]
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = self.download_image(image_link, img_path)
        
        self.image_cache[img_name] = img
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_link = row['image_link']
        entity_name = row['entity_name']
        
        image = self.load_image(image_link)
        
        if self.transform:
            image = self.transform(image)
        
        if not self.is_test:
            entity_value = row['entity_value']
            sample = {
                'image': image,
                'group_id': row['group_id'],
                'entity_name': entity_name,
                'entity_value': entity_value
            }
        else:
            sample = {
                'image': image,
                'group_id': row['group_id'],
                'entity_name': entity_name
            }
        
        return sample
