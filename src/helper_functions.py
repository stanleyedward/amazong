""" USAGE: 
from helper_functions import download_train
        download_train()
"""
import os
from utils import download_images
import pandas as pd

DATASET_FOLDER = '../dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))

def download_train():
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    download_images(train['image_link'], DATASET_FOLDER + 'train_images')
    print(f"downloaded {len(os.listdir(DATASET_FOLDER + 'train_images'))} images")

def download_test():
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    download_images(test['image_link'], DATASET_FOLDER + 'test_images')
    print(f"downloaded {len(os.listdir(DATASET_FOLDER + 'test_images'))} images")

def download_test_sample():
    sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    download_images(sample_test['image_link'], DATASET_FOLDER + 'sample_test_images', allow_multiprocessing=True)
    print(f"downloaded {len(os.listdir(DATASET_FOLDER + 'sample_test_images'))} images")

def download_train_sample():
    sample_train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    download_images(sample_train['image_link'][:176], DATASET_FOLDER  + 'sample_train_images', allow_multiprocessing=True)
    print(f"downloaded {len(os.listdir(DATASET_FOLDER + 'sample_train_images'))} images")
