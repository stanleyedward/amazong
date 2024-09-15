""" USAGE: 
from helper_functions import download_train
        download_train()
"""
import os
from utils import download_images
import pandas as pd
from torchvision import transforms
import torchvision.transforms.functional as F
import re


DATASET_FOLDER = '../dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))


entity_unit_map = {
    "width": ["centimetre", "foot", "inch", "metre", "millimetre", "yard"],
    "depth": ["centimetre", "foot", "inch", "metre", "millimetre", "yard"],
    "height": ["centimetre", "foot", "inch", "metre", "millimetre", "yard"],
    "item_weight": ["milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"],
    "maximum_weight_recommendation": ["milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"],
    "voltage": ["millivolt", "kilovolt", "volt"],
    "wattage": ["kilowatt", "watt"],
    "item_volume": ["cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", "pint", 
                   "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"]
}

unit_variations_by_entity = {
    # "width": {"cm": "centimetre", "mm": "millimetre", "m": "metre", "feet": "foot", "ft": "foot", "in"  : "inch", "yd": "yard"},
    # "depth": {"cm": "centimetre", "mm": "millimetre", "m": "metre", "feet": "foot", "ft": "foot", "in": "inch", "yd": "yard"},
    # "height": {"cm": "centimetre", "mm": "millimetre", "m": "metre", "feet": "foot", "ft": "foot", "in": "inch", "yd": "yard"},
    "width": {"cm": "centimetre", "mm": "millimetre", "m": "metre", "feet": "foot", "ft": "foot","'":"foot", "in": "inch", "''": "inch" ,"yd": "yard"},
    "depth": {"cm": "centimetre", "mm": "millimetre", "m": "metre", "feet": "foot", "ft": "foot", "'":"foot","in": "inch","''": "inch" , "yd": "yard"},
    "height": {"cm": "centimetre", "mm": "millimetre", "m": "metre", "feet": "foot", "ft": "foot","'":"foot", "in": "inch","''": "inch" , "yd": "yard"},
    "item_weight": {"mg": "milligram", "kg": "kilogram", "g": "gram", "lb": "pound", "oz": "ounce", "t": "ton"},
    "maximum_weight_recommendation": {"mg": "milligram", "kg": "kilogram", "g": "gram", "lb": "pound", "oz": "ounce", "t": "ton"},
    "voltage": {"mv": "millivolt", "kv": "kilovolt", "v": "volt", "voltage": "volt"},
    "wattage": {"kw": "kilowatt", "w": "watt", "wattage": "watt"},              
    "item_volume": {"ml": "millilitre", "l": "litre", "cl": "centilitre", "dl": "decilitre", "oz": "fluid ounce", 
                    "pt": "pint", "qt": "quart", "gal": "gallon", "cup": "cup", "cu ft": "cubic foot", 
                    "cu in": "cubic inch"}
}

def normalize_unit(entity_name, unit):
    unit = unit.lower().rstrip('s')  
    return unit_variations_by_entity.get(entity_name, {}).get(unit, unit)  # Map based on entity

def extract_value_and_unit(entity_name, generated_caption):
    allowed_units = entity_unit_map.get(entity_name, [])
    
    # Combine allowed units and their abbreviations into a single regex pattern
    unit_pattern = r"|".join([re.escape(unit) for unit in allowed_units] + 
                             [re.escape(abbr) for abbr in unit_variations_by_entity.get(entity_name, {}).keys()])
    
    # for a number followed by one of the allowed units or abbrev.
    pattern = fr"([-+]?\d*\.?\d+)\s*({unit_pattern})"
    
    # Search in the caption for the matching value and unit
    match = re.search(pattern, generated_caption, re.IGNORECASE)
    
    if match:
        value = match.group(1)
        unit = normalize_unit(entity_name, match.group(2))
        if unit in allowed_units:  
            return f"{value} {unit}"
    
    return ""

def add_entity_value(df):
    """
test_df = pd.read_csv("../dataset/test_captions.csv")
df_with_entity_value = add_entity_value(test_df)
df_with_entity_value.to_csv('../dataset/cleaned_test_captions.csv')

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df['entity_value'] = df.apply(lambda row: extract_value_and_unit(row['entity_name'], row['generated_caption']), axis=1)
    
    return df


def concatenate_csv_files(directory, file_order):
    """
# Specify the directory containing your CSV files
directory = '../dataset/'

# Specify the order of files you want to concatenate
file_order = [
    'test_captions_30000_45000.csv',
    'test_captions_45000_60000.csv',
    'test_captions_60000_75000.csv',
    'test_captions_75000_90000.csv',
]

# Call the function to concatenate the files
concatenate_csv_files(directory, file_order)   

    Args:
        directory (_type_): _description_
        file_order (_type_): _description_
    """
    df_list = []

    for filename in file_order:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_list.append(df)
        else:
            print(f"Warning: File {filename} not found in the directory.")

    if df_list:
        result = pd.concat(df_list, ignore_index=True)
        
        output_file = 'concatenated_output.csv'
        result.to_csv(output_file, index=False)
        print(f"Concatenation complete. Output saved as '{output_file}'")
    else:
        print("Error: No valid CSV files were found to concatenate.")

class MaintainAspectRatioAndPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        aspect_ratio = w / h
        
        if w < h:
            new_w = int(self.target_size * aspect_ratio)
            new_h = self.target_size
        else:
            new_w = self.target_size
            new_h = int(self.target_size / aspect_ratio)
        
        image = F.resize(image, (new_h, new_w))
        
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        
        padding = (pad_w, pad_h, self.target_size - new_w - pad_w, self.target_size - new_h - pad_h)
        image = F.pad(image, padding, 0, 'constant')  
        
        return image

def get_transform(target_size):
    return transforms.Compose([
        MaintainAspectRatioAndPad(target_size=target_size),  
        transforms.ToTensor()  
    ])

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
