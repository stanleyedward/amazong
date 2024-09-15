# !wget https://huggingface.co/spaces/OpenGVLab/InternVL/raw/main/requirements.txt
# pip install -r requirements.txt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

df = pd.read_csv('test.csv')

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Load model and tokenizer
path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# Function to generate caption for a single image
def generate_caption_for_image(model, tokenizer, image, entity_name):
    transform = build_transform(input_size=448)
    image_tensor = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    question = f'''<image>
Extract the numerical value and unit from the text for the entity "{entity_name}".

Allowed units for "{entity_name}" are:
{{
    'width': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'depth': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'height': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'item_weight': ['gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'],
    'maximum_weight_recommendation': ['gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'],
    'voltage': ['kilovolt', 'millivolt', 'volt'],
    'wattage': ['kilowatt', 'watt'],
    'item_volume': ['centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart']
}}[entity_name]

Return the extracted value and unit in the format "x unit" (e.g. "2.0 gram", "12.5 centimetre", "2.56 ounce"). If no explicit mention of "{entity_name}" is found or no value is found, return an empty string (""). '''
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    caption = model.chat(tokenizer, image_tensor, question, generation_config)

    return caption

def generate_captions_for_dataset(df):
    df['generated_caption'] = ''  # Initialize new column for captions

    # Slice the DataFrame from the 5000th to the 10000th row (adjusting for 0-based indexing)
    df_subset = df.iloc[0:131287]

    for index, row in df_subset.iterrows():
        try:
            image_url = row['image_link']
            entity_name = row['entity_name']

            image = load_image_from_url(image_url)

            caption = generate_caption_for_image(model, tokenizer, image, entity_name)
            df_subset.at[index, 'generated_caption'] = caption

            print(f"Processed image at index {index} with group_id {row['group_id']}")

        except Exception as e:
            print(f"Error processing image at index {index}: {e}")

    return df_subset

# Generate captions for the dataset
df_with_captions = generate_captions_for_dataset(df)

# Save the updated DataFrame to a new CSV file
df_with_captions.to_csv('test_captions_5000.csv', index=False)

print("Captions generated and saved successfully for the specified rows!")
