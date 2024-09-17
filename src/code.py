import pandas as pd
from unsloth import FastLanguageModel
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer
max_seq_length = 2048  # Adjust as needed
dtype = torch.float16  # Use float16 for T4 GPU
load_in_4bit = True    # Use 4-bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-70B-Instruct",
    max_seq_length=512,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map='auto'
)

FastLanguageModel.for_inference(model)  # Enable faster inference

# Load the dataframe
file_id = '1LhYl3Y1pBGY24y4YpwOZkSCBLZZlrZIR'
url = f'https://drive.google.com/uc?id={file_id}'
df = pd.read_csv(url,header=None)

# Define a function to extract numerical value and unit
def extract_value_and_unit(row):
    entity_name = row['entity_name']
    generated_caption = row['generated_caption']
    question = f'''<text>
Extract the numerical value and unit from the text: "{generated_caption}" for the entity "{entity_name}". 

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

Return only the extracted value and unit in the format "x unit" (e.g. "2 gram", "12.5 centimetre", "2.56 ounce"). If no explicit mention of "{entity_name}" is found or no value is found, return an empty string (""). Do not include any additional text or explanations. Provide a single output with no text other than the value and unit or an empty string. STRICTLY NO OTHER TEXT IS ALLOWED.'''

    # Tokenize the input and move to device
    inputs = tokenizer(question, return_tensors="pt").to(device)

    # Generate the response
    outputs = model.generate(**inputs, max_length=100)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean the response
    if ':' in response or 'Since' in response or 'However' in response:
        response = ''

    return response.strip()

# Apply the function to each row in the dataframe
df['extracted_value_and_unit'] = df.apply(extract_value_and_unit, axis=1)

# Save the results to a new CSV file
df[['extracted_value_and_unit']].reset_index().rename(
    columns={'index': 'index', 'extracted_value_and_unit': 'prediction'}
).to_csv('predictions.csv', index=False)

print("Results saved to 'predictions.csv'")
