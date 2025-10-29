import easyocr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import re
from datetime import datetime

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the fine-tuned TinyBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = BertForSequenceClassification.from_pretrained('tinybert_model')  # Path to your fine-tuned model

# Load your dataset
df = pd.read_csv('expense_data_1500.csv')

# Normalize text function
def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# OCR function using EasyOCR
def extract_text_from_image(image_path):
    result = reader.readtext(image_path, detail=0)
    text = ' '.join(result)
    
    # Extract Product, Amount, Date, and Bill Number
    extracted_data = {
        "Description": extract_product(text),
        "Amount": extract_amount(text),
        "Date": extract_date(text),
        "Bill_Number": extract_bill_number(text)
    }
    
    return extracted_data

def extract_product(text):
    # Extract product description
    match = re.search(r'Description:\s*(.*?)(?=Amount:|Date:|Bill Number:|$)', text)
    return match.group(1).strip() if match else "Unknown Product"

def extract_amount(text):
    # Extract amount using regex
    match = re.search(r'Amount:\s*([\d,]+\.\d+)', text)
    return float(match.group(1).replace(',', '')) if match else 0.0

def extract_date(text):
    # Extract date in 'YYYY-MM-DD' format
    match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', text)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")

def extract_bill_number(text):
    # Extract bill number
    match = re.search(r'Bill Number:\s*(\w+)', text)
    return match.group(1) if match else "Unknown Bill Number"

# Function to predict if the bill is a duplicate
def is_duplicate_bill(extracted_data, df, model, tokenizer):
    text = normalize_text(f"{extracted_data['Description']} {extracted_data['Amount']}")
    
    for _, row in df.iterrows():
        description = normalize_text(row['Description'])
        amount = row['Amount']
        input_text = normalize_text(f"{description} {amount}")
        
        encoding = tokenizer(input_text, text, return_tensors='pt', truncation=True, padding=True)
        
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(dim=-1)
        
        if predictions[0].item() == 1:  # Assuming 1 indicates duplicate
            return True

    return False

# Function to add a new record to the dataset
def add_to_dataset(extracted_data, df):
    # Create a new record
    new_record = {
        "Expense_ID": df['Expense_ID'].max() + 1,  # Generate a new ID
        "Description": extracted_data["Description"],       # Extracted Product
        "Amount": extracted_data["Amount"],         # Extracted Amount
        "Date": extracted_data["Date"],             # Extracted Date
        "Bill_Number": extracted_data["Bill_Number"] # Use extracted Bill Number
    }
    
    # Convert new record to DataFrame and append to the existing DataFrame
    new_df = pd.DataFrame([new_record], columns=df.columns)  # Ensure columns match
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save the updated DataFrame to a CSV file
    df.to_csv('expense_data_1500.csv', index=False)

# Example usage
image_path = 'generated_bill.png'
extracted_data = extract_text_from_image(image_path)

if is_duplicate_bill(extracted_data, df, model, tokenizer):
    print("Duplicate")
else:
    print("Not Duplicate")
    add_to_dataset(extracted_data, df)
