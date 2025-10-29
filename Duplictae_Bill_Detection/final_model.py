import streamlit as st
import easyocr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import re
from datetime import datetime
from PIL import Image
import numpy as np

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
def extract_text_from_image(image):
    image_np = np.array(image)  # Convert the image to a NumPy array
    result = reader.readtext(image_np, detail=0)
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
    match = re.search(r'Description:\s*(.*?)(?=Amount:|Date:|Bill Number:|$)', text)
    return match.group(1).strip() if match else "Unknown Product"

def extract_amount(text):
    match = re.search(r'Amount:\s*([\d,]+\.\d+)', text)
    return float(match.group(1).replace(',', '')) if match else 0.0

def extract_date(text):
    match = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', text)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")

def extract_bill_number(text):
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
        
        if predictions[0].item() == 1:
            return True

    return False

# Function to add a new record to the dataset
def add_to_dataset(extracted_data, df):
    new_record = {
        "Expense_ID": df['Expense_ID'].max() + 1,
        "Description": extracted_data["Description"],
        "Amount": extracted_data["Amount"],
        "Date": extracted_data["Date"],
        "Bill_Number": extracted_data["Bill_Number"]
    }
    new_df = pd.DataFrame([new_record], columns=df.columns)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv('expense_data_1500.csv', index=False)

# Streamlit UI
st.set_page_config(page_title="Duplicate Bill Detection System", page_icon=":receipt:", layout="centered")
st.image("C:/Users/HP/Desktop/bachground.jpg", use_column_width=True)  # Add a header image

st.title("Duplicate Bill Detection System")
st.write("Upload a bill image to check if it is a duplicate.")

# Image Upload
uploaded_file = st.file_uploader("Choose a bill image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Bill Image', use_column_width=True)

    extracted_data = extract_text_from_image(image)
    st.subheader("Extracted Data")
    st.write(f"**Description:** {extracted_data['Description']}")
    st.write(f"**Amount:** {extracted_data['Amount']}")
    st.write(f"**Date:** {extracted_data['Date']}")
    st.write(f"**Bill Number:** {extracted_data['Bill_Number']}")

    if is_duplicate_bill(extracted_data, df, model, tokenizer):
        st.error("This bill is a **Duplicate**.")
    else:
        st.success("This bill is **Not a Duplicate**.")
        add_to_dataset(extracted_data, df)

