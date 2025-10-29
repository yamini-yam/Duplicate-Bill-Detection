🧾 Intelligent Expense Receipt Processor (TinyBERT + EasyOCR)

This project automates expense extraction and duplicate detection from receipts using OCR and a fine-tuned TinyBERT model.

🚀 Overview

This system extracts key details such as Description, Amount, Date, and Bill Number from scanned or generated bills, then uses a fine-tuned TinyBERT model to detect duplicate entries in an expense dataset.

⚙️ Features

🧠 Text Extraction: Extracts text from bills using EasyOCR.

🔍 Field Extraction: Automatically identifies key fields with regex.

🤖 Duplicate Detection: Fine-tuned TinyBERT identifies duplicates accurately.

📊 Dataset Update: Appends new, unique records to the CSV file.

🏋️ Model Training: Fine-tunes TinyBERT on generated pairwise data for duplicate classification.

🧩 Tech Stack

Python 3.10+

Libraries: easyocr, transformers, pandas, datasets, scikit-learn, torch

Model: huawei-noah/TinyBERT_General_4L_312D

Framework: Hugging Face Transformers

🧾 Workflow

Extract text from the bill using EasyOCR.

Parse and normalize extracted fields.

Check for duplicates using the TinyBERT model.

Update dataset automatically if unique.

🧠 Model Training

Generate text pairs using expense data.

Label pairs (1 = duplicate, 0 = non-duplicate).

Train TinyBERT using:

Learning rate: 2e-5

Batch size: 16

Epochs: 5

Early stopping enabled

📊 Results

OCR Accuracy: High accuracy on printed receipts.

Duplicate Detection: Reliable classification with balanced training data.

Automation: Prevents duplicate expense entries and ensures data integrity.

👩‍💻 Author

Yamini R

📧 Contact: yaminirameshkumar94@gmail.com
