import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

# Load the dataset
df = pd.read_csv('C:/Users/HP/Desktop/SAP_PROJECT/expense_data_1500.csv')

# Generate pairs
def generate_pairs(df):
    pairs = []
    labels = []
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            is_duplicate = 1 if df.iloc[i]['Description'] == df.iloc[j]['Description'] and df.iloc[i]['Amount'] == df.iloc[j]['Amount'] and df.iloc[i]['Date'] == df.iloc[j]['Date'] else 0
            pairs.append([df.iloc[i]['Description'], df.iloc[j]['Description']])
            labels.append(is_duplicate)
    
    return pd.DataFrame({'text_a': [pair[0] for pair in pairs], 'text_b': [pair[1] for pair in pairs], 'label': labels})

pairs_df = generate_pairs(df)

# Balance the dataset
duplicates = pairs_df[pairs_df['label'] == 1]
non_duplicates = pairs_df[pairs_df['label'] == 0].sample(n=len(duplicates), random_state=42)  # Undersample non-duplicates
balanced_pairs_df = pd.concat([duplicates, non_duplicates])

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(balanced_pairs_df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text_a'], examples['text_b'], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Set training arguments with learning rate tuning, more epochs, and early stopping
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Increase number of epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,  # Lower learning rate
    logging_dir='./logs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model='eval_loss',  # Use evaluation loss to determine the best model
)

# Initialize Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('C:/Users/HP/Desktop/SAP_PROJECT/tinybert_model/')
