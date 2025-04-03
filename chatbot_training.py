import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_dataset():
    """
    Prepare the dataset for training with improved formatting
    """
    print("Loading and preparing dataset...")
    
    # Load the processed dataset
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # Get the text column name (assuming it's the first column)
    text_column = df.columns[0]
    
    # Create conversation pairs with better formatting
    conversations = []
    for i in range(0, len(df)-1, 2):
        if i+1 < len(df):
            # Format with clear separation and context
            conversations.append({
                'text': f"<|startoftext|>Human: {df.iloc[i][text_column]}\nAssistant: Let me help you with that. {df.iloc[i+1][text_column]}<|endoftext|>"
            })
    
    # Convert to DataFrame
    df_conversations = pd.DataFrame(conversations)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df_conversations, test_size=0.1, random_state=42)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def tokenize_function(examples, tokenizer):
    """
    Tokenize the texts with improved handling
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,  # Increased max length
        return_tensors='pt',
        return_special_tokens_mask=True
    )

def main():
    # Create output directories
    os.makedirs('./chatbot_model', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model_name = "gpt2"  # Using full GPT-2 instead of DistilGPT-2
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        'pad_token': '<|pad|>',
        'bos_token': '<|startoftext|>',
        'eos_token': '<|endoftext|>',
        'additional_special_tokens': ['Human:', 'Assistant:']
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_dataset()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Set up training arguments with improved parameters
    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        num_train_epochs=5,  # Increased epochs
        per_device_train_batch_size=2,  # Reduced batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Added gradient accumulation
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        learning_rate=5e-5,  # Explicit learning rate
    )
    
    # Initialize data collator with improved settings
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model("./chatbot_model/final")
    tokenizer.save_pretrained("./chatbot_model/final")
    
    print("Training completed!")

if __name__ == "__main__":
    main() 