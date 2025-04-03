import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from evaluate import load
import pandas as pd
from tqdm import tqdm

def load_and_prepare_data():
    # Load the customer support dataset
    df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
    
    # Rename columns to match our expected format
    df = df.rename(columns={
        'instruction': 'question',
        'response': 'answer',
        'category': 'context'  # Using category as context
    })
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    return train_df, val_df, test_df

def prepare_dataset(df, tokenizer, max_length=512):
    def preprocess_function(examples):
        # Combine input and target for T5 format
        inputs = [f"question: {q}\ncontext: {c}" for q, c in zip(examples['question'], examples['context'])]
        targets = examples['answer']
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert DataFrame to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def compute_metrics(eval_preds):
    metric = load("sacrebleu")
    logits, labels = eval_preds
    
    # Decode predictions and labels
    predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU score
    result = metric.compute(predictions=predictions, references=labels)
    
    return {"bleu": result["score"]}

def train_model():
    # Initialize tokenizer and model
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)
    test_dataset = prepare_dataset(test_df, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("./chatbot_model_final")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test set results: {test_results}")

if __name__ == "__main__":
    train_model() 