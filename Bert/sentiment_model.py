import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# Initialize wandb project
wandb.init(project="multimedia_final")

model_name = 'bert-base-chinese'

# Load the CSV file
df = pd.read_csv('../labeledData.csv')

# Convert DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['review'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Further split the validation set into validation and test sets
val_test_split = val_dataset.train_test_split(test_size=0.5)
val_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Define a compute_metrics function to use during evaluation
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Grid search over number of epochs
epoch_values = [8]
for num_epochs in epoch_values:
    # Define the model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/epochs_{num_epochs}',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        report_to="wandb",                # Report to wandb
        run_name=f"multimedia-final-epochs-{num_epochs}"  # Custom run name
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results for {num_epochs} epochs: {eval_results}")
    
    # Log the evaluation results to wandb
    wandb.log({"eval_results": eval_results, "num_epochs": num_epochs})
    
    # Test the model
    test_results = trainer.predict(test_dataset)
    print(f"Test results for {num_epochs} epochs: {test_results.metrics}")
    
    # Log the test results to wandb
    wandb.log({"test_results": test_results.metrics, "num_epochs": num_epochs})
    model.save_pretrained(f'/root/mutimedia_final/saved_models/epochs_{num_epochs}')
    
    # End the wandb run
    wandb.finish()
