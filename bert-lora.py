import os
import json
import torch
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from peft import PeftModel, PeftConfig
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

os.environ["WANDB_DISABLED"] = "true"

# 1. Load your dataset
df = pd.read_feather("data/df_emails_cleaned.feather")
df = df[["ProcessedTextBody", "Complaint"]].dropna()

# Rename column early
df = df.rename(columns={"Complaint": "label"})

# 2. Split the dataset
use_sequential_split = True  # Change to False to use stratified random split

# Dataset splitting
if use_sequential_split:
    print("👉 Using sequential split")
    n_total = len(df)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
else:
    print("👉 Using stratified random split")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Optional: check splits
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# df["label"] = df["label"].astype("int64")
# train_df["label"] = train_df["label"].astype("int64")
# val_df["label"] = val_df["label"].astype("int64")
# test_df["label"] = test_df["label"].astype("int64")

# Optional: Sample a smaller subset for faster training
fraction = 1.00  # Adjust this fraction as needed
train_df = train_df.sample(frac=fraction, random_state=42)
val_df = val_df.sample(frac=fraction, random_state=42)
test_df = test_df.sample(frac=fraction, random_state=42)

# 3. Convert to Hugging Face Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
})

# 4. Tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_function(example):
    return tokenizer(example["ProcessedTextBody"], truncation=True, max_length=256)


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. Load the model and LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.config.problem_type = "single_label_classification"
model = get_peft_model(model, peft_config)

# Optional: Show trainable parameters
model.print_trainable_parameters()

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none",
)


# 7. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# 8. Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_datasets["train"].column_names)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 10. Train the model
try:
    trainer.train()
except Exception as e:
    print("❗️ An error occurred:")
    traceback.print_exc()

# Save model & tokenizer
model.save_pretrained("models/bert-email-complaint-lora")
tokenizer.save_pretrained("models/bert-email-complaint-lora")

# 11. Evaluate on test set
trainer.evaluate(tokenized_datasets["test"])

# 12. Classification report
predictions = trainer.predict(tokenized_datasets["test"])
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

print(classification_report(y_true, y_pred, target_names=["No Complaint", "Complaint"]))

# 13. Add predictions to original test set
test_df = test_df.reset_index(drop=True)
test_df["TrueLabel"] = y_true
test_df["PredictedLabel"] = y_pred
test_df["Correct"] = test_df["TrueLabel"] == test_df["PredictedLabel"]

# Map labels to bool
test_df["TrueLabel"] = test_df["TrueLabel"].map({0: False, 1: True})
test_df["PredictedLabel"] = test_df["PredictedLabel"].map({0: False, 1: True})

# Save predictions
test_df[["ProcessedTextBody", "TrueLabel", "PredictedLabel", "Correct"]].to_csv("data/bert_test_predictions.csv",
                                                                                index=False)

# Save incorrect predictions
incorrect_predictions = test_df[~test_df["Correct"]]
incorrect_predictions.to_csv("data/bert_incorrect_predictions.csv", index=False)

# 14. Reload fine-tuned model
config = PeftConfig.from_pretrained("models/bert-email-complaint-lora")
base_model = BertForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=2)
model = PeftModel.from_pretrained(base_model, "models/bert-email-complaint-lora")

# === Save training logs ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path(f"data/training_logs_{timestamp}.json")

with open(log_file, "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"✅ Training logs saved to: {log_file}")

# === Visualize training curve ===
training_logs = trainer.state.log_history

# Extract epochs & values
train_epochs = [log["epoch"] for log in training_logs if "loss" in log and "epoch" in log]
train_loss = [log["loss"] for log in training_logs if "loss" in log and "epoch" in log]

eval_epochs = [log["epoch"] for log in training_logs if "eval_loss" in log]
eval_loss = [log["eval_loss"] for log in training_logs if "eval_loss" in log]
eval_f1 = [log["eval_f1"] for log in training_logs if "eval_f1" in log]

# Plot Loss
plt.figure(figsize=(10, 4))
plt.plot(train_epochs, train_loss, label="Train Loss", marker="o")
plt.plot(eval_epochs, eval_loss, label="Validation Loss", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot F1
plt.figure(figsize=(10, 4))
plt.plot(eval_epochs, eval_f1, label="Validation F1", color="green", marker="o")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1-score over Epochs")
plt.grid(True)
plt.legend()
plt.show()