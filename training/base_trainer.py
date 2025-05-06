import os
import json
import shutil
import traceback
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    AutoModelForSequenceClassification, BitsAndBytesConfig,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel

from custom_trainer import WeightedLossTrainer


class BaseTrainer:
    def __init__(self,
                 model_name="bert-base-uncased",
                 precision_mode="16bit",
                 num_labels=2,
                 num_train_epochs=3,
                 model_dir="models/default",
                 data_file="data/df_emails_cleaned.feather",
                 split_ratio=(0.7, 0.15, 0.15)):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.precision_mode = precision_mode
        self.num_train_epochs = num_train_epochs
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.data_file = data_file
        self.split_ratio = split_ratio

        os.makedirs(self.model_dir, exist_ok=True)

        self.model = None  # to avoid IDE warning

    def load_and_prepare_data(self):
        """
        Must be implemented in subclass.
        Should return: dataset, train_df, val_df, test_df
        """
        raise NotImplementedError

    def preprocess(self, dataset):
        tokenizer = self.tokenizer

        def tokenize(example):
            return tokenizer(example["ProcessedTextBody"], truncation=True, max_length=256)

        return dataset.map(tokenize, batched=True), tokenizer

    def build_model(self):

        # Dynamically determine target_modules
        target_modules_map = {
            "xlm-roberta": ["q_proj", "v_proj"],
            "distilbert": ["q_lin", "v_lin"],
            "bert": ["query", "value"],
            "roberta": ["query", "value"],
        }

        # Default fallback (can be empty, or raise if not found)
        matched_modules = None
        for key, modules in target_modules_map.items():
            if key in self.model_name.lower():
                matched_modules = modules
                break

        if matched_modules is None:
            raise ValueError(f"Could not determine `target_modules` for model: {self.model_name}")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=matched_modules,
        )

        if self.precision_mode in ["4bit", "8bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(self.precision_mode == "4bit"),
                load_in_8bit=(self.precision_mode == "8bit"),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                device_map="auto"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                quantization_config=bnb_config
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )

        model.config.problem_type = "single_label_classification"
        model = get_peft_model(model, peft_config)
        return model

    def compute_class_weights(self, labels):
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()

        avg = "binary" if self.num_labels == 2 else "macro"
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=avg)
        acc = accuracy_score(labels, predictions)

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self):
        dataset, train_df, val_df, test_df = self.load_and_prepare_data()
        tokenized_datasets, tokenizer = self.preprocess(dataset)
        self.model = self.build_model()

        training_args = TrainingArguments(
            output_dir=self.model_dir,
            logging_dir=os.path.join(self.model_dir, "logs"),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=(self.precision_mode == "16bit"),
            logging_steps=10,
            report_to="tensorboard",
        )

        class_weights = self.compute_class_weights(train_df["label"].values).to(self.model.device)

        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights
        )

        try:
            trainer.train()
        except Exception:
            traceback.print_exc()

        if isinstance(self.model, PeftModel):
            # Save adapter model separately for inspection/debug
            adapter_dir = self.model_dir + "_adapter"
            print(f"💾 Saving adapter model to: {adapter_dir}")
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)

            # Merge adapter into base model
            print("🔁 Merging LoRA adapter into base model...")
            merged_model = self.model.merge_and_unload()

            # Clean main model directory
            if os.path.exists(self.model_dir):
                print(f"🧹 Cleaning model directory: {self.model_dir}")
                for filename in os.listdir(self.model_dir):
                    file_path = os.path.join(self.model_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}: {e}")

            # Save merged model
            merged_model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            print(f"✅ Merged model saved to: {self.model_dir}")
        else:
            # If not a PEFT model, just save the model directly
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)

        # Save quantization info
        with open(os.path.join(self.model_dir, "quantization_config.json"), "w") as f:
            json.dump({"quantization_bits": self.precision_mode}, f, indent=2)

        # Evaluate on test set
        predictions = trainer.predict(tokenized_datasets["test"])
        y_true = predictions.label_ids
        y_pred = predictions.predictions.argmax(-1)

        test_df = test_df.reset_index(drop=True)
        test_df["TrueLabel"] = y_true
        test_df["PredictedLabel"] = y_pred
        test_df["Correct"] = test_df["TrueLabel"] == test_df["PredictedLabel"]

        test_df.to_csv(os.path.join(self.model_dir, "test_predictions.csv"), index=False)
        test_df[~test_df["Correct"]].to_csv(os.path.join(self.model_dir, "incorrect_predictions.csv"), index=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.model_dir, f"training_logs_{timestamp}.json")
        with open(log_file, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

        avg = "binary" if self.num_labels == 2 else "macro"
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        acc = accuracy_score(y_true, y_pred)

        final_metrics = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        self.save_final_metrics(final_metrics)

        self.plot_training_curves(trainer.state.log_history)
        print("✅ Training logs saved to:", log_file)

    def plot_training_curves(self, training_logs):
        train_epochs = [log["epoch"] for log in training_logs if "loss" in log and "epoch" in log]
        train_loss = [log["loss"] for log in training_logs if "loss" in log and "epoch" in log]
        eval_epochs = [log["epoch"] for log in training_logs if "eval_loss" in log]
        eval_loss = [log["eval_loss"] for log in training_logs if "eval_loss" in log]
        eval_f1 = [log["eval_f1"] for log in training_logs if "eval_f1" in log]

        plt.figure(figsize=(10, 4))
        plt.plot(train_epochs, train_loss, label="Train Loss", marker="o")
        plt.plot(eval_epochs, eval_loss, label="Validation Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(eval_epochs, eval_f1, label="Validation F1", color="green", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Validation F1-score over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, "f1_curve.png"))
        plt.close()

    def print_dataset_stats(self, train_df, val_df, test_df, label_encoder=None):
        print("📊 Dataset Summary")
        print("------------------")

        stats = {
            "total": len(train_df) + len(val_df) + len(test_df),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "splits": {}
        }

        print(f"Total records:      {stats['total']}")
        print(f"Training records:   {stats['train']}")
        print(f"Validation records: {stats['validation']}")
        print(f"Test records:       {stats['test']}")

        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            print(f"\n🔍 {name.capitalize()} label distribution:")
            label_dist = {}

            value_counts = df["label"].value_counts().sort_index()
            for label, count in value_counts.items():
                label_name = (
                    label_encoder.inverse_transform([label])[0]
                    if label_encoder is not None else str(label)
                )
                percent = count / len(df) * 100
                label_dist[label_name] = {
                    "count": int(count),
                    "percent": round(percent, 2)
                }
                print(f"  {label_name:15}: {count:4} ({percent:.2f}%)")

            stats["splits"][name] = label_dist

        # Save to JSON file
        stats_path = os.path.join(self.model_dir, "dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n📝 Dataset stats saved to: {stats_path}")

    def save_final_metrics(self, metrics: dict):
        metrics_path = os.path.join(self.model_dir, "final_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📄 Final evaluation metrics saved to: {metrics_path}")
