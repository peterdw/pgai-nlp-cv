import pandas as pd
from datasets import DatasetDict, Dataset
from base_trainer import BaseTrainer

class ComplaintClassifierTrainer(BaseTrainer):
    def __init__(self, precision_mode="16bit", num_train_epochs=10):
        super().__init__(
            model_name="bert-base-uncased",
            precision_mode=precision_mode,
            num_labels=2,
            num_train_epochs=num_train_epochs,
            model_dir=f"models/bert-complaint-classifier-{precision_mode}"
        )

    def load_and_prepare_data(self):
        df = pd.read_feather("data/df_emails_cleaned.feather")
        df = df[["ProcessedTextBody", "Complaint"]].dropna()
        df = df.rename(columns={"Complaint": "label"})

        # Convert boolean to int (if needed)
        if df["label"].dtype == "bool":
            df["label"] = df["label"].astype(int)

        n_total = len(df)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        train_df = df.iloc[:n_train].sample(frac=1.0, random_state=42)
        val_df = df.iloc[n_train:n_train + n_val].sample(frac=1.0, random_state=42)
        test_df = df.iloc[n_train + n_val:].sample(frac=1.0, random_state=42)

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })
        self.print_dataset_stats(train_df, val_df, test_df)

        return dataset, train_df, val_df, test_df
